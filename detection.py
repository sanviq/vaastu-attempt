"""
Owner: Harshika

Contract (do not change shapes without messaging Sanvi immediately):
    detect_rooms(image_path, north_angle) -> list of dicts, each:
        {
            "room_label": str,       # text from OCR or fallback "Room N"
            "direction": str,        # one of N/NE/E/SE/S/SW/W/NW/C
            "bbox": [x, y, w, h],   # bounding box in image pixels
            "north_angle": float,   # echoed back for downstream reference
        }

Pipeline (each step guarded so a failure degrades gracefully):
    1. Load image (supports PNG/JPG; PDF -> first-page raster via pdf2image if present)
    2. Build a wall mask (Otsu) and dilate it so hairline gaps in walls are sealed
    3. Segment rooms as the enclosed floor regions *between* walls
       (cv2.connectedComponentsWithStats on the inverted wall mask), filtered by
       area and by whether the region touches the image edge (= outdoor space).
       Fallback: if that yields almost nothing (open-plan drawings, broken walls)
       drop back to the contour + Hough Line Transform grid below.
    4. OCR each room crop with Tesseract (falls back to room-number fallback if absent)
    5. Map each room centroid to a Vastu zone using the rotated Purusha Mandala grid

Why segmentation, not a line grid, drives step 3: crossing every detected wall line
produces boxes that span several rooms, so each OCR crop then contains text from
several rooms at once and Tesseract returns unusable strings. Segmenting the floor
regions gives one box per room, which fixes the labels as a side effect.

Prototype constraints (match the brief):
    - Input is a clean, digitally-generated blueprint with consistent labels.
    - Arbitrary scans / photos are out of scope for this sprint.
"""

from __future__ import annotations

import difflib
import math
import re
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional OCR backends — Tesseract preferred, EasyOCR as graceful fallback
# ---------------------------------------------------------------------------
try:
    import pytesseract  # type: ignore

    _TESS_OK = True
except ImportError:
    _TESS_OK = False
    warnings.warn(
        "pytesseract not installed — room labels will use centroid-based fallback names.",
        RuntimeWarning,
        stacklevel=1,
    )


def ocr_available() -> bool:
    """
    True only if the Tesseract *binary* can actually be run.

    The pip package importing is not enough — without the system binary every
    room silently falls back to "Room N", no rule matches, and the whole plan
    scores as Moderate. The UI needs to be able to say so out loud rather than
    show a plausible-looking but meaningless result.
    """
    if not _TESS_OK:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Optional PDF rasteriser
# ---------------------------------------------------------------------------
try:
    from pdf2image import convert_from_path  # type: ignore

    _PDF_OK = True
except ImportError:
    _PDF_OK = False

# ---------------------------------------------------------------------------
# Vastu Purusha Mandala — 8 cardinal/inter-cardinal zones + centre
# ---------------------------------------------------------------------------
# Angles are measured clockwise from North (image-up before rotation).
# Each zone spans 45°; centre is the inner 1/3 × 1/3 rectangle.
_ZONES_CW: list[tuple[float, float, str]] = [
    (337.5, 360.0, "N"),
    (0.0,   22.5,  "N"),
    (22.5,  67.5,  "NE"),
    (67.5,  112.5, "E"),
    (112.5, 157.5, "SE"),
    (157.5, 202.5, "S"),
    (202.5, 247.5, "SW"),
    (247.5, 292.5, "W"),
    (292.5, 337.5, "NW"),
]


def _angle_to_zone(angle_deg: float) -> str:
    """Map a bearing (0–360 clockwise from North) to a Vastu cardinal zone."""
    angle_deg = angle_deg % 360
    for lo, hi, zone in _ZONES_CW:
        if lo <= angle_deg < hi:
            return zone
    return "N"  # fallback (covers exactly 360 → 0 edge)


def _centroid_to_direction(cx: int, cy: int, img_w: int, img_h: int, north_angle: float) -> str:
    """
    Map pixel centroid (cx, cy) to one of N/NE/E/SE/S/SW/W/NW/C.

    north_angle: degrees clockwise — the direction in the *image* that
    corresponds to geographic North (0 = image-top is North).

    The centre 1/3 × 1/3 of the image is always Brahmasthan (C).
    """
    centre_x, centre_y = img_w / 2, img_h / 2
    third_w, third_h = img_w / 3, img_h / 3

    # Centre zone check first
    if (
        abs(cx - centre_x) < third_w / 2
        and abs(cy - centre_y) < third_h / 2
    ):
        return "C"

    # Bearing from image centre to centroid (image +Y = down, so negate dy)
    dx = cx - centre_x
    dy = -(cy - centre_y)  # flip so positive = up = visual North when angle=0
    raw_angle = math.degrees(math.atan2(dx, dy)) % 360  # clockwise from image-up

    # Rotate by north_angle so that geographic North aligns correctly
    bearing = (raw_angle - north_angle) % 360
    return _angle_to_zone(bearing)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image(image_path: str) -> np.ndarray:
    """Load PNG/JPG (or PDF page-1 if pdf2image available)."""
    path = Path(image_path)
    if path.suffix.lower() == ".pdf":
        if not _PDF_OK:
            raise RuntimeError(
                "PDF input requires pdf2image (`pip install pdf2image`). "
                "Convert to PNG/JPG and re-upload."
            )
        pages = convert_from_path(str(path), dpi=150, first_page=1, last_page=1)
        img = np.array(pages[0])
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary (0/255) mask where walls are white and rooms are black.
    Steps: grayscale → Gaussian blur → adaptive threshold → morphological close.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=4,
    )
    # Close small gaps in wall lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


# ---------------------------------------------------------------------------
# Room segmentation — enclosed floor regions between walls (primary strategy)
# ---------------------------------------------------------------------------

def _wall_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Return a mask where walls and printed text are white (255) and floor is black.

    Otsu rather than adaptive threshold: blueprints are high-contrast line art, and
    adaptive thresholding turns large blank floor areas into speckle, which then
    fragments the floor regions we are about to segment.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _thresh, wall = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Seal hairline breaks so one room cannot leak into the next.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(wall, kernel, iterations=1)


def _segment_rooms(
    bgr: np.ndarray,
    min_area_fraction: float = 0.004,
    max_area_fraction: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    """
    Return one (x, y, w, h) per enclosed floor region.

    A room is a connected patch of floor fully enclosed by walls. Regions touching
    the image border are the outdoor area around the building, not rooms, and
    regions above max_area_fraction are usually the whole plan leaking through a
    gap — both are dropped.
    """
    h, w = bgr.shape[:2]
    total_area = h * w
    interior = cv2.bitwise_not(_wall_mask(bgr))

    count, _labels, stats, _cents = cv2.connectedComponentsWithStats(interior, connectivity=4)

    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, count):                      # 0 is the background label
        x, y, bw, bh, area = (int(v) for v in stats[i])
        fraction = area / total_area
        if not (min_area_fraction <= fraction <= max_area_fraction):
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                               # outdoor space
        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# ---------------------------------------------------------------------------
# Contour detection — find room bounding boxes
# ---------------------------------------------------------------------------

def _find_room_contours(binary: np.ndarray, min_area_fraction: float = 0.005) -> list[tuple[int, int, int, int]]:
    """
    Return list of (x, y, w, h) for each candidate room region.
    Filters out contours smaller than min_area_fraction of total image area.
    """
    total_area = binary.shape[0] * binary.shape[1]
    min_area = total_area * min_area_fraction

    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    # Sort top-to-bottom, left-to-right for deterministic numbering
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# ---------------------------------------------------------------------------
# Hough Line Transform — derive room grid from dominant wall lines
# ---------------------------------------------------------------------------

def _hough_room_boxes(
    binary: np.ndarray,
    min_area_fraction: float = 0.005,
) -> list[tuple[int, int, int, int]]:
    """
    Use Probabilistic Hough Line Transform (cv2.HoughLinesP) to find dominant
    wall lines, then infer room bounding boxes from the grid they form.

    Strategy:
        1. Run HoughLinesP on the binary wall mask.
        2. Bucket lines into horizontal / vertical groups by angle.
        3. Sort each group by position; the gaps between adjacent parallel
           lines are candidate room spans.
        4. Cross horizontal × vertical spans → candidate boxes.
        5. Filter by minimum area.
    """
    h, w = binary.shape
    total_area = h * w
    min_area = total_area * min_area_fraction

    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(30, min(h, w) // 10),
        maxLineGap=20,
    )

    if lines is None:
        return []

    h_positions: list[int] = []   # y-coords of horizontal walls
    v_positions: list[int] = []   # x-coords of vertical walls

    for line in lines:
        # OpenCV <5 returns (N, 1, 4); OpenCV >=5 returns (N, 4) — flatten handles both.
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx == 0 and dy == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))   # 0=horizontal, 90=vertical
        if angle < 20:          # horizontal wall
            h_positions.append((y1 + y2) // 2)
        elif angle > 70:        # vertical wall
            v_positions.append((x1 + x2) // 2)

    if not h_positions or not v_positions:
        return []

    # Cluster nearby positions (within 15 px) to one representative
    def _cluster(positions: list[int], gap: int = 15) -> list[int]:
        positions = sorted(set(positions))
        clustered: list[int] = []
        for p in positions:
            if clustered and p - clustered[-1] < gap:
                clustered[-1] = (clustered[-1] + p) // 2  # merge
            else:
                clustered.append(p)
        return clustered

    ys = _cluster(h_positions)
    xs = _cluster(v_positions)

    # Add image edges so rooms touching the border are captured
    if not ys or ys[0] > 5:    ys = [0] + ys
    if ys[-1] < h - 5:         ys = ys + [h]
    if not xs or xs[0] > 5:    xs = [0] + xs
    if xs[-1] < w - 5:         xs = xs + [w]

    boxes: list[tuple[int, int, int, int]] = []
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            bx, by = xs[j], ys[i]
            bw, bh = xs[j + 1] - xs[j], ys[i + 1] - ys[i]
            if bw * bh >= min_area:
                boxes.append((bx, by, bw, bh))

    return boxes


# ---------------------------------------------------------------------------
# Merge contour + Hough boxes — deduplicate by IoU
# ---------------------------------------------------------------------------

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-Union for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _merge_boxes(
    contour_boxes: list[tuple[int, int, int, int]],
    hough_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """
    Return contour_boxes plus any Hough box that does not substantially
    overlap (IoU < iou_threshold) an existing contour box.
    Sort result top-to-bottom, left-to-right.
    """
    merged = list(contour_boxes)
    for hb in hough_boxes:
        if all(_iou(hb, cb) < iou_threshold for cb in merged):
            merged.append(hb)
    merged.sort(key=lambda b: (b[1], b[0]))
    return merged


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _clean_ocr_text(raw: str) -> str:
    """Strip non-alphabetic noise from a Tesseract result."""
    cleaned = re.sub(r"[^A-Za-z /\-]", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# Words that alone identify a room type. A detected phrase must contain one of
# these to be treated as a room — that is what keeps furniture callouts
# ("WARDROBE", "TV UNIT") and bare qualifiers ("SITTING SPACE", "OPEN AREA") out
# of the room list. Those are drawn on plans but are not rooms to be judged.
_HEAD_WORDS = {
    "kitchen", "bedroom", "toilet", "wc", "bath", "bathroom", "washroom",
    "dining", "drawing", "living", "hall", "lounge", "puja", "pooja", "prayer",
    "meditation", "study", "office", "library", "basement", "entrance", "foyer",
    "lobby", "porch", "veranda", "verandah", "balcony", "terrace", "store",
    "storage", "utility", "laundry", "pantry", "staircase", "stair", "stairs",
    "parking", "garage", "courtyard", "passage", "corridor", "garden", "lawn",
}

# Words that refine a head word but never stand alone as a room.
_MODIFIER_WORDS = {
    "master", "guest", "children", "kids", "family", "common", "attached",
    "open", "area", "space", "sitting", "sit", "dress", "dressing", "bed",
    "room", "wash", "main", "front", "rear",
}

# Everything Tesseract is allowed to match against.
_ROOM_VOCAB = _HEAD_WORDS | _MODIFIER_WORDS


def _match_vocab(token: str) -> Optional[str]:
    """
    Map one OCR token to a vocabulary word, tolerating mild corruption.

    A fuzzy match must agree on the first character. Tesseract garbles interior
    glyphs constantly ("BEDROpPM", "PUA") but rarely loses the leading one, while
    the false positives that matter are precisely the ones differing at the front:
    "All", from the "All Size House Plans" watermark, scores 0.86 against "hall"
    and invented a second living room in the middle of the test plan.
    """
    lowered = token.lower()
    if lowered in _ROOM_VOCAB:
        return lowered
    candidates = [v for v in _ROOM_VOCAB if v[0] == lowered[0]]
    close = difflib.get_close_matches(lowered, candidates, n=1, cutoff=0.78)
    return close[0] if close else None


def _has_head_word(words: list[str]) -> bool:
    """
    True if this phrase names an actual room.

    Checked against the concatenated phrase as well as the individual words, so
    that a label Tesseract splits as "BED" + "ROOM" still reads as a bedroom.
    """
    if any(w in _HEAD_WORDS for w in words):
        return True
    compact = "".join(words)
    return any(head in compact for head in _HEAD_WORDS)


def _sanitise_label(text: str) -> Optional[str]:
    """
    Keep only the tokens that look like real room words; return None if none do.

    Tolerates mild OCR corruption ("KTCHEN") via close matching, but rejects
    strings with no recognisable room word at all. Used by the per-crop fallback
    path; the primary path goes through _find_room_labels.
    """
    tokens = [t for t in _clean_ocr_text(text).upper().split() if len(t) >= 2]
    kept = [t for t in tokens if _match_vocab(t)]
    if not kept or not _has_head_word([t.lower() for t in kept]):
        return None
    return " ".join(kept)


def _ocr_crop(bgr: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[str]:
    """
    Run Tesseract on the interior of one room box.
    Returns a sanitised label, or None if OCR is unavailable / found nothing usable.
    """
    if not _TESS_OK:
        return None

    # Inset rather than pad: the surrounding walls and any label belonging to the
    # neighbouring room are exactly what we do not want inside the crop.
    inset = 4
    x0, y0 = max(0, x + inset), max(0, y + inset)
    x1, y1 = min(bgr.shape[1], x + w - inset), min(bgr.shape[0], y + h - inset)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    # Tesseract is much steadier on ~200px-tall text regions than on raw crops.
    scale = max(1.0, 200.0 / max(1, min(crop.shape[:2])))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _thresh, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    try:
        # psm 11 (sparse text) suits a label floating in a mostly empty room.
        text = pytesseract.image_to_string(crop, config="--psm 11 --oem 3")
    except Exception:
        return None
    return _sanitise_label(text)


# ---------------------------------------------------------------------------
# Whole-plan label detection (primary strategy)
# ---------------------------------------------------------------------------
#
# Cropping a region and OCR-ing it only works if the region is right, and on a
# real blueprint it usually is not: doorways are drawn as gaps, so the floor is
# one connected blob and segmentation returns furniture cells plus one leaking
# L-shaped region. Reading the printed labels off the whole plan first inverts
# the problem — the labels are what the draughtsman guarantees, so they become
# the room list, and segmentation is demoted to supplying each label a box.


def _text_mask(gray: np.ndarray) -> np.ndarray:
    """
    White-on-black mask holding only glyph-sized ink, walls and furniture removed.

    Tesseract reads printed labels far more reliably once the long wall strokes
    and hatching next to them are gone: on the test plan this is the difference
    between finding 5 labels and finding all 10. Size limits are in pixels of the
    original image, which is fine because plan labels are drawn at a similar
    physical size regardless of how large the sheet is.
    """
    _thresh, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _cents = cv2.connectedComponentsWithStats(ink, connectivity=8)

    glyphs = np.zeros_like(ink)
    for i in range(1, count):
        _x, _y, w, h, area = (int(v) for v in stats[i])
        if 3 <= h <= 26 and 2 <= w <= 60 and area >= 5:
            glyphs[labels == i] = 255
    return glyphs


def _ocr_scales(h: int, w: int) -> list[float]:
    """
    Upscale factors to try, capped so a large sheet does not blow up memory.

    Blueprint labels are often only ~8px tall, well under the ~30px Tesseract
    wants, and the useful factor depends on the sheet size rather than on the
    crop — which is exactly what the old per-crop scaling got wrong.
    """
    cap = 4200.0 / max(h, w)
    scales = [s for s in (3.0, 4.0, 5.0) if s <= cap]
    return scales or [max(1.0, min(cap, 2.0))]


def _ocr_word_boxes(image: np.ndarray, scale: float, psm: int) -> list[dict]:
    """One Tesseract pass; returns vocabulary words with original-pixel boxes."""
    up = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    try:
        data = pytesseract.image_to_data(
            up, config=f"--psm {psm} --oem 3", output_type=pytesseract.Output.DICT
        )
    except Exception:
        return []

    found: list[dict] = []
    for i, raw in enumerate(data["text"]):
        token = re.sub(r"[^A-Za-z]", "", raw).strip()
        try:
            conf = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            continue
        if len(token) < 3 or conf < 35:
            continue
        word = _match_vocab(token)
        if not word:
            continue
        found.append({
            "word": word,
            "conf": conf,
            "x": data["left"][i] / scale,
            "y": data["top"][i] / scale,
            "w": data["width"][i] / scale,
            "h": data["height"][i] / scale,
        })
    return found


def _collect_words(bgr: np.ndarray) -> list[dict]:
    """
    Union of several OCR passes over the whole plan, deduplicated by position.

    No single (preprocessing, scale, page-segmentation-mode) combination finds
    every label — on the test plan the glyph mask at psm 6 finds KITCHEN while
    only plain Otsu finds the lower BEDROOM. Taking the union and letting the
    vocabulary reject the noise gives far better recall than tuning one pass.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scales = _ocr_scales(h, w)

    glyphs = 255 - _text_mask(gray)                       # black text on white
    _thresh, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Both preprocessors are needed: the glyph mask is the only one that reads
    # KITCHEN on the test plan, and plain Otsu is the only one that reads the
    # lower BEDROOM, whose lettering the glyph filter clips to "BEDR".
    passes = [(glyphs, s, psm) for s in (scales[0], scales[-1]) for psm in (6, 11)]
    passes += [(otsu, s, psm) for s in (scales[0], scales[len(scales) // 2])
               for psm in (6, 11)]

    hits: list[dict] = []
    for image, scale, psm in passes:
        hits.extend(_ocr_word_boxes(image, scale, psm))

    # Same word found by several passes: keep the most confident instance.
    hits.sort(key=lambda d: -d["conf"])
    unique: list[dict] = []
    for hit in hits:
        near = max(12.0, hit["h"] * 1.5)
        if any(
            u["word"] == hit["word"]
            and abs(u["x"] - hit["x"]) < near
            and abs(u["y"] - hit["y"]) < near
            for u in unique
        ):
            continue
        unique.append(hit)
    return unique


def _phrase_tokens(tokens: list[str]) -> list[str]:
    """
    Deduplicate the words of one label for display.

    Drops any token contained in another, so a label read as both "BED" and
    "BEDROOM" by different passes renders as "BEDROOM" rather than "BED BEDROOM".
    """
    kept: list[str] = []
    for token in sorted(set(tokens), key=len, reverse=True):
        if not any(token in other for other in kept):
            kept.append(token)
    return [t.upper() for t in sorted(kept, key=lambda t: tokens.index(t))]


def _group_words(words: list[dict]) -> list[dict]:
    """
    Merge adjacent words into one label, so "MASTER" + "BEDROOM" is one room.

    Two words join if they sit within roughly a line height of each other, which
    covers both a label written on one line and one wrapped onto two.
    """
    groups: list[list[dict]] = []
    for word in sorted(words, key=lambda d: (d["y"], d["x"])):
        for group in groups:
            if any(
                abs((word["y"] + word["h"] / 2) - (o["y"] + o["h"] / 2)) < o["h"] * 1.8
                and word["x"] < o["x"] + o["w"] + o["h"] * 3.0
                and o["x"] < word["x"] + word["w"] + word["h"] * 3.0
                for o in group
            ):
                group.append(word)
                break
        else:
            groups.append([word])

    labels: list[dict] = []
    for group in groups:
        group.sort(key=lambda d: (d["y"], d["x"]))
        tokens = [g["word"] for g in group]
        if not _has_head_word(tokens):
            continue                       # furniture callout or bare qualifier
        x0 = min(g["x"] for g in group)
        y0 = min(g["y"] for g in group)
        x1 = max(g["x"] + g["w"] for g in group)
        y1 = max(g["y"] + g["h"] for g in group)
        labels.append({
            "text": " ".join(_phrase_tokens(tokens)),
            "cx": (x0 + x1) / 2,
            "cy": (y0 + y1) / 2,
            "conf": max(g["conf"] for g in group),
        })
    return labels


def _find_room_labels(bgr: np.ndarray) -> list[dict]:
    """Printed room labels on the plan, as {"text", "cx", "cy", "conf"}."""
    if not _TESS_OK:
        return []
    return _group_words(_collect_words(bgr))


def _box_for_label(
    label: dict,
    regions: list[tuple[int, int, int, int]],
    taken: set[int],
    img_w: int,
    img_h: int,
    max_area_fraction: float = 0.30,
) -> tuple[int, int, int, int]:
    """
    Best available segmented region for one label, or a synthesised box.

    Takes the *largest* unclaimed region containing the anchor that is still under
    max_area_fraction. Largest rather than smallest because plans draw wardrobes
    and fixtures as sub-cells inside a room, and the smallest containing region is
    reliably one of those rather than the room. The cap does the opposite job: on
    a plan with open doorways one component leaks across half the sheet, and
    drawing that as a "room" is worse than an approximate box in the right place.
    """
    cx, cy = label["cx"], label["cy"]
    limit = img_w * img_h * max_area_fraction

    candidates = [
        (i, b) for i, b in enumerate(regions)
        if i not in taken
        and b[0] <= cx <= b[0] + b[2]
        and b[1] <= cy <= b[1] + b[3]
        and b[2] * b[3] <= limit
    ]
    if candidates:
        index, box = max(candidates, key=lambda pair: pair[1][2] * pair[1][3])
        taken.add(index)
        return box

    # Nothing usable — a modest box centred on the label still places the room
    # correctly on the overlay, and the direction comes from the anchor anyway.
    bw = max(56, int(img_w * 0.17))
    bh = max(34, int(img_h * 0.09))
    x = int(min(max(cx - bw / 2, 0), img_w - bw))
    y = int(min(max(cy - bh / 2, 0), img_h - bh))
    return (x, y, bw, bh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_rooms(image_path: str, north_angle: float) -> list[dict]:
    """
    Main entry point called by app.py.

    Parameters
    ----------
    image_path : str
        Absolute path to a PNG, JPG/JPEG, or PDF floor plan.
    north_angle : float
        Degrees clockwise: the compass direction that corresponds to
        the *top* of the image (0 = image-top is North).

    Returns
    -------
    list of dict, each:
        {
            "room_label"  : str,
            "direction"   : str,   # N/NE/E/SE/S/SW/W/NW/C
            "bbox"        : [x, y, w, h],
            "north_angle" : float,
        }
    """
    bgr = _load_image(image_path)
    img_h, img_w = bgr.shape[:2]

    regions = _segment_rooms(bgr)

    # Primary strategy: the printed labels are the room list. Direction comes from
    # the label anchor rather than the box centroid — the anchor is inside the
    # room by construction, whereas a synthesised or leaked box may not be.
    labels = _find_room_labels(bgr)
    if len(labels) >= 2:
        rooms: list[dict] = []
        taken: set[int] = set()
        for label in sorted(labels, key=lambda d: -d["conf"]):
            x, y, w, h = _box_for_label(label, regions, taken, img_w, img_h)
            rooms.append(
                {
                    "room_label": label["text"],
                    "direction": _centroid_to_direction(
                        int(label["cx"]), int(label["cy"]), img_w, img_h, north_angle
                    ),
                    "bbox": [x, y, w, h],
                    "north_angle": north_angle,
                }
            )
        rooms.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return rooms

    # No readable labels (scan, handwriting, or no Tesseract): fall back to
    # segmenting regions and OCR-ing each crop, then to the contour + Hough grid.
    boxes = regions
    if len(boxes) < 2:
        binary = _preprocess(bgr)
        boxes = _merge_boxes(_find_room_contours(binary), _hough_room_boxes(binary))

    rooms = []
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cx, cy = x + w // 2, y + h // 2
        rooms.append(
            {
                "room_label": _ocr_crop(bgr, x, y, w, h) or f"Room {idx}",
                "direction": _centroid_to_direction(cx, cy, img_w, img_h, north_angle),
                "bbox": [x, y, w, h],
                "north_angle": north_angle,
            }
        )

    return rooms
