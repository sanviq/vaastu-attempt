"""
Owner: Harshika  (visual pass by Sanvi — see CHANGES.md, agreed 05 Sep)

Contract (do not change shapes without messaging Sanvi immediately):
    draw_overlay(image_path, room_results, north_angle=0.0) -> np.ndarray (RGB)
        image_path   : str — original floor plan (PNG/JPG; rasterise PDFs first)
        room_results : list of dicts from rules_engine.score_layout()
                       Each dict has at minimum:
                           room_label, direction, classification, bbox
        north_angle  : float — optional, only used to orient the compass rose

    draw_remodel_tier(image_path, tier_suggestions, north_angle=0.0) -> np.ndarray (RGB)
        image_path      : str — original floor plan
        tier_suggestions: list of dicts from rules_engine.generate_remodel_tiers()
                          Each dict has at minimum:
                              room_label, current_direction,
                              suggested_direction, classification, bbox
        north_angle     : float — image bearing of North. Needed so the "move to X"
                          arrows point at the real direction, not at image-up.

    Both new parameters default to 0.0, so old call sites keep working.
    Both functions return an RGB numpy array (Matplotlib native).
    Streamlit's st.image() accepts RGB arrays directly.

Tool used: Matplotlib (PSF-based / BSD-compatible, per the brief's approved stack).

Layout: the plan is padded with a header band (title + compass rose) and a footer
band (legend) before drawing, so no annotation is ever painted over the drawing
itself. Room boxes are offset by the header height to compensate.

Colour coding:
    Compliant     → green  #1f9d55
    Moderate      → amber  #d97706
    Non-Compliant → red    #dc2626
"""

from __future__ import annotations

import io
import math

import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Streamlit threads
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (hex, Matplotlib-compatible)
# ---------------------------------------------------------------------------
_COLOURS = {
    "Compliant":     "#1f9d55",
    "Moderate":      "#d97706",
    "Non-Compliant": "#dc2626",
    # A room type the dataset has no rule for. Grey, not amber: it was never
    # judged, and colouring it like a partial pass overstates what is known.
    "Unknown":       "#64748b",
}
_DEFAULT_COLOUR = "#64748b"
_LEGEND_LABELS = {"Unknown": "Not in dataset"}

_INK = "#0f172a"          # near-black for body text
_MUTED = "#64748b"        # secondary text
_BAND = "#f1f5f9"         # header / footer band fill
_RULE = "#e2e8f0"         # hairline separating band from plan
_FILL_ALPHA = 0.10        # low: the blueprint underneath must stay readable
_BORDER_LW = 1.6

# Every plan is resampled to this width before annotating, so all the sizes
# below are absolute and the output is consistent across uploads.
_RENDER_W = 1000.0
_RENDER_MAX_H = 1300.0

_HEADER_H = 58            # title + compass rose
_FOOTER_H = 40            # colour key
_TITLE_FS = 15.0
_SUB_FS = 9.5
_LEGEND_FS = 9.0
_CHIP_FS_MIN, _CHIP_FS_MAX = 6.5, 11.0

# Compass bearing (degrees clockwise from North) at the centre of each zone.
_DIR_BEARING: dict[str, float] = {
    "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
    "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rgb(image_path: str) -> np.ndarray:
    """Load image as RGB ndarray. Handles BGR (OpenCV) → RGB conversion."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_plan(image_path: str) -> tuple[np.ndarray, float]:
    """
    Load the plan resampled to a fixed render width, plus the scale applied.

    Every size below — band heights, font sizes, swatches — is in pixels of this
    canonical canvas, so the output looks the same whatever the upload's
    resolution. Drawing at the source resolution instead is what made the result
    look crude: a 454px-wide plan was annotated with 5pt text and then stretched
    to about 1400px in the browser, magnifying every label and border with it.

    Callers must scale incoming bboxes by the returned factor.
    """
    rgb = _load_rgb(image_path)
    h, w = rgb.shape[:2]
    scale = min(_RENDER_W / w, _RENDER_MAX_H / h)
    if abs(scale - 1.0) < 0.02:
        return rgb, 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=interp)
    return resized, scale


def _scale_box(bbox, scale: float) -> tuple[float, float, float, float]:
    """Convert a detection bbox from source pixels to render pixels."""
    x, y, w, h = bbox
    return x * scale, y * scale, w * scale, h * scale


def _bearing_to_vector(direction: str, north_angle: float) -> tuple[float, float]:
    """
    Unit vector in image pixel space (+y is DOWN) pointing at a compass direction.

    detection._centroid_to_direction does `bearing = image_angle - north_angle`,
    so going the other way is `image_angle = bearing + north_angle`. Getting this
    backwards silently points every arrow the wrong way, so it is derived here
    rather than hard-coded per direction.
    """
    theta = math.radians(_DIR_BEARING[direction] + north_angle)
    return math.sin(theta), -math.cos(theta)


def _pad_canvas(rgb: np.ndarray, header: int, footer: int) -> np.ndarray:
    """White bands above and below the plan, so annotations never cover it."""
    h, w = rgb.shape[:2]
    canvas = np.full((h + header + footer, w, 3), 255, dtype=np.uint8)
    canvas[header:header + h] = rgb
    return canvas


def _make_figure(rgb: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """Create a frameless Matplotlib figure sized 1:1 to the image."""
    h, w = rgb.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """
    Render a Matplotlib figure to an RGB uint8 numpy array.

    No bbox_inches="tight" — it crops each figure to its own content, which made
    the four remodel tiers come out at slightly different sizes in the 2x2 grid.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0, facecolor="white")
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)[:, :, ::-1]   # BGR→RGB


def _band(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    """Flat background strip for the header / footer."""
    ax.add_patch(
        mpatches.Rectangle((x, y), w, h, facecolor=_BAND, edgecolor="none", zorder=2)
    )


def _draw_header(
    ax: plt.Axes, w: float, title: str, subtitle: str, north_angle: float
) -> None:
    """Title block across the top, with a compass rose showing which way North is."""
    _band(ax, 0, 0, w, _HEADER_H)
    ax.add_patch(mpatches.Rectangle((0, _HEADER_H - 1), w, 1,
                                    facecolor=_RULE, edgecolor="none", zorder=3))

    pad = 16.0
    ax.text(pad, _HEADER_H * 0.38, title, fontsize=_TITLE_FS, color=_INK,
            fontweight="bold", va="center", ha="left", zorder=3)
    ax.text(pad, _HEADER_H * 0.72, subtitle, fontsize=_SUB_FS, color=_MUTED,
            va="center", ha="left", zorder=3)

    # Compass rose, right-aligned. Radii stay inside the band height or the "N"
    # gets clipped off the top of the canvas.
    ring, r = 21.0, 11.0
    cx, cy = w - 34.0, _HEADER_H * 0.5
    ax.add_patch(mpatches.Circle((cx, cy), ring, facecolor="white",
                                 edgecolor="#cbd5e1", linewidth=1.0, zorder=3))
    dx, dy = _bearing_to_vector("N", north_angle)
    ax.annotate(
        "", xy=(cx + dx * r, cy + dy * r), xytext=(cx - dx * r, cy - dy * r),
        arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.3, mutation_scale=11),
        zorder=4,
    )
    ax.text(
        cx + dx * r * 1.5, cy + dy * r * 1.5, "N",
        fontsize=8.0, color=_INK, fontweight="bold",
        ha="center", va="center", zorder=5,
        bbox=dict(boxstyle="circle,pad=0.10", facecolor="white",
                  edgecolor="none", alpha=0.9),
    )


def _draw_legend(ax: plt.Axes, w: float, y0: float, note: str) -> None:
    """Colour key across the bottom, with the caption only if it actually fits."""
    _band(ax, 0, y0, w, _FOOTER_H)
    ax.add_patch(mpatches.Rectangle((0, y0), w, 1,
                                    facecolor=_RULE, edgecolor="none", zorder=3))

    swatch = 11.0
    char_w = _LEGEND_FS * 0.83   # ~0.6 em wide at dpi 100, see _fit_fontsize
    mid = y0 + _FOOTER_H * 0.5
    x = 16.0
    for key, colour in _COLOURS.items():
        label = _LEGEND_LABELS.get(key, key)
        ax.add_patch(mpatches.Rectangle((x, mid - swatch / 2), swatch, swatch,
                                        facecolor=colour, edgecolor="none", zorder=3))
        ax.text(x + swatch * 1.7, mid, label, fontsize=_LEGEND_FS, color=_INK,
                va="center", ha="left", zorder=3)
        x += swatch * 1.7 + len(label) * char_w + 22.0

    # The caption is decoration; dropping it beats overprinting the key, which is
    # what happened when it was drawn unconditionally on a narrow plan.
    if note and w - x > len(note) * _LEGEND_FS * 0.62 + 16.0:
        ax.text(w - 16.0, mid, note, fontsize=8.0, color=_MUTED,
                va="center", ha="right", zorder=3)


def _fit_fontsize(text: str, box_w: float, lo: float, hi: float) -> float:
    """
    Largest font size (points) that keeps `text` inside `box_w` pixels.

    At dpi=100 one point is 100/72 px tall and a bold glyph averages ~0.6 of its
    height in width, so a character costs roughly 0.83 * fontsize pixels.
    """
    if not text:
        return lo
    return max(lo, min(hi, box_w * 0.92 / (len(text) * 0.83)))


def _draw_room_box(
    ax: plt.Axes,
    x: float, y: float, w: float, h: float,
    colour: str,
    title: str,
    subtitle: str,
) -> None:
    """Tinted box + border + a centred label chip, sized to fit the room."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=_BORDER_LW,
            edgecolor=colour,
            facecolor=colour,
            alpha=_FILL_ALPHA,
            zorder=3,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=_BORDER_LW,
            edgecolor=colour,
            facecolor="none",
            zorder=4,
        )
    )

    # Tiny slivers cannot hold legible text; the coloured border still reads.
    if w < 46 or h < 26:
        return

    fs = _fit_fontsize(title, w, _CHIP_FS_MIN, _CHIP_FS_MAX)
    ax.text(
        x + w / 2, y + fs * 1.5, title,
        fontsize=fs, color="white", fontweight="bold",
        ha="center", va="center", clip_on=True, zorder=5,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=colour,
                  edgecolor="none", alpha=0.94),
    )

    if subtitle and h > fs * 5:
        sub_fs = _fit_fontsize(subtitle, w, 5.5, fs * 0.88)
        ax.text(
            x + w / 2, y + fs * 3.5, subtitle,
            fontsize=sub_fs, color=_INK,
            ha="center", va="center", clip_on=True, zorder=5,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                      edgecolor="none", alpha=0.82),
        )


# ---------------------------------------------------------------------------
# Public API — draw_overlay
# ---------------------------------------------------------------------------

def draw_overlay(
    image_path: str,
    room_results: list[dict],
    north_angle: float = 0.0,
) -> np.ndarray:
    """
    Draw the green/amber/red compliance overlay on the floor plan.

    Returns an RGB numpy array suitable for st.image().
    """
    rgb, scale = _load_plan(image_path)
    plan_h, plan_w = rgb.shape[:2]
    canvas = _pad_canvas(rgb, _HEADER_H, _FOOTER_H)
    fig, ax = _make_figure(canvas)

    counts = {k: 0 for k in _COLOURS}
    for room in room_results:
        classification = room.get("classification", "")
        if classification in counts:
            counts[classification] += 1

    for room in room_results:
        bbox = room.get("bbox")
        if not bbox:
            continue
        x, y, w, h = _scale_box(bbox, scale)
        classification = room.get("classification", "")
        colour = _COLOURS.get(classification, _DEFAULT_COLOUR)
        _draw_room_box(
            ax,
            x, y + _HEADER_H, w, h,         # shift down past the header band
            colour,
            str(room.get("room_label", "?"))[:24],
            f"{room.get('direction', '')} · {classification}",
        )

    n = len(room_results)
    _draw_header(
        ax, plan_w,
        "Vastu compliance overlay",
        f"{n} room(s) analysed · {counts['Compliant']} compliant · "
        f"{counts['Moderate']} moderate · {counts['Non-Compliant']} non-compliant"
        + (f" · {counts['Unknown']} not in dataset" if counts["Unknown"] else ""),
        north_angle,
    )
    _draw_legend(
        ax, plan_w, _HEADER_H + plan_h,
        "Zones follow the Vastu Purusha Mandala",
    )
    return _fig_to_rgb(fig)


# ---------------------------------------------------------------------------
# Public API — draw_remodel_tier
# ---------------------------------------------------------------------------

def draw_remodel_tier(
    image_path: str,
    tier_suggestions: list[dict],
    north_angle: float = 0.0,
) -> np.ndarray:
    """
    Annotate the floor plan with the remodel suggestions for one tier.

    For each violation: a highlighted box, an arrow pointing at the recommended
    direction, and "Move to X (now Y)".

    Returns an RGB numpy array.
    """
    rgb, scale = _load_plan(image_path)
    plan_h, plan_w = rgb.shape[:2]
    canvas = _pad_canvas(rgb, _HEADER_H, _FOOTER_H)
    fig, ax = _make_figure(canvas)

    for suggestion in tier_suggestions:
        bbox = suggestion.get("bbox")
        classification = suggestion.get("classification", "Non-Compliant")
        colour = _COLOURS.get(classification, _COLOURS["Non-Compliant"])

        room_label = str(suggestion.get("room_label", "?"))[:24]
        current_dir = suggestion.get("current_direction", "?")
        suggested_dir = suggestion.get("suggested_direction") or "—"

        if bbox:
            x, y, w, h = _scale_box(bbox, scale)
        else:
            # No bbox — fall back to a banner strip rather than dropping the room.
            x, y = plan_w / 4, 8.0
            w, h = plan_w / 2, max(52.0, plan_h * 0.07)
        y += _HEADER_H

        _draw_room_box(
            ax, x, y, w, h, colour,
            room_label,
            f"→ {suggested_dir} (now {current_dir})",
        )

        # Arrow toward the recommended zone, rotated into image space.
        if suggested_dir in _DIR_BEARING:
            dvx, dvy = _bearing_to_vector(suggested_dir, north_angle)
            cx, cy = x + w / 2, y + h * 0.78
            arrow_len = max(14.0, min(w, h) * 0.28)
            # Keep the head inside the room it belongs to, otherwise a south-facing
            # arrow on a bottom-row room shoots out into the legend band.
            tip_x = min(max(cx + dvx * arrow_len, x + 3), x + w - 3)
            tip_y = min(max(cy + dvy * arrow_len, y + 3), y + h - 3)
            ax.annotate(
                "",
                xy=(tip_x, tip_y),
                xytext=(cx - dvx * arrow_len * 0.15, cy - dvy * arrow_len * 0.15),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2,
                                mutation_scale=16, shrinkA=0, shrinkB=0),
                annotation_clip=True,
                zorder=6,
            )

    n = len(tier_suggestions)
    _draw_header(
        ax, plan_w,
        "Remodelling plan",
        f"{n} room(s) addressed · arrows show the recommended direction",
        north_angle,
    )
    _draw_legend(
        ax, plan_w, _HEADER_H + plan_h,
        "Arrow = move this room toward that zone",
    )
    return _fig_to_rgb(fig)
