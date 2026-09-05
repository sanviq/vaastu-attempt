# Change log — `feat/app-ui-overhaul`

Branch cut from `main`. Everything below is on this branch only; `main` is untouched
except for one commit noted at the bottom.

Purpose of the branch: make the prototype presentable and actually correct. Two of
the fixes were real logic bugs that produced confident but wrong output.

---

## 1. Changes to Harshika's files

Harshika owns `detection.py`, `overlay.py`, `requirements.txt` and the rules CSV.
She gave verbal go-ahead on 05 Sep to edit them. Her public function signatures are
unchanged — anything calling the old API still works.

### `detection.py`

| # | Change | Why |
|---|---|---|
| 1.1 | **Room finding rewritten**: rooms are now segmented as *enclosed floor regions between walls* (`_wall_mask` + `_segment_rooms`, using Otsu → dilate → `connectedComponentsWithStats` on the inverted mask), filtered by area fraction and by whether the region touches the image edge (= outdoor space). | The old approach crossed every Hough-detected wall line into a grid, so one box could span several rooms. On the 8-room test plan it produced **49 boxes**. |
| 1.2 | Her original contour + Hough path is **kept as a fallback**, triggered when segmentation finds fewer than 2 regions. | Open-plan drawings, or plans with walls too broken to enclose anything, still need a strategy. Her code is the right tool there. |
| 1.3 | **`_ocr_crop` rewritten**: insets 4px past the walls instead of padding outward, converts to greyscale, upscales the crop to ~200px on its short side, applies Otsu, and switches Tesseract from `--psm 6` to `--psm 11` (sparse text). | Padding outward pulled the *neighbouring* room's label into the crop. Tesseract is also much steadier on ~200px text than on raw small crops. |
| 1.4 | **Added `_ROOM_VOCAB` + `_sanitise_label`**, which keeps only tokens that match a known room word (fuzzy, via `difflib`, cutoff 0.8) and returns `None` otherwise. Added `import difflib`. | OCR noise like `'rPyuVvA Rue'` used to become the room label. A clean `"Room 4"` is more useful downstream than a corrupted string. Tolerates mild corruption (`KTCHEN` → kept). |
| 1.5 | **Added public `ocr_available()`** — actually invokes `pytesseract.get_tesseract_version()` rather than just checking the import. | The pip package importing does not mean the Tesseract *binary* exists. Without the binary every room silently becomes "Room N", nothing matches a rule, and the plan scores as Moderate — a plausible-looking but meaningless result. The UI now says so out loud. |
| 1.6 | `HoughLinesP` unpacking now goes through `np.asarray(line).reshape(-1)`. | OpenCV 5 changed the return shape from `(N, 1, 4)` to `(N, 4)`; the old unpack raised `TypeError: cannot unpack non-iterable numpy.int32 object`. Originally fixed on `fix/opencv5-houghlines-unpack`, merged in here. |

**Measured effect on the 8-room test plan:** 49 boxes → **8**; correct labels **0/8 → 8/8**;
classifications went from uniformly "Moderate" to a genuine 5 Moderate / 3 Non-Compliant split.
Blank image → 0 rooms, no crash.

### `overlay.py`

Rewritten visually. No behavioural contract broken — both functions still take
`(image_path, list_of_dicts)` and return an RGB ndarray.

| # | Change | Why |
|---|---|---|
| 2.1 | **`draw_remodel_tier` now accepts `north_angle=0.0`**, and direction vectors are derived from it via `_bearing_to_vector` instead of a hard-coded `_DIR_VEC` table. | **Real bug.** The old table assumed North was image-up. On any rotated plan every "move to X" arrow pointed the wrong way. Derived as `image_angle = bearing + north_angle`, the exact inverse of what `_centroid_to_direction` does. |
| 2.2 | **Header and footer bands** are padded onto the canvas before drawing; room boxes are offset down by the header height. | The legend used to sit *on top of* the plan in the upper-right, covering whatever room was there. |
| 2.3 | **Compass rose** in the header showing which way North is, rotated by `north_angle`. | Makes the orientation setting visible and checkable instead of invisible state. |
| 2.4 | **Palette softened**: green `#00b800`→`#1f9d55`, orange `#ffa500`→`#d97706`, red `#dc0000`→`#dc2626`. Fill alpha `0.30`→`0.13`. | The saturated fills washed out the blueprint underneath — the drawing has to stay readable through the overlay. |
| 2.5 | **Label chips are centred and auto-sized** to fit the room (`_fit_fontsize`), and are skipped entirely on boxes under 34×22px. | Labels were pinned to the top-left corner where they overlapped walls, and overflowed small rooms. |
| 2.6 | **Arrow tips are clamped** inside their own room box. | A south-facing arrow on a bottom-row room used to shoot out of the plan and into the legend band. |
| 2.7 | **Dropped `bbox_inches="tight"`** from `_fig_to_rgb`; dpi fixed at 100. | `tight` crops each figure to its own content, so the four remodel tiers came out at *different sizes* and the 2×2 grid looked ragged. All four are now identical dimensions. |
| 2.8 | Replaced the `ax.set_title(backgroundcolor=...)` banner with the proper header band. Removed the now-unused `math` import and dead `img_h/img_w` locals. | Cosmetic + tidy. |

### `requirements.txt`

| # | Change | Why |
|---|---|---|
| 3.1 | `opencv-python>=4.5.0` → `opencv-python>=4.5.0,<5.0.0` | The unbounded pin is what let OpenCV 5.0.0 install and trigger 1.6. Code handles both now, but a demo build should not pull in other v5 API changes. |
| 3.2 | Documented the **system** dependencies (Tesseract, Poppler) in a comment block. | Neither is pip-installable, and missing Tesseract silently degrades every result rather than erroring. |

---

## 2. Changes to Sanvi's files

### `app.py` — rewritten

| # | Change | Why |
|---|---|---|
| 4.1 | Real branding: page title/icon, hero block, "Vastu Blueprint Compliance Checker". | It just said "Vastu". |
| 4.2 | Raw "north angle" degree input replaced with a plain-English compass dropdown ("Which direction does the **top** of your plan point to?"). | Nobody knows what a north angle in degrees means. |
| 4.3 | **The dropdown value is inverted before it is passed on**: `(360 - facing) % 360`. | **Real bug I introduced.** `detect_rooms(north_angle=…)` means *the image bearing of North*, which is the inverse of *the direction the top of the plan faces*. Passing it straight through mislabelled the direction of every room on every non-North selection. Verified empirically across all 8 headings. |
| 4.4 | Remodel slider replaced with **all four tiers rendered in a 2×2 grid**, each captioned. | The slider did not move, and only 25% was ever visible. |
| 4.5 | Three tabs: Compliance overlay / Room-by-room / Remodelling options. Four headline metrics + a verdict line. Room table gained "Matched rule" and "Source agreement" columns. | Readability; and surfacing source agreement is the honest thing to do given the sources disagree. |
| 4.6 | **PDFs are rasterised to PNG up front** (`_materialise`), with clear errors for missing `pdf2image` / Poppler. | The uploader accepted `.pdf`, `detection.py` could open it, but `overlay.py` uses `cv2.imread`, which cannot — so any PDF upload crashed at the overlay step. |
| 4.7 | Guards added: no rooms detected, no violations to remodel, OCR unavailable, and a count of rooms that matched no Vastu rule. | Each of these previously produced a confident-looking but empty or meaningless screen. |
| 4.8 | `@st.cache_data` on the analysis step. | Re-running detection + OCR on every widget interaction was slow. |

### `rules_engine.py` — on `main`, commit `3aa3492`

| # | Change | Why |
|---|---|---|
| 5.1 | `bbox` is now passed through `_classify_room` (both return paths) and `generate_remodel_tiers`. | **Contract gap.** `overlay.py` reads `bbox` off these dicts; without it `draw_overlay` skipped every room and `draw_remodel_tier` fell back to a generic banner. Nothing rendered. |

---

## 3. Still open

- Validation used a **synthetic** labelled plan (`real_plan.png`), not a real blueprint. Needs a run against an actual plan before demo.
- Segmentation assumes rooms are *enclosed*. Plans with wide doorway gaps may merge two rooms into one region — the fallback path does not catch this because segmentation still returns ≥2 boxes.
- History contains one ugly `wip ui` commit (`04c545c`). A squash-merge cleans it up; not force-pushed since the branch is shared.
