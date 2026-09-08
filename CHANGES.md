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

| 1.7 | **Room finding rewritten again — detection is now label-first** (`_find_room_labels` → `_collect_words` → `_group_words`, then `_box_for_label`). The printed room names are read off the whole plan and *become* the room list; segmentation is demoted to supplying each label a bounding box. Both older strategies are retained as fallbacks when fewer than 2 labels are readable. | **Real bug, found on the first real blueprint.** Segmentation assumes rooms are enclosed, but real plans draw doorways as gaps — so the floor is one connected blob. On the 30x40 test plan it returned 15 "rooms": wardrobe niches, a sitting space, a stair cell, and one L-shaped region leaking across 55% of the sheet. See §4 for the measured effect. |
| 1.8 | **`_text_mask`** removes everything that is not glyph-sized before OCR. | Wall strokes and hatching next to a label wreck the read. This is the only pass that finds KITCHEN on the test plan. |
| 1.9 | **`_ocr_scales`** picks the upscale factor from the *sheet* size, capped at ~4200px on the long side. | The old per-crop scaling targeted 200px on the crop's short side, which gave a 119px-wide kitchen a factor of 1.7 — nowhere near enough for 8px lettering. This is the single biggest reason labels were unreadable. |
| 1.10 | **`_collect_words` unions 8 OCR passes** (2 preprocessors × 2 scales × psm 6 and 11), deduplicating by position and keeping the most confident read. | No single combination finds everything: the glyph mask is the only one that reads KITCHEN, plain Otsu the only one that reads the lower BEDROOM. Union + vocabulary filtering beats tuning one pass. Costs ~4s, cached by `@st.cache_data`. |
| 1.11 | **Vocabulary split into `_HEAD_WORDS` and `_MODIFIER_WORDS`**; a phrase is only a room if it contains a head word. | This is what drops "WARDROBE", "TV UNIT", "SITTING SPACE" and "OPEN AREA" — drawn on plans, but not rooms to be judged. Directly fixes "it's considering things like sitting space as a room". |
| 1.12 | **Fuzzy matching now requires the first character to agree** (cutoff 0.78, was a blanket 0.8). | Tesseract garbles interior glyphs constantly ("BEDROpPM", "PUA") but rarely the leading one. Meanwhile "All", from the plan's "All Size House Plans" watermark, scores 0.86 against "hall" and invented a second living room mid-plan. |
| 1.13 | **`_box_for_label` takes the *largest* containing region under 30% of the sheet**, not the smallest, and synthesises a box when nothing qualifies. | Smallest reliably picked a wardrobe sub-cell drawn inside the room; the 30% cap rejects the leaking region. |

**Measured effect on the 8-room synthetic plan:** 49 boxes → **8**; correct labels **0/8 → 8/8**.
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
| 2.9 | **Every plan is resampled to a fixed 1000px render width** (`_load_plan`), and all band heights, font sizes and swatches are now absolute pixel constants instead of fractions of the plan width. Callers scale bboxes via `_scale_box`. | **This is why the UI looked crude.** A 454px-wide upload was annotated with 5pt text and 2px borders, then stretched to ~1400px in the browser — magnifying every label, chip and line with it. Output is now identical whatever the upload resolution. |
| 2.10 | Legend caption is **only drawn if it fits** beside the colour key. | On a narrow plan "Zones follow the Vastu Purusha Mandala" printed straight through "Non-Compliant". |
| 2.11 | Fill alpha `0.13`→`0.10`, border `2.0`→`1.6`, chip padding and minimum box size retuned. | The tinted boxes were dominating the drawing they are supposed to annotate. |

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
| 5.2 | Added `hall` and `lounge` to the `Drawing / living room` keywords. | Indian plans almost always print the main living room as "HALL", so the largest room in the house matched no rule at all. |
| 5.3 | `generate_remodel_tiers` now **excludes rooms with no matching rule** from the violation list. | PARKING has no rule in the dataset, so it was listed as a remodel step with a blank target direction — a row telling the user to move a room somewhere unspecified. |
| 5.4 | **A direction outside every stated list is now judged by distance from the nearest preferred zone** — ≤90° Moderate, >90° Non-Compliant — instead of falling through to "Moderate, 0 sources, direction not addressed". | **The real bug behind the all-Moderate wash.** This dataset is almost entirely *preferred-only*; `avoid` is blank on 14 of 18 rows. So a kitchen in NW, which no row mentions, was reported as "not addressed" rather than as 180° from the SE the sources actually ask for. On the real plan 7 of 8 rooms landed there — nothing was ever non-compliant, so remodelling had nothing to rank. |
| 5.5 | **`Bedroom` and `Master bedroom / bedroom` merged into one rule family** (`_RULE_FAMILIES`). | The dataset splits bedrooms across two rows and only the second carries an avoid list. A plain "BEDROOM" label matched the first alone — 1 source out of 2, and it threw away the only row that could ever mark a bedroom non-compliant. The NE bedroom is now correctly flagged. |
| 5.6 | Unmatched room types return **`classification: "Unknown"`** instead of `"Moderate"`, and `score_layout` **excludes them from the score** (also returns `n_judged` / `n_rooms`). | Scoring a car park against rules that never mention car parks moved the number without meaning anything. |
| 5.7 | **Every verdict now carries a written reason** and the rule's source titles, replacing `notes: None` and the bare "direction not addressed" string. | This was the actual complaint: the table said nothing about why a room passed or failed. |
| 5.8 | Centre (`C`) handled explicitly rather than by angle. | The Brahmasthan is not on the compass, so "60° from SE" is meaningless for it. It now cites the dataset's own Brahmasthan row about keeping the centre open. |

### `app.py` — presentation

| # | Change | Why |
|---|---|---|
| 6.1 | Overlay is shown at its natural width in a centred column (capped 920px) rather than `use_container_width=True`; page max-width 1400→1180px. | Streamlit was stretching the rendered overlay across the full container, undoing the fixed render width in 2.9. |
| 6.2 | **Dashboard restyled after the SPENDLY dashboard**: dark radial ground, translucent cards with hairline borders, uppercase tracked stat labels, large colour-coded figures, verdict pills. | Requested. The page was four bare `st.metric` calls and a default dataframe. |
| 6.3 | `st.metric` row replaced with custom stat cards + a **proportional compliance bar**. | Reads the split at a glance instead of four unrelated numbers. |
| 6.4 | `st.dataframe` replaced with **room cards** carrying a verdict-coloured left edge, the matched rule, source agreement, and the full reason. The verdict hues match the boxes overlay.py paints on the plan. | The reason text is the point of the table and a dataframe cell truncated it. |
| 6.5 | Added a **"Sources cited" expander** listing the classical texts behind the verdicts. | The app leans on 18 rows from 5 papers; naming them is the honest thing to do. |

---

## 3. Measured effect on the real blueprint

Run against `30X40 NORTH FACING HOUSE PLANS`, the first genuine blueprint tested.

| | Before | After |
|---|---|---|
| Rooms found | 15 (wardrobes, sitting space, stair cell, one 55% leak) | **8** — every real room, nothing spurious |
| Labels read | 1 usable (`TOILET`); rest `Room N`, plus `BEDRODM PARKIN` and `TOWLET` | **8/8 correct** |
| Matched to a Vastu rule | 1/15 | **7/8** (only PARKING unmatched — correctly, the dataset has no parking rule) |
| Directions | meaningless (boxes were furniture) | KITCHEN NW · BEDROOM NE · TOILET N · TOILET E · HALL C · PARKING SE · BEDROOM SW · PUJA S — all correct |
| Verdicts | 1 Compliant, 14 Moderate, 0 Non-Compliant | **2 Compliant · 2 Moderate · 3 Non-Compliant · 1 Unknown** |
| Explained verdicts | 1 of 15 | **8 of 8** |
| Remodel tiers | 4 identical screens | 2 / 3 / 4 / 5 rooms — genuinely different |
| Overall score | 53.3% (meaningless) | 42.9%, scored on the 7 rooms the dataset covers |

Also verified: rotating `north_angle` to 90° shifts all 8 directions consistently; a blank
image yields 0 rooms without crashing; the synthetic plan still reads 8/8 including the
multi-word `MASTER BEDROOM`, `PUJA ROOM` and `LIVING HALL`; Streamlit boots clean (HTTP 200).

---

## 4. Still open

- OCR now costs ~4s per new upload (8 Tesseract passes). Cached per file, so it is paid once.
- The label-first path needs ≥2 readable labels. Scans and handwritten plans fall through to
  the old segmentation path, which is still weak on plans with open doorways.
- Room boxes come from segmentation where one fits and are otherwise approximated around the
  label, so a box occasionally covers slightly more than the room. Position and direction are
  taken from the label anchor, so the *verdict* is unaffected — only the drawn rectangle.
- History contains one ugly `wip ui` commit (`04c545c`). A squash-merge cleans it up; not
  force-pushed since the branch is shared.

---

## 5. Sidebar navigation, plans and About page

All in `app.py` (Sanvi's file). No change to `detection.py`, `overlay.py`,
`rules_engine.py` or the rules CSV in this round.

| # | Change | Why |
|---|---|---|
| 7.1 | Added a sidebar with a three-way nav: **Analyser · Pricing · About** | The analyser calls `st.stop()` when no plan is uploaded, so anything rendered after it was unreachable until you uploaded a file. Pricing and About have to exist before that point. |
| 7.2 | Moved the analyser body into `render_analyser()`, added `render_pricing()` and `render_about()`, dispatched from a `_PAGES` dict | Three top-level pages need three entry points. Analysis logic is unchanged — only indented. |
| 7.3 | Added an **About** page: what the tool is, a four-step "how it works", the rule set (13 room types, 5 published works, listed by title), who built it, mentor, contact, and a liability disclaimer | Asked for. The source list is read off the same CSV the engine uses, so the page cannot drift from the rules. |
| 7.4 | Added a **Pricing** page: Free vs Pro cards, feature lists, `PRICE` constant at the top of the file | Asked for. Pricing is a single constant so it changes in one place. |
| 7.5 | Gated remodel tiers — Free gets **25%**, Pro gets all four; locked panel + upgrade CTA in the Remodelling tab | Gives away the diagnosis, charges for the full prescription. Free still ends in something actionable rather than a dead end. |
| 7.6 | Free 25% tier renders in a half-width column | Alone at full width it was the only oversized plan in the app — the same stretched look fixed in round 4. |
| 7.7 | Repainted Streamlit's primary button in `--accent` | Default Streamlit red read as a warning next to the red *Non-Compliant* verdict pills. |
| 7.8 | `.who.one` caps a single-person card at one column width | A lone card in an `auto-fit` grid stretched full width and read as a banner. |
| 7.9 | Rewrote `.gitignore` | Its git blob (`7a60b85`) was missing from the object store in the working clone — `git fsck` reported it. Content is identical (`__pycache__/`, `*.pyc`). |

### On the billing

**This is demo billing and nothing more.** `st.session_state.pro` is a session flag.
There is no gateway, no account, no database, and **no payment field of any kind** — the
upgrade is a single button, deliberately not a card form, because a realistic-looking
card-entry screen is the wrong thing for a demo build to ship. Pro resets on reload.
Both the sidebar and the Pricing page say so in plain language.

Making this real is not a matter of swapping the button. It needs, in order:

1. **Accounts** — a subscription has to belong to someone. Streamlit has no auth.
2. **A database** — subscription state must outlive the session.
3. **A gateway** — hosted checkout, so card details are entered on the gateway's page
   and never touch this app.
4. **Legal** — terms, refund policy, and the disclaimer already on the About page.

Steps 1 and 2 are the real work; step 3 is the small part. Deferred until hosting is settled.

### Placeholders left in `app.py`

- `PRICE` / `PRICE_PERIOD` — `₹299/month`, provisional.
- About → Mentor: `[Full name]`, `[Designation, Department]`, `[Institution]`, `[email]`.
- About → Contact: `[your email]`.

### Verified

Ran against the same `30X40 NORTH FACING HOUSE PLANS` blueprint: score still 42.9% on
8 rooms, overlay unchanged, all three nav pages render, Free shows one tier plus the lock
panel, upgrading switches the sidebar to **Pro** and renders all four tiers two-up, and
"Switch back to Free" re-locks them.
