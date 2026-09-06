"""
Owner: Sanvi
Thin glue only — wires detection -> rules_engine -> overlay in order. No logic lives here.
"""

import hashlib
import html
import tempfile
from pathlib import Path

import streamlit as st

from detection import detect_rooms, ocr_available
from rules_engine import score_layout, generate_remodel_tiers
from overlay import draw_overlay, draw_remodel_tier

st.set_page_config(
    page_title="Vastu Blueprint Compliance Checker",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
# Dark translucent cards, uppercase tracked labels, big colour-coded figures.
# The three verdict hues are the same ones overlay.py paints on the plan, lifted
# for legibility on a dark ground — the pill next to a room must read as the same
# colour as the box drawn around it.
st.markdown(
    """
    <style>
      :root {
        --bg:#07070F; --bg2:#0E0E1B; --card:rgba(255,255,255,.035);
        --card-hi:rgba(255,255,255,.06); --border:rgba(255,255,255,.08);
        --text:#E8E8F4; --muted:#6E6E9A;
        --good:#35C77E; --warn:#E8A33D; --bad:#F0556B; --none:#6E6E9A;
        --accent:#00E5FF;
      }
      .stApp { background: radial-gradient(1200px 600px at 15% -10%, #14142B 0%, var(--bg) 55%); }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }

      .hero { position:relative; border:1px solid var(--border); border-radius:16px;
              padding:1.5rem 1.7rem; margin-bottom:1.3rem; overflow:hidden;
              background: linear-gradient(135deg, rgba(0,229,255,.10) 0%, rgba(157,78,221,.10) 100%), var(--bg2); }
      .hero h1 { color:#fff; margin:0 0 .3rem 0; font-size:1.5rem; letter-spacing:-.5px; font-weight:700; }
      .hero p  { color:#A9A9C8; margin:0; font-size:.88rem; max-width:64ch; }

      .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:.2rem 0 1rem 0; }
      .stat { background:var(--card); border:1px solid var(--border); border-radius:14px;
              padding:.95rem 1.1rem; transition:background .15s ease; }
      .stat:hover { background:var(--card-hi); }
      .stat .k { font-size:.66rem; text-transform:uppercase; letter-spacing:1px;
                 color:var(--muted); font-weight:600; margin-bottom:.35rem; }
      .stat .v { font-size:1.95rem; font-weight:700; letter-spacing:-1px; line-height:1.05; }
      .stat .s { font-size:.72rem; color:var(--muted); margin-top:.2rem; }
      .v.good{color:var(--good)} .v.warn{color:var(--warn)} .v.bad{color:var(--bad)}
      .v.accent{color:var(--accent)} .v.none{color:var(--none)}

      .bar { display:flex; height:8px; border-radius:99px; overflow:hidden;
             border:1px solid var(--border); margin:.1rem 0 .35rem 0; }
      .bar i { display:block; height:100%; }

      .room { display:grid; grid-template-columns:1.4fr .5fr .9fr 2.4fr; gap:.9rem;
              align-items:center; padding:.72rem .95rem; border:1px solid var(--border);
              border-left:3px solid var(--edge); border-radius:11px;
              background:var(--card); margin-bottom:.45rem; }
      .room:hover { background:var(--card-hi); }
      .room .nm { font-weight:650; color:var(--text); font-size:.92rem; letter-spacing:.2px; }
      .room .rule { font-size:.72rem; color:var(--muted); margin-top:.12rem; }
      .room .note { font-size:.78rem; color:#9C9CBE; line-height:1.35; }
      .dir { font-size:.9rem; font-weight:700; color:var(--text); letter-spacing:.5px; }
      .pill { display:inline-block; padding:.2rem .6rem; border-radius:99px;
              font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.6px; }
      .agree { font-size:.68rem; color:var(--muted); margin-top:.28rem; }

      .thead { display:grid; grid-template-columns:1.4fr .5fr .9fr 2.4fr; gap:.9rem;
               padding:0 .95rem .4rem .95rem; font-size:.66rem; text-transform:uppercase;
               letter-spacing:1px; color:var(--muted); font-weight:600; }

      .tier { border:1px solid var(--border); background:var(--card);
              border-radius:12px; padding:.6rem .85rem; margin-bottom:.5rem; }
      .tier .t { font-weight:700; font-size:.95rem; color:var(--text); }
      .tier .c { color:var(--muted); font-size:.76rem; margin-top:.1rem; }

      .stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--border); }
      .stTabs [data-baseweb="tab"] { font-size:.85rem; font-weight:600; letter-spacing:.2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧭 Vastu Blueprint Compliance Checker</h1>
      <p>Upload a floor plan. Each room is read off the drawing, placed on the Vastu Purusha
         Mandala, and checked against classical sources — with what to change, ranked by how
         much you are willing to remodel.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Human-readable compass picker instead of a raw degree value.
#
# detect_rooms(north_angle=...) wants the image bearing of North, i.e. where North
# sits once the plan is on screen. The question below asks the inverse (where the
# top of the plan points), so the two are related by (360 - facing) % 360 —
# keep these in step, they are easy to invert by accident.
_FACING = {
    "North ↑ (top of plan faces North)": 0.0,
    "North-East ↗": 315.0,
    "East → (top of plan faces East)": 270.0,
    "South-East ↘": 225.0,
    "South ↓ (top of plan faces South)": 180.0,
    "South-West ↙": 135.0,
    "West ← (top of plan faces West)": 90.0,
    "North-West ↖": 45.0,
}

_TIER_CAPTION = {
    "25": "Quick wins — fix the single worst violation.",
    "50": "Balanced — address the most serious half.",
    "75": "Major remodel — most issues resolved.",
    "100": "Full alignment — every violation addressed.",
}

_VERDICT_VAR = {
    "Compliant": "--good",
    "Moderate": "--warn",
    "Non-Compliant": "--bad",
    "Unknown": "--none",
}


def _materialise(file_bytes: bytes, filename: str) -> str:
    """
    Write the upload to a temp file and return a path the vision code can read.

    PDFs are rasterised here rather than downstream: detection.py can open a PDF
    but overlay.py uses cv2.imread, which cannot, so a PDF upload used to crash
    at the overlay step. Converting once up front keeps both on the same image.
    """
    suffix = Path(filename).suffix.lower() or ".png"
    digest = hashlib.md5(file_bytes).hexdigest()[:12]
    raw_path = Path(tempfile.gettempdir()) / f"vastu_{digest}{suffix}"
    raw_path.write_bytes(file_bytes)

    if suffix != ".pdf":
        return str(raw_path)

    png_path = raw_path.with_suffix(".png")
    if not png_path.exists():
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise RuntimeError(
                "PDF support needs pdf2image — run `pip install pdf2image`, "
                "or export your plan as PNG/JPG."
            ) from exc
        try:
            pages = convert_from_path(str(raw_path), dpi=150, first_page=1, last_page=1)
        except Exception as exc:
            raise RuntimeError(
                "Could not read that PDF. pdf2image needs Poppler installed "
                "(`brew install poppler`). Exporting the plan as PNG also works."
            ) from exc
        pages[0].save(png_path, "PNG")
    return str(png_path)


@st.cache_data(show_spinner=False)
def _analyse(file_bytes: bytes, filename: str, north_angle: float):
    image_path = _materialise(file_bytes, filename)
    rooms = detect_rooms(image_path, north_angle)
    result = score_layout(rooms)
    tiers = generate_remodel_tiers(result["room_results"])
    return image_path, result, tiers


def _stat(label: str, value: str, sub: str, tone: str) -> str:
    return (f'<div class="stat"><div class="k">{label}</div>'
            f'<div class="v {tone}">{value}</div><div class="s">{sub}</div></div>')


left, right = st.columns([2, 1])
with left:
    uploaded_file = st.file_uploader(
        "Floor plan", type=["png", "jpg", "jpeg", "pdf"], label_visibility="collapsed"
    )
with right:
    facing = st.selectbox(
        "Which direction does the **top** of your plan point to?",
        list(_FACING),
        help="Check the north arrow on your blueprint. If there isn't one, most plans are drawn with North at the top.",
    )
north_angle = _FACING[facing]

if not ocr_available():
    st.warning(
        "**Tesseract OCR is not installed**, so the room names printed on the plan cannot "
        "be read — and room names are how rooms are found. Install it with "
        "`brew install tesseract` (macOS) or `sudo apt install tesseract-ocr` (Linux), "
        "then restart the app."
    )

if uploaded_file is None:
    st.info("Upload a floor plan (PNG, JPG or PDF) to run the compliance check.")
    st.stop()

try:
    with st.spinner("Reading room labels and checking Vastu compliance…"):
        image_path, result, tiers = _analyse(
            uploaded_file.getvalue(), uploaded_file.name, north_angle
        )
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

room_results = result["room_results"]

if not room_results:
    st.error(
        "**No rooms could be detected in this plan.** Rooms are found by reading the "
        "printed room names (KITCHEN, BEDROOM, TOILET…) off the drawing, so this "
        "usually means the labels are too small to read, handwritten, or absent. "
        "Try a higher-resolution export where the room names are clearly legible."
    )
    st.stop()

counts = {k: sum(1 for r in room_results if r["classification"] == k)
          for k in ("Compliant", "Moderate", "Non-Compliant", "Unknown")}

score = result["overall_score"]
judged = result["n_judged"]
verdict = "Strong Vastu alignment" if score >= 75 else (
    "Partial alignment — worth remodelling" if score >= 45 else "Weak alignment — major changes advised"
)
tone = "good" if score >= 75 else ("warn" if score >= 45 else "bad")

st.markdown(
    '<div class="stats">'
    + _stat("Overall compliance", f"{score}%", verdict, tone)
    + _stat("Compliant", str(counts["Compliant"]), "in a preferred zone", "good")
    + _stat("Moderate", str(counts["Moderate"]), "partly aligned", "warn")
    + _stat("Non-compliant", str(counts["Non-Compliant"]), "against the sources", "bad")
    + "</div>",
    unsafe_allow_html=True,
)

# Proportional bar — the same split as the numbers above, read at a glance.
segments = [("Compliant", "--good"), ("Moderate", "--warn"),
            ("Non-Compliant", "--bad"), ("Unknown", "--none")]
total = len(room_results)
st.markdown(
    '<div class="bar">'
    + "".join(
        f'<i style="width:{counts[name] / total * 100:.2f}%;background:var({var})"></i>'
        for name, var in segments if counts[name]
    )
    + "</div>",
    unsafe_allow_html=True,
)

scored_note = f"Scored on {judged} of {total} rooms"
if counts["Unknown"]:
    scored_note += f" · {counts['Unknown']} room type not covered by the dataset"
st.caption(scored_note)

tab_overview, tab_rooms, tab_remodel = st.tabs(
    ["Compliance overlay", "Room-by-room", "Remodelling options"]
)

with tab_overview:
    # Rendered at its natural size, not stretched to the container. overlay.py
    # draws at a fixed ~1000px width; letting Streamlit scale that up again is
    # what made the labels and borders look coarse.
    overlay_img = draw_overlay(image_path, room_results, north_angle)
    pad_l, mid, pad_r = st.columns([1, 8, 1])
    with mid:
        st.image(overlay_img, width=min(920, overlay_img.shape[1]))

with tab_rooms:
    st.markdown(
        '<div class="thead"><div>Room</div><div>Zone</div>'
        '<div>Verdict</div><div>Why</div></div>',
        unsafe_allow_html=True,
    )
    rows = []
    for r in room_results:
        var = _VERDICT_VAR.get(r["classification"], "--none")
        agreement = (
            f'{r["n_sources_confirming"]}/{r["n_sources_total"]} sources'
            if r["n_sources_total"] else "not in dataset"
        )
        rows.append(
            f'<div class="room" style="--edge:var({var})">'
            f'<div><div class="nm">{html.escape(r["room_label"])}</div>'
            f'<div class="rule">{html.escape(r["canonical_room"] or "no matching rule")}</div></div>'
            f'<div class="dir">{html.escape(r["direction"])}</div>'
            f'<div><span class="pill" style="background:color-mix(in srgb,var({var}) 18%,transparent);'
            f'color:var({var})">{html.escape(r["classification"])}</span>'
            f'<div class="agree">{agreement}</div></div>'
            f'<div class="note">{html.escape(r["notes"] or "")}</div>'
            f"</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)

    cited = sorted({s for r in room_results for s in r.get("sources", [])})
    if cited:
        with st.expander(f"Sources cited ({len(cited)})"):
            for title in cited:
                st.markdown(f"- {title}")
    st.caption(
        "Vastu is not a codified standard — classical sources genuinely disagree, and most "
        "state where a room *should* go rather than where it must not. A room outside every "
        "stated zone is judged by how far it sits from the nearest preferred one."
    )

with tab_remodel:
    st.markdown(
        "Each option is cumulative — 25% fixes only the worst violation, "
        "100% addresses every one."
    )
    if not tiers["100"]:
        st.success("Every room already sits in a compliant direction — nothing to remodel.")
    else:
        row1 = st.columns(2)
        row2 = st.columns(2)
        for col, pct in zip(row1 + row2, ("25", "50", "75", "100")):
            with col:
                st.markdown(
                    f'<div class="tier"><div class="t">{pct}% remodel</div>'
                    f'<div class="c">{_TIER_CAPTION[pct]} · {len(tiers[pct])} room(s)</div></div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    draw_remodel_tier(image_path, tiers[pct], north_angle),
                    use_container_width=True,
                )
