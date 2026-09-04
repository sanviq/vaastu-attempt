"""
Owner: Sanvi
Thin glue only — wires detection -> rules_engine -> overlay in order. No logic lives here.
"""

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from detection import detect_rooms
from rules_engine import score_layout, generate_remodel_tiers
from overlay import draw_overlay, draw_remodel_tier

st.set_page_config(
    page_title="Vastu Blueprint Compliance Checker",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 1400px; }
      .hero { border-radius: 14px; padding: 1.6rem 1.8rem; margin-bottom: 1.4rem;
              background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8c 100%); }
      .hero h1 { color: #fff; margin: 0 0 .35rem 0; font-size: 1.9rem; }
      .hero p  { color: #cfe0f5; margin: 0; font-size: .95rem; }
      .tier-head { font-weight: 700; font-size: 1rem; margin: .2rem 0 .1rem 0; }
      .tier-sub  { color: #8b98a8; font-size: .82rem; margin-bottom: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧭 Vastu Blueprint Compliance Checker</h1>
      <p>Upload a floor plan to check each room against classical Vastu Shastra
         direction rules — and see what to change, ranked by how much you want to remodel.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Human-readable compass picker instead of a raw degree value.
_FACING = {
    "North ↑ (top of plan faces North)": 0.0,
    "North-East ↗": 45.0,
    "East → (top of plan faces East)": 90.0,
    "South-East ↘": 135.0,
    "South ↓ (top of plan faces South)": 180.0,
    "South-West ↙": 225.0,
    "West ← (top of plan faces West)": 270.0,
    "North-West ↖": 315.0,
}

_TIER_CAPTION = {
    "25": "Quick wins — fix the single worst violation.",
    "50": "Balanced — address the most serious half.",
    "75": "Major remodel — most issues resolved.",
    "100": "Full alignment — every violation addressed.",
}


@st.cache_data(show_spinner=False)
def _analyse(file_bytes: bytes, filename: str, north_angle: float):
    suffix = Path(filename).suffix or ".png"
    digest = hashlib.md5(file_bytes).hexdigest()[:12]
    image_path = str(Path(tempfile.gettempdir()) / f"vastu_{digest}{suffix}")
    with open(image_path, "wb") as f:
        f.write(file_bytes)

    rooms = detect_rooms(image_path, north_angle)
    result = score_layout(rooms)
    tiers = generate_remodel_tiers(result["room_results"])
    return image_path, result, tiers


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

if uploaded_file is None:
    st.info("Upload a floor plan (PNG, JPG or PDF) to run the compliance check.")
    st.stop()

with st.spinner("Detecting rooms and checking Vastu compliance…"):
    image_path, result, tiers = _analyse(uploaded_file.getvalue(), uploaded_file.name, north_angle)

room_results = result["room_results"]
counts = {k: sum(1 for r in room_results if r["classification"] == k)
          for k in ("Compliant", "Moderate", "Non-Compliant")}

score = result["overall_score"]
verdict = "Strong Vastu alignment" if score >= 75 else (
    "Partial alignment — worth remodelling" if score >= 45 else "Weak alignment — major changes advised"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Overall compliance", f"{score}%", help=verdict)
m2.metric("Compliant rooms", counts["Compliant"])
m3.metric("Moderate", counts["Moderate"])
m4.metric("Non-compliant", counts["Non-Compliant"])
st.caption(verdict)

tab_overview, tab_rooms, tab_remodel = st.tabs(
    ["Compliance overlay", "Room-by-room", "Remodelling options"]
)

with tab_overview:
    st.image(draw_overlay(image_path, room_results), use_container_width=True)

with tab_rooms:
    st.dataframe(
        [
            {
                "Room": r["room_label"],
                "Direction": r["direction"],
                "Matched rule": r["canonical_room"] or "—",
                "Verdict": r["classification"],
                "Source agreement": f"{int(r['agreement_level'] * 100)}%"
                                    f" ({r['n_sources_confirming']}/{r['n_sources_total']})",
                "Notes": r["notes"] or "",
            }
            for r in room_results
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Vastu is not a codified standard — classical sources genuinely disagree. "
        "'Source agreement' shows how many of the cited texts back each verdict."
    )

with tab_remodel:
    st.markdown(
        "Each option below is cumulative — 25% fixes only the worst violation, "
        "100% addresses every one."
    )
    row1 = st.columns(2)
    row2 = st.columns(2)
    for col, pct in zip(row1 + row2, ("25", "50", "75", "100")):
        with col:
            st.markdown(f'<div class="tier-head">{pct}% remodel</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="tier-sub">{_TIER_CAPTION[pct]} · '
                f'{len(tiers[pct])} room(s)</div>',
                unsafe_allow_html=True,
            )
            st.image(draw_remodel_tier(image_path, tiers[pct]), use_container_width=True)
