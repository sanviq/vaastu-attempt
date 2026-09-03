"""
Owner: Sanvi
Thin glue only — wires detection -> rules_engine -> overlay in order. No logic lives here.
"""

import tempfile
from pathlib import Path

import streamlit as st

from detection import detect_rooms
from rules_engine import score_layout, generate_remodel_tiers
from overlay import draw_overlay, draw_remodel_tier

st.title("Vastu Blueprint Compliance Checker")

uploaded_file = st.file_uploader("Upload floor plan", type=["png", "jpg", "jpeg", "pdf"])
north_angle = st.number_input("North angle (degrees, 0 = image-up)", min_value=0.0, max_value=360.0, value=0.0)

if uploaded_file is not None:
    image_path = str(Path(tempfile.gettempdir()) / uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    rooms = detect_rooms(image_path, north_angle)
    result = score_layout(rooms)
    tiers = generate_remodel_tiers(result["room_results"])

    st.metric("Overall compliance score", f"{result['overall_score']}%")

    overlay_image = draw_overlay(image_path, result["room_results"])
    st.image(overlay_image, caption="Compliance overlay")

    tier_pct = st.select_slider("Remodelling tier", options=["25", "50", "75", "100"])
    remodel_image = draw_remodel_tier(image_path, tiers[tier_pct])
    st.image(remodel_image, caption=f"{tier_pct}% remodel suggestions")
