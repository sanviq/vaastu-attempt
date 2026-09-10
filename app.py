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
from payments import (
    TEST_CARD,
    cleanup_stray_payment_params,
    create_order,
    handle_payment_return,
    is_configured,
    render_checkout,
    sync_pending_payment,
)

st.set_page_config(
    page_title="VAASTU WISE",
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
        /* Japanese Cherry Blossom (Yozakura / 夜桜) Theme Palette */
        --bg: #0E0913;
        --bg2: #190F22;
        --card: rgba(255, 238, 246, 0.038);
        --card-hi: rgba(255, 218, 238, 0.075);
        --border: rgba(255, 182, 213, 0.14);
        --text: #FDF0F6;
        --muted: #A88EA2;
        --good: #3ED185;
        --warn: #F5A623;
        --bad: #FF4D6D;
        --none: #8E7688;
        --accent: #FF7597;
        --accent-glow: rgba(255, 117, 151, 0.28);
        --accent-soft: rgba(255, 117, 151, 0.14);
        --accent-hover: #FF8EA9;
      }
      .stApp { background: radial-gradient(1200px 600px at 15% -10%, #2A132C 0%, #1A0D22 35%, var(--bg) 70%); }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }

      .hero { position:relative; border:1px solid var(--border); border-radius:16px;
              padding:1.5rem 1.7rem; margin-bottom:1.3rem; overflow:hidden;
              background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 12%, transparent) 0%, rgba(186, 104, 200, 0.10) 100%), var(--bg2); }
      .hero h1 { color:var(--text); margin:0 0 .3rem 0; font-size:1.5rem; letter-spacing:-.5px; font-weight:700; }
      .hero p  { color:#D6C2D0; margin:0; font-size:.88rem; max-width:64ch; }

      .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:.2rem 0 1rem 0; }
      .stat { background:var(--card); border:1px solid var(--border); border-radius:14px;
              padding:.95rem 1.1rem; transition:background .15s ease, border-color .15s ease; }
      .stat:hover { background:var(--card-hi); border-color:rgba(255, 182, 213, 0.25); }
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
              background:var(--card); margin-bottom:.45rem; transition:background .15s ease; }
      .room:hover { background:var(--card-hi); }
      .room .nm { font-weight:650; color:var(--text); font-size:.92rem; letter-spacing:.2px; }
      .room .rule { font-size:.72rem; color:var(--muted); margin-top:.12rem; }
      .room .note { font-size:.78rem; color:#DFC8D5; line-height:1.35; }
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

      /* --- plan badge, pricing, locked panels ------------------------------- */
      .chip { display:inline-block; padding:.16rem .55rem; border-radius:99px;
              font-size:.64rem; font-weight:700; text-transform:uppercase; letter-spacing:.8px; }
      .chip.free { background:rgba(168,142,162,.18); color:#D6C2D0; }
      .chip.pro  { background:color-mix(in srgb, var(--accent) 18%, transparent); color:var(--accent); }
      .chip.demo { background:color-mix(in srgb, var(--warn) 16%, transparent); color:var(--warn); }

      .plan { border:1px solid var(--border); background:var(--card); border-radius:16px;
              padding:1.3rem 1.4rem; height:100%; transition:border-color .15s ease; }
      .plan.hi { border-color:color-mix(in srgb, var(--accent) 40%, transparent);
                 background:linear-gradient(160deg, color-mix(in srgb, var(--accent) 10%, transparent) 0%, var(--card) 60%);
                 box-shadow:0 4px 28px rgba(255, 117, 151, 0.08); }
      .plan h3 { color:var(--text); font-size:1.05rem; margin:.5rem 0 .1rem 0; font-weight:700; }
      .plan .price { font-size:2rem; font-weight:700; color:var(--text); letter-spacing:-1.5px;
                     margin:.35rem 0 .1rem 0; }
      .plan .price small { font-size:.8rem; font-weight:500; color:var(--muted); letter-spacing:0; }
      .plan .blurb { color:var(--muted); font-size:.78rem; margin-bottom:.9rem; }
      .plan ul { list-style:none; padding:0; margin:0; }
      .plan li { color:#DFC8D5; font-size:.82rem; padding:.3rem 0 .3rem 1.35rem;
                 position:relative; line-height:1.4; }
      .plan li:before { content:"✓"; position:absolute; left:0; color:var(--good); font-weight:700; }
      .plan li.off { color:var(--muted); }
      .plan li.off:before { content:"—"; color:var(--muted); }

      .lock { border:1px dashed color-mix(in srgb, var(--accent) 35%, transparent); border-radius:16px; padding:1.6rem 1.5rem;
              background:linear-gradient(160deg, color-mix(in srgb, var(--accent) 6%, transparent) 0%, var(--card) 70%);
              text-align:center; margin:.6rem 0 1rem 0; }
      .lock .t { color:var(--text); font-weight:700; font-size:1.02rem; margin-bottom:.3rem; }
      .lock .c { color:var(--muted); font-size:.82rem; max-width:56ch; margin:0 auto; line-height:1.5; }

      .sec { border:1px solid var(--border); background:var(--card); border-radius:16px;
             padding:1.3rem 1.5rem; margin-bottom:.9rem; }
      .sec h3 { color:var(--text); font-size:1rem; margin:0 0 .55rem 0; font-weight:700;
                letter-spacing:-.2px; }
      .sec p, .sec li { color:#DFC8D5; font-size:.86rem; line-height:1.62; }
      .sec p { margin:0 0 .6rem 0; }
      .sec p:last-child { margin-bottom:0; }
      .sec ol, .sec ul { margin:0; padding-left:1.15rem; }
      .sec li { margin-bottom:.4rem; }
      .sec b, .sec strong { color:var(--text); font-weight:650; }
      .sec a { color:var(--accent); text-decoration:none; }
      .sec a:hover { text-decoration:underline; }

      .who { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.7rem; }
      /* A single card in an auto-fit grid stretches the full width and reads as a
         banner rather than a person. Cap it at roughly one column. */
      .who.one { grid-template-columns:minmax(210px,340px); }
      .who .p { border:1px solid var(--border); border-radius:12px; padding:.85rem 1rem;
                background:rgba(255,255,255,.02); }
      .who .p .n { color:var(--text); font-weight:700; font-size:.9rem; }
      .who .p .r { color:var(--accent); font-size:.68rem; text-transform:uppercase;
                   letter-spacing:.9px; font-weight:600; margin:.2rem 0 .35rem 0; }
      .who .p .d { color:var(--muted); font-size:.78rem; line-height:1.45; }

      .note-box { border:1px solid rgba(245,166,35,.25); background:rgba(245,166,35,.05);
                  border-radius:12px; padding:.85rem 1.1rem; }
      .note-box p { color:#E5C396; font-size:.8rem; margin:0; line-height:1.55; }

      /* Streamlit's primary button is red by default, which reads as a warning
         next to green/amber/red verdict pills. Repaint it in the accent. */
      .stButton button[kind="primary"],
      .stButton button[data-testid="stBaseButton-primary"] {
        background:var(--accent); border-color:var(--accent); color:#1B0612;
        font-weight:700; letter-spacing:.2px;
      }
      .stButton button[kind="primary"]:hover,
      .stButton button[data-testid="stBaseButton-primary"]:hover {
        background:var(--accent-hover, #FF8EA9); border-color:var(--accent-hover, #FF8EA9); color:#1B0612;
        box-shadow:0 0 16px rgba(255, 117, 151, 0.35);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Plan state
# ---------------------------------------------------------------------------
# Paid access is session-scoped. With Razorpay keys configured, upgrading runs
# through real test-mode checkout; without keys it falls back to one-click demo.
if "pro" not in st.session_state:
    st.session_state.pro = False
if "plan_tier" not in st.session_state:
    st.session_state.plan_tier = None
if "pending_checkout" not in st.session_state:
    st.session_state.pending_checkout = None
if "checkout_opened_for" not in st.session_state:
    st.session_state.checkout_opened_for = None
if "last_order" not in st.session_state:
    st.session_state.last_order = None

cleanup_stray_payment_params()

if handle_payment_return():
    st.toast(f"Payment successful — {st.session_state.plan_tier.title()} plan unlocked.", icon="✅")
    st.rerun()

if (
    not st.session_state.pro
    and is_configured()
):
    for pending in (
        st.session_state.pending_checkout,
        st.session_state.last_order,
    ):
        if pending and sync_pending_payment(pending):
            st.toast(
                f"Payment successful — {st.session_state.plan_tier.title()} plan unlocked.",
                icon="✅",
            )
            st.rerun()
            break

PRICE_SMALL = "₹30"
PRICE_BIG = "₹100"
PRICE_PERIOD = "/month"

_FREE_FEATURES = [
    ("Overall compliance score", True),
    ("Zone overlay on your plan", True),
    ("Room-by-room verdicts with cited sources", True),
    ("25% quick-wins remodel plan", True),
    ("50 / 75 / 100% remodel plans", False),
    ("Unlimited plans", False),
]

_SMALL_FEATURES = [
    ("Apartment & small floor plans", True),
    ("Everything in Free", True),
    ("All four remodel tiers, side by side", True),
    ("50 / 75 / 100% remodel overlays", True),
    ("Unlimited small plans", True),
]

_BIG_FEATURES = [
    ("Villa, bungalow & large floor plans", True),
    ("Everything in Small", True),
    ("All four remodel tiers, side by side", True),
    ("Unlimited large plans", True),
    ("Priority support", True),
]

# ---------------------------------------------------------------------------
# Sidebar — navigation and plan state
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.05rem;font-weight:700;color:#fff;letter-spacing:-.3px;">'
        "🧭 VAASTU WISE</div>"
        '<div style="color:#6E6E9A;font-size:.72rem;margin-bottom:.9rem;">'
        "Blueprint compliance, with sources</div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Section", ["Analyser", "Pricing", "About"], label_visibility="collapsed"
    )

    if st.session_state.pro:
        tier = (st.session_state.plan_tier or "paid").title()
        plan_chip = f'<span class="chip pro">{tier} plan</span>'
    else:
        plan_chip = '<span class="chip free">Free plan</span>'
    billing_chip = (
        '<span class="chip demo">Test payments</span>'
        if is_configured()
        else '<span class="chip demo">Demo billing</span>'
    )
    st.markdown(
        f'<div style="margin:.9rem 0 .5rem 0;">{plan_chip} {billing_chip}</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.pro:
        if st.button("Switch back to Free", use_container_width=True):
            st.session_state.pro = False
            st.session_state.plan_tier = None
            st.session_state.pop("payment_id", None)
            st.session_state.pending_checkout = None
            st.session_state.checkout_opened_for = None
            st.rerun()
    if is_configured():
        st.caption(
            f"Razorpay test mode — use RuPay test card {TEST_CARD}, any future expiry, any CVV."
        )
    else:
        st.caption(
            "Demo mode — no real payment. Add Razorpay test keys in "
            "`.streamlit/secrets.toml` to enable checkout."
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

# Tiers above this are Pro-only. 25% stays free so the free tier still ends in
# something actionable rather than a diagnosis with no prescription.
_FREE_TIER = "25"


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


def _feature_list(features) -> str:
    return "".join(
        f'<li class="{"" if on else "off"}">{html.escape(text)}</li>'
        for text, on in features
    )


def _pay_button(plan_id: str, key: str, label: str) -> None:
    """Start Razorpay checkout, or instant demo unlock when keys are not set."""
    if st.button(label, key=key, type="primary", use_container_width=True):
        if is_configured():
            try:
                order = create_order(plan_id)
                st.session_state.pending_checkout = order
                st.session_state.last_order = order
                st.session_state.checkout_opened_for = None
            except Exception as exc:
                st.error(f"Could not start checkout: {exc}")
                return
        else:
            st.session_state.pro = True
            st.session_state.plan_tier = plan_id
        st.rerun()

    pending = st.session_state.get("pending_checkout")
    if pending and pending.get("plan_id") == plan_id and is_configured():
        if st.session_state.checkout_opened_for != pending["id"]:
            st.session_state.checkout_opened_for = pending["id"]
            render_checkout(pending)
        else:
            st.caption("Razorpay should open over the full page. Click Pay again if it did not.")


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------
def render_analyser() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🧭 VAASTU WISE</h1>
          <p>Upload a floor plan. Each room is read off the drawing, placed on the Vastu Purusha
             Mandala, and checked against classical sources — with what to change, ranked by how
             much you are willing to remodel.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        elif st.session_state.pro:
            row1 = st.columns(2)
            row2 = st.columns(2)
            for col, pct in zip(row1 + row2, ("25", "50", "75", "100")):
                with col:
                    _render_tier(image_path, tiers, pct, north_angle)
        else:
            # Free tier still ends in something you can act on: the single worst
            # violation, fixed. The remaining three tiers are the paid step up.
            #
            # Held to half width on purpose. In Pro these render two-up, so a lone
            # full-width tier would be the only oversized plan in the app — the same
            # coarse, stretched look the fixed render size was meant to end.
            tier_l, tier_r = st.columns(2)
            with tier_l:
                _render_tier(image_path, tiers, _FREE_TIER, north_angle)
            locked = [p for p in ("50", "75", "100") if tiers[p]]
            st.markdown(
                '<div class="lock">'
                f'<div class="t">🔒 {len(locked)} more remodel plans in Pro</div>'
                '<div class="c">The 50%, 75% and 100% plans show every remaining room '
                'that sits against the sources, with the direction each should move to. '
                'Free covers the single worst violation only.</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            pad_l, mid, pad_r = st.columns([1, 2, 1])
            with mid:
                _pay_button("small", "up_remodel", f"Unlock all remodel plans — from {PRICE_SMALL}{PRICE_PERIOD}")


def _render_tier(image_path: str, tiers: dict, pct: str, north_angle: float) -> None:
    st.markdown(
        f'<div class="tier"><div class="t">{pct}% remodel</div>'
        f'<div class="c">{_TIER_CAPTION[pct]} · {len(tiers[pct])} room(s)</div></div>',
        unsafe_allow_html=True,
    )
    st.image(
        draw_remodel_tier(image_path, tiers[pct], north_angle),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def render_pricing() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Plans</h1>
          <p>Check any plan for free and see exactly where it stands. Paid plans unlock the full
             set of remodelling options — priced by plan size.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_free, col_small, col_big = st.columns(3)
    with col_free:
        st.markdown(
            '<div class="plan">'
            '<span class="chip free">Free</span>'
            "<h3>Check a plan</h3>"
            f'<div class="price">₹0</div>'
            '<div class="blurb">Everything you need to know where you stand.</div>'
            f"<ul>{_feature_list(_FREE_FEATURES)}</ul>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_small:
        st.markdown(
            '<div class="plan hi">'
            '<span class="chip pro">Small</span>'
            "<h3>Small plans</h3>"
            f'<div class="price">{PRICE_SMALL}<small>{PRICE_PERIOD}</small></div>'
            '<div class="blurb">Apartments, flats and compact floor plans.</div>'
            f"<ul>{_feature_list(_SMALL_FEATURES)}</ul>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.session_state.pro:
            tier = (st.session_state.plan_tier or "paid").title()
            st.success(f"You are on the {tier} plan for this session.")
        else:
            _pay_button("small", "up_pricing_small", f"Pay {PRICE_SMALL}{PRICE_PERIOD}")
    with col_big:
        st.markdown(
            '<div class="plan hi">'
            '<span class="chip pro">Big</span>'
            "<h3>Big plans</h3>"
            f'<div class="price">{PRICE_BIG}<small>{PRICE_PERIOD}</small></div>'
            '<div class="blurb">Villas, bungalows and large residential layouts.</div>'
            f"<ul>{_feature_list(_BIG_FEATURES)}</ul>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.session_state.pro:
            tier = (st.session_state.plan_tier or "paid").title()
            st.success(f"You are on the {tier} plan for this session.")
        else:
            _pay_button("big", "up_pricing_big", f"Pay {PRICE_BIG}{PRICE_PERIOD}")

    st.write("")
    if is_configured():
        st.markdown(
            '<div class="note-box"><p><b>Razorpay test mode.</b> Payments are real in Razorpay\'s '
            "sandbox — no money leaves a real account. Use RuPay test card "
            f"<b>{TEST_CARD}</b>, any future expiry, any CVV. After a successful payment the paid "
            "plan unlocks for this browser session. For production you will switch to live keys "
            "and add accounts so subscriptions survive a reload.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="note-box"><p><b>Demo billing.</b> Razorpay keys are not configured, so '
            "Pay buttons unlock the plan instantly with no checkout. To show real payment flow, "
            "copy <code>.streamlit/secrets.toml.example</code> to "
            "<code>.streamlit/secrets.toml</code> and add your Razorpay <b>test</b> API keys.</p></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
def render_about() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>About VAASTU WISE 🌸</h1>
          <p>Automated architectural spatial intelligence bridging computer vision, the Vastu Purusha Mandala, and peer-reviewed architectural research.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sec">
          <h3>What This Is</h3>
          <p>Traditional Vastu consultations frequently depend on subjective personal interpretations without empirical reasoning or literature citations. Blueprints are evaluated by eye, and conflicting guidelines are rarely disclosed.</p>
          <p><b>VAASTU WISE</b> transforms this process through computational objectivity. By ingesting actual floor plans, the system applies image processing and optical character recognition to locate rooms, mathematically projects them onto the calibrated <b>Vastu Purusha Mandala</b>, and evaluates each space against published academic and architectural literature. When authoritative sources disagree, that divergence is transparently reported with source-level attribution rather than obscured by a generic average.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sec">
          <h3>How the Pipeline Works</h3>
          <ol>
            <li><b>Blueprint Vision &amp; OCR Ingestion.</b> The system processes architectural drawings through Gaussian filtering, adaptive thresholding, and morphological operations. Wall lines and room boundaries are detected via contour analysis and Probabilistic Hough Line Transforms. Room labels (such as <i>Kitchen</i>, <i>Master Bedroom</i>, <i>Pooja</i>, or <i>Toilet</i>) are detected via OCR directly from blueprint annotations to serve as ground-truth spatial anchors.</li>
            <li><b>Mandala Coordinate Calibration.</b> The blueprint is mapped to the sacred 9-zone Vastu Purusha Mandala (North, North-East, East, South-East, South, South-West, West, North-West, and Central Brahmasthan). The grid dynamically aligns with the user-specified North compass orientation, calculating geometric centroids for each detected space.</li>
            <li><b>Cross-Referenced Compliance Engine.</b> Each room and orientation pair is verified against our rule dataset. Placements are classified as <b>Compliant</b> (in preferred zones), <b>Moderate</b> (in acceptable or neutral zones), or <b>Non-Compliant</b> (in zones explicitly cautioned against by scholars). An agreement metric quantifies consensus across published sources.</li>
            <li><b>Progressive Tiered Remodelling.</b> Deviations are prioritized by severity and published consensus. Four actionable remediation tiers — <b>25%</b> (quick wins resolving the most critical conflicts), <b>50%</b>, <b>75%</b>, and <b>100%</b> (comprehensive structural harmony) — provide incremental layout solutions with directional relocation vectors.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sec">
          <h3>The Literature &amp; Rule Dataset</h3>
          <p>Our rule corpus comprises <b>13 functional room types</b> cross-referenced across <b>five foundational publications</b> in Vastu Shastra and computational architectural layout design:</p>
          <ul>
            <li><b>Optimal floor plan for residential houses using Eastern concepts of Vaastu Shastra: an artificial intelligence approach</b> — <i>Asian Journal of Civil Engineering</i> (Springer, 2023).</li>
            <li><b>Utility of the Ancient Indian Science of Vaastu in Modern Architecture</b> — Peertechz Journal of Engineering.</li>
            <li><b>Vastu Shastra: An Established Science</b> (Parts II &amp; VI: Guidelines on Bedrooms and Cardinal Orientations).</li>
            <li><b>Vastu Shastra: A Traditional Indian Architectural Science</b> — Research Compendium.</li>
            <li><b>Sense of Direction in Vaastu Shastra</b> — Spatial Alignment Monograph.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sec">
          <h3>Engineering Team</h3>
          <div class="who">
            <div class="p">
              <div class="n">Sanvi Agarwal</div>
              <div class="r">Compliance Engine &amp; Product Architecture</div>
              <div class="d">Engineered the Vastu rules engine, multi-source scoring algorithms, progressive remediation tiers, payment flow, and application core.</div>
            </div>
            <div class="p">
              <div class="n">Harshika</div>
              <div class="r">Computer Vision &amp; Visual Intelligence</div>
              <div class="d">Developed the blueprint ingestion pipeline, contour and Hough line boundary detection, Tesseract OCR room recognition, and the visual compliance overlays.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sec">
          <h3>Academic Guidance &amp; Advisory</h3>
          <div class="who one">
            <div class="p">
              <div class="n">Faculty Project Advisory</div>
              <div class="r">Department of Computer Science &amp; Engineering</div>
              <div class="d">Academic and research supervision supporting computer vision evaluation, rule-engine formalization, and design architecture.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sec">
          <h3>Research &amp; Feedback</h3>
          <p>Have inquiries regarding rule formulations, blueprint detection fidelity, or dataset expansion? We welcome feedback from architects, researchers, and engineers.</p>
          <p>Reach out to the development team at <a href="mailto:vaastuwise.project@gmail.com">vaastuwise.project@gmail.com</a>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="note-box"><p><b>Advisory Disclaimer.</b> This platform provides analytical guidance derived from published historical and contemporary Vastu literature. It is not architectural, structural, or civil engineering advice, and does not evaluate structural load, soil mechanics, or municipal building regulations. Consult a licensed architect or structural engineer before making any structural changes to a property.</p></div>',
        unsafe_allow_html=True,
    )


_PAGES = {"Analyser": render_analyser, "Pricing": render_pricing, "About": render_about}
_PAGES[page]()