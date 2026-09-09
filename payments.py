"""
Razorpay checkout for VAASTU WISE paid plans.

Set keys in .streamlit/secrets.toml (see secrets.toml.example). Without keys the
app falls back to one-click demo upgrades so the rest of the UI still works.
"""

from __future__ import annotations

import json
import time
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

PLANS: dict[str, dict[str, Any]] = {
    "small": {
        "amount_paise": 3000,
        "label": "Small plans — apartments & compact floor plans",
        "price_display": "₹30",
    },
    "big": {
        "amount_paise": 10000,
        "label": "Big plans — villas & large layouts",
        "price_display": "₹100",
    },
}

# Shown in the UI — RuPay test card works on Razorpay India; 4111… is often rejected.
TEST_CARD = "6074 8200 0000 0009"


def _qp(name: str) -> str | None:
    """Read a query param (Streamlit may return a list when duplicated)."""
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _secrets() -> dict[str, str] | None:
    try:
        cfg = st.secrets["razorpay"]
        key_id = str(cfg["key_id"]).strip()
        key_secret = str(cfg["key_secret"]).strip()
        if key_id and key_secret and "xxxx" not in key_id:
            return {"key_id": key_id, "key_secret": key_secret}
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    return None


def is_configured() -> bool:
    return _secrets() is not None


def create_order(plan_id: str) -> dict[str, Any]:
    if plan_id not in PLANS:
        raise ValueError(f"Unknown plan: {plan_id}")

    cfg = _secrets()
    if not cfg:
        raise RuntimeError("Razorpay keys are not configured.")

    import razorpay

    plan = PLANS[plan_id]
    client = razorpay.Client(auth=(cfg["key_id"], cfg["key_secret"]))
    receipt = f"vaastu_{plan_id}_{int(time.time())}"
    order = client.order.create(
        {
            "amount": plan["amount_paise"],
            "currency": "INR",
            "receipt": receipt,
            "notes": {"plan": plan_id, "product": "VAASTU WISE"},
        }
    )
    return {
        "id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "plan_id": plan_id,
    }


def verify_payment(payment_id: str, order_id: str, signature: str) -> bool:
    cfg = _secrets()
    if not cfg:
        return False

    import razorpay

    client = razorpay.Client(auth=(cfg["key_id"], cfg["key_secret"]))
    try:
        client.utility.verify_payment_signature(
            {
                "razorpay_order_id": order_id,
                "razorpay_payment_id": payment_id,
                "razorpay_signature": signature,
            }
        )
        return True
    except razorpay.errors.SignatureVerificationError:
        return False


def sync_pending_payment(pending: dict[str, Any]) -> bool:
    """
    OTP / netbanking flows sometimes skip the JS callback. Ask Razorpay directly
    whether the pending order was captured.
    """
    order_id = pending.get("id")
    plan_id = pending.get("plan_id", "small")
    if not order_id or plan_id not in PLANS:
        return False

    cfg = _secrets()
    if not cfg:
        return False

    import razorpay

    client = razorpay.Client(auth=(cfg["key_id"], cfg["key_secret"]))
    try:
        resp = client.order.payments(order_id)
        for payment in resp.get("items", []):
            if payment.get("status") == "captured":
                st.session_state.pro = True
                st.session_state.plan_tier = plan_id
                st.session_state.payment_id = payment.get("id")
                st.session_state.pending_checkout = None
                st.session_state.checkout_opened_for = None
                return True
    except Exception:
        pass
    return False


def cleanup_stray_payment_params() -> None:
    """Drop ?plan= from failed Razorpay POST redirects (Streamlit only accepts GET)."""
    if _qp("plan") and not _qp("razorpay_payment_id"):
        st.query_params.clear()


def handle_payment_return() -> bool:
    """
    If the URL carries Razorpay success params, verify and unlock the plan.
    Returns True when a verified payment was applied.
    """
    payment_id = _qp("razorpay_payment_id")
    order_id = _qp("razorpay_order_id")
    signature = _qp("razorpay_signature")
    plan_id = _qp("plan") or "small"

    if not (payment_id and order_id and signature):
        return False

    if plan_id not in PLANS:
        plan_id = "small"

    if verify_payment(payment_id, order_id, signature):
        st.session_state.pro = True
        st.session_state.plan_tier = plan_id
        st.session_state.payment_id = payment_id
        st.session_state.pending_checkout = None
        st.session_state.checkout_opened_for = None
        st.query_params.clear()
        return True

    # URL params present but signature failed — still check order on Razorpay's side.
    if order_id and sync_pending_payment({"id": order_id, "plan_id": plan_id}):
        st.query_params.clear()
        return True

    st.session_state.pop("pending_checkout", None)
    st.query_params.clear()
    st.error("Payment could not be verified. If money was debited, contact support with your payment ID.")
    return False


def render_checkout(order: dict[str, Any], file_digest: str = "", filename: str = "", north_angle: float = 0.0) -> None:
    """Open Razorpay as a full-page modal (not inside Streamlit's iframe)."""
    cfg = _secrets()
    if not cfg:
        return

    plan_id = order["plan_id"]
    plan = PLANS[plan_id]
    options = {
        "key": cfg["key_id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "name": "VAASTU WISE",
        "description": plan["label"],
        "order_id": order["id"],
        "theme": {"color": "#00E5FF"},
        "prefill": {},
        "notes": {"file_digest": file_digest, "filename": filename, "north_angle": str(north_angle)},
    }

    # Streamlit renders this inside a small iframe. Razorpay's modal must be
    # created on window.top or it opens trapped in that iframe (tiny scroll box).
    html = f"""
    <script>
      (function () {{
        const options = {json.dumps(options)};
        const planId = {json.dumps(plan_id)};
        const topWin = window.top;
        const topDoc = topWin.document;

        function redirectAfterPay(response) {{
                    const params = new URLSearchParams({{
            razorpay_payment_id: response.razorpay_payment_id,
            razorpay_order_id: response.razorpay_order_id,
            razorpay_signature: response.razorpay_signature,
            plan: planId,
            file_digest: {json.dumps(file_digest)},
            filename: {json.dumps(filename)},
            north_angle: {json.dumps(str(north_angle))},
          }});
          const base = topWin.location.origin + topWin.location.pathname;
          topWin.location.href = base + "?" + params.toString();
        }}

        function openCheckout() {{
          // No redirect:true / callback_url here on purpose — that combination
          // makes Razorpay do a server-side POST to callback_url instead of
          // calling this handler, and Streamlit's server returns "Method Not
          // Allowed" on that POST because it only serves GET. The handler
          // below does a client-side JS redirect (a GET), which is what
          // handle_payment_return() in this file is built to parse.
          options.handler = redirectAfterPay;
          options.modal = {{ ondismiss: function () {{}} }};
          const rzp = new topWin.Razorpay(options);
          rzp.open();
        }}

        function loadAndOpen() {{
          if (topWin.Razorpay) {{
            openCheckout();
            return;
          }}
          let script = topDoc.querySelector(
            'script[src="https://checkout.razorpay.com/v1/checkout.js"]'
          );
          if (!script) {{
            script = topDoc.createElement("script");
            script.src = "https://checkout.razorpay.com/v1/checkout.js";
            script.onload = openCheckout;
            topDoc.head.appendChild(script);
          }} else {{
            script.onload = openCheckout;
            if (topWin.Razorpay) openCheckout();
          }}
        }}

        try {{
          loadAndOpen();
        }} catch (err) {{
          document.body.innerHTML =
            "<p style='color:#F0556B;font-family:sans-serif;font-size:13px;'>" +
            "Could not open checkout here. Refresh and try again.</p>";
        }}
      }})();
    </script>
    """
    components.html(html, height=0, scrolling=False)