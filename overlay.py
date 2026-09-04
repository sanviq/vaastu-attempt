"""
Owner: Harshika

Contract (do not change shapes without messaging Sanvi immediately):
    draw_overlay(image_path, room_results) -> np.ndarray (RGB)
        image_path   : str — original floor plan
        room_results : list of dicts from rules_engine.score_layout()
                       Each dict has at minimum:
                           room_label, direction, classification, bbox

    draw_remodel_tier(image_path, tier_suggestions) -> np.ndarray (RGB)
        image_path      : str — original floor plan
        tier_suggestions: list of dicts from rules_engine.generate_remodel_tiers()
                          Each dict has at minimum:
                              room_label, current_direction,
                              suggested_direction, classification

    Both functions return an RGB numpy array (Matplotlib native).
    Streamlit's st.image() accepts RGB arrays directly.

Tool used: Matplotlib (PSF-based / BSD-compatible, per the brief's approved stack).
    - Blueprint image is loaded with Matplotlib's imread (supports PNG/JPG).
    - FancyBboxPatch draws the coloured room overlays.
    - FancyArrowPatch draws remodel suggestion arrows.
    - All drawing happens on a Matplotlib Figure, then rendered to an RGB ndarray
      via fig.canvas.draw() + np.frombuffer — no file I/O needed.

Colour coding (matches brief):
    Compliant     → green   #00b800
    Moderate      → orange  #ffa500   (yellow on white backgrounds is unreadable)
    Non-Compliant → red     #dc0000
"""

from __future__ import annotations

import io
import math

import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Streamlit threads
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (hex, Matplotlib-compatible)
# ---------------------------------------------------------------------------
_COLOURS = {
    "Compliant":     "#00b800",
    "Moderate":      "#ffa500",
    "Non-Compliant": "#dc0000",
}
_DEFAULT_COLOUR = "#808080"
_OVERLAY_ALPHA = 0.30
_BORDER_ALPHA = 0.90
_BORDER_LW = 2.5

# Direction unit vectors (dx, dy in image pixel space — +y is DOWN)
_DIR_VEC: dict[str, tuple[float, float]] = {
    "N":  ( 0.0, -1.0),
    "NE": ( 0.7, -0.7),
    "E":  ( 1.0,  0.0),
    "SE": ( 0.7,  0.7),
    "S":  ( 0.0,  1.0),
    "SW": (-0.7,  0.7),
    "W":  (-1.0,  0.0),
    "NW": (-0.7, -0.7),
    "C":  ( 0.0,  0.0),
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


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Render a Matplotlib figure to an RGB uint8 numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)[:, :, ::-1]   # BGR→RGB


def _make_figure(rgb: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """Create a frameless Matplotlib figure sized to the image."""
    h, w = rgb.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(rgb)
    ax.axis("off")
    return fig, ax


def _add_legend(ax: plt.Axes) -> None:
    """Add a Matplotlib legend patch in the upper-right of the axes."""
    handles = [
        mpatches.Patch(color=colour, label=label, alpha=0.85)
        for label, colour in _COLOURS.items()
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.75,
        edgecolor="#333333",
        facecolor="#f5f5f5",
    )
    legend.get_frame().set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Public API — draw_overlay
# ---------------------------------------------------------------------------

def draw_overlay(image_path: str, room_results: list[dict]) -> np.ndarray:
    """
    Draw a green/yellow/red compliance overlay on the floor plan using Matplotlib.

    Returns an RGB numpy array suitable for st.image().
    """
    rgb = _load_rgb(image_path)
    img_h, img_w = rgb.shape[:2]
    fig, ax = _make_figure(rgb)

    for room in room_results:
        bbox = room.get("bbox")
        if not bbox:
            continue

        x, y, w, h = bbox
        classification = room.get("classification", "")
        colour = _COLOURS.get(classification, _DEFAULT_COLOUR)
        label = room.get("room_label", "?")
        direction = room.get("direction", "")

        # Semi-transparent filled patch
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=_BORDER_LW,
            edgecolor=colour,
            facecolor=colour,
            alpha=_OVERLAY_ALPHA,
        )
        ax.add_patch(patch)

        # Solid border (drawn as a separate patch at full alpha)
        border = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=_BORDER_LW,
            edgecolor=colour,
            facecolor="none",
            alpha=_BORDER_ALPHA,
        )
        ax.add_patch(border)

        # Room label text
        font_size = max(5, min(10, w / 14))
        ax.text(
            x + 4, y + font_size + 4,
            label[:22],
            fontsize=font_size,
            color="white",
            fontweight="bold",
            va="top",
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=colour,
                alpha=0.70,
                linewidth=0,
            ),
        )

        # Direction · classification sub-label
        ax.text(
            x + 4, y + font_size * 2.4 + 4,
            f"{direction} · {classification}",
            fontsize=max(4, font_size * 0.80),
            color=colour,
            va="top",
            clip_on=True,
        )

    _add_legend(ax)
    return _fig_to_rgb(fig)


# ---------------------------------------------------------------------------
# Public API — draw_remodel_tier
# ---------------------------------------------------------------------------

def draw_remodel_tier(image_path: str, tier_suggestions: list[dict]) -> np.ndarray:
    """
    Annotate the floor plan with remodel suggestions for a given tier using Matplotlib.

    For each violation:
        - Highlighted coloured border box
        - Arrow pointing toward the suggested Vastu direction
        - Text: room name + "Move to X (now Y)"

    Returns an RGB numpy array.
    """
    rgb = _load_rgb(image_path)
    img_h, img_w = rgb.shape[:2]
    fig, ax = _make_figure(rgb)

    for suggestion in tier_suggestions:
        bbox = suggestion.get("bbox")
        classification = suggestion.get("classification", "Non-Compliant")
        colour = _COLOURS.get(classification, _COLOURS["Non-Compliant"])

        room_label = suggestion.get("room_label", "?")
        current_dir = suggestion.get("current_direction", "?")
        suggested_dir = suggestion.get("suggested_direction") or "—"

        if bbox:
            x, y, w, h = bbox
        else:
            # No bbox — place a banner strip
            x, y = img_w // 4, 10
            w, h = img_w // 2, max(60, int(img_h * 0.07))

        cx, cy = x + w / 2, y + h / 2

        # Highlighted border box (thicker than overlay)
        border = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=_BORDER_LW + 1.5,
            edgecolor=colour,
            facecolor=colour,
            alpha=0.18,
        )
        ax.add_patch(border)

        # Arrow toward suggested direction
        if suggested_dir in _DIR_VEC:
            dvx, dvy = _DIR_VEC[suggested_dir]
            arrow_len = min(w, h) * 0.30
            ax.annotate(
                "",
                xy=(cx + dvx * arrow_len, cy + dvy * arrow_len),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colour,
                    lw=2.0,
                    mutation_scale=14,
                ),
            )

        # Room label
        font_size = max(5, min(10, w / 14))
        ax.text(
            x + 4, y + font_size + 4,
            room_label[:22],
            fontsize=font_size,
            color="white",
            fontweight="bold",
            va="top",
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=colour,
                alpha=0.80,
                linewidth=0,
            ),
        )

        # Suggestion text
        msg = f"Move to {suggested_dir}  (now {current_dir})"
        ax.text(
            x + 4, y + font_size * 2.4 + 4,
            msg,
            fontsize=max(4, font_size * 0.80),
            color=colour,
            va="top",
            clip_on=True,
        )

    # Header banner using Matplotlib axes title
    n = len(tier_suggestions)
    ax.set_title(
        f"Remodel suggestions — {n} room(s) addressed",
        fontsize=9,
        color="#e8e8e8",
        loc="left",
        pad=4,
        backgroundcolor="#282828",
    )

    return _fig_to_rgb(fig)
