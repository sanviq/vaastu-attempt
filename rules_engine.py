"""
Owner: Sanvi

Contract (do not change shapes without messaging Harshika immediately):
    score_layout(rooms) -> {"room_results": [...], "overall_score": float}
        rooms: list of {"room_label": str, "direction": str, "bbox": ...}
               (bbox is opaque here — only room_label and direction are read)

    generate_remodel_tiers(room_results) -> {"25": [...], "50": [...], "75": [...], "100": [...]}
"""

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "vastu_indian_base_rules_polished.csv"

_DIR_RE = re.compile(r"\b(NE|NW|SE|SW|N|E|S|W|C)\b")

# Ordered most-specific-first: "master bedroom" must hit before generic "bedroom".
_ROOM_KEYWORDS = [
    (("master",), "Master bedroom / bedroom"),
    (("bedroom", "bed room"), "Bedroom"),
    (("toilet", "wc"), "WC / toilet"),
    (("bathroom", "washroom", "bath"), "Bathroom"),
    (("kitchen",), "Kitchen"),
    (("dining",), "Dining room"),
    (("drawing", "living"), "Drawing / living room"),
    (("puja", "pooja", "meditation", "prayer"), "Puja / meditation room"),
    (("study",), "Study room"),
    (("basement",), "Basement"),
    (("entrance", "main door", "foyer"), "Main entrance"),
    (("brahmasthan", "central", "centre", "center"), "Brahmasthan / central area"),
    (("open space", "setback"), "Open space / setbacks"),
]


def _extract_directions(cell: str) -> set:
    if not cell:
        return set()
    return set(_DIR_RE.findall(cell))


def _load_rules(path: Path) -> dict:
    rules = defaultdict(lambda: {
        "preferred": Counter(), "acceptable": Counter(), "avoid": Counter(), "n_sources": 0,
    })
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            room = row["room_or_function"].strip()
            r = rules[room]
            r["n_sources"] += 1
            for d in _extract_directions(row["preferred_directions"]):
                r["preferred"][d] += 1
            for d in _extract_directions(row["acceptable_directions"]):
                r["acceptable"][d] += 1
            for d in _extract_directions(row["avoid_directions"]):
                r["avoid"][d] += 1
    return dict(rules)


RULES = _load_rules(DATA_PATH)


def _canonical_room(room_label: str):
    label = re.sub(r"[^a-z\s]", " ", room_label.lower())
    for keywords, canonical in _ROOM_KEYWORDS:
        if any(k in label for k in keywords):
            return canonical
    return None


def _best_direction(canonical_room):
    rule = RULES.get(canonical_room)
    if not rule or not rule["preferred"]:
        return None
    return rule["preferred"].most_common(1)[0][0]


def _classify_room(room: dict) -> dict:
    room_label = room["room_label"]
    direction = room["direction"]
    bbox = room.get("bbox")
    canonical = _canonical_room(room_label)
    rule = RULES.get(canonical) if canonical else None

    if rule is None:
        return {
            "room_label": room_label,
            "direction": direction,
            "bbox": bbox,
            "canonical_room": canonical,
            "classification": "Moderate",
            "agreement_level": 0.0,
            "n_sources_confirming": 0,
            "n_sources_total": 0,
            "notes": "no matching rule in dataset for this room type",
        }

    total = rule["n_sources"]
    avoid_votes = rule["avoid"].get(direction, 0)
    pref_votes = rule["preferred"].get(direction, 0)
    acc_votes = rule["acceptable"].get(direction, 0)

    if avoid_votes:
        classification, votes = "Non-Compliant", avoid_votes
    elif pref_votes:
        classification, votes = "Compliant", pref_votes
    elif acc_votes:
        classification, votes = "Moderate", acc_votes
    else:
        classification, votes = "Moderate", 0

    return {
        "room_label": room_label,
        "direction": direction,
        "bbox": bbox,
        "canonical_room": canonical,
        "classification": classification,
        "agreement_level": round(votes / total, 2) if total else 0.0,
        "n_sources_confirming": votes,
        "n_sources_total": total,
        "notes": None if votes else "direction not addressed by any source for this room type",
    }


def score_layout(rooms: list) -> dict:
    room_results = [_classify_room(r) for r in rooms]
    weights = {"Compliant": 1.0, "Moderate": 0.5, "Non-Compliant": 0.0}
    overall_score = (
        round(100 * sum(weights[r["classification"]] for r in room_results) / len(room_results), 1)
        if room_results else 0.0
    )
    return {"room_results": room_results, "overall_score": overall_score}


def generate_remodel_tiers(room_results: list) -> dict:
    violations = [r for r in room_results if r["classification"] != "Compliant"]
    # Worst first: Non-Compliant before Moderate, then by how corroborated the broken rule is.
    violations.sort(key=lambda r: (r["classification"] == "Moderate", -r["agreement_level"]))

    total = len(violations)
    tiers = {}
    for pct in (25, 50, 75, 100):
        n = math.ceil(total * pct / 100) if total else 0
        tiers[str(pct)] = [
            {
                "room_label": v["room_label"],
                "current_direction": v["direction"],
                "bbox": v["bbox"],
                "suggested_direction": _best_direction(v["canonical_room"]) if v["canonical_room"] else None,
                "classification": v["classification"],
                "agreement_level": v["agreement_level"],
            }
            for v in violations[:n]
        ]
    return tiers
