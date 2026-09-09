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

# Compass bearing at the centre of each zone, used to measure how far a room sits
# from where the sources want it. "C" has no bearing and is handled separately.
_DIR_BEARING = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315,
}

# Rows in the CSV that describe the same room type, merged under one name.
#
# The dataset splits bedrooms across "Bedroom" (S1) and "Master bedroom /
# bedroom" (S2), and only the second carries an avoid list. Matching a plain
# "BEDROOM" label to the first row alone meant consulting 1 source out of 3 and
# throwing away the only rows that could ever mark a bedroom non-compliant.
_RULE_FAMILIES = {
    "Bedroom": "Bedroom / master bedroom",
    "Master bedroom / bedroom": "Bedroom / master bedroom",
}

# Ordered most-specific-first: "master bedroom" must hit before generic "bedroom".
_ROOM_KEYWORDS = [
    (("master",), "Bedroom / master bedroom"),
    (("bedroom", "bed room"), "Bedroom / master bedroom"),
    (("toilet", "wc"), "WC / toilet"),
    (("bathroom", "washroom", "bath"), "Bathroom"),
    (("kitchen",), "Kitchen"),
    (("dining",), "Dining room"),
    # Indian plans almost always print the main living room as "HALL".
    (("drawing", "living", "hall", "lounge"), "Drawing / living room"),
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
        "preferred": Counter(), "acceptable": Counter(), "avoid": Counter(),
        "n_sources": 0, "sources": [],
    })
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["room_or_function"].strip()
            room = _RULE_FAMILIES.get(raw, raw)
            r = rules[room]
            r["n_sources"] += 1
            r["sources"].append(row["source_title"].strip())
            for d in _extract_directions(row["preferred_directions"]):
                r["preferred"][d] += 1
            for d in _extract_directions(row["acceptable_directions"]):
                r["acceptable"][d] += 1
            for d in _extract_directions(row["avoid_directions"]):
                r["avoid"][d] += 1
    return dict(rules)


def _angular_distance(a: str, b: str):
    """
    Degrees between two compass zones, or None if either is the centre.

    The centre is not on the compass, so it cannot be "60 degrees from SE" — any
    comparison involving it has to be handled as its own case.
    """
    if a not in _DIR_BEARING or b not in _DIR_BEARING:
        return None
    diff = abs(_DIR_BEARING[a] - _DIR_BEARING[b]) % 360
    return min(diff, 360 - diff)


def _nearest_preferred(direction: str, rule: dict):
    """Closest preferred zone to `direction`, as (zone, degrees away)."""
    best = (None, None)
    for zone in rule["preferred"]:
        dist = _angular_distance(direction, zone)
        if dist is not None and (best[1] is None or dist < best[1]):
            best = (zone, dist)
    return best


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
            "classification": "Unknown",
            "agreement_level": 0.0,
            "n_sources_confirming": 0,
            "n_sources_total": 0,
            "preferred_directions": "",
            "sources": [],
            "notes": "No rule for this room type in the dataset — not judged.",
        }

    total = rule["n_sources"]
    avoid_votes = rule["avoid"].get(direction, 0)
    pref_votes = rule["preferred"].get(direction, 0)
    acc_votes = rule["acceptable"].get(direction, 0)

    prefer_list = ", ".join(sorted(rule["preferred"], key=lambda d: _DIR_BEARING.get(d, 999)))

    if avoid_votes:
        classification, votes = "Non-Compliant", avoid_votes
        note = f"{avoid_votes} of {total} sources name {direction} a direction to avoid here."
    elif pref_votes:
        classification, votes = "Compliant", pref_votes
        note = f"{pref_votes} of {total} sources place this room in {direction}."
    elif acc_votes:
        classification, votes = "Moderate", acc_votes
        note = f"{acc_votes} of {total} sources accept {direction} as an alternative."
    elif direction == "C":
        # Not on the compass, so distance cannot decide it. The dataset's own
        # Brahmasthan row asks for the centre to be kept open, which makes any
        # other room here a compromise rather than an outright violation.
        classification, votes = "Moderate", 0
        note = ("Sits in the Brahmasthan (centre), which classical sources ask to be "
                f"left open. Preferred zone for this room is {prefer_list}.")
    else:
        # The heart of it: this dataset states where a room *should* go and only
        # rarely lists what to avoid. Treating "not in the preferred list" as
        # "unknown" made every such room Moderate with 0% agreement — 7 of 8 on
        # a real plan — which is exactly the all-Moderate wash. In a prescriptive
        # system the preferred zone is the claim, so distance from it is the
        # verdict; a quarter turn off is a compromise, opposite is a violation.
        zone, dist = _nearest_preferred(direction, rule)
        if zone is None:
            classification, votes = "Moderate", 0
            note = "No source states a direction for this room type."
        elif dist <= 90:
            classification, votes = "Moderate", 0
            note = (f"{direction} is {dist}° off the preferred {zone}. "
                    f"Partly aligned; sources prefer {prefer_list}.")
        else:
            classification, votes = "Non-Compliant", 0
            note = (f"{direction} is {dist}° from the preferred {zone}. "
                    f"Sources place this room in {prefer_list}.")

    return {
        "room_label": room_label,
        "direction": direction,
        "bbox": bbox,
        "canonical_room": canonical,
        "classification": classification,
        "agreement_level": round(votes / total, 2) if total else 0.0,
        "n_sources_confirming": votes,
        "n_sources_total": total,
        "preferred_directions": prefer_list,
        "sources": rule["sources"],
        "notes": note,
    }


def score_layout(rooms: list) -> dict:
    room_results = [_classify_room(r) for r in rooms]
    weights = {"Compliant": 1.0, "Moderate": 0.5, "Non-Compliant": 0.0}

    # Rooms the dataset says nothing about are left out of the score rather than
    # counted as half marks. Scoring a car park against Vastu rules that do not
    # mention car parks moves the number without meaning anything.
    judged = [r for r in room_results if r["classification"] in weights]
    overall_score = (
        round(100 * sum(weights[r["classification"]] for r in judged) / len(judged), 1)
        if judged else 0.0
    )
    return {
        "room_results": room_results,
        "overall_score": overall_score,
        "n_judged": len(judged),
        "n_rooms": len(room_results),
    }


def generate_remodel_tiers(room_results: list) -> dict:
    # A room the dataset has no rule for cannot be "fixed" — it was only marked
    # Moderate because nothing was known about it. Listing it as a remodel step
    # with a blank target direction is noise, so it is left out.
    violations = [
        r for r in room_results
        if r["classification"] != "Compliant"
        and r["canonical_room"]
        and _best_direction(r["canonical_room"])
    ]
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
