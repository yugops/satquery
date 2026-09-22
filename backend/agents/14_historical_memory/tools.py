"""
SatQuery AI — Agent 14: HistoricalMemoryAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Global_Historical_Memory
"""

from __future__ import annotations

import re
from typing import Any


HISTORICAL_DISASTER_DB: list[dict[str, Any]] = [
    {
        "district": "Wayanad",
        "state": "Kerala",
        "event_type": "Landslide & Debris Flow",
        "date": "July 2024",
        "impact_summary": "Chooralmala & Meppadi riverbed debris surge. 4.2 km runout channel triggered by 572mm 48-hr rainfall on steep gneiss bedrock.",
        "baseline_area_sqkm": 8.4,
    },
    {
        "district": "Barpeta",
        "state": "Assam",
        "event_type": "Monsoon Riverine Inundation",
        "date": "June 2022",
        "impact_summary": "Brahmaputra high-discharge breach submerging 38 sq. km across 4 revenue circles. Severe paddy crop loss.",
        "baseline_area_sqkm": 38.0,
    },
    {
        "district": "Chamoli",
        "state": "Uttarakhand",
        "event_type": "Rock-Ice Avalanche & GLOF",
        "date": "February 2021",
        "impact_summary": "Ronti Peak hanging glacier detachment triggering high-energy debris surge through Rishiganga gorge.",
        "baseline_area_sqkm": 3.1,
    },
    {
        "district": "Puri",
        "state": "Odisha",
        "event_type": "Extremely Severe Cyclonic Storm (Fani)",
        "date": "May 2019",
        "impact_summary": "Category 5 storm surge causing 2.8 km inland seawater penetration across coastal agricultural belts.",
        "baseline_area_sqkm": 45.0,
    },
    {
        "district": "Beed",
        "state": "Maharashtra",
        "event_type": "Agricultural Drought",
        "date": "October 2023",
        "impact_summary": "Severe monsoon deficit causing -0.32 NDVI delta and 48% soybean/cotton yield reduction across Marathwada.",
        "baseline_area_sqkm": 120.0,
    }
]

_STOPWORDS = {"and", "the", "of", "in", "a", "an", "risk", "damage", "assessment", "event", "hazard"}


def _word_in_text(word: str, text: str) -> bool:
    """
    Word-boundary match instead of raw substring matching. The previous
    version used plain `in` checks, so a short district name like 'Beed'
    could in principle match inside an unrelated longer word/phrase, and
    single common query words could accidentally collide with an event_type
    string. Word-boundary matching removes that class of false-precedent match.
    """
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def tool_mempalace_recall_precedent(district: str, disaster_type: str = "") -> list[dict[str, Any]]:
    """Retrieves verified historical precedent records from MemPalace archive."""
    d_clean = district.lower().strip()
    matches = [
        item for item in HISTORICAL_DISASTER_DB
        if _word_in_text(item["district"].lower(), d_clean) or _word_in_text(item["state"].lower(), d_clean)
    ]
    if not matches and disaster_type:
        dt_words = [w for w in disaster_type.lower().split() if w not in _STOPWORDS and len(w) > 2]
        matches = [
            item for item in HISTORICAL_DISASTER_DB
            if any(_word_in_text(w, item["event_type"].lower()) for w in dt_words)
        ]
    return matches