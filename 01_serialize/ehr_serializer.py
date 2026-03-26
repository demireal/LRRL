"""
Adaptad from https://github.com/stefanhgm/ehrshot-benchmark

EHR Serializer -- converts FEMR patient events into structured Markdown text.

Serialization structure (naivetext, num_aggregated=0):
  # Electronic Healthcare Record
  Current time: ...
  ## Patient Demographics
  ## General Medical Events 
  ## Detailed Past Medical Visits (most recent first)

"""

from datetime import datetime
from typing import List, Optional, Union, Dict, Callable, Tuple
from abc import ABC, abstractmethod
from femr import Event
from collections import defaultdict
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSTANT_LABEL_TIME = datetime(2024, 1, 1)

EHR_HEADING = "\n\n# Electronic Healthcare Record\n\n"
STATIC_EVENTS_HEADING = "## General Medical Events\n\n"

AGGREGATED_SUB_EVENTS = {
    "Body Metrics": {
        "heading": "## Recent Body Metrics\n",
        "events": [
            "Body weight",
            "Body height",
            "Body mass index / BMI",
            "Body surface area",
        ],
    },
    "Vital Signs": {
        "heading": "## Recent Vital Signs\n",
        "events": [
            "Heart rate",
            "Respiratory rate",
            "Systolic blood pressure",
            "Diastolic blood pressure",
            "Body temperature",
            "Oxygen saturation",
        ],
    },
    "Lab Results": {
        "heading": "## Recent Lab Results\n",
        "events": [
            "Hemoglobin",
            "Hematocrit",
            "Erythrocytes",
            "Leukocytes",
            "Platelets",
            "Sodium",
            "Potassium",
            "Chloride",
            "Carbon dioxide, total",
            "Calcium",
            "Glucose",
            "Urea nitrogen",
            "Creatinine",
            "Anion gap",
        ],
    },
}

# ---------------------------------------------------------------------------
# Aggregated-event metadata (units, ranges, format helpers)
# ---------------------------------------------------------------------------

def _fmt_int(x):
    return f"{int(x)}"

def _fmt_1d(x):
    return f"{x:.1f}"

def _fmt_2d(x):
    return f"{x:.2f}"


AGGREGATED_EVENTS = {
    "Heart rate": {
        "codes": ["LOINC/8867-4", "SNOMED/364075005", "SNOMED/78564009"],
        "min_max": [5, 300],
        "normal_range": [60, 100],
        "unit": "bpm",
        "format": _fmt_int,
    },
    "Systolic blood pressure": {
        "codes": ["LOINC/8480-6", "SNOMED/271649006"],
        "min_max": [20, 300],
        "normal_range": [90, 140],
        "unit": "mmHg",
        "format": _fmt_int,
    },
    "Diastolic blood pressure": {
        "codes": ["LOINC/8462-4", "SNOMED/271650006"],
        "min_max": [20, 300],
        "normal_range": [60, 90],
        "unit": "mmHg",
        "format": _fmt_int,
    },
    "Body temperature": {
        "codes": ["LOINC/8310-5"],
        "min_max": [80, 120],
        "normal_range": [95, 100.4],
        "unit": "\u00b0F",
        "format": _fmt_1d,
    },
    "Respiratory rate": {
        "codes": ["LOINC/9279-1"],
        "min_max": [1, 100],
        "normal_range": [12, 18],
        "unit": "breaths/min",
        "format": _fmt_int,
    },
    "Oxygen saturation": {
        "codes": ["LOINC/LP21258-6"],
        "min_max": [1, 100],
        "normal_range": [95, 100],
        "unit": "%",
        "format": _fmt_int,
    },
    "Body weight": {
        "codes": ["LOINC/29463-7"],
        "min_max": [350, 10000],
        "unit": "oz",
        "format": _fmt_1d,
    },
    "Body height": {
        "codes": ["LOINC/8302-2"],
        "min_max": [5, 100],
        "unit": "inch",
        "format": _fmt_1d,
    },
    "Body mass index / BMI": {
        "codes": ["LOINC/39156-5"],
        "min_max": [10, 100],
        "normal_range": [18.5, 24.9],
        "unit": "kg/m2",
        "format": _fmt_1d,
    },
    "Body surface area": {
        "codes": ["LOINC/8277-6", "SNOMED/301898006"],
        "min_max": [0.1, 10],
        "unit": "m2",
        "format": _fmt_2d,
    },
    "Hemoglobin": {
        "codes": ["LOINC/718-7", "SNOMED/271026005", "SNOMED/441689006"],
        "min_max": [1, 20],
        "normal_range": [12, 17],
        "unit": "g/dL",
        "format": _fmt_1d,
    },
    "Hematocrit": {
        "codes": ["LOINC/4544-3", "LOINC/20570-8", "LOINC/48703-3", "SNOMED/28317006"],
        "min_max": [10, 100],
        "normal_range": [36, 51],
        "unit": "%",
        "format": _fmt_int,
    },
    "Erythrocytes": {
        "codes": ["LOINC/789-8", "LOINC/26453-1"],
        "min_max": [1, 10],
        "normal_range": [4.2, 5.9],
        "unit": "10^6/uL",
        "format": _fmt_2d,
    },
    "Leukocytes": {
        "codes": ["LOINC/20584-9", "LOINC/6690-2"],
        "min_max": [1, 100],
        "normal_range": [4, 10],
        "unit": "10^3/uL",
        "format": _fmt_1d,
    },
    "Platelets": {
        "codes": ["LOINC/777-3", "SNOMED/61928009"],
        "min_max": [10, 1000],
        "normal_range": [150, 350],
        "unit": "10^3/uL",
        "format": _fmt_int,
    },
    "Sodium": {
        "codes": ["LOINC/2951-2", "LOINC/2947-0", "SNOMED/25197003"],
        "min_max": [100, 200],
        "normal_range": [136, 145],
        "unit": "mmol/L",
        "format": _fmt_int,
    },
    "Potassium": {
        "codes": ["LOINC/2823-3", "SNOMED/312468003", "LOINC/6298-4", "SNOMED/59573005"],
        "min_max": [0.1, 10],
        "normal_range": [3.5, 5.0],
        "unit": "mmol/L",
        "format": _fmt_1d,
    },
    "Chloride": {
        "codes": ["LOINC/2075-0", "SNOMED/104589004", "LOINC/2069-3"],
        "min_max": [50, 200],
        "normal_range": [98, 106],
        "unit": "mmol/L",
        "format": _fmt_int,
    },
    "Carbon dioxide, total": {
        "codes": ["LOINC/2028-9"],
        "min_max": [10, 100],
        "normal_range": [23, 28],
        "unit": "mmol/L",
        "format": _fmt_int,
    },
    "Calcium": {
        "codes": ["LOINC/17861-6", "SNOMED/271240001"],
        "min_max": [1, 20],
        "normal_range": [9, 10.5],
        "unit": "mg/dL",
        "format": _fmt_1d,
    },
    "Glucose": {
        "codes": ["LOINC/2345-7", "SNOMED/166900001", "LOINC/2339-0", "SNOMED/33747003", "LOINC/14749-6"],
        "min_max": [10, 1000],
        "normal_range": [70, 100],
        "unit": "mg/dL",
        "format": _fmt_int,
    },
    "Urea nitrogen": {
        "codes": ["LOINC/3094-0", "SNOMED/105011006"],
        "min_max": [1, 200],
        "normal_range": [8, 20],
        "unit": "mg/dL",
        "format": _fmt_int,
    },
    "Creatinine": {
        "codes": ["LOINC/2160-0", "SNOMED/113075003"],
        "min_max": [0.1, 10],
        "normal_range": [0.7, 1.3],
        "unit": "mg/dL",
        "format": _fmt_1d,
    },
    "Anion gap": {
        "codes": ["LOINC/33037-3", "LOINC/41276-7", "SNOMED/25469001"],
        "min_max": [-20, 50],
        "normal_range": [3, 11],
        "unit": "mmol/L",
        "format": _fmt_int,
    },
}

CODES_TO_AGGREGATED_EVENTS = {
    code: name
    for name, meta in AGGREGATED_EVENTS.items()
    for code in meta["codes"]
}
AGGREGATED_EVENTS_CODES = list(CODES_TO_AGGREGATED_EVENTS.keys())
AGGREGATED_EVENTS_CODES_LOINC = [c for c in AGGREGATED_EVENTS_CODES if "LOINC" in c]

MEDICATION_ONTOLOGIES = ["RxNorm", "RxNorm Extension"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_recent_aggregated(events: List[Event]) -> Dict[str, List[Event]]:
    """Group aggregated events by type, sorted most-recent-first."""
    result: Dict[str, List[Event]] = defaultdict(list)
    for ev in events:
        if ev.value is None or isinstance(ev.value, str):
            continue
        name = CODES_TO_AGGREGATED_EVENTS.get(ev.code)
        if name is None:
            continue
        lo, hi = AGGREGATED_EVENTS[name]["min_max"]
        if lo <= ev.value <= hi:
            result[name].append(ev)
    for name in result:
        result[name].sort(key=lambda e: e.start, reverse=True)
    return result


def _datetime_date_md(dt: datetime) -> str:
    d = dt.strftime("%Y-%m-%d")
    return f"[{d}]({d})"


def _visit_time_str(label_dt: datetime, start: datetime, end: Optional[datetime]) -> str:
    date_str = _datetime_date_md(start)
    days = (label_dt - start).days
    days_str = "1 day before prediction time" if days == 1 else f"{days} days before prediction time"
    dur = ""
    if end is not None:
        if end > label_dt:
            dur = "current visit"
        else:
            d = (end - start).days
            if d == 1:
                dur = "duration: 1 day"
            elif d > 1:
                dur = f"duration: {d} days"
    return f"{date_str} ({days_str}, {dur})" if dur else f"{date_str} ({days_str})"


def _visit_heading(label_dt: datetime, visit: "EHRVisit") -> str:
    shifted_start = CONSTANT_LABEL_TIME - (label_dt - visit.start)
    shifted_end = CONSTANT_LABEL_TIME - (label_dt - visit.end) if visit.end else None
    return f"### {visit.description} on {_visit_time_str(CONSTANT_LABEL_TIME, shifted_start, shifted_end)}\n\n"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class EHRVisit:
    def __init__(self, visit_id: int, start: datetime,
                 end: Optional[datetime] = None, description: str = ""):
        self.visit_id = visit_id
        self.start = start
        self.end = end
        self.description = description
        self.events: List["EHREvent"] = []

    def add_event(self, event: "EHREvent") -> None:
        self.events.append(event)

    def __lt__(self, other: "EHRVisit") -> bool:
        if self.start != other.start:
            return self.start < other.start
        if self.end is None and other.end is None:
            return False
        if self.end is None:
            return False
        if other.end is None:
            return True
        return self.end < other.end


class EHREvent:
    def __init__(self, start: datetime, end: Optional[datetime] = None,
                 description: str = "", value=None, unit: Optional[str] = None,
                 code: Optional[str] = None):
        self.start = start
        self.end = end
        self.description = description
        self.value = value
        self.unit = unit
        self.code = code


# ---------------------------------------------------------------------------
# Main serializer
# ---------------------------------------------------------------------------

class EHRSerializer:
    """Loads FEMR events and serializes them into Markdown."""

    def __init__(self):
        self.visits: List[EHRVisit] = []
        self.static_events: List[EHREvent] = []
        self.aggregated_events: List[Event] = []

    # -- Loading ----------------------------------------------------------

    def load_from_femr_events(
        self,
        events: List[Event],
        resolve_code: Callable,
        is_visit_event: Callable[[Event], bool],
        filter_aggregated_events: bool,
    ) -> None:
        if filter_aggregated_events:
            non_agg = []
            for ev in events:
                if ev.code in AGGREGATED_EVENTS_CODES:
                    self.aggregated_events.append(ev)
                else:
                    non_agg.append(ev)
            events = non_agg

        visit_map: Dict[int, EHRVisit] = {}
        for ev in filter(is_visit_event, events):
            desc = resolve_code(ev.code)
            if desc is not None:
                v = EHRVisit(ev.visit_id, ev.start,
                             getattr(ev, "end", None), desc)
                visit_map[ev.visit_id] = v

        for ev in filter(lambda e: not is_visit_event(e), events):
            desc = resolve_code(ev.code)
            if desc is None:
                continue
            ehr_ev = EHREvent(
                start=ev.start,
                end=getattr(ev, "end", None),
                description=desc,
                value=getattr(ev, "value", None),
                unit=getattr(ev, "unit", None),
                code=getattr(ev, "code", None),
            )
            visit = visit_map.get(ev.visit_id)
            if visit is not None:
                visit.add_event(ehr_ev)
            else:
                self.static_events.append(ehr_ev)

        self.visits = sorted(visit_map.values())

    # -- Serialization ----------------------------------------------------

    def serialize(self, num_aggregated: int, label_time: datetime) -> str:
        """Produce the full Markdown EHR text.

        Args:
            num_aggregated: number of recent values per aggregated metric.
                            0 = naivetext (no vitals/labs sections).
            label_time: the prediction time for this patient-label pair.
        """
        parts: List[str] = [EHR_HEADING, _time_text()]

        # 1) Demographics
        parts.append(self._demographics_section())

        # 2) Aggregated events (if num_aggregated > 0)
        if num_aggregated > 0:
            parts.append(self._aggregated_section(num_aggregated))

        # NOTE: "Past Medical Visits" summary list is intentionally omitted.
        # It consumed tokens without adding clinical value beyond what the
        # detailed visits section already provides.

        # 3) General medical events
        parts.append(self._general_events_section())

        # 4) Detailed past medical visits
        parts.append(self._detailed_visits_section(label_time))

        text = "".join(parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    # -- Private section builders -----------------------------------------

    def _demographics_section(self) -> str:
        n = 4
        if (len(self.static_events) >= 3
                and self.static_events[2].code
                and self.static_events[2].code.startswith("Ethnicity")):
            n = 3
        n = min(n, len(self.static_events))
        demo = self.static_events[:n]
        self.static_events = self.static_events[n:]
        return "## Patient Demographics\n\n" + _unique_event_list(demo, values=False) + "\n\n"

    def _aggregated_section(self, num_values: int) -> str:
        recent = _get_recent_aggregated(self.aggregated_events)
        lines: List[str] = []
        for section in ("Body Metrics", "Vital Signs", "Lab Results"):
            lines.append(AGGREGATED_SUB_EVENTS[section]["heading"])
            for etype in AGGREGATED_SUB_EVENTS[section]["events"]:
                if etype in recent:
                    vals = [e for e in recent[etype][:num_values] if e.value is not None]
                    val_strs = [_fmt_aggregated_val(etype, e) for e in vals]
                    lines.append(f"- {etype} ({AGGREGATED_EVENTS[etype]['unit']}): {', '.join(val_strs)}")
                else:
                    lines.append(f"- {etype}: No recent data")
            lines.append("")
        return "\n".join(lines) + "\n"

    def _general_events_section(self) -> str:
        return (STATIC_EVENTS_HEADING
                + _unique_event_list(self.static_events, values=True, keep_last=3)
                + "\n\n")

    def _detailed_visits_section(self, label_time: datetime) -> str:
        blocks: List[str] = []
        for visit in sorted(self.visits, reverse=True):
            heading = _visit_heading(label_time, visit)
            conds = [e for e in visit.events
                     if e.code and e.code.split("/")[0] not in
                     ("RxNorm", "RxNorm Extension", "CPT4", "ICD10PCS", "ICD9Proc", "Visit")]
            meds = [e for e in visit.events
                    if e.code and e.code.split("/")[0] in ("RxNorm", "RxNorm Extension")]
            procs = [e for e in visit.events
                     if e.code and e.code.split("/")[0] in ("CPT4", "ICD10PCS", "ICD9Proc")]
            cats: List[str] = []
            if conds:
                cats.append("#### Conditions\n\n" + _unique_event_list(conds, values=True, keep_last=3))
            if meds:
                cats.append("#### Medications\n\n" + _unique_event_list(meds, values=True, keep_last=3))
            if procs:
                cats.append("#### Procedures\n\n" + _unique_event_list(procs, values=True, keep_last=3))
            blocks.append(heading + "\n\n".join(cats))
        header = "## Detailed Past Medical Visits (most recent first)\n\n"
        return header + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Formatting helpers (module-level, used by the class)
# ---------------------------------------------------------------------------

def _time_text() -> str:
    return f"Current time: {_datetime_date_md(CONSTANT_LABEL_TIME)}\n\n"


def _format_value(value) -> str:
    if isinstance(value, float):
        s = f"{value:.2f}".rstrip("0").rstrip(".")
        return s
    return str(value)


def _unique_event_list(events: List[EHREvent], values: bool = False,
                       keep_last: Optional[int] = None) -> str:
    """Serialize a list of events, deduplicating by description and optionally
    aggregating numeric values."""
    bucket: Dict[str, dict] = {}
    order: List[str] = []
    for ev in events:
        key = ev.description
        if key not in bucket:
            bucket[key] = {"values": [], "unit": ev.unit}
            order.append(key)
        if ev.value is not None:
            bucket[key]["values"].append(ev.value)
        bucket[key]["unit"] = ev.unit  # keep latest unit

    lines: List[str] = []
    for desc in order:
        data = bucket[desc]
        vals = data["values"]
        if keep_last is not None:
            vals = vals[-keep_last:]
        if values and vals:
            val_str = ", ".join(_format_value(v) for v in vals)
            unit_str = f" ({data['unit']})" if data["unit"] else ""
            lines.append(f"- {desc}{unit_str}: {val_str}")
        else:
            lines.append(f"- {desc}")
    return "\n".join(lines)


def _fmt_aggregated_val(etype: str, event: Event) -> str:
    meta = AGGREGATED_EVENTS[etype]
    s = meta["format"](event.value)
    if "normal_range" in meta:
        lo, hi = meta["normal_range"]
        if event.value < lo:
            s += " (low)"
        elif event.value > hi:
            s += " (high)"
        else:
            s += " (normal)"
    return s
