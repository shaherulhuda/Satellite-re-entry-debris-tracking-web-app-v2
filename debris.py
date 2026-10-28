"""Debris catalogue: parse a 3-line TLE file (name, line 1, line 2) into OMM-style records.

The records have the same keys as CelesTrak OMM JSON, so every existing function (altitudes, decay
estimate, SGP4 via skyfield / sgp4, conjunction screening) works on them unchanged.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

DEBRIS_DIR = Path(__file__).with_name("data") / "debris"
DEBRIS_PATH = DEBRIS_DIR / "fengyun-1c.tle"  # the originally supplied file

# group slug (file name) -> (label shown in the app, CelesTrak GROUP query used by tools/refresh_data.py)
GROUPS = {
    "fengyun-1c": ("Fengyun-1C", "1999-025"),
    "iridium-33": ("Iridium-33", "iridium-33-debris"),
    "cosmos-2251": ("Cosmos-2251", "cosmos-2251-debris"),
}
REQUIRED_KEYS = {"OBJECT_NAME", "NORAD_CAT_ID", "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION"}


def _implied_decimal(field: str) -> float:
    """TLE exponent fields like ' 40801-3' -> 0.40801e-3; '00000+0' -> 0."""
    s = field.strip()
    if not s:
        return 0.0
    sign = -1.0 if s[0] == "-" else 1.0
    s = s.lstrip("+-")
    mant, exp = s[:-2], s[-2:]
    return sign * float(f"0.{mant}") * 10.0 ** int(exp)


def _epoch_iso(yy: int, day: float) -> str:
    year = 2000 + yy if yy < 57 else 1900 + yy
    t = dt.datetime(year, 1, 1) + dt.timedelta(days=day - 1.0)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")


def parse_tle_record(name: str, l1: str, l2: str) -> dict:
    """One TLE -> OMM-style dict. Raises ValueError on a malformed pair."""
    if not (l1.startswith("1 ") and l2.startswith("2 ")) or len(l1) < 68 or len(l2) < 68:
        raise ValueError("malformed TLE lines")
    if l1[2:7] != l2[2:7]:
        raise ValueError("catalogue numbers of line 1 and 2 differ")
    norad = int(l1[2:7])
    return {
        "OBJECT_NAME": name.strip(),
        "OBJECT_ID": l1[9:17].strip(),
        "EPOCH": _epoch_iso(int(l1[18:20]), float(l1[20:32])),
        "MEAN_MOTION": float(l2[52:63]),
        "ECCENTRICITY": float("0." + l2[26:33].strip()),
        "INCLINATION": float(l2[8:16]),
        "RA_OF_ASC_NODE": float(l2[17:25]),
        "ARG_OF_PERICENTER": float(l2[34:42]),
        "MEAN_ANOMALY": float(l2[43:51]),
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": l1[7],
        "NORAD_CAT_ID": norad,
        "ELEMENT_SET_NO": int(l1[64:68]),
        "REV_AT_EPOCH": int(l2[63:68]),
        "BSTAR": _implied_decimal(l1[53:61]),
        "MEAN_MOTION_DOT": float(l1[33:43]),
        "MEAN_MOTION_DDOT": _implied_decimal(l1[44:52]),
    }


def parse_tle_text(text: str) -> tuple[list[dict], int]:
    """Parse a whole 3LE file. Returns (records, number_of_skipped_entries)."""
    lines = [ln.rstrip("\r\n") for ln in text.splitlines() if ln.strip()]
    records, skipped = [], 0
    for i in range(0, len(lines) - 2, 3):
        try:
            records.append(parse_tle_record(lines[i], lines[i + 1], lines[i + 2]))
        except (ValueError, IndexError):
            skipped += 1
    return records, skipped


def parse_omm_json(text: str) -> list[dict]:
    """CelesTrak OMM JSON (list of dicts) -> records, dropping entries without the needed fields."""
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of OMM records")
    return [r for r in data if isinstance(r, dict) and REQUIRED_KEYS <= set(r)]


def load_debris_records(directory: Path | str = DEBRIS_DIR) -> list[dict]:
    """All debris groups found in `directory` (.tle / .txt / .json), one record per NORAD id.

    The file name (without extension) is the group slug; each record gets a `GROUP` label so the
    app can filter by breakup event. Missing directory -> [].
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    merged: list[dict] = []
    for path in sorted(d.iterdir()):
        if path.suffix.lower() not in (".tle", ".txt", ".json"):
            continue
        text = path.read_text(encoding="utf-8")
        recs = parse_omm_json(text) if path.suffix.lower() == ".json" else parse_tle_text(text)[0]
        label = GROUPS.get(path.stem, (path.stem,))[0]
        for r in recs:
            r["GROUP"] = label
        merged = merge_records(merged, recs)
    return merged


def merge_records(active: list[dict], debris: list[dict]) -> list[dict]:
    """Active + debris, one record per NORAD id (the newer element set wins)."""
    by_id = {r["NORAD_CAT_ID"]: r for r in active}
    for r in debris:
        cur = by_id.get(r["NORAD_CAT_ID"])
        if cur is None or r["EPOCH"] > cur["EPOCH"]:
            by_id[r["NORAD_CAT_ID"]] = r
    return list(by_id.values())
