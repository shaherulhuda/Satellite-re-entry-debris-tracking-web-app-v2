"""Precompute each high-risk object's closest approach (to any satellite or debris) for the risk score.

Usage:  python tools/compute_conjunctions.py [--top 300] [--hours 24] [--threshold 50]

Screening every object takes too long for the web app, so this runs offline on the objects with the highest
base risk (perigee, decay, inclination, type) and stores the result in data/closest_approaches.json. Re-run it
after tools/refresh_data.py. The window starts at the newest element epoch in the data (the snapshot time).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import conjunction  # noqa: E402
import dataset  # noqa: E402
import decay  # noqa: E402

_STATE: dict = {}


def _one(args):
    nid, band, start_jd, hours, threshold = args
    cat, recs = _STATE["cat"], _STATE["recs"]
    row = {"norad_id": int(nid), "name": recs[nid]["OBJECT_NAME"], "miss_km": None, "rel_speed_km_s": None,
           "hours_to_ca": None, "with_name": None, "with_id": None}
    try:
        ev, _ = conjunction.screen(recs[nid], cat, recs, start_jd, hours=hours, threshold_km=threshold,
                                   primary_band=band, alt_margin_km=20.0)
    except ValueError:  # cannot be propagated over the window (e.g. it decays inside it)
        row["note"] = "not propagated over the window"
        return row
    if len(ev):
        e = ev.iloc[0]
        row.update(miss_km=round(float(e["miss_km"]), 3), rel_speed_km_s=round(float(e["rel_speed_km_s"]), 3),
                   hours_to_ca=round((float(e["tca_jd"]) - start_jd) * 24.0, 2), with_name=str(e["name"]),
                   with_id=int(e["norad_id"]))
    return row


def compute(top: int = 300, hours: float = 24.0, threshold_km: float = 50.0, workers: int | None = None, log=print) -> dict:
    from sgp4.api import jday

    cat, recs = dataset.load_catalogue()
    comp = decay.risk_components(cat)
    base_only = {k: (v if k in decay.BASE_KEYS else 0.0) for k, v in decay.DEFAULT_WEIGHTS.items()}
    cat = cat.assign(base=decay.risk_score(comp, base_only))
    shortlist = cat.sort_values("base", ascending=False).head(top)
    ts = cat["EPOCH_DT"].max()
    jd, fr = jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second + ts.microsecond / 1e6)
    jobs = [(int(r.NORAD_CAT_ID), (float(r.perigee_km), float(r.apogee_km)), jd + fr, hours, threshold_km)
            for r in shortlist.itertuples()]
    _STATE.update(cat=cat, recs=recs)
    workers = workers or max(1, (mp.cpu_count() or 1))
    rows = []
    with mp.get_context("fork").Pool(workers) as pool:
        for i, row in enumerate(pool.imap(_one, jobs, chunksize=2), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                log(f"{i}/{len(jobs)} objects screened")
    return {"start_utc": ts.strftime("%Y-%m-%dT%H:%M:%S"), "window_hours": hours, "threshold_km": threshold_km,
            "computed_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "top": top, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    out = compute(args.top, args.hours, args.threshold, args.workers)
    dataset.CA_PATH.write_text(json.dumps(out, separators=(",", ":")))
    hit = sum(1 for r in out["rows"] if r["miss_km"] is not None)
    print(f"wrote {dataset.CA_PATH} ({len(out['rows'])} objects, {hit} with an approach within {args.threshold} km)")
