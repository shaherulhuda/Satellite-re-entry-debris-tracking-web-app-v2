"""Close-approach screening of one object against the catalogue (SGP4 on the mean elements).

This is a SCREENING tool, not an operational conjunction assessment: mean elements carry
kilometre-level position error that grows with element age, there is no covariance and no
collision probability. Operators use Space-Track conjunction data messages for real decisions.

Method: (1) keep only objects whose altitude band overlaps the primary's; (2) propagate the
primary and candidates on a coarse time grid and find local minima of separation; (3) refine
each promising minimum on a 0.5 s grid; (4) drop pairs that stay within a few tens of km of
each other for the whole window (docked / flying in formation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

V_REL_MAX_KM_S = 16.0  # head-on LEO closing speed bound, sets the coarse-grid safety margin
FORMATION_KM = 20.0


def to_satrec(record: dict) -> Satrec:
    sat = Satrec()
    omm.initialize(sat, record)
    return sat


def _jd_grid(jd0: float, fr0: float, offsets_s: np.ndarray):
    total = fr0 + offsets_s / 86400.0
    whole = np.floor(total)
    return jd0 + whole, total - whole


def screen(primary: dict, catalogue: pd.DataFrame, records_by_norad: dict, start_jd: float,
           hours: float = 6.0, step_s: float = 10.0, threshold_km: float = 10.0,
           alt_margin_km: float = 30.0, chunk: int | None = None,
           primary_band: tuple[float, float] | None = None) -> tuple[pd.DataFrame, int]:
    """Return (events, n_candidates). Events are sorted by miss distance.

    start_jd is a Julian date (UTC, as used by sgp4). Columns: name, norad_id, tca_jd,
    miss_km, rel_speed_km_s. `catalogue` is the set of candidates to screen against; if the
    primary is not in it, pass its (perigee_km, apogee_km) as `primary_band`.
    """
    pid = int(primary["NORAD_CAT_ID"])
    if primary_band is None:
        prow = catalogue[catalogue["NORAD_CAT_ID"] == pid].iloc[0]
        primary_band = (float(prow["perigee_km"]), float(prow["apogee_km"]))
    cand = catalogue[(catalogue["NORAD_CAT_ID"] != pid)
                     & (catalogue["perigee_km"] <= primary_band[1] + alt_margin_km)
                     & (catalogue["apogee_km"] >= primary_band[0] - alt_margin_km)]
    n_cand = len(cand)
    if n_cand == 0:
        return pd.DataFrame(columns=["name", "norad_id", "tca_jd", "miss_km", "rel_speed_km_s"]), 0

    sat_p = to_satrec(primary)
    jd0 = float(np.floor(start_jd))
    fr0 = float(start_jd - jd0)
    offsets = np.arange(0.0, hours * 3600.0 + step_s, step_s)
    jd, fr = _jd_grid(jd0, fr0, offsets)
    ep, rp, _ = sat_p.sgp4_array(jd, fr)
    if np.any(ep != 0):
        raise ValueError("primary object could not be propagated over this window")

    coarse_thr = float(np.hypot(threshold_km, 0.5 * V_REL_MAX_KM_S * step_s)) + 1.0
    events = []
    ids = cand["NORAD_CAT_ID"].to_numpy(int)
    if chunk is None:  # keep each batch's position arrays to a few tens of MB
        chunk = int(max(10, min(200, 1_500_000 // len(offsets))))
    for k in range(0, n_cand, chunk):
        batch_ids = ids[k:k + chunk]
        sats = [to_satrec(records_by_norad[int(i)]) for i in batch_ids]
        e, r, _ = SatrecArray(sats).sgp4(jd, fr)
        d = np.linalg.norm(r - rp[None, :, :], axis=2)  # (S, T)
        d[e != 0] = np.inf
        for si, nid in enumerate(batch_ids):
            di = d[si]
            finite = di[np.isfinite(di)]
            if finite.size == 0 or finite.max() < FORMATION_KM:
                continue
            idx = np.where((di[1:-1] < di[:-2]) & (di[1:-1] <= di[2:]) & (di[1:-1] < coarse_thr))[0] + 1
            if di[0] < coarse_thr and di[0] <= di[1]:
                idx = np.append(idx, 0)
            if di[-1] < coarse_thr and di[-1] < di[-2]:
                idx = np.append(idx, len(di) - 1)
            for i in idx:
                ev = _refine(sat_p, sats[si], jd0, fr0, offsets[i], step_s)
                if ev is not None and ev[1] <= threshold_km:
                    events.append((records_by_norad[int(nid)]["OBJECT_NAME"], int(nid), *ev))
    out = pd.DataFrame(events, columns=["name", "norad_id", "tca_jd", "miss_km", "rel_speed_km_s"])
    return out.sort_values("miss_km").reset_index(drop=True), n_cand


def _refine(sat_p: Satrec, sat_c: Satrec, jd0: float, fr0: float, centre_s: float, step_s: float):
    off = centre_s + np.arange(-step_s, step_s + 0.5, 0.5)
    jd, fr = _jd_grid(jd0, fr0, off)
    e1, r1, v1 = sat_p.sgp4_array(jd, fr)
    e2, r2, v2 = sat_c.sgp4_array(jd, fr)
    ok = (e1 == 0) & (e2 == 0)
    if not ok.any():
        return None
    dist = np.where(ok, np.linalg.norm(r1 - r2, axis=1), np.inf)
    j = int(np.argmin(dist))
    return (jd[j] + fr[j], float(dist[j]), float(np.linalg.norm(v1[j] - v2[j])))


def jd_to_datetime(jd: float):
    """Julian date (UTC) -> timezone-aware datetime."""
    import datetime as dt
    return dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc) + dt.timedelta(days=jd - 2451545.0)
