"""Close-approach screening of one object against the catalogue (SGP4 on the mean elements).

This is a SCREENING tool, not an operational conjunction assessment: mean elements carry
kilometre-level position error that grows with element age, there is no covariance and no
collision probability. Operators use Space-Track conjunction data messages for real decisions.

Method (fast enough for windows of many days):
  1. keep only objects whose altitude band overlaps the primary's;
  2. propagate the primary and the candidates with SGP4 every `step_s` seconds, keeping positions AND velocities;
  3. a distance minimum lies where the relative position r and velocity v satisfy r.v = 0, so a sign change of
     r.v from negative to positive between two samples marks one minimum (no fine time grid needed);
  4. between samples the relative motion is a cubic Hermite curve (error of a few metres at 120 s), so the minimum
     inside the step is found on 17 interpolated points; only minima closer than the threshold go on;
  5. those few are refined with exact SGP4 on a 0.25 s grid;
  6. pairs that stay within a few tens of km for the whole window (docked / flying in formation) are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

STEP_S = 120.0       # sampling step; cubic Hermite interpolation keeps the error to metres at this step
HERMITE_POINTS = 17  # interpolation points per step when locating a minimum
HERMITE_TOL_KM = 1.0  # keep minima this much beyond the threshold, to allow for interpolation error
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
           hours: float = 6.0, step_s: float = STEP_S, threshold_km: float = 10.0,
           alt_margin_km: float = 30.0, chunk: int | None = None,
           primary_band: tuple[float, float] | None = None) -> tuple[pd.DataFrame, int]:
    """Return (events, n_candidates). Events are sorted by miss distance.

    start_jd is a Julian date (UTC, as used by sgp4). Columns: name, norad_id, tca_jd,
    miss_km, rel_speed_km_s. `catalogue` is the set of candidates to screen against; if the
    primary is not in it, pass its (perigee_km, apogee_km) as `primary_band`. If the primary stops
    being propagatable (it re-enters) the window is cut there; `events.attrs["window_hours_effective"]`
    holds the screened length.
    """
    pid = int(primary["NORAD_CAT_ID"])
    if primary_band is None:
        prow = catalogue[catalogue["NORAD_CAT_ID"] == pid].iloc[0]
        primary_band = (float(prow["perigee_km"]), float(prow["apogee_km"]))
    cand = catalogue[(catalogue["NORAD_CAT_ID"] != pid)
                     & (catalogue["perigee_km"] <= primary_band[1] + alt_margin_km)
                     & (catalogue["apogee_km"] >= primary_band[0] - alt_margin_km)]
    n_cand = len(cand)
    empty = pd.DataFrame(columns=["name", "norad_id", "tca_jd", "miss_km", "rel_speed_km_s"])

    sat_p = to_satrec(primary)
    jd0 = float(np.floor(start_jd))
    fr0 = float(start_jd - jd0)
    horizon = hours * 3600.0
    offsets = np.append(np.arange(0.0, horizon, step_s), horizon)
    jd, fr = _jd_grid(jd0, fr0, offsets)
    ep, rp, vp = sat_p.sgp4_array(jd, fr)
    bad = np.flatnonzero(ep != 0)
    if bad.size:
        if bad[0] < 3:
            raise ValueError("primary object could not be propagated over this window")
        keep = int(bad[0])
        offsets, jd, fr, rp, vp = offsets[:keep], jd[:keep], fr[:keep], rp[:keep], vp[:keep]
    effective_hours = float(offsets[-1] / 3600.0)
    if n_cand == 0:
        empty.attrs["window_hours_effective"] = effective_hours
        return empty, 0

    T = len(offsets)
    dts = np.diff(offsets)
    ids = cand["NORAD_CAT_ID"].to_numpy(int)
    if chunk is None:  # keep each batch's arrays to a few tens of MB
        chunk = int(max(8, min(200, 1_200_000 // T)))
    u = np.linspace(0.0, 1.0, HERMITE_POINTS)
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    events = []
    for k in range(0, n_cand, chunk):
        batch = ids[k:k + chunk]
        sats = [to_satrec(records_by_norad[int(i)]) for i in batch]
        e, r, v = SatrecArray(sats).sgp4(jd, fr)
        ok = e == 0
        dr = np.where(ok[..., None], r - rp[None], np.nan)
        dv = np.where(ok[..., None], v - vp[None], np.nan)
        dist = np.linalg.norm(dr, axis=2)
        f = np.einsum("stk,stk->st", dr, dv)  # d(dist^2)/dt / 2
        with np.errstate(invalid="ignore"):
            far_enough = np.nanmax(np.where(np.isfinite(dist), dist, -1.0), axis=1) >= FORMATION_KM
            sel = (f[:, :-1] < 0) & (f[:, 1:] >= 0) & far_enough[:, None]
        si, ti = np.nonzero(sel)
        found = []  # (sat index in batch, centre offset in s)
        if len(si):
            dt = dts[ti][:, None, None]
            pts = (h00[None, :, None] * dr[si, ti][:, None, :] + h10[None, :, None] * dt * dv[si, ti][:, None, :]
                   + h01[None, :, None] * dr[si, ti + 1][:, None, :] + h11[None, :, None] * dt * dv[si, ti + 1][:, None, :])
            dmin = np.linalg.norm(pts, axis=2)
            best = dmin.argmin(axis=1)
            close = dmin[np.arange(len(si)), best] < threshold_km + HERMITE_TOL_KM
            for s_, t_, b_ in zip(si[close], ti[close], best[close]):
                found.append((int(s_), float(offsets[t_] + u[b_] * dts[t_])))
        with np.errstate(invalid="ignore"):  # minima at the window edges
            for s_ in np.flatnonzero(far_enough & (f[:, 0] >= 0) & (dist[:, 0] < threshold_km + HERMITE_TOL_KM)):
                found.append((int(s_), 0.0))
            for s_ in np.flatnonzero(far_enough & (f[:, -1] < 0) & (dist[:, -1] < threshold_km + HERMITE_TOL_KM)):
                found.append((int(s_), float(offsets[-1])))
        for s_, centre in found:
            ev = _refine(sat_p, sats[s_], jd0, fr0, centre, 0.0, float(offsets[-1]))
            if ev is not None and ev[1] <= threshold_km:
                events.append((records_by_norad[int(batch[s_])]["OBJECT_NAME"], int(batch[s_]), *ev))
    out = pd.DataFrame(events, columns=["name", "norad_id", "tca_jd", "miss_km", "rel_speed_km_s"])
    out = out.sort_values("miss_km").reset_index(drop=True)
    out.attrs["window_hours_effective"] = effective_hours
    return out, n_cand


def _refine(sat_p: Satrec, sat_c: Satrec, jd0: float, fr0: float, centre_s: float, lo_s: float, hi_s: float,
            half_s: float = 10.0, res_s: float = 0.25):
    """Exact closest approach near centre_s (within [lo_s, hi_s]): returns (jd, miss_km, rel_speed_km_s) or None."""
    off = np.arange(max(centre_s - half_s, lo_s), min(centre_s + half_s, hi_s) + res_s / 2, res_s)
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
