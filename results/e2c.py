import sys, time
sys.path.insert(0, sys.argv[1])
import pandas as pd, numpy as np
from sgp4.api import jday
import dataset, triage
cat, recs = dataset.load_catalogue()
ref = pd.read_csv("results/e2_socrates_watchlist.csv", parse_dates=["tca"])
ts = pd.Timestamp("2026-10-29T08:00:00"); j, f = jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
rows = []; times = []; app5 = 0
for prim in ref.primary.unique():
    t0 = time.time()
    ev, n = triage.screen_asset(recs[int(prim)], cat, recs, j + f, hours=168, threshold_km=25, against="all")
    times.append(time.time() - t0)
    ev["tca"] = pd.to_datetime((ev.tca_jd - 2440587.5) * 86400, unit="s")
    app5 += int((ev.miss_km <= 5).sum())
    for r in ref[ref.primary == prim].itertuples():
        c = ev[ev.norad_id == r.partner]
        row = dict(primary=prim, partner=r.partner_name, ref_tca=r.tca, ref_km=r.miss, ref_v=r.speed, in_data=int(r.partner) in recs)
        if len(c):
            k = (c.tca - r.tca).abs().idxmin(); m = c.loc[k]; dt = (m.tca - r.tca).total_seconds()
            row.update(app_km=round(m.miss_km, 3), app_v=round(m.rel_speed_km_s, 3), dt_s=round(dt, 1), matched=abs(dt) <= 120)
        else:
            row.update(matched=False)
        rows.append(row)
cmp = pd.DataFrame(rows)
cmp["ahead_days"] = (cmp.ref_tca - pd.Timestamp("2026-10-29T12:59")).dt.total_seconds() / 86400
mm = cmp[cmp.matched]
print(cmp.round(3).to_string(index=False))
print(f"\nreference events {len(cmp)}; partners in data {cmp.in_data.sum()}; matched {len(mm)}")
print(f"|dt| s: median {mm.dt_s.abs().median():.2f}, max {mm.dt_s.abs().max():.2f}")
dk = (mm.app_km - mm.ref_km).abs(); print(f"|dmiss| km: median {dk.median():.2f}, max {dk.max():.2f}; within 1 km {(dk<=1).sum()}")
print(f"|dv| km/s median {(mm.app_v-mm.ref_v).abs().median():.3f}")
print("app also flagged within 5 km (all its events):", app5)
print("runtime per satellite s:", [round(t, 1) for t in times])
um = cmp[~cmp.matched]; print("unmatched:\n", um.to_string(index=False))
cmp.to_csv("results/e2_watchlist_vs_socrates.csv", index=False)
