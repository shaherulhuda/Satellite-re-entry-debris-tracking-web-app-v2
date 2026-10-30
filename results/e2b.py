import sys, time
sys.path.insert(0, sys.argv[1])
import pandas as pd, numpy as np
from sgp4.api import jday
import dataset, triage
cat, recs = dataset.load_catalogue()
ts=pd.Timestamp("2026-10-28T17:00:00"); j,f=jday(ts.year,ts.month,ts.day,ts.hour,ts.minute,ts.second)
t0=time.time(); ev,n = triage.screen_asset(recs[25544], cat, recs, j+f, hours=168, threshold_km=25, against="all")
rt=time.time()-t0
ev["tca"]=pd.to_datetime((ev.tca_jd-2440587.5)*86400, unit="s").dt.round("s")
ref=pd.DataFrame([(46292,"3CAT-5A","2026-11-03 23:47:50.642",0.913,7.713),(60518,"FLOCK 4BE-1","2026-10-29 11:23:14.802",3.020,7.138),
 (60207,"R5-S2","2026-11-02 15:52:12.602",3.867,14.745),(46292,"3CAT-5A","2026-11-04 03:40:18.545",4.823,7.767),(46507,"NETSAT-2","2026-11-04 15:05:08.456",4.893,6.032)],
 columns=["norad","name","tca","miss","speed"]); ref["tca"]=pd.to_datetime(ref.tca)
print(f"candidates {n}, events<=25km {len(ev)}, within 5 km {int((ev.miss_km<=5).sum())}, runtime {rt:.1f} s")
print("partners in catalogue:", {i:(i in recs) for i in ref.norad})
rows=[]
for r in ref.itertuples():
    c=ev[(ev.norad_id==r.norad)]
    if len(c):
        k=(c.tca-r.tca).abs().idxmin(); m=c.loc[k]
        dt=(m.tca-r.tca).total_seconds()
        rows.append(dict(ref=r.name,ref_tca=r.tca,ref_km=r.miss,ref_v=r.speed,app_tca=m.tca,dt_s=dt,app_km=round(m.miss_km,3),app_v=round(m.rel_speed_km_s,3),matched=abs(dt)<=120))
    else: rows.append(dict(ref=r.name,ref_tca=r.tca,ref_km=r.miss,ref_v=r.speed,matched=False))
cmp=pd.DataFrame(rows); print(cmp.to_string(index=False))
mm=cmp[cmp.matched]; print("matched",len(mm),"/5; mean |dmiss| km",round((mm.app_km-mm.ref_km).abs().mean(),2),"mean |dt| s",round(mm.dt_s.abs().mean(),1),"mean |dv| km/s",round((mm.app_v-mm.ref_v).abs().mean(),3))
print("app events within 5 km:"); print(ev[ev.miss_km<=5][["name","norad_id","tca","miss_km","rel_speed_km_s"]].to_string(index=False))
cmp.to_csv("/home/claude/results/e2_iss_vs_socrates.csv",index=False)
