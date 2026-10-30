import sys, json
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import orbits, decay
old = decay.estimate_decay(orbits.catalog_from_records(json.load(open(sys.argv[2]))))
new = decay.estimate_decay(orbits.catalog_from_records(json.load(open(sys.argv[1]+"/active.json"))))
for d in (old,new):
    d["h"] = decay.mean_altitude_km(d["MEAN_MOTION"]); d["type"]=d["OBJECT_NAME"].map(decay.classify_type)
    d["leo"]=(d["h"]<2000)&(d["perigee_km"]<2000)
old=old[old.leo]; 
cut = pd.Timestamp("2026-10-29T12:59:00")
# ---- Part B: predicted re-entries between old epoch and the 29 Oct download
old["elapsed"] = (cut - old["EPOCH_DT"]).dt.total_seconds()/86400
newids=set(new.NORAD_CAT_ID)
old["still_in_catalogue"]=old.NORAD_CAT_ID.isin(newids)
est=old[old.decay_status!="no measurable decay"]
pred_central = est[est.decay_days<=est.elapsed]
pred_window = est[est.decay_days_low<=est.elapsed]
print("LEO objects 6 Oct:",len(old)," with estimate:",len(est))
print("central estimate before 29 Oct:",len(pred_central)," gone from catalogue:",int((~pred_central.still_in_catalogue).sum()))
print("window lower bound before 29 Oct:",len(pred_window)," gone:",int((~pred_window.still_in_catalogue).sum()))
gone = old[~old.still_in_catalogue]
print("all LEO objects gone from active catalogue:",len(gone))
cols=["NORAD_CAT_ID","OBJECT_NAME","EPOCH","h","perigee_km","decay_days","decay_days_low","decay_days_high","decay_status","elapsed","still_in_catalogue"]
cand = pd.concat([pred_window, gone]).drop_duplicates("NORAD_CAT_ID")[cols].sort_values("decay_days")
cand.to_csv("results/e1_candidates.csv",index=False)
print(cand.round(1).to_string(index=False, max_rows=80))
# ---- Part A: predicted vs observed altitude loss for objects in both snapshots
m = old.merge(new[["NORAD_CAT_ID","EPOCH_DT","h"]], on="NORAD_CAT_ID", suffixes=("","_new"))
m = m[m.decay_status=="estimate"]
m["dt"]=(m.EPOCH_DT_new-m.EPOCH_DT).dt.total_seconds()/86400
m=m[m.dt>=7]
def pred_drop(h0, r0, dt, steps=200):
    h=h0; step=dt/steps
    for _ in range(steps):
        h = h - r0*decay.density(h)/decay.density(h0)*step
        if h<=120: return h0-120
    return h0-h
m["pred_drop"]=[pred_drop(a,b,c) for a,b,c in zip(m.h,m.decay_km_day,m.dt)]
m["obs_drop"]=m.h-m.h_new
m["ratio"]=m.obs_drop/m.pred_drop
m["group"]=np.where(m.OBJECT_NAME.str.contains("STARLINK"),"Starlink (manoeuvring)", np.where(m.type=="rocket body","rocket body","other payload"))
def summ(g):
    raised=(g.obs_drop<0).sum(); r=g.ratio[g.obs_drop>0]
    return pd.Series(dict(n=len(g), raised=int(raised), within_x2=round(((g.ratio>=0.5)&(g.ratio<=2)).mean()*100,1),
        median_ratio=round(r.median(),2), q1=round(r.quantile(.25),2), q3=round(r.quantile(.75),2),
        median_abs_err_km=round((g.obs_drop-g.pred_drop).abs().median(),2), median_dt=round(g.dt.median(),1)))
out = pd.concat([m.groupby("group").apply(summ, include_groups=False), summ(m).to_frame("all").T])
print(out.to_string())
# low objects only (where decay matters): h<450
low=m[m.h<450]; print("h<450 km:"); print(pd.concat([low.groupby("group").apply(summ, include_groups=False), summ(low).to_frame("all").T]).to_string())
m.to_csv("results/e1_rate_check.csv",index=False)
