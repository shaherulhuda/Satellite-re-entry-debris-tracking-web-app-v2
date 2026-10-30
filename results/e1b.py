import pandas as pd, numpy as np
c=pd.read_csv("results/e1_candidates.csv"); d=pd.read_csv("results/e1_decay_dates.csv")
m=c.merge(d,left_on="NORAD_CAT_ID",right_on="norad")
ep=pd.to_datetime(m.EPOCH); dd=pd.to_datetime(m.decay_date)
m["actual_days"]=(dd+pd.Timedelta(hours=12)-ep).dt.total_seconds()/86400
lo=(ep+pd.to_timedelta(m.decay_days_low,unit="D")).dt.normalize(); hi=(ep+pd.to_timedelta(m.decay_days_high,unit="D")).dt.normalize()
m["inside"]=(dd>=lo)&(dd<=hi)
m["err_days"]=m.actual_days-m.decay_days
m["ratio"]=m.actual_days/m.decay_days
m["starlink"]=m.OBJECT_NAME.str.contains("STARLINK")
def s(g,l): print(f"{l}: n={len(g)} inside window={g.inside.sum()} ({100*g.inside.mean():.0f}%) median |err|={g.err_days.abs().median():.1f} d, median err={g.err_days.median():+.1f} d, median actual/predicted={g.ratio.median():.2f}, within x2 of central={((g.ratio>=.5)&(g.ratio<=2)).mean()*100:.0f}%")
s(m,"all"); s(m[m.starlink],"Starlink"); s(m[~m.starlink],"other")
for lim in (3,10):
    s(m[m.decay_days<=lim],f"central<= {lim} d")
s(m[m.decay_days>10],"central>10 d")
print(m[["OBJECT_NAME","decay_days","decay_days_low","decay_days_high","actual_days","inside"]].round(1).to_string(index=False))
cut=pd.Timestamp("2026-10-29T12:59")
still=c[(c.still_in_catalogue)&(pd.to_datetime(c.EPOCH)+pd.to_timedelta(c.decay_days,unit="D")<cut)]
print("predicted (central) before 29 Oct but still in orbit:",len(still),"of which Starlink",still.OBJECT_NAME.str.contains("STARLINK").sum())
still2=c[(c.still_in_catalogue)&(pd.to_datetime(c.EPOCH)+pd.to_timedelta(c.decay_days_high,unit="D")<cut)]
print("predicted (whole window) before 29 Oct but still in orbit:",len(still2), list(still2.OBJECT_NAME))
m.to_csv("results/e1_reentry_check.csv",index=False)
