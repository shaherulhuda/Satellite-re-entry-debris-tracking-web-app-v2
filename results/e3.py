import sys, json, time
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import dataset, decay
t0=time.time()
cat, recs = dataset.load_catalogue()
cat = dataset.attach_closest_approaches(cat, dataset.load_closest_approaches())
comp = decay.risk_components(cat)
sat = (cat["source"]=="active") & cat["leo"]
cat, comp = cat[sat], comp[sat]
W = dict(decay.DEFAULT_WEIGHTS)
def rank(w):
    s = decay.risk_score(comp, w); return s.sort_values(ascending=False, kind="mergesort")
base = rank(W)
top20 = list(base.index[:20]); top50 = list(base.index[:50])
def compare(w, label):
    r = rank(w); t20 = set(r.index[:20]); t50=set(r.index[:50])
    # spearman over baseline top 50 positions in new ranking
    full_b = base.rank(ascending=False); full_r = r.rank(ascending=False)
    rho_all = full_b.corr(full_r)
    rho50 = full_b.loc[top50].rank().corr(full_r.loc[top50].rank())
    moves = (full_r.loc[top20]-full_b.loc[top20]).abs()
    return dict(case=label, top20_overlap=len(t20 & set(top20)), top50_overlap=len(t50 & set(top50)),
                rho_top50=round(rho50,3), rho_all_leo=round(rho_all,4), max_rank_move_top20=int(moves.max()),
                median_rank_move_top20=float(moves.median()))
rows=[compare(W,"default")]
for k in W:
    for f in (0.5, 2.0):
        w=dict(W); w[k]=W[k]*f; rows.append(compare(w, f"{k} x{f}"))
    w=dict(W); w[k]=0.0; rows.append(compare(w, f"{k} = 0"))
b=dict(W); 
for k in ("miss","speed","time"): b[k]=0.0
rows.append(compare(b,"base score only (no close-approach terms)"))
eq={k:1.0 for k in W}; rows.append(compare(eq,"equal weights"))
df=pd.DataFrame(rows); pd.set_option("display.width",200); print(df.to_string(index=False))
# drivers of top 10
c = comp.loc[top20[:10]]; contrib = pd.DataFrame({k: 100*W[k]*c[col] for k,col in decay.COMPONENT_OF.items()})
contrib.insert(0,"name",cat.loc[top20[:10],"OBJECT_NAME"]); contrib.insert(1,"score",base.loc[top20[:10]].round(1))
print(contrib.round(1).to_string(index=False))
print("mean share of score in top20:", (pd.DataFrame({k: W[k]*comp.loc[top20][col] for k,col in decay.COMPONENT_OF.items()}).sum()/ (base.loc[top20].sum()/100)).round(3).to_dict())
print("LEO active sats scored", int(sat.sum()), "runtime s", round(time.time()-t0,1))
df.to_csv("/home/claude/results/e3_weight_sensitivity.csv", index=False)
