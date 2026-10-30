import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os; R=os.path.dirname(os.path.abspath(__file__))+"/"; O=R+"figures/"
BLUE,ORANGE,INK,MUTED,GRID="#2a78d6","#eb6834","#222222","#5a5a5a","#e3e3e3"
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","Times New Roman"],"font.size":10,
 "axes.edgecolor":"#9a9a9a","axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.spines.top":False,"axes.spines.right":False})
def fin(fig,ax,name):
    ax.grid(axis="y" if ax.get_ylabel() else "x",color=GRID,lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(O+name,dpi=220); plt.close(fig)

# Fig 2 – E1 error
e1=pd.read_csv(R+"e1_reentry_check.csv")
err=e1.err_days.values; print("E1 n",len(err),"median|err|",np.median(abs(err)).round(2),"inside",e1.inside.sum())
fig,ax=plt.subplots(figsize=(6.3,2.6))
bins=np.arange(np.floor(err.min()),np.ceil(err.max())+1,1)
ax.hist(err,bins=bins,color=BLUE,edgecolor="white",linewidth=1.5)
ax.set_xlabel("Actual minus estimated re-entry time (days)"); ax.set_ylabel("Objects")
ax.axvline(0,color=INK,lw=1,ls="--")
o=e1.loc[e1.err_days.idxmax()]
ax.annotate(f"{o.OBJECT_NAME}\n(only object outside its window)",xy=(o.err_days,1),xytext=(o.err_days-5,8),fontsize=8.5,color=MUTED,
            arrowprops=dict(arrowstyle="-",color=MUTED,lw=0.8),ha="center")
fin(fig,ax,"fig_e1.png")

# Fig 3 – E2 by horizon
e2=pd.read_csv(R+"e2_watchlist_vs_socrates.csv")
e2["h"]=pd.cut(e2.ahead_days,[-1,1,2,99],labels=["Within 1 day","1–2 days","More than 2 days"])
g=e2.groupby("h",observed=False).matched.agg(["sum","count"]); print(g)
fig,ax=plt.subplots(figsize=(6.3,2.5))
y=np.arange(len(g))[::-1]
ax.barh(y,g["sum"],color=BLUE,height=0.55,label="Matched by the app")
ax.barh(y,g["count"]-g["sum"],left=g["sum"],color=ORANGE,height=0.55,label="Missed",edgecolor="white",linewidth=2)
for yi,(m,n) in zip(y,g.values): ax.text(n+0.3,yi,f"{m} of {n}",va="center",fontsize=9.5,color=INK)
ax.set_yticks(y); ax.set_yticklabels(g.index); ax.set_xlabel("SOCRATES close approaches (number)"); ax.set_xlim(0,19); ax.set_xticks(range(0,17,2))
ax.legend(frameon=False,loc="lower right",fontsize=9)
ax.grid(axis="x",color=GRID,lw=0.8); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig(O+"fig_e2.png",dpi=220); plt.close(fig)

# Fig 4 – E3 top-20 overlap by case
e3=pd.read_csv(R+"e3_weight_sensitivity.csv"); e3=e3[e3.case!="default"]; print(e3[["case","top20_overlap"]].to_string())
def lab(c):
    if c.startswith("base"): return "Base score only (no close-approach factors)"
    if c=="equal weights": return "All weights equal"
    k,op=c.split(" ",1); k={"miss":"Miss distance","decay":"Decay time","time":"Time to closest approach","speed":"Relative speed","type":"Object type"}.get(k,k.capitalize())
    return k+(" removed" if op=="= 0" else " weight ×"+op[1:].replace(".0",""))
e3["case"]=e3.case.map(lab); e3=e3.sort_values("top20_overlap")
fig,ax=plt.subplots(figsize=(6.3,0.15*len(e3)+0.8))
ax.barh(range(len(e3)),e3.top20_overlap,color=BLUE,height=0.62)
for i,v in enumerate(e3.top20_overlap): ax.text(v+0.2,i,str(v),va="center",fontsize=8.5,color=INK)
ax.set_yticks(range(len(e3))); ax.set_yticklabels(e3.case,fontsize=8); ax.set_xticks([0,5,10,15,20]); ax.set_xlim(0,21.5)
ax.set_xlabel("Satellites still in the top 20 (of 20)")
ax.grid(axis="x",color=GRID,lw=0.8); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig(O+"fig_e3.png",dpi=220); plt.close(fig)

# Fig 5 – share of top-20 score by factor (E3 run values)
f=pd.Series({"Miss distance":26,"Decay time":24,"Perigee":15,"Time to closest approach":11,"Inclination":11,"Relative speed":10,"Object type":2})[::-1]
fig,ax=plt.subplots(figsize=(6.3,2.5))
ax.barh(range(len(f)),f.values,color=BLUE,height=0.6)
for i,v in enumerate(f.values): ax.text(v+0.4,i,f"{v}%",va="center",fontsize=9,color=INK)
ax.set_yticks(range(len(f))); ax.set_yticklabels(f.index); ax.set_xlim(0,30); ax.set_xlabel("Share of the top-20 risk score (%)")
ax.grid(axis="x",color=GRID,lw=0.8); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig(O+"fig_e3w.png",dpi=220); plt.close(fig)
