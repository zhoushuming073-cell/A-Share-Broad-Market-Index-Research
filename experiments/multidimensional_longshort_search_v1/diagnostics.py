"""One-at-a-time perturbations of the final retrospective winner and historical window summaries."""
from search import *
from backtest import COLS
def extra():
    leaders=pd.read_csv(OUT/"top_return.csv");winner=leaders.iloc[0]
    p=json.loads(winner.params);a=[]
    for name,value in p.items():
        if name=="family":continue
        domain=spec(p["family"])[name]
        if isinstance(domain,list):
            variants=[q for q in domain if q!=value]
        else:variants=[max(domain[0],value*.8),min(domain[1],value*1.2)]
        for v in variants:
            q={**p,name:v};m=measure(q,0,len(R))
            a.append(dict(parameter=name,value=v,cagr=m["cagr"],mdd=m["mdd"],delta_cagr=m["cagr"]-winner.cagr,delta_mdd=m["mdd"]-winner.mdd))
    save(a,"parameter_effects")
    d=pd.read_csv(OUT/"daily_positions.csv",parse_dates=["date"]);rows=[]
    for name,g in d.groupby("strategy"):
        for period,l,r in [("early",2015,2017),("main",2018,2024),("append",2025,2026),("all",2015,2026)]:
            sub=g[g.date.dt.year.between(l,r)]
            if sub.empty:continue
            a=sub[COLS].to_numpy().copy();a[:,1:11]/=a[0,10]
            ix=DATES.get_indexer(sub.date);m=metrics(a,MONTH[ix],YEAR[ix],Y[ix])
            rows.append(dict(strategy=name,period=period,**m))
    save(rows,"periods")
    wf=pd.read_csv(OUT/"walk_forward.csv")
    dates=wf[wf.policy.eq("overall")]
    matched=[]
    for label,q in [("equal",BASE),("retrospective",p)]:
        for window in dates.itertuples():
            l=DATES.get_loc(window.first);r=DATES.get_loc(window.last)+1
            z=measure(q,l,r)
            matched.append(dict(rule=label,fold=window.fold,first=window.first,last=window.last,net=z["net"],cagr=z["cagr"]))
    save(matched,"matched_fixed_folds")
