"""Run from the project root: python scripts/run_pair_search.py"""
from itertools import permutations
from pathlib import Path

from statsmodels.stats.multitest import multipletests

from build_pair_universe import universe
from optimize_hedge_ratio import hac, metrics, rolling_tests, specifications, weights
from research_common import ROOT, TABLES, csv, np, pd
from run_e2_regimes import classify

ARCHIVE=ROOT/"outputs/pair_paths"
CODE_TYPES={"a":str,"b":str,"code":str}


def load(name):
    return pd.read_csv(TABLES/(name+".csv"),dtype=CODE_TYPES)


def inputs():
    p,u=universe()
    r=p.pct_change(fill_method=None).iloc[1:]
    n=len(r)
    return p,r,u,(n*3//5,n*4//5)


def paths(a,b):
    with np.load(ARCHIVE/f"{a}_{b}.npz") as z:
        return {k:z[k] for k in z.files}


def collect():
    p,r,u,(tr,va)=inputs()
    ARCHIVE.mkdir(exist_ok=True,parents=True)
    eg=rolling_tests(p,r.index)
    csv(eg,"pair_rolling_cointegration")
    states=classify(p["000001"])[0].reindex(r.index).fillna("warmup").to_numpy()
    parts={"train":slice(0,tr),"valid":slice(tr,va),"test":slice(va,None),"full":slice(None)}
    main,base,segments=[],[],[]
    for a,b in permutations(p,2):
        x,y=r[a].to_numpy(),r[b].to_numpy()
        for name,sl in parts.items():
            xx,yy=x[sl],y[sl]
            base.append(dict(a=a,b=b,period=name,corr=np.corrcoef(xx,yy)[0,1],
                             price_gap=np.prod(1+xx)-np.prod(1+yy),ols_beta=np.cov(xx,yy)[0,1]/np.var(yy,ddof=1),
                             **metrics((xx-yy)/2)))
        packed={k:[] for k in ["ids","w","v","ret","traded","raw_ret"]}
        for ident,method,window,freq,entry,exit_z,h,side in specifications(p,r,a,b,r.index[:tr-1],eg):
            w,v=weights(h,side)
            ret=w*x+v*y
            raw=side*(x-np.nan_to_num(h)*y)*(w!=0)
            prevnav=np.r_[1,(1+ret[:-1]).cumprod()]
            trades=(np.abs(w*prevnav-np.r_[0,(w*prevnav*(1+x))[:-1]])
                    +np.abs(v*prevnav-np.r_[0,(v*prevnav*(1+y))[:-1]]))
            trades[-1]+=prevnav[-1]*(abs(w[-1]*(1+x[-1]))+abs(v[-1]*(1+y[-1])))
            for key,value in zip(packed,(ident,w,v,ret,trades,raw)):
                packed[key].append(value)
            row=dict(a=a,b=b,spec=ident,method=method,window=window,freq=freq,entry=entry,exit_z=exit_z)
            for part,sl in parts.items():
                rr,ww,vv=ret[sl],w[sl],v[sl]
                z=metrics(rr)
                active=ww!=0
                mkt=r["000985"].to_numpy()[sl]
                z.update(active=active.mean(),gross=np.abs(ww).mean()+np.abs(vv).mean(),net=np.mean(ww+vv),
                         long_weight=np.maximum(ww,0).mean()+np.maximum(vv,0).mean(),
                         short_weight=np.minimum(ww,0).mean()+np.minimum(vv,0).mean(),
                         capital_long_cash_plus_full_short_collateral=np.abs(ww).mean()+np.abs(vv).mean(),
                         mean_h=np.mean(h[sl][active]) if active.any() else np.nan,
                         market_beta=np.cov(rr,mkt)[0,1]/np.var(mkt,ddof=1),
                         turnover=(trades/prevnav)[sl].mean(),
                         raw_total=metrics(raw[sl])["total"],
                         raw_mean_gross=np.where(active,1+np.nan_to_num(h[sl]),0).mean())
                row.update({part+"_"+k:v0 for k,v0 in z.items()})
            se,pv=hac(ret[va:])
            row.update(test_hac_se_daily=se,test_hac_p=pv)
            years=[metrics(ret[va:][r.index[va:].year==yr])["total"] for yr in r.index[va:].year.unique()]
            regimes=[]
            for state in ("advance","sideways","decline","transition"):
                ix=states[va:]==state
                met=metrics(ret[va:][ix])
                segments.append(dict(a=a,b=b,spec=ident,state=state,**met))
                if met["n"]:
                    regimes.append(met["annual"])
            row.update(positive_years=np.mean(np.array(years)>0),worst_regime=min(regimes),
                       train_test_gap=abs(row["test_annual"]-row["train_annual"]))
            main.append(row)
        np.savez_compressed(ARCHIVE/f"{a}_{b}.npz",date=r.index.to_numpy(),**packed)
    frame=pd.DataFrame(main)
    basic=pd.DataFrame(base)
    for part in parts:
        summary=basic.loc[basic.period==part,["a","b","corr","price_gap","ols_beta"]]
        frame=frame.merge(summary.rename(columns={c:f"{part}_{c}" for c in ["corr","price_gap","ols_beta"]}),
                          on=["a","b"],validate="many_to_one",sort=False)
    frame["test_holm_p"]=multipletests(frame.test_hac_p,method="holm")[1]
    frame["test_bh_q"]=multipletests(frame.test_hac_p,method="fdr_bh")[1]
    direction=frame.method.ne("mr")
    score_cols=["test_annual","test_mdd","test_sharpe","positive_years","worst_regime","train_test_gap"]
    scores=frame.loc[direction,score_cols].copy()
    scores["train_test_gap"]*=-1
    frame.loc[direction,"robust_score"]=scores.rank(pct=True).mean(axis=1)
    csv(frame,"all_pair_results")
    csv(basic,"pair_basic_statistics")
    csv(pd.DataFrame(segments),"pair_test_regimes")
    rankings={"return":"test_annual","sharpe":"test_sharpe","robust":"robust_score"}
    for name,col in rankings.items():
        top=frame[direction].sort_values([col,"a","b","spec"],ascending=[False,True,True,True])
        csv(top.drop_duplicates(["a","b"]).head(20),f"top_{name}_pairs")
    return frame


def pick(frame):
    eligible=frame[(frame.method!="mr")&(frame.train_sharpe>0)&(frame.valid_active>=.5)]
    if eligible.empty:
        return None
    eligible=eligible.copy()
    eligible["lock_sharpe"]=eligible.valid_sharpe
    if "test_n" in frame:
        # The last validation close is reserved for execution.
        i=int(frame.train_n.iloc[0]); j=i+int(frame.valid_n.iloc[0])-1
        for (a,b),group in eligible.groupby(["a","b"]):
            z=paths(a,b)
            daily=z["ret"][:,i:j]
            scores=daily.mean(axis=1)*np.sqrt(252)/np.maximum(daily.std(axis=1,ddof=1),1e-15)
            eligible.loc[group.index,"lock_sharpe"]=group.spec.map(dict(zip(z["ids"],scores)))
    return eligible.sort_values(["lock_sharpe","a","b","spec"],ascending=[False,True,True,True]).iloc[0]


if __name__=="__main__":
    from run_walk_forward import walk
    from run_cost_analysis import diagnostics
    from generate_pair_report import report
    from pair_instruments import instruments
    results=collect()
    walk(results)
    diagnostics(results)
    instruments()
    report()
