from run_pair_search import inputs, paths, load, pick
from optimize_hedge_ratio import metrics
from research_common import np, pd, csv


def walk(frame):
    prices,r,_,_=inputs()
    years=r.index.year
    choices=[]; daily=[]; trials=[]; targets=[]
    # Train, validate, trade chronologically. Drop last validation observation for execution latency.
    for year in range(years.min()+4,years.max()+1):
        tr=np.where((years>=year-4)&(years<=year-2))[0]
        va=np.where(years==year-1)[0][:-1]
        te=np.where(years==year)[0]
        candidates=[]; test_paths={}
        for (a,b),group in frame.groupby(["a","b"],sort=True):
            z=paths(a,b)
            x,y=r[a].to_numpy(),r[b].to_numpy()
            h=np.cov(x[tr[:-1]],y[tr[:-1]])[0,1]/np.var(y[tr[:-1]],ddof=1)
            for j,spec in enumerate(z["ids"]):
                if spec.startswith("mr_"):
                    continue
                rr=z["ret"][j] if spec!="static" else (x-h*y)/(1+h)
                training=metrics(rr[tr]); validation=metrics(rr[va])
                rec=dict(a=a,b=b,spec=spec,method=spec.split("_")[0],train_sharpe=training["sharpe"],
                         valid_sharpe=validation["sharpe"],valid_active=np.mean(z["w"][j,va]!=0),
                         year=year,static_fit_h=h if spec=="static" else np.nan)
                candidates.append(rec)
                test_paths[(a,b,spec)]=rr[te]
        candidate=pd.DataFrame(candidates)
        q=pick(candidate)
        key=(q.a,q.b,q.spec)
        trade=test_paths[key]
        positions=pd.DataFrame(0.,index=r.index[te],columns=r.columns)
        z=paths(q.a,q.b)
        idx=list(z["ids"]).index(q.spec)
        if q.spec=="static":
            positions[q.a]=1/(1+q.static_fit_h)
            positions[q.b]=-q.static_fit_h/(1+q.static_fit_h)
        else:
            positions[q.a]=z["w"][idx,te]
            positions[q.b]=z["v"][idx,te]
        np.testing.assert_allclose((positions*r.loc[positions.index]).sum(axis=1),trade,atol=1e-12)
        targets.append(positions)
        for row in candidates:
            k=(row["a"],row["b"],row["spec"])
            trials.append(dict(**row,selected=k==key,**metrics(test_paths[k])))
        choices.append(dict(**q.to_dict(),train_start=r.index[tr[0]],train_end=r.index[tr[-1]],
                            valid_start=r.index[va[0]],valid_end=r.index[va[-1]],
                            test_start=r.index[te[0]],test_end=r.index[te[-1]],**metrics(trade)))
        daily.append(pd.DataFrame(dict(date=r.index[te],ret=trade,a=q.a,b=q.b,spec=q.spec)))
    csv(pd.DataFrame(trials),"walk_forward_trials")
    series=pd.concat(daily)
    series["nav"]=(1+series.ret).cumprod()
    # Carry marked holdings through annual switches, netting trades in shared legs.
    target=pd.concat(targets)
    marked=np.zeros(len(r.columns)); equity=1.; net=[]
    days=prices.index.to_series().diff().dt.days.loc[target.index].to_numpy()
    for k,(weight,change) in enumerate(zip(target.to_numpy(),r.loc[target.index].to_numpy())):
        start=equity
        after=start
        for _ in range(8):
            after=start-.0005*np.abs(weight*after-marked).sum()
        held=weight*after
        volume=np.abs(held-marked).sum()
        borrow=-np.minimum(held,0).sum()*.02*days[k]/365
        marked=held*(1+change)
        equity=after+held@change-borrow
        if k==len(target)-1:
            volume+=np.abs(marked).sum()
            equity-=.0005*np.abs(marked).sum()
        net.append((equity/start-1,equity,volume,borrow))
    series[["net_ret_5bp_2pct","net_nav","traded","borrow_paid"]]=np.array(net)
    for row in choices:
        sub=series[series.date.dt.year==row["year"]]
        row["net_total_5bp_2pct"]=metrics(sub.net_ret_5bp_2pct)["total"]
    csv(pd.DataFrame(choices),"walk_forward_results")
    csv(series,"walk_forward_daily")


if __name__=="__main__":
    walk(load("all_pair_results"))
