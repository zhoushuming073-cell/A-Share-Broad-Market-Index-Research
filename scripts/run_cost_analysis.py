from scipy.stats import norm

from optimize_hedge_ratio import account, hac, metrics
from research_common import csv, factors, fit, np, pd, SEED
from run_pair_search import inputs, load, paths, pick


def diagnostics(frame):
    p,r,_,(tr,va)=inputs()
    dates=r.index[va:]
    days=p.index.to_series().diff().dt.days.loc[dates].to_numpy(float)
    lock=pick(frame)
    winner=frame[frame.method!="mr"].nlargest(1,"test_annual").iloc[0]
    benchmark=frame[(frame.a=="000001")&(frame.b=="000300")&(frame.spec=="static")].iloc[0]
    selected={k:(v.a,v.b,v.spec) for k,v in [("locked",lock),("test_best",winner),("benchmark",benchmark)]}
    ledgers=[]; costs=[]; scaled=[]; sensitivity=[]; splits=[]; regression=[]; bootstrap=[]
    bench=paths(benchmark.a,benchmark.b)["ret"][1,va:]
    rng=np.random.default_rng(SEED)
    n=len(dates); block=20
    starts=rng.integers(n,size=(2000,int(np.ceil(n/block)),1))
    ix=((starts+np.arange(block))%n).reshape(2000,-1)[:,:n]
    for (a,b),group in frame.groupby(["a","b"]):
        data=paths(a,b)
        x,y=r[a].iloc[va:].to_numpy(),r[b].iloc[va:].to_numpy()
        for j,spec in enumerate(data["ids"]):
            w,v=data["w"][j,va:],data["v"][j,va:]
            rr=data["ret"][j,va:]
            key=dict(a=a,b=b,spec=spec)
            hit=[label for label,k in selected.items() if k==(a,b,spec)]
            for gross in (.5,1.,2.):
                scaled.append(dict(**key,target_gross=gross,**metrics(gross*rr)))
            flags=np.ones(len(w),dtype=np.bool_)
            for bp in (0,2,5,10,20):
                for borrow in (0.,.01,.02,.03,.05):
                    z=account(x,y,w,v,days,flags,bp/10000,borrow)
                    if bp==0 and borrow==0:
                        np.testing.assert_allclose(z[:,0],rr,atol=1e-12)
                    costs.append(dict(**key,bp=bp,borrow=borrow,traded=z[:,2].sum(),borrow_paid=z[:,5].sum(),
                                      commission=z[:,6].sum(),**metrics(z[:,0])))
                    if hit and (borrow==.02 or (bp==0 and borrow==0)):
                        for label in hit:
                            ledgers.append(pd.DataFrame(dict(date=dates,role=label,**key,bp=bp,borrow=borrow,
                                  ret=z[:,0],nav=z[:,1],traded=z[:,2],long_weight=z[:,3],short_weight=z[:,4],
                                  short_collateral_proxy=-z[:,4],borrow_paid=z[:,5],commission=z[:,6])))
            if (a,b)==(lock.a,lock.b) and spec.startswith("rolling_120_"):
                freq=spec.split("_")[-1]
                period=dates.to_period({"D":"D","W":"W-FRI","M":"M","Q":"Q"}[freq])
                flags=np.r_[True,period[1:]!=period[:-1]]
                for mode,trade in [("daily_reset",np.ones(len(w),dtype=bool)),("hold_shares",flags)]:
                    z=account(x,y,w,v,days,trade,.0005,.02)
                    sensitivity.append(dict(**key,freq=freq,mode=mode,traded=z[:,2].sum(),
                                            mean_gross=(z[:,3]-z[:,4]).mean(),**metrics(z[:,0])))
            for label in hit:
                for name,sample in [("return",rr),("minus_benchmark",rr-bench)]:
                    means=sample[ix].mean(axis=1)*252
                    se,pval=hac(sample)
                    bootstrap.append(dict(role=label,series=name,annual_mean=sample.mean()*252,
                                          hac_se=se*252,hac_p=pval,block=block,replications=len(ix),
                                          ci_low=np.quantile(means,.025),ci_high=np.quantile(means,.975),
                                          remove_best5_total=metrics(np.sort(sample)[:-5])["total"]))
                for start,end in [("2018","2020"),("2021","2024"),("2025","2026")]:
                    mask=(r.index.year>=int(start))&(r.index.year<=int(end))
                    splits.append(dict(role=label,**key,start=start,end=end,**metrics(data["ret"][j,mask])))
                ser=pd.Series(rr,index=dates)
                ch=factors("CH4")
                for model,design in [(c,r[[c]].loc[dates]) for c in ["000001","000985","000300"]]+[
                        ("CH4",ch.loc[ch.index.intersection(dates),["mktrf","SMB","VMG","PMO"]])]:
                    f=fit(ser.loc[design.index],design)
                    for term,val,se,pv,ci in zip(["intercept",*design.columns],f.params,f.bse,f.pvalues,f.conf_int()):
                        regression.append(dict(role=label,model=model,term=term,beta=val,hac_se=se,p=pv,
                                               lo=ci[0],hi=ci[1],r2=f.rsquared,n=len(design),
                                               start=design.index[0],end=design.index[-1]))
    csv(pd.DataFrame(costs),"cost_sensitivity")
    csv(pd.DataFrame(scaled),"pair_leverage")
    csv(pd.concat(ledgers),"pair_selected_ledger")
    csv(pd.DataFrame(sensitivity),"pair_rebalance_control")
    csv(pd.DataFrame(splits),"pair_historical_splits")
    csv(pd.DataFrame(regression),"pair_factor_exposures")
    csv(pd.DataFrame(bootstrap),"pair_bootstrap")
    selected_pair=frame[(frame.a==lock.a)&(frame.b==lock.b)]
    csv(selected_pair,"hedge_method_comparison")
    # Synchronized centered circular block bootstrap maximum across strategies.
    # Shared date frequencies preserve contemporaneous cross-strategy correlation.
    matrix=np.concatenate([paths(a,b)["ret"][:,va:] for (a,b),_ in frame.groupby(["a","b"])])
    se=np.array([hac(s)[0] for s in matrix])
    good=se>1e-12
    matrix=matrix[good]; se=se[good]
    counts=np.array([np.bincount(row,minlength=n) for row in ix],dtype=float)
    means=matrix.mean(axis=1)
    obs=np.max(means/se)
    null=np.max((counts @ (matrix-means[:,None]).T/n)/se,axis=1)
    csv(pd.DataFrame([dict(trials=len(frame),nonconstant=len(se),observed_max_t=obs,block=block,
                           draws=len(null),max_t_p=(1+(null>=obs).sum())/(len(null)+1))]),"pair_search_max_test")
    # Exact legacy convention using Jan 4 as the inception close.
    old=[]
    for ts,te,ss,ee in [("2015","2017","2018","2020"),("2021-01-05","2024","2025","2026")]:
        train=r.loc[ts:te]; test=r.loc[ss:ee]
        h=train["000001"].cov(train["000300"])/train["000300"].var()
        raw=test["000001"]-h*test["000300"]
        for scale,s in [("original",raw),("gross_one",raw/(1+h))]:
            old.append(dict(start=ss,end=ee,h=h,scale=scale,**metrics(s)))
    csv(pd.DataFrame(old),"pair_legacy_check")


if __name__=="__main__":
    diagnostics(load("all_pair_results"))
