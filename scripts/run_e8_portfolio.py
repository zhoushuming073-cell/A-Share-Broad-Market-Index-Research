from research_common import *
from scipy.optimize import minimize
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


def simulate(r, weights, fee=0):
    """Targets rebalanced at the previous close, then held through current close."""
    weights = weights.reindex(r.index)
    position = np.zeros(r.shape[1])
    rows = []
    for day, ret in r.iterrows():
        target = weights.loc[day]
        previous = position.copy()
        if target.notna().all():
            position = target.to_numpy()
        turnover = np.abs(position-previous).sum()
        gross = position @ ret.to_numpy()
        net = (1-fee*turnover)*(1+gross)-1
        rows.append([day,gross,net,turnover,fee*turnover,*position])
        position = position * (1+ret.to_numpy())/(1+gross)
    return pd.DataFrame(rows,columns=["trade_date","gross","net","traded_notional","cost",*r.columns]).set_index("trade_date")


def target_weights(r, method):
    schedule = pd.DataFrame(np.nan,index=r.index,columns=r.columns)
    if method=="buy_hold":
        schedule.iloc[0]=1/r.shape[1]
        return schedule
    for i,day in enumerate(r.index):
        if i and r.index[i-1].month==day.month:
            continue
        if method=="monthly_equal":
            schedule.iloc[i]=1/r.shape[1]
        elif i>=253:
            # Estimate through t-2, execute at t-1 close.
            covariance=r.iloc[i-253:i-1].cov().to_numpy()*252
            sol=minimize(lambda w: w@covariance@w, np.full(r.shape[1],1/r.shape[1]),
                         jac=lambda w:2*covariance@w,method="SLSQP",bounds=[(.10,.55)]*r.shape[1],
                         constraints=[{"type":"eq","fun":lambda w:w.sum()-1,
                                       "jac":lambda w:np.ones_like(w)}],options={"ftol":1e-12})
            if not sol.success:
                raise ValueError(f"Optimization failed on {day}: {sol.message}")
            schedule.iloc[i]=sol.x
    return schedule


def stats(r, benchmark, rf=0):
    excess = r-rf
    down = np.sqrt(np.minimum(excess,0).pow(2).mean()*252)
    ann_mean = excess.mean()*252
    vol = r.std()*np.sqrt(252)
    return dict(**window(r.index),cagr=np.exp(np.log1p(r).mean()*252)-1,vol_ann=vol,
                sharpe=ann_mean/(excess.std()*np.sqrt(252)),sortino=ann_mean/down,downside_ann=down,
                max_drawdown=drawdown(r),tracking_error_ann=(r-benchmark).std()*np.sqrt(252),
                correlation=r.corr(benchmark))


def spanning(r, benchmarks, candidates):
    # Joint Newey-West Wald on alpha=0 and sum of slopes=1 per added asset.
    x = np.column_stack([np.ones(len(r)), r[benchmarks].to_numpy()])
    y = r[candidates].to_numpy()
    inv = np.linalg.inv(x.T@x)
    coeff = inv@x.T@y
    residual = y-x@coeff
    score = (residual[:,:,None]*x[:,None,:]).reshape(len(r),-1)
    meat = score.T@score
    for lag in range(1,HAC_LAGS+1):
        cross=score[lag:].T@score[:-lag]
        meat+=(1-lag/(HAC_LAGS+1))*(cross+cross.T)
    bread=np.kron(np.eye(len(candidates)),inv)
    cov=bread@meat@bread*len(r)/(len(r)-x.shape[1])
    restriction=np.kron(np.eye(len(candidates)),np.array([[1]+[0]*len(benchmarks),[0]+[1]*len(benchmarks)]))
    delta=restriction@coeff.T.ravel()-np.tile([0,1],len(candidates))
    v=restriction@cov@restriction.T
    statistic=float(delta@np.linalg.solve(v,delta))
    return dict(**window(r.index),benchmarks="+".join(benchmarks),candidates="+".join(candidates),
                wald=statistic,df=len(delta),p=chi2.sf(statistic,len(delta)),
                alpha_daily=str(coeff[0].tolist()),slope_sum=str(coeff[1:].sum(axis=0).tolist()))


def main():
    _,r,_=market()
    target=r[TARGETS]
    dynamic=target_weights(target,"dynamic")
    oos=r.loc[dynamic.dropna().index[0]:]
    policies={"buy_hold":target_weights(target,"buy_hold"),"monthly_equal":target_weights(target,"monthly_equal"),
              "dynamic_minvar":dynamic}
    rows,daily=[],[]
    for sample,d in [("full",r),("common_oos",oos)]:
        for code in CODES:
            rows.append(dict(sample=sample,strategy=code,cost_bps=0,annual_traded_notional=0,total_cost_fraction=0,
                             risk_free="zero",**stats(d[code],d["000001"])))
        for method, w in policies.items():
            if sample=="full" and method=="dynamic_minvar":
                continue
            w=w.loc[d.index].copy()
            if w.iloc[0].isna().all():
                w.iloc[0]=.25
            for fee in [0,5,10,20]:
                s=simulate(d[TARGETS],w,fee/10000)
                rows.append(dict(sample=sample,strategy=method,cost_bps=fee,risk_free="zero",
                                 annual_traded_notional=252*s.traded_notional.mean(),total_cost_fraction=s.cost.sum(),
                                 **stats(s.net,d["000001"])))
                daily.append(s.assign(sample=sample,strategy=method,cost_bps=fee).reset_index())
    rf=factors()["rf_dly"].dropna()
    for x in daily:
        if x["sample"].iloc[0]!="common_oos" or x.cost_bps.iloc[0]!=10:
            continue
        s=x.set_index("trade_date").net
        s=s.loc[s.index.intersection(rf.index)]
        rows.append(dict(sample="oos_rf_overlap",strategy=x.strategy.iloc[0],cost_bps=10,risk_free="CH3",
                         **stats(s,r.loc[s.index,"000001"],rf.loc[s.index])))
    for code in CODES:
        s=oos[code].loc[oos.index.intersection(rf.index)]
        rows.append(dict(sample="oos_rf_overlap",strategy=code,cost_bps=0,risk_free="CH3",
                         **stats(s,r.loc[s.index,"000001"],rf.loc[s.index])))
    csv(pd.DataFrame(rows),"e8_performance")
    paths=pd.concat(daily,ignore_index=True)
    csv(paths,"e8_daily_portfolios")
    csv(r.corr(),"e8_correlations",True)
    tests=pd.DataFrame([spanning(r,base,cand) for base in [[c] for c in BENCHES]+[BENCHES]
                        for cand in [TARGETS]+[[c] for c in TARGETS]])
    tests["p_holm"]=multipletests(tests.p,method="holm")[1]
    csv(tests,"e8_spanning")
    rng=np.random.default_rng(SEED)
    subset=paths[(paths["sample"]=="common_oos") & (paths.cost_bps==10)]
    aligned=subset.pivot(index="trade_date",columns="strategy",values="net").join(oos[["000001","000300"]])
    values=aligned.to_numpy()
    intervals=[]
    for block in (10,20,40):
        starts=rng.integers(0,len(values),size=(1500,int(np.ceil(len(values)/block))))
        indices=((starts[:,:,None]+np.arange(block))%len(values)).reshape(1500,-1)[:,:len(values)]
        samples=values[indices]
        vols=samples.std(axis=1,ddof=1)*np.sqrt(252)
        sharpes=samples.mean(axis=1)*252/vols
        for baseline in ["monthly_equal","000001","000300"]:
            i, j=aligned.columns.get_loc("dynamic_minvar"),aligned.columns.get_loc(baseline)
            for metric, simulations, point in [("vol_ann",vols,aligned.std()*np.sqrt(252)),
                                                ("sharpe_zero",sharpes,aligned.mean()*np.sqrt(252)/aligned.std())]:
                lo,hi=np.quantile(simulations[:,i]-simulations[:,j],[.025,.975])
                intervals.append(dict(**window(aligned.index),baseline=baseline,metric=metric,block_days=block,
                                      delta=point.iloc[i]-point.iloc[j],lo=lo,hi=hi,replicates=1500))
    csv(pd.DataFrame(intervals),"e8_bootstrap_differences")
    fig,ax=plt.subplots(figsize=(10,4.6))
    for code in ["000001","000985","000300"]:
        ax.plot(oos.index,(1+oos[code]).cumprod(),label=NAMES[code],lw=1)
    for name in ["monthly_equal","dynamic_minvar"]:
        p=paths[(paths["sample"]=="common_oos")&(paths.strategy==name)&(paths.cost_bps==10)]
        ax.plot(p.trade_date,(1+p.net).cumprod(),label=name.replace("_"," "),lw=1.4)
    ax.set_ylabel("Price-based wealth (initial 1)")
    ax.legend(ncol=3,fontsize=8)
    figure(fig,"e8_oos_wealth")


if __name__=="__main__":
    main()
