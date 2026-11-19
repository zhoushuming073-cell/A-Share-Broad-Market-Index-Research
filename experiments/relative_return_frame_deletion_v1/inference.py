import statsmodels.api as sm
from scipy import stats
from scipy.stats.mstats import winsorize
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from data import *
def run():
    rows=[];groups=[("full",np.ones(N,bool)),("2015-2017",T.year<2018),("2018-2020",(T.year>=2018)&(T.year<=2020)),
        ("2021-2024",(T.year>=2021)&(T.year<=2024)),("2025-2026",T.year>=2025)]+[(str(year),T.year==year) for year in np.unique(T.year)]
    for name,ix in groups:
        x=D[ix]
        for lag in ([0,5,10,20] if name=="full" else [5]):
            res=sm.OLS(x,np.ones((len(x),1))).fit(**(dict(cov_type="HAC",cov_kwds=dict(maxlags=lag,use_correction=True)) if lag else {}))
            lo,hi=res.conf_int()[0]
            rows.append(dict(group=name,method="HAC" if lag else "iid",lag=lag,n=len(x),estimate=x.mean(),se=res.bse[0],lo=lo,hi=hi,p=res.pvalues[0]))
    table=pd.DataFrame(rows);ix=table.group.str.fullmatch(r"\d{4}");table.loc[ix,"holm"]=multipletests(table.loc[ix,"p"],method="holm")[1]
    add=[]
    for lag,row in acorr_ljungbox(D,lags=[5,10,20],return_df=True).iterrows():add.append(dict(group="full",method="LjungBox",lag=lag,estimate=row.lb_stat,p=row.lb_pvalue))
    design=np.column_stack([np.ones(N)]+[ix.astype(float) for _,ix in groups[2:5]])
    fit=sm.OLS(D,design).fit(cov_type="HAC",cov_kwds=dict(maxlags=10,use_correction=True))
    test=fit.wald_test(np.eye(4)[1:],scalar=True)
    add.append(dict(group="four_periods",method="HAC_joint_equal_means",lag=10,estimate=test.statistic,p=test.pvalue))
    rng=np.random.default_rng(CFG["seed"]+91)
    for length in (5,20,60):
        prefix=np.r_[0,np.cumsum(np.r_[D,D[:length]])];full,tail=divmod(N,length);all_means=[]
        for start in range(0,CFG["bootstrap_trials"],500):
            starts=rng.integers(N,size=(500,full+bool(tail)))
            sums=(prefix[starts[:,:full]+length]-prefix[starts[:,:full]]).sum(axis=1)
            if tail:sums+=prefix[starts[:,-1]+tail]-prefix[starts[:,-1]]
            all_means.extend(sums/N)
        means=np.array(all_means);lo,hi=np.quantile(means,[.025,.975])
        add.append(dict(group="full",method="circular_bootstrap",lag=length,n=N,estimate=D.mean(),se=means.std(ddof=1),
            lo=lo,hi=hi,p=(1+(abs(means-D.mean())>=abs(D.mean())).sum())/(1+len(means))))
    save(pd.concat([table,pd.DataFrame(add)],ignore_index=True),"statistical_tests")
    scale=stats.median_abs_deviation(D,scale="normal");mu=np.median(D)
    for _ in range(200):
        weight=np.minimum(1,1.345*scale/np.maximum(abs(D-mu),1e-16));old=mu;mu=np.average(D,weights=weight)
        if abs(mu-old)<1e-14:break
    save([dict(method=k,estimate=v) for k,v in dict(mean=D.mean(),median=np.median(D),
        trim5=stats.trim_mean(D,.05),winsor5=np.mean(winsorize(D,limits=.05)),huber=mu).items()],"robust_locations")
