"""Recompute the four revisions from the repository's read-only price database."""
from itertools import combinations, permutations
from pathlib import Path
import json
import sys

import exchange_calendars as xc
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V1 = ROOT.parent / "four_pair_tests"
sys.path.insert(0, str(REPO / "scripts"))
from optimize_hedge_ratio import account, metrics
from research_common import db

S, H, C, U = "000001","000300","000903","000132"
NAMES = {S:"上证综指", H:"沪深300", C:"中证100", U:"上证100"}
FEE, BORROW, HOLD = .0005, .02, 252
RES, DATA = ROOT/"results",ROOT/"data"
PHASES = {"train":("2015","2017"),"validation":("2018","2024"),"test":("2025","2026")}
FREQUENCIES = {"D":None,"W":"W-FRI","M":"M","Q":"Q"}


def save(frame,name):
    frame.to_csv(RES/(name+".csv"),index=False,encoding="utf-8-sig",float_format="%.13g")


def prices():
    with db() as c:
        raw=pd.read_sql_query("SELECT trade_date,index_code,close,close_return FROM index_daily "
            "WHERE provider='eastmoney' AND index_code IN (?,?,?,?) AND trade_date BETWEEN ? AND ?",
            c,params=(*NAMES,"2015-01-05","2026-11-17"),parse_dates=["trade_date"])
        master=pd.read_sql_query("SELECT * FROM index_master WHERE provider='eastmoney' AND index_code IN (?,?,?,?)",
                                  c,params=tuple(NAMES))
    assert not raw.duplicated(["trade_date","index_code"]).any()
    p=raw.pivot(index="trade_date",columns="index_code",values="close").sort_index()
    calendar=xc.get_calendar("XSHG").sessions_in_range(p.index[0],p.index[-1]).tz_localize(None)
    np.testing.assert_array_equal(p.index,calendar)
    assert np.isfinite(p).all().all() and (p>0).all().all() and len(p)==2884
    archived=pd.read_csv(V1/"data/input_closes.csv",index_col=0,parse_dates=True)
    pd.testing.assert_frame_equal(p,archived,check_names=False)
    for code in p:
        file,=(REPO/"data/raw_index").glob(f"eastmoney_index_{code}_2015-01-01_2026-11-17_*.csv")
        original=pd.read_csv(file,parse_dates=["date"]).set_index("date")
        np.testing.assert_array_equal(p.index,original.index)
        np.testing.assert_allclose(p[code],original.close,rtol=0,atol=1e-10)
    stored=raw.pivot(index="trade_date",columns="index_code",values="close_return").loc[p.index]
    err=np.abs(stored.iloc[1:].values-p.pct_change(fill_method=None).iloc[1:].values)
    assert np.isfinite(err).all() and err.max()<1e-10
    master.to_csv(DATA/"index_master.csv",index=False,encoding="utf-8-sig")
    p.to_csv(DATA/"input_closes.csv",encoding="utf-8-sig")
    (RES/"data_audit.json").write_text(json.dumps(dict(closes=len(p),symbols=list(p),calendar="XSHG",
        calendar_version=xc.__version__,missing=0,duplicates=0,raw_files_matched=4,max_return_error=err.max()),indent=2))
    return p


def entry_windows(p):
    n=len(p)-HOLD
    prices_l,prices_s=p[S].values,p[H].values
    gaps=np.diff(p.index.to_numpy()).astype("timedelta64[D]").astype(int)
    rows=[]; paths={}
    for rule in ("units","daily"):
        net=np.ones(n); gross=np.ones(n)
        a=np.full(n,.5); b=-a.copy()
        net-=FEE
        fees=np.full(n,FEE); borrow=np.zeros(n)
        net_path=np.ones((n,HOLD+1)); gross_path=net_path.copy()
        for k in range(HOLD):
            x=prices_l[k+1:k+1+n]/prices_l[k:k+n]-1
            y=prices_s[k+1:k+1+n]/prices_s[k:k+n]-1
            if rule=="daily" and k:
                post=net.copy()
                for _ in range(7):
                    post=net-FEE*(np.abs(post/2-a)+np.abs(-post/2-b))
                fees+=net-post
                net=post; a=net/2; b=-a
            carry=-b*BORROW*gaps[k:k+n]/365
            net=net+a*x+b*y-carry
            borrow+=carry
            if rule=="daily":
                gross=gross*(1+(x-y)/2)
            else:
                gross=gross+a*x+b*y
            a=a*(1+x); b=b*(1+y)
            if k==HOLD-1:
                close_fee=FEE*(abs(a)+abs(b))
                net-=close_fee; fees+=close_fee
            gross_path[:,k+1]=gross;net_path[:,k+1]=net
        assert (net_path>0).all()
        rows.append(pd.DataFrame(dict(rule=rule,entry=p.index[:n],exit=p.index[HOLD:],holding=HOLD,
            gross=gross-1,fee=fees,borrow=borrow,net=net-1,
            path_vol=np.std(np.diff(net_path)/net_path[:,:-1],axis=1,ddof=1)*np.sqrt(252))))
        paths[rule+"_net"]=net_path; paths[rule+"_gross"]=gross_path
    all_rows=pd.concat(rows,ignore_index=True)
    save(all_rows,"e1_windows")
    np.savez_compressed(RES/"e1_paths.npz",entry=p.index[:n].values,**paths)
    desc=[]
    for rule,g in all_rows.groupby("rule"):
        for basis in ("net","gross"):
            y=g[basis]
            desc.append(dict(rule=rule,basis=basis,n=len(g),mean=y.mean(),median=y.median(),std=y.std(),
                q25=y.quantile(.25),q75=y.quantile(.75),min=y.min(),max=y.max(),profit=(y>0).mean(),
                loss=(y<0).mean(),mean_fee=g.fee.mean(),mean_borrow=g.borrow.mean(),mean_path_vol=g.path_vol.mean()))
    save(pd.DataFrame(desc),"e1_summary")
    # Resample consecutive entry dates jointly so that both rules see the same blocks.
    paired=all_rows.pivot(index="entry",columns="rule",values="net")[["units","daily"]]
    rng=np.random.default_rng(20261118)
    intervals=[]
    for length in (252,504):
        values=paired.to_numpy()
        extended=np.vstack([values,values[:length]])
        prefix=np.vstack([np.zeros(2),np.cumsum(extended,axis=0)])
        full,tail=divmod(len(values),length)
        starts=rng.integers(len(values),size=(5000,full+bool(tail)))
        sums=(prefix[starts[:,:full]+length]-prefix[starts[:,:full]]).sum(axis=1)
        if tail:
            sums+=prefix[starts[:,-1]+tail]-prefix[starts[:,-1]]
        means=sums/len(values)
        for j,rule in enumerate(("units","daily")):
            lo,hi=np.quantile(means[:,j],[.025,.975])
            intervals.append(dict(rule=rule,length=length,replications=len(means),seed=20261118,lo=lo,hi=hi))
    save(pd.DataFrame(intervals),"e1_block_intervals")


def evaluate(r,p,a,b,h,active,period,experiment,method,frequency,signal_date,fallback,net_same=False):
    ids=r.loc[slice(*PHASES[period])].index
    h=h.loc[ids]; active=active.loc[ids]
    wl=active.astype(float)/(1+h); ws=-wl*h
    x,y=r.loc[ids,a].to_numpy(),r.loc[ids,b].to_numpy()
    w=wl.to_numpy(); v=ws.to_numpy()
    if net_same:
        w=w+v;v=np.zeros(len(w))
    prev_dates=p.index[p.index.get_indexer(ids)-1]
    days=(ids-prev_dates).days.to_numpy(float)
    acct=account(x,y,w,v,days,np.ones(len(w),bool),FEE,BORROW)
    gross=w*x+v*y
    pre=np.r_[1.,acct[:-1,1]]
    # Exposure amounts in the repo ledger are relative to pre-trade capital.
    invested=(acct[:,3]-acct[:,4])*pre
    if net_same:
        # For equal symbols only the common term survives, including when net exposure is zero.
        common_money=(acct[:,3]+acct[:,4])*pre*x
        relative_money=np.zeros(len(w))
    else:
        common_money=invested*(wl+ws).values*(x+y)/2
        relative_money=invested*(x-y)/2
    common=(wl+ws).values*(x+y)/2
    relative=active.values*(x-y)/2
    df=pd.DataFrame(dict(date=ids,execution=prev_dates,signal_cutoff=signal_date.loc[ids].values,
        experiment=experiment,period=period,a=a,b=b,method=method,frequency=frequency,h=h.values,
        active=active.values,fallback=fallback.loc[ids].values,weight_a=w,weight_b=v,
        gross=gross,net=acct[:,0],net_nav=acct[:,1],gross_nav=np.cumprod(1+gross),
        turnover=acct[:,2],fee=acct[:,6],borrow=acct[:,5],common=common,relative=relative,
        common_money=common_money,relative_money=relative_money))
    np.testing.assert_allclose(common+relative,gross,atol=1e-13)
    np.testing.assert_allclose(common_money+relative_money-df.fee-df.borrow,
                               df.net_nav-pre,atol=1e-13)
    return df


def constant(r,value):
    return pd.Series(value,index=r.index)


def directions(r,p):
    accounts=[]; ranking={}
    for a,b in permutations((S,H,C),2):
        for period in PHASES:
            d=evaluate(r,p,a,b,constant(r,1.),constant(r,True),period,2,"one","D",
                       constant(r,pd.NaT),constant(r,False))
            accounts.append(d)
            if period=="validation":
                # Evaluate a standalone validation account liquidated at the ranking close.
                last=d.iloc[-2]; start=d.iloc[-3].net_nav
                x=r.loc[last.date,a];y=r.loc[last.date,b]
                deployed=start-last.fee
                unwind=FEE*deployed*((1+x)+(1+y))/2
                ranking[a,b]=last.net_nav-unwind-1
    test=pd.concat([d for d in accounts if d.period.iloc[0]=="test"])
    selections=[]; tests=[]; differences=[]
    for pair in combinations((S,H,C),2):
        a,b=max((pair,pair[::-1]),key=ranking.get)
        selected=test[(test.a==a)&(test.b==b)].set_index("date")
        reverse=test[(test.a==b)&(test.b==a)].set_index("date")
        diff=selected.net-reverse.net
        differences.append(pd.DataFrame(dict(date=diff.index,a=a,b=b,selected=selected.net.values,
                                              reverse=reverse.net.values,difference=diff.values)))
        win=selected.net_nav.iloc[-1]-1; rev=reverse.net_nav.iloc[-1]-1
        selections.append(dict(a=a,b=b,selection_end="2024-12-30",entry="2024-12-31",
            validation=ranking[a,b],validation_reverse=ranking[b,a],net=win,reverse=rev,
            gap=win-rev,hindsight=max(win,rev),hindsight_gap=max(win,rev)-win))
        for lag in (5,10,20):
            fit=sm.OLS(diff.values,np.ones((len(diff),1))).fit(cov_type="HAC",
                                          cov_kwds=dict(maxlags=lag,use_correction=True))
            lo,hi=fit.conf_int()[0]
            tests.append(dict(a=a,b=b,n=len(diff),lag=lag,mean=diff.mean(),se=fit.bse[0],
                              lo=lo,hi=hi,z=fit.tvalues[0],p=fit.pvalues[0]))
    tests=pd.DataFrame(tests)
    tests["holm"]=tests.groupby("lag").p.transform(lambda s:multipletests(s,method="holm")[1])
    save(tests,"e2_hac");save(pd.DataFrame(selections),"e2_selection")
    save(pd.concat(differences),"e2_paired_daily")
    return accounts


def ratio_schedule(r,a,b,method,freq,warmup):
    x,y=r[a],r[b]
    # To execute at previous close, consume data only through the preceding close.
    t=pd.Series(r.index,index=r.index); cutoff=t.shift(2)
    active=constant(r,True)
    if warmup:
        active.iloc[:121]=False
    if method in ("one","half","double"):
        return constant(r,{"one":1.,"half":.5,"double":2.}[method]),active,constant(r,pd.NaT),constant(r,False)
    if method=="static":
        train=x.loc[:"2017"].iloc[:-1].index
        first=x.index[:120]
        def slope(d):
            return x.loc[d].cov(y.loc[d])/y.loc[d].var()
        raw=constant(r,slope(train))
        cutoff=constant(r,train[-1])
        raw.loc[:"2017"]=slope(first); cutoff.loc[:"2017"]=first[-1]
    elif method=="rolling":
        raw=(x.rolling(120).cov(y)/y.rolling(120).var()).shift(2)
    elif method=="ewma":
        ew=y.ewm(alpha=.03,adjust=False,min_periods=60)
        raw=(ew.cov(x)/ew.var()).shift(2)
    elif method=="vol":
        raw=(x.rolling(120).std()/y.rolling(120).std()).shift(2)
    else:
        raise ValueError(method)
    if a==b:
        raw=raw.where(~(np.isfinite(raw)&(raw>0)),1.)
    if freq!="D":
        calendar=r.index.to_period(FREQUENCIES[freq])
        update=np.r_[True,calendar[1:]!=calendar[:-1]] | (active & ~active.shift(fill_value=False)).values
        latest=np.maximum.accumulate(np.where(update,np.arange(len(r)),0))
        raw=raw.iloc[latest].set_axis(r.index)
        cutoff=cutoff.iloc[latest].set_axis(r.index)
    valid=np.isfinite(raw)&(raw>0)&active
    h=raw.where(valid).ffill().fillna(1.)
    return h,active,cutoff.where(valid).ffill().where(active),active&~valid


def ratios(r,p,experiment):
    records=[]
    pairs=permutations((S,H,C),2) if experiment==3 else [(H,H),(H,U)]
    methods=["one","static","rolling","ewma","vol"]
    if experiment==4: methods+=["half","double"]
    for a,b in pairs:
        for method in methods:
            frequencies=FREQUENCIES if experiment==3 and method in ("rolling","ewma","vol") else ["D"]
            for freq in frequencies:
                h,active,cut,fallback=ratio_schedule(r,a,b,method,freq,experiment==3)
                for period in PHASES if experiment==3 else ["validation","test"]:
                    records.append(evaluate(r,p,a,b,h,active,period,experiment,method,freq,cut,
                                            fallback,net_same=experiment==4 and a==b))
    return records


def summarize(d,r):
    rows=[]
    keys=["experiment","period","a","b","method","frequency"]
    for key,g in d.groupby(keys,sort=False):
        m=metrics(g.net); gm=metrics(g.gross);market=r.loc[g.date,H].values
        prior=np.r_[1.,g.net_nav.values[:-1]]
        pg=np.r_[1.,g.gross_nav.values[:-1]]
        rows.append(dict(zip(keys,key))|dict(first=g.date.iloc[0],last=g.date.iloc[-1],n=len(g),
            gross=gm["total"],net=m["total"],annual=m["annual"],vol=m["vol"],sharpe=m["sharpe"],
            mdd=m["mdd"],gross_vol=gm["vol"],beta=np.cov(g.gross,market)[0,1]/np.var(market,ddof=1),
            mean_h=g.loc[g.active,"h"].mean(),turnover=np.sum(g.turnover/prior),
            annual_turnover=252*np.mean(g.turnover/prior),fee=g.fee.sum(),borrow=g.borrow.sum(),
            inactive=int((~g.active).sum()),fallback=int(g.fallback.sum()),
            gross_common=pg@g.common,gross_relative=pg@g.relative,
            net_common=g.common_money.sum(),net_relative=g.relative_money.sum()))
    return pd.DataFrame(rows)


def main():
    for path in (RES,DATA): path.mkdir(exist_ok=True)
    p=prices()
    r=p.pct_change(fill_method=None).iloc[1:]
    entry_windows(p)
    records=directions(r,p)
    records.extend(ratios(r,p,3))
    records.extend(ratios(r,p,4))
    d=pd.concat(records,ignore_index=True)
    for exp,g in d.groupby("experiment"):
        g.to_csv(RES/f"e{exp}_daily.csv.gz",index=False,float_format="%.13g",
                 compression=dict(method="gzip",mtime=0))
    save(summarize(d,r),"phase_statistics")


if __name__=="__main__":
    main()
