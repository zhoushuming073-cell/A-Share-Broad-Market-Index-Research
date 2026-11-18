"""Rebuild the four experiments from the repository's read-only SQLite snapshot."""
import sys
from itertools import permutations
from pathlib import Path
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from optimize_hedge_ratio import account, metrics, weights
from research_common import db, np, pd, plt, sm
from matplotlib.ticker import PercentFormatter

S, H, C, U = "000001", "000300", "000903", "000132"
NAME = {S:"上证综指", H:"沪深300", C:"中证100", U:"上证100"}
METHOD = {"one":"1∶1","static":"静态OLS","rolling":"120日OLS","ewma":"EWMA 0.97",
          "vol":"120日波动率比","half":"固定0.5","double":"固定2"}
PERIODS = {"训练":"2015:2017","验证前段":"2018:2020","验证后段":"2021:2024",
           "验证合计":"2018:2024","测试":"2025:2026"}
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
FIG = HERE / "figures"
DATA = HERE / "data"
for folder in (FIG,DATA):
    folder.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False})


def save(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig",float_format="%.12g")


def plot_done(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/(name+".png"),dpi=240,bbox_inches="tight")
    with plt.rc_context({"svg.fonttype":"none"}):
        fig.savefig(FIG/(name+".svg"),bbox_inches="tight")
    plt.close(fig)


def mask(idx, span):
    a,b=map(int,span.split(":"))
    return (idx.year>=a)&(idx.year<=b)


def label(a,b):
    return NAME[a]+" / "+NAME[b]


def observations():
    with db() as conn:
        data=pd.read_sql_query("SELECT trade_date,index_code,close FROM index_daily WHERE provider='eastmoney' "
                               "AND index_code IN (?,?,?,?) AND trade_date BETWEEN '2015-01-05' AND '2026-11-17'",
                               conn,params=(S,H,C,U),parse_dates=["trade_date"])
    p=data.pivot(index="trade_date",columns="index_code",values="close").sort_index()
    assert len(p)==2884 and p.notna().all().all() and (p>0).all().all()
    p.to_csv(DATA/"input_closes.csv",encoding="utf-8-sig")
    return p


def entry_test(p,r,days):
    length=252
    ones=np.ones(length)
    rows=[]; paths=[]
    for i in range(len(r)-length+1):
        seg=r.iloc[i:i+length]
        gross=np.cumprod(1+(seg[S].values-seg[H].values)/2)
        acc=account(seg[S].values,seg[H].values,.5*ones,-.5*ones,days[i:i+length],
                    ones.astype(bool),.0005,.02)
        rows.append((p.index[i],seg.index[-1],length,gross[-1]-1,acc[-1,1]-1,
                     acc[:,5].sum(),acc[:,6].sum()))
        paths.append(np.stack([np.r_[1,gross],np.r_[1,acc[:,1]]]))
    d=pd.DataFrame(rows,columns=["entry","exit","holding_days","gross","net","borrow_cost","fees"])
    save(d,"e1_all_entries")
    np.savez_compressed(OUT/"e1_holding_paths.npz",entry=d.entry.values,days=np.arange(length+1),
                        basis=np.array(["gross","net"]),nav=np.array(paths))
    fig,(ax,bx)=plt.subplots(2,1,figsize=(10,6.2))
    for key,title in [("gross","毛收益"),("net","成本后")]:
        ax.plot(d.entry,d[key],label=title,lw=1)
    ax.axhline(0,color="gray",lw=.7); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("每个建仓日随后252个交易日的收益");ax.legend()
    for k in (d.net.idxmin(),d.net.idxmax()):
        bx.plot(paths[k][1],label=f"{d.entry[k]:%Y-%m-%d} 建仓")
    bx.set(xlabel="持有交易日",ylabel="成本后净值",title="按期末净收益回看的最高与最低窗口")
    bx.legend()
    plot_done(fig,"e1_entry")
    fig,ax=plt.subplots(figsize=(8,3.7))
    bins=np.linspace(min(d.net.min(),d.gross.min()),max(d.net.max(),d.gross.max()),32)
    for c,n in [("gross","毛收益"),("net","成本后")]:
        ax.hist(d[c],bins=bins,alpha=.55,label=n)
    ax.set(xlabel="每个窗口的252日持有收益",ylabel="建仓日期数",title="相同持有期限下的收益分布")
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.axvline(0,color="black",lw=.8);ax.legend()
    plot_done(fig,"e1_distribution")
    return d


def hedge_ratios(x,y):
    # Leave the final calibration close free for the first 2018 trade.
    train=x.loc[:"2017"].iloc[:-1].index
    static=x.loc[train].cov(y.loc[train])/y.loc[train].var()
    ew=y.ewm(alpha=.03,adjust=False,min_periods=60)
    return {"one":1., "static":static,
            "rolling":(y.rolling(120).cov(x)/y.rolling(120).var()).shift(2),
            "ewma":(ew.cov(x)/ew.var()).shift(2),
            "vol":(x.rolling(120).std()/y.rolling(120).std()).shift(2)}


def simulate(r,days,a,b,rule,h,experiment):
    sample=r if experiment==2 else r.loc["2018":]
    x,y=sample[a].values,sample[b].values
    h=pd.Series(h,index=r.index).loc[sample.index].values
    assert np.all(np.isfinite(h)&(h>0))
    w,v=weights(h,np.ones(len(h)))
    g=w*x+v*y
    acc=account(x,y,w,v,days[-len(h):],np.ones(len(h),bool),.0005,.02)
    # Taking the long leg as market reference is the requested CSI300 decomposition in E4.
    direction=(w+v)*x
    relative=-v*(x-y)
    np.testing.assert_allclose(direction+relative,g,atol=1e-14)
    d=pd.DataFrame(dict(date=sample.index,experiment=experiment,a=a,b=b,method=rule,h=h,w_long=w,w_short=v,
                        gross=g,net=acc[:,0],gross_nav=np.cumprod(1+g),net_nav=acc[:,1],
                        turnover=acc[:,2],borrow=acc[:,5],fee=acc[:,6],direction=direction,relative=relative))
    if a==b:
        np.testing.assert_allclose(relative,0)
        if rule not in ("half","double"):
            np.testing.assert_allclose(g,0,atol=1e-13)
    return d


def stats(d,r):
    rows=[]
    for (e,a,b,m),group in d.groupby(["experiment","a","b","method"],sort=False):
        for name,span in dict(PERIODS,全段="2015:2026").items():
            cut=group.loc[mask(pd.DatetimeIndex(group.date),span)]
            if cut.empty: continue
            market=r.loc[cut.date,H].to_numpy()
            pre=np.r_[1,(1+cut.gross.to_numpy()[:-1]).cumprod()]
            gm=metrics(cut.gross); nm=metrics(cut.net)
            dc=pre@cut.direction; rc=pre@cut.relative
            np.testing.assert_allclose(dc+rc,gm["total"],atol=1e-12)
            rows.append(dict(experiment=e,a=a,b=b,method=m,period=name,first=cut.date.iloc[0],
                last=cut.date.iloc[-1],n=len(cut),h_mean=cut.h.mean(),gross=gm["total"],net=nm["total"],
                annual=gm["annual"],vol=gm["vol"],mdd=gm["mdd"],net_vol=nm["vol"],net_mdd=nm["mdd"],
                beta=np.cov(cut.gross,market)[0,1]/market.var(ddof=1),direction_pnl=dc,relative_pnl=rc))
    return pd.DataFrame(rows)


def comparison_figures(d):
    def draw(ax,g,text):
        ix=pd.DatetimeIndex(g.date)
        ax.plot(ix,g.gross_nav,label=text,lw=1)
        ax.axvline(pd.Timestamp("2024-12-31"),color="grey",lw=.35)
    fig,ax=plt.subplots(figsize=(10,4.8))
    for (a,b),g in d[d.experiment==2].groupby(["a","b"]): draw(ax,g,label(a,b))
    ax.set(title="六个多空方向的1∶1模拟毛净值",ylabel="净值");ax.legend(ncol=3,fontsize=8)
    plot_done(fig,"e2_directions")
    fig,axes=plt.subplots(3,2,figsize=(12,8),sharex=True)
    for ax,((a,b),g) in zip(axes.flat,d[d.experiment==3].groupby(["a","b"])):
        for m,s in g[g.method.isin(METHOD)].groupby("method"): draw(ax,s,METHOD[m])
        ax.set_title(label(a,b))
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle("相同持有期的五种对冲方法 · 毛净值（竖线后为测试期）")
    plot_done(fig,"e3_methods")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,((a,b),g) in zip(axes,d[d.experiment==4].groupby(["a","b"])):
        for m,s in g.groupby("method"): draw(ax,s,METHOD[m])
        ax.set_title(label(a,b));ax.legend(fontsize=7,ncol=2)
    fig.suptitle("同指数与跨指数多空 · 毛净值（竖线后为测试期）")
    plot_done(fig,"e4_comparison")


def estimate_difference(y,x,lag,pos):
    model=sm.OLS(y,x).fit(cov_type="HAC",cov_kwds={"maxlags":lag,"use_correction":True})
    lo,hi=np.asarray(model.conf_int())[pos]
    return dict(estimate=model.params[pos],se=model.bse[pos],p=model.pvalues[pos],lo=lo,hi=hi,lag=lag)


def main():
    p=observations();r=p.pct_change().iloc[1:]
    days=p.index.to_series().diff().iloc[1:].dt.days.to_numpy(float)
    entries=entry_test(p,r,days)
    records=[]
    for a,b in permutations((S,H,C),2):
        records.append(simulate(r,days,a,b,"one",1.,2))
        for method,h in hedge_ratios(r[a],r[b]).items():
            records.append(simulate(r,days,a,b,method,h,3))
            if method in ("rolling","ewma","vol"):
                for freq,period in [("W","W-FRI"),("M","M"),("Q","Q")]:
                    stamps=r.index.to_period(period)
                    rebalance=np.r_[True,stamps[1:]!=stamps[:-1]]
                    ratio=h.iloc[np.maximum.accumulate(np.where(rebalance,np.arange(len(r)),0))].to_numpy()
                    records.append(simulate(r,days,a,b,method+"_"+freq,pd.Series(ratio,index=r.index),3))
    for b in (H,U):
        methods=dict(hedge_ratios(r[H],r[b]),half=.5,double=2.)
        for method,h in methods.items():
            records.append(simulate(r,days,H,b,method,h,4))
    daily=pd.concat(records,ignore_index=True)
    daily.to_csv(OUT/"e2_e4_daily.csv.gz",index=False,float_format="%.12g",compression={"method":"gzip","mtime":0})
    summary=stats(daily,r);save(summary,"e2_e4_statistics")
    # Freeze direction at the penultimate 2024 close, allowing execution at the last close.
    rank=daily[(daily.experiment==2)&daily.date.between("2018","2024-12-30")].groupby(["a","b"]).net.agg(lambda x:(1+x).prod()-1)
    selected=[]
    for a,b in [(S,H),(S,C),(H,C)]:
        long,short=max([(a,b),(b,a)],key=rank.get)
        rows=summary[(summary.experiment==2)&(summary.period=="测试")]
        get=lambda i,j: rows[(rows.a==i)&(rows.b==j)].net.iloc[0]
        selected.append(dict(a=long,b=short,selection_net=rank[long,short],opposite_validation=rank[short,long],
                             test_net=get(long,short),opposite_test=get(short,long)))
    chosen=pd.DataFrame(selected);save(chosen,"e2_direction_selection")
    # An investor acting on the frozen ranking starts flat at the final 2024 close.
    # The main phase table instead measures the subperiod of continuous campaigns.
    fresh=[]
    test_returns=r.loc["2025":]
    for a,b in permutations((S,H,C),2):
        n=len(test_returns)
        acc=account(test_returns[a].to_numpy(),test_returns[b].to_numpy(),
                    np.full(n,.5),np.full(n,-.5),days[-n:],np.ones(n,bool),.0005,.02)
        fresh.append(pd.DataFrame(dict(date=test_returns.index,a=a,b=b,net=acc[:,0],net_nav=acc[:,1])))
    deployed=pd.concat(fresh,ignore_index=True)
    save(deployed,"e2_fresh_test_paths")
    final=deployed.groupby(["a","b"]).net_nav.last()-1
    hindsight=[]
    for row in chosen.itertuples():
        held=(row.a,row.b)
        opposite=(row.b,row.a)
        winner=max((held,opposite),key=final.get)
        hindsight.append(dict(long_at_selection=row.a,short_at_selection=row.b,
                              long_in_hindsight=winner[0],short_in_hindsight=winner[1],
                              selected_net=final[held],reversed_net=final[opposite],
                              hindsight_net=final[winner],hindsight_gap=final[winner]-final[held]))
    save(pd.DataFrame(hindsight),"e2_hindsight_comparison")
    fig,ax=plt.subplots(figsize=(9,4.8))
    for (a,b),g in deployed.groupby(["a","b"]):
        ax.plot([p.index[p.index<g.date.iloc[0]][-1],*g.date],[1.,*g.net_nav],label=label(a,b),lw=1)
    ax.set(ylabel="成本后净值（期初为1）",xlabel="日期",title="测试期从空仓建仓的六组策略");ax.legend(ncol=3,fontsize=8)
    plot_done(fig,"e2_fresh_test")
    design=np.column_stack([np.ones(len(entries)),entries.entry.dt.year.ge(2021)])
    timing=pd.DataFrame([estimate_difference(entries.net.to_numpy(),design,l,1) for l in (251,504)])
    save(timing,"e1_start_period_contrast")
    paired=[]
    for row in chosen.itertuples():
        sample=daily[(daily.experiment==2)&(daily.date.dt.year>=2025)]
        selected_returns=sample[(sample.a==row.a)&(sample.b==row.b)].net.to_numpy()
        reversed_returns=sample[(sample.b==row.a)&(sample.a==row.b)].net.to_numpy()
        delta=(selected_returns-reversed_returns)*252
        paired.append(dict(a=row.a,b=row.b,**estimate_difference(delta,np.ones((len(delta),1)),5,0)))
    paired=pd.DataFrame(paired);paired["holm_p"]=multipletests(paired.p,method="holm")[1]
    save(paired,"e2_paired_tests")
    e3=summary[(summary.experiment==3)&(summary.period=="测试")&summary.method.isin(METHOD)].copy()
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    for axis,column,title in [(ax,"beta","对沪深300的实现 Beta"),(bx,"vol","年化波动率")]:
        vals=e3.pivot(index=["a","b"],columns="method",values=column)[["one","static","rolling","ewma","vol"]]
        vals.index=[label(*k) for k in vals.index]
        vals.rename(columns=METHOD).plot.bar(ax=axis,rot=25,legend=axis==ax)
        axis.set_title(title);axis.set_xlabel("多头 / 空头")
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("最终测试期的市场暴露与风险");plot_done(fig,"e3_beta_risk")
    comparison_figures(daily)


if __name__=="__main__":
    main()
