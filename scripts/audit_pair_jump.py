"""Audit the September 2024 discontinuity against archived independent providers."""
from pathlib import Path
import inspect
import hashlib
import akshare as ak
from run_pair_search import inputs, paths
from optimize_hedge_ratio import account, metrics
from research_common import ROOT, REPORTS, FIGURES, csv, np, pd, plt


def main():
    p,r,_,(_,test_start)=inputs()
    cache=ROOT/"outputs/inputs/pair_jump_audit"
    checks=[]; independent={}
    for source in ["CSI","Sina","Tencent"]:
        for code in ["000001","000132"]:
            file=cache/f"{source}_{code}.csv"
            raw=pd.read_csv(file)
            close=raw.iloc[:,9] if source=="CSI" else raw["close"]
            series=pd.Series(close.to_numpy(),index=pd.to_datetime(raw.iloc[:,0]))
            assert series.index.is_unique
            common=p.loc["2024-07-01":"2024-10-31"].index.intersection(series.index)
            difference=series.loc[common]-p.loc[common,code]
            checks.append(dict(source=source,code=code,n=len(common),max_abs_points=difference.abs().max(),
                               start=common[0],end=common[-1],sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            assert len(common)==82 and difference.abs().max()<.006
            independent[source,code]=series
    csv(pd.DataFrame(checks),"pair_jump_source_checks")
    z=paths("000001","000132")
    j=list(z["ids"]).index("ewma_0.94_M")
    w,v=z["w"][j],z["v"][j]
    ret=z["ret"][j]
    period=r.loc["2024-09-24":"2024-09-30"].index
    sep=r.index.get_loc("2024-09-02")
    cutoff=r.index[sep-2]
    # Independent finite weighted-sum reconstruction of adjust=False EWMA slope.
    past=r.loc[:cutoff,["000001","000132"]].to_numpy()
    n=len(past)
    weights=.06*.94**np.arange(n-1,-1,-1,dtype=float)
    weights[0]=.94**(n-1)
    average=weights@past
    centered=past-average
    h=(weights*centered[:,0])@centered[:,1]/((weights*centered[:,1])@centered[:,1])
    np.testing.assert_allclose(h,-v[sep]/w[sep],atol=1e-12)
    # Verify official August inputs are sufficient to reproduce the September estimate.
    official=pd.concat({c:independent["CSI",c] for c in ["000001","000132"]},axis=1)
    official_returns=official.pct_change(fill_method=None).dropna()
    np.testing.assert_allclose(official_returns.loc["2024-07-02":cutoff],r.loc["2024-07-02":cutoff,["000001","000132"]],atol=1e-12)
    # Independently mark the two signed cash positions then rebalance each day.
    equity=1.; rebuilt=[]
    for k,(a,b) in enumerate(r[["000001","000132"]].to_numpy()):
        long,short=w[k]*equity,v[k]*equity
        equity+=long*a+short*b
        rebuilt.append(equity)
    np.testing.assert_allclose(rebuilt,(1+ret).cumprod(),rtol=1e-12)
    daily=pd.DataFrame(dict(close_sse=p["000001"],close_sse100=p["000132"],
                          r_sse=r["000001"],r_sse100=r["000132"]),index=p.index)
    daily["long_weight"]=pd.Series(w,index=r.index)
    daily["short_weight"]=pd.Series(v,index=r.index)
    daily["net_weight"]=daily.long_weight+daily.short_weight
    daily["h"]=-daily.short_weight/daily.long_weight
    daily["long_contribution"]=daily.long_weight*daily.r_sse
    daily["short_contribution"]=daily.short_weight*daily.r_sse100
    daily["strategy_return"]=daily.long_contribution+daily.short_contribution
    daily["full_nav"]=pd.Series(np.r_[1,rebuilt],index=p.index)
    daily["direction_contribution"]=daily.net_weight*daily.r_sse
    daily["relative_contribution"]=-daily.short_weight*(daily.r_sse-daily.r_sse100)
    np.testing.assert_allclose(daily.loc[r.index,"direction_contribution"]+daily.loc[r.index,"relative_contribution"],ret,atol=1e-12)
    csv(daily.loc["2024-08-26":"2024-10-15"],"pair_jump_daily_audit",index=True)
    d=daily.loc[period]
    jump=(1+d.strategy_return).prod()-1
    previous=(1+d.strategy_return).cumprod().shift(fill_value=1)
    direction=(previous*d.direction_contribution).sum()
    relative=(previous*d.relative_contribution).sum()
    counter=[]
    for spec in ["ewma_0.94_D","ewma_0.94_W","ewma_0.94_M","ewma_0.94_Q","one"]:
        row=list(z["ids"]).index(spec)
        s=pd.Series(z["ret"][row],index=r.index)
        counter.append(dict(spec=spec,event_return=metrics(s.loc[period])["total"],
                            test_return=metrics(s.iloc[test_start:])["total"],
                            test_excluding_event=metrics(s.iloc[test_start:][~s.iloc[test_start:].index.isin(period)])["total"]))
    csv(pd.DataFrame(counter),"pair_jump_counterfactuals")
    cost=account(r["000001"].to_numpy(),r["000132"].to_numpy(),w,v,
                 p.index.to_series().diff().dt.days.loc[r.index].to_numpy(float),
                 np.ones(len(r),bool),.0005,.02)
    old=ROOT/"outputs/tables/pair_champion_full_history.csv"
    if old.exists():
        saved=pd.read_csv(old)
        np.testing.assert_allclose(saved.gross_nav,rebuilt,atol=1e-9)
        np.testing.assert_allclose(saved.scenario_net_nav,cost[:,1],atol=1e-9)
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False})
    fig,axes=plt.subplots(2,1,figsize=(11,7.4),layout="constrained")
    dates=p.loc["2024-09-20":"2024-10-11"].index
    for title,values in [("上证指数",p["000001"]),("上证100",p["000132"]),("原策略毛净值",daily.full_nav)]:
        s=values.loc[dates];axes[0].plot(dates,s/s.iloc[0],marker="o",ms=3,label=title)
    axes[0].axvspan(pd.Timestamp("2024-09-24"),pd.Timestamp("2024-09-30"),color="#eed6ac",alpha=.4)
    axes[0].set_title("放大直线拉升所在的交易日 · 2024年9月24—30日",loc="left")
    axes[0].set_ylabel("9月20日收盘 = 1");axes[0].legend()
    x=np.arange(len(period))
    axes[1].bar(x-.2,d.long_contribution*100,width=.2,label="多头贡献",color="#26786a")
    axes[1].bar(x,d.short_contribution*100,width=.2,label="空头贡献",color="#ad593c")
    axes[1].bar(x+.2,d.strategy_return*100,width=.2,label="合计收益",color="#597ba0")
    axes[1].set_xticks(x,[date.strftime("%m-%d") for date in period])
    axes[1].set_ylabel("当日收益贡献 / 百分点")
    axes[1].set_title("持仓固定在76.65%多头、23.35%空头 · 五个交易日复利 +9.83%",loc="left")
    axes[1].legend(ncol=3)
    fig.savefig(FIGURES/"pair_jump_audit.png",dpi=160,facecolor="white");plt.close(fig)
    lines=["# 上证指数／上证100在2024年9月的净值跃升审计","",
           f"原曲线的直线拉升对应2024-09-24至2024-09-30五个收益日，累计模拟毛收益{jump:.4%}。原始点位、收益计算、资金复利和冻结持仓可重建，未发现这段由日期错位、净值重置或拼接产生的异常。行情已经指数公司及两家独立供应商核对。该盈利高度依赖当时的净多头方向暴露。",
           "","![事件放大与逐日贡献](../outputs/figures/pair_jump_audit.png)",
           "","## 逐日验算","",
           "|日期|上证收盘|上证100收盘|上证涨跌|上证100涨跌|多头贡献|空头贡献|组合涨跌|",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for date,q in d.iterrows():
        lines.append(f"|{date:%Y-%m-%d}|{q.close_sse:.2f}|{q.close_sse100:.2f}|{q.r_sse:.4%}|{q.r_sse100:.4%}|{q.long_contribution:.4%}|{q.short_contribution:.4%}|{q.strategy_return:.4%}|")
    lines+=["",f"事件前收盘到末日，上证涨{(1+d.r_sse).prod()-1:.2%}，上证100涨{(1+d.r_sse100).prod()-1:.2%}。9月30日组合收益为 0.766450 × 8.0637% − 0.233550 × 10.8575% = 3.6447%。",
            "","## 持仓为何偏多","",
            f"9月2日为9月首个收益日，当月参数只使用截至{cutoff:%Y-%m-%d}的日收益，计划在8月30日收盘执行。用全历史有限指数权重手工重建，h={h:.9f}，与保存头寸一致。9月全月冻结比例，净多头{w[sep]+v[sep]:.4%}。8月28日估计h约0.437，8月29日上证跌0.50%、上证100涨2.05%，短期协方差估计降至0.305；8月30日当日收益不参与9月的冻结估计。",
            "","10月首个收益日是10月8日，代码在9月30日收盘换至新比例，信息截止9月27日。重新定权只改变头寸，净值不因定权跳变。每日按净值恢复既定权重仍会产生交易。",
            "","## 方向暴露与配对差额","",
            "使用恒等式 r_p = (w_A + w_B) × r_A + (−w_B) × (r_A − r_B)，将第一项解释为净多头按上证收益获得的贡献，第二项为等金额配对部分。按事件期初资本1元，逐日按此前净值加权，得出完全相加的盈亏。",
            f"净多头部分贡献{direction:.4%}，相对收益部分贡献{relative:.4%}，合计{jump:.4%}。这是一种会计分解，基准取上证指数；净名义敞口不等同于市场Beta。该窗口的行情变化令历史估计的对冲比例未能消除市场方向。",
            "","## 频率与极端事件依赖","",
            "|保持其他规则的对照|五日毛收益|全测试期毛收益|剔除五日后毛收益|",
            "|---|---:|---:|---:|"]
    for q in counter:
        lines.append(f"|{q['spec']}|{q['event_return']:.2%}|{q['test_return']:.2%}|{q['test_excluding_event']:.2%}|")
    lines+=["","剔除日仅用于收益集中度诊断，各日比例采用原路径，假设当日收益归零，不代表提前规避或重新优化。所有频率参数来自原始已搜索网格。",
            "","## 独立来源与复算","",
            "|来源|指数|核对日期数|最大点位差|","|---|---|---:|---:|"]
    for q in checks:lines.append(f"|{q['source']}|{q['code']}|{q['n']}|{q['max_abs_points']:.4f}|")
    lines+=["","逐日比较范围2024-07-01至2024-10-31，包含参数估计末端和拉升窗口。CSI官方和腾讯与库内点位一致，新浪高精度点位最大相差0.005点。新浪与腾讯是独立供应商，其底层指数仍来自相同官方发布体系。",
            "","原始响应的标准化快照和SHA-256见 outputs/inputs/pair_jump_audit 与 [来源核对表](../outputs/tables/pair_jump_source_checks.csv)。[逐日贡献](../outputs/tables/pair_jump_daily_audit.csv) · [频率对照](../outputs/tables/pair_jump_counterfactuals.csv)。",
            "","在项目根目录运行 python scripts/audit_pair_jump.py，使用已获取的独立来源快照。重建了全2883个收益日的毛净值，并与原图保存的成本净值核对。",
            "","本次审计支持这段模拟收益可由已核对的历史行情与既定规则产生。该收益的实盘可实现性仍依赖上证100空头工具、成交价格与借券条件。"]
    (REPORTS/"pair_jump_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("PASS",jump,direction,relative,h)


if __name__=="__main__":
    main()
