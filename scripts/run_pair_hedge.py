"""Price-index spread portfolios. Run from the project root with Python."""
import base64
from itertools import permutations

from research_common import FIGURES, REPORTS, TARGETS, csv, figure, market, np, pd, plt

CN = dict(zip(TARGETS, ["国证1000", "中证A100", "上证100", "沪深300"]))
FEE = .001
BORROW = .02


def rank_pairs(close):
    levels = close[TARGETS].div(close[TARGETS].iloc[0])
    rank = levels.iloc[-1].rank(ascending=False, method="min")
    rows = []
    for long, short in permutations(TARGETS, 2):
        above = levels[long].iloc[1:] > levels[short].iloc[1:]
        ahead = levels[long].iloc[-1] > levels[short].iloc[-1]
        rows.append(dict(long=long, short=short, short_rank=rank[short], n=len(above),
                         above_days=above.sum(), coverage=above.mean(), long_return=levels[long].iloc[-1]-1,
                         short_return=levels[short].iloc[-1]-1, end_advantage=levels[long].iloc[-1]-levels[short].iloc[-1],
                         eligible=(rank[short] in (2, 3) and ahead and above.mean()>.5)))
    return pd.DataFrame(rows).sort_values(["eligible","coverage","end_advantage"], ascending=False)


def simulate(prices, mode, fee, borrow):
    """Positions are signed market values at closing prices, expressed per initial NAV."""
    wealth = 1.
    position = np.zeros(2)
    last = prices.index[0]
    prev_price = prices.iloc[0].to_numpy()
    rows = []
    direction = np.array([1., -1.])
    for i, (day, price) in enumerate(prices.iterrows()):
        change = price.to_numpy() / prev_price - 1
        pnl = position * change
        borrow_cost = -position[1] * borrow * (day-last).days / 365
        before = wealth + pnl.sum() - borrow_cost
        marked = position * (1+change)
        if day == prices.index[-1]:
            target = np.zeros(2)
        elif i == 0 or mode == "daily":
            # Solve for positions at 100% of NAV after paying the rebalancing commission.
            size = before
            for _ in range(20):
                updated = before - fee*np.abs(direction*size-marked).sum()
                if abs(updated-size)<1e-14:
                    size = updated
                    break
                size=updated
            target = direction * size
        else:
            target = marked
        traded = np.abs(target - marked).sum()
        cost = fee*traded
        wealth = before - cost
        if wealth <= 0:
            raise ValueError("NAV exhausted; account requires a liquidation rule")
        rows.append(dict(date=day,nav=wealth,long_pnl=pnl[0],short_pnl=pnl[1],
                         borrow_cost=borrow_cost,trade_cost=cost,traded=traded))
        position, prev_price, last = target, price.to_numpy(), day
    return pd.DataFrame(rows).set_index("date")


def describe(path):
    nav = path.nav
    returns = nav.pct_change().iloc[1:].copy()
    # Attribute entry fees to the first holding period.
    returns.iloc[0] = nav.iloc[1]-1
    years=(nav.index[-1]-nav.index[0]).days/365.2425
    capital=np.r_[1.,nav.to_numpy()]
    return dict(start=nav.index[0].date(),end=nav.index[-1].date(),n=len(returns),terminal=nav.iloc[-1],
                total=nav.iloc[-1]-1,cagr=nav.iloc[-1]**(1/years)-1,vol=returns.std()*np.sqrt(252),
                mdd=(capital/np.maximum.accumulate(capital)-1).min(),win_rate=(returns>0).mean(),
                long_pnl=path.long_pnl.sum(),short_pnl=path.short_pnl.sum(),
                borrow_cost=path.borrow_cost.sum(),trade_cost=path.trade_cost.sum())


def percent(x):
    return f"{x:.2%}"


def main():
    prices,_,_=market()
    cutoff = pd.Timestamp("2020-12-31")
    entry = prices.index[prices.index>cutoff][0]
    settings=[("lookback","全期回看",prices,prices),("holdout","分期检验",prices.loc[:cutoff],prices.loc[entry:])]
    candidates, daily, summary=[],[],[]
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    fig,axes=plt.subplots(2,2,figsize=(13,7.8),layout="constrained")
    for col,(spec,label,train,test) in enumerate(settings):
        ranked=rank_pairs(train)
        candidates.append(ranked.assign(sample=spec,selection_start=train.index[0],selection_end=train.index[-1]))
        selected=ranked.loc[ranked.eligible].iloc[0]
        long, short=selected.long,selected.short
        legs=test[[long,short]]
        levels=legs/legs.iloc[0]
        top,lower=axes[0,col],axes[1,col]
        top.plot(levels[long],color="#175981",label=f"多头标的 · {CN[long]}",lw=1.2)
        top.plot(levels[short],color="#af6c28",label=f"空头标的 · {CN[short]}",lw=1.2)
        top.set_title(f"{label}    {legs.index[0]:%Y.%m}—{legs.index[-1]:%Y.%m}",loc="left")
        top.set_ylabel("指数点位（建仓日 = 1）")
        top.legend()
        coverage=(levels[long].iloc[1:]>levels[short].iloc[1:]).mean()
        for mode in ("fixed","daily"):
            for fee,borrow in [(0,0),(FEE,0),(FEE,.01),(FEE,BORROW),(FEE,.04)]:
                p=simulate(legs,mode,fee,borrow)
                description=describe(p)
                args=dict(sample=spec,mode=mode,long=long,short=short,fee=fee,borrow=borrow)
                summary.append(dict(**args,coverage_selection=selected.coverage,coverage_holding=coverage,**description))
                if fee==0 or borrow==BORROW:
                    daily.append(p.assign(**args,long_level=levels[long],short_level=levels[short]).reset_index())
                if (mode=="fixed" and fee==0) or borrow==BORROW:
                    title=("固定份额 · 毛" if fee==0 else ("固定份额 · 净" if mode=="fixed" else "每日等额 · 净"))
                    lower.plot(p.nav,label=title,lw=1.1,color={"固定份额 · 毛":"#7d8791","固定份额 · 净":"#16695c","每日等额 · 净":"#704f8c"}[title],
                               linestyle="--" if fee==0 else "-")
                if fee==0 and mode=="fixed":
                    np.testing.assert_allclose(p.nav,1+levels[long]-levels[short],atol=1e-12)
                if fee==0 and mode=="daily":
                    np.testing.assert_allclose(p.nav.iloc[1:],(1+legs.pct_change().iloc[1:]@np.array([1.,-1.])).cumprod(),atol=1e-12)
                np.testing.assert_allclose(description["total"],description["long_pnl"]+description["short_pnl"]
                                           -description["trade_cost"]-description["borrow_cost"],atol=1e-12)
        lower.axhline(1,color="gray",lw=.7)
        lower.set_ylabel("多空组合净值（初始资本 = 1）")
        lower.legend(fontsize=9)
    figure(fig,"pair_hedge_curves")
    scores=pd.DataFrame(summary)
    csv(scores,"pair_hedge_results")
    csv(pd.concat(candidates),"pair_hedge_candidates")
    csv(pd.concat(daily),"pair_hedge_daily")
    write_report(scores)


def write_report(results):
    # Produce a portable report with its chart embedded (no web dependencies).
    content=[]
    def add(s): content.append(s)
    add("<h1>A股宽基指数多空配对收益曲线</h1>")
    add("<p>价格指数实验｜数据截至 2026 年 11 月 17 日｜初始本金 100 元</p>")
    selected=results[(results.fee==FEE)&(results.borrow==BORROW)&(results['mode']=='fixed')]
    for row in selected.itertuples():
        title="全期回看" if row.sample=="lookback" else "分期检验"
        add(f"<p><strong>{title}</strong>　多 {CN[row.long]}（{row.long}），空 {CN[row.short]}（{row.short}）。选对期曲线领先占比 {row.coverage_selection:.2%}，持有期占比 {row.coverage_holding:.2%}。固定份额持有并平仓后，100 元剩余 <strong>{row.terminal*100:.2f} 元</strong>。</p>")
    gross=results[(results.fee==0)&(results["mode"]=="fixed")].set_index("sample")
    add(f"<p>全期回看中，两腿终点只拉开了相当于初始本金 {gross.at['lookback','total']:.2%} 的差额，覆盖不了持仓费用。分期检验中，锁定方向后连毛收益也为 {gross.at['holdout','total']:.2%}，前段领先关系在后段发生反转。</p>")
    image=base64.b64encode((FIGURES/"pair_hedge_curves.png").read_bytes()).decode()
    add(f'<img alt="两组多空配对的标的走势与组合净值" src="data:image/png;base64,{image}">')
    add("<h2>如何选出两只指数</h2><p>研究池沿用前文四条目标宽基。上证综指、中证全指承担基准角色。每个选对区间都从同日收盘归一化为 1。把期末价格收益排名第 2 或第 3 的指数定义为“收益一般”；多头须在期末跑赢空头，且曲线严格高于空头的交易日超过一半。对四只指数的 12 种多空方向排序，优先选择领先天数占比最高的合格组合，同分取期末差额较大的组合。占比不计起始日。</p>")
    add("<p>全期回看使用整段历史选择配对，只展示事后挑选的结果。分期检验使用 2015-01-05 至 2020-12-31 选对，锁定后在 2021-01-04 收盘建仓，首笔持有收益发生在 1 月 5 日，两条标的曲线从建仓日重新归一化。分界点是本次回顾性实验的设定，项目先前已分析过全期行情。</p>")
    add("<h2>差额如何变成净值</h2><p>主方案以 100 元资本建立大约 100 元多头和 100 元空头，总名义敞口约 200 元。固定两边份额，零费用时组合净值恰为 1＋多头归一化点位－空头归一化点位。补充方案每日收盘把两腿调回各为净资产的 100%，次日毛收益等于多头日收益减空头日收益，并逐日复利。两种持仓的风险暴露随时间不同。</p>")
    add("<p>主费用情景按每笔单边成交额收 10 个基点，含双腿开仓、调仓、平仓；空头按前收市值和实际日历天数支付年费 2%，现金及卖空所得利息设为 0。费用从净资产扣除。价格指数未计多头收息和空头补息，也未模拟保证金追缴、产品跟踪、期货基差或展期。此处假定六位代码对应的指数收益可按点位买卖，空头可持续借到。等额配置只控制初始净名义敞口，市场 Beta 可偏离零。</p>")
    add("<h2>组合绩效</h2><table><thead><tr><th>选择方式</th><th>持仓</th><th>口径</th><th>收益日</th><th>累计收益</th><th>年化收益</th><th>年化波动</th><th>最大回撤</th></tr></thead><tbody>")
    for row in results[(results.fee==0)|(results.borrow==BORROW)].itertuples():
        add(f"<tr><td>{'全期回看' if row.sample=='lookback' else '分期检验'}</td><td>{'固定份额' if row.mode=='fixed' else '每日等额'}</td><td>{'毛' if row.fee==0 else '净'}</td><td>{row.n}</td>"+"".join(f"<td>{percent(v)}</td>" for v in [row.total,row.cagr,row.vol,row.mdd])+"</tr>")
    add("</tbody></table><p>年化收益按实际持有年数复合计算，波动率按 252 个交易日年化。最大回撤包含从初始本金到平仓的全路径。</p>")
    add("<h2>每 100 元的固定份额盈亏</h2>")
    for row in selected.itertuples():
        add(f"<p><strong>{'全期回看' if row.sample=='lookback' else '分期检验'}</strong>　多头盈亏 {100*row.long_pnl:+.2f} 元，空头盈亏 {100*row.short_pnl:+.2f} 元，借券费 {100*row.borrow_cost:.2f} 元，交易费 {100*row.trade_cost:.2f} 元。合计 {100*row.total:+.2f} 元。</p>")
    add("<h2>借券费敏感性</h2><table><tr><th>固定份额净收益</th><th>0% 年费</th><th>1% 年费</th><th>2% 年费</th><th>4% 年费</th></tr>")
    for label,spec in [("全期回看","lookback"),("分期检验","holdout")]:
        d=results[(results['sample']==spec)&(results['mode']=='fixed')&(results.fee==FEE)].sort_values("borrow")
        add("<tr><td>"+label+"</td>"+"".join("<td>"+percent(x)+"</td>" for x in d.total)+"</tr>")
    add("</table><p>领先占比描述此前累计走势的位置。固定份额组合的期末毛盈亏取决于两腿从建仓到平仓的涨跌差，持续借券费用会进一步改变投资者实际留下的差额。</p>")
    add("<p>数据取自项目 SQLite 中的东方财富指数日收盘价，来源接口为 AKShare index_zh_a_hist。已有价格及收益覆盖经项目审计，本次对固定份额价差恒等式、每日等额复利恒等式和净盈亏分项逐项复核。</p>")
    css="body{max-width:1080px;margin:40px auto;padding:0 18px;font:16px/1.7 'Microsoft YaHei',sans-serif;color:#19303a}h1,h2{line-height:1.3}h2{margin-top:32px;font-size:21px}img{width:100%}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:6px 8px;text-align:right;border-bottom:1px solid #ddd}th{background:#edf3f2}td:first-child,th:first-child{text-align:left}"
    (REPORTS/"pair_hedge_report.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A股宽基多空收益曲线</title><style>'+css+'</style><main>'+"\n".join(content)+'</main></html>',encoding="utf-8")


if __name__=="__main__":
    main()
