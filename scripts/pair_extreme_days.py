"""Measure the sensitivity of price-index rankings to common-date exclusions."""
from itertools import combinations
from math import ceil

from matplotlib.ticker import PercentFormatter

from research_common import csv, figure, market, np, pd, plt


def exclusions(simple, spread):
    count = len(simple)
    market_move = simple["000985"].abs()
    yield "all", simple.index[:0]
    for fraction in (.005, .01, .05):
        yield f"market_{fraction:g}", market_move.nlargest(ceil(count*fraction)).index
    for size in (2, 5, 10):
        yield f"spread_{size}", spread.abs().nlargest(size).index
    yield "spread_0.01", spread.abs().nlargest(ceil(count*.01)).index
    yield "2024_sep_24_30", simple.loc["2024-09-24":"2024-09-30"].index


def calculate(prices, returns):
    rows, dates = [], []
    for sample, simple in [("full", returns), ("from_2021", returns.loc["2021-01-05":])]:
        logs = np.log1p(simple)
        for a, b in combinations(sorted(simple), 2):
            pair = simple[[a,b]]
            spread = logs[a]-logs[b]
            for rule, removed in exclusions(simple,spread):
                kept = pair.drop(index=removed)
                kept_log = logs.loc[kept.index,[a,b]].sum()
                log_gap = kept_log[a]-kept_log[b]
                deleted = spread.loc[removed].sum()
                np.testing.assert_allclose(log_gap + deleted, spread.sum(), atol=1e-12)
                if rule == "all":
                    base = prices.index[prices.index.get_loc(kept.index[0])-1]
                    np.testing.assert_allclose(np.exp(kept_log), prices.loc[kept.index[-1],[a,b]] /
                                               prices.loc[base,[a,b]], atol=1e-12)
                rows.append(dict(sample=sample,a=a,b=b,rule=rule,sample_days=len(pair),
                                 kept_days=len(kept),removed_days=len(removed),
                                 pearson=kept[a].corr(kept[b]),spearman=kept[a].corr(kept[b],method="spearman"),
                                 a_total=np.expm1(kept_log[a]),b_total=np.expm1(kept_log[b]),
                                 relative=np.expm1(log_gap),kept_log_gap=log_gap,removed_log_gap=deleted,
                                 ranking_reversed=np.sign(log_gap)*np.sign(spread.sum())<0))
                for date in removed:
                    dates.append(dict(sample=sample,a=a,b=b,rule=rule,date=date,
                                      a_return=pair.at[date,a],b_return=pair.at[date,b],
                                      market_return=simple.at[date,"000985"],relative_log=spread.at[date]))
    return pd.DataFrame(rows), pd.DataFrame(dates)


def plot(prices, returns, summary):
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False})
    plt.rcParams["figure.dpi"] = 160
    fig, axs = plt.subplots(2,2,figsize=(11,6.2),layout="constrained")
    for col,(b,name) in enumerate([("000132","上证100"),("000300","沪深300")]):
        spread = np.log1p(returns["000001"])-np.log1p(returns[b])
        masks = dict(exclusions(returns,spread))
        ax = axs[0,col]
        for rule, line, color in [("all","保留全部日","#16786f"),("market_0.01","剔除市场极端29日","#a86522"),
                                  ("spread_0.01","剔除配对分化最大29日","#6554a2")]:
            adjusted = spread.copy()
            adjusted.loc[masks[rule]]=0
            relative = pd.Series(np.expm1(np.r_[0,adjusted.cumsum()]),index=prices.index)
            ax.plot(relative,color=color,lw=1.35,label=f"{line} {relative.iloc[-1]:+.2%}")
        ax.axhline(0,color="#71827f",lw=.8)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(f"上证指数相对{name}的累计表现",loc="left")
        if col==0:
            ax.set_ylabel("期末净值之比减1")
        ax.legend(fontsize=8)
        ax=axs[1,col]
        rows=summary.query("sample=='full' and a=='000001' and b==@b").set_index("rule")
        choices=["all","market_0.005","market_0.01","market_0.05","spread_0.01"]
        values=rows.loc[choices,"pearson"]
        ax.barh(np.arange(len(values)),values,color=["#16786f","#a86522","#a86522","#a86522","#6554a2"])
        ax.set_yticks(np.arange(len(values)),["全样本","市场最极端0.5%","市场最极端1%","市场最极端5%","配对分化最大1%"])
        ax.invert_yaxis()
        ax.set_xlim(0,1.07)
        for k,value in enumerate(values):
            ax.text(value+.012,k,f"{value:.3f}",va="center",fontsize=9)
        ax.set_xlabel("保留日期的日收益 Pearson 相关系数")
    fig.suptitle("日收益仍高度联动，长期领先者却可换位",fontsize=16,weight="bold")
    figure(fig,"pair_extreme_days")


if __name__=="__main__":
    prices,returns,_=market()
    summary,dates=calculate(prices,returns)
    csv(summary,"pair_extreme_days_summary")
    csv(dates,"pair_extreme_days_exclusions")
    plot(prices,returns,summary)
