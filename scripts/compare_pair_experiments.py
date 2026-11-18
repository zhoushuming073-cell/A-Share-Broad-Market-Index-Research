"""Align the two discussed experiments by date, exposure and methodology."""
from run_pair_search import inputs, paths
from optimize_hedge_ratio import metrics
from research_common import FIGURES, REPORTS, csv, np, pd, plt


def main():
    p,r,_,_=inputs()
    dates=r.loc["2021-01-05":].index
    train=r.loc["2021-01-05":"2024"]
    h=train["000001"].cov(train["000300"])/train["000300"].var()
    old=(r["000001"]-h*r["000300"])/(1+h)
    curves={"上证 / 沪深300 · 静态0.820":old.loc[dates]}
    for b,title in [("000300","上证 / 沪深300 · EWMA"),("000132","上证 / 上证100 · EWMA")]:
        z=paths("000001",b)
        row=list(z["ids"]).index("ewma_0.94_M")
        curves[title]=pd.Series(z["ret"][row],index=r.index).loc[dates]
    data=pd.DataFrame(curves)
    csv(data,"pair_aligned_comparison_daily",index=True)
    rows=[]
    for title,s in curves.items():
        for period,part in [("2021年以来",s),("2025年以来",s.loc["2025":])]:
            rows.append(dict(strategy=title,period=period,**metrics(part)))
    summary=pd.DataFrame(rows)
    csv(summary,"pair_aligned_comparison")
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout="constrained")
    for ax,start,base,label in [(axes[0],"2021-01-05","2021-01-04","2021年初开始"),
                                (axes[1],"2025-01-01","2024-12-31","2025年初开始")]:
        for (title,s),color in zip(curves.items(),["#16776a","#c38825","#6c6299"]):
            piece=s.loc[start:]
            nav=pd.Series(np.r_[1,(1+piece).cumprod()],index=pd.DatetimeIndex([pd.Timestamp(base),*piece.index]))
            ax.plot(nav,color=color,lw=1.4,label=f"{title}（{nav.iloc[-1]-1:+.2%}）")
        ax.axhline(1,color="#777",lw=.7)
        ax.set_title(label+" · 同为1倍总敞口、未扣成本",loc="left")
        ax.legend(fontsize=8);ax.set_ylabel("净值")
    fig.savefig(FIGURES/"pair_aligned_comparison.png",dpi=160,facecolor="white")
    plt.close(fig)
    report="""# 两次配对实验的比较口径

原实验是2021年开始的上证指数/沪深300，静态比例约0.820，每1元净资产做多1元、做空0.820元，总敞口1.820倍。全期累计27.03%。改成1倍总敞口后，累计14.26%。图中的新实验是2015年开始的上证指数/上证100，EWMA每月更新，另画了扣成本曲线。

![同日期、同敞口的比较](../outputs/figures/pair_aligned_comparison.png)

上图静态比例用2021—2024年估计，左图含估计期，右图在估计期之后。两个EWMA曲线用过去可得收益更新比例。配对与参数已经被历史研究查看过，这张图用于解释口径差异。

上证/上证100的图中配置在2015—2022年训练段亏损8.64%、夏普−0.218，未通过训练夏普大于零的筛选条件。它是测试期回看收益最高的配置。真正经过验证选出的配置是沪深300/上证100的60日波动率比。

训练段用于静态Beta估计、动态规则历史回放与盈利门槛筛选。验证段在通过门槛的配置中按夏普选择配对、方向、方法和参数。测试段固定这些选择，动态比例继续按既定算法更新。现有规则没有预测未来相对收益，也没有在预期不利时平仓或切换多空方向。

OLS和EWMA以历史协方差/方差估计对冲比率，目标是减少共同波动。即使估计完全正确，扣除市场部分后的平均收益仍然可能小于零。三段切分检查这类收益能否延续，切分本身不产生收益。

当前搜索只覆盖六条指数和既定规则。结果支持该搜索尚未找到可靠的事前选择方法。对于所有宽基相对价值策略是否可行，现有实验不能给出普遍结论。
"""
    (REPORTS/"pair_experiment_explanation.md").write_text(report,encoding="utf-8")


if __name__=="__main__":
    main()
