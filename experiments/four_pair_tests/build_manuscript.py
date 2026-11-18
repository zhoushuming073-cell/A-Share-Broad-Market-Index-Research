"""Regenerate the initial editable Markdown manuscript from the verified tables."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
assert json.loads((OUT/"verification.json").read_text())["status"] == "passed"
names = {"000001":"上证综指","000300":"沪深300","000903":"中证100","000132":"上证100"}
methods = {"one":"1∶1","static":"静态OLS","rolling":"滚动OLS","ewma":"EWMA","vol":"波动率比",
           "half":"固定0.5","double":"固定2"}

def read(name):
    return pd.read_csv(OUT/(name+".csv"), dtype={"a":str, "b":str})

def pct(value):
    return f"{100*value:.2f}%".replace("-","−")

def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str,row)) + " |" for row in
                     [headers, ["---"]*len(headers), *rows])

def pair(a,b):
    return names[a]+"／"+names[b]

s = read("e2_e4_statistics")
e1 = read("e1_all_entries")
selection = read("e2_direction_selection")
hindsight = pd.read_csv(OUT/"e2_hindsight_comparison.csv",dtype={col:str for col in
    ("long_at_selection","short_at_selection","long_in_hindsight","short_in_hindsight")})
test = read("e2_paired_tests")
fresh = read("e2_fresh_test_paths").groupby(["a","b"]).net_nav.last()-1
phases = ["训练","验证前段","验证后段","测试","全段"]
e2 = s[s.experiment.eq(2)].pivot(index=["a","b"],columns="period",values="net")
e3 = s[s.experiment.eq(3) & s.period.eq("测试")]
e4 = s[s.experiment.eq(4) & s.period.eq("测试")]
tables = {
    "hindsight":table(["验证期所选多头／空头","测试期事后较优多头／空头","所选净收益","事后较优净收益","差额（百分点）"],
        [[pair(row.long_at_selection,row.short_at_selection),pair(row.long_in_hindsight,row.short_in_hindsight),
          pct(row.selected_net),pct(row.hindsight_net),f"{100*row.hindsight_gap:.2f}"] for row in hindsight.itertuples()]),
    "mean_selected":pct(hindsight.selected_net.mean()),
    "mean_hindsight":pct(hindsight.hindsight_net.mean()),
    "mean_gap":f"{100*hindsight.hindsight_gap.mean():.2f}",
    "entries":table(["口径","平均","中位数","第25百分位","第75百分位","盈利窗口"],[
        [label,pct(v.mean()),pct(v.median()),pct(v.quantile(.25)),pct(v.quantile(.75)),pct((v>0).mean())]
        for label,v in [("毛收益",e1.gross),("成本后",e1.net)]]),
    "phases":table(["多头／空头","2015–17","2018–20","2021–24","2025–26¹","全期"],
        [[pair(a,b),*[pct(row[k]) for k in phases]] for (a,b),row in e2.iterrows()]),
    "selection":table(["历史所选多头／空头","选择期净收益","测试期新开仓","反向新开仓","原始 p"],
        [[pair(row.a,row.b),pct(row.selection_net),pct(fresh[row.a,row.b]),pct(fresh[row.b,row.a]),
          f"{test.iloc[i].p:.3f}"] for i,row in enumerate(selection.itertuples())]),
    "methods":table(["方法","平均净收益","平均年化波动","平均绝对Beta","平均净值最大回撤"],
        [[methods[m],pct(g.net.mean()),pct(g.vol.mean()),f"{g.beta.abs().mean():.3f}",pct(g.net_mdd.mean())]
         for m in list(methods)[:5] for g in [e3[e3.method.eq(m)]]]),
    "freq":table(["更新频率","滚动OLS","EWMA","波动率比"],
        [[f,*[pct(e3.loc[e3.method.eq(m+suffix),"vol"].mean()) for m in ["rolling","ewma","vol"]]]
         for f,suffix in [("每日",""),("每周","_W"),("每月","_M"),("每季","_Q")]]),
    "attribution":table(["空头","方法","方向贡献¹","相对贡献¹","毛收益","净收益","Beta"],
        [[names[row.b],methods[row.method],f"{100*row.direction_pnl:.2f}",f"{100*row.relative_pnl:.2f}",
          pct(row.gross),pct(row.net),f"{row.beta:.3f}"] for row in e4.itertuples()
         if row.b!="000300" or row.method in ("one","half","double")]),
}
text = r"""---
title: A股宽基指数多空对冲策略的收益稳定性、对冲比率敏感性与事后选择偏差研究
subtitle: "Return Stability, Hedge-Ratio Sensitivity, and Ex-Post Selection Bias in Long–Short Broad-Market Index Strategies: Evidence from China's A-Share Market"
date: 2026年11月18日
lang: zh-CN
---

# 摘要

本文以四项历史模拟考察宽基指数之间的相对收益能否由投资者持续取得。利用2015年1月至2026年11月的共同收盘价，固定总名义敞口为本金的一倍，同时记录毛收益及扣除模拟成本后的收益。做多上证综指、做空沪深300，在2,632个建仓日分别持有252个交易日，平均净收益为−1.83%，盈利窗口占29.75%。按验证期净收益在三对指数中分别选择方向，到最终测试期只有一组从空仓建仓后盈利，三组相对反向的优势均未达到统计显著。每日EWMA对冲使六方向的平均年化波动从2.77%降至2.48%，平均净收益仍为负。同指数等额多空的毛收益为零；跨指数等额多空可以保留明显的相对损益以及非零市场Beta。这些结果支持对冲比率具有风险管理作用，尚不足以支持依据历史强弱稳定获利。

**关键词**　宽基指数；相对收益；最小方差对冲；建仓时点；事后选择偏差

# 研究背景、文献与假说

两条指数净值曲线分叉，投资者在哪一端建仓、投入几成资金、能否承担持有成本，共同决定可取得的收益。一条长期领先的曲线可以在所选起点之后跑输。本文从一笔具体的交易出发，将同一个问题分别落实到起点、方向、比率和损益归因四个可复核的实验。

Gatev等[1]研究形成期配对与交易期价差规则，Do与Faff[2]考察简单配对策略的时期差异。Avellaneda与Lee[3]从残差构造统计套利信号，Zhang与Urquhart[4]研究内地与香港市场的配对。本文持续持有给定方向，没有设置价差偏离或回归的进出场阈值。这些研究提供形成期与评价期的设计依据，其收益率不能直接用来评价本文的组合。Ederington[5]以方差降低评价对冲，为比较OLS与等名义策略提供理论参照。Lo与MacKinlay[6]、White[7]和Hansen[8]解释了反复检索历史规则后评价赢家的问题。本文采用时间切分及三项方向检验的多重校正，没有实施White或Hansen的策略族检验。

建仓实验检验两个历史起点组的平均持有收益是否相等。方向实验检验验证期选定方向与反向在测试期的平均日净收益差是否为零。对冲实验考察最小方差估计能否同时降低实现波动和方向选错的损失。同指数实验提供可精确计算的零相对收益对照，检验跨指数损益中还剩多少相对贡献。后两项以会计恒等式和实现风险衡量，不作模型优越性的显著性推断。

# 数据、资金账户与实验设计

输入为仓库基线 fe47059 的 data/a_share_indices.sqlite，使用 index_daily 表中 provider 为 eastmoney 的价格指数收盘价。四条序列为上证综指000001、沪深300 000300、中证100（现称中证A100）000903及上证100 000132。000132仅用于实验四。共同区间为2015-01-05至2026-11-17，含2,884组正收盘价，各序列均无缺值。首日用于建仓或计算基期，产生2,883个收盘到收盘收益。输入快照与原库可逐日核对；本次核验针对仓库数据及账户运算，未逐日比对指数公司行情。代码连续使用同一指数代码，因此跨期结论涵盖样本中实际经历的编制规则变化。

对每单位本金，令 $h_t$ 为第 $t$ 日空头与多头的目标名义金额之比，$r_{L,t}$、$r_{S,t}$ 为两腿价格收益。设目标多头权重为 $w_t$、空头权重为 $v_t$，则

$$w_t=\frac{1}{1+h_t},\qquad v_t=-\frac{h_t}{1+h_t},\qquad g_t=w_t r_{L,t}+v_t r_{S,t}.$$

1∶1时为多0.5元、空0.5元。以1元作多、再以1元作空的日收益公式使用两倍总敞口，不能直接与表中数字比较。每个持仓日开头按目标权重调回一倍总敞口。成交按上一收盘价模拟，动态信号仅用下单收盘日前一个交易日及更早的信息。假设可借券、卖空、按收盘点位成交，现金利率为零。每单位双腿实际成交金额收单边5个基点，空头另按年率2%、实际自然日数除以365计提。费用按扣费后可建仓资本计算；两腿分别记账，不做同标的内部净额结算。没有现金分红、ETF跟踪差、期货基差、滑点或保证金追缴模型。

令 $A_t,B_t$ 为调仓后两个头寸的有符号金额，$Q_t$ 为成交绝对金额之和（末日含平仓），$\Delta d_t$ 为相邻收盘的自然日数。每日记账及复利为

$$V_t=V_{t-1}+A_t r_{L,t}+B_t r_{S,t}-0.0005Q_t-0.02|B_t|\frac{\Delta d_t}{365}.$$

$$u_t=V_t/V_{t-1}-1,\qquad R_{a:b}=\prod_{t=a}^{b}(1+u_t)-1.$$

风险指标使用样本标准差、日度峰值与相对沪深300的毛收益敏感度。$n$ 为阶段交易日数，$m$ 为沪深300日收益，$s$ 为样本标准差，阶段回撤包含初始净值1。

$$R_{\mathrm{ann}}=(1+R_{a:b})^{252/n}-1,\quad \sigma_{\mathrm{ann}}=\sqrt{252}\,s(g),\quad \beta_m=\frac{\mathrm{Cov}(g,m)}{\mathrm{Var}(m)}.$$

$$\mathrm{MDD}=\min_{a\le t\le b}\left(\frac{V_t}{\max_{a-1\le j\le t} V_j}-1\right).$$

实验一每个日期独立开平仓。实验二从2015年起连续持仓；2015–17年为训练期，2018–24年为验证期，最后453个收益日（2025-01-02至2026-11-17）为按时间留出的测试期。实验三、四共同自2018-01-02起计收益，之前的数据用于估计。连续策略在阶段边界按已有持仓结转，只在整条路径起止日计开平仓费。实验二另从2024年最后收盘空仓部署六个方向，计入完整进入与退出成本。整个研究在既有探索之后开展，时间隔离不能消除研究者已经看过部分历史结果的影响。

# 实验一　相同持有期的建仓差异

对每一个具有完整后续252个交易日的收盘日，等额做多上证综指、做空沪深300。每个窗口独立从1元出发，期末清仓。报告的平均数给予每个可选建仓日相同权重。

表：逐日起点的252日收益分布

{{entries}}

成本后最差窗口始于2020-02-03，至2021-02-08亏损9.08%；最佳窗口始于2021-02-10，至2022-03-02盈利8.18%。这两个起点仅相隔约一年，方向、资金与期限完全相同。图中的最高与最低曲线按期末表现回看选取。

![](figures/e1_entry.png){width=95%}

图：各建仓日的持有期收益及事后极值窗口净值

![](figures/e1_distribution.png){width=86%}

图：2,632个完整窗口的毛收益与成本后收益分布

以建仓日是否进入2021年将窗口分组。拟合 $Y_i=\alpha+\delta D_i+e_i$，其中 $Y_i$ 是窗口净收益，$D_i$ 是后期起点指示变量。双侧统计量为 $z=\hat\delta/\mathrm{SE}_{HAC}(\hat\delta)$，采用Newey–West型异方差与自相关一致协方差估计。后期起点均值高出4.78个百分点；滞后251日的95%区间为[2.77, 6.78]个百分点，滞后504日的区间为[2.59, 6.97]。相邻窗口共享251日收益；窗口数不能解释为独立试验次数。分界点来自既有研究关注的时期，结果只能描述阶段差异，无法提供可事前操作的择时信号。

# 实验二　多空方向的事后选择

表：六个等额方向分阶段的累计成本后收益

{{phases}}

¹ 2026年止于11月17日。全期按日复利；各阶段合计应连乘。图中绘制六条连续毛净值，竖线右侧为最终测试期。与成本后数据并读，可以看到在同一对指数上，两个方向都亏损的历史阶段。

![](figures/e2_directions.png){width=95%}

图：包含训练期的六方向毛净值

每对指数仅用2018年首个收益日至2024-12-30的连续净收益选取方向，将12月31日收盘留作执行。表中“反向”使用完全相同的开平仓及资金规则。右栏的双侧检验在连续策略口径上比较同一天的两方向日净收益，以5阶HAC估计均值差的标准误；三次检验用Holm方法校正。

表：验证期选向与最终测试期从空仓部署的结果

{{selection}}

历史所选方向有两组在测试期胜过反向，仅一组取得正净收益。三个校正后 p 值均为1.000。沪深300相对中证100在验证期损失较小，测试期反向损失较小；两个方向都未覆盖成本。把最终测试期收益较高的一侧反标为“正确方向”，会将当时无法获知的信息带入选向。这里的三对指数共享成分与交易日，2/3也不宜外推为预测准确率。

![](figures/e2_fresh_test.png){width=95%}

图：最终测试期从空仓开始的成本后净值

表：同一测试期内事后挑选方向带来的表现差额

{{hindsight}}

对每一对指数，从两条已经实现的测试期净值中选期末较高者，可以量化看完结果再选方向所增加的历史收益。三对指数等权平均，事先选向的净收益为{{mean_selected}}，事后选向为{{mean_hindsight}}，差额为{{mean_gap}}个百分点。差额全部来自沪深300与中证100这一对。即使允许事后选方向，三对中也有两对的两个方向均亏损。该差额取决于本次留出区间和两方向候选集，不能当作事后选择偏差的总体估计。

# 实验三　对冲比率、更新频率与风险

以空头收益解释多头收益，带截距OLS的斜率为协方差除以方差。静态OLS仅使用2015年至2017-12-28的收益，固定用于其后全段。滚动OLS和波动率比均取120个收益日；EWMA用衰减系数0.97、递推加权均值和协方差，至少60个历史值。历史波动率比仅按两腿波动规模匹配，忽略相关系数。令 $s_L,s_S$ 为估计期波动，$\rho$ 为相关系数，$C_\lambda$ 为指数加权中心协方差：

$$h_{\mathrm{OLS}}=\rho\,s_L/s_S,\qquad h_{\mathrm{EWMA}}=C_\lambda(L,S)/C_\lambda(S,S),\qquad h_{\mathrm{vol}}=s_L/s_S.$$

动态比率分别在日、周、月、季的首个收益日采用新值，期间维持该比率。每次使用截至收益日之前第二个收盘的观测。即使比率不变，仍每日恢复目标权重。每个方向包括1个等额设定、1个静态估计和12个动态设定，共84条路径。下表对六个方向的单策略指标取算术平均，没有将这些高度重叠的策略合并成一只基金。

表：测试期六方向的平均表现（比率可变者每日更新）

{{methods}}

![](figures/e3_methods.png){width=98%}

图：六个方向、五种方法在共同可评价区间的毛净值

表：不同比率更新频率下六个方向的平均年化毛波动率

{{freq}}

![](figures/e3_beta_risk.png){width=98%}

图：各方向的实现市场Beta与年化毛波动率

滚动估计与EWMA在测试期降低了平均波动；静态OLS的平均波动略高于等额法。以验证期选中的多沪深300、空中证100为例，每日EWMA令毛波动由1.45%降至1.32%，实现Beta由−0.028变为+0.011，成本后最大回撤从−3.95%收窄至−3.86%，累计净收益却从−2.82%降为−3.54%。最小方差目标关注当期两腿如何共同波动，不包含对未来相对均值的预测。表中的频率差异仅描述本次测试，尚不能确立普遍最优更新周期。

# 实验四　同指数对照与跨指数损益归因

多头均为沪深300。空头依次为沪深300和上证100。除每日更新的五种方法外，加入固定 $h=0.5$ 和 $h=2$，分别获得正、负名义净敞口。给定沪深300为参照，可逐日拆分毛收益并链接至阶段复利收益。$G_{t-1}$ 为阶段起点归一至1的前一日毛净值：

$$d_t=(w_t+v_t)r_{300,t},\qquad q_t=-v_t(r_{300,t}-r_{S,t}),\qquad g_t=d_t+q_t.$$

$$G_b-1=\sum_{t=a}^b G_{t-1}d_t+\sum_{t=a}^b G_{t-1}q_t.$$

表：测试期沪深300多头组合的复利损益归因

{{attribution}}

¹ 方向与相对贡献单位为期初本金的百分点，按未取整值相加等于毛收益。同指数的静态OLS、滚动OLS、EWMA和波动率比均与等额路径重合，仅列等额一行。所有跨指数设定在该测试期均为净亏损。

![](figures/e4_comparison.png){width=98%}

图：同指数与跨指数不同名义比率的毛净值

在同指数对照中，相对项恒为零；等额两腿的毛净值始终为1。测试期净亏2.08%来自两腿独立记账下的借券与交易成本。在跨指数等额组合中，名义方向项也恒为零，而毛亏2.53%来自两指数的相对变化，实现市场Beta仍为−0.059。相对项自身可与市场同向或反向变动，因此这个以沪深300为基准的会计分解并未提取无市场风险的因子收益。跨指数静态OLS的方向贡献为+0.61个百分点、相对贡献为−2.04个百分点，二者共同决定毛亏1.43%。

# 讨论、研究局限与结论

四项实验证据指向一个具体区分。历史选向解决的是哪一条指数会领先；对冲估计解决的是给定方向后，另一条腿应承担多少共同波动。更频繁地调整第二条腿能够改变风险，即使领先方向选错，账户仍会承担损失。2015年起等额做多上证综指、做空沪深300的全期净收益为−15.29%，其中2021–24年单段却盈利7.07%。对这条曲线的评价会随观察起点以及是否扣费而改变。

一种可能解释是，规模和行业权重差异随时期改变相对收益；四项实验只使用指数收盘价，没有用成分及行业数据检验这一机制。指数价格缺少分红以及实际多空工具成本；交易成本与借券费为情景假设。样本外为一次历史留出，且研究动机源于既有回测。六个方向及多个比率的比较不能证明未来最优。同指数组合单列两腿的成本也不能解释为现实中净额结算后必须发生的费用。

综上，在本样本、总敞口与成本口径下，有证据支持宽基多空可以降低部分共同波动；对于历史选向能否在成本后持续赚钱，三对指数给出的证据不足。已有结果适合用于检查建仓日期、样本切分与资金定义如何改变结论。

# 参考文献

[1] Gatev E, Goetzmann W N, Rouwenhorst K G. Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies*, 2006, 19(3), 797–827. [DOI](https://doi.org/10.1093/rfs/hhj020)

[2] Do B, Faff R. Does Simple Pairs Trading Still Work? *Financial Analysts Journal*, 2010, 66(4), 83–95. [DOI](https://doi.org/10.2469/faj.v66.n4.1)

[3] Avellaneda M, Lee J H. Statistical Arbitrage in the US Equities Market. *Quantitative Finance*, 2010, 10(7), 761–782. [DOI](https://doi.org/10.1080/14697680903124632)

[4] Zhang H, Urquhart A. Pairs Trading Across Mainland China and Hong Kong Stock Markets. *International Journal of Finance & Economics*, 2019, 24(2), 698–726. [DOI](https://doi.org/10.1002/ijfe.1687)

[5] Ederington L H. The Hedging Performance of the New Futures Markets. *Journal of Finance*, 1979, 34(1), 157–170. [DOI](https://doi.org/10.1111/j.1540-6261.1979.tb02077.x)

[6] Lo A W, MacKinlay A C. Data-Snooping Biases in Tests of Financial Asset Pricing Models. *Review of Financial Studies*, 1990, 3(3), 431–467. [DOI](https://doi.org/10.1093/rfs/3.3.431)

[7] White H. A Reality Check for Data Snooping. *Econometrica*, 2000, 68(5), 1097–1126. [DOI](https://doi.org/10.1111/1468-0262.00152)

[8] Hansen P R. A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics*, 2005, 23(4), 365–380. [DOI](https://doi.org/10.1198/073500105000000063)

# 附录　数据、图表与复现

在仓库根目录运行 **python experiments/four_pair_tests/run.py**，随后运行同目录的 **verify.py**。脚本只读原始数据库，复用仓库账户及指标函数。完整输入见 data/input_closes.csv。每一笔252日实验的进出日期及结果见 results/e1_all_entries.csv；e1_holding_paths.npz 中 nav 的三个维度依次为建仓日、毛／净口径、持有日（0–252）。其余连续策略逐日权重、比率、成交、费用、收益、归因及双净值见 e2_e4_daily.csv.gz，各阶段统计见 e2_e4_statistics.csv。空仓开始的测试路径见 e2_fresh_test_paths.csv。

独立账户核验从收盘价重算2,632个窗口的逐日净值，并逐日核对另外104条连续策略的228,194条记录。归因加总、同指数零毛收益与样本边界均设有数值断言。核验清单见 results/verification.json。图表可从CSV和run.py重新绘制；figures目录同时留存PNG和可编辑SVG。研究者可直接修订Markdown源稿或Word正文。重新生成Markdown会刷新初始稿，手工修订后应直接从Markdown生成Word。
"""
for key, value in tables.items():
    text = text.replace("{{"+key+"}}",value)
(ROOT/"research_report.md").write_text(text,encoding="utf8")
