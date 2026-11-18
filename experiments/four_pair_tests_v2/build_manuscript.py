"""Revise the original manuscript, retaining its title, question, literature, and bibliography."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
OLD=ROOT.parent/"four_pair_tests"
old=(OLD/"research_report.md").read_text(encoding="utf8")
verification=json.loads((OUT/"verification.json").read_text())
assert verification["status"]=="passed"
def read(name):
    return pd.read_csv(OUT/(name+".csv"),dtype={"a":str,"b":str})

s=read("phase_statistics")
e1=read("e1_summary").set_index(["rule","basis"])
sel=read("e2_selection")
hac=read("e2_hac")
block=read("e1_block_intervals")
windows=read("e1_windows")
e2=s[s.experiment==2].pivot(index=["a","b"],columns="period",values="net")
e3=s[s.experiment==3].copy()
e4=s[(s.experiment==4)&(s.period=="test")]
d3=e3[(e3.period=="test")&(e3.frequency=="D")]
rules={"units":"固定数量","daily":"每日恢复"}
methods={"one":"等额","static":"静态OLS","rolling":"滚动OLS","ewma":"EWMA","vol":"波动率比",
         "half":"固定0.5","double":"固定2"}
periods={"train":"训练期","validation":"验证期","test":"测试期"}
names={"000001":"上证综指","000300":"沪深300","000903":"中证100","000132":"上证100"}
def pair(a,b): return names[a]+"／"+names[b]
def num(v,d=2): return (f"{v:.{d}f}" if abs(v)>=.5/10**d else f"{0:.{d}f}").replace("-","−")
def pct(v,d=2): return num(v*100,d)+"%"
def avg(rule,col):return pct(e1.loc[(rule,"net"),col])
def row3(a,b,m):
    return d3[(d3.a==a)&(d3.b==b)&(d3.method==m)].iloc[0]
parts=[old.split("# 摘要")[0].replace("date: 2026年11月18日","date: 2026年11月18日 · 修订稿")]
tno=fno=0
def add(text):parts.append(text.strip())
def heading(title):add("# "+title)
def table(title,heads,rows):
    global tno
    tno+=1
    add(f"表 {tno}　{title}")
    add("\n".join("| "+" | ".join(str(a) for a in row)+" |" for row in [heads,["---"]*len(heads),*rows]))
def figure(name,title):
    global fno
    fno+=1
    add(f"![](figures/{name}.png)")
    add(f"图 {fno}　{title}")
heading("摘要")
add(f"本文沿用四项宽基指数配对实验，区分不同建仓规则、历史选向、对冲比率与组合损益来源。基于2015年1月至2026年11月的日收盘价，做多上证综指、做空沪深300，在每个可行起点各持有252个交易日，固定数量账户平均净收益为{avg('units','mean')}，每日恢复等额权重账户为{avg('daily','mean')}。三对指数在验证期选定方向后分别新开测试账户，两组跑赢反向，仅一组盈利。三组方向差异的双侧检验在多重校正后均未达到5%显著性。六个方向采用每日EWMA对冲后，测试期平均波动下降，平均累计净收益仍为负。同指数等额多空按净头寸结算时收益、交易和借券费用均为零；等额做多沪深300、做空上证100则承担相对收益和完整的两腿成本。")
add("**关键词**　宽基指数；相对收益；最小方差对冲；建仓时点；事后选择偏差")
heading("研究背景与研究问题")
add(next(p for p in old.split("\n\n") if p.startswith("两条指数净值曲线分叉")))
add("本文依次回答四个问题。第一，相同期限下，进入日期与是否恢复权重如何改变持有收益？第二，选定历史较优方向后，在新账户中能否继续跑赢反向并覆盖费用？第三，估计比率和更新频率对风险与净收益的影响是否一致？第四，当双腿为同一标的或不同标的时，实际支付的成本、共同方向损益和相对损益各有多少？")
heading("相关文献")
add(next(p for p in old.split("\n\n") if p.startswith("Gatev等")))
add("对冲比率的协方差估计对应固定多头单位下的最小残差方差。本文再按总名义敞口归一，长期样本外波动仍需用实际权重检验；对冲后表现良好也不构成价差均值回归的证据。")
heading("数据来源与样本说明")
add("数据来自项目 SQLite 库 index_daily 表，提供方字段为 eastmoney，价格取 close。四条序列均与仓库留存的原始行情CSV逐日一致；日期与 exchange_calendars 4.13.2 的上海证券交易所交易日历在全区间完全相符。共同区间2015-01-05至2026-11-17，共2,884个收盘点、2,883个相邻收盘收益。没有重复日期、缺值或非正收盘价，每日重新计算的收益与库内 close_return 一致。原始文件与数据库为同一来源，本核验检验了数据处理链及交易日覆盖。")
table("研究对象与用途",["代码","指数","使用范围"],[
    ["000001","上证综合指数","实验一至三"],["000300","沪深300","全部四项，另作Beta基准"],
    ["000903","中证100（现称中证A100）","实验二、三"],["000132","上证100","仅实验四的跨指数空头"]])
add("训练期涵盖2015—2017年（731个收益日），验证期涵盖2018—2024年（1,699日），最终测试期为2025-01-02至2026-11-17（453日）。仅实验一遍历全部可行起点。实验二至四每个列报阶段均从净资产1、空仓开始，阶段末平仓。各期间的开平仓均在本阶段账户计费，初次成交为首个收益日之前一个交易日收盘。")
heading("数学模型与账户核算方法")
add(r"设 $P_{i,t}$ 为价格指数收盘点位，$r_{i,t}=P_{i,t}/P_{i,t-1}-1$。以 $h_t$ 表示空方相对于多方的目标金额比，$w_t$、$v_t$ 是按一倍总名义敞口归一的带符号权重。对恢复目标仓位的策略，有")
add(r"$$w_t=\frac{1}{1+h_t},\quad v_t=-\frac{h_t}{1+h_t},\quad g_t=w_t r_{L,t}+v_t r_{S,t}=\frac{r_{L,t}-h_t r_{S,t}}{1+h_t}.$$")
add(r"设 $A_t,B_t$ 为成交后的带符号头寸，$Q_t$ 是调仓绝对成交额之和，末日还包括平仓成交额。$N^-_t$ 为结算后实际承担的空头名义金额，$\Delta d_t$ 为跨越的自然日数。每单位成交收5个基点，借券年率2%，现金收益率为零。日末净资产和阶段复利收益为")
add(r"$$V_t=V_{t-1}+A_t r_{L,t}+B_t r_{S,t}-0.0005Q_t-0.02N^-_t\frac{\Delta d_t}{365},\qquad u_t=\frac{V_t}{V_{t-1}}-1.$$ ")
add(r"$$R=\prod_t(1+u_t)-1,\quad R_{\mathrm{ann}}=(1+R)^{252/n}-1,\quad \sigma_{\mathrm{ann}}=\sqrt{252}\,s(u),\quad S=\sqrt{252}\,\bar{u}/s(u).$$")
add(r"$$\mathrm{MDD}=\min_t\left(V_t/\max_{j\le t}V_j-1\right),\qquad \beta=\mathrm{Cov}(g,r_{300})/\mathrm{Var}(r_{300}).$$")
add("年化收益、波动、Sharpe与最大回撤按成本后账户计算，峰值序列含期初1；收益恒为零的Sharpe未定义。Beta用毛收益度量指数敞口。累计换手为每日成交额除以当天期初净资产后的总和（双腿绝对额相加，包括进出），年化换手为日均值乘252。费用以每单位初始本金计。成交发生在前收盘、借券费覆盖该收盘至本收盘；台账按收益日记录整段现金流，首笔交易费并入首日，末日另计平仓费。")
add("实验一首次严格按初始本金各建0.5元，交易费从现金扣除，因此成交后净资产为0.9995；对照账户从下一次调仓起按扣费后的可配置净资产恢复目标。实验二至四沿用原模型，全程按扣除当次交易费后的资本配置目标权重。交易量按市值漂移后的原仓位与新仓位之差计算。")
heading("实验一　建仓时点敏感性")
add(r"在每个具有252个完整后续收益日的收盘日 $s$ 建仓。主账户持有 $0.5/P_{L,s}$ 单位多头和 $0.5/P_{S,s}$ 单位空头，数量保持至平仓。其未扣费累计收益可由两端价格独立计算：")
add(r"$$R^{\mathrm{gross}}_{s,t}=\frac{1}{2}\left(P_{L,t}/P_{L,s}-P_{S,t}/P_{S,s}\right).$$")
add("固定数量账户在两次成交之间仍按日盯市，空头计费基数随指数点位变化。对照账户逐日恢复目标权重，按实际调仓额付费。两者各有2,632个窗口，首个起点2015-01-05。")
table("所有252日窗口的收益分布（每个建仓日等权）",["规则／口径","均值","中位数","标准差","25%分位","75%分位","最小","最大"],[
    [rules[r]+"/"+basis,*[pct(e1.loc[(r,b),c]) for c in ("mean","median","std","q25","q75","min","max")]]
    for r in rules for b,basis in [("gross","毛"),("net","净")]])
table("盈利概率的历史频数、持有期风险与成本",["规则","毛收益>0","毛收益<0","净收益>0","净收益<0","路径波动¹","交易费²","借券费²"],[
    [rules[r],*[pct(e1.loc[(r,b),c]) for b,c in [("gross","profit"),("gross","loss"),("net","profit"),("net","loss"),("net","mean_path_vol"),("net","mean_fee"),("net","mean_borrow")]]] for r in rules])
add("¹ 各窗口日净收益的年化波动，再对建仓日取平均。上一表的标准差衡量各窗口终值的分散程度。² 平均累计费用，占该笔初始本金。")
g=windows[windows.rule=="units"];low=g.loc[g.net.idxmin()];high=g.loc[g.net.idxmax()]
add(f"固定数量账户最差窗口为{low.entry}至{low.exit}（{pct(low.net)}），最佳为{high.entry}至{high.exit}（{pct(high.net)}）。每日恢复权重提高了平均净收益，但盈利日期占比反而较低。这两个指标分别对损益幅度和正负号敏感。")
figure("e1_entries","每个可行建仓日起持有252日的成本后收益")
figure("e1_distribution","固定数量与每日恢复权重的持有期净收益分布")
add("相邻起点共享251个收益日。为展示均值对样本路径的敏感性，沿建仓日序列按连续起点区块有放回抽样，区块越过样本尾部时循环到序列开头，两规则使用相同抽样索引，每次截取2,632个起点，重复5,000次，随机种子20261118。下表为百分位95%区间；区块长度改变后，固定数量规则的上界跨越零。有限历史中仅有少量长区块，均值应首先作描述性理解。")
table("净收益均值的循环移动区块Bootstrap区间",["规则","区块长度（起点数）","95%下界","95%上界"],[
    [rules[r.rule],r.length,pct(r.lo,4),pct(r.hi,4)] for r in block.itertuples()])
heading("实验二　多空方向选择偏差")
add("每对指数双向模拟。为在2024-12-31收盘执行，先以验证期起点至2024-12-30收盘的可获行情比较两方向，从空仓开始并按该日假设退出计费。此后锁定方向。完整验证期表另计至12月31日。测试期六个账户在12月31日收盘从空仓独立建立，不继承任何持仓或费用；六条净值和全部检验均来自这六个账户。")
table("六个方向各阶段独立账户的累计净收益",["多头／空头",*periods.values()],[
    [pair(a,b),*[pct(row[k]) for k in periods]] for (a,b),row in e2.iterrows()])
table("冻结方向与反向的比较（累计净收益）",["所选多头／空头","选向期所选","选向期反向","测试所选","测试反向","差额¹"],[
    [pair(r.a,r.b),*[pct(getattr(r,c)) for c in ("validation","validation_reverse","net","reverse")],num(100*r.gap)] for r in sel.itertuples()])
add("¹ 单位为百分点，按未取整数值计算。三个历史所选方向在选向期均为相对亏损较少的一方。进入测试期后，两组继续跑赢反向，仅上证综指多头／沪深300空头覆盖全部成本。")
add(r"每日配对差为 $d_t=u_{\mathrm{selected},t}-u_{\mathrm{reverse},t}$。检验 $H_0:E(d_t)=0$。用Bartlett权重的Newey–West标准误及有限样本修正，以正态近似作双侧检验。下式中的 $\hat{\gamma}_k$ 为按样本量 $n$ 归一的样本自协方差：")
add(r"$$\widehat{\mathrm{SE}}^2(\bar{d})=\frac{1}{n-1}\left(\hat{\gamma}_0+2\sum_{k=1}^K\left[1-\frac{k}{K+1}\right]\hat{\gamma}_k\right),\quad z=\bar{d}/\widehat{\mathrm{SE}}(\bar{d}).$$")
table("测试期配对检验（453个交易日，均值及区间单位为基点／日）",["历史所选多头／空头","滞后","日均差","95%下界","95%上界","原始p","Holm p"],[
    [pair(r.a,r.b),r.lag,num(r.mean*1e4,3),num(r.lo*1e4,3),num(r.hi*1e4,3),num(r.p,4),num(r.holm,4)] for r in hac.itertuples()])
add(f"每个滞后阶数内分别校正三对检验。主规格为5日，10日及20日下结论相同。若允许看完测试期再选方向，三对的平均累计净收益为{pct(sel.hindsight.mean())}，依历史选向则为{pct(sel.net.mean())}，事后多出{num(sel.hindsight_gap.mean()*100)}个百分点。差额完全来自沪深300与中证100。三对共同使用相同市场和日期，这一差额不能外推到整个历史策略搜索空间。")
figure("e2_curves","六方向累计净值，逐阶段从空仓及净资产1开始")
heading("实验三　对冲比例敏感性")
add(r"静态与滚动OLS均为带截距的日收益回归，指数加权法采用递推的中心协方差。令 $s_L,s_S$ 为估计期标准差，$\rho$ 为同期相关系数，$C_\lambda$ 为指数加权中心协方差，三种估计为")
add(r"$$h_{\mathrm{OLS}}=\rho\,s_L/s_S,\qquad h_{\mathrm{EWMA}}=C_\lambda(L,S)/C_\lambda(S,S),\qquad h_{\mathrm{vol}}=s_L/s_S.$$ ")
add("滚动OLS及波动率比使用120个收益日。EWMA使用衰减0.97、最少60个收益、递推均值（adjust=False）。静态OLS在验证和测试阶段均冻结于2017-12-28收盘的训练估计，留出12月29日用于执行。训练期自身使用最初120个训练收益估计一次并冻结至期末。为让五种方法训练期可比，统一在前121个收益日持有现金（120个观测及1个执行滞后），从2015-07-06开始获得持仓收益；现金日计入训练期长度。这样可以既报告训练期账户，又避免将2017年底才知道的斜率用于2015年交易。")
add("动态比率按日、周、月、季更新。以首个适用收益日为 t，仅使用截至 t−2 交易日收盘的数据，在 t−1 收盘执行。低频规则在相应自然周期的第一个收益日启用新比率，其余日仍逐日恢复目标权重。无效、缺失或非正的新估计沿用上一个有效值；若此前没有有效值用1。测试、验证及训练启用后的回退次数均为零。训练期每条路径121/731日按共同规则空仓（16.55%），验证、测试期为0%。完整日表保留执行日期、估计截止日期、目标比率和回退标记。")
table("最终测试期五种方法的六方向等权统计均值",["方法","毛收益","净收益","净年化","净波动","净Sharpe","净回撤","绝对Beta"],[
    [methods[m],*[pct(g[c].mean()) for c in ("gross","net","annual","vol")],num(g.sharpe.mean()),pct(g.mdd.mean()),num(g.beta.abs().mean(),3)]
    for m,g in d3.groupby("method",sort=False)])
table("最终测试期的平均头寸与成本（六方向，每日更新）",["方法","平均h","累计换手¹","交易费","借券费"],[
    [methods[m],num(g.mean_h.mean(),3),num(g.turnover.mean(),2),pct(g.fee.mean()),pct(g.borrow.mean())] for m,g in d3.groupby("method",sort=False)])
add("¹ 单位为倍，按各策略已实现日换手累计后取平均。各列为独立策略指标的算术均值。附录逐方向列示每个阶段的净收益；phase_statistics.csv 对每个方向、阶段及频率逐列保存所有指标。")
table("比率更新频率与平均年化净波动（测试期）",["更新频率","滚动OLS","EWMA","波动率比"],[
    [lbl,*[pct(e3[(e3.period=="test")&(e3.frequency==freq)&(e3.method==m)].vol.mean()) for m in ("rolling","ewma","vol")]]
    for freq,lbl in [("D","每日"),("W","每周"),("M","每月"),("Q","每季")]])
u=row3("000300","000903","one");v=row3("000300","000903","ewma")
add(f"对验证期选中的沪深300多头／中证100空头，改用每日EWMA后，测试期波动由{pct(u.vol)}降至{pct(v.vol)}，净收益由{pct(u.net)}降至{pct(v.net)}，Beta由{num(u.beta,3)}变为{num(v.beta,3)}。另一方向沪深300多头／上证综指空头的亏损则由{pct(row3('000300','000001','one').net)}收窄至{pct(row3('000300','000001','ewma').net)}。调仓有时减轻、有时扩大亏损。静态OLS的六方向平均绝对Beta高于等额法，也说明估计空头比率可能增加实现市场暴露。")
figure("e3_risk","五种方法在各方向的年化净波动与实现毛收益Beta")
figure("e3_curves","六方向、五种方法的独立测试账户净值")
heading("实验四　同指数与跨指数比较")
add(r"当两腿均为沪深300，设名义多头为 $L$、空头为 $S$，执行前合并为同一标的的净头寸。每单位名义总额对应的毛收益为")
add(r"$$g_t=\frac{L-S}{L+S}r_{300,t}=\frac{1-h_t}{1+h_t}r_{300,t}.$$ ")
add("h=1时零持仓，成交与借券均为零。h=0.5时持有三分之一净多头，h=2时持有三分之一净空头。沪深300与上证100是不同标的，对两腿各自市值变动和成交额计费。两种账户的名义输入均按本金一倍配置，净额结算后的实际持仓随净敞口缩减。")
add(r"定义 $r_{M,t}=(r_{L,t}+r_{S,t})/2$，$r_{R,t}=(r_{L,t}-r_{S,t})/2$。恒等式给出对称的共同方向与相对贡献：")
add(r"$$g_t=\frac{1-h_t}{1+h_t}r_{M,t}+r_{R,t}.$$ ")
add(r"对实际带符号头寸 $A_t,B_t$，共同损益金额为 $(A_t+B_t)r_{M,t}$，相对损益金额为 $(A_t-B_t)r_{R,t}$。逐日累加两个金额，扣除实际交易费、借券费，恰等于期末净资产减1。若归因对象是无费用毛净值，改用前一日毛净值乘当日两项收益后求和。两种累计基数分别保存在统计表中。此分解中的共同项是两指数收益均值；相对项可以与市场相关，均不承担资产定价因子识别功能。")
table("测试期净账户损益桥（每1元初始资金的百分点）",["空头","方法","共同损益","相对损益","交易费","借券费","净收益¹"],[
    [names[r.b],methods[r.method],*[num(100*getattr(r,c),3) for c in ("net_common","net_relative","fee","borrow","net")]]
    for r in e4.itertuples() if r.b!="000300" or r.method in ("one","half","double")])
add("¹ 共同损益＋相对损益−交易费−借券费。各项独立取整。同指数的四种估计法均得h=1，与等额行完全重合。最终测试期跨指数的七种设定均净亏损。")
cross=e4[(e4.b=="000132")&(e4.method=="static")].iloc[0]
eq=e4[(e4.b=="000132")&(e4.method=="one")].iloc[0]
add(f"跨指数等额账户的共同贡献为零，全部毛收益{pct(eq.gross)}来自相对变化，成本后净收益{pct(eq.net)}，实现Beta为{num(eq.beta,3)}。跨指数静态OLS在毛账户中得到共同贡献{num(100*cross.gross_common)}、相对贡献{num(100*cross.gross_relative)}个百分点，合成毛收益{pct(cross.gross)}。正的共同方向敞口抵消了部分相对亏损。")
figure("e4_curves","沪深300多头分别对冲同一标的和上证100的测试期净值")
heading("综合讨论")
add("实验一表明，选择起点与是否允许权重漂移会同时改变尾部亏损、交易量和盈利频数。实验二控制交易规则后，历史所选方向可以继续少亏，但盈利仍取决于相对变化能否覆盖两腿成本。实验三对协方差的响应有助于控制部分波动；交易方向错误时，提高风险估计频率并不能稳定修复净收益。实验四进一步说明，账面上的两条名义腿是否形成真实的两个头寸，直接决定投资者实际承担的费用。")
heading("研究局限")
add("本研究使用价格指数，没有现金分红，也未模拟ETF跟踪误差、申赎及借券可得性、股指期货基差、展期、保证金或真实滑点。固定数量按指数单位盯市仅为价格敞口模拟。交易费用和年借券率是情景设定。行业或规模权重差异可能解释相对表现，本次没有成分归因数据来识别这一机制。")
add("本研究的样本外是一次历史留出。研究动机来自已知的历史曲线，研究者层面的事后关注仍可能存在。回测按当时可用的信号时序执行；三组配对的Holm校正只覆盖指定检验族。这里没有对已消亡指数与留存指数作比较，不能据此推断幸存者偏差。区块重抽样保留局部依赖，对长期制度或风格变化仍有局限。")
heading("结论")
add("在给定的四条指数与资金、成本口径下，建仓时间与持仓规则影响收益的分布。样本没有给出可靠证据证明验证期较优的方向能在测试期持续产生正净收益。动态比率可降低部分风险，但其净收益效果随方向而异。判断跨指数策略是否有价值，须同时观察相对损益、残余方向敞口和持有成本。")
heading("参考文献")
add(old.split("# 参考文献")[1].split("# 附录")[0])
heading("附录　实验参数、统计结果与复现")
for period,name in periods.items():
    q=e3[(e3.period==period)&(e3.frequency=="D")].pivot(index=["a","b"],columns="method",values="net")
    table(f"{name}六方向累计净收益（动态法每日更新）",["多头／空头",*[methods[m] for m in list(methods)[:5]]],[
        [pair(a,b),*[pct(values[m]) for m in list(methods)[:5]]] for (a,b),values in q.iterrows()])
comp=read("version_comparison")
keys=[("mean_net_units","实验一主规格均值"),("mean_net_daily","实验一每日对照均值"),("net_000300_000300_one","实验四同指数等额净收益"),
      ("net_000300_000132_one","实验四跨指数等额净收益"),("gross_common_000300_000132_static","实验四跨指数静态共同贡献¹")]
table("与原报告的口径对照",["统计量","原版","修订版","变动来源"],[
    [label,pct(v.v1),pct(v.v2),reason] for (k,label),reason in zip(keys,["每日恢复改为固定数量","首次各按初始本金50%成交","同标的净额计费","测试期重新开户","对称收益分解"])
    for v in [comp[comp.measure==k].iloc[0]]])
add("¹ 毛账户按初始本金归一。原版等额同指数测试期的−2.08%来自连续持有、两腿分账计费；修订后整个路径零成交。实验二已公布的空仓测试净值可以重现；发生变化的是统计检验所用账户。完整逐项差额见 results/version_comparison.csv。")
add(f"复算顺序为 run.py、verify.py、figures.py、build_manuscript.py、build_word.py --word-fields（均在本目录）。分析依赖沿用仓库 requirements.txt，日历另用 exchange_calendars==4.13.2；Word生成用原版的 pypandoc_binary、python-docx 与 Windows Word。核验覆盖两套建仓规则的逐日净值、{verification['account_paths']}个阶段账户、{verification['daily_rows']:,}条账户记录、{verification['distinct_signal_estimates_checked']:,}种去重后的动态及静态估计输入组合，以及全部9项HAC结果。")
add("data/input_closes.csv 保存共同价格；results/e1_windows.csv 与 e1_paths.npz 保存逐窗口明细及毛净路径；e2_paired_daily.csv、e2_hac.csv 和 e2_selection.csv 对应配对差、显著性与历史锁定记录；e2_daily.csv.gz、e3_daily.csv.gz、e4_daily.csv.gz 保存逐日账本；phase_statistics.csv 提供每种设置的各阶段毛净收益、净年化、波动、Sharpe、回撤、毛Beta、平均比率、换手、费用和金额归因。figures 目录含PNG与可编辑SVG。可直接编辑本报告的Word或Markdown；手工编辑Markdown后只需重新运行Word排版脚本。")
(ROOT/"research_report.md").write_text("\n\n".join(parts)+"\n",encoding="utf-8")
