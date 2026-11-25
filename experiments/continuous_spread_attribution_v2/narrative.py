from models import B,C,V1
import numpy as np
import pandas as pd

def display(df,cols=None):
    t=df.copy() if cols is None else df[cols.split()].copy()
    for c in t.select_dtypes('float'):
        t[c]=t[c].map(lambda v:f'{v:.4g}')
    return t.to_markdown(index=False,disable_numparse=True)
def fig(i,alt):return f'![图{i}　{alt}](figures/fig_{i:02}.png)'
def write(r,period,cases,mask,attr,year,metric,tests,claims,ht,reg,changes,increment,refits):
    g=mask.query("rule=='local_win' and K in [1,2,5]").copy()
    ends=[str(s[-1].date()) for s in cases.values()]
    g=g[g.end.isin(ends)].pivot(index=['start','end','raw_bp'],columns='K',values='net_bp').reset_index()
    p=attr.query("rule=='raw' and ((panel=='CH' and model=='M4A') or (case=='2026_trough' and panel=='full' and model=='M2'))")
    aux=tests.query("rule=='residual_win' and block==20").iloc[0]
    full_m=metric.query("W==60 and panel=='CH'").set_index('model')
    threshold=full_m.threshold_bp.iloc[0]
    case_stats=[]
    for label,s in cases.items():
        h=r.loc[s]; side=np.sign(h.d.sum())
        case_stats.append([label,s[0].date(),s[-1].date(),h.d.sum()*1e4,h.market.sum()*1e4,h.d.std()*1e4,(h.d*side>0).mean()])
    cs=pd.DataFrame(case_stats,columns=['案例','开始','结束','实际bp','市场bp','日差标准差bp','同向日占比'])
    raw=mask.query("rule=='raw'");abs_rule=mask.query("rule=='local_abs' and K==2")
    bigger=(abs_rule.net_bp.abs()>abs_rule.raw_bp.abs()+1e-8).sum()
    extreme=period.query("kind=='extreme'")
    changes=changes[changes.model.isin(['M2','M4A'])]
    second=pd.read_csv(B/"results/secondary_checks.csv")
    local=pd.read_csv(B/'results/local_loss.csv.gz').query("panel=='full' and base=='M2'")
    best=local.loc[local.groupby('W').sum_gain_bp2.idxmax()]
    rb=tests.query("rule=='local_win' and K==2").pivot(index='null',columns='block',values='p').reset_index()
    longest=period.query("window==60 and kind=='speed_run'").sort_values('n',ascending=False).groupby('sign').head(1)
    retained=attr.query("((panel=='CH' and model=='M5') or (panel=='full' and model=='M3' and case=='2026_trough')) and ((rule=='local_win' and K in [2,5]) or rule=='raw')").copy()
    retained['explained_bp']=retained.raw_bp-retained.error
    text=f"""# 沪深300与上证综指连续相对收益偏移的多因子归因与动态暴露局部性

## 摘要

沪深300与上证综指可以共享大部分市场波动，又在特定持有期内持续拉开距离。以2015-01-06至2026-11-17的2,883个日收益为基础，逐窗中和赢家贡献最大的交易日后，四个历史案例均保留原方向。2017年正峰删两日后保留600基点；2021年初正峰的规模因子可解释约549基点。解释日度差额与解释持有期累积差额的结果存在明显差别。相比固定非对称市场模型，随机游走Beta的60日平方归因误差仅下降1.23%；控制风格后的额外降幅为1.13%，两项均未通过主检验。观察得到模型未解释的连续偏移，其异常性和可持续收益含义尚未确立。

## 一、冻结设计、样本与模型

[冻结协议](protocol.md)的提交为7771bf3，研究基线为c99d3fe。沿用V1逐日核对过的收盘序列。令d为沪深300减上证综指的日对数收益，1基点=0.0001对数收益。表中累计偏移取每日d的和。图1的W日速度取窗口和除以W。对于滚动速度同号的阶段，以第一个到最后一个合格**窗口结束日**为阶段边界，仅累计落在这些实际日期上的d。[全部阶段与极值](results/periods.csv)列出边界、实际贡献、同期市场、日差标准差和同向比例。[分阶段模型归因](results/period_attribution.csv.gz)覆盖具有完整顺序估计的阶段。

全样本市场代理为中证全指日对数收益。共同面板使用CH原始简单市场超额收益、SMB、VMG及CH4的PMO。两个工作簿共用市场列；各自的SMB按相应版本取值。共同面板含2,816日，止于2026-08-31；2026-06-05至06-30之间的17个交易日缺因子。估计只使用可匹配日；评价窗口必须在原始市场日历上连续完整，不把缺口拼接。CH模型以简单相对财富收益 exp(d)−1 为响应，所有模型在这个面板内使用相同响应和日期。对模型给出的简单收益和s，将每个加项乘以 log(1+s)/s，s=0时系数为1，剩余项取实际d减各归因项之和。这是总拟合值的单位转换；风格多空收益保持原始取值。两套面板的模型排名只在各自内部解释。[因子数据口径与V1核验](../relative_speed_common_market_v1/research_report.md)

M0为扩展均值，M1加入固定市场，M2再加入市场收益的正部，因而market项表示下行斜率乘市场收益，up_extra表示上行斜率的额外部分。M3用V1随机游走斜率；滚动OLS另用60、120、252日估计截距和斜率。M4_3和M4_4分别加入对应风格，M4A在CH4基础上增加非对称控制。M5先拟合M4A，再对剩余项估计随机游走市场暴露，允许状态方差为零。M3与M2是竞争规格；M5的静态部分和状态超参数每63日联合刷新一次，M4A逐日扩展，因此实际损失差也包含刷新频率差异。各模型前504日只训练；当天的条件归因仅使用前收盘可得的参数及当日实现的因子。价格指数不计现金分红。

## 二、连续区间怎样形成

{fig(1,'连续相对速度及探索性案例')}

**表1　完整案例窗口及实际收益差**

{display(cs)}

**表2　不同观察期限的正、负速度极值**

{display(extreme,'window start end mean_speed_bp raw_bp agree')}

60日速度连续同号时间最长的两个阶段如下。阶段起点处的速度已经包含起点之前的59个收益日；raw_bp只加总起止日期之间实际发生的相对收益差。

{display(longest,'start end sign n raw_bp market_log_bp vol_bp agree')}

同向日占比按原窗口整体方向计算。强窗口内可有多日反向波动。2021年两例是从结束于该自然年的所有完整60日窗口中取正负极值；这些案例用于解释构成，显著性必须回到全部窗口。

## 三、全历史删日与区间内部中和

对{len(raw)}个完整60日窗口分别排序。按原始区间赢家选择1、2、5日，另以绝对日差选日；历史名单亦按同样方向分别构造。所有操作均保留60日分母。图2比较四种名单。表3的1、2、5列为中和**窗内赢家日**后的累计差，单位为基点。

{display(g)}

{fig(2,'选日范围与中和梯度')}

2017年最强正窗的前两大全历史赢家日均在窗外，按全历史删两日仍领先810.20基点，改为窗内删两日后余600.02基点。2026年负峰的两天按有利于上证综指排序，中和后仍为−779.71基点。在全部窗口中，按绝对差中和两日后有{bigger}窗的绝对偏移反而扩大，原因是被中和的收益原先抵消了部分区间领先。[全部中和结果](results/neutralized_windows.csv.gz)

## 四、实际偏移由哪些模型项解释

**表4　主要案例的逐项归因（累计对数基点）**

{display(p,'case panel model raw_bp intercept market up_extra SMB VMG PMO error')}

2021年初的正峰中，规模项与沪深300的领先同向，市场、非对称和风格控制后的残差约141基点。2017年同样有约810基点的领先，规模项只有约102基点，剩余约663基点。同等幅度的相对收益对风格变量的依赖程度可以相差很大。2021年负峰的规模项约−283基点，仍有约−637基点残差；2026年负峰超出因子覆盖范围，只列市场模型。

**表5　动态规格相对固定规格的条件拟合增量（基点）**

{display(increment)}

表5为两个模型的总拟合差，截距也可能改变，不能再把该差额加入动态模型本身的市场项。完整记录把M5额外状态贡献标为dynamic。[多模型分解及同日掩码反事实](results/case_attribution.csv.gz) · [全窗口反事实误差](results/counterfactual_errors.csv.gz)

{fig(3,'四个案例的逐日累计条件归因')}

同时中和实际差额与该日全部模型贡献，得到下列保留日归因。2026年案例因CH数据不足改列全期M3。

{display(retained,'case model rule K raw_bp explained_bp error')}

主规格保持原历史训练。敏感性在四个案例中按K=2重估M2、M3、M4A、M5：从训练观测和滤波更新中排除指定日，再对同一保留日求残差合计。窗内规则只影响进入窗口之后的更新；全期名单可能影响早期训练，只作事后敏感性。与固定原模型相比，这些配对敏感性中，残差绝对变化的中位数为{(refits.refit_error_bp-refits.frozen_error_bp).abs().median():.2f}基点，最大值为{(refits.refit_error_bp-refits.frozen_error_bp).abs().max():.2f}基点。[重估明细](results/case_refits.csv.gz)

## 五、动态暴露增量集中于何时

以日平方误差之差评价动态模型，正数表示动态模型误差更小。2021年占全样本相对固定Beta日度改善总额的51.8%，占相对非对称Beta改善总额的76.0%。仅去掉2021年的评价日、保留原训练历史后，两种降幅分别为4.81%和1.61%；后者的单侧HAC检验p=0.251。CH面板M5相对M4A的2021年份额可大于100%，因为其余年份合计抵消了部分改善。

{fig(4,'每日平方损失差的滚动合计')}

{fig(5,'年度改善贡献')}

{fig(6,'逐年剔除评价观测')}

**表6　市场波动与斜率变化的局部性（同组每日平均改善，基点²）**

{display(reg.query("panel=='full' and base=='M2'"),'variable group n mean_gain_bp2')}

相对全期M2，三个持有长度中日损失改善最集中的窗口如下。这是对实现损失按时间定位，不能据这些日期事前切换模型。

{display(best,'W start end sum_gain_bp2')}

波动较高、斜率变化较快时，相对非对称模型的平均改善较大。平静组点估计仍为正；分组相关不识别波动导致模型改善的因果关系。每个60、120、252日区间的起止及日损失改善总额可在[局部损失表](results/local_loss.csv.gz)中复核。[各年贡献与逐年剔除](results/yearly.csv)

**表7　相邻年市场斜率变化**

{display(changes,'panel model year next_year down_change up_change p holm')}

允许相邻两年全部系数各自取值，用十阶HAC联合检验两条市场斜率。CH面板控制风格后，两次年度比较仍达到校正后5%水平。该结果支持指定年间的平均暴露差，但不能定位结构断点，也不能说明额外状态参数改善60日解释。

## 六、持有期累计误差

对连续W日取 E=Σ(实际差额−当日模型拟合差额)，再计算绝对值和平方。方向以窗口拟合总和与实际总和是否同号计。训练期内60日绝对累计差的90%分位数为{threshold:.1f}基点，冻结作为大幅窗口阈值；错向或幅度未达阈值均为漏判。阈值只用于样本内定义的分组，尚缺投资者外部风险预算。

**表8　CH共同验证期的60日累计误差、方向与漏判**

{display(metric.query("panel=='CH' and W==60"),'model n mean_error_bp mae_bp mse_bp2 accuracy large miss')}

{fig(7,'同样本日度与持有期误差')}

**补充诊断　60日滚动OLS的改善有多少来自更新斜率？**

在看到持有期结果后，另固定当时的扩展市场斜率，只用过去60个观测更新截距。下表区分相对M2的整体改善和相对这个局部截距对照的改善，并列出静态风格项的增量。全样本滚动OLS相对M2降低60日平方误差40.6%，相对只更新截距的对照仅降低4.2%，后者未达到显著水平。各行属于单独的探索性检验族；gain为累计平方损失降幅，HAC滞后59阶。

{display(second)}



在CH共同样本中，M4A将60日均方误差较M2降低{1-full_m.loc['M4A','mse_bp2']/full_m.loc['M2','mse_bp2']:.1%}。方向判断由61.0%提高至78.9%。加入动态项后，漏判减少，但方向准确率降至76.9%，累计平方误差的额外改善很小。60日滚动OLS在这段历史的累计误差也较低；它同时更新局部截距和斜率，不能把全部改善归给时变Beta。不同模型并无统一的指标排序。[各期限与每日评价](results/holding_metrics.csv) · [逐窗实际与误差](results/holding_windows.csv.gz)

{fig(8,'加入风格控制前后的累计残差路径')}

**表9　60日累计平方误差改善：重叠和非重叠口径**

{display(ht.query("W==60 and ((panel=='full' and base=='M2') or base=='M4A')"),'panel base model stride n gain mean se p')}

主规格以59阶HAC处理重叠；每60个交易日取一个窗口，从首个合格窗口固定起算，作为对照。mean与se单位均为平方基点，p为单侧。日度损失之和与持有期误差平方不同，后者还包含不同日期误差的交叉乘积。

## 七、重新搜索区间后的推断

每个零模型和区块长度各模拟2,000次。零漂移模型同步重抽实际收益向量所决定的日差，先扣除平均差；固定市场与非对称模型同步抽取市场和残差区块，残差整块随机换号；风格零模型在CH可用日期采用同样处理，同时抽取风格、市场和缺测标志。每次重新搜索全部合格60日窗口、重判区间赢家并按指定规则选日。区块换号假定块内残差可共同翻转；只保留块内的时间依赖。主统计量是中和赢家两日后**重新搜索得到的最大绝对累计差**；它不必与原始峰值来自同一窗口。风格零模型参照其自身可观察日历。

{fig(9,'全窗搜索极值的模拟参考分布')}

**表10　预设六项主检验，主区块20日**

{display(claims)}

不同区块长度的未校正高尾p值如下，四个基准在三种设定下均未达到5%。

{display(rb)}

区块长度5、60的结果及K=1、5等敏感性见[完整零模型比较](results/null_tests.csv)。固定非对称+CH4下，另逐次重估固定系数，检查调整后残差自身的窗内赢家两日中和极值。观察值{aux.observed_bp:.1f}基点，在20日区块下的零模型95%参考范围为[{aux.lo:.1f}, {aux.hi:.1f}]基点，单侧p={aux.p:.3f}。该项是对固定风格基准的辅助检验；没有把对照模型的全期拟合残差当成真实交易期可获得的预测误差。

## 八、研究结论

**存在模型尚未解释完的连续相对价格偏移。现有检验不足以说明它系统性地超出普通市场、非对称暴露和风格暴露叠加相关残差所能产生的范围。** 对历史基准比较，2017年与2021年两段规模近似的领先应给出不同的风险解释；对模型选择，固定风格控制已提供明显的描述性改善，复杂动态规格在60日累计评价中没有通过增量检验。本轮将M5保留为对照。指数价格未计股息，因子可能遗漏行业、调样及其他风险，稀有大幅行情也限制有效独立区间数。保持正式投稿论文与本轮探索分开。
"""
    (B/'research_report.md').write_text(text,encoding='utf8')
