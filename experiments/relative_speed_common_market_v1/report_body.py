import numpy as np
import pandas as pd
from core import BASE,OUT,CFG

def render(r,s,models,p,peaks,decomp,reg,rr,rs,validation,changes,sim,null,titles,o):
    from presentation import table,interval
    def pick(df,cols):return table(df[cols.split()].copy())
    def fig(i):return f'![图{i} {titles[i-1]}](figures/fig_{i:02d}.png)'
    base=models.query("proxy=='000985' and K==0 and method=='A' and lag==10")
    result=decomp.query("K==0 and method=='A'").iloc[0]
    diagnostics=[]
    for (m,k),g in pd.read_csv(OUT/'daily.csv.gz').groupby(['method','K']):
        a=g.query('not removed'); y=a.original_bp.to_numpy();z=a.residual.to_numpy()
        diagnostics.append(dict(method=m,K=k,mean_retained_bp=y.mean(),residual_mean_bp=z.mean()*1e4,
            top2_abs_fraction=np.sort(abs(y))[-2:].sum()/abs(y).sum(),
            top2_positive_fraction=np.sort(y)[-2:].sum()/np.maximum(y,0).sum(),
            beta_max_delta=np.max(abs(g.post_beta.values-s.post_beta.values))))
    dg=pd.DataFrame(diagnostics)
    dg.to_csv(OUT/'gradient_diagnostics.csv',index=False)
    basers=rs.query("method=='A' and K in [0,2,3] and series=='daily'")
    nu=null.query("block==20 and K==0 and method=='A' and term!='beta'")
    ref=sim.query("block==20 and method=='A' and K==0 and metric in ['max_60','min_60','positive_run','negative_run']")
    rd=sim.query("block==0 and method=='A' and K==2 and metric in ['mean','max_60','positive_run']")
    val=validation.query("period=='full'")
    oosmean=interval(o.dynamic_error.values*1e4); ny=validation.query("period!='full' and comparator=='fixed_error'")
    ch=models.query("proxy.str.contains('matched') and term=='gamma' and lag==10")
    raw=s.speed_bp
    gamma=result.fixed_beta
    # Conditional variance shares and signed terminal attribution have different denominators.
    fixmarket=gamma*r.market.sum()*1e4
    diff=o.dynamic_error.sum()*1e4
    from statsmodels.stats.multitest import multipletests
    claims=pd.DataFrame(dict(hypothesis=['mean spread','fixed beta','up vs down','dynamic innovation mean',
        'validation innovation mean','gain vs fixed','gain vs asymmetric'],
        p=[base.loc[base.term==t,'p'].iloc[0] for t in ['drift','gamma','up_minus_down','residual_mean']]+
        [oosmean[3],*val.p.tolist()]))
    claims['holm']=multipletests(claims.p,method='holm')[1]
    claims.to_csv(OUT/'primary_hypotheses.csv',index=False)
    examples=rr.query("method=='A' and K==2 and series=='daily' and sign==1").nlargest(3,'contribution_bp')
    from analysis import oos
    from core import filt,kalman
    common=r.dropna(subset=['CH3','CH4'])
    proxy_rows=[]
    for proxy in ['market','CH3']:
        sub=common.copy()
        sub['market']=common[proxy]
        x=sub.market.values;y=sub.d.values
        a,b,noise,q=kalman(y,x)
        prior=filt(y,x,a,b,noise,q)[1]; u=y-a-prior*x
        er=oos(sub,output=None)
        proxy_rows.append(dict(proxy=proxy,n=len(y),beta_300=np.polyfit(x,sub.a,1)[0],
            beta_SSE=np.polyfit(x,sub.b,1)[0],r2_300=np.corrcoef(x,sub.a)[0,1]**2,
            r2_SSE=np.corrcoef(x,sub.b)[0,1]**2,market_bp=(prior*x).sum()*1e4,
            resid_mean_bp=u.mean()*1e4,resid_p=interval(u)[3],
            gain_vs_fixed=1-(er.dynamic_error**2).sum()/(er.fixed_error**2).sum()))
    proxies=pd.DataFrame(proxy_rows)
    proxies.to_csv(OUT/'proxy_checks.csv',index=False)
    report=f"""# 沪深300与上证综指的相对收益速度、共同市场暴露及连续偏移

## 摘要

以沪深300减上证综指定义日相对对数收益。两指数在2015-01-05至2026-11-17之间有{len(r)+1:,}个共同收盘点。沪深300取得{np.expm1(r.d.sum()):.2%}的期末相对价格财富优势。最有利两日中和后，终值符号翻转，但60日速度峰值依然出现在2017-11-21，达到13.50基点/日。中证全指的固定市场斜率差为{gamma:.4f}，只解释3.28%的日差方差。动态暴露对顺序验证期的同期归因有所改善；固定暴露模型的残差也可形成多个正向区间。在允许区块依赖和极值搜索的比较中，未取得普通日存在独立持续溢价的证据。

## 一、数据、变量与反事实

以2015-01-05收盘归一，首个收益日为{r.index[0]:%Y-%m-%d}。数据库、归档收盘及旧实验日差逐日核对，主样本完整，收盘为正，日期无重复。中证全指只作市场代理。价格指数不计现金分红。数据检查、关键日清单复用和数学口径见[执行方案](README.md)及[分析输入](results/inputs.csv)。明汯因子文件使用日频小数，mktrf 为市场组合减无风险收益，rf_dly 为日无风险收益。将二者相加再取 log1p 得到市场代理，另按工作簿原超额简单收益比较。CH-3、CH-4的市场列相同，模型另含的规模、价值或换手因子未进入此次单因子回归。工作簿不能独立确认每只股票的分红修订史。[因子来源](https://www.mingshiim.com/database) · [因子构造论文](https://doi.org/10.1016/j.jfineco.2019.03.008)

每日速度 d 为两条指数日对数收益之差；累计偏移为 d 的前缀和，W日平均速度为窗口和除以W。下文1基点指0.0001对数收益。A按d降序、B按d升序、C按绝对值降序取前K日，K为0至10以及15、20。路径上将相应贡献置零，日历长度固定；估计模型时跳过这些收益观测，状态方差继续逐日传播。删除名单事后形成。

## 二、速度与关键日期

原始日速度最高{raw.max():.2f}基点（{raw.idxmax():%Y-%m-%d}），最低{raw.min():.2f}基点（{raw.idxmin():%Y-%m-%d}）。累计偏移的最高点却在2021-02-10（{s.cum_bp.max():.2f}基点）。表1的日期均为完整窗口结束日。局部峰谷根据平滑速度前后增量变号及中心二阶差分标注，只能在下一交易日确认。[全部局部转折、斜率与曲率](results/turns.csv.gz)

**表1　未经中和的窗口速度极值（基点/日）**

{pick(p,'window max_bp max_date min_bp min_date')}

{fig(1)}

{fig(2)}

**表2　关键日数与主窗口峰值**

{pick(peaks.query("window==60 and K in [2,5,10,20]"),'method K max_bp max_date min_bp min_date')}

对A法，中和最有利的两日并未触及原60日最高速度；在已检验梯度中，首次迁移发生于K=15。删除某天只影响包含该日的窗口。原始终点对两天敏感与普通日期间仍能持续跑赢可以同时成立。[全部梯度及窗口累计收益](results/daily.csv.gz)

{fig(3)}

{fig(4)}

## 三、市场敏感度及涨跌非对称性

以 d 对常数项和同期中证全指日对数收益回归；两指数使用相同样本和解释变量，因此差额回归斜率严格等于两条单独回归斜率之差。非对称回归将市场收益正部和负部同时放入方程。表3列Newey–West十阶标准误及逐项95%区间；[全表](results/models.csv)保留5、10、20阶、调整R²、残差方差、均值和梯度校正。HAC在原日期间距上计算。

**表3　固定市场暴露与分段斜率（日对数小数）**

{pick(base.query("term in ['drift','gamma','up','down','up_minus_down']"),'term value se lo hi p adj_r2')}

上行斜率为0.1025，下行斜率接近零，二者之差的十阶HAC检验 p<0.0001。上涨日沪深300的胜率为55.39%，下跌日为41.02%。下跌日较常跑输与下跌幅度越大、相对损失就越大是不同命题。两组按当日市场收益符号划分。

{pick(reg.query("method=='A' and K==0"),'state n win raw_bp market_bp resid_bp')}

共同日期为{int(ch.n.iloc[0])}日，首末为{r.CH3.dropna().index[0]:%Y-%m-%d}至{r.CH3.dropna().index[-1]:%Y-%m-%d}；收盘后无法获取的因子值不作前向填充。同样本更换代理后，固定斜率方向一致。

{pick(ch,'proxy n value lo hi adj_r2')}

在相同日期进一步分别估计两条指数的市场斜率，并以各市场代理重新训练动态模型。r2_300与r2_SSE是市场代理对**单条指数**的解释率，区别于表3对两条指数**差额**的解释率。两种口径的验证期仅含有匹配观测的日期，结果与全期验证分列。

{table(proxies)}

{fig(6)}

## 四、暴露的时间变化与累计归因

随机游走斜率模型令状态创新和观测创新互不相关、截距固定。以高斯创新似然估计噪声方差，比较零状态方差的边界解。图7给出更新后的状态及给定全期超参数的条件区间，不含参数估计误差。归因则取吸收当日相对收益之前的斜率乘当日实际市场收益。估计的日状态方差为{result.state_variance:.6f}，条件斜率一日创新标准差为{np.sqrt(result.state_variance):.4f}。为避免据曲线目测断点，另用相邻自然年截距和斜率交互检验年度暴露升降；2026年为不完整年度。表4列相邻年差幅最大的上升、下降，各区间从全部年度检验族校正。[全部年度检验](results/annual_changes.csv)

{pick(changes.loc[[changes.change.idxmax(),changes.change.idxmin()]],'first second change lo hi holm')}

{fig(5)}

{fig(7)}

全期日差方差与终点归因使用不同分母。固定模型仅有约3.28%的方差解释率，其市场项累计{fixmarket:.2f}基点。动态模型中，实际差额{result.total_bp:.2f}基点等于截距{result.intercept_bp:.2f}、市场项{result.market_bp:.2f}和一步创新{result.residual_bp:.2f}之和。动态市场项为负，更高市场敏感度不足以解释期末略微领先。估计时点和模型设定均会影响这组分解。[逐K分解](results/decomposition.csv)

{pick(decomp.query("K in [2,20]"),'method K fixed_beta state_variance total_bp intercept_bp market_bp residual_bp')}

{fig(8)}

## 五、关键日之后的连续偏移与推断

区间定义为日差严格同号的最大连续交易日段；零或缺失中断。平滑速度同号区间另存，不能把一个滚动均值持续为正解读为每个交易日都跑赢。表5中count为段数、longest为天数，最后一列为各同号区间中与市场累计方向一致的比例。每段起止、贡献与同期市场差额见[完整区间](results/runs.csv.gz)。

{pick(basers,'K sign count longest sum_bp fraction_market_same')}

A法删两日后，按累计正贡献取前三段，列示实际起止与市场方向。即使一段内沪深300每日都相对领先，中证全指本身仍可能下跌。

{pick(examples,'start end days contribution_bp market_bp')}

下表比较中和后保留日的平均速度、残差速度（基点/日）、剩余前两正日占全部正贡献的比例，以及滤波斜率相对原始路径的最大绝对变化。与计算路径时使用的全日历分母不同，此处速度只除以保留日数。[全部梯度诊断](results/gradient_diagnostics.csv)

{pick(dg.query("K in [2,20]"),'method K mean_retained_bp residual_mean_bp top2_positive_fraction beta_max_delta')}

{fig(9)}

固定Beta残差带截距，样本内残差和机械归零。另用动态模型的一步创新检验平均及累计漂移；在全期拟合超参数条件下，HAC十阶均值检验 p={base.query("term=='residual_mean'").p.iloc[0]:.3f}，对累计量的标准误等于均值标准误乘有效天数。图10的区间只对单个窗口有效。正向区间存在，不足以说明漂移稳定或异常。

{fig(10)}

零平均差基准将联合日收益按同步循环区块抽样，对相对收益差移除全样本均值。每个长度5、20、60分别10,000次，每次重选赢家日、输家日和绝对差日，重新搜索各期限的最大及最小速度。模拟比较的是历史极值在稳定零漂移过程中的位置；不会为峰值日期本身提供置信区间。等量随机删除另在实际日历抽取10,000次。表6给20日区块下原始序列的双尾经验p值，以及按同类指标跨K、方向和窗口的Holm校正。原始60日峰谷均落在宽泛的模拟范围内。

{pick(ref,'metric observed median p holm')}

A法中和2日与均匀中和2日的对照如下，速度与均值为日对数小数，游程为天数。定向删除有意挑出尾部，对终值有影响不等于对最长连续段也有影响。[全部模拟](results/simulations.csv)

{pick(rd,'metric observed median p holm')}

另以固定市场斜率为基准，同步抽市场和残差区块，对每块残差随机换号，保留块内连续结构；每次重新排名、删日、拟合截距与斜率，检验市场调整残差的最长同号段与60日极值。为衡量是否全靠一两个区间，逐次排除最强的两个互不重叠60日窗口，再搜索第三段。表7为未删日的20日区块结果；下行极值以绝对幅度显示。第三段正向速度为{nu.query("term=='third_positive60'").observed.iloc[0]*1e4:.2f}基点/日，因此正向积累可在多段观察到；多重校正后仍与固定暴露基准相容。这一检验依赖块残差符号对称假设。[逐梯度和区块结果](results/residual_null.csv)

{pick(nu,'term observed lo hi p holm')}

## 六、扩展窗口验证与结论

首504日用于初始化，固定及非对称回归按日扩展；动态模型每63日仅用历史重新估计超参数，每天用上一日滤波状态归因。验证期{o.index[0]}至{o.index[-1]}。表8的gain为相对于各对照的平方损失降幅，loss_bp2及其HAC区间为“对照损失减动态损失”，以平方基点计。

{pick(val,'comparator n gain loss_bp2 lo hi p')}

相对固定线性规格，年度切片有{int((ny.gain>0).sum())}/{len(ny)}个取得正改善。动态模型虽降低整体条件归因误差，各时段的增量须结合年度差异理解。验证期动态误差合计{diff:.1f}基点，平均{oosmean[0]:.3f}基点/日（十阶HAC区间[{oosmean[1]:.3f}, {oosmean[2]:.3f}]，p={oosmean[3]:.3f}）。验证期均值及累计量均不足以识别可持续的独立收益。[逐年比较](results/validation.csv)

将原始样本三项暴露或漂移问题、两项创新均值及两个验证期损失比较合并为七项主检验（HAC十阶），Holm结果如下。相对固定Beta的改善通过校正，相对固定非对称模型的证据较弱。各设计另有逐梯度或峰值搜索校正，不把同一行情的多个口径计作独立验证。

{table(claims)}

共同市场暴露存在，但固定线性模型留下大部分日相对收益方差。上涨和下跌日的斜率明显不同。删去最有利的两三个交易日，正向区间和60日正峰仍存在；控制市场和区间搜索后，持续获利的解释缺乏支持。因子代理对固定斜率的符号影响较小，但价格指数无法检验分红后的财富分解。动态模型使用同期已实现的市场收益，其改善衡量条件归因能力。行业、估值、权重重叠和制度事件的作用仍无法由这两条指数的日收盘序列识别。
"""
    (BASE/'research_report.md').write_text(report,encoding='utf8')
