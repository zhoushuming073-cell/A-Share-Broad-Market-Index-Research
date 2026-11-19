"""Build all figures and the editable manuscript, then typeset it to PDF in desktop Word."""
import json,shutil,tempfile,importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from data import *
from diagnostics import extra
F=ROOT/"figures";F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,
                     "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","font.size":9})
names={"base":"固定1:1","static_best":"事后最佳固定配比","return":"事后收益冠军","ideal_curve":"理想评分参照","overall":"事后综合冠军",
       "wf_return":"滚动收益选参","wf_ideal":"滚动理想选参","wf_overall":"滚动综合选参"}
families=dict(fixed="固定",momentum="相对动量",reversion="相对反转",trend="共同趋势",voltarget="波动预算",hedge="动态对冲",regime="状态切换",fusion="信号融合")
def read(name):return pd.read_csv(OUT/(name+".csv"))
def pct(x):return f"{x:.2%}"
def build():
    extra()
    nav=pd.read_csv(OUT/"daily_nav.csv",index_col=0,parse_dates=True)
    d=read("daily_positions");d["date"]=pd.to_datetime(d.date)
    cand=read("candidate_metrics");pareto=read("pareto_front");wf=read("walk_forward")
    trial=read("all_trials");conv=read("convergence");rob=read("robustness");attr=read("attribution").set_index("strategy")
    seg=read("periods").set_index(["strategy","period"]);effects=read("parameter_effects")
    winner=read("top_return").iloc[0];overall=read("top_overall").iloc[0];ideal=read("top_ideal_curve")
    near=read("ideal_nearest").iloc[0] if ideal.empty else ideal.iloc[0]
    audit=json.loads((OUT/"verification.json").read_text())
    parts=['---\ntitle: "A股宽基多空策略的收益—风险边界"\nsubtitle: "多策略族联合搜索、多目标评价与半年滚动验证"\ndate: "2026年11月19日　|　历史计算实验"\nlang: zh-CN\n---']
    def add(s):parts.append(s)
    def head(s):add("# "+s)
    def tab(title,h,rows):
        add("表："+title);add("\n".join("| "+" | ".join(map(str,row))+" |" for row in [h,["---"]*len(h),*rows]))
    def fig(key,title,x,y,note):
        f=plt.gcf();ax=plt.gca();ax.set(xlabel=x,ylabel=y);ax.grid(axis="y",alpha=.18)
        f.suptitle(title);f.text(.08,.008,note,fontsize=8);f.tight_layout(rect=(0,.05,1,.96))
        for ext in ("png","svg"):f.savefig(F/(key+"."+ext),dpi=220,bbox_inches="tight")
        plt.close(f);add(f"![](figures/{key}.png)");add("图："+title+"。"+note)
    def axes():return plt.subplots(figsize=(8.6,3.8))[1]
    def curves(columns,water=False):
        ax=axes()
        # Combine completely identical curves into a single explicit legend entry.
        groups={}
        for k in columns:
            h=tuple(nav[k].fillna(-9).round(11));groups.setdefault(h,[]).append(k)
        for keys in groups.values():
            s=nav[keys[0]].dropna();s=pd.concat([pd.Series(1.,index=[P.index[P.index.get_loc(s.index[0])-1]]),s])
            if water:s=s/s.cummax()-1
            ax.plot(s.index,s,label=" / ".join(names[k] for k in keys),lw=1.05)
        ax.legend(fontsize=7,ncol=2);return ax
    head("摘要")
    add(f"以同一组2,884个价格指数收盘点为输入，在八类规则中生成初始固定库{len(cand):,}个去重候选。全历史事后收益冠军扣费后累计{pct(winner.net)}、年化{pct(winner.cagr)}、最大回撤{pct(winner.mdd)}。事前逐窗执行的收益选择规则从2018年起累计{pct(seg.loc[('wf_return','all'),'net'])}，综合选择规则累计{pct(seg.loc[('wf_overall','all'),'net'])}。预定理想门槛下共有{len(ideal)}个合格解。报告分别列示参数回看、固定库后续历史和18次实时序选参；三个对象回答不同问题。")
    head("数据、时序与账户")
    add("上证综指000001为A、沪深300指数000300为B。使用仓库v2价格快照并核对上海交易日历，2015-01-05收盘至2026-11-17收盘，2,883个日收益。所有统计采用252个交易日年化，现金及Sharpe的无风险利率为零。价格指数不含分红；不能按指数点位实际成交，尚未纳入ETF跟踪、期指基差、保证金制度、滑点及实际可借规模。")
    ax=axes();(P/P.iloc[0]).set_axis(["上证综指","沪深300"],axis=1).plot(ax=ax);fig("01_prices","两条价格指数的全历史走势","交易日","基期=1","2015–2026，收盘价归一，未计产品费用。")
    add("收益日t的特征仅使用截至t−2收盘的收益，t−1收盘执行。信号确认后仅在设定频率的交易日调整，期间持仓随市价漂移。名义总敞口最高为自有权益的2.5倍，对应40%总名义保证金；漂移超限于下一次收盘决策强制恢复目标。成交按费后权益迭代确定，末日清仓。空头卖出款全额受限，自有权益可用于多头，超额多头借款年率3%，空头借券2%，单边成交费5bp，按相邻收盘自然日/365计息。权益耗尽触发破产异常，本轮没有破产路径。")
    head("决策函数与参数空间")
    add("组合按目标总敞口G、空多比h及方向s配置，A权重=sG/(1+h)，B权重=−sGh/(1+h)。固定家族始终多A空B。允许反向的扩展家族在负信号下交换符号。静态总敞口0.5–2.5，h取0.35–2.2；频率为每1、5或20个交易日。方向确认1或3日，寿命5、20或60日，到期退出。动量、趋势的标准化强度为窗口平均日收益乘根号窗口长度再除以窗口标准差。所有窗口属于10、20、40、60、120日。")
    tab("八类规则及条件变量",["家族","方向、对冲或持仓规则"],[
        ["固定","固定h与G；另保留G=h=1的日调仓对照"],["相对动量","A−B日收益强度越过0.2–1.8阈值；确认、平仓或反向"],
        ["相对反转","对平稳性要求更弱的收益差采用反向强度信号；不假设价格协整"],["共同趋势","用(A+B)/2收益强度控制是否交易与方向"],
        ["波动预算","按过去(A−B)/2波动率将总仓缩放至上限；目标2%–10%，为代理风险预算"],
        ["动态对冲","等额、首120日静态OLS、滚动OLS、EWMA、波动比、共同基准Beta匹配、总敞口最小方差"],
        ["状态切换","共同趋势强度绝对值超1时追随相对动量，否则反转；波动冲击比率过滤"],
        ["信号融合","0.65相对动量+0.35共同趋势；波动上限、目标风险及4%–15%回撤后10日冷却"]])
    add("传统OLS为Cov(A,B)/Var(B)，固定多头金额时最小化剩余方差。一倍总敞口下令wA−wB=1，最小方差h=(Var(A)+Cov)/(Var(B)+Cov)。相对等权市场的Beta匹配在本模型与该式等价。动态比率裁剪至0.15–5，缺失退回1；未形成有效方向信号时空仓。策略库未把所有可想象的连续交易阈值或机器学习规则纳入搜索。")
    head("分层搜索与评分协议")
    add("首个外层窗之前：2015–16用于训练，2017为内层验证，截止至2017-12-28。固定随机种子7319、9319；每个种子随机128、打乱Sobol 256、分组多变量TPE 128、NSGA-II 128次。随机与Sobol在八族内均匀分层，类别离散映射，连续值保留四位小数。TPE代理为0.35训练Sharpe+0.65验证Sharpe+2验证CAGR−验证回撤（Sharpe截至±4）。NSGA-II种群32，以验证CAGR、负回撤、Sharpe最大化，继承上阶段8个代理高分起点。种子对比是顺序热启动试验，非独立重复。初次Sobol加随机共768次；TPE采用有限预算，把更多计算用于逐窗重新选择。")
    add("之后每半年取最近约三年为内层，最后一年为验证，两年为训练；将外层首笔执行收盘留出，搜索截止到再前一交易日。每窗重评冻结库并追加24次TPE、64次NSGA-II。验证Pareto集中仅考虑活跃率至少15%的候选，邻域总敞口与阈值或配比同时±10%扰动。A按CAGR、B按理想评分、C按综合评分冻结。若当窗没有合格B，以分数最高者作为近似策略，明示不合格。外层平仓后下一窗重新开仓，按窗末本金顺序复利。信号没有重叠的未来标签；时间分隔用于保障收盘执行顺序。")
    add("全历史榜单专评首窗冻结的参数库，按完整历史事后排序。A与C要求15%活跃率；B另要求年化≥1.5%、Sharpe≥0.65、回撤≤12%、活跃≥30%、盈利年度比例≥60%且对数净值斜率为正。B的增长、回撤及持续时间、Sharpe/Sortino、盈利月份及Ulcer、盈利年度、成本换手权重为20/25/20/15/15/5。C对增长、风险、Sharpe组合、邻域、后续窗口表现、成本、绝对Beta与参数数目分别加权25/15/12/15/12/8/8/5。内层用已知训练表现代替后续表现。固定尺度截至[0,1]，CAGR以12%、回撤25%、持续500日、年化波动20%、Sharpe 2、Sortino 3、Ulcer 10%、年化成本4%、年化换手80倍为尺度。具体映射见scoring.py。全部阈值先于最终排名设定。")
    head("全历史榜单与日度净值")
    tab("代表策略的固定参数及全历史表现",["榜单","参数ID","家族","累计","CAGR","回撤","Sharpe"],[
        [lbl,row.id,families[row.family],pct(row.net),pct(row.cagr),pct(row.mdd),f"{row.sharpe:.2f}"] for lbl,row in
        [("收益",winner),("理想近似" if ideal.empty else "理想",near),("综合",overall)]])
    add("收益冠军参数为 "+winner.params+"。正、负动量均可开仓，方向反转时两腿同时交换。图中完全相同的净值合并图例，完整排名和每一个候选的具体决策参数随报告发布。")
    curves(["base","static_best","return","ideal_curve","overall","wf_overall"])
    fig("02_nav","完整日度账户净值","交易日","初始净值=1","2015–2026，固定参数事后筛选；滚动账户从2018年起。均已扣费。")
    curves(["return","ideal_curve","wf_return","wf_ideal"]);fig("03_champions","收益目标与理想目标的实际路径","交易日","成本后净值","固定策略同一起点；两条滚动规则均始于2018年，独立本金1。")
    curves(["base","return","ideal_curve","wf_overall"],True);plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    fig("04_drawdowns","从历史高点到当日的水下深度","交易日","回撤","逐日净值含初始本金，2015–2026或2018–2026。")
    add(f"冠军最长水下期{int(winner.duration)}个交易日，水下日占比{pct(winner.underwater)}，盈利月占比{pct(winner.months)}。理想榜{'为空，近似者虽获高分仍超过回撤上限' if ideal.empty else '有策略达标'}。收益榜与综合榜{'选出同一参数组合' if winner.id==overall.id else '选出不同组合'}。这些曲线的事后识别不构成盈利预测。")
    ax=axes();pts=ax.scatter(cand.mdd,cand.cagr,c=cand.sharpe,s=8,cmap="viridis",alpha=.4)
    ax.scatter(pareto.mdd,pareto.cagr,s=24,facecolors="none",edgecolors="red",label="三目标非支配");ax.legend();plt.colorbar(pts,ax=ax,label="Sharpe")
    fig("05_pareto","全历史三目标非支配集合","最大回撤（小数）","净CAGR（小数）","固定库事后计分。红圈含第三维Sharpe，二维投影可能重叠。")
    ax=axes();groups=cand.groupby("family");ax.boxplot([g.cagr for _,g in groups],tick_labels=[families[f] for f,_ in groups],showfliers=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1));fig("06_families","各族固定参数的全历史收益分布","策略族","成本后CAGR","2015–2026，事后；箱线省略离群点，完整数据保存在候选表。")
    family=cand.groupby("family").agg(n=("id","size"),median=("cagr","median"),best=("cagr","max"),dd=("mdd","median"))
    tab("各族样本分布（非等概率投资组合）",["家族","候选数","CAGR中位","CAGR最高","回撤中位"],[[families[k],int(r.n),pct(r["median"]),pct(r.best),pct(r.dd)] for k,r in family.iterrows()])
    grid=rob[(rob.id==winner.id)&(rob.scenario=="grid")].drop_duplicates(["gross_factor","other_factor"]).pivot(index="gross_factor",columns="other_factor",values="cagr")
    ax=axes();im=ax.imshow(grid*100,aspect="auto",cmap="YlGnBu");ax.set_xticks(range(len(grid.columns)),[f"{x:.1f}" for x in grid.columns]);ax.set_yticks(range(len(grid)),[f"{x:.1f}" for x in grid.index]);plt.colorbar(im,ax=ax,label="净CAGR（%）")
    fig("07_grid","收益冠军参数邻域","第二参数倍率","总敞口倍率","全历史2015–2026，二参数5×5扰动；独立复算每一条成本后路径。")
    ax=axes();init=conv[conv.fold==0]
    for (alg,seed),g in init.groupby(["algo","seed"]):ax.plot(g.n,g.hv,marker=".",label=f"{alg} {seed}")
    ax.legend(fontsize=7,ncol=4);fig("08_search","不同预算的Pareto超体积","算法内累计评价数","三维超体积","首窗内层验证；参考劣点为CAGR −50%、回撤75%、Sharpe −6。")
    tab("初始探索各算法成本与末次诊断",["算法","种子","次数","最优验证CAGR","HV","搜索耗时秒¹"],[
        [r.algo,r.seed,r.n,pct(r.best),f"{r.hv:.2f}",f"{r.wall_seconds:.2f}"] for r in init.groupby(["algo","seed"]).tail(1).itertuples()])
    add("¹ 从各组首次评价结束至最后一次评价结束，包括Optuna候选生成；最优值及超体积轨迹见 convergence.csv。初始种群继承既有高分点，算法间的差异同时包含热启动与搜索预算影响。")
    head("严格时序选择及后续历史")
    matched=read("matched_fixed_folds").groupby("rule").net.agg(lambda u:(1+u).prod()-1)
    add(f"采用与18个外窗完全相同的开仓日、清仓日及调仓日历，冻结参数的等额基准累计{pct(matched['equal'])}，事后收益冠军累计{pct(matched['retrospective'])}。这一对照将切窗引起的再平衡及双边交易费也对齐。后一个参数的冠军身份使用了后续历史，仍属事后分析；逐窗事前选参的结果单独列示。")
    tab("阶段累计净收益（空白表示尚未启动）",["账户","2015–17","2018–24","2025–26","全段累计","全段回撤"],[
        [names[k],*[pct(seg.loc[(k,per),"net"]) if (k,per) in seg.index else "—" for per in ["early","main","append","all"]],pct(seg.loc[(k,"all"),"mdd"])] for k in names])
    ax=axes();g=wf[wf.policy=="overall"];ax.plot(g.fold+1,g.valid_cagr,label="内层验证");ax.plot(g.fold+1,g.outer_cagr,label="下一半年");ax.axhline(0,color="gray",lw=.7);ax.legend();ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig("09_walk","综合选择规则的内外层表现","外层窗口序号","年化净收益","18个按时间冻结的窗口。最后窗截至2026-11-17，短窗年化易放大波动。")
    add(f"以相同TPE代理在各窗前16次与全部预算中选择，再以外层真实账本拼接，两者累计分别为{pct(seg.loc[('wf_budget16','all'),'net'])}和{pct(seg.loc[('wf_budget_full','all'),'net'])}。更多尝试能提高库内最优值，不保证下一窗改善。2025–2026的原始历史在先前研究中已被查看，本实验不能把它当成全新盲测。")
    head("头寸、资金成本与风险归因")
    ax=axes();g=d[d.strategy=="return"].set_index("date")
    for arr,lab in [(g.a/g.start,"上证"),(g.b/g.start,"沪深300"),(g.gross_weight,"总敞口"),(g.net_weight,"净敞口")]:ax.plot(arr,label=lab,lw=.7)
    ax.legend(ncol=4);fig("10_weights","事后收益冠军的逐日持仓","交易日","期初净资产倍数","2015–2026；带符号头寸来自实际成交，空白风险日持仓为零。")
    ax=axes();cols=["constant_direction","dynamic_direction","relative","fee","borrow","finance"]
    aa=attr.loc[["base","return","wf_return","wf_overall"],cols].copy();aa.loc[:,cols[3:]]*=-1
    aa.index=[names[k] for k in aa.index];aa.columns=["平均净敞口","动态净敞口","相对变动","交易费","借券","融资"];aa.plot.bar(ax=ax,rot=0);ax.legend(ncol=3,fontsize=8)
    fig("11_attribution","期初单位本金的累计损益分解","策略","本金单位","固定策略2015起；滚动规则2018起；正负两组条形之和为累计净收益。")
    a=attr.loc["return"]
    add(f"将(A+B)/2作为共同收益，(A−B)/2作为相对收益，分别乘以当日实际头寸之和与之差。共同项再分解成全段平均净权重与偏离平均的部分，仅为描述性恒等式。收益冠军共同损益{a.common:+.3f}元，其中动态净敞口{a.dynamic_direction:+.3f}元；相对损益{a.relative:+.3f}元，交易费{a.fee:.3f}、借券{a.borrow:.3f}、融资{a.finance:.4f}元。标签为{a.driver}，绝对Beta较低仍可有大的条件方向暴露。120日实现Beta在daily_positions.csv。")
    head("稳健性、经济解释与结论")
    stress=rob[(rob.id==winner.id)&rob.scenario.str.startswith("fee=")].drop_duplicates("scenario")
    tab("冠军的单独成本冲击",["费率／借券／融资情景","净CAGR"],[[r.scenario,pct(r.cagr)] for r in stress.itertuples()])
    by=effects.groupby("parameter")[["delta_cagr","delta_mdd"]].agg(lambda x:x.abs().max())
    tab("冠军一次仅改一个变量的最大绝对变化",["参数","CAGR变化","回撤变化"],[[k,pct(r.delta_cagr),pct(r.delta_mdd)] for k,r in by.sort_values("delta_cagr",ascending=False).iterrows()])
    alt=cand[cand.active>=.15].nlargest(1,"alternate").iloc[0]
    pinfo=wf[wf.policy.isin(["return","ideal","overall"])].copy();pinfo["gap"]=pinfo.outer_cagr-pinfo.valid_cagr
    correlation=pinfo.complexity.corr(pinfo.gap,method="spearman")
    add(f"对该冠军，在所试邻域内，对收益最敏感的是{by.delta_cagr.idxmax()}，对回撤最敏感的是{by.delta_mdd.idxmax()}。二维邻域收益范围{pct(grid.min().min())}至{pct(grid.max().max())}。±10%邻域稳定度{winner.stability:.3f}（1为最稳）；早期、主段、追加段分别重开账户后的CAGR为{pct(winner.early_cagr)}、{pct(winner.main_cagr)}、{pct(winner.append_cagr)}。局部稳定性无法抵消重复筛选产生的乐观偏差。")
    add(f"三目标全历史前沿有{len(pareto)}个参数。不同风险容忍度对应不同可选点。提高风险与邻域权重后，综合第一名为{alt.id}（原为{overall.id}）。内外层年化差与参数数目的Spearman相关为{correlation:.2f}，重复窗口和相同赢家使其只能作描述性诊断。多信号融合族中位CAGR为{pct(family.loc['fusion','median'])}；这些观察没有隔离交互项的因果效果。")
    fixed_dir=cand[~cand.params.map(lambda s:json.loads(s).get("reverse",False))]
    allow_dir=cand[cand.params.map(lambda s:json.loads(s).get("reverse",False))]
    add(f"含择时而始终只允许多上证、空沪深300的候选共{len(fixed_dir)}个，最高CAGR为{pct(fixed_dir.cagr.max())}，中位{pct(fixed_dir.cagr.median())}；允许反向的{len(allow_dir)}个候选分别为{pct(allow_dir.cagr.max())}、{pct(allow_dir.cagr.median())}。这两个候选集合的参数分布不同，不能将收益差直接归于方向反转本身。")
    add(f"相较固定等额，全部候选中有{int((cand.excess_cagr>0).sum())}/{len(cand)}个在完整历史上获得更高CAGR。但固定族不得反向，最高值为{pct(family.loc['fixed','best'])}；历史收益最高的结构属{families[winner.family]}。绝对收益、低回撤和持续性需同时评价，最高历史收益不等于最稳的未来策略。本次{'没有合格理想策略' if ideal.empty else '出现符合形态门槛的历史策略'}，滚动收益、理想、综合规则的真实时序表现见表。即便事后固定赢家在部分后续年份盈利，也没有证据证明滚动选择能反复识别它。")
    add("本次是有成本、有限结构和有限预算的经验边界。相互依赖的候选、共享市场路径、研究者已知历史及短而不独立的半年窗限制统计推断。没有把搜索次数等同于独立试验次数，也未对参数挑选的收益进行显著性宣称。")
    head("复现与数据索引")
    add("逐笔试验见 all_trials.csv，内层与事后前沿分别见 inner_fronts.csv、pareto_front.csv；候选完整参数、风险成本、三段统计与相对基准差见 candidate_metrics.csv。榜单与最近似候选独立存档。walk_forward.csv 提供每次冻结的ID、执行日、信息截止和参数。daily_nav.csv 与 daily_positions.csv 保存日度净值及会计桥。verification.json 记录账户及排名核验。")
    md="\n\n".join(parts)+"\n";(ROOT/"research_report.md").write_text(md,encoding="utf-8")
    # Reuse the repository's field, table and Chinese typography pipeline.
    spec=importlib.util.spec_from_file_location("layout",ROOT.parent/"four_pair_tests_v2/build_word.py")
    layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
    import win32com.client
    with tempfile.TemporaryDirectory() as tmp:
        layout.ROOT=Path(tmp);(layout.ROOT/"research_report.md").write_text(md,encoding="utf8")
        shutil.copytree(F,layout.ROOT/"figures");docx=layout.build()
        word=win32com.client.DispatchEx("Word.Application")
        try:
            word.Visible=False;word.DisplayAlerts=0
            doc=word.Documents.Open(str(docx),AddToRecentFiles=False)
            doc.Fields.Update();doc.Repaginate();doc.TablesOfContents(1).Update()
            doc.ExportAsFixedFormat(str(ROOT/"research_report.pdf"),17);doc.Close(False)
        finally:word.Quit()
if __name__=="__main__":build()
