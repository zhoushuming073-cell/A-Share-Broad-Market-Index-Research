import json,shutil,tempfile,importlib.util
from pathlib import Path
import fitz
from data import *
from figures import make,LABEL
from deletion import flip
def pct(x):return f"{x:.2%}"
def build():
    make()
    o=read("original_metrics").iloc[0];un=read("random_deletion_summary");core=un[un.reps==100000]
    year=read("annual_contributions");yr=year[year.rule=="original"]
    det=read("deterministic_deletion");sel=det.set_index(["rule","k"])
    top=read("top_positive_days");low=read("top_negative_days")
    st=read("statistical_tests");hac=st[(st.group=="full")&(st.method=="HAC")&(st.lag==10)].iloc[0]
    u=core.set_index("k");v=json.loads((OUT/"verification.json").read_text())
    partial=read("stratified_deletion_summary");blocks=read("block_deletion_summary")
    bridge=read("hedge_bridge").iloc[0]
    md=['---\ntitle: "两天能改变终值：宽基指数相对收益的关键交易日依赖性"\nsubtitle: "沪深300与上证综指的同步删帧、路径拟合与漂移检验"\ndate: 2026年11月19日\nlang: zh-CN\n---']
    def head(s):md.append("# "+s)
    def add(s):md.append(s)
    tables=0
    def table(title,headers,rows):
        nonlocal tables
        tables+=1
        add(f"表 {tables}　"+title)
        add("\n".join("| "+" | ".join(map(str,row))+" |" for row in [headers,["---"]*len(headers),*rows]))
    def fig(i,title):
        add(f"![](figures/figure_{i:02}.png)")
        add(f"图 {i}　"+title)
    head("摘要")
    add(f"2015-01-05至2026-11-17，沪深300比上证综指的终值高{pct(o.Q)}。累积正负相对对数收益抵消了{pct(o.cancellation)}；删除两个事后最有利日即可使领先消失。对固定历史做{v['total_simulations']:,}次同步无放回抽帧，其中七个核心比例各含100,000次均匀抽样。随机删除10%交易日后，领先翻负的频率为{pct(u.loc[count(.1),'p_flip'])}。平均日相对对数收益仅{o.daily_mean*10000:.3f}个基点，HAC双侧检验p={hac.p:.3f}。数据支持终值对关键日期敏感，而不足以证明沪深300享有稳定正漂移。")
    head("数据、时间坐标与财富口径")
    add("使用仓库 four_pair_tests_v2/data/input_closes.csv 的上证综指000001（A）和沪深300 000300（B），与只读SQLite库及上海交易日日历复核。2,884个正收盘价配对形成2,883组日收益。价格指数未含现金分红。先逐日从相邻真实收盘价计算两条对数收益，再使用同一日期掩码过滤。没有跨越已删除日期重新计算任何收益。")
    add(r"令 $d_t=\log(B_t/B_{t-1})-\log(A_t/A_{t-1})$。对删除集合 $S$，保留日的累计对数差 $D^{(-S)}=\sum_{t\notin S}d_t=D-\sum_{t\in S}d_t$，相对财富优势 $Q^{(-S)}=\exp(D^{(-S)})-1$。图中用原日历轴呈现，删除日维持前值；均值、相关性、回归与路径偏离统计只计保留日。所有终值均为累计值。由于删帧不改变原日历跨度，本文不把保留日压缩为新的实际投资年数。")
    add(f"原始买入持有累计价格收益为上证综指{pct(o.wealth_a-1)}、沪深300 {pct(o.wealth_b-1)}。两者相差{100*(o.wealth_b-o.wealth_a):.2f}个百分点，财富比值减一则为{pct(o.Q)}。相对日标准差{o.daily_sd*1e4:.2f}bp。B领先的交易日{int(o.positive)}个（{pct(o.positive/N)}），落后{int(o.negative)}个（{pct(o.negative/N)}）。正日对数差合计{o.pos:.5f}、负日{o.neg:.5f}、合计{o.D:.5f}。抵消率定义为1−|净和|/绝对贡献之和。")
    fig(1,"两指数全时期买入持有价格净值");fig(2,"沪深300相对上证综指的累计对数收益轨迹");fig(3,"日相对收益分布及两个尾端日期")
    head("定向删帧：哪些日期能改变结论")
    add(f"按d从大到小删除，删1日尚余{pct(sel.loc[('best',1),'Q'])}，删2日为{pct(sel.loc[('best',2),'Q'])}；因此最小翻转数k⁺={flip()}。反过来，删去最差2日，领先扩大到{pct(sel.loc[('worst',2),'Q'])}。单侧删帧是事后极值构造，恰好给出固定k的可达下界和上界。即使极值超出10万次随机结果，也不能用它计算普通显著性p值。")
    table("翻转所需的最佳两日及相应的最差两日（两指数为原始简单收益）",["类别","日期","上证","沪深300","d（bp）"],[
        [lab,r.date,pct(r.return_a),pct(r.return_b),f"{r.d*1e4:+.2f}"] for lab,g in [("最佳",top),("最差",low)] for r in g.head(2).itertuples()])
    table("同删k日后的财富优势Q",["k","删最佳","删最差","删最大绝对差"],[[k,*[pct(sel.loc[(rule,k),"Q"]) for rule in ("best","worst","absolute")]] for k in sorted(set(CFG["counts"])|{count(p) for p in [.01,.02,.05,.10,.20]})])
    fig(4,"依次删除最佳相对收益日的合成路径");fig(5,"三种尾部排序的路径比较");fig(6,"终点相对差随删除比例变化")
    head("随机删帧：历史优势能抵抗多大扰动")
    add("均匀无放回抽样使用Floyd算法，在候选位置已入选时改选此前不可达的当前末位置。每次重新从全体日期抽样，批次种子由主种子、机制、删除数和批次起点共同派生；各档日数按比例四舍五入。每次逐保留日重建双指数对数净值，计算终点、收益相关性及R²、均值、标准差、截距斜率、残差、累计偏离RMS与峰值等统计量。日收益回归包含截距，故R²等于相关系数平方。")
    table("均匀无放回，每档100,000次；财富比值的历史敏感性分布",["约删比例","天数","Q的5%分位","中位","95%分位","Q>0频率","翻负频率","翻负频率95%区间"],[
        [f"{prop:g}%",r.k,pct(r.Q_p05),pct(r.Q_p50),pct(r.Q_p95),pct(r.p_positive),pct(r.p_flip),f"{pct(r.p_flip_lo)}–{pct(r.p_flip_hi)}"] for prop,r in zip(np.array(CFG["proportions"])*100,core.itertuples())])
    add("表内的5%与95%分位描述不同删除集合导致的历史波动；末列Wilson区间只衡量有限模拟次数对共同抽样机制下频率估计的精度。两者都不表示未来市场收益的预测区间。删帧比例越大，随机扰动区间越宽；例如删除5%时，即使中位数仍为正，左尾已可出现明显落后。")
    fig(7,"随机删帧后的终值分布");fig(8,"随机删帧下保持正优势的比例")
    head("分层、连续区块与同预算比较")
    add("年度和逐日历月份分层使用随机起点的累计配额取整，各层精确到其理论配额上下1日以内、各层之和严格为k，然后在每层内等概率抽取。连续删帧将长度固定的块压缩成一个位置，无放回选择不重叠块的位置，再展开；最后不足整块的余数随机分配给一个块。允许相邻块相接，无首尾环绕。该机制在区间边缘的单日入选概率可以不同于中段。另对k小于指定块长的情形删一个长度k的块。")
    comb=pd.concat([core,partial,blocks]);g=comb[comb.k.eq(count(.10))]
    table("同删288日的机制差异",["机制","试验数","Q的5%分位","中位","95%分位","翻负频率","蒙特卡洛标准误¹"],[
        [r.rule,r.reps,pct(r.Q_p05),pct(r.Q_p50),pct(r.Q_p95),pct(r.p_flip),pct(r.mcse_flip)] for r in g.itertuples()])
    add("¹ 频率的标准误。均匀、分层、区块回答不同的固定样本条件问题。比较机制差异时需结合各自的模拟误差；不能将本样本中较高的区块翻转频率解释成未来出现相同连续行情的概率。")
    comparison=read("equal_budget_comparison");k=count(.1)
    table("定向删帧在同预算均匀随机分布中的位置（k=288）",["定向规则","定向Q","随机Q严格低于它的频率","同值频率"],[
        [LABEL[r.rule],pct(r.target_Q),pct(r.below),pct(r.equal)] for r in comparison[(comparison.k==k)&(comparison.random_method=="uniform")].itertuples()])
    add("完整 equal_budget_comparison.csv 包含每个定向预算与七种随机机制的经验位置。两端等量删帧要求偶数预算；各删0.5%、1%、2.5%、5%时按每端日数取整，所以有时总日数与直接对总比例取整相差一天。对照始终按实际k匹配。")
    head("路径接近、日收益拟合与稳健漂移")
    add("偏离指标为保留各日累计对数净值差的均方根及最大绝对值。另以同向日占比度量短周期方向一致性；终值Q单列，用于区分路径是否贴合、终点是否接近和日收益是否高度相关。日收益回归用B作因变量、A作自变量，残差标准差以n−2作分母。")
    table("删帧后的日收益拟合与累计路径距离",["规则","k","ρ","R²","Beta","截距bp","残差bp","RMS","最大偏离","Q"],[
        [LABEL.get(r.rule,"原始"),r.k,*[f"{x:.3f}" for x in (r.corr,r.r2,r.beta)],f"{r.alpha*1e4:.2f}",f"{r.residual_sd*1e4:.2f}",f"{r.rms:.3f}",f"{r.max_dev:.3f}",pct(r.Q)] for r in det[(det.k.eq(0))|((det.rule=="absolute")&det.k.isin([29,288]))|(det.rule=="symmetric")].itertuples()])
    absrow=sel.loc[("absolute",29)]
    add(f"删去分化最大的29日，日收益R²从{o.r2:.3f}升至{absrow.r2:.3f}，RMS由{o.rms:.3f}降至{absrow.rms:.3f}，终值Q却由{pct(o.Q)}变为{pct(absrow.Q)}。这三项给出的信息不同。均匀随机删除10%时R²中位数为{u.loc[288,'r2_p50']:.3f}，没有相似幅度的提高。")
    loc=read("robust_locations")
    table("相对日收益的稳健位置估计",["估计量","bp／日"],[[r.method,f"{r.estimate*1e4:+.3f}"] for r in loc.itertuples()])
    add("trim5删去两端各5%观测后求均值；winsor5将两端各5%拉回边界；Huber以中位数和固定MAD正态尺度起步，用1.345倍尺度对大偏差降权。中位数与Huber估计为负，普通均值略正，符号对尾部处理敏感。合成截尾曲线没有给投资者提供预知极端日的方法。")
    fig(9,"日收益R²与删帧比例");fig(10,"原始、正尾及预定首抽随机路径；被删日按日历标出")
    fig(11,"连续区块、分层和离散随机抽帧的同预算比较")
    head("年度贡献、市场状态与漂移检验")
    yc=year[(year.rule=="best")&(year.k==20)].set_index("year")
    table("各年对长期相对收益的贡献（2026年截至11月17日）",["年份","天数","正对数贡献","负对数贡献","年度Q","最佳20日入选数"],[
        [r.year,r.n,f"{r.pos:.4f}",f"{r.neg:.4f}",pct(r.Q),int(yc.loc[r.year,"removed"])] for r in yr.itertuples()])
    add(f"{len(yr)}个自然年（含末年部分区间）中，{int((yr.Q>0).sum())}年为正。最佳20日中2015年有7日、2024年有4日；但两个翻转关键日相隔多年。尾部贡献有所聚集，不能仅据两个日期给经济事件作因果归因。")
    state=read("market_states")
    table("以上证前120个交易日累计对数收益划分状态",["状态","日数","d均值bp","d标准差bp","最佳20日数"],[[r.state,r.n,f"{r.mean*1e4:.2f}",f"{r.sd*1e4:.2f}",r.top20] for r in state.itertuples()])
    add("以±0.05为趋势阈值，前120个收益日列为预热。状态仅用于分组，所有观测保留。")
    daily=read("daily_relative_returns")
    table("各长度完整滚动窗口的相对财富变化",["交易日","窗口数","最小Q","中位Q","最大Q","正窗口比例"],[
        [k,len(z),pct(z.min()),pct(z.median()),pct(z.max()),pct((z>0).mean())] for k in (20,60,120,252) for z in [daily[f"rolling_Q_{k}"].dropna()]])
    add("滚动窗口存在重叠。这些占比为描述性统计，不将窗口当成独立样本计算显著性。")
    fig(12,"正负贡献的年度分布与关键日期集中度");fig(13,"总正负贡献及相抵后的净额")
    test=st[(st.group=="full")&st.lo.notna()]
    table("原始日相对收益的双侧均值检验（bp／日）",["估计","滞后／块长","均值","标准误","95%下界","95%上界","p"],[
        [r.method,r.lag,f"{r.estimate*1e4:.3f}",f"{r.se*1e4:.3f}",f"{r.lo*1e4:.3f}",f"{r.hi*1e4:.3f}",f"{r.p:.4f}"] for r in test.itertuples()])
    add("普通检验用截距模型与常规t区间；Newey–West使用Bartlett权重、有限样本修正及双侧正态近似。循环移动区块Bootstrap每次有放回抽取连续块，直到截足原长，三个块长各20,000次；区间取均值百分位，p值由以样本均值中心化的双侧重抽样分布估计。这里从时间序列推断均值，区别于固定历史删帧实验。")
    lj=st[st.method=="LjungBox"];joint=st[st.method=="HAC_joint_equal_means"].iloc[0]
    add(f"Ljung–Box检验在5、10、20阶的p值依次为"+", ".join(f"{x:.3f}" for x in lj.p)+f"。按既有研究使用的2015–17、2018–20、2021–24、2025–26四段，带10阶HAC的联合等均值检验p={joint.p:.4f}。这是对指定边界的阶段差异检验，不是对未知断点位置的估计。逐年均值检验在12年内以Holm方法校正，最小校正p为{st.holm.min():.3f}。阶段均值有变化与全期均值显著为正是两个不同命题。")
    table("四个指定时期的日漂移",["时期","均值bp","HAC 95%下界","95%上界"],[[r.group,f"{r.estimate*1e4:+.3f}",f"{r.lo*1e4:.3f}",f"{r.hi*1e4:.3f}"] for r in st[st.group.str.match(r"^\d{4}-\d{4}$")].itertuples()])
    fig(14,"日均漂移的区间推断")
    head("解析结果与模拟核验")
    add(r"设 $I_t$ 为日t被删除的指示变量，$p=k/N$。固定k的无放回设计给出 $E[I_t]=p$，对不同日期有 $\mathrm{Cov}(I_i,I_j)=-p(1-p)/(N-1)$。将 $D^{(-S)}=D-\sum I_t d_t$ 展开，并以 $s_d^2=\sum(d_t-\bar{d})^2/(N-1)$ 表示有限总体方差，可得")
    add(r"$$E[D^{(-S)}]=(1-k/N)D,\qquad \mathrm{Var}(D^{(-S)})=k(1-k/N)s_d^2.$$")
    add("指数函数为凸函数，因此Q的条件期望须直接对每次exp(D)−1求平均；不能将D的解析期望简单指数化代替。")
    ac=read("analytic_checks");ac=ac[(ac.rule=="uniform")&(ac.reps==100000)]
    table("完全随机抽帧的有限总体解析核验（对数单位）",["k","理论均值","模拟均值","理论SD","模拟SD","均值误差／MCSE"],[
        [r.k,f"{r.theory_mean:.5f}",f"{r.observed_mean:.5f}",f"{r.theory_sd:.5f}",f"{r.observed_sd:.5f}",f"{r.z_mean:+.2f}"] for r in ac.itertuples()])
    add(f"独立种子对七档各再运行5,000次，与主种子的均值差最大为{v['independent_seed_max_abs_z']:.2f}个合并蒙特卡洛标准误。核验以原始价格比连乘重算{v['cases_verified']}组定向／原始终值，另以独立实现逐项检查{v['first_draw_masks_verified']}组首抽的全部指标与日期掩码。每次模拟均断言删除计数等于k。")
    fig(15,"随模拟次数增加的条件频率")
    head("与多空对冲研究衔接")
    section=f"原始价格指数的买入持有财富比高出{pct(o.Q)}。沿用前报告的资本与成本口径，每日多0.5单位沪深300、空0.5单位上证综指，从空仓建立、逐日恢复目标权重并在样本末平仓，毛累计收益为{pct(bridge.gross)}，成本后为{pct(bridge.net)}。单边成交费5bp，年借券费2%、按实际自然日，现金收益为零。财富比、等额再平衡收益以及择时策略收益的计算对象各异。净财富差很小、关键日期可以改变符号、跨期均值变化均削弱固定方向长期盈利的实证依据。动态策略还须证明事前预测与成本后的滚动表现；本实验没有检验新的预测信号。"
    add(section)
    head("结论与研究限制")
    add(f"用净额和总额作分母，关键日的重要性看起来截然不同。最有利两日贡献之和为{top.head(2).d.sum():.5f}，相当于最终净对数优势的{pct(top.head(2).d.sum()/o.D)}，但只占所有正日贡献的{pct(top.head(2).d.sum()/o.pos)}。即使取最有利20日，合计也仅占总正贡献的{pct(top.head(20).d.sum()/o.pos)}。因此，两天决定的是历史终点的正负号；全年市场中正向分化的规模远大于这两天。")
    add(f"就终值符号而言，两天足以抹去沪深300的领先；就全期收益形成而言，正负两端各累积了约3.7个对数单位，两个日期的重要性来自抵消后分母极小。它不能证明全部相对表现由两次孤立事件造成。删去最差两日反而把优势提高到{pct(sel.loc[('worst',2),'Q'])}，说明这种敏感性具有两面性。")
    add("在均匀删帧中，随机结果并不总是翻负，且翻负频率依赖删除强度和抽样机制。去掉绝对分化最大的日期常会改善日收益拟合，但保留路径仍可在多年内累积非零差异。均值区间包含零，不能由此证明真实期望严格为零，更不能推断未来不存在超额收益。历史事件解释、总收益指数、实际ETF可交易性与动态择时需要另行研究。")
    head("复现与交付")
    add("在仓库根目录运行 python experiments/relative_return_frame_deletion_v1/run.py。每批1,000次，主种子20261119，复核种子19470517。均匀七档共700,000次；补充预算、年度和月份分层及四种区块每个设定3,000次；复核种子35,000次。另有均值推断的60,000次有放回区块Bootstrap，不计入删帧试验数。每次迭代保存终点或汇总指标，不保存所有逐日模拟轨迹；批次缓存可断点续算，Git归档保存终值Q、抽样协议、首抽日期和完整分位统计。")
    add("结果目录内，daily_relative_returns.csv 含全部原始日数据、滚动优势、状态；top_positive_days.csv、top_negative_days.csv 按贡献排序；synthetic_paths.csv.gz 含定向删帧路径与逐日标志；年度贡献、相关回归、同预算经验位置及时间序列推断各有独立CSV。整合版PDF先完整保留既有多维对冲报告，再附本实验。")
    manuscript="\n\n".join(md)+"\n";(ROOT/"research_report.md").write_text(manuscript,encoding="utf8")
    (ROOT/"research_report_addendum.md").write_text("# 关键交易日依赖性与固定方向对冲\n\n"+md[2]+"\n\n"+section+"\n\n详见[独立实验报告](research_report.md)。\n",encoding="utf8")
    render(manuscript)
def render(md):
    spec=importlib.util.spec_from_file_location("layout",ROOT.parent/"four_pair_tests_v2/build_word.py")
    layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
    import win32com.client
    with tempfile.TemporaryDirectory() as tmp:
        layout.ROOT=Path(tmp);(layout.ROOT/"research_report.md").write_text(md,encoding="utf8")
        shutil.copytree(ROOT/"figures",layout.ROOT/"figures");docx=layout.build()
        word=win32com.client.DispatchEx("Word.Application")
        try:
            word.Visible=False;word.DisplayAlerts=0
            doc=word.Documents.Open(str(docx),AddToRecentFiles=False);doc.Fields.Update();doc.Repaginate();doc.TablesOfContents(1).Update()
            doc.ExportAsFixedFormat(str(ROOT/"research_report.pdf"),17);doc.Close(False)
        finally:word.Quit()
    old=ROOT.parent/"multidimensional_longshort_search_v1/research_report.pdf"
    with fitz.open() as out:
        toc=[]
        for label,p in [("第一部分　多维多空策略搜索",old),("第二部分　关键交易日与随机抽帧",ROOT/"research_report.pdf")]:
            with fitz.open(p) as source:
                offset=len(out);toc.append([1,label,offset+1])
                toc.extend([[lvl+1,title,page+offset] for lvl,title,page in source.get_toc() if page>0])
                out.insert_pdf(source)
        out.set_toc(toc);out.save(ROOT/"combined_research_report.pdf",deflate=True)
if __name__=="__main__":build()
