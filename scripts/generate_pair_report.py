from functools import lru_cache

from matplotlib.ticker import PercentFormatter
from optimize_hedge_ratio import metrics
from run_pair_search import inputs, load, paths, pick
from research_common import csv, figure, np, pd, plt, REPORTS

plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False})
METHOD={"fixed":"等金额","static":"静态OLS","rolling":"滚动OLS","vol":"波动率比","ewma":"EWMA","mr":"均值回归"}
FREQ={"D":"日","W":"周","M":"月","Q":"季"}
STATE={"advance":"主升","decline":"下跌","sideways":"震荡","transition":"过渡"}
ROLES={"locked":"验证锁定","test_best":"测试事后冠军","benchmark":"原配对等敞口基准"}


def tab(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),
                     *["| "+" | ".join(str(v) for v in row)+" |" for row in rows]])


def report():
    p,r,u,(tr,va)=inputs()
    names=u.set_index("code").name.to_dict()
    names["000001"]="上证指数"
    dates=r.index[va:]
    f=load("all_pair_results")
    directional=f[f.method!="mr"]
    tops={k:load("top_"+k+"_pairs") for k in ("return","sharpe","robust")}
    win=tops["return"].iloc[0]; lock=pick(f)
    bench=directional.query("a=='000001' and b=='000300' and spec=='static'").iloc[0]
    roles={"locked":lock,"test_best":win,"benchmark":bench}
    cmp=load("hedge_method_comparison")
    bymethod=cmp[cmp.method!="mr"].sort_values("valid_sharpe",ascending=False).drop_duplicates("method")
    costs=load("cost_sensitivity")
    reg=load("pair_test_regimes")
    boot=load("pair_bootstrap"); exposures=load("pair_factor_exposures")
    wf=load("walk_forward_results"); wfd=load("walk_forward_daily"); wfm=metrics(wfd.ret)
    wftr=load("walk_forward_trials")
    wfrank=(wftr.assign(log_growth=np.log1p(wftr.total)).groupby(["a","b","spec"]).log_growth.sum()
            .sort_values(ascending=False))
    a,b,s=wfrank.index[0]
    hist=load("pair_historical_splits"); legacy=load("pair_legacy_check")
    test=load("pair_search_max_test").iloc[0]
    mr=f[f.method=="mr"]; gates=load("pair_rolling_cointegration")
    lev=load("pair_leverage"); fr=load("pair_rebalance_control")
    def label(q):
        return names[q.a]+" / "+names[q.b]
    def how(q):
        return {"one":"等金额","static":"训练期静态OLS"}.get(q.spec,
               f"{METHOD[q.method]} {q.window:g} · {FREQ[q.freq]}更")
    def sub(df,q):
        return df[(df.a==q.a)&(df.b==q.b)&(df.spec==q.spec)]
    def cost(q,bp=5,borrow=.02):
        return sub(costs,q).query("bp==@bp and borrow==@borrow").iloc[0]
    @lru_cache(maxsize=16)
    def curve(a,b):
        return paths(a,b)
    def series(q):
        z=curve(q.a,q.b); k=list(z["ids"]).index(q.spec)
        return z["ret"][k,va:]
    def nav(s):
        return pd.Series(np.r_[1,(1+np.asarray(s)).cumprod()],index=p.index[va:])
    def fig(title,xlabel="",ylabel=""):
        g,ax=plt.subplots(figsize=(10,4.9),layout="constrained")
        ax.set_title(title,loc="left",fontweight="bold",pad=12)
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
        return g,ax
    figures=[]
    def save(g,num,name):
        file=f"pair_search_{num:02}_{name}"
        figure(g,file);figures.append(file)
    g,ax=fig("测试期事后收益前十个多空方向",ylabel="净值 · 起点为1")
    for q in tops["return"].head(10).itertuples():
        ax.plot(nav(series(q)),lw=1.1,label=label(q))
    ax.legend(ncol=2,fontsize=8);save(g,1,"top_curves")
    for num,key,field,title in [(2,"return","test_annual","测试期年化收益前十"),(3,"sharpe","test_sharpe","测试期夏普前十")]:
        g,ax=fig(title,xlabel="年化收益" if num==2 else "年化夏普（无风险率为零）")
        top=tops[key].head(10).iloc[::-1]
        ax.barh([label(q) for q in top.itertuples()],top[field],color="#24796c")
        if num==2: ax.xaxis.set_major_formatter(PercentFormatter(1))
        save(g,num,key)
    g,ax=fig("测试收益冠军的两条价格指数",ylabel="归一化点位")
    for c in (win.a,win.b):
        s=p[c].iloc[va:];ax.plot(s/s.iloc[0],label=names[c])
    ax.legend();save(g,4,"legs")
    g,ax=fig("同一测试窗口的三条策略净值",ylabel="1倍总敞口净值")
    for k,q in roles.items(): ax.plot(nav(series(q)),label=ROLES[k])
    ax.legend();save(g,5,"hedged")
    g,ax=fig("固定验证锁定配对，改变对冲方法",ylabel="净值")
    for q in bymethod.itertuples(): ax.plot(nav(series(q)),label=how(q))
    ax.legend(fontsize=9);save(g,6,"methods")
    g,ax=fig("测试期各市场状态的条件年化收益",ylabel="年化收益")
    rg=pd.concat([sub(reg,q).set_index("state").annual.rename(ROLES[k]) for k,q in roles.items()],axis=1)
    rg.rename(index=STATE).plot.bar(ax=ax,rot=0);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("");save(g,7,"regimes")
    g,ax=fig("相同方向与参数的训练、验证和测试收益",ylabel="年化收益")
    pd.DataFrame({ROLES[k]:[q.train_annual,q.valid_annual,q.test_annual] for k,q in roles.items()},
                 index=["训练","验证","测试"]).plot.bar(ax=ax,rot=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1));save(g,8,"splits")
    g,ax=fig("验证锁定策略的实际成交成本敏感性",ylabel="净值")
    ledger=load("pair_selected_ledger")
    for (bp,borrow),d in ledger[ledger.role=="locked"].groupby(["bp","borrow"]):
        ax.plot(nav(d.ret),label=f"单边{bp}bp · 借券{borrow:.0%}")
    ax.legend(fontsize=9);save(g,9,"cost")
    g,ax=fig("逐年重新选择后，八个年度窗口有赚有亏",ylabel="当年累计毛收益")
    ax.bar(wf.year,wf.total,color=np.where(wf.total>=0,"#24796c","#b75b40"))
    ax.axhline(0,color="#687377",lw=.7);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("2026年截至11月17日");save(g,10,"walk")
    champion=sub(ledger[ledger.role=="test_best"],win)
    g,(ax,ddax)=plt.subplots(2,1,sharex=True,figsize=(10,6.0),
                              height_ratios=[2.5,1],layout="constrained")
    g.suptitle(f"测试期收益冠军　｜　做多{names[win.a]}，做空{names[win.b]}",
               fontsize=17,fontweight="bold")
    ax.set_title(f"{dates[0]:%Y.%m.%d} — {dates[-1]:%Y.%m.%d}　·　{how(win)}　·　总敞口 1 倍",
                 fontsize=10,loc="left")
    for bp,br,title,color in [(0,0,"毛收益","#087d71"),(5,.02,"成本后","#ad502b")]:
        holding=champion[(champion.bp==bp)&(champion.borrow==br)].sort_values("date")
        np.testing.assert_array_equal(pd.to_datetime(holding.date).to_numpy(),dates.to_numpy())
        path=nav(holding.ret)
        np.testing.assert_allclose(path.iloc[1:],holding.nav,atol=1e-8)
        dd=path/path.cummax()-1
        ax.plot(path,color=color,label=f"{title}　期末 {path.iloc[-1]:.4f}（{path.iloc[-1]-1:+.2%}）")
        ddax.plot(dd,color=color,lw=1.1,label=f"{title}最大回撤 {dd.min():.2%}")
    ax.axhline(1,color="#808b89",lw=.8,ls="--")
    ax.set_ylabel("模拟净值（起始为 1）")
    ddax.set_ylabel("距前高回撤")
    ddax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.legend(loc="upper left",frameon=False);ddax.legend(loc="lower left",ncol=2,frameon=True,fontsize=9)
    ddax.set_xlabel("成本情景按实际成交额单边收取 5 bp，空头年化借券费为 2%")
    save(g,11,"champion_nav")
    def chart(i,title):
        return f"![{title}](../outputs/figures/{figures[i-1]}.png)"
    def perf(q):
        return [label(q),how(q),f"{q.test_mean_h:.3f}",f"{q.test_total:.2%}",f"{q.test_annual:.2%}",
                f"{q.test_vol:.2%}",f"{q.test_sharpe:.2f}",f"{q.test_mdd:.2%}",f"{cost(q).total:.2%}"]
    headings=["角色","多头 / 空头","方法","平均 h","累计","年化","波动","夏普","最大回撤","成本后累计¹"]
    data=curve(lock.a,lock.b); idx=list(data["ids"]).index(lock.spec)
    ratio=-data["v"][idx,va:]/data["w"][idx,va:]
    selection=pd.DataFrame([dict(role=k,a=q.a,b=q.b,spec=q.spec,**{t:float(q[t]) for t in
                       ["train_annual","valid_sharpe","test_total","test_annual","test_vol","test_sharpe","test_mdd"]})
                       for k,q in roles.items()])
    csv(selection,"pair_selection")
    note=["# A股宽基指数两两配对交易研究","",
          f"**{dates[0]:%Y-%m-%d} 至 {dates[-1]:%Y-%m-%d} 的留出测试中，验证期锁定的{label(lock)}累计毛收益 {lock.test_total:.2%}，计入单边5bp及年化2%借券假设后为 {cost(lock).total:.2%}。** 测试期回看可找到累计赚 {win.test_total:.2%} 的{label(win)}。独立按年重复选对后，累计毛收益为 {wfm['total']:.2%}。结合成本、统计证据与空头工具缺口，暂不支持进入以收益为考核目标的模拟基金阶段。",
          "","## 测试期收益冠军的模拟净值","",chart(11,"冠军多空策略毛净值、成本后净值与回撤"),
          "","## 冻结选择与历史冠军","",tab(headings,[[ROLES[k],*perf(q)] for k,q in roles.items()]),
          "","¹ 两腿实际买卖金额分别收费，计入进入与退出；空头按期初市值和实际日历天数/365计息。各表除注明外均为价格指数毛收益，现金利息和无风险率设为零。h 为每1元多头对应的空头金额。动态方法的平均 h 仅用于描述，不能作为每日固定值。",
          "",chart(5,"验证锁定、测试冠军与基准净值"),"","## 候选池与研究设计","",
          f"项目库有{len(u)}条具有日线的指数，均属于本任务指定的宽基。覆盖 {p.index[0]:%Y-%m-%d} 至 {p.index[-1]:%Y-%m-%d}，每条 {len(p):,} 个共同收盘点、{len(r):,} 个收益日，缺失率均为零。本次搜索边界是已入库的宽基。无已入库行业或主题指数。逐指数起止、类别、纳入状态见[候选池](../outputs/tables/index_universe.csv)。",
          "",tab(["阶段","首个收益日","末个收益日","天数"],[[k,f"{idx[0]:%Y-%m-%d}",f"{idx[-1]:%Y-%m-%d}",len(idx)] for k,idx in [("训练",r.index[:tr]),("验证",r.index[tr:va]),("测试",dates)]]),
          "",f"先计算15个无序配对、30个方向的基础指标，再在每个方向测试52个配置。固定方向有{len(directional):,}种、双向均值回归{len(mr)}种，合计{len(f):,}。训练夏普必须为正、验证期持仓日占比至少50%，按验证夏普选一组，保留最后一个验证收盘作为建仓时间。测试期的冠军、前三张榜单与全期冠军仅作回顾。该项目此前已使用过部分测试年份，时间切分可以约束本程序的信息流，不能恢复研究者从未看过的历史。",
          "", "固定方向的对冲网格包括等金额、训练期含截距的固定OLS、60/120/252/504日滚动OLS、衰减系数0.94/0.97/0.99的EWMA、四个同窗口的波动率比。后三种方法均按日/周/月/季更新 h，EWMA至少60日预热。全日线在同一日历上评价，预热或无效比例时持现；训练期包含冷启动及静态OLS样本内拟合。动态策略持有 t−1 至 t 的头寸使用截至 t−2 收盘的信息，静态OLS保留最后一个训练交易日作为执行缓冲。",
          "", "主实验每天将总名义归一为1倍净资产，多头权重1/(1+h)，空头权重−h/(1+h)。周/月/季指 h 的更新频率；下文另测期间完全不交易、允许权重漂移。相同的毛敞口不保证对市场 Beta 中性。逐日实际金额按初始资本1元记录，乘以初始资金即可换算；净敞口为两腿带符号权重之和。现金多头全额占用为多头权重，另以空头名义100%作为抵押压力示例，两者之和为名义占用，实际保证金依工具另核。原始口径 rA−h×rB 保存在逐日档及全结果表，活跃时原始总敞口1+h。",
          "","## 收益由什么决定","",
          f"**配对选择。** 固定1∶1后，30个方向的测试年化介于 {directional[directional.spec=='one'].test_annual.min():.2%} 和 {directional[directional.spec=='one'].test_annual.max():.2%}。下图每个方向取该方向事后年化最高的配置，因此包含方向内选优。两条指数原始曲线从测试建仓收盘归一为1。",
          "",chart(1,"十个方向的净值"),"",chart(4,"测试冠军标的"),"",
          f"**对冲比例。** 锁定策略的60日波动率比在测试期介于 {ratio.min():.3f} 和 {ratio.max():.3f}，合计平均多头权重{lock.test_long_weight:.2%}、空头权重{lock.test_short_weight:.2%}、净敞口{lock.test_net:.2%}。下面固定这一配对，每个方法用验证夏普最高的规格进行描述性对照。",
          "",tab(["方法","验证夏普","测试累计","测试波动","测试回撤"],[[how(q),f"{q.valid_sharpe:.2f}",f"{q.test_total:.2%}",f"{q.test_vol:.2%}",f"{q.test_mdd:.2%}"] for q in bymethod.itertuples()]),"",chart(6,"方法比较"),
          "","**估计窗口。** 固定配对、每日更新滚动OLS，各窗口的测试表现如下。","",
          tab(["窗口","平均 h","累计收益","年化波动"],[[int(q.window),f"{q.test_mean_h:.3f}",f"{q.test_total:.2%}",f"{q.test_vol:.2%}"] for q in cmp.query("method=='rolling' and freq=='D'").itertuples()]),
          "","**实际调仓。** 固定同一配对和120日滚动OLS，均按5bp与2%借券假设计费。持股数模式只在规定频率交易，期间总敞口随价格漂移。成交额为每1元初始资产在全测试期的双边累计成交金额。","",
          tab(["频率","净值每日重置年化","仅调仓日交易年化","仅调仓日成交额"],[[FREQ[k],f"{d.iloc[0].annual:.2%}",f"{d.iloc[1].annual:.2%}",f"{d.iloc[1].traded:.2f}"] for k,d in fr.groupby("freq",sort=False)]),
          "","## 市场状态、历史反转与风险暴露","",
          "沿用既有上证指数状态规则，仅用前一日可见的120日累计收益和均线。趋势超过±10%且位于相应均线一侧分别划为主升/下跌，累计收益在±5%为震荡，其余为过渡。下表按状态内的收益日复利年化；短状态年化可能很大，不能视为可连续持有一整年的收益。",
          "",tab(["测试状态","日数",*ROLES.values()],[[STATE[k],int(sub(reg,lock).set_index("state").loc[k,"n"]),*[f"{v:.2%}" for v in row]] for k,row in rg.iterrows()]),"",chart(7,"市场状态"),"",
          tab(["既定方向历史切片","2018—2020","2021—2024","2025—2026-11-17"],[[ROLES[k],*[f"{z:.2%}" for z in hist[hist.role==k].total]] for k in roles]),
          "","切片间重叠于主实验的训练、验证或测试段，仅作归因，其中静态基准回看训练前段使用了训练内估计。按原实验独立复算，2021—2024年 h=0.819943，2025年以来原始累计+5.56%、1倍总敞口+3.07%；2015—2017年估计 h=0.973105，用于2018—2020年同方向分别为−19.18%及−10.17%。",
          "",chart(8,"阶段对比"),"",
          tab(["组合","上证 Beta","中证全指 Beta","沪深300 Beta","CH4规模暴露"],[[ROLES[k],*[f"{exposures[(exposures.role==k)&(exposures.term==term)].iloc[0].beta:.3f}" for term in ["000001","000985","000300","SMB"]]] for k in roles]),
          "","三条市场基准分别回归，CH4联合回归包含市场、规模、价值及PMO，因子覆盖至2026-08-31。系数、HAC标准误、置信区间及样本日数见[风险暴露](../outputs/tables/pair_factor_exposures.csv)。",
          "","## 检验强度与赢家偏差","",
          tab(["对象","算术年化均值","HAC年化标准误","双侧 p","20日区块95%区间","去掉最好5日累计"],[[ROLES[q.role],f"{q.annual_mean:.2%}",f"{q.hac_se:.2%}",f"{q.hac_p:.3f}",f"[{q.ci_low:.2%}, {q.ci_high:.2%}]",f"{q.remove_best5_total:.2%}"] for q in boot[boot.series=="return"].itertuples()]),
          "",f"日度Newey–West取5阶。对全部{len(f):,}个均值检验进行Holm与BH校正，5%水平分别通过{sum(f.test_holm_p<.05)}和{sum(f.test_bh_q<.05)}个。保留策略间同期相关性的共同日期循环区块Bootstrap，20日为块、固定种子、2,000次。去均值后比较全网格最大单侧HAC t统计量，非恒定序列{int(test.nonconstant)}条，调整后 p={test.max_t_p:.3f}。该检验直接处理搜索最高收益的机会，不假设1,560条策略彼此独立。",
          "",tab(["配对减去同期原基准","年化平均差","HAC p","配对Bootstrap 95%区间"],[[ROLES[q.role],f"{q.annual_mean:.2%}",f"{q.hac_p:.3f}",f"[{q.ci_low:.2%}, {q.ci_high:.2%}]"] for q in boot[(boot.series=="minus_benchmark")&(boot.role!="benchmark")].itertuples()]),
          "","## 均值回归独立实验","",
          f"每季使用此前504日对数点位，分别检查两腿水平ADF p>0.05、一阶差分ADF p<0.05、带常数Engle–Granger、残差ADF和AR(1)半衰期。季度内对30个方向EG检验作BH校正，要求 q<0.05、残差ADF p<0.05、正价格斜率且半衰期1—126日。{len(gates)}个配对季度中通过{int(gates.pass_gate.sum())}个。上季模型在新季度重置，以最近可用收盘计算Z，±1.5/2/2.5入场，回归0或同侧0.5出场，同时按4倍标准差止损、最长60日。价差过高做空第一腿、过低做多，反向时抵押与借券跟随实际空头。",
          "",f"验证期全部{len(mr)}个阈值配置的持仓率最大为{mr.valid_active.max():.1%}，无法据此选择阈值，冻结的均值回归决策是持现，测试收益为0。事后网格中有{(mr.test_active>0).sum()}个配置在测试期交易，最高测试年化{mr.test_annual.max():.2%}，不作为已选策略。见[滚动检验明细](../outputs/tables/pair_rolling_cointegration.csv)。",
          "","## 成本与杠杆","",
          "以下为锁定策略累计净收益，列为空头年化借券率，行为单边成交费。全部1,560种策略的25种组合均已运行。","",
          tab(["单边费","0%","1%","2%","3%","5%"],[[f"{bp} bp",*[f"{cost(lock,bp,br).total:.2%}" for br in (0,.01,.02,.03,.05)]] for bp in (0,2,5,10,20)]),"",chart(9,"成本曲线"),"",
          tab(["锁定策略总敞口","测试累计毛收益","波动","最大回撤"],[[f"{q.target_gross:g}倍",f"{q.total:.2%}",f"{q.vol:.2%}",f"{q.mdd:.2%}"] for q in sub(lev,lock).itertuples()]),
          "","## 逐年向前检验","",
          "每轮前三个日历年训练、下一年验证、再下一年交易。静态Beta逐轮重估，动态估计只读当时可得历史；最后一个验证交易日留作成交。按与主实验相同的训练盈利门槛和验证夏普规则选固定方向配置，不用当年收益修改选择。年度间按净值连续复利，表列毛收益。",
          "",tab(["交易年","多头 / 空头","冻结规格","验证夏普","当年毛收益","成本后收益¹","当年毛回撤"],[[q.year,label(q),q.spec,f"{q.valid_sharpe:.2f}",f"{q.total:.2%}",f"{q.net_total_5bp_2pct:.2%}",f"{q.mdd:.2%}"] for q in wf.itertuples()]),
          "",f"年度切换在同一账户内按六条指数实际持仓净额交易。按单边5bp、年化2%借券，连续持有的净值由1降至{wfd.net_nav.iloc[-1]:.4f}，累计成本后收益{wfd.net_nav.iloc[-1]-1:.2%}。",
          "",f"{len(wf)}段中{(wf.total>0).sum()}段盈利，累计{wfm['total']:.2%}、年化{wfm['annual']:.2%}、夏普{wfm['sharpe']:.2f}、最大回撤{wfm['mdd']:.2%}。若事后在各年度均跟踪同一个规格，跨年度累计最高的是{names[a]} / {names[b]}（{s}，{np.expm1(wfrank.iloc[0]):.2%}）；年度交易策略没有事先锁定这一规格。",
          "",chart(10,"逐年测试"),"","## 榜单与可复算材料","",
          "各阶段冠军以下按年化收益排列，验证锁定另按预设风险调整规则。三张前20名榜以多空方向为单位，同方向仅保留该榜指标最佳的规格。综合榜取测试年化、回撤、夏普、测试覆盖日历年盈利比例、最差状态年化、训练测试年化差绝对值的反向排名，六项百分位等权。首尾非完整年度按实际覆盖计算。",
          "",tab(["阶段（回顾）","多头 / 空头","规格","该阶段年化"],[[n,label(q),q.spec,f"{q[col]:.2%}"] for n,col in [("训练","train_annual"),("验证","valid_annual"),("最终测试","test_annual"),("全样本事后","full_annual")] for q in [directional.nlargest(1,col).iloc[0]]]),
          "",chart(2,"收益排行榜"),"",chart(3,"夏普排行榜")]
    for key,title in [("return","绝对收益"),("sharpe","夏普"),("robust","综合稳定性")]:
        note+=["",f"### {title}前20个方向","",tab(["名次","多头 / 空头","规格","测试年化","夏普","回撤","综合分"],[[i,label(q),q.spec,f"{q.test_annual:.2%}",f"{q.test_sharpe:.2f}",f"{q.test_mdd:.2%}",f"{q.robust_score:.3f}"] for i,q in enumerate(tops[key].itertuples(),1)])]
    note+=["","## 结论与实际工具","",
           f"在这六条宽基中，收益冠军的多数收益集中在测试前段，2025年以来累计只有{hist[hist.role=='test_best'].iloc[-1].total:.2%}。验证锁定策略在同一段为{hist[hist.role=='locked'].iloc[-1].total:.2%}。两者均未显示显著优于等敞口原基准。年度重复选择与成本压力进一步削弱了将其作为模拟基金收益策略的依据。",
           "","两组共同的上证100空头尚缺已核实的直接工具和逐日券源。另行检验了三只场内ETF与IF单月合约，原配对可观察到工具收益、流动性及换月基差。[实际可交易性与工具收益报告](pair_trading_implementation.md)",
           "","**复算入口**　在项目根目录运行 `python scripts/run_pair_search.py`。工具报告可单独运行 `python scripts/pair_instruments.py`，优先读取已缓存的ETF与期货快照。SQLite全程以只读方式打开。指数收盘取AKShare/东方财富项目归档，无全收益、分红与成分股复权重建。",
           "","**主要数据**　[全参数结果](../outputs/tables/all_pair_results.csv) · [基础配对统计](../outputs/tables/pair_basic_statistics.csv) · [固定配对方法对照](../outputs/tables/hedge_method_comparison.csv) · [成本压力测试](../outputs/tables/cost_sensitivity.csv) · [年度逐候选记录](../outputs/tables/walk_forward_trials.csv) · [持仓与成交明细](../outputs/tables/pair_selected_ledger.csv)。",
           "","每个有向配对的完整逐日档位于 outputs/pair_paths/A_B.npz。date 为收益日期，ids 为行顺序；w、v 为两腿签名权重，ret 为1倍毛收益，raw_ret 为原始敞口毛收益，traded 为每1元初始资本实际买卖额（首笔开仓、末日含平仓）。逐日净值为 (1+ret) 累乘。all_pair_results 的 turnover 是沿完整路径的日均成交额/期初净资产，单独成本情景则在测试起点从空仓开始。"]
    (REPORTS/"pair_trading_full_research.md").write_text("\n".join(note)+"\n",encoding="utf-8")


if __name__=="__main__":
    report()
