"""Reproduce the specified price-index trade and write the PDF, chart and audit tables."""
from base64 import b64encode
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib.ticker import PercentFormatter
from playwright.sync_api import sync_playwright
from statsmodels.tsa.stattools import coint

from research_common import ROOT, FIGURES, REPORTS, csv, db, drawdown, fit, market, np, pd, plt, sm

L, S = "000001", "000300"
NAMES = {L: "上证指数", S: "沪深300", "000903": "中证A100", "000132": "上证100", "399311": "国证1000"}
PCT = lambda x: f"{x:.2%}"
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})


def beta(r, long=L, short=S):
    return r[long].cov(r[short]) / r[short].var()


def performance(s):
    nav = (1 + s).prod()
    ann = nav ** (252 / len(s)) - 1
    vol = s.std() * np.sqrt(252)
    return dict(n=len(s), cumulative=nav-1, annual=ann, volatility=vol,
                sharpe=s.mean()*252/vol, geometric_ratio=ann/vol, max_dd=drawdown(s))


def spread_test(p):
    logp = np.log(p)
    model = sm.OLS(logp[L], sm.add_constant(logp[S])).fit()
    z = model.resid
    k = sm.OLS(z.diff().iloc[1:], sm.add_constant(z.shift().iloc[1:])).fit().params.iloc[1]
    test = coint(logp[L], logp[S], trend="c", autolag="aic")
    return model.params.iloc[1], k, 1+k, np.log(.5)/np.log1p(k), test[1]


def table(headers, rows):
    return ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row)
            + "</tr>" for row in rows) + "</tbody></table>")


def chart(p, nav, b):
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.9), sharex=True, height_ratios=(1, 1))
    for ax, series, labels in [
        (axes[0], [p[L]/p[L].iloc[0], p[S]/p[S].iloc[0]], ["上证指数", "沪深300"]),
        (axes[1], [nav["beta_nav"], nav["one_nav"]], [f"固定 Beta 对冲（{b:.3f}）", "等金额对冲（1∶1）"])]:
        ax.axvspan(pd.Timestamp("2024-12-31"), p.index[-1], color="#e2f1ed", alpha=.7)
        ax.axhline(1, color="#899491", lw=.7)
        for s, label, color in zip(series, labels, ["#10675e", "#a44b27"]):
            ax.plot(s, label=label, color=color, lw=1.3)
            ax.annotate(f"{s.iloc[-1]:.3f}", (s.index[-1],s.iloc[-1]), xytext=(6,0),
                        textcoords="offset points", color=color, va="center", fontsize=9)
        ax.margins(x=0)
        ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
        ax.spines[["left","bottom"]].set_visible(False)
    axes[0].set_ylabel("价格指数 · 起点为 1")
    axes[1].set_ylabel("策略毛净值 · 起点为 1")
    axes[1].set_xlabel("浅绿色为检验期（2025-01-02 起）")
    fig.subplots_adjust(left=.085, right=.93, top=.97, bottom=.11, hspace=.14)
    dest = FIGURES / "sse_csi300_replication.png"
    fig.savefig(dest, dpi=165, facecolor="white")
    plt.close(fig)
    return dest


def main():
    closes, historical_returns, _ = market()
    p = closes.loc["2021-01-04":"2026-11-17", list(NAMES)]
    assert p.index.is_unique and p.notna().all().all() and len(p) == 1422
    # Compare every sampled close with the CSV ingested into SQLite.
    with db() as conn:
        for code in NAMES:
            paths = conn.execute("SELECT DISTINCT b.raw_file FROM index_daily d JOIN ingest_batch b "
                                 "ON d.batch_id=b.batch_id WHERE d.index_code=? AND d.provider='eastmoney' "
                                 "AND d.trade_date BETWEEN '2021-01-04' AND '2026-11-17'", (code,)).fetchall()
            assert len(paths) == 1
            raw = pd.read_csv(ROOT / paths[0][0], parse_dates=["date"]).set_index("date")
            np.testing.assert_array_equal(p[code], raw.loc[p.index, "close"])
    r = p.pct_change(fill_method=None).iloc[1:]
    train = r.loc[:"2024"]
    b = beta(train)
    one, hedged = r[L]-r[S], r[L]-b*r[S]
    out = hedged.loc["2025":]
    daily = p.add_prefix("close_").join(r.add_prefix("return_"))
    for key,s in [("beta",hedged),("one",one)]:
        daily[key+"_return"] = s
        daily[key+"_nav"] = (1+daily[key+"_return"].fillna(0)).cumprod()
    csv(daily, "sse_csi300_replication_daily", index=True)
    periods = {"训练期": slice(None,"2024"), "2025年": "2025", "2026年至11月17日": "2026",
               "检验期合计": slice("2025",None), "全期": slice(None)}
    stats = pd.DataFrame([{"strategy":name, "period":key, **performance(s.loc[cut])}
                          for name,s in [("Beta",hedged),("1:1",one)] for key,cut in periods.items()])
    csv(stats, "sse_csi300_replication_performance")
    g = stats[stats.strategy=="Beta"].set_index("period")
    base = stats[stats.strategy=="1:1"].iloc[-1]
    prices_tr = p.loc[:"2024"]
    gains = prices_tr.iloc[-1] / p.iloc[0] - 1
    pairs = []
    for a,c in combinations(NAMES, 2):
        i,j = sorted((a,c),key=gains.get,reverse=True)
        h = beta(train,i,j)
        test = (r[i]-h*r[j]).loc["2025":]
        pairs.append(dict(long=i,short=j,train_corr=train[i].corr(train[j]),full_corr=r[i].corr(r[j]),
                          train_gap=gains[i]-gains[j],train_beta=h,oos_annual=performance(test)["annual"]))
    pairs = pd.DataFrame(pairs).sort_values("train_gap",ascending=False)
    csv(pairs, "sse_csi300_replication_pairs")
    eligible = pairs.query("train_corr >= 0.90")
    mean = sm.OLS(out, np.ones((len(out),1))).fit(cov_type="HAC", cov_kwds={"maxlags":5})
    ci = mean.conf_int().iloc[0]*252
    exp = fit(out,r.loc[out.index,S])
    earlier = historical_returns.loc["2018":"2020"]
    early_b = beta(historical_returns.loc["2015":"2017"])
    early_perf = performance(earlier[L]-early_b*earlier[S])
    early_one = performance(earlier[L]-earlier[S])
    spread = spread_test(p)
    spread_tr = spread_test(prices_tr)
    rel = (p[L]/p[L].iloc[0] > p[S]/p[S].iloc[0]).iloc[1:]
    image = chart(p,daily,b)
    image_uri = "data:image/png;base64,"+b64encode(image.read_bytes()).decode()
    perf_table = table(["固定 Beta","收益日","累计","年化","年化波动","标准夏普","最大回撤"],
                       [[i,int(q.n),PCT(q.cumulative),PCT(q.annual),PCT(q.volatility),
                         f"{q.sharpe:.2f}",PCT(q.max_dd)] for i,q in g.iterrows()])
    ranks = table(["训练期相对强 / 弱","训练期相关","训练期收益差¹","全期相关","检验期年化²"],
                  [[NAMES[q.long]+" / "+NAMES[q.short],f"{q.train_corr:.3f}",
                    f"{100*q.train_gap:.2f}",f"{q.full_corr:.3f}",PCT(q.oos_annual)] for q in pairs.itertuples()])
    levels = p.iloc[-1]/p.iloc[0]-1
    corr = r[L].corr(r[S])
    ddnav = pd.concat([pd.Series([1.],index=[prices_tr.index[-1]]),(1+out).cumprod()])
    dd = ddnav/ddnav.cummax()-1
    trough = dd.idxmin()
    peak = ddnav.loc[:trough].idxmax()
    costs = table(["年化总拖累假设","0%","1%","2%","3%","4%"],[["检验期累计收益"]+
                  [PCT(((1+out)*(1-c)**(1/252)).prod()-1) for c in [0,.01,.02,.03,.04]]])
    be = 1-(1+g.loc["检验期合计","annual"])**-1
    ratios = "、".join(f"{v:.2f}" for v in g.iloc[:4].geometric_ratio)
    navpath = " → ".join(f"{x:.4f}" for x in [1,*daily.groupby(daily.index.year).last().loc[2024:,"beta_nav"]])
    summary = f"""<h1>上证指数做多、沪深300做空</h1><p class="lead">收益曲线复现报告　｜　2021年1月4日至2026年11月17日</p>
    <h2>5.56%的检验期毛收益可以复现</h2>
    <p>按2024年底估计的比例，每1元净资产做多1元上证指数、做空{b:.3f}元沪深300，每日恢复目标权重。此后453个收益日，1元增至{1+g.loc["检验期合计","cumulative"]:.4f}元，年化波动4.88%，最大回撤8.00%。更早的2018—2020年，同方向、同样先估计再固定比例，累计亏损{abs(early_perf['cumulative']):.2%}。</p>
    <img src="{image_uri}" alt="两条价格指数及两种配对策略净值曲线">
    <p class="note">图1　上图为同起点的价格指数，下图按日将多空收益差复利。固定 Beta 在白色区间用该区间全部数据拟合，浅绿色区间才使用事前估计的比例。两幅图纵轴刻度不同。</p>
    <p>从首日收盘后算起，上证指数量值领先沪深300的天数占{rel.mean():.2%}。同期策略日收益为正的天数仅占{hedged.gt(0).mean():.2%}。只要历史上积累了领先优势，即便当天跑输，归一化曲线也可以继续在上方。</p>"""
    method = f"""<h1>原文数字逐项核对</h1>
    <p>五条指数均有1,422个收盘点。首日用来建仓，从次日开始得到1,421个日收益。2025年及以后的年度收益均包含跨年首个交易日的涨跌。</p>
    {table(["指标","原文","按收盘点复算"],[
        ["上证指数 / 沪深300累计价格收益","+2.33% / −24.51%",f"{PCT(levels[L])} / {PCT(levels[S])}"],
        ["日收益相关 / 含截距回归 R²","0.945 / 0.894",f"{corr:.5f} / {corr**2:.5f}"],
        ["1∶1 全期累计 / 年化 / 最大回撤","31.00% / 4.91% / −8.81%",f"{PCT(base.cumulative)} / {PCT(base.annual)} / {PCT(base.max_dd)}"],
        ["训练期日收益 Beta","0.820",f"{b:.6f}"],
        ["国证1000价格收益 / 与沪深300相关","−19.04% / 0.980",f"{PCT(levels['399311'])} / {r['399311'].corr(r[S]):.5f}"]])}
    <h2>固定 Beta 的分期结果</h2>{perf_table}
    <p>净值依次为 <strong>{navpath}</strong>（建仓、2024年末、2025年末、实验末日）。检验期回撤高点在{peak:%Y-%m-%d}，低点在{trough:%Y-%m-%d}。同期上证指数与沪深300的年化波动分别为{r.loc[out.index,L].std()*np.sqrt(252):.2%}、{r.loc[out.index,S].std()*np.sqrt(252):.2%}。</p>
    <h2>夏普比率有一处口径差异</h2>
    <p>年化收益按净值增长倍数的252/N次方减1，年化波动按日收益样本标准差乘√252。无风险收益设为零时，标准夏普采用日均收益乘252再除以年化波动。原文四行夏普用“几何年化收益 ÷ 年化波动”可以复现为{ratios}，两种算法应明确区分。</p>
    <p>1∶1 全期末，两条价格曲线的端点收益相减为{(levels[L]-levels[S])*100:.2f}个百分点，对应固定初始名义头寸。31.00%对应每日调仓后的复利策略。Beta 策略初始总敞口为{1+b:.3f}倍，净多头为{1-b:.3f}倍，收益分母均为基金净资产。回归包含截距，建仓时只使用斜率。</p>"""
    selection = f"""<h1>只看到2024年底，会选中谁？</h1>
    <p>原文没有给出唯一的筛选规则。为核验选对是否依赖事后判断，本次设置一条可重复的规则：先要求训练期日收益相关系数至少0.90，再按同起点的累计价格收益差从大到小排序，做多强者。下表展示全部十组，包括低于门槛的配对。</p>
    {ranks}<p class="note">¹ 单位为百分点。² 每一组都仅用训练期日收益估计自己的 Beta，检验期固定多空方向与比例；检验期一列仅用于评价。</p>
    <p>门槛内共有{len(eligible)}组。排在首位的是上证指数 / 中证A100，检验期年化{eligible.iloc[0].oos_annual:.2%}。上证指数 / 沪深300排第{eligible['short'].tolist().index(S)+1}，检验期年化3.06%。国证1000 / 沪深300按同一算法得到1.23%，与原文吻合。门槛和排序规则本次才明确，因此这项检验不能还原当时的选对决定。</p>
    <h2>同一交易方向在较早时期发生反转</h2>
    <p>另取库中2015—2017年估计 Beta，固定为{early_b:.4f}，从2017年末收盘持有到2020年末。三年累计{early_perf['cumulative']:.2%}，最大回撤{early_perf['max_dd']:.2%}。同期1∶1组合累计{early_one['cumulative']:.2%}。即使不改变指数、多空方向和估计方法，换一段市场历史也会承担持续亏损。</p>"""
    evidence = f"""<h1>对冲掉了多少风险？</h1>
    <p>检验期对冲后的日收益再次回归沪深300，剩余 Beta 为{exp.params[1]:.3f}，95%区间为[{exp.conf_int()[1,0]:.3f}, {exp.conf_int()[1,1]:.3f}]。在这段历史中，对沪深300日涨跌的线性敏感度接近零；行业、规模等其他暴露仍可能贡献盈亏。</p>
    <p>检验期算术年化均值为{out.mean()*252:.2%}，95%区间为[{ci.iloc[0]:.2%}, {ci.iloc[1]:.2%}]，检验均值等于零的双侧 p 值为{mean.pvalues.iloc[0]:.3f}。均值和剩余 Beta 均使用5阶 Newey–West 标准误。区间未校正配对筛选，现有观察不足以确认长期正期望。</p>
    <h2>价差的272日半衰期如何理解</h2>
    {table(["拟合窗口","价格斜率","差分系数 κ","水平系数 φ","半衰期","协整检验 p"],
    [[n,f"{h:.3f}",f"{k:.5f}",f"{phi:.5f}",f"{hl:.1f}日",f"{pv:.4f}"] for n,(h,k,phi,hl,pv) in [("训练期",spread_tr),("全期",spread)]])}
    <p>先将上证指数对数点位对沪深300对数点位做含截距回归，再用价差的一阶滞后解释价差变化。原文的−0.00254是差分系数 κ；水平 AR(1) 系数 φ = 1 + κ。半衰期按 log(0.5) / log(φ) 计算。Engle–Granger 双步检验含常数、按 AIC 选滞后，全期无法拒绝无协整，训练期结果接近5%边界。272日只是一项依赖窗口的点估计。</p>
    <h2>多少成本会吃光检验期收益？</h2>
    {costs}<p>情景把所有摩擦合并为占净资产的固定年化拖累 c，每交易日净值额外乘 (1−c) 的1/252次方。盈亏平衡拖累约为{be:.2%}/年。费率为假设；本实验的价格指数不含现金分红，现金和融资产生的利息均按零计。ETF分红与跟踪偏离、期货基差和展期、保证金现金收益、交易费用仍需要用工具实盘历史核算。</p>
    <h2>数据与复算</h2>
    <p class="note">来源为项目 SQLite 的 index_daily 表，AKShare 采集东方财富日线，五条序列已逐日对照归档 CSV。公开点位抽查包括上证指数首日3502.96、2025年末上证3968.84及沪深300为4629.94、末日分别为3584.58和3976.46。仅比对指定日期。<a href="https://www.sse.com.cn/aboutus/publication/monthly/documents/c/10061129/files/2d35673d99d847c0a12ed0432505f2bb.pdf">上交所2021年1月月报</a> · <a href="https://english.sse.com.cn/news/publications/newsletter/c/10804613/files/3009d96b0f3048ceac3e47569bd025dc.html">上交所2025年12月简报</a> · <a href="https://yuanchuang.10jqka.com.cn/20261117/c681901356.shtml">同花顺末日收盘快讯</a></p>
    <p class="note">项目根目录运行 python scripts/replicate_sse_csi300.py。日线、分期绩效与十组候选的 CSV 存在 outputs/tables，下游可直接检查逐日持仓收益与复利。</p>"""
    pages = [summary,method,selection,evidence]
    css = """@page{size:A4;margin:0}*{box-sizing:border-box}body{margin:0;color:#172c32;font:10.9pt/1.62 'Microsoft YaHei',sans-serif}
    section{position:relative;width:210mm;height:297mm;padding:15mm 16mm 13mm;break-after:page}section:last-child{break-after:auto}
    header,footer,.note{color:#516668}header,footer{font-size:9pt;letter-spacing:.05em}header{border-bottom:1px solid #a5c1bb;padding-bottom:7px}
    footer{position:absolute;bottom:8mm;left:16mm;right:16mm;border-top:1px solid #c6d4d0;padding-top:5px;display:flex;justify-content:space-between}
    h1{font-size:23pt;line-height:1.25;margin:20px 0 8px}h2{font-size:14.5pt;margin:18px 0 6px;color:#155b54}p{margin:8px 0}
    .lead{font-size:12pt;color:#48635f}.note{font-size:9pt}table{width:100%;border-collapse:collapse;font-size:9.5pt;margin:11px 0 6px}
    th,td{text-align:right;border-bottom:1px solid #d3dfdb;padding:5px 4px}th{background:#e6efec;color:#184d46}
    td:first-child,th:first-child{text-align:left}img{display:block;width:100%;margin-top:8px}a{color:#17635a}"""
    html = "<html lang='zh-CN'><meta charset='utf-8'><title>上证指数与沪深300配对复现</title><style>"+css+"</style>" + "".join(
        f"<section><header>A股宽基指数研究　/　配对交易复现</header>{content}<footer><span>价格指数　·　模拟毛收益　·　数据截至 2026-11-17</span><span>{i} / 4</span></footer></section>"
        for i,content in enumerate(pages,1))+"</html>"
    with TemporaryDirectory() as tmp, sync_playwright() as pw:
        source = Path(tmp)/"report.html"
        source.write_text(html,encoding="utf-8")
        browser = pw.chromium.launch(channel="msedge")
        page = browser.new_page()
        page.goto(source.as_uri())
        page.emulate_media(media="print")
        overlaps = page.locator("section").evaluate_all("""els=>els.map(e=>e.querySelector('footer').getBoundingClientRect().top-e.querySelector('footer').previousElementSibling.getBoundingClientRect().bottom)""")
        if min(overlaps)<0:
            raise ValueError(f"Report overflow {overlaps}")
        page.pdf(path=REPORTS/"sse_csi300_replication.pdf",prefer_css_page_size=True,print_background=True)
        browser.close()


if __name__ == "__main__":
    main()
