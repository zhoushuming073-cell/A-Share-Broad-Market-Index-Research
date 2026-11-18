"""Plot the previously identified test-period champion over every available date."""
from optimize_hedge_ratio import account, metrics
from run_pair_search import inputs, load, paths
from research_common import FIGURES, REPORTS, csv, np, pd, plt

plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})


def main():
    prices, returns, _, (train_end, valid_end) = inputs()
    champion = load("top_return_pairs").iloc[0]
    pair = paths(champion.a, champion.b)
    row = list(pair["ids"]).index(champion.spec)
    dates = returns.index
    gross = pair["ret"][row]
    long_weight = pair["w"][row]
    short_weight = pair["v"][row]
    days = prices.index.to_series().diff().dt.days.loc[dates].to_numpy(float)
    net_account = account(
        returns[champion.a].to_numpy(),
        returns[champion.b].to_numpy(),
        long_weight, short_weight, days,
        np.ones(len(dates), dtype=np.bool_), .0005, .02,
    )
    full = pd.DataFrame({
        "date": dates,
        "phase": np.select(
            [np.arange(len(dates)) < train_end, np.arange(len(dates)) < valid_end],
            ["训练", "验证"], default="测试",
        ),
        "gross_return": gross,
        "gross_nav": np.cumprod(1 + gross),
        "scenario_net_return": net_account[:, 0],
        "scenario_net_nav": net_account[:, 1],
        "long_weight": long_weight,
        "short_weight": short_weight,
        "traded_per_initial_capital": net_account[:, 2],
    })
    csv(full, "pair_champion_full_history")

    # The test-period gross path must reconcile with the published leaderboard.
    test = metrics(gross[valid_end:])
    np.testing.assert_allclose(test["total"], champion.test_total, atol=1e-9)
    assert (full.scenario_net_nav > 0).all()

    chart_dates = prices.index
    gross_nav = pd.Series(np.r_[1., full.gross_nav.to_numpy()], index=chart_dates)
    net_nav = pd.Series(np.r_[1., full.scenario_net_nav.to_numpy()], index=chart_dates)
    phases = [
        ("训练", chart_dates[0], dates[train_end - 1], "#edf3f1"),
        ("验证", dates[train_end], dates[valid_end - 1], "#eaf1fb"),
        ("测试", dates[valid_end], dates[-1], "#fff1e7"),
    ]
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    fig.subplots_adjust(left=.085, right=.97, top=.79, bottom=.22)
    fig.suptitle(
        "测试期事后收益第一名的历史回看  |  上证指数 / 上证100",
        fontsize=17, fontweight="bold",
    )
    ax.set_title(
        "2015-01-05 至 2026-11-17  ·  EWMA β（0.94，每月更新）  ·  总名义敞口 1 倍",
        fontsize=10, loc="left", color="#526563",
    )
    for name, start, end, color in phases:
        ax.axvspan(start, end, color=color, alpha=.85, lw=0)
        ax.text(start + (end - start) / 2, .98, name,
                ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontsize=10, color="#4d6260")
    ax.plot(gross_nav, color="#087d71", lw=1.7,
            label=f"模拟毛净值  {gross_nav.iloc[-1]:.3f}（全期 {gross_nav.iloc[-1]-1:+.2%}）")
    ax.plot(net_nav, color="#b2542e", lw=1.5,
            label=f"成本情景净值  {net_nav.iloc[-1]:.3f}（全期 {net_nav.iloc[-1]-1:+.2%}）")
    ax.axhline(1, color="#778482", lw=.8, ls="--")
    for boundary in (dates[train_end], dates[valid_end]):
        ax.axvline(boundary, color="#71817f", lw=.8, ls=":")
    ax.set_ylabel("模拟净值（2015-01-05 = 1）")
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(alpha=.15)
    fig.text(
        .085, .07,
        "成本情景：单边成交费 5 bp，空头年化借券费 2%；按实际成交额逐日计提。"
        " 此配置训练期夏普为负，未通过事前筛选。",
        fontsize=9, color="#526563",
    )
    image = FIGURES / "pair_champion_full_history.png"
    fig.savefig(image, dpi=180, facecolor="white")
    plt.close(fig)

    periods = {
        "训练": slice(0, train_end),
        "验证": slice(train_end, valid_end),
        "测试": slice(valid_end, None),
        "全期": slice(None),
    }
    rows = []
    for label, segment in periods.items():
        gm = metrics(gross[segment])
        nm = metrics(net_account[segment, 0])
        rows.append((label, gm["n"], gm["total"], nm["total"],
                     gm["annual"], gm["mdd"]))
    summary = pd.DataFrame(rows, columns=[
        "phase", "return_days", "gross_total", "scenario_net_total",
        "gross_annual", "gross_max_drawdown",
    ])
    csv(summary, "pair_champion_full_history_summary")

    report = REPORTS / "pair_trading_full_research.md"
    text = report.read_text(encoding="utf-8")
    heading = "## 测试期收益冠军的模拟净值"
    block = (
        "## 冠军配对的完整历史曲线\n\n"
        "![冠军配对从训练期到测试期的完整净值]"
        "(../outputs/figures/pair_champion_full_history.png)\n\n"
        "曲线从2015-01-05的1元起始，覆盖训练、验证和测试三个阶段。"
        "同一组EWMA 0.94、每月更新的参数贯穿全段。"
        "该组合是看过测试期收益后选出的冠军，训练期与验证期曲线用于展示历史表现。"
        "逐日净值见[完整历史数据]"
        "(../outputs/tables/pair_champion_full_history.csv)。\n\n"
        "| 阶段 | 收益日 | 毛收益 | 成本情景收益 | 毛收益年化 | 毛收益最大回撤 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for label, n, total, net_total, annual, drawdown in rows:
        block += (
            f"| {label} | {n:,} | {total:+.2%} | {net_total:+.2%} | "
            f"{annual:+.2%} | {drawdown:.2%} |\n"
        )
    block += "\n阶段收益分别从阶段首日净值重新计量。\n\n"
    if "## 冠军配对的完整历史曲线" in text:
        before, after = text.split("## 冠军配对的完整历史曲线", 1)
        _, suffix = after.split(heading, 1)
        text = before + block + heading + suffix
    else:
        text = text.replace(heading, block + heading, 1)
    report.write_text(text, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
