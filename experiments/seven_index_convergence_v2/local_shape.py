"""Re-evaluate the archived V2 masks on the original trading-day axis."""
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = HERE / "figures"
WINDOWS = (60, 120, 252)
NAMES = {"000001": "上证综指", "000985": "中证全指", "000300": "沪深300",
         "000903": "中证A100", "000132": "上证100", "399311": "国证1000", "000905": "中证500"}


def metrics(paths, daily):
    gap = paths.iloc[:, 0] - paths.iloc[:, 1]
    centered = gap.std(ddof=0)
    level = np.sqrt(np.mean(gap**2))
    assert np.isclose(level**2, gap.mean()**2 + centered**2)
    nav = np.exp(paths)
    result = dict(level=level, centered=centered, mean_gap=gap.mean(),
                  offset_share=(gap.mean()/level)**2 if level else np.nan,
                  nav_centered=(nav.iloc[:, 0]-nav.iloc[:, 1]).std(ddof=0),
                  tracking_error=(daily.iloc[:, 0]-daily.iloc[:, 1]).std()*np.sqrt(252))
    for w in WINDOWS:
        result[f"rolling_{w}"] = gap.rolling(w).std(ddof=0).mean()
    return result


def save_figure(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=200)
    svg = FIG / f"{stem}.svg"
    fig.savefig(svg, metadata={"Date": None})
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines)+"\n", encoding="utf-8")
    plt.close(fig)


def draw(example, table, initial_date):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "svg.fonttype": "none", "svg.hashsalt": "v2-shape"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for col, method in enumerate(("original", "pair_K10")):
        paths = example[method]
        half = len(paths)//2
        full = pd.concat([pd.DataFrame(0., index=[initial_date], columns=paths.columns), paths])
        late = paths.iloc[half-1:]-paths.iloc[half-1]
        for row, (panel, label) in enumerate(((full, "从共同起点累计"),
                                              (late, "从中点前一收盘重新累计"))):
            ax = axes[row, col]
            for code, color in zip(panel, ("#16588b", "#ae5124")):
                ax.plot(panel.index, 100*panel[code], color=color, lw=1, label=NAMES[code])
            ax.set_title(("原始行情" if col == 0 else "本配对独立删 10 日") + " · " + label)
            ax.set_ylabel("累计对数收益 × 100")
            if row == 0:
                ax.axvline(paths.index[half], color="#505860", ls=":", lw=1)
            ax.legend(frameon=False, ncol=2, fontsize=8)
            ax.grid(axis="y", alpha=.2)
    for row in axes:
        low = min(ax.get_ylim()[0] for ax in row)
        high = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(low, high)
    save_figure(fig, "10_rebased_example")
    view = table.query("period == 'late' and method == 'pair_K10'").set_index("pair")
    columns = ["level", "centered", "rolling_120", "nav_centered"]
    vals = 100*view[[c+"_gain" for c in columns]].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    limit = max(abs(vals.min()), vals.max())
    ax.imshow(vals, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    labels = [" / ".join(NAMES[c] for c in p.split("_")) for p in view.index]
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xticks(range(4), ["起点固定距离", "段内中心化", "120日局部距离", "线性净值中心化"])
    ax.xaxis.tick_top()
    for (i, j), value in np.ndenumerate(vals):
        ax.text(j, i, f"{value:+.1f}%", ha="center", va="center", fontsize=8,
                color="white" if abs(value) > .62*limit else "#142536")
    ax.set_xlabel("删日前后距离的改善率（蓝色为缩小，红色为扩大）")
    save_figure(fig, "11_late_pair_shapes")


def main():
    close = pd.read_csv(OUT/"input_closes_main.csv", index_col=0, parse_dates=True)
    assert close.columns.tolist() == list(NAMES) and close.index.is_unique and close.index.is_monotonic_increasing
    assert np.isfinite(close).all().all() and (close > 0).all().all()
    returns = np.log(close).diff().iloc[1:]
    mid = len(returns)//2
    periods = {"full": slice(None), "early": slice(None, mid), "late": slice(mid, None)}
    pair_dates = pd.read_csv(OUT/"pairwise_key_dates.csv", parse_dates=["date"]).query("sample=='main'")
    common = pd.read_csv(OUT/"deletion_dates.csv", parse_dates=["date"]).query(
        "sample=='main' and kind=='natural' and K==10 and method in ['A','U','R2','B']")
    rows, example = [], {}
    for a, b in combinations(close.columns, 2):
        pair = a+"_"+b
        masks = {"original": [], **{f"pair_K{k}": g.date for k, g in
                 pair_dates.loc[pair_dates.pair.eq(pair)].groupby("K")},
                 **{m: g.date for m, g in common.groupby("method")}}
        for method, dates in masks.items():
            daily = returns[[a,b]].copy()
            assert len(set(dates)) == len(dates) and set(dates).issubset(daily.index)
            daily.loc[list(dates)] = 0
            paths = daily.cumsum()
            if pair == "000001_000300" and method in ("original", "pair_K10"):
                example[method] = paths
            for period, s in periods.items():
                p, d = paths.iloc[s], daily.iloc[s]
                rows.append(dict(pair=pair, method=method, period=period,
                                 start=p.index[0].date(), end=p.index[-1].date(), days=len(p),
                                 removed=p.index.isin(dates).sum(), **metrics(p, d)))
    table = pd.DataFrame(rows)
    distances = ["level", "centered", "nav_centered", "tracking_error", *[f"rolling_{w}" for w in WINDOWS]]
    original = table.query("method=='original'").set_index(["pair","period"])
    for col in distances:
        base = original[col].reindex(pd.MultiIndex.from_frame(table[["pair","period"]])).to_numpy()
        table[col+"_gain"] = 1-table[col]/base
    archive = pd.read_csv(OUT/"independent_pair_deletion.csv").query("sample=='main'")
    for k, g in archive.groupby("K"):
        actual = table.query("period=='full' and method==@method_name",
                             local_dict={"method_name":f"pair_K{k}"}).set_index("pair")
        np.testing.assert_allclose(actual.loc[g.pair,"level"], g.rms, atol=1e-12)
    table.to_csv(OUT/"local_shape_pairs.csv", index=False)
    summary = []
    for (method, period), group in table.groupby(["method", "period"], sort=False):
        for col in distances:
            summary.append(dict(method=method, period=period, metric=col, pairs=len(group),
                                median=group[col].median(), median_gain=group[col+"_gain"].median(),
                                improved=(group[col+"_gain"] > 1e-12).sum()))
    pd.DataFrame(summary).to_csv(OUT/"local_shape_summary.csv", index=False)
    draw(example, table, close.index[0])


if __name__ == "__main__":
    main()
