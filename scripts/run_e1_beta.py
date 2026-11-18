from research_common import *
from statsmodels.stats.multitest import multipletests


def main():
    _, r, _ = market()
    rows, rolling = [], {}
    groups = {"full": r, "2015-2019": r.loc[:"2019"], "2020-2022": r.loc["2020":"2022"],
              "2023-2026": r.loc["2023":], "2015-2025": r.loc[:"2025"]}
    rf = factors()["rf_dly"].reindex(r.index).dropna()
    for index in TARGETS:
        for benchmark in BENCHES:
            for name, d in groups.items():
                rows.append(dict(index=index, benchmark=benchmark, sample=name, model="market_price",
                                 **regression_row(d[index], d[benchmark])))
            for name, mask in [("up_day", r[benchmark]>0), ("down_day", r[benchmark]<0)]:
                rows.append(dict(index=index, benchmark=benchmark, sample=name, model="market_price",
                                 **regression_row(r.loc[mask, index], r.loc[mask, benchmark], r.index)))
            d = r.loc[rf.index]
            for name, risk_free in [("matched_raw", 0), ("matched_excess", rf)]:
                rows.append(dict(index=index, benchmark=benchmark, sample="rf_overlap", model=name,
                                 **regression_row(d[index]-risk_free, d[benchmark]-risk_free)))
            for length in [60, 252]:
                rolling[index, benchmark, length] = r[index].rolling(length).cov(r[benchmark]) / r[benchmark].rolling(length).var()
    results = pd.DataFrame(rows)
    mask = results["sample"].eq("full")
    for col in ["alpha_p", "beta_eq_one_p"]:
        results.loc[mask, col+"_holm"] = multipletests(results.loc[mask, col], method="holm")[1]
    csv(results, "e1_regression")
    rr = pd.DataFrame(rolling)
    rr.columns = pd.MultiIndex.from_tuples(rr.columns, names=["index", "benchmark", "window"])
    csv(rr.stack(["index", "benchmark", "window"], future_stack=True).dropna().rename("beta").reset_index(), "e1_rolling_beta")
    for benchmark in BENCHES:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for a, length in zip(ax, [60, 252]):
            for code in TARGETS:
                a.plot(rr.index, rr[code, benchmark, length], label=NAMES[code], linewidth=1)
            a.axhline(1, color="gray", lw=.7)
            a.set_ylabel(f"{length}-day beta")
        ax[0].legend(ncol=4, loc="upper right")
        fig.suptitle(f"Rolling beta vs {NAMES[benchmark]}")
        figure(fig, f"e1_rolling_{benchmark}")
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True, layout="constrained")
    for i, benchmark in enumerate(BENCHES):
        for j, code in enumerate(TARGETS):
            a = axs[i, j]
            p = results.loc[mask & results["index"].eq(code) & results.benchmark.eq(benchmark)].iloc[0]
            a.scatter(r[benchmark]*100, r[code]*100, s=4, alpha=.20)
            x = np.array([r[benchmark].min(), r[benchmark].max()])
            a.plot(x*100, (p.alpha_daily+p.beta*x)*100, c="#aa493e")
            a.set_title(f"{NAMES[code]} · R² {p.r2:.2f}")
            a.set_xlabel(f"{NAMES[benchmark]} (%)")
            if j == 0:
                a.set_ylabel("Index return (%)")
    figure(fig, "e1_scatter")
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, benchmark in enumerate(BENCHES):
        p = results.loc[mask & results.benchmark.eq(benchmark)].set_index("index").loc[TARGETS]
        ax.errorbar(np.arange(4)+(.12 if j else -.12), p.beta, yerr=[p.beta-p.beta_lo,p.beta_hi-p.beta],
                    fmt="o", capsize=4, label=NAMES[benchmark])
    ax.axhline(1, color="gray", lw=.8)
    ax.set_xticks(range(4), [NAMES[c] for c in TARGETS])
    ax.set_ylabel("Beta and 95% HAC CI")
    ax.legend()
    figure(fig, "e1_exposure")


if __name__ == "__main__":
    main()
