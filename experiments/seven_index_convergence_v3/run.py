"""Seven-index rolling reset paths on the unchanged V2 price panel."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/"seven_index_convergence_v2/results/input_closes_main.csv"
OUT = ROOT/"results"
FIG = ROOT/"figures"
NAMES = {"000001":"上证综指", "000985":"中证全指", "000300":"沪深300", "000903":"中证A100",
         "000132":"上证100", "399311":"国证1000", "000905":"中证500"}
PAIRS = list(combinations(NAMES, 2))
A, B = np.array(list(combinations(range(7), 2))).T
WINDOWS = (60, 120, 252)
SEED, DRAWS = 20261122, 10000
SPLIT = pd.Timestamp("2023-09-14")
METRICS = ["reset_60", "reset_120", "reset_252", "center_60", "center_120", "center_252",
           "detrend", "drift", "correlation", "daily_rms"]


def segment(x):
    gaps = x[:, A]-x[:, B]
    paths = gaps.cumsum(axis=0)
    design = np.column_stack((np.ones(len(x)), np.arange(len(x))))
    fit = design@np.linalg.lstsq(design, paths, rcond=None)[0]
    return pd.DataFrame(dict(pair=["_".join(p) for p in PAIRS],
        reset=np.sqrt(np.mean(paths**2, axis=0)), detrend=np.sqrt(np.mean((paths-fit)**2, axis=0)),
        drift=252*abs(gaps.mean(axis=0)), correlation=np.corrcoef(x.T)[A, B],
        daily_rms=np.sqrt(np.mean(gaps**2, axis=0))))


def early_offset(r, split):
    """Signed contributions measured in the direction of the observed early endpoint."""
    records = []
    for a, b in PAIRS:
        early = (r[a]-r[b]).iloc[:split]
        endpoint = early.sum()
        if endpoint == 0:
            continue  # No terminal lead to explain in this pair.
        directed = np.sign(endpoint)*early
        for span in (1,20):
            sums = directed.rolling(span).sum().dropna()
            end = sums.idxmax()
            contribution = sums.at[end]
            records.append(dict(pair=a+"_"+b, span=span, start=early.index[early.index.get_loc(end)-span+1],
                                end=end, endpoint=endpoint, signed_contribution=np.sign(endpoint)*contribution,
                                share=contribution/abs(endpoint),
                                endpoint_without=endpoint-np.sign(endpoint)*contribution))
    return pd.DataFrame(records)


def observed(r, split):
    x = r.to_numpy()
    summary, windows, rolling_pairs = [], [], []
    details = pd.concat([segment(v).assign(period=label) for label, v in
                         [("early",x[:split]), ("late",x[split:])]], ignore_index=True)
    for w in WINDOWS:
        starts = np.arange(len(x)-w+1)
        gaps = x[:, A]-x[:, B]
        path = np.vstack([np.zeros(len(PAIRS)), gaps.cumsum(axis=0)])
        pair_reset = np.array([np.sqrt(np.mean((path[i+1:i+w+1]-path[i])**2, axis=0)) for i in starts])
        corr = np.array([np.corrcoef(x[i:i+w].T)[A,B] for i in starts])
        strict = np.where(starts+w<=split, "early", np.where(starts>=split, "late", "cross"))
        central = np.where(starts+w//2<split, "early", "late")
        pool = np.sqrt(np.mean(pair_reset**2, axis=1))
        index = pd.DataFrame(dict(window=w, start=r.index[starts], end=r.index[starts+w-1],
                                  center=r.index[starts+w//2], strict=strict, midpoint=central))
        windows.append(index.assign(reset=pool, correlation=np.median(corr, axis=1)))
        for k, p in enumerate(PAIRS):
            rolling_pairs.append(index.assign(pair="_".join(p), reset=pair_reset[:,k], correlation=corr[:,k]))
        for name, groups in [("strict", strict), ("midpoint", central)]:
            for label in ["early","late"]:
                selected = groups==label
                summary.append(dict(rule=name, window=w, period=label, count=int(selected.sum()),
                                    crossing=int((selected & (strict=="cross")).sum()), reset=np.median(pool[selected])))
    return details, pd.concat(windows), pd.concat(rolling_pairs), pd.DataFrame(summary)


@njit(cache=True)
def pooled_windows(x, split):
    # Mean squared pairwise differences = 2/(n-1) times the sum across centered indices.
    n, m = x.shape
    path = np.zeros((n+1,m))
    prefix = np.zeros_like(path)
    sq = np.zeros(n+1)
    for t in range(n):
        path[t+1] = path[t]+x[t]-np.mean(x[t])
        prefix[t+1] = prefix[t]+path[t+1]
        sq[t+1] = sq[t]+np.dot(path[t+1], path[t+1])
    changes = np.empty(6)
    for k, w in enumerate(WINDOWS):
        series = np.empty(n-w+1)
        for s in range(len(series)):
            e = s+w
            ms = (sq[e]-sq[s]-2*np.dot(path[s], prefix[e]-prefix[s]))/w+np.dot(path[s], path[s])
            series[s] = np.sqrt(max(0., ms)*2/(m-1))
        changes[k] = np.median(series[split:])-np.median(series[:split-w+1])
        boundary = split-w//2
        changes[k+3] = np.median(series[boundary:])-np.median(series[:boundary])
    return changes


@njit(cache=True)
def auxiliary(x):
    n, m = x.shape
    time = np.arange(n)-(n-1)/2
    denom = np.dot(time,time)
    drifts = np.empty(21)
    corrs = np.empty(21)
    ss = daily = 0.
    k = 0
    for i in range(m):
        for j in range(i+1,m):
            a, b = x[:,i], x[:,j]
            diff = a-b
            wealth = np.cumsum(diff)
            centered = wealth-np.mean(wealth)
            resid = centered-time*(np.dot(centered,time)/denom)
            ss += np.mean(resid**2)
            daily += np.mean(diff**2)
            drifts[k] = abs(np.mean(diff))*252
            corrs[k] = np.mean((a-a.mean())*(b-b.mean()))/(a.std()*b.std())
            k += 1
    return np.array([np.sqrt(ss/k),np.median(drifts),np.median(corrs),np.sqrt(daily/k)])


@njit(parallel=True, cache=True)
def null_draws(x, split, block, starts):
    draws = np.empty((len(starts),10))
    for k in prange(len(starts)):
        z = np.empty_like(x)
        for t in range(len(x)):
            z[t] = x[(starts[k,t//block]+t%block)%len(x)]
        draws[k,:6] = pooled_windows(z,split)
        draws[k,6:] = auxiliary(z[split:])-auxiliary(z[:split])
    return draws


def inference(x, split, observed_change):
    rng = np.random.default_rng(SEED)
    rows, archive = [], {}
    for block in (5,20,60):
        starts = rng.integers(len(x),size=(DRAWS,(len(x)+block-1)//block))
        sims = null_draws(x,split,block,starts)
        archive[f"block_{block}"] = sims
        for j, metric in enumerate(METRICS):
            v = sims[:,j]
            delta = observed_change[j]
            tail = min(np.sum(v<=delta),np.sum(v>=delta))
            p = min(1.,2*(1+tail)/(DRAWS+1))
            rows.append(dict(block=block, metric=metric, delta=delta, null_low=np.quantile(v,.025),
                             null_high=np.quantile(v,.975), p=p, family=("strict","midpoint","aux")[int(j>=3)+int(j>=6)]))
    df = pd.DataFrame(rows)
    df["holm"] = df.groupby(["block","family"]).p.transform(lambda p: multipletests(p,method="holm")[1])
    np.savez_compressed(OUT/"bootstrap_draws.npz", metrics=np.array(METRICS), **archive)
    return df


def figures(windows, detail, rolling_pairs):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,
                         "svg.fonttype":"none","svg.hashsalt":"seven-v3"})
    fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True,layout="constrained")
    for w, g in windows.groupby("window"):
        axes[0].plot(g.end,100*g.reset,label=f"{w}日",lw=1.25)
        axes[1].plot(g.end,g.correlation,label=f"{w}日",lw=1.05)
    for ax, ylabel in zip(axes,["窗口重置 RMS（对数点）","配对日收益相关系数中位数"]):
        ax.axvline(SPLIT,ls="--",color="#575757",lw=1)
        ax.grid(axis="y",alpha=.25)
        ax.set_ylabel(ylabel)
        ax.legend(ncol=3,frameon=False)
    axes[1].set_xlabel("窗口结束日期")
    save(fig,"rolling_pool")
    med = rolling_pairs.query("strict!='cross'").groupby(["pair","window","strict"]).reset.median().unstack()
    heat = (100*(med.late/med.early-1)).unstack()
    levels = detail.pivot(index="pair",columns="period",values="reset")
    heat["segment"] = 100*(levels.late/levels.early-1)
    order = ["_".join(p) for p in PAIRS]
    heat = heat.loc[order]
    heat.to_csv(OUT/"pair_changes_pct.csv")
    fig, ax = plt.subplots(figsize=(8,8),layout="constrained")
    limit = 100
    ax.imshow(heat.to_numpy(),aspect="auto",cmap="RdBu_r",vmin=-limit,vmax=limit)
    ax.set_yticks(np.arange(len(heat)), [" / ".join(NAMES[c] for c in p) for p in PAIRS],fontsize=8)
    ax.set_xticks(range(4),["60日","120日","252日","整段重置"])
    ax.xaxis.tick_top()
    for (i,j), value in np.ndenumerate(heat.to_numpy()):
        ax.text(j,i,f"{value:+.1f}%",ha="center",va="center",fontsize=8,
                color="white" if abs(value)>65 else "black")
    ax.set_xlabel("后半段相对前半段变化率　蓝色为距离缩小，色阶在 ±100% 截断")
    save(fig,"pair_changes")


def save(fig, name):
    fig.savefig(FIG/(name+".png"),dpi=200)
    svg = FIG/(name+".svg")
    fig.savefig(svg,metadata={"Date":None})
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    closes = pd.read_csv(SOURCE,index_col=0,parse_dates=True)
    assert list(closes)==list(NAMES) and closes.index.is_unique and closes.index.is_monotonic_increasing
    assert np.isfinite(closes).all().all() and (closes>0).all().all()
    r = np.log(closes).diff().iloc[1:]
    split = r.index.get_loc(SPLIT)
    x = np.ascontiguousarray(r.to_numpy())
    detail, windows, pairs, summary = observed(r,split)
    early_offset(r,split).to_csv(OUT/"early_offset.csv", index=False)
    detail.to_csv(OUT/"segment_pairs.csv",index=False)
    windows.to_csv(OUT/"rolling_pool.csv",index=False)
    pairs.to_csv(OUT/"rolling_pairs.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    summary.to_csv(OUT/"window_summary.csv",index=False)
    metrics = []
    for period, g in detail.groupby("period"):
        for label in ["detrend","daily_rms","reset","drift","correlation"]:
            pooled = np.sqrt(np.mean(g[label]**2)) if label in ("detrend","daily_rms","reset") else g[label].median()
            metrics.append(dict(period=period,metric=label,pool=pooled,q25=g[label].quantile(.25),
                                median=g[label].median(),q75=g[label].quantile(.75)))
    stats = pd.DataFrame(metrics)
    stats.to_csv(OUT/"segment_summary.csv",index=False)
    changes = summary.pivot(index=["rule","window"],columns="period",values="reset")
    aux = stats.pivot(index="metric",columns="period",values="pool")
    delta = np.r_[(changes.late-changes.early).loc["strict"],(changes.late-changes.early).loc["midpoint"],
                  (aux.late-aux.early).loc[METRICS[6:]]]
    np.testing.assert_allclose(delta,np.r_[pooled_windows(x,split),auxiliary(x[split:])-auxiliary(x[:split])],atol=1e-12)
    inference(x,split,delta).to_csv(OUT/"bootstrap_tests.csv", index=False)
    figures(windows,detail,pairs)
    (OUT/"manifest.json").write_text(json.dumps(dict(input=SOURCE.name,sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        closes=len(closes),days=len(r),start=str(closes.index[0].date()),end=str(r.index[-1].date()),first_late=str(SPLIT.date()),
        windows=WINDOWS,draws=DRAWS,seed=SEED,blocks=[5,20,60]),indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
