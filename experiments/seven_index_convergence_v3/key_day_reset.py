"""Apply V2's archived common date masks to V3 rolling reset distances."""
import hashlib
import json

import numpy as np
import pandas as pd
from numba import njit, prange
from statsmodels.stats.multitest import multipletests

from run import A, B, FIG, NAMES, OUT, PAIRS, SOURCE, SPLIT, WINDOWS, plt, save

V2 = SOURCE.parent
METHODS = ("A","U","R2","B")
PERIODS = ("full","early","late")
DRAWS = 10000
SEED = 20261122


def paths(x, window):
    """Compute each pair's squared reset RMS via two prefix sums."""
    diff = x[:,A]-x[:,B]
    zero = np.zeros((1,len(PAIRS)))
    path = np.vstack([zero,diff.cumsum(axis=0)])
    prefix = np.vstack([zero,path[1:].cumsum(axis=0)])
    squared = np.vstack([zero,(path[1:]**2).cumsum(axis=0)])
    base = path[:-window]
    variance = ((squared[window:]-squared[:-window])-2*base*(prefix[window:]-prefix[:-window]))/window + base**2
    return np.sqrt(np.maximum(variance,0))


def masks(index):
    dates = pd.read_csv(V2/"deletion_dates.csv",parse_dates=["date"]).query("sample=='main'")
    info = pd.read_csv(V2/"convergence_results.csv").query("sample=='main' and method in @METHODS")
    cases = []
    for row in info.itertuples():
        if row.kind != "natural" and not (row.kind=="budget" and row.deleted==row.K):
            continue
        days = dates.loc[dates.scenario.eq(row.scenario),"date"]
        slots = np.sort(index.get_indexer(days))
        assert slots.min()>=0 and len(np.unique(slots))==len(slots)==row.deleted
        signature = hashlib.sha256(b"main"+slots.astype("<i4").tobytes()).hexdigest()[:16]
        assert signature==row.mask_id
        cases.append((row,slots))
    assert len(cases)==24
    return cases


@njit(cache=True,parallel=True)
def random_distances(centered, slots, split):
    """Joint masks on all seven indices. RMS across pairs from cross-sectional dispersion."""
    n, m = centered.shape
    result = np.empty((len(slots),3,3))
    for k in prange(len(slots)):
        keep = np.ones(n,np.bool_)
        keep[slots[k]] = False
        path = np.zeros((n+1,m))
        sums = np.zeros_like(path)
        squares = np.zeros(n+1)
        for t in range(n):
            for j in range(m):
                path[t+1,j] = path[t,j]+centered[t,j]*keep[t]
                sums[t+1,j] = sums[t,j]+path[t+1,j]
            squares[t+1] = squares[t]+np.dot(path[t+1],path[t+1])
        for j,w in enumerate(WINDOWS):
            v = np.empty(n-w+1)
            for s in range(len(v)):
                total = squares[s+w]-squares[s]
                for c in range(m):
                    y = path[s,c]
                    total += y*(w*y-2*(sums[s+w,c]-sums[s,c]))
                v[s] = np.sqrt(max(0.,total)*2/(w*(m-1)))
            result[k,j,0] = np.median(v)
            result[k,j,1] = np.median(v[:split-w+1])
            result[k,j,2] = np.median(v[split:])
    return result


def evaluate(r, cases):
    split = r.index.get_loc(SPLIT)
    pooled, pairs, series = [], [], []
    x = r.to_numpy()
    reference = {w:paths(x,w) for w in WINDOWS}
    for row, slots in cases:
        z = x.copy()
        z[slots] = 0
        meta = dict(scenario=row.scenario, mask_id=row.mask_id, method=row.method, kind=row.kind,
                    K=row.K, deleted=len(slots), deleted_early=int(np.sum(slots<split)),
                    deleted_late=int(np.sum(slots>=split)), v2_gain=row.improvement)
        for w, before in reference.items():
            after = paths(z,w)
            original = np.sqrt(np.mean(before**2,axis=1))
            modified = np.sqrt(np.mean(after**2,axis=1))
            if row.K==10:
                series.append(pd.DataFrame(dict(scenario=row.scenario, window=w, start=r.index[:len(before)],
                    end=r.index[w-1:], original=original, modified=modified)))
            for period, part in zip(PERIODS,(slice(None),slice(None,split-w+1),slice(split,None))):
                a, b = np.median(original[part]), np.median(modified[part])
                p, q = np.median(before[part],axis=0), np.median(after[part],axis=0)
                pooled.append(dict(**meta,window=w,period=period,windows=len(original[part]),
                                    original=a,modified=b,gain=1-b/a,closer_pairs=int((p-q>1e-12).sum())))
                for label, p0, p1 in zip(PAIRS,p,q):
                    pairs.append(dict(scenario=row.scenario,window=w,period=period,pair="_".join(label),
                                      original=p0,modified=p1,gain=1-p1/p0))
    return pd.DataFrame(pooled), pd.DataFrame(pairs), pd.concat(series,ignore_index=True)


def placebo(r, results):
    primary = results.query("K==10").drop_duplicates(["mask_id","window","period"]).copy()
    split = r.index.get_loc(SPLIT)
    x = r.to_numpy()
    centered = np.ascontiguousarray(x-x.mean(axis=1,keepdims=True))
    rng = np.random.default_rng(SEED)
    records = []
    archive = {}
    for (early,late), group in primary.groupby(["deleted_early","deleted_late"]):
        slots = np.empty((DRAWS,early+late),dtype=int)
        for k in range(DRAWS):
            slots[k,:early] = rng.choice(split,early,replace=False)
            slots[k,early:] = split+rng.choice(len(r)-split,late,replace=False)
        distances = random_distances(centered,slots,split)
        archive[f"early{early}_late{late}"] = distances
        for row in group.itertuples():
            draws = 1-distances[:,WINDOWS.index(row.window),PERIODS.index(row.period)]/row.original
            records.append(dict(mask_id=row.mask_id,window=row.window,period=row.period,
                         random_median=np.median(draws),random_p025=np.quantile(draws,.025),
                         random_p975=np.quantile(draws,.975),p=(1+np.count_nonzero(draws>=row.gain-1e-12))/(DRAWS+1)))
    df = pd.DataFrame(records)
    df["holm"] = df.groupby("period").p.transform(lambda p:multipletests(p,method="holm")[1])
    np.savez_compressed(OUT/"key_day_random_draws.npz",windows=WINDOWS,periods=PERIODS,**archive)
    return results.query("K==10").merge(df,on=["mask_id","window","period"],validate="many_to_one")


def figures(results, series):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,
                         "svg.fonttype":"none","svg.hashsalt":"key-day-reset"})
    fig, axes = plt.subplots(1,2,figsize=(10,4),sharey=True,layout="constrained")
    for kind,ax,title in zip(("natural","budget"),axes,("原始名单","每种方法同删10日")):
        v = results.query("K==10 and period=='full' and kind==@kind")
        for k,method in enumerate(METHODS):
            g = v.loc[v.method.eq(method)].sort_values("window")
            ax.bar(np.arange(3)+(k-1.5)*.19,100*g.gain,width=.18,label=f"{method}（{g.deleted.iloc[0]}日）")
        ax.set_xticks(range(3),["60日","120日","252日"])
        ax.axhline(0,color="#444444",lw=.7)
        ax.grid(axis="y",alpha=.2)
        ax.set_title(title)
        ax.legend(fontsize=8,ncol=2)
    axes[0].set_ylabel("全样本滚动距离改善率（%）")
    save(fig,"key_day_budgets")
    fig, axes = plt.subplots(3,1,figsize=(10,5.6),sharex=True,layout="constrained")
    for ax,w in zip(axes,WINDOWS):
        for method in METHODS:
            g = series[(series.scenario==f"{method}_K10") & (series.window==w)]
            ax.plot(g.end,100*(g.original-g.modified),lw=.9,label=method)
        ax.axhline(0,color="black",lw=.7)
        ax.axvline(SPLIT,ls=":",color="#555555",lw=.8)
        ax.set_ylabel(f"{w}日差值（对数点）")
        ax.grid(axis="y",alpha=.2)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("窗口结束日期　正值表示剔除后距离缩小")
    save(fig,"key_day_series")


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    closes = pd.read_csv(SOURCE,index_col=0,parse_dates=True)
    assert list(closes)==list(NAMES) and closes.index.is_unique and closes.index.is_monotonic_increasing
    assert np.isfinite(closes).all().all() and (closes>0).all().all()
    r = np.log(closes).diff().iloc[1:]
    results,pairs,series = evaluate(r,masks(r.index))
    tests = placebo(r,results)
    results.to_csv(OUT/"key_day_pool.csv",index=False)
    pairs.to_csv(OUT/"key_day_pairs.csv",index=False)
    tests.to_csv(OUT/"key_day_random_tests.csv",index=False)
    series.to_csv(OUT/"key_day_series.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    figures(results,series)
    files = (SOURCE,V2/"deletion_dates.csv",V2/"convergence_results.csv")
    manifest = dict(seed=SEED,draws_per_budget=DRAWS,
                    sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (OUT/"key_day_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
