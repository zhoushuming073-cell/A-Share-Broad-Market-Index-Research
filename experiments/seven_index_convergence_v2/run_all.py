import hashlib
import json
from importlib.metadata import version

import numpy as np
import pandas as pd
from data_validation import CFG, OUT, load
from key_dates import scores
from deletion_analysis import analyze
from bootstrap_inference import random_test, date_test, bootstrap


def run():
    OUT.mkdir(exist_ok=True)
    main,audit,extremes = load(CFG["start"],"main")
    long,extra,tail = load(CFG["long_start"],"long")
    pd.DataFrame(audit+extra).to_csv(OUT/"data_audit.csv",index=False)
    pd.DataFrame(extremes+tail).to_csv(OUT/"flagged_returns.csv",index=False)
    tables,masks,baseline = analyze(main,"main")
    other,_,_ = analyze(long,"long")
    for name,frame in tables.items():
        pd.concat([frame,other.get(name,pd.DataFrame())],ignore_index=True).to_csv(OUT/f"{name}.csv",index=False)
    jaccard = pd.concat([tables["pairwise_jaccard"],other["pairwise_jaccard"]])
    by_share = jaccard.assign(group=np.where(jaccard.shared_index,"shared","disjoint"))
    by_share = pd.concat([by_share,jaccard.assign(group="all")])
    summary = by_share.groupby(["sample","K","group"]).jaccard.agg(
        count="size",mean="mean",median="median",sd="std",minimum="min",maximum="max",
        q05=lambda a:a.quantile(.05),q25=lambda a:a.quantile(.25),
        q75=lambda a:a.quantile(.75),q95=lambda a:a.quantile(.95))
    summary.to_csv(OUT/"jaccard_summary.csv")
    convergence = pd.concat([tables["convergence_results"],other["convergence_results"]])
    convergence.query("kind=='budget'").to_csv(OUT/"deletion_budget_comparison.csv",index=False)
    manifest = dict(config=CFG,versions={name:version(name) for name in
                    ("numpy","pandas","numba","exchange_calendars","statsmodels","matplotlib","PyYAML")},
                    generator="PCG64",random_seed=CFG["seed"],
                    date_seed=CFG["seed"]+1,block_seeds={str(k):CFG["seed"]+100+k for k in CFG["blocks"]},
                    input_sha256={s:hashlib.sha256((OUT/f"input_closes_{s}.csv").read_bytes()).hexdigest()
                                  for s in ("main","long")})
    (OUT/"simulation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    random_test(main,tables["convergence_results"],baseline,tables["pair_convergence"])
    print("Completed equal-count random deletion",flush=True)
    pair_scores=scores(main.values)
    date_test(main,pair_scores,np.argsort(-pair_scores,axis=1,kind="stable"))
    print("Completed quarter-stratified date randomization",flush=True)
    bootstrap(main)
    from figures import build
    build()


if __name__=="__main__":
    run()
