from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd
from data_validation import CFG, CODES, LABELS, OUT, PAIRS
from key_dates import detail, memberships, rank, scores
from convergence_metrics import evaluate, rms_and_gap


def analyze(r,sample):
    x = np.ascontiguousarray(r.values)
    pair_scores = scores(x)
    order = np.argsort(-pair_scores,axis=1,kind="stable")
    overall = pair_scores.mean(axis=0)
    overall_rank = rank(overall)
    reference = rms_and_gap(x,np.zeros(len(x),bool))
    tables = defaultdict(list)
    masks = {}
    plotted = {"original":np.expm1(x.cumsum(axis=0))}

    def record(method,k,kind,slots,scenario):
        summary,pairs,ends,curve = evaluate(x,slots,reference)
        signature = hashlib.sha256(sample.encode()+np.sort(slots).astype("<i4").tobytes()).hexdigest()[:16]
        meta = dict(sample=sample,K=k,kind=kind,method=method,scenario=scenario,mask_id=signature)
        tables["convergence_results"].append(dict(**meta,**summary))
        tables["pair_convergence"].append(pairs.assign(**meta))
        tables["index_endpoints"].append(ends.assign(**meta))
        masks[signature] = slots
        tables["deletion_dates"].extend(dict(**meta,rank=i,date=r.index[day],score=overall[day])
                                       for i,day in enumerate(slots,1))
        if sample=="main" and (kind=="natural" and k==CFG["primary_k"] or scenario in ("A_1pct","A_5pct")):
            plotted[scenario] = curve

    record("original",0,"original",np.empty(0,int),"original")
    overall_daily = pd.DataFrame(dict(date=r.index[overall_rank],score=overall[overall_rank],
        rank=np.arange(1,len(x)+1)))
    for f in CFG["fractions"]:
        amount = int(np.floor(f*len(x)+.5))
        tag = f"{100*f:g}pct"
        record("A",amount,"fraction",overall_rank[:amount],"A_"+tag)
        overall_daily[tag] = overall_daily["rank"].le(amount)
    for k in CFG["k"]:
        overall_daily[f"K{k}"] = overall_daily["rank"].le(k)
        sets,votes,vote_rank,chosen = memberships(pair_scores,order,overall,k)
        pair_days,day,comp,diff,jrows,jac,freq,summary = detail(r,pair_scores,overall,k,sets,votes,chosen)
        tables["pairwise_key_dates"].append(pair_days)
        tables["date_frequency"].append(day)
        tables["method_overlap"].extend(comp)
        tables["method_overlap_dates"].extend(diff)
        tables["pairwise_jaccard"].extend(jrows)
        tables["frequency_distribution"].extend(freq)
        tables["set_summary"].append(summary)
        pd.DataFrame(jac,index=LABELS,columns=LABELS).to_csv(OUT/f"jaccard_{sample}_K{k}.csv",index_label="pair")
        for name,subset in sets.items():
            record(name,k,"natural",subset,f"{name}_K{k}")
            file = "union_dates" if name=="U" else "vote_dates" if name=="B" else "repeated_dates"
            if name!="A":
                tables[file].append(day.iloc[subset].assign(method=name,method_rank=np.arange(1,len(subset)+1)))
        for budget in sorted({k,len(sets["U"]),len(sets["R2"]),*CFG["k"]} - {0}):
            choices = dict(A=overall_rank,B=vote_rank,U=sets["U"],R2=sets["R2"])
            for name,subset in choices.items():
                if len(subset)>=budget:
                    record(name,k,"budget",subset[:budget],f"{name}_K{k}_M{budget}")
                else:
                    tables["unavailable_budgets"].append(dict(K=k,budget=budget,method=name,available=len(subset)))
        # These pair-specific paths do not define a common seven-index experiment.
        for p,label in enumerate(LABELS):
            a,b = PAIRS[p]
            spread = x[:,a]-x[:,b]
            spread[chosen[p]] = 0
            value = np.sqrt(np.mean(spread.cumsum()**2))
            tables["independent_pair_deletion"].append(dict(K=k,pair=label,rms=value,
                improvement=1-value/reference[0][p],common_mask=False))
    tables["overall_key_dates"].append(overall_daily)
    curves = pd.concat({key:pd.DataFrame(value,columns=CODES,index=r.index) for key,value in plotted.items()},axis=1)
    curves.loc[pd.Timestamp(CFG["start"] if sample=="main" else CFG["long_start"])] = 0
    curves.sort_index().to_csv(OUT/f"convergence_paths_{sample}.csv",index_label="date")
    frames = {name:(pd.concat(parts,ignore_index=True) if isinstance(parts[0],pd.DataFrame) else pd.DataFrame(parts))
              .assign(sample=sample) for name,parts in tables.items()}
    return frames,masks,reference
