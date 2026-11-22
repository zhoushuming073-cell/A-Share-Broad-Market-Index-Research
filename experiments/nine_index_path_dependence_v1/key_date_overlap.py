from itertools import combinations
import numpy as np
import pandas as pd
from data_validation import OUT, LABELS

def overlap(keys, dates, save=False):
    aggregate = {}
    freq = pd.DataFrame(index=dates)
    pairs = []
    for k in (5,10):
        chosen = keys[keys["rank"]<=k]
        groups = chosen.groupby("pair").date.apply(set)
        counts = chosen.date.value_counts().reindex(dates,fill_value=0)
        freq[f"top{k}"] = counts
        jaccards=[]
        for a,b in combinations(groups.index,2):
            n = len(groups[a] & groups[b])
            jaccards.append(n/(2*k-n))
            pairs.append(dict(K=k,pair_a=a,pair_b=b,intersection=n,jaccard=jaccards[-1]))
        aggregate[f"J{k}"]=np.mean(jaccards)
        aggregate[f"max_count{k}"]=counts.max()
        aggregate[f"date_hhi{k}"]=np.square(counts/counts.sum()).sum()
    if save:
        freq.to_csv(OUT/"key_date_overlap.csv",index_label="date")
        pd.DataFrame(pairs).to_csv(OUT/"pair_jaccard.csv",index=False)
        freq.groupby(freq.index.to_period("M")).sum().to_csv(OUT/"monthly_overlap.csv")
    return aggregate
