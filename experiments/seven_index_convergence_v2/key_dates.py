from itertools import combinations

import numpy as np
import pandas as pd

from data_validation import LABELS, PAIRS, PAIR_IDX


def scores(returns):
    return np.abs(returns[:, PAIR_IDX[:, 0]] - returns[:, PAIR_IDX[:, 1]]).T


def rank(score):
    return np.argsort(-score, kind="stable")


def memberships(pair_scores, order, overall, k):
    """A day is a column numbered in ascending calendar order."""
    chosen = order[:, :k]
    votes = np.bincount(chosen.ravel(), minlength=len(overall))
    strength = np.zeros(len(overall))
    np.maximum.at(strength, chosen.ravel(), np.take_along_axis(pair_scores, chosen, 1).ravel())
    vote_rank = np.lexsort((np.arange(len(overall)), -overall, -votes))
    set_rank = np.lexsort((np.arange(len(overall)), -overall, -strength))
    return dict(A=rank(overall)[:k], U=set_rank[votes[set_rank]>0],
                **{f"R{q}": set_rank[votes[set_rank]>=q] for q in (2,3,5,10)},
                I21=set_rank[votes[set_rank]==21], B=vote_rank[:k]), votes, vote_rank, chosen


def overlap(a,b):
    a,b = set(a),set(b)
    common = len(a & b)
    union = len(a | b)
    return dict(intersection=common, union=union, jaccard=common/union if a and b else np.nan,
                cover_A=common/len(a) if a and b else np.nan,
                cover_other=common/len(b) if a and b else np.nan,
                only_A=len(a-b), only_other=len(b-a))


def detail(r,pair_scores,overall,k,sets,votes,chosen):
    dates = r.index
    rank_by_date = np.empty(len(overall), int)
    rank_by_date[rank(overall)] = np.arange(1,len(overall)+1)
    rows = []
    for p, label in enumerate(LABELS):
        for pos, day in enumerate(chosen[p],1):
            rows.append(dict(K=k, date=dates[day], pair=label, rank=pos, absolute_gap=pair_scores[p,day]))
    pair_days = pd.DataFrame(rows)
    references = pair_days.assign(tag=lambda f: f.pair + "@" + f["rank"].astype(str)).groupby("date").tag.agg("|".join)
    table = pd.DataFrame(dict(K=k, date=dates, votes=votes, score=overall, overall_rank=rank_by_date))
    table["pair_and_rank"] = table.date.map(references).fillna("")
    for q in sets:
        table[q] = np.isin(np.arange(len(dates)), sets[q])
    compared = []
    differences = []
    for method in ("U","R2","B"):
        a,b = set(sets["A"]), set(sets[method])
        compared.append(dict(K=k, other=method, size_A=len(a), size_other=len(b), **overlap(a,b)))
        for status, chosen_days in (("both",a&b),("A_only",a-b),(method+"_only",b-a)):
            differences.extend(dict(K=k, other=method, status=status, date=dates[i]) for i in sorted(chosen_days))
    jac = np.eye(len(PAIRS))
    jrows = []
    for p,q in combinations(range(len(PAIRS)),2):
        num = len(set(chosen[p]) & set(chosen[q]))
        jac[p,q] = jac[q,p] = num/(2*k-num)
        jrows.append(dict(K=k, first=LABELS[p], second=LABELS[q], common=num, jaccard=jac[p,q],
                          shared_index=bool(set(PAIRS[p]) & set(PAIRS[q]))))
    categories = [(1,1),(2,2),(3,4),(5,9),(10,20),(21,21)]
    counts = table.loc[table.votes.gt(0),"votes"]
    freq = []
    for lo,hi in categories:
        selected = table[table.votes.between(lo,hi)]
        freq.append(dict(K=k, low=lo, high=hi, days=len(selected), share=len(selected)/len(counts),
                         slots=selected.votes.sum(),
                         dates="|".join(selected.date.dt.strftime("%Y-%m-%d"))))
    summary = dict(K=k, unique=len(sets["U"]), repeat=len(sets["R2"]), all_21=len(sets["I21"]),
                   repeat_fraction=len(sets["R2"])/len(sets["U"]),
                   repeat_slots=votes[votes>=2].sum()/(21*k), all_fraction=len(sets["I21"])/len(sets["U"]))
    return pair_days, table, compared, differences, jrows, jac, freq, summary
