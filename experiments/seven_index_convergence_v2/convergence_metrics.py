import numpy as np
import pandas as pd
from numba import njit
from data_validation import CODES, LABELS, PAIR_IDX


@njit(cache=True)
def rms_and_gap(x, removed):
    """Keep all time positions, including dates on which the path is held flat."""
    t,n = x.shape
    wealth = np.zeros(n)
    pair_ss = np.zeros(len(PAIR_IDX))
    for day in range(t):
        if not removed[day]:
            wealth += x[day]
        for p in range(len(PAIR_IDX)):
            a,b = PAIR_IDX[p]
            pair_ss[p] += (wealth[a]-wealth[b])**2
    return np.sqrt(pair_ss/t), np.exp(wealth.max())-np.exp(wealth.min())


def evaluate(x, slots, baseline):
    mask = np.zeros(len(x),dtype=bool)
    mask[slots] = True
    rms,gap = rms_and_gap(x,mask)
    before, old_gap = baseline
    improve = 1-rms/before
    cum = np.where(mask[:,None],0,x).cumsum(axis=0)
    retained = x[~mask]
    corr = np.corrcoef(retained,rowvar=False)[tuple(PAIR_IDX.T)]
    old_corr = np.corrcoef(x,rowvar=False)[tuple(PAIR_IDX.T)]
    pool = np.sqrt(np.mean(rms**2))
    result = dict(deleted=len(slots),fraction=len(slots)/len(x),rms=pool,
                  improvement=1-pool/np.sqrt(np.mean(before**2)),terminal_log_sd=cum[-1].std(ddof=0),
                  terminal_gap=gap,terminal_gap_change=old_gap-gap,mean_corr=corr.mean(),
                  mean_corr_change=(corr-old_corr).mean(),converged=int((improve>1e-12).sum()),
                  widened=int((improve< -1e-12).sum()),same=int((abs(improve)<=1e-12).sum()))
    pair = pd.DataFrame(dict(pair=LABELS,rms=rms,improvement=improve,corr=corr,corr_change=corr-old_corr))
    original = np.expm1(x.sum(axis=0))
    final = np.expm1(cum[-1])
    ends = pd.DataFrame(dict(code=CODES,original_return=original,after_return=final,change=final-original))
    return result,pair,ends,np.expm1(cum)
