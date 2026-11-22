import numpy as np
import pandas as pd
from numba import njit, prange
from statsmodels.stats.multitest import multipletests
from data_validation import CFG, LABELS, OUT, PAIRS, PAIR_IDX
from convergence_metrics import rms_and_gap

DATE_METRICS = ("J_A_U","J_A_R","J_A_B","mean_pair_J","repeat_fraction","repeat_slots")
METHODS = ("A","U","R2","B")


@njit(cache=True)
def date_stats(a,u,r,b,chosen,votes):
    out = np.empty(6)
    for i,other in enumerate((u,r,b)):
        union = np.sum(a|other)
        out[i] = np.sum(a&other)/union if np.any(other) else np.nan
    k = chosen.shape[1]
    pair_count = len(chosen)
    marked = np.zeros((pair_count,len(a)),np.bool_)
    for p in range(pair_count):
        marked[p,chosen[p]] = True
    total = 0.
    for p in range(pair_count):
        for q in range(p):
            common = np.sum(marked[q,chosen[p]])
            total += common/(2*k-common)
    out[3] = total/(pair_count*(pair_count-1)/2)
    out[4] = r.sum()/u.sum()
    out[5] = votes[r].sum()/(pair_count*k)
    return out


@njit(cache=True)
def selection(chosen, overall_order):
    n = len(overall_order)
    k = chosen.shape[1]
    votes = np.zeros(n,np.int64)
    for p in range(len(chosen)):
        for j in chosen[p]:
            votes[j]+=1
    a = np.zeros(n,np.bool_)
    b = np.zeros(n,np.bool_)
    a[overall_order[:k]] = True
    # Stable sort retains descending overall score and then calendar order.
    vote_rank = overall_order[np.argsort(-votes[overall_order],kind="mergesort")]
    b[vote_rank[:k]] = True
    return a,votes>0,votes>=2,b,votes


@njit(parallel=True,cache=True)
def equal_random(x, permutations, budgets, before, old_gap):
    sims = np.empty((len(permutations),len(budgets),23))
    pool = np.sqrt(np.mean(before**2))
    for draw in prange(len(permutations)):
        removed = np.zeros(len(x),np.bool_)
        last = 0
        for b in range(len(budgets)):
            count = budgets[b]
            removed[permutations[draw,last:count]] = True
            last = count
            rms,gap = rms_and_gap(x,removed)
            sims[draw,b,0] = 1-np.sqrt(np.mean(rms**2))/pool
            sims[draw,b,1] = old_gap-gap
            sims[draw,b,2:] = 1-rms/before
    return sims


@njit(parallel=True,cache=True)
def block_paths(x, indices, permutations, ks):
    output=np.empty((len(indices),len(ks),6))
    effects=np.empty((len(indices),len(ks),4,5))
    fixed=np.full((len(indices),len(ks),4),np.nan)
    for draw in prange(len(indices)):
        sample = x[indices[draw]]
        t = len(sample)
        overall = np.zeros(t)
        absolute = np.empty((len(PAIRS),t))
        order = np.empty((len(PAIRS),t),np.int64)
        for p in range(len(PAIRS)):
            a,b = PAIR_IDX[p]
            gap = np.abs(sample[:,a]-sample[:,b])
            absolute[p] = gap
            overall += gap
            order[p] = np.argsort(-gap,kind="mergesort")
        overall_rank = np.argsort(-overall,kind="mergesort")
        before,_ = rms_and_gap(sample,np.zeros(t,np.bool_))
        base = np.sqrt(np.mean(before**2))
        for i,k in enumerate(ks):
            a,u,r,b,votes = selection(order[:,:k],overall_rank)
            output[draw,i] = date_stats(a,u,r,b,order[:,:k],votes)
            # Fixed A and B both use k. All methods additionally use a nested,
            # equal-size uniform random mask on the same synthetic trajectory.
            for j,mask in enumerate((a,u,r,b)):
                count = mask.sum()
                rms,gap = rms_and_gap(sample,mask)
                imp = 1-np.sqrt(np.mean(rms**2))/base
                random = np.zeros(t,np.bool_)
                random[permutations[draw,:count]]=True
                random_rms,random_gap = rms_and_gap(sample,random)
                random_imp = 1-np.sqrt(np.mean(random_rms**2))/base
                effects[draw,i,j,0]=count
                effects[draw,i,j,1]=imp
                effects[draw,i,j,2]=random_imp
                effects[draw,i,j,3]=imp-random_imp
                effects[draw,i,j,4]=gap-random_gap
            fixed[draw,i,0] = effects[draw,i,0,1]
            fixed[draw,i,3] = effects[draw,i,3,1]
            strength=np.zeros(t)
            for p in range(len(order)):
                for day in order[p,:k]:
                    strength[day]=max(strength[day],absolute[p,day])
            ranked=overall_rank[np.argsort(-strength[overall_rank],kind="mergesort")]
            for j,pool_mask in enumerate((u,r),1):
                if pool_mask.sum()>=k:
                    truncated=np.zeros(t,np.bool_)
                    truncated[ranked[pool_mask[ranked]][:k]]=True
                    selected,_=rms_and_gap(sample,truncated)
                    fixed[draw,i,j]=1-np.sqrt(np.mean(selected**2))/base
    return output,effects,fixed


@njit(parallel=True,cache=True)
def shifted_sets(chosen, overall, starts, sizes, group, shifts):
    n,k=chosen.shape
    out=np.empty((len(shifts),6))
    for i in prange(len(shifts)):
        shifted=np.empty_like(chosen)
        for p in range(n):
            for j in range(k):
                day=chosen[p,j]
                h=group[day]
                shifted[p,j]=starts[h]+(day-starts[h]+shifts[i,p,h])%sizes[h]
        a,u,r,b,votes=selection(shifted,overall)
        out[i]=date_stats(a,u,r,b,shifted,votes)
    return out


def quantiles(values):
    lo,mid,hi=np.nanquantile(values,[.025,.5,.975])
    return dict(q025=lo,median=mid,q975=hi)


def upper(values,value):
    valid=values[np.isfinite(values)]
    return (1+(valid>=value-1e-13).sum())/(len(valid)+1) if len(valid) and np.isfinite(value) else np.nan


def permute(rng,n,t):
    # Independent uniform random permutations, then take nested prefixes.
    return rng.random((n,t)).argsort(axis=1).astype(np.int32)


def random_test(r, summary, reference, pair_summary):
    budgets=np.array(sorted(summary.deleted.unique()))
    rng=np.random.default_rng(CFG["seed"])
    x=np.ascontiguousarray(r.values)
    n=CFG["draws"]
    before,gap=reference
    draws=np.concatenate([equal_random(x,permute(rng,min(CFG["batch"],n-i),len(x)),budgets,before,gap)
                         for i in range(0,n,CFG["batch"])])
    np.savez_compressed(OUT/"random_deletion_draws.npz",draws=draws,budgets=budgets,
                        metrics=["pool_improvement","terminal_gap_change",*LABELS])
    rows=[]
    scores=summary.drop_duplicates("mask_id")
    for row in scores.itertuples():
        values=draws[:,np.searchsorted(budgets,row.deleted)]
        rows.append(dict(mask_id=row.mask_id,deleted=row.deleted,observed=row.improvement,
                         p=upper(values[:,0],row.improvement),**quantiles(values[:,0]),
                         gap_observed=row.terminal_gap_change,gap_p=upper(values[:,1],row.terminal_gap_change),
                         **{"gap_"+k:v for k,v in quantiles(values[:,1]).items()}))
    df=pd.DataFrame(rows)
    valid=df.deleted.gt(0)
    for metric in ("p","gap_p"):
        df.loc[valid,metric+"_holm"]=multipletests(df.loc[valid,metric],method="holm")[1]
    df.to_csv(OUT/"random_deletion_results.csv",index=False)
    pair_rows=[]
    for row in scores.itertuples():
        if row.deleted==0:
            continue
        values=draws[:,np.searchsorted(budgets,row.deleted),2:]
        observed=pair_summary.loc[pair_summary.mask_id.eq(row.mask_id)].drop_duplicates("pair")
        comparisons=[upper(values[:,j],p.improvement) for j,p in enumerate(observed.itertuples())]
        adjusted=multipletests(comparisons,method="holm")[1]
        for j,p in enumerate(observed.itertuples()):
            pair_rows.append(dict(mask_id=row.mask_id,deleted=row.deleted,pair=p.pair,
                                  observed=p.improvement,p=comparisons[j],holm=adjusted[j],**quantiles(values[:,j])))
    pd.DataFrame(pair_rows).to_csv(OUT/"random_pair_results.csv",index=False)
    return draws,budgets


def date_test(r, pair_scores, order):
    overall=np.argsort(-pair_scores.mean(axis=0),kind="stable")
    group,levels=pd.factorize(r.index.to_period("Q"),sort=False)
    sizes=np.bincount(group)
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    rng=np.random.default_rng(CFG["seed"]+1)
    rows=[]
    controls=[]
    for k in CFG["k"]:
        chosen=order[:,:k]
        a,u,rep,b,votes=selection(chosen,overall)
        observed=date_stats(a,u,rep,b,chosen,votes)
        shifts=rng.integers(sizes,size=(CFG["draws"],len(PAIRS),len(sizes)),dtype=np.int32)
        draws=shifted_sets(chosen,overall,starts,sizes,group,shifts)
        controls.append(draws)
        for name,v,values in zip(DATE_METRICS,observed,draws.T):
            rows.append(dict(K=k,metric=name,observed=v,p=upper(values,v),valid=np.isfinite(values).sum(),
                             **quantiles(values)))
    frame=pd.DataFrame(rows)
    valid=frame.p.notna()
    frame.loc[valid,"holm"]=multipletests(frame.loc[valid,"p"],method="holm")[1]
    frame.to_csv(OUT/"date_randomization_results.csv",index=False)
    np.savez_compressed(OUT/"date_randomization_draws.npz",draws=np.stack(controls),K=CFG["k"],metrics=DATE_METRICS)


def bootstrap(r):
    x=np.ascontiguousarray(r.values)
    n,t=CFG["draws"],len(x)
    rows=[]
    for length in CFG["blocks"]:
        rng=np.random.default_rng(CFG["seed"]+100+length)
        overlap,effect,fixed=[],[],[]
        for i in range(0,n,CFG["batch"]):
            batch=min(CFG["batch"],n-i)
            starts=rng.integers(t,size=(batch,(t+length-1)//length))
            ids=((starts[:,:,None]+np.arange(length))%t).reshape(batch,-1)[:,:t].copy()
            dates,values,matched=block_paths(x,ids,permute(rng,batch,t),np.array(CFG["k"]))
            overlap.append(dates)
            effect.append(values)
            fixed.append(matched)
        overlap,effect,fixed=np.concatenate(overlap),np.concatenate(effect),np.concatenate(fixed)
        fields=("deleted","improvement","random_improvement","excess_improvement","gap_minus_random")
        np.savez_compressed(OUT/f"bootstrap_block_{length}.npz",overlap=overlap,effects=effect,
                            fixed_budget_improvement=fixed,K=CFG["k"],
                            overlap_metrics=DATE_METRICS,effect_metrics=fields,methods=METHODS)
        for j,k in enumerate(CFG["k"]):
            for m,name in enumerate(DATE_METRICS):
                rows.append(dict(block=length,K=k,method="date",metric=name,**quantiles(overlap[:,j,m])))
            for m,method in enumerate(METHODS):
                for c,name in enumerate(fields):
                    values=effect[:,j,m,c]
                    rows.append(dict(block=length,K=k,method=method,metric=name,
                                     positive_share=np.mean(values>1e-13),**quantiles(values)))
                values=fixed[:,j,m]
                rows.append(dict(block=length,K=k,method=method,metric="fixed_K_improvement",
                                 valid=np.isfinite(values).sum(),**quantiles(values)))
        print(f"Completed {n:,} synchronized paths, block={length}",flush=True)
    pd.DataFrame(rows).to_csv(OUT/"bootstrap_results.csv",index=False)
