import hashlib
import json
import platform

import numba
import numpy as np
import pandas as pd
from numba import njit, prange
from statsmodels.stats.multitest import multipletests
from data_validation import CFG, OUT, LABELS, METRICS, audit, spreads
from pairwise_metrics import describe
from key_date_overlap import overlap

OVERLAP = ["J5","max_count5","date_hhi5","J10","max_count10","date_hhi10"]

def draw(rng,n,t,method,length):
    if method=="circular":
        starts=rng.integers(t,size=(n,(t+length-1)//length))
        return ((starts[:,:,None]+np.arange(length))%t).reshape(n,-1)[:,:t].copy()
    starts=rng.integers(t,size=(n,t))
    reset=rng.random((n,t))<1/length
    for day in range(1,t):
        starts[:,day]=np.where(reset[:,day],starts[:,day],(starts[:,day-1]+1)%t)
    return starts

@njit(parallel=True,cache=True)
def evaluate(paths,d,orders):
    n,t=paths.shape
    pairs=len(d)
    stats=np.empty((n,pairs,9))
    shared=np.empty((n,6))
    for b in prange(n):
        # Linked lists enumerate repeated source days in simulated chronological order.
        head=np.full(t,-1)
        following=np.empty(t,np.int64)
        counts=np.zeros(t,np.int64)
        for pos in range(t-1,-1,-1):
            day=paths[b,pos]
            following[pos]=head[day]
            head[day]=pos
            counts[day]+=1
        flags=np.zeros((pairs,t),np.uint8)
        top=np.empty((pairs,10),np.int64)
        for p in range(pairs):
            pos_sum=neg_sum=pos_sq=neg_sq=0.
            for day in range(t):
                v=d[p,day]
                amount=counts[day]*v
                if v>0:
                    pos_sum+=amount
                    pos_sq+=amount*v
                else:
                    neg_sum-=amount
                    neg_sq+=amount*v
            signed=pos_sum-neg_sum
            gain=abs(signed)
            direction=int(signed<0)
            den=neg_sum if direction else pos_sum
            hhi=(neg_sq if direction else pos_sq)/(den*den)
            stats[b,p,5]=hhi
            stats[b,p,6]=1/hhi
            stats[b,p,7]=1-gain/(pos_sum+neg_sum)
            stats[b,p,8]=gain
            n_seen=0
            running=0.
            kstar=0
            for day in orders[direction,p]:
                pos=head[day]
                while pos>=0:
                    n_seen+=1
                    running+=abs(d[p,day])
                    if n_seen<=10:
                        top[p,n_seen-1]=pos
                        flags[p,pos]=2 if n_seen<=5 else 1
                    if kstar==0 and running>=gain:
                        kstar=n_seen
                    if n_seen==1: stats[b,p,1]=running/den
                    if n_seen==5: stats[b,p,2]=running/den
                    if n_seen==10: stats[b,p,3]=running/den
                    if n_seen==20: stats[b,p,4]=running/den
                    pos=following[pos]
                if kstar and n_seen>=20: break
            stats[b,p,0]=kstar if gain else 0
        for idx,k in enumerate((5,10)):
            jsum=0.
            frequency=np.zeros(t,np.int64)
            for p in range(pairs):
                for pos in top[p,:k]: frequency[pos]+=1
                for other in range(p):
                    common=0
                    for pos in top[p,:k]:
                        common+=flags[other,pos]>=(2 if k==5 else 1)
                    jsum+=common/(2*k-common)
            shared[b,idx*3]=jsum/(pairs*(pairs-1)/2)
            shared[b,idx*3+1]=frequency.max()
            shared[b,idx*3+2]=(frequency**2).sum()/(pairs*k)**2
    return stats,shared

def compare(sample,obs,lower):
    n=len(sample)
    p=(1+(sample<=obs if lower else sample>=obs).sum())/(n+1)
    return dict(observed=obs,p=p,mcse=np.sqrt(p*(1-p)/n),
                **dict(zip(["q025","median","q975"],np.quantile(sample,[.025,.5,.975]))))

def run(r,observed,overlaps):
    centered=r-r.mean()+r.values.mean()
    assert np.ptp(centered.mean().values)<1e-15
    d=spreads(centered)
    orders=np.ascontiguousarray([np.argsort(-d,axis=1,kind="stable"),np.argsort(d,axis=1,kind="stable")])
    digest=hashlib.sha256((OUT/"input_closes.csv").read_bytes()).hexdigest()
    rows=[]
    for setting,(method,length) in enumerate(CFG["designs"]):
        seed=CFG["seed"]+setting
        rng=np.random.default_rng(seed)
        results=[]
        globals_=[]
        n=CFG["replications"]
        for first in range(0,n,CFG["batch"]):
            ix=draw(rng,min(CFG["batch"],n-first),len(r),method,length)
            stats,shared=evaluate(ix,d,orders)
            if first==0:
                replay=centered.iloc[ix[0]].reset_index(drop=True)
                expected,keys=describe(replay)
                np.testing.assert_allclose(stats[0],expected[METRICS],rtol=1e-10,atol=1e-12)
                check=overlap(keys,replay.index)
                np.testing.assert_allclose(shared[0],[check[k] for k in OVERLAP],atol=1e-12)
            results.append(stats)
            globals_.append(shared)
        stats=np.concatenate(results)
        shared=np.concatenate(globals_)
        tag=f"{method}_{length}"
        meta=dict(config=CFG,method=method,length=length,seed=seed,input_sha256=digest,
                  numpy=np.__version__,numba=numba.__version__,python=platform.python_version(),
                  generator=type(rng.bit_generator).__name__,ties="source date then simulated position")
        np.savez_compressed(OUT/f"simulation_{tag}.npz",pair_stats=stats,overlap=shared,
                            pairs=LABELS,metrics=METRICS,overlap_metrics=OVERLAP,metadata=json.dumps(meta))
        for m,name in enumerate(METRICS):
            lower=name in ("kstar","N_eff")
            tests=[compare(stats[:,p,m],observed.iloc[p][name],lower) for p in range(len(LABELS))]
            corrected=multipletests([x["p"] for x in tests],method="holm")[1]
            for pair,res,adj in zip(LABELS,tests,corrected):
                rows.append(dict(design=tag,metric=name,pair=pair,holm=adj,**res))
            rows.append(dict(design=tag,metric=name,pair="pool_median",
                **compare(np.median(stats[:,:,m],axis=1),observed[name].median(),lower)))
        for m,name in enumerate(OVERLAP):
            rows.append(dict(design=tag,metric=name,pair="pool",**compare(shared[:,m],overlaps[name],False)))
        print("Completed",tag,n,flush=True)
    pd.DataFrame(rows).to_csv(OUT/"bootstrap_results.csv",index=False)

if __name__=="__main__":
    from data_validation import OUT
    r=audit()
    obs,keys=describe(r)
    obs.to_csv(OUT/"pairwise_summary.csv",index=False)
    keys.to_csv(OUT/"key_days.csv",index=False)
    run(r,obs,overlap(keys,r.index,save=True))

