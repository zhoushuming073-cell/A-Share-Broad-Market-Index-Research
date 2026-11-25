import numpy as np
import pandas as pd
from numba import njit,prange
from models import C,O,inputs,design

@njit(cache=True)
def extremes(y,valid):
    # [raw, local directed/absolute, global directed/absolute] by deletion budget.
    n=len(y);w=60
    out=np.zeros((5,3))
    z=y.copy();z[~valid]=0.
    ranks=np.empty((3,n),np.int64)
    for i,v in enumerate((z,-z,np.abs(z))):
        score=v.copy();score[~valid]=-np.inf
        ranks[i]=np.argsort(-score,kind='mergesort')
    global_cs=np.zeros((3,3,n+1))
    for mode in range(3):
        for j,k in enumerate((1,2,5)):
            a=np.zeros(n); ids=ranks[mode,:k];a[ids]=z[ids]
            global_cs[mode,j,1:]=np.cumsum(a)
    cs=np.zeros(n+1);cs[1:]=np.cumsum(z)
    missing=np.zeros(n+1);missing[1:]=np.cumsum(~valid)
    sorted_w=np.sort(z[:w])
    for t in range(w,n+1):
        start=t-w
        if t>w:
            old=np.searchsorted(sorted_w,z[start-1]);new=z[t-1]
            while old<w-1 and sorted_w[old+1]<new:
                sorted_w[old]=sorted_w[old+1];old+=1
            while old>0 and sorted_w[old-1]>new:
                sorted_w[old]=sorted_w[old-1];old-=1
            sorted_w[old]=new
        if missing[t]!=missing[start]:continue
        total=cs[t]-cs[start];side=0 if total>=0 else 1
        low=0;high=w-1; abs_sum=direct=0.;j=0
        for k in range(1,6):
            direct+=sorted_w[-k] if side==0 else sorted_w[k-1]
            if abs(sorted_w[low])>abs(sorted_w[high]):
                abs_sum+=sorted_w[low];low+=1
            else:abs_sum+=sorted_w[high];high-=1
            if k==1 or k==2 or k==5:
                out[0,j]=max(out[0,j],abs(total))
                out[1,j]=max(out[1,j],abs(total-direct))
                out[2,j]=max(out[2,j],abs(total-abs_sum))
                for h,mode in enumerate((side,2)):
                    agg=total-global_cs[mode,j,t]+global_cs[mode,j,start]
                    out[3+h,j]=max(out[3+h,j],abs(agg))
                j+=1
    return out

@njit(cache=True,parallel=True)
def simulate(X,target,valid,fit,resid,block,seed,N,zero,style):
    n=len(target); result=np.empty((N,5,3)); rem=np.empty(N)
    for b in prange(N):
        np.random.seed(seed+b)
        ids=np.empty(n,np.int64);sg=np.empty(n)
        for a in range(0,n,block):
            i=np.random.randint(n);v=2*np.random.randint(2)-1
            for j in range(min(block,n-a)):
                ids[a+j]=(i+j)%n;sg[a+j]=v
        ok=valid[ids]; y=resid[ids] if zero else fit[ids]+resid[ids]*sg
        d=np.log1p(y) if style else y
        result[b]=extremes(d,ok)
        rem[b]=0.
        if style:
            xx=X[ids][ok]; co=np.linalg.lstsq(xx,y[ok])[0]
            delta=d-np.log1p(X[ids]@co)
            rem[b]=extremes(delta,ok)[1,1]
    return result,rem

def run(r):
    rows=[];dist={}
    for i,model in enumerate(['zero','M1','M2','M4A']):
        style=model=='M4A';panel='CH' if style else 'full'
        X=design(r,'M0' if model=='zero' else model,panel)
        ok=X.notna().all(axis=1).values
        X=np.ascontiguousarray(X.fillna(0).to_numpy())
        y=np.expm1(r.d.values) if style else r.d.values
        co=np.linalg.lstsq(X[ok],y[ok],rcond=None)[0];fit=X@co;e=y-fit
        obs=extremes(r.d.values,ok)
        for block in C['blocks']:
            samples,rem=simulate(X,y,ok,fit,e,block,C['seed']+100000*i+1000*block,C['replications'],model=='zero',style)
            for h,rule in enumerate(['raw','local_win','local_abs','global_win','global_abs']):
                for j,k in enumerate(C['k']):
                    s=samples[:,h,j]*1e4;v=obs[h,j]*1e4
                    rows.append([model,block,rule,k,v,*np.quantile(s,[.025,.5,.975]),(1+(s>=v).sum())/(1+len(s))])
            dist[f'{model}_{block}']=samples[:,1,1]*1e4
            if style:
                v=extremes(r.d.values-np.log1p(fit),ok)[1,1]
                rows.append([model,block,'residual_win',2,v*1e4,*np.quantile(rem*1e4,[.025,.5,.975]),(1+(rem>=v).sum())/(1+len(rem))])
    pd.DataFrame(rows,columns='null block rule K observed_bp lo median hi p'.split()).to_csv(O/'null_tests.csv',index=False)
    np.savez_compressed(O/'null_draws.npz',**dist)
if __name__=='__main__':run(inputs())
