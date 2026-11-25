import numpy as np
import pandas as pd
from numba import njit,prange
from statsmodels.stats.multitest import multipletests
from core import CFG,OUT
from simulation import measure

@njit(cache=True)
def residual_stats(d,x,order,k):
    keep=np.ones(len(d),np.bool_)
    keep[order[:k]]=False
    xx=x[keep]; yy=d[keep]
    slope=((xx-xx.mean())*(yy-yy.mean())).sum()/((xx-xx.mean())**2).sum()
    alpha=yy.mean()-slope*xx.mean()
    u=np.where(keep,d-alpha-slope*x,0.)
    z=measure(u,np.array([60]))
    cs=np.empty(len(u)+1);cs[0]=0.;cs[1:]=np.cumsum(u)
    av=(cs[60:]-cs[:-60])/60
    # Remove the two strongest non-overlapping 60-day windows in each direction.
    tops=np.empty(2)
    for side in range(2):
        score=av.copy() if side==0 else -av.copy()
        for j in range(2):
            i=np.argmax(score);score[max(0,i-59):min(len(score),i+60)]=-1e10
        tops[side]=score.max()
    return np.array([slope,z[1],z[2],z[3],-z[4],tops[0],tops[1]])

@njit(cache=True,parallel=True)
def wild(d,x,block,seed,ks,nrep):
    beta=((x-x.mean())*(d-d.mean())).sum()/((x-x.mean())**2).sum()
    fitted=d.mean()+beta*(x-x.mean());e=d-fitted
    res=np.empty((nrep,3,len(ks),7))
    for i in prange(nrep):
        np.random.seed(seed+i); yy=np.empty(len(d));xx=np.empty(len(d))
        for j in range(0,len(d),block):
            start=np.random.randint(len(d));sign=2*np.random.randint(2)-1
            for h in range(min(block,len(d)-j)):
                t=(start+h)%len(d)
                xx[j+h]=x[t];yy[j+h]=fitted[t]+sign*e[t]
        for m in range(3):
            rank=np.argsort(-(yy if m==0 else -yy if m==1 else np.abs(yy)),kind='mergesort')
            for j,k in enumerate(ks):res[i,m,j]=residual_stats(yy,xx,rank,k)
    return res

def run(r):
    y=r.d.values;x=r.market.values;ks=np.array([0,1,2,5,10,20]);rows=[]
    for b in CFG['blocks']:
        sim=wild(y,x,b,CFG['seed']+b*123456,ks,CFG['replications'])
        for m,method in enumerate('ABC'):
            rank=np.argsort(-(y if m==0 else -y if m==1 else abs(y)),kind='stable')
            for j,k in enumerate(ks):
                obs=residual_stats(y,x,rank,k)
                for h,term in enumerate(['beta','positive_run','negative_run','max60','negative_min60','third_positive60','third_negative60']):
                    a=sim[:,m,j,h];lo,hi=np.quantile(a,[.025,.975])
                    p=(1+(a>=obs[h]-1e-13).sum())/(1+len(a)) if h else np.nan
                    rows.append([b,method,k,term,obs[h],lo,hi,p])
    tab=pd.DataFrame(rows,columns='block method K term observed lo hi p'.split())
    for b,g in tab.dropna(subset=['p']).groupby('block'):
        ids=g[(g.K>0)|(g.method=='A')].index
        tab.loc[ids,'holm']=multipletests(tab.loc[ids,'p'],method='holm')[1]
        for i,row in g[g.K==0].iterrows():
            tab.loc[i,'holm']=tab.loc[(tab.block==b)&(tab.K==0)&(tab.method=='A')&(tab.term==row.term),'holm'].iloc[0]
    tab.to_csv(OUT/'residual_null.csv',index=False)
