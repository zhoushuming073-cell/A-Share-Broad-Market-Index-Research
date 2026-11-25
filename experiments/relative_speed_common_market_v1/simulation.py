import numpy as np
import pandas as pd
from numba import njit,prange
from statsmodels.stats.multitest import multipletests
from core import CFG,OUT,load_data

@njit(cache=True)
def measure(y,windows):
    n=len(y)
    cum=np.empty(n+1);cum[0]=0.
    pos=neg=pmax=nmax=0
    for t in range(n):
        cum[t+1]=cum[t]+y[t]
        pos=(pos+1) if y[t]>0 else 0
        neg=(neg+1) if y[t]<0 else 0
        pmax=max(pos,pmax);nmax=max(neg,nmax)
    out=np.empty(3+2*len(windows))
    out[0],out[1],out[2]=cum[-1]/n,pmax,nmax
    for j,w in enumerate(windows):
        diff=(cum[w:]-cum[:-w])/w
        out[3+2*j]=diff.max();out[4+2*j]=diff.min()
    return out

@njit(cache=True,parallel=True)
def batch(panel,block,seed,ks,ws,reps):
    n=len(panel)
    out=np.empty((reps,3,len(ks),3+2*len(ws)))
    drift=np.mean(panel[:,0]-panel[:,1])
    for i in prange(reps):
        np.random.seed(seed+i)
        v=np.empty(n)
        if block:
            for s in range(0,n,block):
                first=np.random.randint(n)
                for j in range(min(block,n-s)):
                    # Shared row index selects both legs and matching market observation.
                    row=panel[(first+j)%n]
                    v[s+j]=row[0]-row[1]-drift
        else:
            v=panel[:,0]-panel[:,1]
        for m in range(3):
            if block:
                z=v if m==0 else (-v if m==1 else np.abs(v))
                order=np.argsort(-z,kind='mergesort')
            else:
                order=np.random.permutation(n)
            y=v.copy()
            for j,k in enumerate(ks):
                if j:
                    y[order[ks[j-1]:k]]=0
                out[i,m,j]=measure(y,ws)
    return out

def run(r):
    ks=np.array(CFG['k']);ws=np.array(CFG['windows'])
    names=['mean','positive_run','negative_run']+[f'{s}_{w}' for w in ws for s in ['max','min']]
    rows=[]
    for b in [0,*CFG['blocks']]:
        sims=batch(r[['a','b','market']].to_numpy().copy(order='C'),b,CFG['seed']+b*100000,ks,ws,CFG['replications'])
        for m,method in enumerate('ABC'):
            y=r.d.to_numpy()
            order=np.argsort(-(y if m==0 else -y if m==1 else abs(y)),kind='stable')
            for j,k in enumerate(ks):
                obs=y.copy();obs[order[:k]]=0
                for h,(name,v) in enumerate(zip(names,measure(obs,ws))):
                    x=sims[:,m,j,h]
                    lo,med,hi=np.quantile(x,[.025,.5,.975])
                    p=min(1.,2*min((1+(x>=v-1e-14).sum())/(1+len(x)),(1+(x<=v+1e-14).sum())/(1+len(x))))
                    rows.append([b,method,k,name,v,lo,med,hi,p])
    t=pd.DataFrame(rows,columns='block method K metric observed q025 median q975 p'.split())
    t['family']=t.metric.str.split('_').str[0]
    for _,g in t.groupby(['block','family']):
        unique=g[(g.K!=0)|(g.method=='A')]
        adj=dict(zip(unique.index,multipletests(unique.p,method='holm')[1]))
        for i,row in g.iterrows():
            ix=i if i in adj else unique.index[(unique.K==0)&(unique.metric==row.metric)][0]
            t.loc[i,'holm']=adj[ix]
    t.to_csv(OUT/'simulations.csv',index=False)

if __name__=='__main__':run(load_data())
