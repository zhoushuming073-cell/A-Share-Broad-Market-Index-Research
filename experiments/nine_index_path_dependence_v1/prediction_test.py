import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from data_validation import OUT,CFG,spreads,LABELS

def features(d,k):
    w=np.lib.stride_tricks.sliding_window_view(d,k)
    trend=w.mean(axis=1)
    z=w*np.where(trend>=0,1,-1)[:,None]
    c5=np.partition(z,-5,axis=1)[:,-5:].sum(axis=1)/z.clip(min=0).sum(axis=1)
    sd=w.std(axis=1)
    extreme=(abs(w)>2*sd[:,None]).mean(axis=1)
    return np.column_stack([np.ones(len(w)),sd,abs(w).mean(axis=1),c5,trend,extreme])

def dm(loss_diff,h):
    x=loss_diff-loss_diff.mean()
    n=len(x)
    lrv=x@x/n
    for lag in range(1,h):
        lrv+=2*(1-lag/h)*(x[lag:]@x[:-lag])/n
    stat=loss_diff.mean()/np.sqrt(lrv/n)
    return stat,2*norm.sf(abs(stat))

def run(r):
    k,train=CFG["lookback"],CFG["train"]
    scores=[]
    daily=[]
    for pair,d in zip(LABELS,spreads(r)):
        x=features(d,k)
        for h in CFG["horizons"]:
            future=np.sqrt(np.lib.stride_tricks.sliding_window_view(d[k:]**2,h).sum(axis=1))
            predictions=[]
            rows=range(train+h-1,len(future))
            for t in rows:
                end=t-h+1
                xx=x[end-train:end]
                yy=future[end-train:end]
                forecast=[max(0.,x[t,:n]@np.linalg.lstsq(xx[:,:n],yy,rcond=None)[0]) for n in (3,4,6)]
                predictions.append([yy.mean(),*forecast])
            preds=np.asarray(predictions)
            y=future[rows]
            err=y[:,None]-preds
            loss=err**2
            dates=r.index[np.asarray(rows)+k-1]
            daily.append(pd.DataFrame(dict(pair=pair,h=h,origin=dates,target=y,mean=preds[:,0],
                base=preds[:,1],expanded=preds[:,2],robust=preds[:,3])))
            for col,name in enumerate(("base","expanded","robust"),1):
                diff=loss[:,1]-loss[:,col]
                stat,p=dm(diff,h) if col>1 else (np.nan,np.nan)
                scores.append(dict(pair=pair,h=h,model=name,start=dates[0],end=dates[-1],n=len(y),
                    R2_vs_mean=1-loss[:,col].sum()/loss[:,0].sum(),
                    R2_vs_base=1-loss[:,col].sum()/loss[:,1].sum(),mae=abs(err[:,col]).mean(),
                    rmse=np.sqrt(loss[:,col].mean()),dm=stat,p=p))
    scores=pd.DataFrame(scores)
    for (_,model),g in scores.groupby(["h","model"]):
        if model!="base":
            scores.loc[g.index,"holm"]=multipletests(g.p,method="holm")[1]
    scores.to_csv(OUT/"prediction_results.csv",index=False)
    pd.concat(daily).to_csv(OUT/"predictions.csv.gz",index=False)
