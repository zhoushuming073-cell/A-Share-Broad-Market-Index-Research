import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
B=Path(__file__).resolve().parent
V1=B.parent/'relative_speed_common_market_v1'
sys.path.insert(0,str(V1))
from core import load_data,kalman,filt,hac
C=json.loads((B/'config.json').read_text(encoding='utf-8-sig'))
O=B/'results'
def inputs():
    r=load_data()
    r['pos']=np.arange(len(r))
    for n in (3,4):
        f=pd.read_excel(B.parents[1]/f'outputs/inputs/CH{n}_factors_daily_202608.xlsx')
        f.index=pd.to_datetime(f.date.astype(str))
        for col in ['SMB','VMG']+(['PMO'] if n==4 else []):
            r[f'{col}{n}']=f[col]
    assert np.allclose(r.CH3_excess.dropna(),r.CH4_excess.dropna())
    return r

def design(r,model,panel):
    x=r.market if panel=='full' else r.CH4_excess
    X=pd.DataFrame({'intercept':1.,'market':x},index=r.index)
    if model=='M0':return X[['intercept']]
    if model in ('M2','M4A','M5'):X['up_extra']=x.clip(lower=0)
    if model.startswith('M4') or model=='M5':
        n=3 if model=='M4_3' else 4
        for col in ['SMB','VMG']+(['PMO'] if n==4 else []):X[col]=r[col+str(n)]
    return X

def estimate(r,panel,model,exclude=None,first=None,last=None):
    X=design(r,model,panel); names=list(X);X=X.values
    x=(r.market if panel=='full' else r.CH4_excess).values
    y=r.d.values.copy() if panel=='full' else np.expm1(r.d.values)
    if exclude is not None:y[r.index.isin(exclude)]=np.nan
    first=C['warmup'] if first is None else first
    last=len(r) if last is None else last
    # Replay from last scheduled hyperparameter refresh before the requested window.
    first=max(C['warmup'],first)
    start=C['warmup']+(first-C['warmup'])//C['refresh']*C['refresh']
    records=[]
    for t in range(start,last):
        hist=y[:t];keep=np.isfinite(hist)
        roll=int(model[3:]) if model.startswith('OLS') else None
        if roll:keep[:max(0,t-roll)]=False
        if model in ('M3','M5'):
            if (t-C['warmup'])%C['refresh']==0:
                co=np.zeros(X.shape[1])
                if model=='M5':co=np.linalg.lstsq(X[:t][keep],hist[keep],rcond=None)[0]
                target=hist-X[:t]@co
                a,b,noise,q=kalman(target,x[:t])
                _,_,_,bs,ps=filt(target,x[:t],a,b,noise,q)
                b=bs[-1];p=ps[-1]
            parts=X[t]*co
            parts[0]+=a
            extra=b*x[t]
            prediction=parts.sum()+extra
            p+=q
            if np.isfinite(y[t]):
                den=noise+p*x[t]**2
                gain=p*x[t]/den
                next_b=b+gain*(y[t]-prediction);p*=noise/den
            else:next_b=b
            beta=b+co[1];b=next_b
        else:
            co=np.linalg.lstsq(X[:t][keep],hist[keep],rcond=None)[0]
            parts=X[t]*co;extra=0.;prediction=parts.sum()
            beta=co[1] if len(co)>1 else 0.
        if t<first:continue
        if panel=='CH' and prediction<=-1:raise ValueError('conditional wealth <= 0')
        scale=np.log1p(prediction)/prediction if panel=='CH' and prediction!=0 else 1.
        row=dict(zip(names,parts*scale))
        row.update(date=r.index[t],panel=panel,model=model,beta=beta,dynamic=extra*scale,
                   fitted=prediction*scale,d=r.d.iloc[t],error=r.d.iloc[t]-prediction*scale,
                   q=q if model in ('M3','M5') else 0)
        records.append(row)
    return pd.DataFrame(records)

def loss_test(values,lag):
    a=np.asarray(values,float)
    if np.isfinite(a).sum()<3:return (np.nan,)*3
    fit,cov=hac(a,np.ones((len(a),1)),lag)
    v=fit.params[0];se=np.sqrt(max(0,cov[0,0]))
    return v,se,norm.sf(v/se) if se else np.nan
