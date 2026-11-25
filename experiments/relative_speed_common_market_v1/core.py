from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import yaml
from numba import njit
from scipy.optimize import minimize
import statsmodels.api as sm

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=BASE/'results'
CFG=yaml.safe_load((BASE/'config.yaml').read_text(encoding='utf-8-sig'))
OLD=ROOT/'experiments/relative_return_frame_deletion_v1/results'

def load_data():
    def prices(file):
        return pd.read_csv(ROOT/file,index_col=0,parse_dates=True)
    ab=prices('experiments/four_pair_tests_v2/data/input_closes.csv')[['000300','000001']]
    long=prices('experiments/seven_index_convergence_v2/results/input_closes_long.csv')
    assert ab.index.equals(long.index) and ab.index.is_unique and ab.index.is_monotonic_increasing
    np.testing.assert_allclose(ab,long[ab.columns],atol=1e-8,rtol=0)
    p=long[['000300','000001','000985']]
    assert np.isfinite(p).all().all() and (p>0).all().all()
    assert list(p.index[[0,-1]].strftime('%Y-%m-%d'))==CFG['sample']
    with sqlite3.connect((ROOT/'data/a_share_indices.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        sql=pd.read_sql_query("select trade_date,index_code,close,close_return from index_daily where index_code in ('000300','000001','000985')",db,parse_dates=['trade_date'])
    for col,expected in [('close',p),('close_return',p.pct_change())]:
        frame=sql.pivot(index='trade_date',columns='index_code',values=col).reindex(columns=p.columns)
        assert frame.index.equals(p.index)
        np.testing.assert_allclose(frame.iloc[1:] if col=='close_return' else frame,expected.iloc[1:] if col=='close_return' else expected,atol=1e-8,rtol=0)
    r=np.log(p).diff().dropna()
    r.columns=['a','b','market']
    r['d']=r.a-r.b
    old=pd.read_csv(OLD/'daily_relative_returns.csv',index_col='date',parse_dates=True)
    assert old.index.equals(r.index)
    np.testing.assert_allclose(old.d,r.d,atol=1e-13)
    for tail,asc in [('positive',False),('negative',True)]:
        ranks=pd.read_csv(OLD/f'top_{tail}_days.csv')
        assert list(r.d.sort_values(ascending=asc).index[:20].strftime('%Y-%m-%d'))==list(ranks.date[:20])
    for model in ['CH3','CH4']:
        f=pd.read_excel(ROOT/f'outputs/inputs/{model}_factors_daily_202608.xlsx')
        f.index=pd.to_datetime(f.date.astype(str))
        assert f.index.is_unique and f[['mktrf','rf_dly']].notna().all().all()
        r[model]=np.log1p(f.mktrf+f.rf_dly)
        r[model+'_excess']=f.mktrf
    return r

def ranked(d,method):
    score={'A':d,'B':-d,'C':abs(d)}[method]
    return np.argsort(-score,kind='stable')

@njit(cache=True)
def filt(y,x,alpha,beta,noise,q):
    n=len(y)
    pred=np.empty(n); var=np.empty(n); post=np.empty(n); pv=np.empty(n)
    p=0.; ll=0.
    for t in range(n):
        p+=q
        pred[t]=beta; var[t]=p
        if np.isfinite(y[t]):
            e=y[t]-alpha-beta*x[t]
            s=noise+p*x[t]**2
            ll+=np.log(s)+e**2/s
            beta+=p*x[t]/s*e
            p*=noise/s
        post[t]=beta; pv[t]=p
    return ll,pred,var,post,pv

def kalman(y,x):
    keep=np.isfinite(y)
    alpha,beta=np.linalg.lstsq(np.column_stack([np.ones(keep.sum()),x[keep]]),y[keep],rcond=None)[0]
    noise=np.mean((y[keep]-alpha-beta*x[keep])**2)
    null=(alpha,beta,noise,0.)
    def unpack(v):return v[0]*.001,v[1],np.exp(v[2]),np.exp(v[3])
    def loss(v):return filt(y,x,*unpack(v))[0]/keep.sum()
    res=minimize(loss,[alpha*1000,beta,np.log(noise),-10.],method='L-BFGS-B',bounds=[(None,None),(None,None),(-24,-5),(-28,0)])
    if not res.success:raise RuntimeError(res.message)
    fit=unpack(res.x)
    return fit if filt(y,x,*fit)[0]<filt(y,x,*null)[0]-1e-6 else null

def hac(y,X,lag):
    # Preserve calendar spacing when selected dates are omitted from estimation.
    keep=np.isfinite(y)&np.isfinite(X).all(axis=1)
    fit=sm.OLS(y[keep],X[keep]).fit()
    score=np.zeros_like(X); score[keep]=X[keep]*fit.resid[:,None]
    meat=score.T@score
    for j in range(1,lag+1):
        cov=score[j:].T@score[:-j]
        meat+=(1-j/(lag+1))*(cov+cov.T)
    bread=fit.normalized_cov_params
    return fit,bread@meat@bread
