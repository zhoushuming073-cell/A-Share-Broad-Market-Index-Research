"""Causal daily hedge signals, portfolio accounting and statistics."""
from itertools import permutations
import warnings

from numba import njit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

from research_common import np, pd

warnings.filterwarnings("ignore",message="adfuller currently returns a plain tuple",category=FutureWarning)


def rolling_tests(prices, dates):
    logp = np.log(prices.loc[dates])
    quarters = dates.to_period("Q")
    rows = []
    for k in np.flatnonzero(np.r_[True,quarters[1:]!=quarters[:-1]]):
        if k < 505:
            continue
        history = logp.iloc[k-505:k-1]
        unit = {c: (adfuller(history[c],maxlag=5)[1],adfuller(history[c].diff().dropna(),maxlag=5)[1]) for c in logp}
        for a,b in permutations(logp.columns,2):
            x,y = history[b],history[a]
            slope = x.cov(y)/x.var()
            intercept = y.mean()-slope*x.mean()
            z = y-intercept-slope*x
            phi = z.autocorr() * z.iloc[1:].std()/z.iloc[:-1].std()
            stat,p,_ = coint(y,x,maxlag=5,autolag="aic")
            rows.append(dict(effective=dates[k],fit_end=history.index[-1],a=a,b=b,intercept=intercept,
                             slope=slope,scale=z.std(),eg_p=p,eg_stat=stat,adf_p=adfuller(z,maxlag=5)[1],
                             level_a=unit[a][0],level_b=unit[b][0],diff_a=unit[a][1],diff_b=unit[b][1],
                             half_life=np.log(.5)/np.log(phi) if 0<phi<1 else np.nan))
    df = pd.DataFrame(rows)
    df["eg_q"] = df.groupby("effective").eg_p.transform(lambda x:multipletests(x,method="fdr_bh")[1])
    df["pass_gate"] = (df.eg_q.lt(.05)&df.adf_p.lt(.05)&df.slope.gt(0)&df.half_life.between(1,126)
                       &df.level_a.gt(.05)&df.level_b.gt(.05)&df.diff_a.lt(.05)&df.diff_b.lt(.05))
    return df


@njit
def signals(z, allowed, boundary, entry, exit_z):
    result = np.zeros(len(z))
    side = age = 0
    for k in range(len(z)):
        if boundary[k]:
            side = age = 0
        if not allowed[k] or not np.isfinite(z[k]):
            side = age = 0
        elif side:
            age += 1
            if side*z[k] >= -exit_z or side*z[k] <= -4 or age >= 60:
                side = age = 0
        elif entry<=abs(z[k])<4:
            side = -1 if z[k]>0 else 1
        result[k] = side
    return result


def specifications(prices,r,a,b,train,tests):
    x,y = r[b],r[a]
    h = x.loc[train].cov(y.loc[train])/x.loc[train].var()
    yield "one","fixed",0,"D",0,0,np.ones(len(r)),np.ones(len(r))
    yield "static","static",0,"D",0,0,np.full(len(r),h),np.ones(len(r))
    for method,windows in [("rolling",[60,120,252,504]),("ewma",[.94,.97,.99]),("vol",[60,120,252,504])]:
        for w in windows:
            if method=="ewma":
                ex=x.ewm(alpha=1-w,adjust=False,min_periods=60)
                est=ex.cov(y)/ex.var()
            elif method=="vol":
                est=y.rolling(w).std()/x.rolling(w).std()
            else:
                est=x.rolling(w).cov(y)/x.rolling(w).var()
            est=est.shift(2)  # earn close t-1 -> t using information through close t-2
            for freq,period in [("D","D"),("W","W-FRI"),("M","M"),("Q","Q")]:
                stamps=r.index.to_period(period)
                update=np.r_[True,stamps[1:]!=stamps[:-1]]
                # Retain invalid updates as cash until the next scheduled estimate.
                positions=np.maximum.accumulate(np.where(update,np.arange(len(r)),0))
                values=est.to_numpy()[positions]
                yield f"{method}_{w}_{freq}",method,w,freq,0,0,values,np.ones(len(r))
    t=tests[(tests.a==a)&(tests.b==b)].set_index("effective")
    estimates=t.reindex(r.index,method="ffill")
    z=(np.log(prices[a]).shift(2).reindex(r.index)-estimates.intercept
       -estimates.slope*np.log(prices[b]).shift(2).reindex(r.index))/estimates.scale
    for ent in (1.5,2.,2.5):
        for ex in (0.,.5):
            side=signals(z.to_numpy(),estimates.pass_gate.eq(True).to_numpy(),
                         r.index.isin(t.index),ent,ex)
            yield f"mr_{ent}_{ex}","mr",504,"Q",ent,ex,estimates.slope.to_numpy(),side


def weights(h,side):
    active=np.isfinite(h)&(h>0)&(side!=0)
    w=np.divide(side,1+h,out=np.zeros_like(side),where=active)
    return w,np.where(active,-w*h,0.)


@njit
def account(x,y,w,v,days,trade,fee,borrow,liquidate=True):
    # Holdings and traded amounts in currency per one unit of initial capital.
    n=len(x)
    result=np.zeros((n,8))
    equity=1.
    pos_a=pos_b=0.
    for k in range(n):
        start=equity
        traded=0.
        if trade[k] or k==0:
            after=equity
            for _ in range(8):
                after=equity-fee*(abs(w[k]*after-pos_a)+abs(v[k]*after-pos_b))
            a,b=w[k]*after,v[k]*after
            traded=abs(a-pos_a)+abs(b-pos_b)
            pos_a,pos_b=a,b
            equity-=fee*traded
        short=max(-pos_a,0.)+max(-pos_b,0.)
        long=max(pos_a,0.)+max(pos_b,0.)
        carry=short*borrow*days[k]/365.
        equity+=pos_a*x[k]+pos_b*y[k]-carry
        pos_a*=1+x[k]
        pos_b*=1+y[k]
        if k==n-1 and liquidate:
            unwind=abs(pos_a)+abs(pos_b)
            equity-=fee*unwind
            traded+=unwind
        if equity<=0:
            raise ValueError("portfolio insolvent")
        result[k]=np.array([equity/start-1,equity,traded,long/start,-short/start,carry,fee*traded,traded/start])
    return result


def metrics(r):
    r=np.asarray(r); n=len(r)
    if not n:
        return dict(n=0,total=np.nan,annual=np.nan,vol=np.nan,sharpe=np.nan,mdd=np.nan,win=np.nan)
    nav=np.r_[1.,(1+r).cumprod()]
    vol=r.std(ddof=1)*np.sqrt(252) if n>1 else np.nan
    return dict(n=n,total=nav[-1]-1,annual=np.expm1(np.log1p(r).mean()*252),vol=vol,
                sharpe=r.mean()*252/vol if vol>1e-12 else np.nan,
                mdd=np.min(nav/np.maximum.accumulate(nav)-1),win=np.mean(r>0))


def hac(r):
    e=np.asarray(r)-np.mean(r)
    n=len(e)
    variance=e@e
    for lag in range(1,min(n,6)):
        variance+=2*(1-lag/6)*(e[lag:]@e[:-lag])
    se=np.sqrt(max(variance,0))/n
    return se,2*norm.sf(abs(np.mean(r))/se) if se else 1.
