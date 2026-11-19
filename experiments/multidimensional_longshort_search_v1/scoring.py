from data import np,CFG

def longest(mask):
    edge=np.diff(np.r_[False,mask,False].astype(int))
    return int(np.max(np.flatnonzero(edge==-1)-np.flatnonzero(edge==1),initial=0))
def metrics(a,month,year,market):
    u,v=a[:,0],a[:,1];n=len(u);dd=1-v/np.maximum.accumulate(np.r_[1.,v])[1:]
    vol=u.std(ddof=1)*np.sqrt(252);cagr=v[-1]**(252/n)-1
    neg=np.sqrt(np.mean(np.minimum(u,0)**2))*np.sqrt(252)
    def grouped(keys):
        ends=np.r_[np.where(keys[1:]!=keys[:-1])[0],n-1]
        return v[ends]/np.r_[1.,v[ends[:-1]]]-1
    mo=grouped(month);yr=grouped(year)
    lv=np.log(v);t=np.arange(n);slope=np.dot(t-t.mean(),lv-lv.mean())/np.sum((t-t.mean())**2)
    gains=np.maximum(u,0)
    underwater=dd>1e-10
    ends=np.flatnonzero(np.diff(np.r_[False,underwater,False].astype(int))==-1)
    starts=np.flatnonzero(np.diff(np.r_[False,underwater,False].astype(int))==1)
    duration=ends-starts
    return dict(net=v[-1]-1,cagr=cagr,vol=vol,sharpe=u.mean()*252/max(vol,1e-12),
        sortino=u.mean()*252/max(neg,1e-12),mdd=dd.max(),calmar=cagr/max(dd.max(),.001),
        duration=longest(underwater),recovery=np.mean(duration[ends<n]) if np.any(ends<n) else 0.,
        underwater=underwater.mean(),ulcer=np.sqrt(np.mean(dd**2)),months=(mo>0).mean(),
        losing_months=longest(mo<0),years=(yr>0).mean(),worst_year=yr.min(),slope=slope*252,
        residual=np.std(lv-slope*t),concentration=np.sort(gains)[-5:].sum()/max(gains.sum(),1e-12),
        active=np.mean(a[:,11]>1e-7),beta=np.cov(u,market)[0,1]/market.var(ddof=1),
        turnover=np.sum(a[:,4]/a[:,10])*252/n,cost=np.sum(a[:,5:8].sum(axis=1)/a[:,10])*252/n,
        fee=a[:,5].sum(),borrow=a[:,6].sum(),finance=a[:,7].sum())
def score(m,complexity,stability=.5,forward=None):
    clip=lambda x:np.clip(x,0.,1.)
    growth=clip(m["cagr"]/.12);risk=(clip(1-m["mdd"]/.25)+clip(1-m["duration"]/500))/2
    sharp=(clip(m["sharpe"]/2)+clip(m["sortino"]/3))/2
    smooth=(m["months"]+clip(1-m["ulcer"]/.10))/2
    cost=(clip(1-m["cost"]/.04)+clip(1-m["turnover"]/80))/2
    ideal=np.dot(CFG["ideal_weights"],[growth,risk,sharp,smooth,m["years"],cost])
    fw=m["years"] if forward is None else clip(.5+forward/.15)
    dims=[growth,(risk+clip(1-m["vol"]/.20))/2,sharp,stability,fw,cost,clip(1-abs(m["beta"])),clip(1-complexity/15)]
    overall=np.dot(CFG["overall_weights"],dims)
    alt=np.dot([.15,.22,.15,.18,.12,.08,.06,.04],dims)
    gate=CFG["ideal_gate"]
    qualified=(m["cagr"]>=gate["cagr"] and m["mdd"]<=gate["drawdown"] and m["sharpe"]>=gate["sharpe"] and
        m["active"]>=gate["active"] and m["years"]>=gate["positive_years"] and m["slope"]>0)
    return float(ideal),float(overall),bool(qualified),float(alt)
