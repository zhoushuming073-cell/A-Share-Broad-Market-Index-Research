from metrics import *
ORDER={"best":np.argsort(-D,kind="stable"),"worst":np.argsort(D,kind="stable"),"absolute":np.argsort(-abs(D),kind="stable")}
def mask(rule,k):
    deleted=np.r_[ORDER["worst"][:k//2],ORDER["best"][:k//2]] if rule=="symmetric" else ORDER[rule][:k]
    keep=np.ones(N,bool);keep[deleted]=False
    assert (~keep).sum()==k
    return keep
def flip():
    if D.sum()<=0:return None
    hits=np.flatnonzero(D.sum()-np.cumsum(D[ORDER["best"]])<=0)
    return int(hits[0]+1) if len(hits) else None
def counts():
    regular={count(p) for p in CFG["proportions"]}
    return sorted(regular|set(CFG["counts"])|{2*count(p) for p in CFG["tail_each"]}|({flip()} if flip() else set()))
def build():
    orig=describe(np.ones(N,bool));save([orig],"original_metrics")
    daily=pd.DataFrame(dict(date=T,a=A,b=B,return_a=np.expm1(A),return_b=np.expm1(B),d=D,abs_d=abs(D),cum_D=np.cumsum(D)))
    spread=pd.Series(D,index=T)
    for k in (20,60,120,252):daily[f"rolling_Q_{k}"]=np.expm1(spread.rolling(k).sum()).values
    # Lagged 120-session benchmark momentum; state describes the day before each return.
    market=LOG.iloc[:,0].rolling(120).sum().shift(1)
    daily["state"]=np.select([market.isna(),market>.05,market<-.05],["warmup","up","down"],default="range")
    save(daily,"daily_relative_returns")
    for rule,f in [("best","positive"),("worst","negative")]:
        sub=daily.iloc[ORDER[rule]].copy()
        sub=sub[sub.d>0] if rule=="best" else sub[sub.d<0]
        sub.insert(0,"rank",np.arange(1,len(sub)+1));save(sub,"top_"+f+"_days")
    results=[];annual=[];paths=[]
    cases=[("original",0)]+[(rule,k) for k in counts() for rule in ORDER]+[("symmetric",2*count(p)) for p in CFG["tail_each"]]
    for rule,k in cases:
        keep=mask(rule,k) if k else np.ones(N,bool)
        results.append(dict(rule=rule,k=k,**describe(keep)))
        p=path(keep);p["rule"]=rule;p["k"]=k;paths.append(p)
        for y in np.unique(T.year):
            sel=T.year==y;d=D[sel&keep]
            annual.append(dict(rule=rule,k=k,year=y,removed=int((sel&~keep).sum()),n=len(d),pos=d[d>0].sum(),neg=d[d<0].sum(),D=d.sum(),Q=np.expm1(d.sum())))
    save(results,"deterministic_deletion");save(annual,"annual_contributions")
    save(pd.DataFrame(results)[["rule","k","n","corr","r2","alpha","beta","residual_sd","rms","max_dev","Q","same_sign"]],"fit_metrics")
    pd.concat(paths).to_csv(OUT/"synthetic_paths.csv.gz",index=False,float_format="%.12g",compression={"method":"gzip","mtime":0})
    state=[]
    top=set(ORDER["best"][:20])
    for s in daily.state.unique():
        ix=np.flatnonzero(daily.state==s);state.append(dict(state=s,n=len(ix),mean=D[ix].mean(),sd=D[ix].std(ddof=1),Q=np.expm1(D[ix].sum()),top20=len(top.intersection(ix))))
    save(state,"market_states")
