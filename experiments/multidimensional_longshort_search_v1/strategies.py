import hashlib,json
from functools import lru_cache
from data import np
from features import features

FAMILIES=["fixed","momentum","reversion","trend","voltarget","hedge","regime","fusion"]
def spec(f):
    fields={"gross":(.5,2.5),"ratio":(.35,2.2),"frequency":[1,5,20]}
    if f!="fixed": fields["window"]=[10,20,40,60,120]
    if f in ("momentum","reversion","trend","regime","fusion"):
        fields.update(threshold=(.2,1.8),reverse=[False,True],confirm=[1,3],hold=[5,20,60])
    if f in ("voltarget","fusion"):fields["risk_target"]=(.02,.10)
    if f=="hedge":
        fields.pop("ratio");fields["model"]=["static","rolling","ewma","ratio","minimum","beta_match","equal"]
    if f in ("fusion","regime"):fields["shock_limit"]=(1.,2.)
    if f=="fusion":fields["dd_limit"]=(.04,.15)
    return fields
def canonical(p):
    p=dict(p)
    if p.get("model") in ("equal","static"):p.pop("window",None)
    return {k:round(float(v),4) if isinstance(v,(float,np.floating)) else v for k,v in p.items()}
def key(p):return hashlib.sha256(json.dumps(canonical(p),sort_keys=True).encode()).hexdigest()[:16]
def params(f,u):
    p={"family":f}
    for t,(name,domain) in enumerate(spec(f).items()):
        p[name]=domain[min(int(u[t]*len(domain)),len(domain)-1)] if isinstance(domain,list) else domain[0]+u[t]*(domain[1]-domain[0])
    return canonical(p)
def targets(p):
    f=p["family"];d=features(p.get("window",120))
    h=p.get("ratio",1.)
    if f=="hedge" and p["model"]!="equal":h=d[p["model"]].fillna(1.).clip(.15,5.).to_numpy()
    signal=np.ones(len(d))
    m=d.momentum.to_numpy();t=d.trend.to_numpy()
    if f in ("momentum","reversion","trend","regime","fusion"):
        score={"momentum":m,"reversion":-m,"trend":t,"regime":np.where(abs(t)>1,m,-m),"fusion":.65*m+.35*t}[f]
        signal=np.where(score>p["threshold"],1.,np.where((score < -p["threshold"]) & p["reverse"],-1.,0.))
        if p["confirm"]>1:
            ok=np.r_[False,False,(signal[2:]==signal[1:-1])&(signal[2:]==signal[:-2])]
            signal=np.where(ok,signal,0.)
        if "shock_limit" in p:signal*=d.shock.le(p["shock_limit"]).values
    exposure=p["gross"]*signal
    if "risk_target" in p:exposure*=np.clip(p["risk_target"]/d.vol.fillna(np.inf).clip(lower=.001).values,0,1)
    return np.ascontiguousarray(exposure/(1+h)),np.ascontiguousarray(-exposure*h/(1+h))
