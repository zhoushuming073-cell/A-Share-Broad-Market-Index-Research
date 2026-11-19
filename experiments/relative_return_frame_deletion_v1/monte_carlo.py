import os
from numba import njit
from scipy.stats import norm
from data import *
from deletion import counts
from block_sampling import fill

FIELDS="D Q wealth_a wealth_b corr r2 alpha beta residual_sd daily_mean daily_sd rms max_dev same_sign".split()
CACHE=ROOT/".cache";CACHE.mkdir(exist_ok=True)
@njit(cache=True)
def batch(a,b,k,mode,bounds,length,seed,reps):
    np.random.seed(seed)
    n=len(a);m=n-k;out=np.empty((reps,14));first=np.zeros(n,np.bool_)
    for s in range(reps):
        mask=np.zeros(n,np.bool_)
        fill(mask,k,mode,bounds,length)
        if s==0:first=mask.copy()
        la=lb=aa=ab=bb=rms=mx=removed=0.;sign=hits=0
        for j in range(n):
            x,y=a[j],b[j]
            if mask[j]:
                removed+=y-x;hits+=1;continue
            la+=x;lb+=y;aa+=x*x;ab+=x*y;bb+=y*y
            diff=lb-la;rms+=diff*diff;mx=max(mx,abs(diff));sign+=(x*y>0 or (x==0 and y==0))
        assert hits==k
        diff=lb-la;xx=aa-la*la/m;yy=bb-lb*lb/m;xy=ab-la*lb/m
        rho=xy/np.sqrt(xx*yy);beta=xy/xx
        out[s]=np.array([diff,np.expm1(diff),np.exp(la),np.exp(lb),rho,rho*rho,(lb-beta*la)/m,beta,
            np.sqrt(max(0.,yy-beta*xy)/(m-2)),diff/m,np.sqrt((xx+yy-2*xy)/(m-1)),np.sqrt(rms/m),mx,sign/m])
        if s==0:assert abs(diff+removed-(b-a).sum())<1e-10
    return out,first

def bounds(rule):
    group=T.year if rule=="year" else T.to_period("M").asi8
    return np.r_[0,np.flatnonzero(group[1:]!=group[:-1])+1,N].astype("int64")
def wilson(h,n):
    z=norm.ppf(.975);p=h/n
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return mid-half,mid+half
def run():
    fingerprint=hashlib.sha256(FILE.read_bytes()+b"".join((ROOT/name).read_bytes() for name in ["config.yaml","monte_carlo.py","block_sampling.py","data.py"])).hexdigest()[:12]
    core={count(p) for p in CFG["proportions"]}
    methods=["uniform","year","month"]+[f"block{b}" for b in CFG["blocks"]]
    summaries=[];convergence=[];audit=[];quantiles=[];replay=[];draws={}
    for mi,rule in enumerate(methods+["replica"]):
        for k in (sorted(core) if rule=="replica" else counts()):
            reps=CFG["replica_trials"] if rule=="replica" else CFG["uniform_trials"] if rule=="uniform" and k in core else CFG["secondary_trials"]
            mode=0 if rule in ("replica","uniform") else 2 if rule.startswith("block") else 1
            length=int(rule[5:]) if mode==2 else 1
            blocks=[];first=None
            for start in range(0,reps,CFG["batch"]):
                seed=int(np.random.SeedSequence([CFG["replica_seed"] if rule=="replica" else CFG["seed"],mi,k,start]).generate_state(1)[0])
                file=CACHE/f"{fingerprint}_{rule}_{k}_{start}.npz"
                if not file.exists():
                    vals,mask=batch(A,B,k,mode,bounds(rule),length,seed,min(CFG["batch"],reps-start))
                    temp=file.with_suffix(".tmp.npz");np.savez_compressed(temp,values=vals,deleted=np.flatnonzero(mask));os.replace(temp,file)
                with np.load(file) as z:
                    blocks.append(z["values"])
                    if start==0:first=z["deleted"]
            data=np.vstack(blocks);assert len(data)==reps and np.isfinite(data).all()
            q=data[:,1];Dsim=data[:,0];flipped=(Dsim<=0) if D.sum()>0 else (Dsim>=0)
            row=dict(rule=rule,k=k,reps=reps,fraction=k/N,p_positive=(q>0).mean(),p_flip=flipped.mean(),
                mcse_flip=flipped.std(ddof=1)/np.sqrt(reps),p_flip_lo=wilson(flipped.sum(),reps)[0],p_flip_hi=wilson(flipped.sum(),reps)[1])
            values=np.quantile(data,[.01,.05,.25,.5,.75,.95,.99],axis=0)
            for j,col in enumerate(FIELDS):
                row[col+"_mean"]=data[:,j].mean();row[col+"_sd"]=data[:,j].std(ddof=1);row[col+"_mcse"]=row[col+"_sd"]/np.sqrt(reps)
                for l,tag in enumerate(["01","05","25","50","75","95","99"]):row[col+"_p"+tag]=values[l,j]
            summaries.append(row)
            kept=np.ones(N,bool);kept[first]=False
            from metrics import describe
            ref=describe(kept)
            np.testing.assert_allclose(data[0],[ref[c] for c in FIELDS],atol=1e-11)
            replay.extend(dict(rule=rule,k=k,date=str(T[j].date())) for j in first)
            if rule in ("uniform","replica"):
                mu=(1-k/N)*D.sum();sd=np.sqrt(k*(1-k/N)*D.var(ddof=1))
                audit.append(dict(rule=rule,k=k,reps=reps,theory_mean=mu,observed_mean=Dsim.mean(),theory_sd=sd,observed_sd=Dsim.std(ddof=1),
                    z_mean=(Dsim.mean()-mu)/(sd/np.sqrt(reps))))
            if rule=="uniform":
                for n in (100,1000,3000,10000,30000,100000):
                    if n<=reps:convergence.append(dict(k=k,n=n,mean_D=Dsim[:n].mean(),p_positive=(q[:n]>0).mean(),mean_Q=q[:n].mean()))
            if rule!="replica":draws[rule+"_"+str(k)]=q
        print(rule,"complete",flush=True)
    for filename,rules in [("random_deletion_summary",["uniform"]),("stratified_deletion_summary",["year","month"]),
                           ("block_deletion_summary",methods[3:]),("replica_summary",["replica"])]:
        save([s for s in summaries if s["rule"] in rules],filename)
    save(audit,"analytic_checks");save(convergence,"convergence");save(replay,"first_draw_dates")
    np.savez_compressed(OUT/"terminal_Q.npz",**draws)
if __name__=="__main__":run()
