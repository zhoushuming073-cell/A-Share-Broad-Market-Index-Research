import json,time,warnings
import optuna
from scipy.stats import qmc
from data import *
from strategies import FAMILIES,params,key,spec,canonical
from backtest import simulate
from scoring import metrics,score
from multiobjective import front,volume
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore",category=optuna.exceptions.ExperimentalWarning)
BASE={"family":"fixed","ratio":1.,"gross":1.,"frequency":1}
def measure(p,l,r,**kw):
    a=simulate(p,l,r,**kw);return metrics(a,MONTH[l:r],YEAR[l:r],Y[l:r])
def suggest(tr):
    f=tr.suggest_categorical("family",FAMILIES);p={"family":f}
    for name,space in spec(f).items():
        k=f+"_"+name
        p[name]=tr.suggest_categorical(k,space) if isinstance(space,list) else tr.suggest_float(k,*space)
    return canonical(p)
def optimize(l,mid,r,fold,bank,initial):
    rows=[];cache={};diag=[]
    started=time.perf_counter()
    def evalp(p,alg,seed):
        begin=time.perf_counter();i=key(p)
        if i not in cache:
            train=measure(p,l,mid);val=measure(p,mid,r)
            proxy=.35*np.clip(train["sharpe"],-4,4)+.65*np.clip(val["sharpe"],-4,4)+2*val["cagr"]-val["mdd"]
            cache[i]=(train,val,proxy)
        tr,v,z=cache[i]; banklocal[i]=p
        rows.append(dict(fold=fold,algo=alg,seed=seed,id=i,params=json.dumps(p,sort_keys=True),family=p["family"],
            train_cagr=tr["cagr"],valid_cagr=v["cagr"],valid_dd=v["mdd"],valid_sharpe=v["sharpe"],
            active=v["active"],proxy=z,seconds=time.perf_counter()-begin,elapsed=time.perf_counter()-started,status="complete"))
        return z,(v["cagr"],-v["mdd"],v["sharpe"])
    banklocal={}
    for p in bank.values():evalp(p,"frozen_bank",0)
    for offset,seed in enumerate([CFG["seed"],CFG["second_seed"]] if initial else [CFG["seed"]+fold]):
        if initial:
            for algo,n in [("random",CFG["random_per_seed"]),("sobol",CFG["sobol_per_seed"])]:
                for j,f in enumerate(FAMILIES):
                    dim=len(spec(f));size=n//8
                    draws=(np.random.default_rng(seed+j).random((size,dim)) if algo=="random" else
                           qmc.Sobol(dim,scramble=True,seed=seed+j).random_base2(int(np.log2(size))))
                    for u in draws:evalp(params(f,u),algo,seed)
        for algo in ["tpe","nsga"]:
            sampler=(optuna.samplers.TPESampler(seed=seed,multivariate=True,group=True) if algo=="tpe" else
                     optuna.samplers.NSGAIISampler(seed=seed,population_size=CFG["population"]))
            study=optuna.create_study(sampler=sampler,directions=["maximize"]*(1 if algo=="tpe" else 3))
            # Seed the population with high-quality members of the preceding stage.
            elites=sorted(banklocal,key=lambda i:cache[i][2],reverse=True)[:8]
            for i in elites:
                p=banklocal[i];f=p["family"]
                study.enqueue_trial({"family":f,**{f+"_"+k:v for k,v in p.items() if k!="family"}})
            n=CFG[algo+"_per_seed"] if initial else CFG["fold_"+algo]
            def objective(tr):
                z,vec=evalp(suggest(tr),algo,seed)
                return z if algo=="tpe" else vec
            study.optimize(objective,n_trials=n)
    df=pd.DataFrame(rows)
    for (alg,seed),g in df.groupby(["algo","seed"]):
        if alg=="frozen_bank":continue
        for n in sorted(set([min(i,len(g)) for i in [16,32,64,128,256,len(g)]])):
            h=g.iloc[:n];v=h[["valid_cagr","valid_dd","valid_sharpe"]].values.copy();v[:,1]*=-1
            diag.append(dict(fold=fold,algo=alg,seed=seed,n=n,best=h.valid_cagr.max(),best_proxy=h.proxy.max(),hv=volume(v),seconds=h.seconds.sum(),wall_seconds=h.elapsed.iloc[-1]-h.elapsed.iloc[0]))
    return banklocal,cache,rows,diag
