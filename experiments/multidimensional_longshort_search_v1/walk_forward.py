import pickle,sqlite3,hashlib
from search import *
from multiobjective import front
from backtest import COLS
def perturb(p,factor):
    q=dict(p);q["gross"]=min(2.5,max(.5,q["gross"]*factor))
    if "threshold" in q:q["threshold"]*=factor
    elif "ratio" in q:q["ratio"]*=factor
    return canonical(q)
def run_folds():
    db=sqlite3.connect(OUT/"checkpoint.sqlite")
    db.execute("create table if not exists folds(k integer primary key, payload blob)")
    fingerprint=hashlib.sha256(b"".join((ROOT/f).read_bytes() for f in
        ("walk_forward.py","backtest.py","features.py","strategies.py","scoring.py","search.py","config.yaml","data.py","multiobjective.py"))+FILE.read_bytes()).hexdigest()
    db.execute("create table if not exists manifest(version text)")
    if db.execute("select version from manifest").fetchone()!=(fingerprint,):
        db.execute("delete from folds");db.execute("delete from manifest")
        db.execute("insert into manifest values(?)",(fingerprint,));db.commit()
    # Calendar-aligned half years; the last period ends at the final stored close.
    bounds=[DATES.searchsorted(pd.Timestamp(y,m,1)) for y in range(2018,2027) for m in (1,7)]+[len(R)]
    records=[];convergence=[];walk=[];paths={};bank={key(BASE):BASE}
    for k,(a,b) in enumerate(zip(bounds,bounds[1:])):
        cutoff=a-1;mid=DATES.searchsorted(DATES[a]-pd.DateOffset(years=1))
        l=max(0,DATES.searchsorted(DATES[a]-pd.DateOffset(years=3)))
        stored=db.execute("select payload from folds where k=?",(k,)).fetchone()
        if stored:local,cache,rows,diag=pickle.loads(stored[0])
        else:
            local,cache,rows,diag=optimize(l,mid,cutoff,k,bank,k==0)
            db.execute("insert into folds values(?,?)",(k,pickle.dumps((local,cache,rows,diag))));db.commit()
        if k==0:bank=local.copy()
        records.extend(rows);convergence.extend(diag)
        ids=list(local);vec=[[cache[i][1]["cagr"],-cache[i][1]["mdd"],cache[i][1]["sharpe"]] for i in ids]
        shortlist=[ids[j] for j in front(vec) if cache[ids[j]][1]["active"]>=.15] or [key(BASE)]
        ratings={}
        for i in shortlist:
            train,v,_=cache[i];p=local[i]
            ns=[measure(perturb(p,f),mid,cutoff)["cagr"] for f in (.9,1.1)]
            stability=max(0.,1-max(abs(t-v["cagr"]) for t in ns)/.08)
            ratings[i]=score(v,len(p),stability,train["cagr"])
        best=max(shortlist,key=lambda i:cache[i][1]["cagr"])
        qualified=[i for i in shortlist if ratings[i][2]]
        ideal=max(qualified or shortlist,key=lambda i:ratings[i][0])
        overall=max(shortlist,key=lambda i:ratings[i][1])
        early=max([t for t in rows if t["algo"]=="tpe"][:16],key=lambda t:t["proxy"])["id"]
        late=max([t for t in rows if t["algo"]=="tpe"],key=lambda t:t["proxy"])["id"]
        for name,i in [("return",best),("ideal",ideal),("overall",overall),("budget16",early),("budget_full",late)]:
            p=local[i];acct=simulate(p,a,b);outer=metrics(acct,MONTH[a:b],YEAR[a:b],Y[a:b])
            walk.append(dict(fold=k,policy=name,id=i,params=json.dumps(p,sort_keys=True),train_start=DATES[l],
                validation_start=DATES[mid],info_end=DATES[cutoff-1],execution=P.index[a],first=DATES[a],last=DATES[b-1],
                valid_cagr=cache[i][1]["cagr"],outer_cagr=outer["cagr"],outer_net=outer["net"],
                ideal_gate=ratings.get(i,(0,0,False))[2],complexity=len(p)))
            paths.setdefault(name,[]).append(acct)
        print("fold",k+1,"of",len(bounds)-1,"candidates",len(local),flush=True)
    db.close();save(records,"all_trials");save(convergence,"convergence");save(walk,"walk_forward")
    return bank,paths,bounds
