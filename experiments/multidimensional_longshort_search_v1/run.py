from walk_forward import *
from attribution import ledger
from backtest import COLS
def main():
    bank,paths,bounds=run_folds()
    rows=[];rob=[];a=bounds[0];e=len(R);base=measure(BASE,0,e)
    for i,p in bank.items():
        m=measure(p,0,e);forward=measure(p,a,e);early=measure(p,0,a)
        neighbors=[]
        for f in (.9,1.1):
            q=perturb(p,f);n=measure(q,0,e)["cagr"];neighbors.append(n)
            rob.append(dict(id=i,scenario="near",factor=f,cagr=n,params=json.dumps(q,sort_keys=True)))
        stability=max(0.,1-max(abs(n-m["cagr"]) for n in neighbors)/.08)
        ideal,overall,qualified,alt=score(m,len(p),stability,forward["cagr"])
        rows.append(dict(id=i,family=p["family"],params=json.dumps(p,sort_keys=True),**m,ideal=ideal,overall=overall,
            qualified=qualified,alternate=alt,stability=stability,complexity=len(p),forward_cagr=forward["cagr"],
            early_cagr=early["cagr"],main_cagr=measure(p,a,bounds[14])["cagr"],append_cagr=measure(p,bounds[14],e)["cagr"],
            excess_cagr=m["cagr"]-base["cagr"]))
    df=pd.DataFrame(rows);save(df,"candidate_metrics")
    trials=pd.read_csv(OUT/"all_trials.csv")
    validation_fronts=[]
    for k,g in trials.drop_duplicates(["fold","id"]).groupby("fold"):
        values=g[["valid_cagr","valid_dd","valid_sharpe"]].values.copy();values[:,1]*=-1
        validation_fronts.append(g.iloc[front(values)])
    save(pd.concat(validation_fronts),"inner_fronts")
    vec=df[["cagr","mdd","sharpe"]].values.copy();vec[:,1]*=-1
    save(df.iloc[front(vec)],"pareto_front")
    active=df[df.active>=.15];rankings={}
    save(active.nlargest(10,"ideal"),"ideal_nearest")
    for lab,sub,col in [("return",active,"cagr"),("ideal_curve",df[df.qualified],"ideal"),("overall",active,"overall")]:
        ranked=sub.sort_values([col,"id"],ascending=[False,True]).copy();ranked.insert(0,"rank",range(1,len(ranked)+1))
        save(ranked,"top_"+lab);rankings[lab]=(ranked if len(ranked) else active.sort_values(col,ascending=False)).iloc[0]
    static=df[df.family=="fixed"].nlargest(1,"cagr").iloc[0]
    blocks=[]
    for name,i in [("base",key(BASE)),("static_best",static.id)]+[(k,v.id) for k,v in rankings.items()]:
        p=bank[i];acct=simulate(p,0,e);d=ledger(name,acct,DATES);d["id"]=i;blocks.append(d)
        for fee,bc,fc in [(0,0,0),(.001,.04,.06),(.0005,.02,.06)]:
            z=measure(p,0,e,fee=fee,borrow=bc,finance=fc)
            rob.append(dict(id=i,scenario=f"fee={fee},borrow={bc},finance={fc}",cagr=z["cagr"]))
        for f in (.8,.9,1,1.1,1.2):
            for h in (.8,.9,1,1.1,1.2):
                q=dict(p);q["gross"]=np.clip(q["gross"]*f,.5,2.5)
                knob=next((k for k in ["ratio","threshold","window","frequency"] if k in q),"gross")
                q[knob]=q[knob]*h
                if knob in ("window","frequency"):q[knob]=max(1,round(q[knob]))
                rob.append(dict(id=i,scenario="grid",gross_factor=f,other_factor=h,knob=knob,cagr=measure(q,0,e)["cagr"]))
    for name,parts in paths.items():
        scale=1.;joined=[]
        for part in parts:
            part=part.copy();part[:,1:11]*=scale;scale=part[-1,1];joined.append(part)
        blocks.append(ledger("wf_"+name,np.vstack(joined),DATES[a:]))
    daily=pd.concat(blocks,ignore_index=True)
    save(daily,"daily_positions");save(daily.pivot(index="date",columns="strategy",values="nav").reset_index(),"daily_nav")
    attr=daily.groupby("strategy")[["constant_direction","dynamic_direction","common","relative","fee","borrow","finance"]].sum()
    attr["driver"]=np.where(abs(attr.relative)>abs(attr.common),"相对收益主导",np.where(abs(attr.dynamic_direction)>abs(attr.constant_direction),"动态方向主导","固定方向主导"))
    save(attr.reset_index(),"attribution");save(rob,"robustness")
    from verify import verify
    verify()
    from report import build
    build()
if __name__=="__main__":main()
