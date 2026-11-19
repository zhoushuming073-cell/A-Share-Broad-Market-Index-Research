"""Independent audit of the persisted daily ledgers, as-of feature calculations and rankings."""
import json
from scipy.stats import linregress
from data import *
from features import features
from scoring import score

def verify():
    d=pd.read_csv(OUT/"daily_positions.csv",parse_dates=["date"])
    nav=pd.read_csv(OUT/"daily_nav.csv",index_col=0,parse_dates=True)
    wf=pd.read_csv(OUT/"walk_forward.csv",parse_dates=["info_end","execution","first","last"])
    assert ((wf.info_end<wf.execution)&(wf.execution<wf["first"])).all()
    def ck(a,b):np.testing.assert_allclose(a,b,atol=3e-10,rtol=2e-9)
    checks=0
    for name,g in d.groupby("strategy"):
        g=g.set_index("date");idx=g.index
        x,y=R.loc[idx].values.T
        # Independently reconstruct trades as differences versus prior marked holdings.
        old_a=np.r_[0.,(g.a*(1+x)).values[:-1]];old_b=np.r_[0.,(g.b*(1+y)).values[:-1]]
        ends=np.zeros(len(g),bool);ends[-1]=True
        if name.startswith("wf_"):
            ends=idx.isin(wf[wf.policy.eq(name[3:])]["last"])
            starts=np.r_[True,ends[:-1]]
            old_a[starts]=0.;old_b[starts]=0.
        turnover=abs(g.a-old_a)+abs(g.b-old_b)+ends*(abs(g.a*(1+x))+abs(g.b*(1+y)))
        ck(turnover,g.turnover);ck(turnover*CFG["fee"],g.fee)
        equity=g.start-(turnover-ends*(abs(g.a*(1+x))+abs(g.b*(1+y))))*CFG["fee"]
        gap=GAPS[DATES.get_indexer(idx)]/365
        ck((g.a.clip(upper=0)+g.b.clip(upper=0))*(-CFG["borrow"])*gap,g.borrow)
        ck((g.a.clip(lower=0)+g.b.clip(lower=0)-equity).clip(lower=0)*CFG["finance"]*gap,g.finance)
        ck(g.start+g.a*x+g.b*y-g.fee-g.borrow-g.finance,g.nav)
        ck((1+g.net).cumprod(),g.nav);ck(g.nav,nav.loc[idx,name])
        ck(g.common+g.relative,g.a*x+g.b*y);ck(g.common,g.dynamic_direction+g.constant_direction)
        assert g.gross_weight.max()<=2.5+1e-9
        checks+=len(g)
    candidates=pd.read_csv(OUT/"candidate_metrics.csv")
    for label,mask,col in [("return",candidates.active>=.15,"cagr"),("ideal_curve",candidates.qualified,"ideal"),("overall",candidates.active>=.15,"overall")]:
        expected=candidates[mask].sort_values([col,"id"],ascending=[False,True])
        actual=pd.read_csv(OUT/("top_"+label+".csv"))
        assert list(actual.id)==list(expected.id)
    for row in candidates.to_dict("records"):
        i,o,q,alt=score(row,row["complexity"],row["stability"],row["forward_cagr"])
        ck([i,o,alt],[row["ideal"],row["overall"],row["alternate"]]);assert q==row["qualified"]
    base=d[d.strategy.eq("base")]
    ck(base.a,-base.b)
    for k in (10,20,40,60,120):
        f=features(k)
        for day in [250,738,1500,2800]:
            hist=R.iloc[day-k-1:day-1];x,y=hist.values.T
            ck(f["rolling"].iloc[day],linregress(y,x).slope)
            ck(f.momentum.iloc[day],(x-y).mean()*np.sqrt(k)/np.std(x-y,ddof=1))
    # Independent implementation for the published momentum winner, including confirmation,
    # age-based exits, unscheduled margin calls and next-close execution.
    winner=pd.read_csv(OUT/"top_return.csv").iloc[0]
    p=json.loads(winner.params)
    if p["family"]=="momentum":
        g=d[d.strategy.eq("return")];signals=[];age=0;pa=pb=0.;v=1.
        spread=X-Y;k=p["window"]
        for i,row in enumerate(g.itertuples()):
            hist=spread[i-k-1:i-1] if i>=k+1 else np.array([])
            z=hist.mean()*np.sqrt(k)/hist.std(ddof=1) if len(hist) else 0
            signals.append(1 if z>p["threshold"] else -1 if p["reverse"] and z < -p["threshold"] else 0)
            sign=signals[-1] if len(set(signals[-p["confirm"]:]))==1 and len(signals)>=p["confirm"] else 0
            expired=age>=p["hold"]
            if i%p["frequency"]==0 or expired or abs(pa)+abs(pb)>2.5*v+1e-12:
                w=(0 if expired else sign)*p["gross"]/(1+p["ratio"]);b=-p["ratio"]*w
                post=v
                for _ in range(8):post=v-CFG["fee"]*(abs(w*post-pa)+abs(b*post-pb))
                pa,pb=w*post,b*post
            ck([pa,pb],[row.a,row.b])
            age=age+1 if abs(pa)+abs(pb)>1e-12 else 0
            pa*=1+X[i];pb*=1+Y[i];v=row.nav
    out=dict(status="passed",daily_ledger_rows=checks,ranked_candidates=len(candidates),calendar_sessions=len(P),
        winner_decisions_rebuilt=p["family"]=="momentum",
        input_sha256=hashlib.sha256(FILE.read_bytes()).hexdigest(),outer_windows=wf.fold.nunique(),
        controls=["as-of dates","historical feature samples","unit NAV compounding","marked trading volume including fold exits",
        "short collateral excluded from funding","margin ceiling","daily algebraic attribution","all leaderboard memberships and orders"])
    (OUT/"verification.json").write_text(json.dumps(out,indent=2))
if __name__=="__main__":verify()
