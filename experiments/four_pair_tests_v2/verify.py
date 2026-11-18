"""Independent checks on exported observations, not calls to the simulation functions."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
OLD=ROOT.parent/"four_pair_tests"/"results"
P=pd.read_csv(ROOT/"data/input_closes.csv",index_col=0,parse_dates=True)
R=P.pct_change(fill_method=None).iloc[1:]
F=.0005
B=.02


def read(name,root=OUT):
    return pd.read_csv(root/name,dtype={"a":str,"b":str})


def same(x,y,tol=4e-10):
    np.testing.assert_allclose(x,y,rtol=1e-10,atol=tol,equal_nan=True)


def check_windows():
    z=np.load(OUT/"e1_paths.npz")
    table=read("e1_windows.csv")
    n=len(P)-252
    x,y=(P[k].values for k in ("000001","000300"))
    gaps=np.diff(P.index.values).astype("timedelta64[D]").astype(int)
    fixed_borrow=np.zeros(n)
    capital=np.ones(n)
    prev_a=np.zeros(n);prev_b=np.zeros(n);fees=np.zeros(n);borrow=np.zeros(n)
    gn=np.ones(n)
    for i in range(252):
        lx=x[i+1:i+n+1]/x[i:i+n]-1
        sy=y[i+1:i+n+1]/y[i:i+n]-1
        expiry=F*(x[i+1:i+1+n]/x[:n]+y[i+1:i+1+n]/y[:n])/2 if i==251 else 0
        fixed_borrow+=.5*y[i:i+n]/y[:n]*B*gaps[i:i+n]/365
        fg=1+.5*(x[i+1:i+1+n]/x[:n]-y[i+1:i+1+n]/y[:n])
        same(fg,z["units_gross"][:,i+1])
        same(fg-F-expiry-fixed_borrow,z["units_net"][:,i+1])
        if i==0:
            sized=np.full(n,.5)
        else:
            # Solve the piecewise-linear fee equation by choosing the valid trade signs.
            sized=np.full(n,np.nan)
            for sa,sb in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                candidate=(capital+F*(sa*prev_a+sb*prev_b))/(2+F*(sa-sb))
                ok=(sa*(candidate-prev_a)>=-1e-14)&(sb*(-candidate-prev_b)>=-1e-14)
                sized[ok]=candidate[ok]
            assert np.isfinite(sized).all()
        fee=F*(abs(sized-prev_a)+abs(-sized-prev_b))
        carry=B*sized*gaps[i:i+n]/365
        prev_a=sized*(1+lx);prev_b=-sized*(1+sy)
        if i==251: fee+=F*(abs(prev_a)+abs(prev_b))
        fees+=fee;borrow+=carry;capital+=sized*(lx-sy)-fee-carry
        gn*=1+(lx-sy)/2
        same(capital,z["daily_net"][:,i+1]);same(gn,z["daily_gross"][:,i+1])
    for rule,trade,cost in [("units",F+expiry,fixed_borrow),("daily",fees,borrow)]:
        rows=table[table.rule==rule]
        np.testing.assert_array_equal(pd.to_datetime(rows.entry),P.index[:n])
        np.testing.assert_array_equal(pd.to_datetime(rows.exit),P.index[252:])
        assert (rows.holding==252).all()
        same(rows.fee,trade);same(rows.borrow,cost)
        same(rows.net,z[rule+"_net"][:,-1]-1);same(rows.gross,z[rule+"_gross"][:,-1]-1)
    return n


def check_signals(d):
    # Pool repeated schedule dates; reconstruct slopes directly from price returns at their recorded cutoff.
    signal=d[d.active & d.method.isin(["static","rolling","ewma","vol"])].copy()
    assert (signal.signal_cutoff<signal.execution).all()
    unique=signal.drop_duplicates(["a","b","method","signal_cutoff"])
    for (a,b,method),g in unique.groupby(["a","b","method"]):
        for row in g.itertuples():
            pos=R.index.get_loc(row.signal_cutoff)+1
            window=120 if method in ("rolling","vol") else pos
            x=R[a].values[pos-window:pos];y=R[b].values[pos-window:pos]
            if method=="vol": estimate=x.std()/y.std()
            elif method=="ewma":
                power=.97**np.arange(pos-1,-1,-1)
                weight=.03*power;weight[0]=power[0]
                mx=weight@x;my=weight@y
                estimate=(weight*(x-mx))@(y-my)/((weight*(y-my))@(y-my))
            else:
                estimate=(x-x.mean())@(y-y.mean())/((y-y.mean())@(y-y.mean()))
            same(row.h,estimate,2e-9)
    # Check update schedule against the exchange calendar rather than against the stored ratio.
    for (_,period,a,b,m,f),g in signal.groupby(["experiment","period","a","b","method","frequency"]):
        if m=="static":
            cutoff=R.index[119] if period=="train" else R.loc[:"2017"].index[-2]
            assert (g.signal_cutoff==cutoff).all()
            continue
        ix=R.index.get_indexer(g.date)
        if f=="D": same_cut=R.index[ix-2]
        else:
            periods=R.index.to_period({"W":"W-FRI","M":"M","Q":"Q"}[f])
            init=np.r_[True,periods[1:]!=periods[:-1]]
            init[121]=True
            positions=np.maximum.accumulate(np.where(init,np.arange(len(R)),0))
            same_cut=R.index[positions[ix]-2]
        np.testing.assert_array_equal(same_cut,g.signal_cutoff)
    return len(unique)


def accounts():
    buckets=[];summary=read("phase_statistics.csv")
    keys=["experiment","period","a","b","method","frequency"]
    for exp in (2,3,4):
        d=read(f"e{exp}_daily.csv.gz")
        for c in ("date","execution","signal_cutoff"):d[c]=pd.to_datetime(d[c])
        for key,g in d.groupby(keys):
            s=summary.set_index(keys).loc[key]
            pos=P.index.get_indexer(g.date)
            np.testing.assert_array_equal(P.index[pos-1],g.execution)
            x,y=(R.loc[g.date,c].values for c in (key[2],key[3]))
            w,v=g.weight_a.values,g.weight_b.values
            net=g.net_nav.to_numpy()
            start=np.r_[1.,net[:-1]]
            # Start with equity, solve the actual trade equation, then use marked preceding holdings.
            pa=np.zeros(len(g));pb=pa.copy();volume=pa.copy();short=pa.copy()
            old_a=old_b=0.
            for i in range(len(g)):
                post=start[i]
                for _ in range(5):
                    post=start[i]-F*(abs(post*w[i]-old_a)+abs(post*v[i]-old_b))
                pa[i]=post*w[i];pb[i]=post*v[i]
                volume[i]=abs(pa[i]-old_a)+abs(pb[i]-old_b)
                old_a=pa[i]*(1+x[i]);old_b=pb[i]*(1+y[i])
            volume[-1]+=abs(old_a)+abs(old_b)
            days=(g.date-g.execution).dt.days
            carry=(np.maximum(-pa,0)+np.maximum(-pb,0))*B*days/365
            same(volume,g.turnover);same(F*volume,g.fee);same(carry,g.borrow)
            same(start+pa*x+pb*y-volume*F-carry,net)
            same(net/start-1,g.net);same(w*x+v*y,g.gross)
            same(np.cumprod(1+w*x+v*y),g.gross_nav)
            same((pa+pb)*(x+y)/2,g.common_money)
            same((pa-pb)*(x-y)/2,g.relative_money)
            same(s.net,s.net_common+s.net_relative-s.fee-s.borrow)
            same(s.gross,s.gross_common+s.gross_relative)
            same(s.net,net[-1]-1)
            same(s.vol,np.std(net/start-1,ddof=1)*np.sqrt(252))
            if exp==4 and key[2]==key[3] and key[4] not in ("half","double"):
                same(net,1,0);same(g.turnover,0,0);same(g.borrow,0,0)
        buckets.append(d)
    all_days=pd.concat(buckets,ignore_index=True)
    checks=check_signals(all_days)
    return all_days,checks


def hac(d):
    active=d[(d.experiment==2)&(d.period=="test")].set_index(["a","b","date"]).sort_index()
    diff=read("e2_paired_daily.csv");h=read("e2_hac.csv")
    selections=read("e2_selection.csv")
    for row in selections.itertuples():
        assert row.validation>=row.validation_reverse
        for a,b,score in [(row.a,row.b,row.validation),(row.b,row.a,row.validation_reverse)]:
            g=d[(d.experiment==2)&(d.period=="validation")&(d.a==a)&(d.b==b)].iloc[:-1]
            last=g.iloc[-1]
            assert str(last.date.date())==row.selection_end
            # Mark both legs at the final ranking close and pay the explicit exit fee.
            scale=(g.iloc[-2].net_nav-last.fee)/2
            marked=scale*(2+R.loc[last.date,a]+R.loc[last.date,b])
            same(last.net_nav-1-F*marked,score)
    for (a,b),g in h.groupby(["a","b"]):
        s=active.loc[a,b];rev=active.loc[b,a]
        z=(s.net-rev.net).to_numpy()
        ds=diff[(diff.a==a)&(diff.b==b)]
        np.testing.assert_array_equal(s.index,pd.to_datetime(ds.date))
        same(ds.selected,s.net);same(ds.reverse,rev.net);same(ds.difference,z)
        n=len(z);e=z-z.mean()
        for row in g.itertuples():
            meat=e@e+2*sum((1-k/(row.lag+1))*(e[k:]@e[:-k]) for k in range(1,row.lag+1))
            se=np.sqrt(meat*n/(n-1))/n
            same(row.se,se);same(row.p,2*norm.sf(abs(z.mean())/se));same(row.mean,z.mean())
    for _,g in h.groupby("lag"):
        g=g.sort_values("p")
        same(np.minimum(1,np.maximum.accumulate(g.p*np.arange(len(g),0,-1))),g.holm)


def version_table():
    rows=[]
    def add(exp,item,old,new,reason):
        rows.append(dict(experiment=exp,measure=item,v1=old,v2=new,delta=new-old,reason=reason))
    e=read("e1_windows.csv");old=read("e1_all_entries.csv",OLD)
    for key in ("units","daily"):
        add(1,"mean_net_"+key,old.net.mean(),e.loc[e.rule.eq(key),"net"].mean(),
            "fixed units" if key=="units" else "initial positions exactly 0.5 before fees")
    for row in read("e2_hac.csv").query("lag==5").itertuples():
        prev=read("e2_paired_tests.csv",OLD)
        add(2,"hac5_p_"+row.a+"_"+row.b,prev[(prev.a==row.a)&(prev.b==row.b)].p.iloc[0],row.p,
            "HAC now uses test accounts opened from flat")
    prior=read("e2_e4_statistics.csv",OLD)
    current=read("phase_statistics.csv")
    for row in current[(current.period=="test")&(current.frequency=="D")&(current.experiment>=3)].itertuples():
        q=prior[(prior.experiment==row.experiment)&(prior.a==row.a)&(prior.b==row.b)&
                (prior.method==row.method)&(prior.period=="测试")].iloc[0]
        add(row.experiment,"net_"+row.a+"_"+row.b+"_"+row.method,q.net,row.net,
            "netted same symbol and reset at test start" if row.a==row.b else "reset at test start")
        if row.experiment==4:
            add(4,"gross_common_"+row.a+"_"+row.b+"_"+row.method,q.direction_pnl,row.gross_common,
                "long-leg reference replaced by symmetric common return")
    pd.DataFrame(rows).to_csv(OUT/"version_comparison.csv",index=False,encoding="utf-8-sig")


if __name__=="__main__":
    n=check_windows()
    ledger,sigs=accounts()
    hac(ledger)
    version_table()
    record=dict(status="passed",entry_dates=n,entry_rules=2,holding_sessions=252,
        daily_rows=len(ledger),account_paths=ledger.groupby(["experiment","period","a","b","method","frequency"]).ngroups,
        distinct_signal_estimates_checked=sigs,independent_HAC_checks=9,raw_prices_and_XSHG_calendar="matched")
    (OUT/"verification.json").write_text(json.dumps(record,indent=2))
    print(record)
