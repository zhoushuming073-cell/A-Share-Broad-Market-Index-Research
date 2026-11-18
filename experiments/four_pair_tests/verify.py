"""Check exported paths against price levels, turnover and the funding ledger."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
p=pd.read_csv(ROOT/"data/input_closes.csv",index_col=0,parse_dates=True)
r=p.pct_change().iloc[1:]
daily=pd.read_csv(OUT/"e2_e4_daily.csv.gz",dtype={"a":str,"b":str},parse_dates=["date"])
count=0
for (exp,a,b,method),d in daily.groupby(["experiment","a","b","method"]):
    x,y=(r.loc[d.date,c].values for c in (a,b))
    np.testing.assert_allclose(d.gross,d.w_long*x+d.w_short*y,atol=1e-12)
    np.testing.assert_allclose(d.gross_nav,(1+d.gross).cumprod(),atol=1e-9)
    start=np.r_[1,d.net_nav.values[:-1]]
    # Reconstruct post-trade equity from pre-trade capital. Last day's fee also includes unwind.
    after=(d.net_nav+d.borrow)/(1+d.gross)
    after.iloc[-1]=((d.net_nav.iloc[-1]+d.borrow.iloc[-1])/
        (1+d.gross.iloc[-1]-.0005*(abs(d.w_long.iloc[-1])*(1+x[-1])+abs(d.w_short.iloc[-1])*(1+y[-1]))))
    wa=after*d.w_long; wb=after*d.w_short
    volume=abs(wa-np.r_[0,(wa*(1+x)).values[:-1]])+abs(wb-np.r_[0,(wb*(1+y)).values[:-1]])
    volume.iloc[-1]+=abs(wa.iloc[-1]*(1+x[-1]))+abs(wb.iloc[-1]*(1+y[-1]))
    np.testing.assert_allclose(d.turnover,volume,atol=2e-10)
    np.testing.assert_allclose(d.fee,volume*.0005,atol=1e-12)
    gaps=p.index.to_series().diff().dt.days.loc[d.date].to_numpy()
    np.testing.assert_allclose(d.borrow,-wb*.02*gaps/365,atol=1e-12)
    np.testing.assert_allclose(start+wa*x+wb*y-d.fee-d.borrow,d.net_nav,atol=2e-10)
    if exp==3 and method=="rolling":
        day=d.date.iloc[-1]
        hist=r.iloc[r.index.get_loc(day)-121:r.index.get_loc(day)-1]
        np.testing.assert_allclose(d.h.iloc[-1],np.polyfit(hist[b],hist[a],1)[0],atol=1e-10)
    count+=1
e=pd.read_csv(OUT/"e1_all_entries.csv",parse_dates=["entry","exit"])
z=np.load(OUT/"e1_holding_paths.npz")
n=len(e)
np.testing.assert_array_equal(e.entry,p.index[:n])
np.testing.assert_array_equal(e.exit,p.index[252:])
v=np.ones(n); pos_a=np.zeros(n);pos_b=np.zeros(n)
for offset in range(252):
    before=v.copy()
    for _ in range(6):
        sized=.5*v
        v=before-.0005*(abs(sized-pos_a)+abs(-sized-pos_b))
    sized=.5*v
    x=r["000001"].iloc[offset:offset+n].values;y=r["000300"].iloc[offset:offset+n].values
    gap=(p.index[offset+1:offset+1+n]-p.index[offset:offset+n]).days
    pos_a=sized*(1+x);pos_b=-sized*(1+y)
    v=np.asarray(v+sized*(x-y)-sized*.02*gap/365)
    if offset==251:
        v-=.0005*(abs(pos_a)+abs(pos_b))
    np.testing.assert_allclose(v,z["nav"][:,1,offset+1],atol=1e-12)
    np.testing.assert_allclose(z["nav"][:,0,offset+1],z["nav"][:,0,offset]*(1+(x-y)/2),atol=1e-12)
np.testing.assert_allclose(e.net,v-1,atol=1e-12)
deployed=pd.read_csv(OUT/"e2_fresh_test_paths.csv",dtype={"a":str,"b":str},parse_dates=["date"])
for (a,b),d in deployed.groupby(["a","b"]):
    value=1.
    pos_a=pos_b=0.
    for row in d.itertuples():
        start=value
        for _ in range(7):
            value=start-.0005*(abs(.5*value-pos_a)+abs(-.5*value-pos_b))
        gap=(p.index[p.index.get_loc(row.date)]-p.index[p.index.get_loc(row.date)-1]).days
        sized=value/2
        x,y=r.loc[row.date,[a,b]]
        pos_a,pos_b=sized*(1+x),-sized*(1+y)
        value+=sized*(x-y)-sized*.02*gap/365
        if row.date==d.date.iloc[-1]:
            value-=.0005*(abs(pos_a)+abs(pos_b))
        np.testing.assert_allclose([row.net_nav,row.net],[value,value/start-1],atol=1e-10)
continuous=daily[daily.experiment.eq(2)]
ranking=continuous[continuous.date.between("2018-01-01","2024-12-30")].groupby(["a","b"]).net.apply(
    lambda returns: np.expm1(np.log1p(returns).sum()))
realized=deployed.groupby(["a","b"]).net.apply(lambda returns: np.expm1(np.log1p(returns).sum()))
hindsight=pd.read_csv(OUT/"e2_hindsight_comparison.csv",dtype={col:str for col in
    ("long_at_selection","short_at_selection","long_in_hindsight","short_in_hindsight")})
assert len(hindsight)==3 and len(realized)==6
seen=set()
for row in hindsight.itertuples():
    selected=(row.long_at_selection,row.short_at_selection)
    opposite=selected[::-1]
    seen.add(tuple(sorted(selected)))
    assert ranking[selected]>=ranking[opposite]
    best=max((selected,opposite),key=realized.get)
    assert best==(row.long_in_hindsight,row.short_in_hindsight)
    np.testing.assert_allclose([row.selected_net,row.reversed_net,row.hindsight_net,row.hindsight_gap],
        [realized[selected],realized[opposite],realized[best],realized[best]-realized[selected]],atol=1e-10)
assert len(seen)==3
checks=dict(status="passed",entry_windows=n,holding_sessions=252,other_paths=count,selection_pairs=len(seen),
            other_path_days=len(daily),fresh_test_path_days=len(deployed),gross_and_cost_checks=True)
(OUT/"verification.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf8")
print(checks)
