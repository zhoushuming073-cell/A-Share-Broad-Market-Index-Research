from data import *
from backtest import COLS
def ledger(label,acct,idx):
    d=pd.DataFrame(acct,columns=COLS,index=idx)
    d["strategy"]=label
    net=d.a/d.start+d.b/d.start
    market=R.loc[idx].mean(axis=1)
    d["constant_direction"]=net.mean()*d.start*market
    d["dynamic_direction"]=d.common-d.constant_direction
    d["rolling_beta"]=d.net.rolling(120).cov(R.loc[idx].iloc[:,1])/R.loc[idx].iloc[:,1].rolling(120).var()
    return d.rename_axis("date").reset_index()
