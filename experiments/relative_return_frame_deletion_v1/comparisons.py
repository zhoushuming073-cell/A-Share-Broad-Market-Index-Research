import sys
from data import *
sys.path.insert(0,str(ROOT.parents[1]/"scripts"))
from optimize_hedge_ratio import account

def build():
    q=np.load(OUT/"terminal_Q.npz");rows=[]
    for r in read("deterministic_deletion").itertuples():
        if r.k==0:continue
        for method in ["uniform","year","month","block5","block10","block20","block60"]:
            sample=q[method+"_"+str(r.k)]
            rows.append(dict(rule=r.rule,k=r.k,random_method=method,target_Q=r.Q,
                below=np.mean(sample<r.Q-1e-12),equal=np.mean(abs(sample-r.Q)<=1e-12),random_median=np.median(sample),
                random_p05=np.quantile(sample,.05),random_p95=np.quantile(sample,.95)))
    save(rows,"equal_budget_comparison")
    days=np.diff(P.index.values).astype("timedelta64[D]").astype(float)
    acct=account(np.expm1(B),np.expm1(A),np.full(N,.5),np.full(N,-.5),days,np.ones(N,bool),.0005,.02)
    gross=(np.expm1(B)-np.expm1(A))/2
    save([dict(gross=np.prod(1+gross)-1,net=acct[-1,1]-1,fee=acct[:,6].sum(),borrow=acct[:,5].sum())],"hedge_bridge")
