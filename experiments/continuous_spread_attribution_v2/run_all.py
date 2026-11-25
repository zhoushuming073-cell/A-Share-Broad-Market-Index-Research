from models import C,O,inputs,estimate
import pandas as pd
import numpy as np
from windows import summarize,COMP
from evaluate import evaluate
import nulls

def estimate_all(r):
    records=[]
    for panel,sample in [('full',r),('CH',r.dropna())]:
        names=['M0','M1','M2','M3',*['OLS'+str(w) for w in (60,120,252)]]
        if panel=='CH':names+=['M4_3','M4_4','M4A','M5']
        records.extend(estimate(sample,panel,m) for m in names)
    daily=pd.concat(records,ignore_index=True)
    daily[COMP]=daily[COMP].fillna(0)
    np.testing.assert_allclose(daily[COMP].sum(axis=1),daily.d,atol=1e-12)
    daily.to_csv(O/'daily.csv.gz',index=False)
    return daily

def main():
    r=inputs()
    daily=estimate_all(r)
    summarize(r,daily)
    evaluate(r,daily)
    nulls.run(r)
    from reporting import report
    report()
if __name__=='__main__':main()
