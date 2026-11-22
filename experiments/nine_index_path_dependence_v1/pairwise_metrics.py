import numpy as np
import pandas as pd
from data_validation import METRICS, OUT, LABELS, PAIRS, spreads

def describe(r):
    rows, keys = [], []
    for (i,j),label,d in zip(PAIRS,LABELS,spreads(r)):
        total = d.sum()
        z = d if total>=0 else -d
        order = np.argsort(-z,kind="stable")
        cum = z[order].cumsum()
        positive = z.clip(min=0)
        hhi = np.square(positive/positive.sum()).sum()
        k = int(np.searchsorted(cum[:(z>0).sum()],abs(total))+1) if total else 0
        values = [k,*[cum[n-1]/positive.sum() for n in (1,5,10,20)],hhi,1/hhi,
                  1-abs(total)/abs(z).sum(),abs(total)]
        row = dict(pair=label,winner=j if total>=0 else i,T=len(d),D=total,Q=np.expm1(abs(total)),
                   k_fraction=k/len(d),**dict(zip(METRICS,values)))
        for n in (1,2,5,10,20):
            row[f"Q_delete_{n}"]=np.expm1(abs(total)-cum[n-1])
        rows.append(row)
        keys.extend(dict(pair=label,rank=n,date=r.index[t],z=z[t]) for n,t in enumerate(order[:20],1))
    return pd.DataFrame(rows),pd.DataFrame(keys)
