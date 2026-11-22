import numpy as np
import pandas as pd
from data_validation import OUT,CODES,CFG
from pairwise_metrics import describe
from key_date_overlap import overlap

def run(r):
    rows=[]
    for period,data in (("full",r),("first_half",r.iloc[:len(r)//2]),("second_half",r.iloc[len(r)//2:])):
        metrics,keys=describe(data)
        for omit in ["none"]+CODES:
            chosen=metrics if omit=="none" else metrics[~metrics.pair.str.contains(omit)]
            m=overlap(keys[keys.pair.isin(chosen.pair)],data.index)
            rows.append(dict(period=period,omit=omit,T=len(data),pairs=len(chosen),
                median_C5=chosen.C5.median(),median_k_fraction=chosen.k_fraction.median(),
                joint_share=((chosen.C5>=CFG["concentrated_c5"])&(chosen.k_fraction<=CFG["fragile_fraction"])).mean(),**m))
    pd.DataFrame(rows).to_csv(OUT/"robustness_results.csv",index=False)
