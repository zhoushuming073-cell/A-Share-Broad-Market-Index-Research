"""Secondary comparisons, outside the frozen six-test family."""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from models import O,loss_test

def rolling_check(r,daily):
    rows=[]
    for panel,g in daily.groupby('panel'):
        sample=r if panel=='full' else r.dropna()
        errors=g.pivot(index='date',columns='model',values='error').reindex(r.index)
        fixed=g[g.model=='M1'].set_index('date').beta
        y=sample.d if panel=='full' else np.expm1(sample.d)
        x=sample.market if panel=='full' else sample.CH4_excess
        # A daily expanding slope, with only the intercept recalibrated on the most recent 60 observations.
        b=fixed.reindex(sample.index)
        fit=y.rolling(60).mean().shift()+b*(x-x.rolling(60).mean().shift())
        errors['local_intercept']=sample.d-(fit if panel=='full' else np.log1p(fit))
        pairs=[('M2','OLS60'),('local_intercept','OLS60')]
        if panel=='CH':pairs.append(('M2','M4A'))
        for base,model in pairs:
            eb=errors[base].rolling(60).sum(); em=errors[model].rolling(60).sum()
            v,se,p=loss_test((eb**2-em**2)*1e8,59)
            rows.append([panel,base,model,1-em.pow(2).sum()/eb.pow(2).sum(),v,se,p])
    df=pd.DataFrame(rows,columns='panel baseline model gain mean_bp2 se p'.split())
    df['holm_secondary']=multipletests(df.p,method='holm')[1]
    df.to_csv(O/'secondary_checks.csv',index=False)
    return df
