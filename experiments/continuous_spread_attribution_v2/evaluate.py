import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from models import C,O,loss_test,design,hac

def evaluate(r,models):
    metrics=[]; rows=[]; comparisons=[];years=[];local=[];regimes=[];breaks=[]
    for panel,p in models.groupby('panel'):
        panel_r=r if panel=='full' else r.dropna()
        calibration=panel_r.iloc[:C['warmup']].d.reindex(r.index)
        series={}
        for model,g in p.groupby('model'):
            g=g.set_index('date').reindex(r.index)
            series[model]=g
            for w in [1,*C['windows']]:
                h=g[['d','fitted','error']].rolling(w).sum().dropna()
                threshold=calibration.rolling(w).sum().abs().quantile(.9)
                agree=np.sign(h.d)==np.sign(h.fitted)
                big=h.d.abs()>=threshold
                miss=(~agree)|(h.fitted.abs()<threshold)
                metrics.append(dict(panel=panel,model=model,W=w,n=len(h),mean_error_bp=h.error.mean()*1e4,
                    mae_bp=h.error.abs().mean()*1e4,mse_bp2=(h.error**2).mean()*1e8,accuracy=agree.mean(),
                    large=int(big.sum()),threshold_bp=threshold*1e4,miss=miss[big].mean()))
                rows.append(h.assign(panel=panel,model=model,W=w,abs_error_bp=h.error.abs()*1e4,
                                     squared_error_bp2=(h.error*1e4)**2,agree=agree,large=big,miss=miss&big).rename_axis('end').reset_index())
        pairs=[('M1','M3'),('M2','M3')]+([('M4A','M5')] if panel=='CH' else [])
        for base,dyn in pairs:
            b=series[base];a=series[dyn]
            daily=(b.error**2-a.error**2)*1e8
            for y,g in daily.dropna().groupby(daily.dropna().index.year):
                outside=daily.where(daily.index.year!=y)
                v,se,pv=loss_test(outside,10)
                years.append(dict(panel=panel,base=base,model=dyn,year=y,n=g.size,sum_gain=g.sum(),
                    share=g.sum()/daily.sum(),leave_gain=outside.sum()/(b.error.where(outside.notna())**2*1e8).sum(),leave_p=pv,
                    leave_mean_bp2=v))
            vol=r.market.rolling(60).std().shift()
            change=a.beta.diff(60).abs()
            for name,z in [('volatility',vol),('beta_change',change)]:
                cut=z.expanding(min_periods=60).median().shift()
                for group,flag in [('high',z>cut),('low',z<=cut)]:
                    v=daily.where(flag&cut.notna()).dropna()
                    regimes.append(dict(panel=panel,base=base,model=dyn,variable=name,group=group,n=len(v),mean_gain_bp2=v.mean()))
            for w in [1,*C['windows']]:
                if w in (60,120,252):
                    roll=daily.rolling(w).sum().dropna()
                    local.append(pd.DataFrame(dict(panel=panel,base=base,model=dyn,W=w,start=r.index[r.index.get_indexer(roll.index)-w+1],
                                                   end=roll.index,sum_gain_bp2=roll.values)))
                eb=b.error.rolling(w).sum();ea=a.error.rolling(w).sum()
                diff=(eb**2-ea**2)*1e8
                for stride in [1,w] if w>1 else [1]:
                    sample=diff
                    if stride>1:
                        # Deterministic grid anchored at first valid endpoint; omit incomplete windows.
                        first=diff.first_valid_index(); start=r.index.get_loc(first)
                        sample=diff.iloc[start::w]
                    v,se,pv=loss_test(sample,max(10,w-1) if stride==1 else 1)
                    comparisons.append(dict(panel=panel,base=base,model=dyn,W=w,stride=stride,n=sample.count(),
                        gain=sample.sum()/(eb.reindex(sample.index).where(sample.notna())**2*1e8).sum(),mean=v,se=se,p=pv))
        for model in (['M1','M2'] if panel=='full' else ['M1','M2','M4A']):
            for y0 in [2020,2021]:
                sub=panel_r.loc[str(y0):str(y0+1)]
                X=design(sub,model,panel)
                marker=(sub.index.year==y0+1).astype(float)
                # Allow every coefficient to change, test only market slopes.
                A=np.c_[X.values,marker[:,None]*X.values]
                y=sub.d.values if panel=='full' else np.expm1(sub.d.values)
                f,cov=hac(y,A,10)
                idx=[X.shape[1]+X.columns.get_loc(n) for n in ['market','up_extra'] if n in X]
                beta=f.params[idx]; v=cov[np.ix_(idx,idx)]
                breaks.append(dict(panel=panel,model=model,year=y0,next_year=y0+1,down_change=beta[0],up_change=beta.sum(),
                    p=chi2.sf(beta@np.linalg.solve(v,beta),len(idx))))
    br=pd.DataFrame(breaks);br['holm']=multipletests(br.p,method='holm')[1]
    for name,frame in [('holding_metrics',pd.DataFrame(metrics)),('holding_tests',pd.DataFrame(comparisons)),
        ('yearly',pd.DataFrame(years)),('regimes',pd.DataFrame(regimes)),('changes',br)]:
        frame.to_csv(O/(name+'.csv'),index=False)
    pd.concat(local).to_csv(O/'local_loss.csv.gz',index=False)
    pd.concat(rows).to_csv(O/'holding_windows.csv.gz',index=False,float_format='%.10g')
