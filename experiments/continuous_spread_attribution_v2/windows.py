import numpy as np
import pandas as pd
from models import C,O,estimate
COMP=['intercept','market','up_extra','dynamic','SMB','VMG','PMO','error']
def periods(r):
    rows=[];cases={}
    for w in C['windows']:
        speed=r.d.rolling(w).mean()
        def record(a,b,kind,sign):
            g=r.iloc[a:b+1]
            rows.append(dict(window=w,kind=kind,start=g.index[0],end=g.index[-1],sign=sign,n=len(g),
                raw_bp=g.d.sum()*1e4,market_log_bp=g.market.sum()*1e4,vol_bp=g.d.std()*1e4,
                mean_speed_bp=g.d.mean()*1e4,agree=(g.d*sign>0).mean()))
        for t,sign in [(speed.idxmax(),1),(speed.idxmin(),-1)]:
            b=r.index.get_loc(t);record(b-w+1,b,'extreme',sign)
        signs=np.sign(speed.fillna(0))
        for _,g in signs.groupby(signs.ne(signs.shift()).cumsum()):
            if g.iloc[0]:record(r.index.get_loc(g.index[0]),r.index.get_loc(g.index[-1]),'speed_run',g.iloc[0])
        if w==60:
            for label,t in [('2017_peak',speed.idxmax()),('2021_high',speed.loc['2021'].idxmax()),
                ('2021_low',speed.loc['2021'].idxmin()),('2026_trough',speed.idxmin())]:
                b=r.index.get_loc(t);cases[label]=r.index[b-59:b+1]
    df=pd.DataFrame(rows);df.to_csv(O/'periods.csv',index=False)
    return df,cases

def masks(r):
    ix=np.lib.stride_tricks.sliding_window_view(np.arange(len(r)),60)
    v=r.d.values[ix];sign=np.where(v.sum(axis=1)>=0,1,-1)
    ranks={'local_win':np.argsort(-v*sign[:,None],axis=1,kind='stable'),'local_abs':np.argsort(-abs(v),axis=1,kind='stable')}
    pos,neg,ab=[np.argsort(-u,kind='stable') for u in [r.d.values,-r.d.values,abs(r.d.values)]]
    yield 'raw',0,np.zeros(ix.shape,bool)
    for k in C['k']:
        for name,rank in ranks.items():
            deleted=np.zeros(ix.shape,bool);np.put_along_axis(deleted,rank[:,:k],True,axis=1)
            yield name,k,deleted
        yield 'global_win',k,np.where(sign[:,None]>0,np.isin(ix,pos[:k]),np.isin(ix,neg[:k]))
        yield 'global_abs',k,np.isin(ix,ab[:k])

def summarize(r,models):
    pdf,cases=periods(r); ix=np.lib.stride_tricks.sliding_window_view(np.arange(len(r)),60)
    masklist=list(masks(r));allrows=[]; attr=[];sens=[];cf=[];period_attr=[]
    for rule,k,mask in masklist:
        y=np.where(mask,0,r.d.values[ix]);allrows.append(pd.DataFrame(dict(start=r.index[ix[:,0]],end=r.index[ix[:,-1]],rule=rule,K=k,
            removed=mask.sum(axis=1),raw_bp=r.d.values[ix].sum(axis=1)*1e4,net_bp=y.sum(axis=1)*1e4)))
    pd.concat(allrows).to_csv(O/'neutralized_windows.csv.gz',index=False)
    for (panel,model),g in models.groupby(['panel','model']):
        g=g.set_index('date').reindex(r.index)
        cols=g[COMP].fillna(0); active=g.error.notna()
        for row in pdf.itertuples():
            seg=cols.loc[row.start:row.end]
            if active.loc[seg.index].all():
                period_attr.append(dict(panel=panel,model=model,start=row.start,end=row.end,kind=row.kind,W=row.window,
                                    raw_bp=row.raw_bp,**(seg.sum()*1e4).to_dict()))
        errors=g.error.values[ix];fitted=g.fitted.values[ix]
        ok=np.isfinite(errors).all(axis=1)
        for rule,k,mask in masklist:
            e=np.where(mask,0,errors).sum(axis=1);pred=np.where(mask,0,fitted).sum(axis=1)
            cf.append(pd.DataFrame(dict(panel=panel,model=model,end=r.index[59:][ok],rule=rule,K=k,
                                     fitted_bp=1e4*pred[ok],error_bp=1e4*e[ok])))
            for label,dates in cases.items():
                i=r.index.get_loc(dates[-1])-59
                if not ok[i]:continue
                keep=~mask[i];total=r.loc[dates[keep],'d'].sum()*1e4
                parts=cols.loc[dates[keep]].sum()*1e4
                np.testing.assert_allclose(parts.sum(),total,atol=1e-8)
                attr.append(dict(case=label,start=dates[0],end=dates[-1],panel=panel,model=model,rule=rule,K=k,raw_bp=total,**parts.to_dict()))
                if model not in ('M2','M3','M4A','M5') or k!=2:continue
                train=r if panel=='full' else r.dropna()
                denied=dates[~keep]
                if rule.startswith('global'):
                    v=r.d.values if rule=='global_win' else abs(r.d.values)
                    if rule=='global_win' and r.loc[dates,'d'].sum()<0:v=-v
                    denied=r.index[np.argsort(-v,kind='stable')[:k]]
                if not len(denied):continue
                a=train.index.get_loc(dates[0]);b=train.index.get_loc(dates[-1])+1
                refit=estimate(train,panel,model,denied,a,b).set_index('date')
                sens.append(dict(case=label,panel=panel,model=model,rule=rule,K=k,
                    frozen_error_bp=parts.error,refit_error_bp=refit.loc[dates[keep],'error'].sum()*1e4))
    for filename,records in [('case_attribution',attr),('case_refits',sens),('period_attribution',period_attr)]:
        pd.DataFrame(records).to_csv(O/(filename+'.csv.gz'),index=False)
    pd.concat(cf).to_csv(O/'counterfactual_errors.csv.gz',index=False)
