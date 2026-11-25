import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from core import BASE,OUT,CFG,ranked,kalman,filt,hac

def intervals(y,market,index):
    sign=np.sign(np.nan_to_num(y))
    bounds=np.r_[0,np.flatnonzero(np.diff(sign))+1,len(y)]
    a,b=bounds[:-1],bounds[1:]
    keep=sign[a]!=0
    a,b=a[keep],b[keep]
    def totals(x):
        cs=np.r_[0,np.nancumsum(x)]
        return cs[b]-cs[a]
    return pd.DataFrame(dict(start=index[a],end=index[b-1],sign=sign[a],days=b-a,
                             contribution_bp=1e4*totals(y),market_bp=1e4*totals(market)))

def add_test(rows,label,fit,cov,contrast,term,lag,meta):
    c=np.asarray(contrast)
    val=c@fit.params
    se=np.sqrt(max(0,c@cov@c))
    rows.append(dict(**meta,model=label,term=term,lag=lag,value=val,se=se,lo=val-1.96*se,
                     hi=val+1.96*se,p=2*norm.sf(abs(val)/se) if se else np.nan,
                     r2=fit.rsquared,adj_r2=fit.rsquared_adj,resid_variance=np.var(fit.resid),n=len(fit.resid)))

def model_tests(y,x,rows,meta):
    ones=np.ones(len(x))
    for lag in CFG['hac']:
        for name,X,labels in [('mean',ones[:,None],['drift']),('fixed',np.column_stack([ones,x]),['intercept','gamma']),
                ('asymmetric',np.column_stack([ones,np.maximum(x,0),np.minimum(x,0)]),['intercept','up','down'])]:
            f,c=hac(y,X,lag)
            for vec,term in zip(np.eye(len(labels)),labels):add_test(rows,name,f,c,vec,term,lag,meta)
            if name=='asymmetric':add_test(rows,name,f,c,[0,1,-1],'up_minus_down',lag,meta)

def generate(r):
    x=r.market.to_numpy();d=r.d.to_numpy();idx=r.index
    frames=[];turns=[];runs=[];peaks=[];tests=[];fits=[];regimes=[]
    for method in 'ABC':
        order=ranked(d,method)
        for k in CFG['k']:
            meta=dict(method=method,K=k)
            y=d.copy(); y[order[:k]]=np.nan
            neutral=np.nan_to_num(y); s=pd.Series(neutral,index=idx)
            daily=pd.DataFrame(dict(date=idx,method=method,K=k,removed=np.isnan(y),original_bp=1e4*d,
                                    speed_bp=1e4*neutral,cum_bp=1e4*s.cumsum().values))
            model_tests(y,x,tests,dict(**meta,proxy='000985'))
            params=kalman(y,x)
            ll,prior,var,post,pv=filt(y,x,*params)
            alpha,beta,noise,q=params
            f,_=hac(y,np.column_stack([np.ones(len(y)),x]),10)
            fixed=f.params[0]+f.params[1]*x
            market=prior*x
            u=y-alpha-market
            fits.append(dict(**meta,alpha=alpha,initial_beta=beta,obs_variance=noise,state_variance=q,
                        fixed_beta=f.params[1],fixed_sse=np.nansum((y-fixed)**2),dynamic_sse=np.nansum(u*u),
                        total_bp=np.nansum(y)*1e4,intercept_bp=np.isfinite(y).sum()*alpha*1e4,
                        market_bp=np.sum(market[np.isfinite(y)])*1e4,residual_bp=np.nansum(u)*1e4))
            daily['prior_beta']=prior;daily['post_beta']=post;daily['post_se']=np.sqrt(pv)
            daily['market_component']=np.where(np.isfinite(y),market,np.nan)
            daily['intercept']=np.where(np.isfinite(y),alpha,np.nan)
            daily['residual']=u;daily['fixed_beta']=f.params[1]
            for direction,mask in [('up',x>0),('down',x<0)]:
                ok=mask&np.isfinite(y)
                regimes.append(dict(**meta,state=direction,n=ok.sum(),win=np.mean(y[ok]>0),mean_beta=prior[ok].mean(),
                            raw_bp=y[ok].sum()*1e4,market_bp=market[ok].sum()*1e4,resid_bp=u[ok].sum()*1e4))
            for lag in CFG['hac']:
                fit,cov=hac(u,np.ones((len(x),1)),lag)
                add_test(tests,'innovation',fit,cov,[1],'residual_mean',lag,dict(**meta,proxy='000985'))
            for label,z in [('daily',neutral),('residual',u)]:
                runs.append(intervals(z,x,idx).assign(**meta,series=label))
            for w in CFG['windows']:
                speed=s.rolling(w).mean()*1e4
                daily[f'v{w}_bp']=speed.to_numpy()
                daily[f'sum{w}_bp']=(speed*w).to_numpy()
                runs.append(intervals(speed.to_numpy()/1e4,x,idx).assign(**meta,series=f'v{w}'))
                delta=speed.diff()
                hit=(delta*delta.shift(-1)<0)
                turns.append(pd.DataFrame(dict(date=idx[hit],window=w,method=method,K=k,
                    kind=np.where(delta[hit]>0,'peak','trough'),speed_bp=speed[hit],slope_bp=delta[hit],
                    curvature_bp=(delta.shift(-1)-delta)[hit])).reset_index(drop=True))
                peaks.append(dict(**meta,window=w,max_bp=speed.max(),max_date=speed.idxmax(),
                    min_bp=speed.min(),min_date=speed.idxmin(),cum_max_date=s.cumsum().idxmax(),
                    cum_min_date=s.cumsum().idxmin(),cum_max_bp=s.cumsum().max()*1e4,cum_min_bp=s.cumsum().min()*1e4))
            for w in [60,120,252]:
                ys=pd.Series(y,index=idx);xs=r.market.where(ys.notna());n=ys.rolling(w,min_periods=1).count()
                # Nominal W calendar days, excluding removed observations; require full calendar warmup.
                rolling=xs.rolling(w,min_periods=w-20).cov(ys)/xs.rolling(w,min_periods=w-20).var()
                daily[f'ols{w}']=rolling.where(np.arange(len(y))>=w-1).values
            frames.append(daily)
    ch=r.dropna(subset=['CH3','CH4'])
    for col in ['market','CH3','CH4','CH3_excess','CH4_excess']:
        model_tests(ch.d.to_numpy(),ch[col].to_numpy(),tests,dict(method='A',K=0,proxy=col+'_matched'))
    df=pd.DataFrame(tests)
    for _,g in df.groupby(['proxy','lag','model','term']):
        sel=g[(g.K!=0)|(g.method=='A')].dropna(subset=['p'])
        if len(sel):
            df.loc[sel.index,'holm']=multipletests(sel.p,method='holm')[1]
            if (sel.K==0).any():df.loc[g.index[g.K==0],'holm']=df.loc[sel.index[sel.K==0][0],'holm']
    for name,t in [('daily.csv.gz',pd.concat(frames)),('turns.csv.gz',pd.concat(turns)),('runs.csv.gz',pd.concat(runs)),
                   ('peaks.csv',pd.DataFrame(peaks)),('models.csv',df),('decomposition.csv',pd.DataFrame(fits)),('regimes.csv',pd.DataFrame(regimes))]:
        t.to_csv(OUT/name,index=False,float_format='%.12g')

def oos(r,output="oos.csv"):
    y=r.d.values;x=r.market.values;rows=[]
    for t in range(CFG['warmup'],len(y)):
        if (t-CFG['warmup'])%CFG['refresh']==0:
            params=kalman(y[:t],x[:t]); a,b,noise,q=params
            _,_,_,bf,pf=filt(y[:t],x[:t],*params)
            b,p=bf[-1],pf[-1]
        co=np.polyfit(x[:t],y[:t],1); e=y[t]-a-b*x[t]
        rows.append((r.index[t],y[t]-np.polyval(co,x[t]),e,b,co[0]))
        p+=q; den=noise+p*x[t]**2;b+=p*x[t]/den*e;p*=noise/den
    errors=pd.DataFrame(rows,columns=['date','fixed_error','dynamic_error','prior_beta','fixed_beta'])
    if output:errors.to_csv(OUT/output,index=False)
    return errors
