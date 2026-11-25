import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from report_body import render
from core import OUT,BASE,CFG,hac

def read(file):return pd.read_csv(OUT/file)
def table(df):
    display=df.copy()
    for col in display.select_dtypes('number'):
        if pd.api.types.is_integer_dtype(display[col]):
            display[col]=display[col].map(str)
        else:
            display[col]=display[col].map(lambda x: format(x,'.4g'))
    return display.to_markdown(index=False,disable_numparse=True)
def interval(y,lag=10):
    f,c=hac(y,np.ones((len(y),1)),lag)
    v=f.params[0];se=np.sqrt(c[0,0])
    return v,v-1.96*se,v+1.96*se,2*norm.sf(abs(v)/se)
def build(r):
    d=read('daily.csv.gz');d.date=pd.to_datetime(d.date)
    s=d.query("method=='A' and K==0").set_index('date')
    models=read('models.csv');peaks=read('peaks.csv');decomp=read('decomposition.csv')
    sim=read('simulations.csv');null=read('residual_null.csv');reg=read('regimes.csv')
    p=peaks.query("method=='A' and K==0")
    o=read('oos.csv').set_index('date')
    es=[]
    for j,t in enumerate(range(CFG['warmup'],len(r))):
        m=r.market.values; X=np.c_[np.ones(t),np.maximum(m[:t],0),np.minimum(m[:t],0)]
        co=np.linalg.lstsq(X,r.d.values[:t],rcond=None)[0]
        es.append(r.d.iloc[t]-np.array([1,max(0,m[t]),min(0,m[t])])@co)
    o['asym_error']=es
    o.to_csv(OUT/'oos_comparison.csv')
    validation=[]
    for lab,g in [('full',o),*list(o.groupby(o.index.str[:4]))]:
        for bench in ['fixed_error','asym_error']:
            diff=(g[bench]**2-g.dynamic_error**2)*1e8
            v,lo,hi,pv=interval(diff.values)
            validation.append([lab,bench,len(g),1-(g.dynamic_error**2).sum()/(g[bench]**2).sum(),v,lo,hi,pv])
    validation=pd.DataFrame(validation,columns='period comparator n gain loss_bp2 lo hi p'.split())
    validation.to_csv(OUT/'validation.csv',index=False)
    # Adjacent calendar-year slopes, with a HAC interaction test across the shared boundary.
    years=sorted(set(r.index.year)); changes=[]
    for first,second in zip(years,years[1:]):
        g=r[r.index.year.isin([first,second])]
        later=(g.index.year==second).astype(float)
        X=np.c_[np.ones(len(g)),g.market,later,later*g.market]
        fit,c=hac(g.d.values,X,10);v=fit.params[-1];se=np.sqrt(c[-1,-1])
        changes.append([first,second,v,v-1.96*se,v+1.96*se,2*norm.sf(abs(v)/se)])
    changes=pd.DataFrame(changes,columns='first second change lo hi p'.split())
    changes['holm']=multipletests(changes.p,method='holm')[1]
    changes.to_csv(OUT/'annual_changes.csv',index=False)
    runs=read('runs.csv.gz')
    rr=runs.query("series in ['residual','daily']").copy()
    rr['same_market']=rr.sign*rr.market_bp>0
    rr['abs_bp']=rr.contribution_bp.abs()
    by=rr.groupby(['method','K','series','sign'])
    rs=by.agg(count=('days','size'),longest=('days','max'),sum_bp=('contribution_bp','sum'),
              fraction_market_same=('same_market','mean'),trading_days=('days','sum'),
              largest_bp=('abs_bp','max')).reset_index()
    rs['largest_share']=rs.largest_bp/rs.sum_bp.abs()
    rs.to_csv(OUT/'run_summary.csv',index=False)
    # Pointwise window HAC interval around local mean prediction error, using Bartlett lag 10.
    band=[]
    for t in range(59,len(s)):
        v,l,h,_=interval(s.residual.values[t-59:t+1]*1e4)
        band.append([s.index[t],v,l,h])
    band=pd.DataFrame(band,columns=['date','mean','lo','hi']).set_index('date')
    band.to_csv(OUT/'residual_band.csv')
    titles=['日相对收益与累计偏移','不同期限的平均相对收益速度','关键日中和后的60日速度','最大60日速度及窗口结束日期','固定斜率与滚动OLS','上涨日与下跌日的相对斜率','随机游走相对Beta','累计相对收益的条件归因','单日同号区间的长度分布','市场调整残差的60日平均速度']
    plt.rcParams.update({'font.family':['Microsoft YaHei'],'axes.unicode_minus':False})
    for j,title in enumerate(titles,1):
        n=2 if j in [1,4] else 3 if j==3 else 1
        fig,axes=plt.subplots(n,1,figsize=(10,3+n*1.35),layout='constrained')
        ax=np.atleast_1d(axes)[0]
        if j==1:
            ax.plot(s.index,s.speed_bp,lw=.55);axes[1].plot(s.index,s.cum_bp)
            axes[1].set_ylabel('累计对数基点')
        if j==2:
            for w in CFG['windows']:ax.plot(s.index,s[f'v{w}_bp'],lw=.9,label=f'{w}日')
        if j==3:
            for a,method in zip(axes,'ABC'):
                for k in [0,2,10,20]:
                    z=d.query('method==@method and K==@k');a.plot(z.date,z.v60_bp,lw=.85,label=f'K={k}')
                a.set_ylabel(method+' 基点/日');a.legend(ncol=4,fontsize=8)
        if j==4:
            for m,g in peaks.query('window==60').groupby('method'):
                ax.plot(g.K,g.max_bp,'o-',label=m)
                axes[1].plot(g.K,pd.to_datetime(g.max_date),'o-',label=m)
            axes[1].set_ylabel('峰值窗口结束日');axes[1].set_xlabel('中和天数 K')
        if j==5:
            for w in [60,120,252]:ax.plot(s.index,s[f'ols{w}'],lw=.85,label=f'OLS {w}日')
            ax.axhline(s.fixed_beta.iloc[0],label='固定Beta',color='black')
        if j==6:
            z=models.query("method=='A' and K==0 and proxy=='000985' and lag==10 and term in ['up','down']")
            ax.errorbar(['上涨日','下跌日'],z.value,yerr=1.96*z.se,fmt='o',capsize=5);ax.axhline(0,c='grey',lw=.8)
        if j==7:
            ax.plot(s.index,s.post_beta,label='收盘更新值',lw=1)
            ax.fill_between(s.index,s.post_beta-1.96*s.post_se,s.post_beta+1.96*s.post_se,alpha=.2,label='条件状态95%带')
        if j==8:
            for label,values in [('实际差额',s.cum_bp),('市场',s.market_component.cumsum()*1e4),('截距',s.intercept.cumsum()*1e4),('残差',s.residual.cumsum()*1e4)]:
                ax.plot(s.index,values,label=label)
        if j==9:
            for k,m,l in [(0,'A','原始'),(2,'A','A删2日'),(20,'C','C删20日')]:
                for sig,style in [(1,'-'),(-1,'--')]:
                    z=rr.query("method==@m and K==@k and sign==@sig and series=='daily'").days.value_counts().sort_index()
                    ax.plot(z.index,z,style,label=l+(' 正' if sig==1 else ' 负'))
            ax.set_yscale('log');ax.set_xlabel('连续交易日数')
        if j==10:
            ax.plot(band.index,band['mean'],lw=1,label='60日均值')
            ax.fill_between(band.index,band.lo,band.hi,alpha=.23,label='逐窗HAC 95%区间')
        ax.set_title(title)
        ax.set_ylabel('基点/交易日' if j in [1,2,3,4,10] else '相对Beta' if j in [5,6,7] else '累计对数基点' if j==8 else '区间数')
        if ax.get_legend_handles_labels()[0]:ax.legend(ncol=4,fontsize=8)
        for a in np.atleast_1d(axes):a.grid(alpha=.18)
        fig.savefig(BASE/f'figures/fig_{j:02d}.png',dpi=180);plt.close(fig)
    render(r,s,models,p,peaks,decomp,reg,rr,rs,validation,changes,sim,null,titles,o)
