import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from models import B,O,C,inputs
from windows import periods

def load(n):
    path=O/(n+'.csv')
    return pd.read_csv(path if path.exists() else O/(n+'.csv.gz'))

def figures(r,cases,mask,case,local,year,metric,daily,tests):
    plt.rcParams.update({'font.family':['Microsoft YaHei'],'axes.unicode_minus':False})
    dest=B/'figures'
    def save(fig,n,title):
        fig.suptitle(title);fig.tight_layout();fig.savefig(dest/f'fig_{n:02}.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for w in C['windows']:ax.plot(r.index,r.d.rolling(w).mean()*1e4,label=f'{w}日',lw=1.5 if w==60 else .75)
    for dates in cases.values():ax.axvspan(dates[0],dates[-1],color='gray',alpha=.16)
    fast=r.d.rolling(60).mean()
    ax.fill_between(r.index,0,.018,where=(fast>0),transform=ax.get_xaxis_transform(),color='#278565')
    ax.fill_between(r.index,0,.018,where=(fast<0),transform=ax.get_xaxis_transform(),color='#bf584e')
    ax.axhline(0,color='gray',lw=.6);ax.set_ylabel('平均对数基点/交易日');ax.legend(ncol=4)
    save(fig,1,'沪深300减上证综指的相对速度｜阴影为案例；底边绿/红标示60日速度正/负')
    fig,axs=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for ax,(label,dates) in zip(axs.flat,cases.items()):
        z=mask[mask.end==str(dates[-1].date())]
        for name,g in z.groupby('rule'):
            if name=='raw':continue
            ax.plot([0,*g.K],[z.raw_bp.iloc[0],*g.net_bp],'o-',label=name)
        ax.set_title(label);ax.set_ylabel('累计对数基点');ax.set_xlabel('中和天数')
    axs.flat[0].legend(fontsize=8);save(fig,2,'相同窗口内两种选日范围、两种排序的影响')
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,(label,dates) in zip(axs.flat,cases.items()):
        panel='full' if label=='2026_trough' else 'CH'
        sub=daily[(daily.panel==panel)&daily.date.isin(dates)]
        for m in ['M1','M2','M3','M4A','M5']:
            g=sub[sub.model==m]
            if len(g):ax.plot(g.date,g.fitted.cumsum()*1e4,label=m)
        g=sub[sub.model=='M1'];ax.plot(g.date,g.d.cumsum()*1e4,'k--',label='实际')
        ax.set_ylabel('累计对数基点');ax.set_title(label+' · '+panel);ax.tick_params(axis='x',rotation=25)
    axs.flat[0].legend(ncol=3,fontsize=8);save(fig,3,'使用前收盘参数的累计归因')
    fig,ax=plt.subplots(figsize=(10,4))
    for base,g in local.query("panel=='full' and W==60").groupby('base'):
        ax.plot(pd.to_datetime(g.end),g.sum_gain_bp2,label='M3 相对 '+base,lw=1)
    ax.axhline(0,c='gray');ax.set_ylabel('60日内日平方损失改善之和（基点²）');ax.legend();save(fig,4,'动态暴露的局部解释增量')
    for n,col,title,yl in [(5,'sum_gain','不同年份对日度损失改善的贡献','年度合计（基点²）'),(6,'leave_gain','排除该年评价观测后的日度改善','相对平方误差降幅')]:
        fig,ax=plt.subplots(figsize=(10,4));year.query("panel=='full'").pivot(index='year',columns='base',values=col).plot.bar(ax=ax)
        ax.axhline(0,c='gray');ax.set_xlabel('评价年份');ax.set_ylabel(yl);save(fig,n,title)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,w in zip(axs,[1,60]):
        g=metric.query("panel=='CH' and W==@w");ax.bar(g.model,np.sqrt(g.mse_bp2));ax.set_title(f'{w}日累计误差');ax.tick_params(axis='x',rotation=50);ax.set_ylabel('均方根误差（基点）')
    save(fig,7,'相同CH验证期的日度与持有期误差')
    fig,ax=plt.subplots(figsize=(11,4))
    for m in ['M2','M3','M4A','M5']:
        g=daily.query("panel=='CH' and model==@m").set_index('date').error
        ax.plot(r.index,g.cumsum().reindex(r.index)*1e4,label=m)
    ax.set_ylabel('验证期已观察日累计残差（基点）');ax.legend();save(fig,8,'加入风格控制前后的剩余路径｜缺测日断线')
    sim=np.load(O/'null_draws.npz');fig,axs=plt.subplots(2,2,figsize=(10,6))
    for ax,m in zip(axs.flat,['zero','M1','M2','M4A']):
        ax.hist(sim[f'{m}_20'],bins=35,color='#83a9bc')
        v=tests.query("null==@m and block==20 and rule=='local_win' and K==2").observed_bp.iloc[0]
        ax.axvline(v,c='firebrick',label=f'实际 {v:.0f} bp');ax.set_title(m);ax.legend();ax.set_xlabel('搜索后最大绝对60日累计差（基点）');ax.set_ylabel('模拟次数')
    save(fig,9,'窗内中和赢家两日后的极值与同步20日区块零模型')

def report():
    from narrative import write
    r=inputs();period,cases=periods(r);daily=load('daily')
    daily.date=pd.to_datetime(daily.date)
    tests=load('null_tests'); ht=load('holding_tests')
    primary=[]
    for row in tests.query("block==20 and rule=='local_win' and K==2").itertuples(index=False,name=None):
        m,b,rule,k,obs,lo,med,hi,p=row
        primary.append([m+' 窗内中和两日极值',obs,'对数基点',p])
    for panel,base,dyn in [('full','M2','M3'),('CH','M4A','M5')]:
        g=ht.query("panel==@panel and base==@base and model==@dyn and W==60 and stride==1").iloc[0]
        primary.append([dyn+' 相对 '+base+' 的60日改善',g.gain*100,'%',g.p])
    claims=pd.DataFrame(primary,columns=['主检验','观察值','单位','p'])
    claims['Holm_p']=multipletests(claims.p,method='holm')[1]
    claims.to_csv(O/'primary.csv',index=False)
    attr=load('case_attribution');raw=attr.query("rule=='raw'")
    extra=[]
    for (label,panel),g in raw.groupby(['case','panel']):
        g=g.set_index('model')
        for base,m in [('M1','M3'),('M2','M3'),('M4A','M5')]:
            if m in g.index:extra.append([label,panel,base,m,g.error[base]-g.error[m]])
    inc=pd.DataFrame(extra,columns='case panel baseline model increment_bp'.split());inc.to_csv(O/'case_increments.csv',index=False)
    mask=load('neutralized_windows');year=load('yearly');met=load('holding_metrics')
    from diagnostics import rolling_check
    rolling_check(r,daily)
    figures(r,cases,mask,attr,load('local_loss'),year,met,daily,tests)
    write(r,period,cases,mask,attr,year,met,tests,claims,ht,load('regimes'),load('changes'),inc,load('case_refits'))

if __name__=='__main__':report()
