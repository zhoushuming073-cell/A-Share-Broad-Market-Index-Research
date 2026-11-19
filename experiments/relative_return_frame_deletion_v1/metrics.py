from scipy.stats import skew
from data import *
def describe(keep):
    a=A[keep];b=B[keep];d=b-a
    s=d.sum();path=np.cumsum(d);corr=np.corrcoef(a,b)[0,1]
    beta=np.cov(a,b)[0,1]/a.var(ddof=1);alpha=b.mean()-beta*a.mean()
    return dict(n=len(d),D=s,Q=np.expm1(s),wealth_a=np.exp(a.sum()),wealth_b=np.exp(b.sum()),
        daily_mean=d.mean(),daily_sd=d.std(ddof=1),skew=skew(d,bias=False),corr=corr,r2=corr**2,
        alpha=alpha,beta=beta,residual_sd=np.sqrt(np.sum((b-alpha-beta*a)**2)/(len(d)-2)),
        rms=np.sqrt(np.mean(path**2)),max_dev=abs(path).max(),same_sign=np.mean(np.sign(a)==np.sign(b)),
        pos=d[d>0].sum(),neg=d[d<0].sum(),positive=int((d>0).sum()),negative=int((d<0).sum()),
        cancellation=1-abs(s)/abs(d).sum())
def path(keep):
    la=np.r_[0.,np.cumsum(np.where(keep,A,0))];lb=np.r_[0.,np.cumsum(np.where(keep,B,0))]
    return pd.DataFrame(dict(date=P.index,wealth_a=np.exp(la),wealth_b=np.exp(lb),D=lb-la,deleted=np.r_[False,~keep]))
