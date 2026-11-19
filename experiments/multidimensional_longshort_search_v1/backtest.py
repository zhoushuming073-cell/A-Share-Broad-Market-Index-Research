from numba import njit
from data import np,X,Y,GAPS,CFG
from strategies import targets
COLS="net nav a b turnover fee borrow finance common relative start gross_weight net_weight".split()
@njit(cache=True)
def engine(x,y,gap,wa,wb,freq,hold,limit,fee,borrow,fin):
    n=len(x);out=np.zeros((n,13));v=peak=lagged=1.;a=b=0.;age=cool=0
    for i in range(n):
        start=v;turn=0.
        forced=abs(a)+abs(b)>2.5*v+1e-12
        stop=(limit>0 and lagged/peak<1-limit and cool==0)
        if stop:cool=10;peak=lagged
        expired=age>=hold
        if i%freq==0 or forced or stop or expired or cool:
            w,z=wa[i],wb[i]
            if cool or expired:w=z=0.
            post=v
            for _ in range(7):post=v-fee*(abs(w*post-a)+abs(z*post-b))
            turn=abs(w*post-a)+abs(z*post-b);a=w*post;b=z*post;v-=fee*turn
        age=age+1 if abs(a)+abs(b)>1e-12 else 0
        cool=max(0,cool-1)
        short=max(-a,0.)+max(-b,0.);long=max(a,0.)+max(b,0.)
        bc=short*borrow*gap[i]/365;fc=max(long-v,0.)*fin*gap[i]/365
        cm=(a+b)*(x[i]+y[i])/2;rel=(a-b)*(x[i]-y[i])/2
        av,bv=a,b
        v+=cm+rel-bc-fc;a*=1+x[i];b*=1+y[i]
        if i==n-1:turn+=abs(a)+abs(b);v-=fee*(abs(a)+abs(b))
        if v<=0:raise ValueError("insolvent")
        lagged=start;peak=max(peak,lagged)
        out[i]=np.array([v/start-1,v,av,bv,turn,turn*fee,bc,fc,cm,rel,start,(abs(av)+abs(bv))/start,(av+bv)/start])
    return out
def simulate(p,l,r,fee=None,borrow=None,finance=None):
    a,b=targets(p);s=slice(l,r)
    return engine(X[s],Y[s],GAPS[s],a[s],b[s],p["frequency"],p.get("hold",100000),p.get("dd_limit",0),
        CFG["fee"] if fee is None else fee,CFG["borrow"] if borrow is None else borrow,CFG["finance"] if finance is None else finance)
