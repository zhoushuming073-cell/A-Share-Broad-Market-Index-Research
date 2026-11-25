import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from core import OUT,CFG,load_data,ranked
from analysis import intervals
from simulation import batch

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=load_data()
    def test_interruption(self):
        a=intervals(np.array([1.,2.,0.,3.,np.nan,-1.]),np.ones(6),np.arange(6))
        self.assertEqual(a.days.tolist(),[2,1,1])
    def test_all_results(self):
        r=self.r
        for (m,k),g in pd.read_csv(OUT/'daily.csv.gz').groupby(['method','K']):
            mask=g.removed.values
            expected=np.zeros(len(r),bool);expected[ranked(r.d.values,m)[:k]]=True
            np.testing.assert_array_equal(mask,expected)
            y=r.d.mask(mask,0)
            np.testing.assert_allclose(y.rolling(60).mean()*1e4,g.v60_bp,atol=1e-8)
            np.testing.assert_allclose(y.cumsum()*1e4,g.cum_bp,atol=1e-7)
            parts=g[['residual','intercept','market_component']]
            self.assertTrue(parts[mask].isna().all().all())
            np.testing.assert_allclose(parts.sum(axis=1),y,atol=1e-10)
            fit=np.polyfit(r.market[~mask],r.a[~mask],1)-np.polyfit(r.market[~mask],r.b[~mask],1)
            self.assertAlmostEqual(fit[0],g.fixed_beta.iloc[0],10)
    def test_seed(self):
        p=self.r[['a','b','market']].values[:90].copy()
        args=(p,5,123,np.array([0,2]),np.array([20,60]),4)
        np.testing.assert_array_equal(batch(*args),batch(*args))
    def test_wealth(self):
        from core import ROOT
        p=pd.read_csv(ROOT/'experiments/four_pair_tests_v2/data/input_closes.csv')
        wealth=p['000300']/p['000001']
        self.assertAlmostEqual(np.exp(self.r.d.sum()),wealth.iloc[-1]/wealth.iloc[0],12)
    def test_oos_chronology(self):
        from core import kalman,filt
        o=pd.read_csv(OUT/'oos.csv')
        n=CFG['warmup'];r=self.r
        a,b,noise,q=kalman(r.d.values[:n],r.market.values[:n])
        *_,beta,_=filt(r.d.values[:n],r.market.values[:n],a,b,noise,q)
        self.assertAlmostEqual(o.dynamic_error.iloc[0],r.d.iloc[n]-a-beta[-1]*r.market.iloc[n],12)


    def test_kalman_against_joint_gaussian(self):
        from core import filt
        from scipy.linalg import solve
        # Independent dense Gaussian likelihood with a missing observation.
        x=np.array([.02,-.03,.01,.025,-.02])
        y=np.array([.001,.002,np.nan,.0004,-.0006])
        a,b,noise,q=.0001,.03,.00002,.0004
        ix=np.flatnonzero(np.isfinite(y));t=ix+1
        cov=q*np.minimum.outer(t,t)*np.outer(x[ix],x[ix])+noise*np.eye(len(ix))
        e=y[ix]-a-b*x[ix]
        expected=np.linalg.slogdet(cov)[1]+e@solve(cov,e,assume_a='pos')
        self.assertAlmostEqual(filt(y,x,a,b,noise,q)[0],expected,10)

    def test_bootstrap_reselects_and_searches(self):
        from simulation import batch
        p=self.r[['a','b','market']].values[:83].copy()
        ks=np.array([0,2,5]);ws=np.array([20,60]);seed=431
        result=batch(p,5,seed,ks,ws,1)[0]
        rng=np.random.RandomState(seed)
        loc=np.concatenate([(rng.randint(len(p))+np.arange(5))%len(p) for _ in range(17)])[:len(p)]
        gap=p[:,0]-p[:,1];d=gap[loc]-gap.mean()
        for m,name in enumerate('ABC'):
            for j,k in enumerate(ks):
                z=d.copy();z[ranked(z,name)[:k]]=0
                for h,w in enumerate(ws):
                    means=np.convolve(z,np.ones(w)/w,mode='valid')
                    np.testing.assert_allclose(result[m,j,3+2*h:5+2*h],[means.max(),means.min()],atol=1e-12)

    def test_hac_against_statsmodels(self):
        import statsmodels.api as sm
        from core import hac
        x=sm.add_constant(self.r.market.to_numpy());y=self.r.d.to_numpy()
        for lag in CFG['hac']:
            fit,cov=hac(y,x,lag)
            other=sm.OLS(y,x).fit(cov_type='HAC',cov_kwds={'maxlags':lag,'use_correction':False})
            np.testing.assert_allclose(cov,other.cov_params(),rtol=1e-10)
    def test_residual_window_selection(self):
        from residual_check import residual_stats
        y=self.r.d.to_numpy();x=self.r.market.to_numpy()
        rank=ranked(y,'C'); keep=np.ones(len(y),bool);keep[rank[:20]]=False
        result=residual_stats(y,x,rank,20)
        co=np.polyfit(x[keep],y[keep],1)
        e=np.where(keep,y-np.polyval(co,x),0.)
        averages=pd.Series(e).rolling(60).mean().dropna().values
        for sign,h in [(1,5),(-1,6)]:
            chosen=[]
            for i in np.argsort(-sign*averages):
                if all(abs(i-j)>=60 for j in chosen):chosen.append(i)
                if len(chosen)==3:break
            self.assertAlmostEqual(result[h],sign*averages[chosen[-1]],12)
