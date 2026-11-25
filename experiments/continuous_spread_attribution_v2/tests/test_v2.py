import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from models import B,O,V1,C,inputs,estimate
from nulls import extremes,simulate
from windows import masks,COMP

class Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=inputs()
        cls.daily=pd.read_csv(O/'daily.csv.gz',parse_dates=['date'])
    def test_v1_replay(self):
        v=pd.read_csv(V1/'results/oos_comparison.csv',parse_dates=['date'])
        for m,col in [('M1','fixed_error'),('M2','asym_error'),('M3','dynamic_error')]:
            g=self.daily.query("panel=='full' and model==@m")
            np.testing.assert_array_equal(v.date.values,g.date.values)
            np.testing.assert_allclose(v[col],g.error,atol=5e-9)
    def test_masks_on_real_cases(self):
        r=self.r;loc=r.index.get_loc('2017-11-21')
        z=r.d.iloc[loc-59:loc+1]
        for rule,k,mask in masks(r):
            removed=set(z.index[mask[loc-59]])
            if rule=='local_win':self.assertEqual(removed,set(z.nlargest(k).index))
            if rule=='global_abs':self.assertEqual(removed,set(r.d.abs().nlargest(k).index)&set(z.index))
    def test_search_reference(self):
        y=np.random.default_rng(418).normal(size=223);ok=np.ones(len(y),bool);ok[123]=False
        out=extremes(y,ok)
        for j,k in enumerate(C['k']):
            expected=[]
            for i in range(len(y)-59):
                sl=slice(i,i+60)
                if not ok[sl].all():continue
                v=y[sl];side=1 if v.sum()>=0 else -1
                ranked=np.argsort(-side*v);ab=np.argsort(-abs(v))
                seq=np.arange(len(y))[ok];gw=seq[np.argsort(-side*y[ok])[:k]];ga=seq[np.argsort(-abs(y[ok]))[:k]]
                def impact(ids):return y[ids[(ids>=i)&(ids<i+60)]].sum()
                expected.append(np.abs([v.sum(),v.sum()-v[ranked[:k]].sum(),v.sum()-v[ab[:k]].sum(),v.sum()-impact(gw),v.sum()-impact(ga)]))
            np.testing.assert_allclose(out[:,j],np.max(expected,axis=0),atol=1e-11)
    def test_seed(self):
        r=self.r.iloc[:180];x=np.c_[np.ones(len(r)),r.market];z=r.d.values
        args=(x,z,np.ones(len(r),bool),np.zeros(len(r)),z,5,88,4,False,False)
        np.testing.assert_array_equal(simulate(*args)[0],simulate(*args)[0])
    def test_daily_interval_identity(self):
        g=self.daily
        np.testing.assert_allclose(g[COMP].sum(axis=1),g.d,atol=1e-12)
        a=pd.read_csv(O/'case_attribution.csv.gz');np.testing.assert_allclose(a[COMP].sum(axis=1),a.raw_bp,atol=1e-8)
        c=pd.read_csv(O/'counterfactual_errors.csv.gz')
        n=pd.read_csv(O/'neutralized_windows.csv.gz')
        merged=c.merge(n,on=['end','rule','K'],validate='many_to_one')
        self.assertEqual(len(c),len(merged))
        np.testing.assert_allclose(merged.fitted_bp+merged.error_bp,merged.net_bp,atol=1e-7)
    def test_ch_gaps(self):
        g=pd.read_csv(O/'holding_windows.csv.gz')
        dates=self.r.index
        for w in C['windows']:
            v=g.query("panel=='CH' and model=='M5' and W==@w")
            expected=self.r.loc[self.r.dropna().index[C['warmup']:],'d'].reindex(dates).rolling(w).sum().dropna()
            np.testing.assert_array_equal(v.end,expected.index.strftime('%Y-%m-%d'))
            np.testing.assert_allclose(v.d,expected,atol=1e-9)
    def test_current_target_does_not_fit_itself(self):
        r=self.r.dropna().iloc[:540].copy();t=530
        g=estimate(r,'CH','M5',first=t,last=t+1)
        r.iloc[t,r.columns.get_loc('d')]+=.1
        alt=estimate(r,'CH','M5',first=t,last=t+1)
        self.assertAlmostEqual(g.fitted[0],alt.fitted[0],12)
    def test_2021_share(self):
        y=pd.read_csv(O/'yearly.csv')
        v=pd.read_csv(V1/'results/oos_comparison.csv')
        gain=v.asym_error**2-v.dynamic_error**2
        ref=gain[v.date.str.startswith('2021')].sum()/gain.sum()
        got=y.query("panel=='full' and base=='M2' and year==2021").share.iloc[0]
        self.assertAlmostEqual(got,ref,5)
