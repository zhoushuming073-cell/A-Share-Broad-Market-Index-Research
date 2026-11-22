import unittest

import numpy as np
import pandas as pd
from data_validation import CFG, CODES, LABELS, OUT, PAIRS
from key_dates import memberships, overlap, scores, rank
from convergence_metrics import evaluate, rms_and_gap
from bootstrap_inference import block_paths, selection, date_stats


class SevenIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = pd.read_csv(OUT/"input_closes_main.csv",index_col=0,parse_dates=True)
        cls.returns = np.log(cls.prices).diff().iloc[1:]
        cls.x = np.ascontiguousarray(cls.returns.values)
        cls.gaps = scores(cls.x)
        cls.order = np.argsort(-cls.gaps,axis=1,kind="stable")
        cls.overall = cls.gaps.mean(axis=0)
        cls.baseline = rms_and_gap(cls.x,np.zeros(len(cls.x),bool))

    def test_fixed_universe(self):
        self.assertEqual(tuple(self.prices.columns),CODES)
        self.assertEqual(len(CODES),7)
        self.assertEqual(len({frozenset(p) for p in PAIRS}),21)
        self.assertEqual(set(CODES),{"000001","000985","000300","000903","000132","399311","000905"})

    def test_membership_and_vote_sets(self):
        for k in CFG["k"]:
            sets, votes, _, chosen = memberships(self.gaps,self.order,self.overall,k)
            self.assertTrue(all(len(set(row))==k for row in chosen))
            actual=np.array([[t in row for t in range(len(votes))] for row in chosen]).sum(axis=0)
            np.testing.assert_array_equal(votes,actual)
            all_lists = [set(row) for row in chosen]
            self.assertEqual(set(sets["U"]),set.union(*all_lists))
            self.assertEqual(set(sets["I21"]),set.intersection(*all_lists))
            self.assertEqual(votes.sum(),21*k)
            for q in (2,3,5,10):
                self.assertEqual(set(sets[f"R{q}"]),set(np.flatnonzero(actual>=q)))
            order=sorted(range(len(votes)),key=lambda t:(-votes[t],-self.overall[t],t))
            np.testing.assert_array_equal(sets["B"],order[:k])

    def test_overlap_denominators_and_matrices(self):
        self.assertEqual(overlap([1,2,3],[3,4])["jaccard"],.25)
        self.assertEqual(overlap([1,2,3],[3,4])["cover_A"],1/3)
        self.assertEqual(overlap([1,2,3],[3,4])["cover_other"],.5)
        self.assertTrue(np.isnan(overlap([1],[])["jaccard"]))
        for k in CFG["k"]:
            m=pd.read_csv(OUT/f"jaccard_main_K{k}.csv",index_col=0)
            self.assertEqual(tuple(m.index),LABELS)
            np.testing.assert_array_equal(m,m.T)
            np.testing.assert_array_equal(np.diag(m),1.)
            self.assertTrue(np.all((m.values>=0)&(m.values<=1)))
            sets,_,_,_=memberships(self.gaps,self.order,self.overall,k)
            report=pd.read_csv(OUT/"method_overlap.csv").query("sample=='main' and K==@k")
            for row in report.itertuples():
                expected=overlap(sets["A"],sets[row.other])
                for field,value in expected.items():
                    self.assertAlmostEqual(getattr(row,field),value)

    def test_synchronous_mask_and_same_axis(self):
        zero=np.empty(0,int)
        result,_,_,curve=evaluate(self.x,zero,self.baseline)
        self.assertEqual(result["improvement"],0.)
        np.testing.assert_allclose(np.expm1(self.x.cumsum(axis=0)),curve)
        paths=pd.read_csv(OUT/"convergence_paths_main.csv",header=[0,1],index_col=0,parse_dates=True)
        self.assertTrue(paths.index.equals(self.prices.index))
        used=pd.read_csv(OUT/"deletion_dates.csv",parse_dates=["date"]).query("sample=='main'")
        for scenario in paths.columns.get_level_values(0).unique():
            mask=self.returns.index.isin(used[used.scenario.eq(scenario)].date)
            expected=self.x.copy()
            expected[mask]=0
            np.testing.assert_allclose(paths[scenario].iloc[1:],np.expm1(expected.cumsum(axis=0)),atol=1e-13)
            if mask.any():
                self.assertTrue(np.all(np.diff(paths[scenario].values,axis=0)[mask]==0))

    def test_ties_and_separate_diagnostic(self):
        self.assertEqual(rank(np.array([1.,1.,3.,1.])).tolist(),[2,0,1,3])
        rows=pd.read_csv(OUT/"independent_pair_deletion.csv").query("sample=='main'")
        self.assertEqual(len(rows),21*len(CFG["k"]))
        self.assertFalse(rows.common_mask.any())
        curves=pd.read_csv(OUT/"convergence_paths_main.csv",header=[0,1],nrows=1)
        self.assertTrue(set(LABELS).isdisjoint(curves.columns.get_level_values(0)))

    def test_block_independently_rebuilds_sets_and_matches_budget(self):
        rng=np.random.default_rng(321)
        times=rng.integers(0,len(self.x),size=(2,120))
        perm=np.array([rng.permutation(times.shape[1]) for _ in times])
        observed,effects,fixed=block_paths(self.x,times,perm,np.array([10]))
        for b,ts in enumerate(times):
            sample=self.x[ts]
            abs_gap=scores(sample)
            order=np.argsort(-abs_gap,axis=1,kind="stable")
            total=abs_gap.mean(axis=0)
            sets,votes,_,chosen=memberships(abs_gap,order,total,10)
            a,u,r,v,w=selection(chosen,rank(total))
            np.testing.assert_allclose(observed[b,0],date_stats(a,u,r,v,chosen,w))
            base=rms_and_gap(sample,np.zeros(len(sample),bool))
            for i,name in enumerate(("A","U","R2","B")):
                slots=sets[name]
                actual,*_=evaluate(sample,slots,base)
                random,*_=evaluate(sample,perm[b,:len(slots)],base)
                self.assertAlmostEqual(actual["improvement"]-random["improvement"],effects[b,0,i,3])
                if len(slots)>=10:
                    matched,*_=evaluate(sample,slots[:10],base)
                    self.assertAlmostEqual(matched["improvement"],fixed[b,0,i])


if __name__=="__main__":
    unittest.main()
