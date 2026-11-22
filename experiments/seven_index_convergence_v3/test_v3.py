import unittest

import numpy as np
import pandas as pd

from run import A, B, METRICS, OUT, SOURCE, SPLIT, auxiliary, early_offset, null_draws, pooled_windows


class ConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closes = pd.read_csv(SOURCE,index_col=0,parse_dates=True)
        cls.r = np.log(cls.closes/cls.closes.shift()).dropna()
        cls.split = cls.r.index.get_loc(SPLIT)

    def test_archived_values_and_windows(self):
        roll = pd.read_csv(OUT/"rolling_pool.csv",parse_dates=["start","end","center"])
        for w, g in roll.groupby("window"):
            self.assertEqual(len(g),len(self.r)-w+1)
            self.assertEqual(g.strict.eq("cross").sum(),w-1)
            np.testing.assert_array_equal(g.midpoint.eq("late"),g.center>=SPLIT)
            for pos in [0,self.split-w+1,len(g)-1]:
                row = g.iloc[pos]
                paths = (self.closes.loc[row.start:row.end]/self.closes.iloc[pos]).apply(np.log).to_numpy()
                self.assertAlmostEqual(row.reset,np.sqrt(np.mean((paths[:,A]-paths[:,B])**2)))
        stats = pd.read_csv(OUT/"window_summary.csv").query("rule=='midpoint'")
        values = stats.pivot(index="window",columns="period",values="reset")
        np.testing.assert_array_equal(np.round(100*(values.late/values.early-1),2),[-5.97,-4.93,-14.32])

    def test_common_market_and_linear_drift(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(620,7))*.01
        shift = rng.normal(size=(620,1))*.02
        np.testing.assert_allclose(pooled_windows(x,310),pooled_windows(x+shift,310),atol=1e-12)
        linear = shift + np.arange(7)*.001
        detrend,drift,corr,daily = auxiliary(linear)
        self.assertAlmostEqual(detrend,0)
        self.assertGreater(drift,0)
        self.assertGreater(daily,0)
        self.assertAlmostEqual(corr,1)

    def test_bootstrap_day_vectors_and_holm(self):
        x = np.ascontiguousarray(self.r.to_numpy())
        starts = np.random.default_rng(15).integers(len(x),size=(1, (len(x)+19)//20))
        positions = ((starts.T+np.arange(20))%len(x)).ravel()[:len(x)]
        z = x[positions]
        expected = np.r_[pooled_windows(z,self.split),auxiliary(z[self.split:])-auxiliary(z[:self.split])]
        np.testing.assert_allclose(null_draws(x,self.split,20,starts)[0],expected,atol=1e-12)
        with np.load(OUT/"bootstrap_draws.npz") as data:
            tests = pd.read_csv(OUT/"bootstrap_tests.csv")
            for row in tests.itertuples():
                draws = data[f"block_{row.block}"][:,METRICS.index(row.metric)]
                p = (1+min(sum(draws<=row.delta),sum(draws>=row.delta)))*2/(len(draws)+1)
                self.assertAlmostEqual(row.p,min(1,p))
                self.assertGreaterEqual(row.holm+1e-12,row.p)

    def test_signed_offset(self):
        t = early_offset(self.r,self.split)
        self.assertEqual(len(t),42)
        for row in t.itertuples():
            a,b = row.pair.split("_")
            gap = (self.r[a]-self.r[b]).iloc[:self.split]
            kept = gap[(gap.index<row.start)|(gap.index>row.end)]
            self.assertEqual(len(gap)-len(kept),row.span)
            self.assertAlmostEqual(kept.sum(),row.endpoint_without)


if __name__ == "__main__":
    unittest.main()
