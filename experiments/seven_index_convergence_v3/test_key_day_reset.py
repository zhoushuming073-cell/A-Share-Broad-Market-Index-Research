import unittest

import numpy as np
import pandas as pd

from key_day_reset import DRAWS, PERIODS, masks, paths, random_distances
from run import A, B, OUT, SOURCE, SPLIT, WINDOWS


class KeyDayResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        close = pd.read_csv(SOURCE,index_col=0,parse_dates=True)
        cls.r = np.log(close).diff().dropna()
        cls.cases = masks(cls.r.index)

    def test_raw_reference_and_flat_days(self):
        old = pd.read_csv(OUT/"rolling_pool.csv")
        x = self.r.to_numpy()
        for w in WINDOWS:
            rms = paths(x,w)
            np.testing.assert_allclose(np.mean(rms**2,axis=1)**.5,
                                       old.loc[old.window.eq(w),"reset"],atol=1e-11)
        for row,slots in self.cases:
            x = self.r.to_numpy().copy()
            x[slots] = 0
            for w in WINDOWS:
                start = min(int(slots[0]),len(x)-w)
                direct = x[start:start+w].cumsum(axis=0)
                expected = np.mean((direct[:,A]-direct[:,B])**2,axis=0)**.5
                np.testing.assert_allclose(paths(x,w)[start],expected,atol=1e-11)

    def test_random_engine_and_archived_tail(self):
        x = self.r.to_numpy()
        split = self.r.index.get_loc(SPLIT)
        row,slots = next((r,s) for r,s in self.cases if r.scenario=="U_K10")
        calc = random_distances(x-x.mean(axis=1,keepdims=True),slots[None,:],split)[0]
        tests = pd.read_csv(OUT/"key_day_random_tests.csv")
        group = tests.loc[tests.scenario.eq(row.scenario)]
        with np.load(OUT/"key_day_random_draws.npz") as f:
            for datum in group.itertuples():
                w,p = WINDOWS.index(datum.window),PERIODS.index(datum.period)
                self.assertAlmostEqual(calc[w,p],datum.modified)
                v = f[f"early{datum.deleted_early}_late{datum.deleted_late}"][:,w,p]
                improvement = 1-v/datum.original
                expected = (1+np.count_nonzero(improvement>=datum.gain-1e-12))/(DRAWS+1)
                self.assertAlmostEqual(expected,datum.p)
        equal = tests.query("kind=='budget' and method in ['U','R2']")
        self.assertEqual(equal.mask_id.nunique(),1)
        self.assertEqual(tests.mask_id.nunique(),5)


if __name__=="__main__":
    unittest.main()
