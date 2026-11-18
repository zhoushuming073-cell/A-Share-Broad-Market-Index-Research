import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from research_common import *
from run_e2_regimes import classify
from run_e4_offsetting import offset_metrics
from run_e5_decomposition import decompose
from run_e8_portfolio import simulate


class ResearchFormulas(unittest.TestCase):
    def test_offset_and_industry_reconciliation(self):
        weights=[.25]*4
        returns=[.08,-.04,.02,-.04]
        m=offset_metrics(weights,returns)
        self.assertAlmostEqual(m["positive"],.025)
        self.assertAlmostEqual(m["negative_abs"],.020)
        self.assertAlmostEqual(m["co"],.04/.045)
        c,n=decompose(np.array(weights)*returns,["X","X","Y","Y"])
        self.assertAlmostEqual(n["X"],.01)
        self.assertAlmostEqual(c["within"],.03)
        self.assertAlmostEqual(c["between"],.01)
        self.assertAlmostEqual(c["total"],m["total_offset"])

    def test_zero_and_invalid_gross(self):
        self.assertTrue(np.isnan(offset_metrics([1],[0])["co"]))
        self.assertAlmostEqual(offset_metrics([.5,.5],[.02,.04])["co"],0)
        with self.assertRaises(ValueError):
            offset_metrics([.5,-.5],[.01,.02])
        with self.assertRaises(ValueError):
            decompose([.01,-.02],["X",None])

    def test_regime_does_not_see_current_close(self):
        ix=pd.bdate_range("2019-01-01",periods=135)
        p=pd.Series(100.,index=ix)
        p.iloc[-2]=115
        expected=classify(p)[0].iloc[-1]
        self.assertEqual(expected,"advance")
        p.iloc[-1]=80
        self.assertEqual(classify(p)[0].iloc[-1],expected)

    def test_drift_cost_and_initial_wealth(self):
        ix=pd.bdate_range("2020-01-01",periods=2)
        r=pd.DataFrame({"a":[.2,0.],"b":[0.,-.1]},index=ix)
        w=pd.DataFrame([[.5,.5],[.5,.5]],index=ix,columns=r.columns)
        s=simulate(r,w,.001)
        self.assertAlmostEqual(s.net.iloc[0],.0989)
        self.assertAlmostEqual(s.traded_notional.iloc[1],1/11)
        self.assertAlmostEqual(drawdown([-.10,.02]),-.10)

    def test_calendar_hac_matches_standard_on_contiguous_sample(self):
        rng=np.random.default_rng(SEED)
        ix=pd.bdate_range("2020-01-01",periods=80)
        x=pd.Series(rng.normal(size=len(ix)),index=ix)
        e=pd.Series(rng.normal(size=len(ix)),index=ix).rolling(4,min_periods=1).mean()
        y=.8*x+e
        np.testing.assert_allclose(fit(y,x,calendar=ix).bse,fit(y,x).bse,rtol=1e-12)


if __name__=="__main__":
    unittest.main()
