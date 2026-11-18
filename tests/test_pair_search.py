import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from optimize_hedge_ratio import account, signals, specifications, weights
from research_common import np, pd


class PairSearchTests(unittest.TestCase):
    def test_roundtrip_with_drift_and_fees(self):
        # One initial long and short, followed by a final close without rebalancing.
        x=np.array([.2,-.1]); y=np.array([-.1,.1]); fee=.001
        z=account(x,y,np.array([.5,.5]),np.array([-.5,-.5]),np.ones(2),
                  np.array([True,False]),fee,0.)
        size=.5/(1+fee)
        profit=size*(1.2*.9-.9*1.1)
        total_volume=2*size+size*(1.2*.9+.9*1.1)
        self.assertAlmostEqual(z[-1,1],1+profit-fee*total_volume)
        self.assertAlmostEqual(z[:,2].sum(),total_volume)

    def test_signal_observation_precedes_entry_close(self):
        ix=pd.bdate_range("2010-01-01",periods=650)
        rng=np.random.default_rng(13)
        r=pd.DataFrame(rng.normal(0,.01,(len(ix),2)),index=ix,columns=["a","b"])
        r["a"]+=r["b"]
        def get(data):
            p=(1+data).cumprod()
            return next(row[-2] for row in specifications(p,data,"a","b",ix[:400],None)
                        if row[0]=="rolling_60_D")
        old=get(r)
        r.iloc[539:]+=rng.normal(0,.1,(len(r)-539,2))
        new=get(r)
        np.testing.assert_allclose(new[:541],old[:541],equal_nan=True)
        self.assertNotAlmostEqual(old[541],new[541])

    def test_unit_exposure_and_flat_when_disallowed(self):
        w,v=weights(np.array([.7,1.2,np.nan,-.5]),np.array([1.,-1.,1.,1.]))
        np.testing.assert_allclose(abs(w)+abs(v),[1.,1.,0.,0.])
        z=np.array([2.2,1.1,.2,.1,-2.2,-4.2])
        out=signals(z,np.ones(6,dtype=bool),np.zeros(6,dtype=bool),2.,.5)
        np.testing.assert_array_equal(out,[-1,-1,0,0,1,0])


if __name__=="__main__":
    unittest.main()
