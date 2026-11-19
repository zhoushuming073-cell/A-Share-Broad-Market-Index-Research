from pathlib import Path
import hashlib
import yaml
import exchange_calendars as xc
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)
CFG=yaml.safe_load((ROOT/"config.yaml").read_text())
FILE=ROOT.parent/"four_pair_tests_v2/data/input_closes.csv"
P=pd.read_csv(FILE,index_col=0,parse_dates=True)[["000001","000300"]]
assert P.shape==(2884,2) and P.index.is_unique and np.isfinite(P).all().all() and (P>0).all().all()
np.testing.assert_array_equal(P.index,xc.get_calendar("XSHG").sessions_in_range(P.index[0],P.index[-1]).tz_localize(None))
LOG=np.log(P/P.shift()).iloc[1:]
A,B=LOG.to_numpy().T
D=B-A; N=len(D); T=LOG.index
def count(p):return int(np.floor(N*p+.5))
def save(df,name):
    pd.DataFrame(df).to_csv(OUT/(name+".csv"),index=False,encoding="utf-8-sig",float_format="%.15g")
def read(name):return pd.read_csv(OUT/(name+".csv"))
