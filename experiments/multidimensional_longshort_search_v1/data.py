from pathlib import Path
import hashlib
import yaml
import numpy as np
import pandas as pd
import exchange_calendars as xc

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
OUT.mkdir(exist_ok=True)
CFG=yaml.safe_load((ROOT/"config.yaml").read_text())
FILE=ROOT.parent/"four_pair_tests_v2/data/input_closes.csv"
P=pd.read_csv(FILE,index_col=0,parse_dates=True)[["000001","000300"]]
assert len(P)==2884 and P.index.is_unique and (P>0).all().all()
np.testing.assert_array_equal(P.index,xc.get_calendar("XSHG").sessions_in_range(P.index[0],P.index[-1]).tz_localize(None))
R=P.pct_change(fill_method=None).iloc[1:]
X,Y=R.to_numpy().T
DATES=R.index
GAPS=np.diff(P.index.values).astype("timedelta64[D]").astype(float)
MONTH=DATES.to_period("M").astype(int).to_numpy()
YEAR=DATES.year.to_numpy()
def save(rows,name):
    pd.DataFrame(rows).to_csv(OUT/(name+".csv"),index=False,encoding="utf-8-sig")
