from pathlib import Path
from itertools import combinations
import hashlib
import sqlite3
import sys

import exchange_calendars as xc
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from index_pool import INDICES, PROVIDERS

CFG = yaml.safe_load((HERE / "config.yaml").read_text())
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
CODES = CFG["codes"]
PAIRS = list(combinations(CODES, 2))
LABELS = [i + "_" + j for i, j in PAIRS]
METRICS = ["kstar","C1","C5","C10","C20","HHI","N_eff","offset","abs_D"]
RELEASE = dict(zip(["000001","000985","000300","000903","000132","399311","000905","399006","000688"],
                   ["1991-07-15","2011-08-02","2005-04-08","2006-05-29","2012-04-20",
                    "2005-02-03","2007-01-15","2010-06-01","2020-07-23"]))

def audit():
    calendar = xc.get_calendar("XSHG").sessions_in_range(CFG["start"], CFG["end"]).tz_localize(None)
    panel = pd.DataFrame(index=calendar)
    records = []
    with sqlite3.connect((ROOT / "data/a_share_indices.sqlite").as_uri()+"?mode=ro", uri=True) as db:
        for code in CODES:
            q = pd.read_sql_query("""SELECT d.trade_date,d.close,m.return_type,b.raw_file,b.raw_sha256
                FROM index_daily d JOIN index_master m USING(provider,index_code)
                JOIN ingest_batch b USING(batch_id) WHERE d.index_code=? AND d.provider=?
                AND trade_date BETWEEN ? AND ? ORDER BY trade_date""", db,
                params=(code,PROVIDERS[code],CFG["start"],CFG["end"]),parse_dates=["trade_date"]).set_index("trade_date")
            assert q.index.equals(calendar) and q.return_type.eq("price").all()
            assert np.isfinite(q.close).all() and q.close.gt(0).all()
            for path, digest in q[["raw_file","raw_sha256"]].drop_duplicates().itertuples(index=False):
                path = ROOT / path.replace("\\","/")
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                raw = pd.read_csv(path,index_col="date",parse_dates=True)
                sub = q[q.raw_sha256.eq(digest)]
                np.testing.assert_allclose(raw.close.loc[sub.index],sub.close,rtol=0,atol=1e-8)
            panel[code] = q.close
            r = np.log(q.close).diff()
            assert q.index[0] >= pd.Timestamp(RELEASE[code])
            source = ("https://www.cnindex.com.cn/html2pdf/preview/jj_399311.pdf" if code=="399311" else
                      "https://www.szse.cn/disclosure/notice/general/t20100531_500454.html" if code=="399006"
                      else f"https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/{code}factsheet.pdf")
            records.append(dict(code=code,name=INDICES[code][0],provider=PROVIDERS[code],type="price",
                release_date=RELEASE[code],release_source=source,
                start=q.index[0].date(),end=q.index[-1].date(),closes=len(q),missing=q.close.isna().sum(),
                max_abs_log_return=r.abs().max(),max_abs_date=r.abs().idxmax().date(),
                abs_log_gt_10pct=r.abs().gt(.10).sum(),raw_sha256="|".join(q.raw_sha256.unique())))
    pd.DataFrame(records).to_csv(OUT/"data_audit.csv",index=False)
    panel.to_csv(OUT/"input_closes.csv",index_label="date",float_format="%.10f")
    return np.log(panel).diff().iloc[1:]

def spreads(r):
    return np.ascontiguousarray([r[j].values-r[i].values for i,j in PAIRS])
