from itertools import combinations
from pathlib import Path
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
CODES = tuple(CFG["codes"])
PAIRS = tuple(combinations(range(len(CODES)), 2))
PAIR_IDX = np.asarray(PAIRS)
LABELS = tuple(f"{CODES[a]}_{CODES[b]}" for a, b in PAIRS)
NAMES = {code: INDICES[code][0] for code in CODES}
NAMES["000001"] = "上证综指"


def load(start, sample):
    calendar = xc.get_calendar("XSHG").sessions_in_range(start, CFG["end"]).tz_localize(None)
    panel = pd.DataFrame(index=calendar)
    records = []
    extremes = []
    uri = (ROOT / "data/a_share_indices.sqlite").as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        for code in CODES:
            query = """
                SELECT d.trade_date, d.close, m.return_type, b.raw_file, b.raw_sha256
                FROM index_daily d JOIN index_master m USING(provider,index_code)
                JOIN ingest_batch b USING(batch_id)
                WHERE d.index_code=? AND d.provider=? AND trade_date BETWEEN ? AND ?
                ORDER BY d.trade_date
            """
            rows = pd.read_sql_query(query, db, params=(code, PROVIDERS[code], start, CFG["end"]),
                                     parse_dates=["trade_date"]).set_index("trade_date")
            if not rows.index.equals(calendar) or not rows.return_type.eq("price").all():
                raise ValueError(f"{code} does not match expected trading dates or price-index basis")
            if not np.isfinite(rows.close).all() or not rows.close.gt(0).all():
                raise ValueError(f"{code} has invalid closes")
            for path, digest in rows[["raw_file", "raw_sha256"]].drop_duplicates().itertuples(index=False):
                path = ROOT / path.replace("\\", "/")
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Source hash differs for {path}")
                raw = pd.read_csv(path, parse_dates=["date"]).set_index("date", verify_integrity=True)
                matching = rows[rows.raw_sha256.eq(digest)]
                np.testing.assert_allclose(raw.loc[matching.index, "close"], matching.close, rtol=0, atol=1e-8)
            panel[code] = rows.close
            daily = np.log(rows.close).diff().iloc[1:]
            flagged = daily[daily.abs().gt(.1)]
            extremes.extend(dict(sample=sample, code=code, date=date, log_return=value)
                            for date, value in flagged.items())
            records.append(dict(sample=sample, code=code, name=NAMES[code], provider=PROVIDERS[code],
                                return_type="price", first_close=calendar[0].date(), last_close=calendar[-1].date(),
                                closes=len(calendar), returns=len(daily), duplicate_dates=0, missing_dates=0,
                                max_abs_log_return=daily.abs().max(), max_date=daily.abs().idxmax().date(),
                                abs_log_gt_10pct=len(flagged), raw_sha256="|".join(rows.raw_sha256.unique())))
    panel.to_csv(OUT / f"input_closes_{sample}.csv", index_label="date", float_format="%.10f")
    return np.log(panel).diff().iloc[1:], records, extremes
