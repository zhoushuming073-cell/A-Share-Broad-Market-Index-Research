"""Download supplemental public inputs without changing the source database."""
import argparse
from concurrent.futures import ThreadPoolExecutor

import requests

from research_common import *

PRICE_BASE = "https://money-tree.eyunzhu.com"
FACTOR_BASE = "https://www.mingshiim.com/static"


def stages():
    with db() as conn:
        dates = conn.execute("SELECT DISTINCT as_of_date FROM constituent_snapshot "
                             "WHERE snapshot_kind='baostock_rebalance_observation' ORDER BY as_of_date").fetchall()
    close, _, _ = market()
    return {pd.Timestamp(s): close.index[close.index.get_loc(s)+1] for s, in dates}


def fetch(pair):
    name, url = pair
    file = INPUTS / name
    try:
        if not file.exists():
            r = requests.get(url, timeout=18)
            r.raise_for_status()
            if name.endswith(".csv"):
                if not r.content.startswith(b"symbol,trade_date,"):
                    raise ValueError("Unrecognized daily CSV format")
            elif not r.content.startswith(b"PK"):
                raise ValueError("Unrecognized factor Excel format")
            file.write_bytes(r.content)
        return {"file":name,"url":url,"status":"ok"}
    except (requests.RequestException, ValueError) as e:
        status = f"HTTP_{e.response.status_code}" if isinstance(e, requests.HTTPError) and e.response is not None else type(e).__name__
        return {"file":name,"url":url,"status":status}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--hs300-stages",action="store_true")
    args=p.parse_args()
    dates={"20261102","20261030"}
    if args.hs300_stages:
        dates.update(d.strftime("%Y%m%d") for d in stages().values())
    pairs=[(f"{model}_factors_daily_202608.xlsx", f"{FACTOR_BASE}/{model}_factors_daily_202608.xlsx")
           for model in ["CH3","CH4"]]
    pairs += [(f"{day}.csv",f"{PRICE_BASE}/daily_kline/date/{day}.csv") for day in sorted(dates)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pd.DataFrame(pool.map(fetch,pairs)).to_csv(LOGS/"input_download.csv",index=False)


if __name__=="__main__":
    main()
