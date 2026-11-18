"""Collect CSI 300 membership snapshots after each June/December regular rebalance."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import baostock as bs

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "a_share_indices.sqlite"
RAW = ROOT / "data" / "raw_constituents"


def stage_dates(conn):
    dates = {r[0] for r in conn.execute("SELECT DISTINCT trade_date FROM index_daily WHERE index_code='000300'")}
    yield '2015-01-05'
    for year in range(2015, 2026):
        for month in (6, 12):
            first = date(year, month, 1)
            first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
            second_friday = first_friday + timedelta(days=7)
            after = sorted(x for x in dates if x > second_friday.isoformat())
            if after:
                yield after[0]


def main():
    RAW.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB)
    result = []
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(f"BaoStock login: {login.error_msg}")
    try:
        for day in stage_dates(conn):
            if conn.execute("SELECT 1 FROM constituent_snapshot WHERE index_code='000300' AND as_of_date=? AND snapshot_kind='baostock_rebalance_observation' LIMIT 1", (day,)).fetchone():
                continue
            try:
                q = bs.query_hs300_stocks(day)
                if q.error_code != '0':
                    raise RuntimeError(q.error_msg)
                rows = []
                while q.next():
                    rows.append(q.get_row_data())
                codes = [r[1] for r in rows]
                if len(rows) != 300 or len(set(codes)) != 300:
                    raise ValueError(f"unexpected members: {len(rows)}")
                path = RAW / f"baostock_hs300_{day}.csv"
                with path.open('w', newline='', encoding='utf-8-sig') as f:
                    w = csv.writer(f)
                    w.writerow(['query_date','update_date','security_code','security_name'])
                    for update, security, name in rows:
                        w.writerow([day, update, security, name])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                sid = hashlib.sha256((day + digest).encode()).hexdigest()[:32]
                with conn:
                    conn.execute("INSERT OR IGNORE INTO constituent_source VALUES (?,?,?,?,?,?,?)",
                                 (sid, '000300', 'baostock.query_hs300_stocks',
                                  __import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds'),
                                  str(path.relative_to(ROOT)), digest, 'historical_membership_snapshot'))
                    for update, security, name in rows:
                        conn.execute("INSERT OR IGNORE INTO constituent_snapshot VALUES (?,?,?,?,?,?,?,?,?,?)",
                                     ('000300', day, security.split('.')[1], name, security.split('.')[0],
                                      None, None, 'baostock_rebalance_observation', 1, sid))
                item = {'date': day, 'members': len(rows), 'updates': sorted(set(r[0] for r in rows))[:3]}
                result.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
            except Exception as exc:
                item = {'date': day, 'error': repr(exc)}
                result.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
            time.sleep(0.3)
    finally:
        bs.logout()
        conn.close()
    (ROOT / 'data' / 'hs300_stage_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
