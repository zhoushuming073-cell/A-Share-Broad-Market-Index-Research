"""Fetch A-share index price OHLCV data and build an auditable SQLite store."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw_index"
DB = DATA / "a_share_indices.sqlite"
INDICES = {
    "399311": ("国证1000", "CNI"),
    "000903": ("中证A100", "CSI"),
    "000132": ("上证100", "SSE/CSI"),
    "000300": ("沪深300", "CSI"),
    "000001": ("上证综合指数", "SSE"),
    "000985": ("中证全指", "CSI"),
}
COLUMNS = ["date", "open", "close", "high", "low", "volume", "turnover_value", "amplitude_pct", "change_pct", "change_points", "turnover_pct"]


def init(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS index_master (
      provider TEXT NOT NULL, index_code TEXT NOT NULL, index_name TEXT NOT NULL,
      publisher TEXT NOT NULL, return_type TEXT NOT NULL,
      PRIMARY KEY (provider,index_code)
    );
    CREATE TABLE IF NOT EXISTS ingest_batch (
      batch_id TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, source TEXT NOT NULL,
      api_function TEXT NOT NULL, akshare_version TEXT NOT NULL,
      request_start TEXT NOT NULL, request_end TEXT NOT NULL,
      raw_file TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
      row_count INTEGER NOT NULL, first_date TEXT, last_date TEXT
    );
    CREATE TABLE IF NOT EXISTS index_daily (
      provider TEXT NOT NULL, index_code TEXT NOT NULL, trade_date TEXT NOT NULL,
      open REAL, high REAL, low REAL, close REAL NOT NULL,
      volume REAL, turnover_value REAL, amplitude_pct REAL, change_pct_source REAL,
      change_points_source REAL, turnover_pct REAL, close_return REAL,
      batch_id TEXT NOT NULL,
      PRIMARY KEY(provider,index_code,trade_date),
      FOREIGN KEY(provider,index_code) REFERENCES index_master(provider,index_code),
      FOREIGN KEY(batch_id) REFERENCES ingest_batch(batch_id),
      CHECK(close > 0), CHECK(open IS NULL OR open > 0),
      CHECK(high IS NULL OR high > 0), CHECK(low IS NULL OR low > 0)
    );
    CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily(trade_date);
    """)


def fetch_one(conn: sqlite3.Connection, code: str, start: str, end: str) -> dict:
    name, publisher = INDICES[code]
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    if df.empty or len(df.columns) != len(COLUMNS):
        raise ValueError(f"{code}: empty or unexpected column count {list(df.columns)}")
    df.columns = COLUMNS  # AKShare Eastmoney API field order, recorded in source.
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError(f"{code}: invalid or duplicated dates")
    if (df["close"] <= 0).any() or df["close"].isna().any():
        raise ValueError(f"{code}: invalid close")
    df = df.sort_values("date").reset_index(drop=True)
    df["close_return"] = df["close"].pct_change(fill_method=None)
    # The first return is unknown because the preceding close was not requested.
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"eastmoney_index_{code}_{start}_{end}_{stamp}.csv"
    df.drop(columns=["close_return"]).to_csv(raw_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    batch_id = uuid.uuid4().hex
    import importlib.metadata
    with conn:
        conn.execute("INSERT OR REPLACE INTO index_master VALUES (?,?,?,?,?)", ("eastmoney", code, name, publisher, "price"))
        conn.execute("INSERT INTO ingest_batch VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            batch_id, fetched_at, "Eastmoney push2his.eastmoney.com", "akshare.index_zh_a_hist",
            importlib.metadata.version("akshare"), start, end, str(raw_path.relative_to(ROOT)), digest,
            len(df), df["date"].iloc[0], df["date"].iloc[-1],
        ))
        for row in df.itertuples(index=False):
            vals = [None if pd.isna(v) else v for v in row]
            conn.execute("""INSERT INTO index_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(provider,index_code,trade_date) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume,
                turnover_value=excluded.turnover_value, amplitude_pct=excluded.amplitude_pct,
                change_pct_source=excluded.change_pct_source,
                change_points_source=excluded.change_points_source,
                turnover_pct=excluded.turnover_pct, close_return=excluded.close_return,
                batch_id=excluded.batch_id""", (
                    "eastmoney", code, vals[0], vals[1], vals[3], vals[4], vals[2],
                    vals[5], vals[6], vals[7], vals[8], vals[9], vals[10], vals[11], batch_id,
                ))
    return {"code": code, "name": name, "rows": len(df), "first": df["date"].iloc[0], "last": df["date"].iloc[-1], "raw": str(raw_path.relative_to(ROOT)), "sha256": digest}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2026-11-17")
    args = p.parse_args()
    DATA.mkdir(exist_ok=True)
    out = []
    with sqlite3.connect(DB) as conn:
        init(conn)
        for code in INDICES:
            try:
                result = fetch_one(conn, code, args.start, args.end)
                out.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                out.append({"code": code, "error": repr(exc)})
                print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    (DATA / "ingest_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
