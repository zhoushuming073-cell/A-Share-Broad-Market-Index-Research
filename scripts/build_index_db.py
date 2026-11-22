"""Fetch A-share index price OHLCV data and build an auditable SQLite store."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd

from index_pool import INDICES, PROVIDERS, SINA_SYMBOLS


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw_index"
DB = DATA / "a_share_indices.sqlite"
COLUMNS = ["date", "open", "close", "high", "low", "volume", "turnover_value", "amplitude_pct", "change_pct", "change_points", "turnover_pct"]
OHLCV = ["date", "open", "high", "low", "close", "volume"]


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


def fetch_daily(code: str, start: str, end: str) -> pd.DataFrame:
    if code in SINA_SYMBOLS:
        return ak.stock_zh_index_daily(symbol=SINA_SYMBOLS[code]).loc[:, OHLCV]
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    if len(df.columns) != len(COLUMNS):
        raise ValueError(f"{code}: empty or unexpected column count {list(df.columns)}")
    df.columns = COLUMNS
    return df


def prepare_daily(df: pd.DataFrame, code: str, start: str, end: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError(f"{code}: invalid or duplicated dates")
    df = df.loc[df["date"].between(start, end)]
    if df.empty:
        raise ValueError(f"{code}: no data in requested period")
    if (df["close"] <= 0).any() or df["close"].isna().any():
        raise ValueError(f"{code}: invalid close")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any() or (
        df["low"] > df[["open", "close"]].min(axis=1)
    ).any():
        raise ValueError(f"{code}: inconsistent OHLC")
    return df.sort_values("date").reset_index(drop=True)


def store_daily(conn: sqlite3.Connection, code: str, start: str, end: str, df: pd.DataFrame) -> dict:
    name, publisher = INDICES[code]
    provider = PROVIDERS[code]
    source, api = ("Sina Finance finance.sina.com.cn", "akshare.stock_zh_index_daily") if provider == "sina" else (
        "Eastmoney push2his.eastmoney.com", "akshare.index_zh_a_hist"
    )
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{provider}_index_{code}_{start}_{end}_{stamp}.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    batch_id = uuid.uuid4().hex
    stored = df.reindex(columns=COLUMNS)
    stored["close_return"] = stored["close"].pct_change(fill_method=None)
    with conn:
        conn.execute("INSERT OR REPLACE INTO index_master VALUES (?,?,?,?,?)", (provider, code, name, publisher, "price"))
        conn.execute("INSERT INTO ingest_batch VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            batch_id, fetched_at, source, api,
            version("akshare"), start, end, raw_path.relative_to(ROOT).as_posix(), digest,
            len(df), df["date"].iloc[0], df["date"].iloc[-1],
        ))
        for row in stored.itertuples(index=False):
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
                    provider, code, vals[0], vals[1], vals[3], vals[4], vals[2],
                    vals[5], vals[6], vals[7], vals[8], vals[9], vals[10], vals[11], batch_id,
                ))
        # Recompute across batch boundaries after a partial or overlapping refresh.
        conn.execute("""
            WITH recalculated AS (
                SELECT trade_date, close / LAG(close) OVER (ORDER BY trade_date) - 1 AS daily_return
                FROM index_daily WHERE provider = ? AND index_code = ?
            )
            UPDATE index_daily SET close_return = recalculated.daily_return
            FROM recalculated WHERE provider = ? AND index_code = ?
              AND index_daily.trade_date = recalculated.trade_date
        """, (provider, code, provider, code))
    return {"code": code, "name": name, "provider": provider, "api": api, "rows": len(df),
            "first": df["date"].iloc[0], "last": df["date"].iloc[-1],
            "raw": raw_path.relative_to(ROOT).as_posix(), "sha256": digest}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2026-11-17")
    p.add_argument("--codes", nargs="+", choices=INDICES, default=list(INDICES))
    p.add_argument("--report", type=Path, default=DATA / "ingest_report_nine.json")
    args = p.parse_args()
    DATA.mkdir(exist_ok=True)
    out = []
    with sqlite3.connect(DB) as conn:
        init(conn)
        for code in dict.fromkeys(args.codes):
            try:
                df = prepare_daily(fetch_daily(code, args.start, args.end), code, args.start, args.end)
                result = store_daily(conn, code, args.start, args.end, df)
                out.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                out.append({"code": code, "error": repr(exc)})
                print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    args.report.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any("error" in row for row in out):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
