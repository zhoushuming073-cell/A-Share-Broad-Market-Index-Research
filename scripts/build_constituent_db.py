"""Archive dated official index membership files; never backfill an old date from a new file."""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw_constituents"
DB = DATA / "a_share_indices.sqlite"
CSI = ["000903", "000132", "000300", "000001", "000985"]
BASE = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile"


def download(url: str, name: str) -> tuple[bytes, str, str]:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    if not (r.content[:4] == b"PK\x03\x04" or r.content[:4] == b"\xd0\xcf\x11\xe0"):
        raise ValueError(f"not an Excel file: {url}, {r.content[:80]!r}")
    RAW.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(r.content).hexdigest()
    path = RAW / f"{name}_{digest[:12]}.xls"
    if not path.exists():
        path.write_bytes(r.content)
    return r.content, str(path.relative_to(ROOT)), digest


def init(c: sqlite3.Connection) -> None:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS constituent_source (
      source_id TEXT PRIMARY KEY, index_code TEXT NOT NULL, source_url TEXT NOT NULL,
      fetched_at TEXT NOT NULL, file_path TEXT NOT NULL, sha256 TEXT NOT NULL,
      file_kind TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS constituent_snapshot (
      index_code TEXT NOT NULL, as_of_date TEXT NOT NULL, security_code TEXT NOT NULL,
      security_name TEXT, exchange_name TEXT, industry TEXT,
      weight_pct REAL, snapshot_kind TEXT NOT NULL, complete INTEGER NOT NULL,
      source_id TEXT NOT NULL,
      PRIMARY KEY(index_code,as_of_date,security_code,snapshot_kind,source_id),
      FOREIGN KEY(source_id) REFERENCES constituent_source(source_id)
    );
    CREATE INDEX IF NOT EXISTS idx_constituent_date ON constituent_snapshot(index_code,as_of_date);
    CREATE TABLE IF NOT EXISTS constituent_adjustment_record (
      index_code TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT,
      security_code TEXT NOT NULL, security_name TEXT, industry TEXT,
      adjustment_type TEXT NOT NULL, source_id TEXT NOT NULL,
      PRIMARY KEY(index_code,start_date,security_code,adjustment_type,source_id),
      FOREIGN KEY(source_id) REFERENCES constituent_source(source_id)
    );
    """)


def source(c, code, url, raw_path, digest, kind):
    sid = hashlib.sha256((code + url + digest).encode()).hexdigest()[:32]
    c.execute("INSERT OR IGNORE INTO constituent_source VALUES (?,?,?,?,?,?,?)",
              (sid, code, url, datetime.now().astimezone().isoformat(timespec="seconds"), raw_path, digest, kind))
    return sid


def date_str(v) -> str:
    if pd.isna(v):
        raise ValueError("missing as-of date")
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def cni(c):
    code = "399311"
    url = f"http://www.cnindex.com.cn/sample-detail/download-adjustment?indexcode={code}"
    content, path, digest = download(url, f"cni_{code}_adjustment")
    df = pd.read_excel(io.BytesIO(content), dtype={2: str})
    if len(df.columns) != 6 or df.empty:
        raise ValueError(f"CNI unexpected adjustment shape {df.shape}")
    df.columns = ["start", "end", "security_code", "name", "industry", "kind"]
    sid = source(c, code, url, path, digest, "official_adjustment_history")
    results = []
    for start, g in df.groupby("start", sort=True):
        start = date_str(start)
        # CNI reports OLD retained members, + incoming, - outgoing, and optional reserves.
        included = g[g["kind"].isin(["OLD", "+"])]
        n = included["security_code"].nunique()
        if not 950 <= n <= 1001 or included["security_code"].duplicated().any():
            raise ValueError(f"CNI {start} unexpected member count {n}")
        for row in g.itertuples(index=False):
            code6 = str(row.security_code).split(".")[0].zfill(6)
            c.execute("INSERT OR IGNORE INTO constituent_adjustment_record VALUES (?,?,?,?,?,?,?,?)",
                      (code, start, date_str(row.end) if pd.notna(row.end) else None,
                       code6, row.name, row.industry, str(row.kind), sid))
        for row in included.itertuples(index=False):
            c.execute("INSERT OR IGNORE INTO constituent_snapshot VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (code, start, str(row.security_code).split(".")[0].zfill(6), row.name,
                       None, row.industry, None, "cni_adjustment_stage", 1, sid))
        results.append({"date": start, "members": n})
    return {"index": code, "kind": "historical_stages", "stages": len(results),
            "first": results[0], "last": results[-1], "raw": path}


def cni_weight(c):
    code = "399311"
    url = f"https://www.cnindex.com.cn/sample-detail/download-history?indexcode={code}"
    content, path, digest = download(url, f"cni_{code}_weight")
    df = pd.read_excel(io.BytesIO(content), dtype={1: str})
    if len(df.columns) != 6 or df.empty:
        raise ValueError(f"CNI unexpected weight shape {df.shape}")
    df.columns = ["date", "security_code", "name", "industry", "market_cap", "weight_pct"]
    sid = source(c, code, url, path, digest, "official_dated_weight_snapshot")
    dates = set()
    for row in df.itertuples(index=False):
        day = date_str(row.date)
        dates.add(day)
        c.execute("INSERT OR IGNORE INTO constituent_snapshot VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (code, day, str(row.security_code).split(".")[0].zfill(6), row.name,
                   None, row.industry, float(row.weight_pct), "cni_weight_snapshot", 1, sid))
    return {"index": code, "kind": "weight_snapshot", "dates": sorted(dates), "members": len(df),
            "weight_sum_pct": round(pd.to_numeric(df.weight_pct).sum(), 4), "raw": path}


def csi(c, code, kind):
    subdir = "cons" if kind == "membership" else "closeweight"
    suffix = "cons" if kind == "membership" else "closeweight"
    url = f"{BASE}/{subdir}/{code}{suffix}.xls"
    content, path, digest = download(url, f"csi_{code}_{kind}")
    df = pd.read_excel(io.BytesIO(content), dtype={1: str, 4: str})
    expected = 9 if kind == "membership" else 10
    if len(df.columns) != expected or df.empty:
        raise ValueError(f"CSI {code} unexpected {kind} shape {df.shape}")
    sid = source(c, code, url, path, digest, f"official_{kind}_snapshot")
    dates = set()
    codes = set()
    weights = []
    for row in df.itertuples(index=False, name=None):
        day = date_str(str(row[0])) if not isinstance(row[0], (int, float)) else date_str(str(int(row[0])))
        security = str(row[4]).split(".")[0].zfill(6)
        dates.add(day)
        codes.add(security)
        w = float(row[9]) if kind == "weight" and pd.notna(row[9]) else None
        if w is not None:
            weights.append(w)
        c.execute("INSERT OR IGNORE INTO constituent_snapshot VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (code, day, security, row[5], row[7], None, w,
                   f"csi_{kind}_snapshot", 1, sid))
    if len(dates) != 1 or len(codes) != len(df):
        raise ValueError(f"CSI {code} {kind}: mixed dates or duplicate codes")
    return {"index": code, "kind": kind, "date": next(iter(dates)),
            "members": len(df), "weight_sum_pct": round(sum(weights), 4) if weights else None,
            "raw": path}


def main():
    DATA.mkdir(exist_ok=True)
    results = []
    with sqlite3.connect(DB) as c:
        init(c)
        for label, fn in [("cni_history", lambda: cni(c)), ("cni_weight", lambda: cni_weight(c))] + [
            (f"csi_{code}_{kind}", lambda code=code, kind=kind: csi(c, code, kind))
            for code in CSI for kind in ["membership", "weight"]
        ]:
            try:
                with c:
                    result = fn()
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                results.append({"task": label, "error": repr(exc)})
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    (DATA / "constituent_ingest_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
