import hashlib
import importlib.metadata
import json

from research_common import *
from run_e4_offsetting import quotes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks, schemas, lineage = {}, [], []
    with db() as c:
        checks["integrity"] = c.execute("PRAGMA integrity_check").fetchone()[0]
        checks["foreign_key_errors"] = len(c.execute("PRAGMA foreign_key_check").fetchall())
        names = c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        for table, in names:
            fields = c.execute(f"PRAGMA table_info([{table}])").fetchall()
            n = c.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            for _, field, declared, required, _, pk in fields:
                summary = c.execute(f"SELECT typeof([{field}]), COUNT(*) FROM [{table}] GROUP BY 1").fetchall()
                start,end=(None,None)
                if "date" in field or field.endswith("_at") or field in ("request_start","request_end"):
                    start,end=c.execute(f"SELECT MIN([{field}]), MAX([{field}]) FROM [{table}]").fetchone()
                schemas.append(dict(table=table,rows=n,field=field,declared=declared,required=required,pk_order=pk,
                                    observed_types=json.dumps(dict(summary)),nulls=dict(summary).get("null",0),first=start,last=end))
        sql = "SELECT raw_file,raw_sha256,source FROM ingest_batch UNION ALL SELECT file_path,sha256,source_url FROM constituent_source"
        for file,expected,url in c.execute(sql):
            p=ROOT/file
            digest=sha(p) if p.is_file() else None
            lineage.append(dict(file=file,expected_sha256=expected,actual_sha256=digest,
                                checksum_match=(digest==expected),source=url))
        s=pd.read_sql_query("SELECT * FROM constituent_snapshot",c)
        q=pd.read_sql_query("SELECT * FROM index_daily",c)
        checks["nonpositive_ohlc"]=int((q[["open","high","low","close"]]<=0).any(axis=1).sum())
        checks["invalid_ohlc"]=int(((q.high<q[["open","close"]].max(axis=1))|(q.low>q[["open","close"]].min(axis=1))).sum())
        checks["duplicate_index_days"]=int(q.duplicated(["provider","index_code","trade_date"]).sum())
        checks["duplicate_members"]=int(s.duplicated(["index_code","as_of_date","snapshot_kind","source_id","security_code"]).sum())
        checks["invalid_security_code"]=int((~s.security_code.str.fullmatch(r"\d{6}")).sum())
        checks["nonpositive_snapshot_weights"]=int(s.weight_pct.lt(0).sum())
    close, ret, _ = market()
    stored = q.pivot(index="trade_date",columns="index_code",values="close_return")
    stored.index=pd.to_datetime(stored.index)
    checks["stored_vs_recomputed_return_max_delta"]=float((stored-ret).abs().max().max())
    checks["common_price_days"]=len(close)
    checks["common_returns"]=len(ret)
    checks["source_checksum_failures"]=sum(not x["checksum_match"] for x in lineage)
    for item in (x for x in lineage if not x["checksum_match"] and "000001_membership" in x["file"]):
        book=pd.read_excel(ROOT/item["file"],dtype={4:str})
        workbook={(str(x[0]),str(x[4]).zfill(6),x[5],x[7]) for x in book.itertuples(index=False,name=None)}
        rows=s[s.index_code.eq("000001") & s.snapshot_kind.eq("csi_membership_snapshot")]
        stored_keys={(x.as_of_date.replace("-",""),x.security_code,x.security_name,x.exchange_name) for x in rows.itertuples()}
        checks["sse_membership_row_reconciliation"]={"workbook_rows":len(workbook),"database_rows":len(stored_keys),
                                                      "workbook_only":len(workbook-stored_keys),"database_only":len(stored_keys-workbook)}
    checks["database_sha256"]=sha(DB)
    checks["dependencies"]={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","matplotlib","statsmodels","openpyxl","requests"]}
    checks["seed"]=SEED
    downloads=pd.read_csv(LOGS/"input_download.csv")
    existing={x["file"].replace("\\","/") for x in lineage}
    raw_files=[p for d in ["raw_index","raw_constituents"] for p in (ROOT/"data"/d).glob("*") if p.is_file()]
    checks["unreferenced_raw_files"]=[p.relative_to(ROOT).as_posix() for p in raw_files if p.relative_to(ROOT).as_posix() not in existing]
    for row in downloads.itertuples(index=False):
        path=INPUTS/row.file
        if path.is_file():
            lineage.append(dict(file=path.relative_to(ROOT).as_posix(),expected_sha256=None,
                                actual_sha256=sha(path),checksum_match=None,source=row.url))
    official=INPUTS/"official_index_snapshot.csv"
    lineage.append(dict(file=official.relative_to(ROOT).as_posix(),actual_sha256=sha(official),
                        source="CNI/CSI via AKShare index_hist_cni/stock_zh_index_hist_csindex"))
    rows=[]
    for code,g in q.groupby("index_code"):
        rows.append(dict(dataset="index_price",index_code=code,source="Eastmoney / AKShare",
                         start=g.trade_date.min(),end=g.trade_date.max(),dates=g.trade_date.nunique(),
                         rows=len(g),usable_rows=g.close_return.notna().sum(),status="available"))
    for code,g in pd.read_csv(official,dtype={"index_code":str}).groupby("index_code"):
        rows.append(dict(dataset="official_index_price_check",index_code=code,source=g.source.iloc[0],
                         start=g.date.min(),end=g.date.max(),dates=len(g),rows=len(g),usable_rows=1,status="available"))
    for (code,kind),g in s.groupby(["index_code","snapshot_kind"]):
        rows.append(dict(dataset=kind,index_code=code,source="constituent_source",start=g.as_of_date.min(),
                         end=g.as_of_date.max(),dates=g.as_of_date.nunique(),rows=len(g),
                         usable_rows=len(g),weight_pct=g.weight_pct.sum(min_count=1),status="snapshot"))
    for kind, g in s[s.industry.notna()].groupby("index_code"):
        rows.append(dict(dataset="industry_labels",index_code=kind,source="CNI weight workbook",
                         start=g.as_of_date.min(),end=g.as_of_date.max(),dates=g.as_of_date.nunique(),
                         rows=len(g),usable_rows=len(g),status="snapshot"))
    for model in ("CH3","CH4"):
        f=factors(model)
        rows.append(dict(dataset=model,index_code="",source="Mingshi",**window(f.index),
                         dates=len(f),rows=len(f),usable_rows=len(f.index.intersection(ret.index)),status="available"))
    for row in downloads[downloads.file.str.endswith(".csv")].itertuples(index=False):
        day=pd.to_datetime(row.file[:8],format="%Y%m%d").strftime("%Y-%m-%d")
        n=len(quotes(day)) if row.status=="ok" else 0
        rows.append(dict(dataset="individual_stock_price",index_code="",source=row.url,
                         start=day if n else None,end=day if n else None,requested_date=day,
                         dates=int(bool(n)),rows=n,usable_rows=n,status=row.status))
    for kind in ("continuous_membership","continuous_daily_weights","total_return_indices",
                 "etf_prices","corporate_actions","historical_industry","adjusted_equity_history"):
        rows.append(dict(dataset=kind,index_code="",dates=0,rows=0,usable_rows=0,status="not_in_project"))
    csv(pd.DataFrame(rows).drop(columns="n"),"data_coverage")
    csv(pd.DataFrame(schemas),"data_schema")
    csv(pd.DataFrame(lineage),"data_lineage")
    (LOGS/"data_quality.json").write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":
    main()
