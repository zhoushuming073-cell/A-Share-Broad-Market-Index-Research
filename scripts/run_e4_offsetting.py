from fetch_research_inputs import stages
from research_common import *

PREV = "2026-10-30"
DAY = "2026-11-02"
STAGE = "baostock_rebalance_observation"


def offset_metrics(w, r):
    w, r = np.asarray(w,float), np.asarray(r,float)
    if w.ndim!=1 or w.shape!=r.shape or len(w)==0 or not np.isfinite(w).all() or not np.isfinite(r).all() or (w<0).any():
        raise ValueError("Need aligned finite returns and nonnegative weights")
    contributions=w*r
    positive=np.maximum(contributions,0).sum()
    negative=np.maximum(-contributions,0).sum()
    gross=positive+negative
    net=positive-negative
    total=2*min(positive,negative)
    return dict(positive=positive,negative_abs=negative,gross=gross,net=net,
                total_offset=total,co=total/gross if gross>0 else np.nan)


def dispersion(r, benchmark):
    return dict(cssd_about_benchmark=np.sqrt(((r-benchmark)**2).sum()/(len(r)-1)) if len(r)>1 else np.nan,
                csad_about_benchmark=(r-benchmark).abs().mean(),
                cross_section_sd=r.std(), mean_abs_dev=(r-r.mean()).abs().mean(),
                up=(r>0).mean(),down=(r<0).mean(),flat=(r==0).mean())


def quotes(day):
    file=INPUTS/f"{pd.Timestamp(day):%Y%m%d}.csv"
    x=pd.read_csv(file,dtype={"symbol":str,"trade_date":str})
    if x.symbol.duplicated().any() or not x.trade_date.eq(f"{pd.Timestamp(day):%Y%m%d}").all():
        raise ValueError(f"Duplicate symbol or wrong trading date: {file.name}")
    return x[x.close.gt(0)&x.pre_close.gt(0)].copy()


def main():
    _,r,_=market()
    official=pd.read_csv(INPUTS/"official_index_snapshot.csv",dtype={"index_code":str})
    official=official.pivot(index="date",columns="index_code",values="close")
    official_returns=official.loc[DAY]/official.loc[PREV]-1
    with db() as c:
        s=pd.read_sql_query("SELECT * FROM constituent_snapshot WHERE as_of_date=? AND weight_pct IS NOT NULL",c,params=[PREV])
        historical=pd.read_sql_query("SELECT * FROM constituent_snapshot WHERE snapshot_kind=?",c,params=[STAGE])
    s["ex"]=s.exchange_name.map({"上海证券交易所":"SH","深圳证券交易所":"SZ","北京证券交易所":"BJ"})
    q,p=quotes(DAY),quotes(PREV)
    if p.symbol.str[:6].duplicated().any():
        raise ValueError("Ambiguous security codes in price snapshots")
    p["security_code"]=p.symbol.str[:6]
    # Match by reported venue when available. For CNI the code must uniquely identify a quote.
    s=s.merge(p[["security_code","symbol","close"]].rename(columns={"close":"prior_close"}),on="security_code",how="left",validate="many_to_one")
    s["symbol"]=s.symbol.where(s.ex.isna(),s.security_code+"."+s.ex)
    s=s.merge(q[["symbol","close","pre_close","pct_chg"]],on="symbol",how="left",validate="many_to_one")
    if s.duplicated(["index_code","security_code"]).any():
        raise ValueError("Multiple competing weight snapshots")
    s["raw_return"]=s.close/s.prior_close-1
    s["ref_return"]=s.close/s.pre_close-1
    s["contribution"]=s.weight_pct/100*s.raw_return
    s["reference_changed"]=(s.pre_close-s.prior_close).abs()>.005
    s["trade_date"]=DAY
    rows=[]
    for index,g in s.groupby("index_code"):
        reported=official_returns[index]
        for key in ["raw_return","ref_return"]:
            valid=g.dropna(subset=[key,"weight_pct"])
            if valid.empty:
                continue
            m=offset_metrics(valid.weight_pct/100,valid[key])
            rows.append(dict(index=index,weight_date=PREV,date=DAY,return_basis=key,n_file=len(g),n_used=len(valid),
                             weight_sum_pct=g.weight_pct.sum(),weight_used_pct=valid.weight_pct.sum(),
                             missing_weight_pct=g.loc[~g.index.isin(valid.index),"weight_pct"].sum(),
                             reference_changed=int(valid.reference_changed.sum()), vendor_delta=(valid.ref_return-valid.pct_chg/100).abs().max(),
                             index_reference_return=reported,eastmoney_return=r.at[pd.Timestamp(DAY),index],
                             reconstructed_minus_reference=m["net"]-reported,
                             **m,**dispersion(valid[key],reported)))
    csv(s[["index_code","as_of_date","trade_date","security_code","symbol","industry","weight_pct","prior_close",
           "close","pre_close","reference_changed","raw_return","ref_return","contribution","source_id"]],"e4_snapshot_contributions")
    out=pd.DataFrame(rows)
    csv(out,"e4_snapshot_metrics")
    stage=[]
    for obs, day in stages().items():
        g=historical[historical.as_of_date.eq(obs.strftime("%Y-%m-%d"))].copy()
        try:
            q=quotes(day).set_index("symbol")
        except FileNotFoundError:
            stage.append(dict(observation_date=obs,date=day,n_file=len(g),n_used=0,status="source_unavailable"))
            continue
        tickers=g.security_code+"."+g.exchange_name.str.upper()
        d=q.reindex(tickers)
        rr=(d.close/d.pre_close-1).dropna()
        stage.append(dict(observation_date=obs,date=day,n_file=len(g),n_used=len(rr),status="available_quotes",
                          **dispersion(rr,r.at[day,"000300"])))
    csv(pd.DataFrame(stage),"e4_hs300_stage_dispersion")
    plot=out[(out.return_basis=="raw_return")&out["index"].isin(TARGETS)].set_index("index").loc[TARGETS]
    fig,ax=plt.subplots(figsize=(8,4))
    (plot[["gross","total_offset"]].rename(index=NAMES)*10000).plot.bar(ax=ax,rot=0)
    ax.set_ylabel("Contribution (bp)")
    ax.set_xlabel("")
    ax.set_title("2026-11-02 · matched securities only")
    figure(fig,"e4_snapshot")


if __name__=="__main__":
    main()
