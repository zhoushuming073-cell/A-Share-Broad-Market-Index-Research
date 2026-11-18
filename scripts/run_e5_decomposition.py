from research_common import *


def decompose(contribution, industry):
    c=pd.Series(np.asarray(contribution,float))
    groups=pd.Series(np.asarray(industry))
    if len(c)==0 or not np.isfinite(c).all() or groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError("Valid named industry required for every contribution")
    net=c.groupby(groups).sum()
    gross=c.abs().sum()
    group_gross=net.abs().sum()
    abs_net=abs(c.sum())
    total=gross-abs_net
    within= gross-group_gross
    between=group_gross-abs_net
    error=total-within-between
    if abs(error)>1e-12 or min(within,between,total)<-1e-12:
        raise ValueError("Triangle inequality or decomposition failed")
    return (dict(total=total,within=within,between=between,identity_error=error,
                 within_share=within/total if total>0 else np.nan,
                 between_share=between/total if total>0 else np.nan,n_industries=len(net)), net)


def main():
    s=pd.read_csv(TABLES/"e4_snapshot_contributions.csv",dtype={"index_code":str,"security_code":str})
    rows, industries=[],[]
    for (code,day),g in s.groupby(["index_code","trade_date"]):
        for basis in ["raw_return","ref_return"]:
            valid=g.dropna(subset=["industry",basis,"weight_pct"])
            if valid.empty:
                continue
            result, net=decompose(valid.weight_pct/100*valid[basis],valid.industry)
            weights=valid.groupby("industry").weight_pct.sum()/valid.weight_pct.sum()
            rows.append(dict(index=code,date=day,basis=basis,n_file=len(g),n_used=len(valid),
                             included_weight_pct=valid.weight_pct.sum(),industry_hhi=weights.pow(2).sum(),**result))
            industries.append(net.rename("net_contribution").rename_axis("industry").reset_index().assign(index=code,date=day,basis=basis))
    csv(pd.DataFrame(rows),"e5_decomposition")
    if industries:
        csv(pd.concat(industries,ignore_index=True),"e5_industry_contributions")
        d=industries[0].sort_values("net_contribution")
        fig,ax=plt.subplots(figsize=(7,4.5))
        ax.barh(d.industry,d.net_contribution*10000,color=np.where(d.net_contribution>=0,"#42785b","#b24b42"))
        plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
        for t in ax.get_yticklabels():
            t.set_fontfamily("Microsoft YaHei")
        ax.set_xlabel("Net contribution (bp)")
        ax.set_title("CNI 1000 · 2026-11-02 · matched subset")
        figure(fig,"e5_industry")


if __name__=="__main__":
    main()
