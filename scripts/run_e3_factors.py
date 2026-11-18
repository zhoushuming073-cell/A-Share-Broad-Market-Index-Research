from research_common import *
from statsmodels.stats.multitest import multipletests


def main():
    _, ret, _ = market()
    ch3, ch4 = factors(), factors("CH4")
    sample = ret.index.intersection(ch3.index).intersection(ch4.index)
    rf = ch3.loc[sample,"rf_dly"]
    if not ch4.loc[sample,"rf_dly"].equals(rf):
        raise ValueError("Factor versions have different risk-free rates")
    fits, coefficients = [], []
    specs = {"market_CH": ch3[["mktrf"]], "CH3": ch3[["mktrf","SMB","VMG"]],
             "CH4": ch4[["mktrf","SMB","VMG","PMO"]]}
    for name, returns in {**{c:ret[c] for c in TARGETS}, "000903_minus_000300":ret["000903"]-ret["000300"]}.items():
        y = returns.loc[sample] - (0 if "_minus_" in name else rf)
        for model, x in specs.items():
            x = x.loc[sample]
            f = fit(y, x)
            fits.append(dict(index=name,model=model,**window(sample),r2=f.rsquared,r2_adj=f.rsquared_adj,
                             alpha_daily=f.params[0],alpha_ann_linear=252*f.params[0],alpha_p=f.pvalues[0],
                             resid_vol_ann=np.sqrt(252*f.ssr/f.df_resid)))
            for term,b,se,p,ci in zip(["alpha",*x.columns], f.params,f.bse,f.pvalues,f.conf_int()):
                coefficients.append(dict(index=name,model=model,term=term,estimate=b,se=se,p=p,lo=ci[0],hi=ci[1]))
    summary=pd.DataFrame(fits)
    summary["alpha_p_holm"]=multipletests(summary.alpha_p,method="holm")[1]
    coeff=pd.DataFrame(coefficients)
    coeff["p_holm_by_model_term"]=coeff.groupby(["model","term"])["p"].transform(lambda p:multipletests(p,method="holm")[1])
    csv(summary,"e3_models")
    csv(coeff,"e3_coefficients")
    fig,ax=plt.subplots(figsize=(8,4))
    d=summary[summary["index"].isin(TARGETS)].pivot(index="index",columns="model",values="r2_adj").loc[TARGETS,list(specs)]
    d.rename(index=NAMES).plot.bar(ax=ax,rot=0)
    ax.set_xlabel("")
    ax.set_ylabel("Adjusted R²")
    figure(fig,"e3_factor_fit")


if __name__ == "__main__":
    main()
