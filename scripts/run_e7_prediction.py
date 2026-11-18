"""Expanding-window M0/M1/M2 forecasts with an outcome maturity embargo."""
import argparse

from research_common import *
from statsmodels.stats.multitest import multipletests

MODELS={"M0":["market_return_20","index_vol_20","log_volume"],
        "M1":["market_return_20","index_vol_20","log_volume","cssd","csad"],
        "M2":["market_return_20","index_vol_20","log_volume","cssd","csad","co","gross","between_share"]}


def forecasts(panel, r):
    predictions,results=[],[]
    for code,g in panel.groupby("index_code"):
        if g.trade_date.duplicated().any() or code not in TARGETS:
            raise ValueError("Expected unique target index-date rows")
        y=r[code]
        x=g.set_index("trade_date").reindex(y.index)[MODELS["M2"]].replace([np.inf,-np.inf],np.nan)
        for h in (5,20):
            rv=(y.pow(2).rolling(h).sum().shift(-h)*252/h).pow(.5)
            dd=y.rolling(h).apply(drawdown,raw=True).shift(-h).abs()
            for label,outcome in [("volatility",rv),("drawdown",dd)]:
                d=x.assign(actual=outcome,position=np.arange(len(y))).dropna()
                rows=[]
                for pos,(day,observation) in enumerate(d.iterrows()):
                    if pos==0 or pos%20==0:
                        train=d.loc[d.position + h <= observation.position]
                        coefs={}
                        if len(train)>=504:
                            for model,cols in MODELS.items():
                                coefs[model]=np.linalg.lstsq(np.column_stack([np.ones(len(train)),train[cols]]),
                                                             train.actual,rcond=None)[0]
                    if not coefs:
                        continue
                    rows.append(dict(index=code,horizon=h,target=label,date=day,n_train=len(train),
                                     training_last_origin=train.index[-1],training_last_outcome=y.index[int(train.position.iloc[-1])+h],
                                     actual=observation.actual,mean_benchmark=train.actual.mean(),
                                     **{m:max(0.,np.r_[1.,observation[MODELS[m]]]@b) for m,b in coefs.items()}))
                if not rows:
                    continue
                f=pd.DataFrame(rows).set_index("date")
                naive_loss=(f.actual-f.mean_benchmark).pow(2)
                m0_loss=(f.actual-f.M0).pow(2)
                m1_loss=(f.actual-f.M1).pow(2)
                for m in MODELS:
                    err=f.actual-f[m]
                    loss=err.pow(2)
                    comparators={"M0":m0_loss}
                    if m=="M2":
                        comparators["M1"]=m1_loss
                    for baseline,baseline_loss in comparators.items():
                        delta=baseline_loss-loss
                        p=None
                        if m!="M0" and delta.std()>0:
                            design=pd.DataFrame(index=delta.index)
                            p=float(fit(delta,design,lags=h,calendar=y.loc[delta.index[0]:delta.index[-1]].index).pvalues[0])
                        results.append(dict(index=code,horizon=h,target=label,model=m,baseline=baseline,
                                            **window(f.index),rmse=np.sqrt(loss.mean()),mae=err.abs().mean(),
                                            oos_r2=1-loss.sum()/naive_loss.sum(),loss_gain=delta.mean(),hac_p=p))
                predictions.append(f.reset_index())
    if not predictions:
        raise ValueError("Insufficient common history for 504 matured training observations plus OOS outcomes")
    result=pd.DataFrame(results)
    has_p=result.hac_p.notna()
    result.loc[has_p,"p_holm"]=multipletests(result.loc[has_p,"hac_p"],method="holm")[1] if has_p.any() else np.nan
    return result,pd.concat(predictions,ignore_index=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",type=Path)
    a=p.parse_args()
    if a.panel is None:
        print("BLOCKED: pass --panel with aligned end-of-day features for all three models")
        return
    input_frame=pd.read_csv(a.panel,parse_dates=["trade_date"],dtype={"index_code":str})
    _,r,_=market()
    results,predictions=forecasts(input_frame,r)
    csv(results,"e7_forecast_metrics")
    csv(predictions,"e7_forecasts")


if __name__=="__main__":
    main()
