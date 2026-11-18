"""Rotation interface for a validated continuous historical industry panel."""
import argparse

from research_common import *
from run_e2_regimes import classify


def rotation(panel, market_close):
    required={"trade_date","index_code","industry","industry_contribution","industry_start_weight","stock_abs_contribution"}
    missing=required-set(panel.columns)
    if missing:
        raise ValueError(f"Missing {sorted(missing)}")
    if panel[list(required)].isna().any().any() or panel.duplicated(["index_code","trade_date","industry"]).any():
        raise ValueError("Need unique complete industry-day inputs")
    states=classify(market_close)[0]
    rows=[]
    for code,group in panel.groupby("index_code"):
        c=group.pivot(index="trade_date",columns="industry",values="industry_contribution").sort_index()
        w=group.pivot(index="trade_date",columns="industry",values="industry_start_weight").reindex_like(c)
        if c.isna().any().any() or w.isna().any().any() or w.lt(0).any().any() or not np.allclose(w.sum(axis=1),1,atol=.002):
            raise ValueError(f"{code}: incomplete panel or invalid start weights")
        if not c.index.equals(market_close.loc[c.index.min():c.index.max()].index):
            raise ValueError(f"{code}: missing trading sessions")
        active=w.gt(0)
        industry_r=c.div(w.where(active))
        s=states.reindex(c.index)
        groups=s.ne(s.shift()).cumsum()
        for _,state in s.groupby(groups):
            if state.iloc[0]!="sideways" or len(state)<5:
                continue
            d=c.loc[state.index]
            net=d.sum(axis=1)
            if not net.gt(-1).all():
                raise ValueError("Invalid net return")
            wealth=(1+net).cumprod().shift(fill_value=1)
            linked=d.mul(wealth,axis=0)
            linked_abs=linked.abs().to_numpy().sum()
            cross=wealth @ (d.abs().sum(axis=1)-net.abs())
            # Keep time-series price reversals distinct from cross-industry offsets.
            reversal=(linked.abs().sum()-linked.sum().abs()).sum()
            if not np.isclose(linked.sum().sum(),(1+net).prod()-1):
                raise ValueError("Linking failed")
            gross=group[group.trade_date.isin(state.index)].groupby("trade_date").stock_abs_contribution.sum()
            if (gross.reindex(d.index) < d.abs().sum(axis=1)-1e-12).any():
                raise ValueError("Stock contributions do not reconcile to industry gross")
            leader=d.idxmax(axis=1)
            ranks=industry_r.loc[state.index].rank(axis=1)
            rank_corr=ranks.corrwith(ranks.shift(),axis=1).iloc[1:].mean()
            rows.append(dict(index=code,**window(state.index),cumulative=linked.sum().sum(),
                             industry_daily_offset_ratio=cross/linked_abs if linked_abs else np.nan,
                             daily_stock_offset_ratio=(wealth@(gross-net.abs()))/(wealth@gross) if (wealth@gross)>0 else np.nan,
                             industry_time_reversal=reversal,leader_switch=leader.ne(leader.shift()).iloc[1:].mean(),
                             rank_persistence=rank_corr,realized_vol_ann=net.std()*np.sqrt(252)))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",type=Path)
    a=p.parse_args()
    if a.panel is None:
        print("BLOCKED: pass --panel with continuous point-in-time industry contributions")
        return
    data=pd.read_csv(a.panel,parse_dates=["trade_date"],dtype={"index_code":str})
    close,_,_=market()
    csv(rotation(data,close["000001"]),"e6_rotation")


if __name__=="__main__":
    main()
