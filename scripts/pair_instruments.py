"""Cache only instrument data not already in the index database."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import akshare as ak
from optimize_hedge_ratio import account, metrics
from run_pair_search import inputs, load
from research_common import csv, np, pd, INPUTS, REPORTS

CACHE=INPUTS/"pair_instruments"
PRODUCTS={"000001":("510210","2011-03-25"),"000300":("510300","2012-05-28"),"000903":("512910","2019-07-01")}


def instrument(code,start,end):
    file=CACHE/f"{code}_{start}_{end}.csv"
    if not file.exists():
        if code.startswith("IF"):
            data=ak.futures_zh_daily_sina(symbol=code).rename(columns={"close":"price"})
        else:
            data=ak.fund_etf_hist_em(symbol=code,period="daily",start_date=start,end_date=end,
                                     adjust="qfq").rename(columns={"日期":"date","收盘":"price","成交额":"amount"})
        data["date"]=pd.to_datetime(data.date)
        data=data.loc[data.date.between(pd.Timestamp(start),pd.Timestamp(end))]
        if data.empty or data.date.duplicated().any() or (data.price<=0).any():
            raise ValueError(f"invalid or empty {code}")
        data.to_csv(file,index=False,encoding="utf-8-sig")
    return pd.read_csv(file,parse_dates=["date"]).set_index("date").sort_index()


def contract(day):
    month=day.to_period("M")
    first=month.start_time
    third_friday=first+pd.Timedelta(days=(4-first.weekday())%7+14)
    if day>=third_friday-pd.Timedelta(days=7):
        month+=1
    return "IF"+month.strftime("%y%m")


def instruments():
    CACHE.mkdir(exist_ok=True,parents=True)
    p,r,u,(_,va)=inputs()
    dates=r.index[va:]; prev=p.index[va:-1]
    start,end=prev[0].strftime("%Y%m%d"),dates[-1].strftime("%Y%m%d")
    calendar=p.index[va:]
    contracts=[contract(d) for d in prev]
    keys=[v[0] for v in PRODUCTS.values()]+sorted(set(contracts))
    quotes={}; gaps=[]
    def fetch(code):
        try:
            return code,instrument(code,start,end),None
        except Exception as e:
            return code,None,str(e)[:160]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for code,data,error in pool.map(fetch,keys):
            if error:
                gaps.append(f"{code} 数据源返回 {error}")
            else:
                quotes[code]=data
    records=[]; etfret={}
    for row in u.itertuples():
        product,listing=PRODUCTS.get(row.code,("",""))
        rec=dict(code=row.code,name=row.name,etf=product,listing=listing,n=0,status="未核实精确跟踪ETF")
        if product in quotes:
            df=quotes[product].reindex(calendar)
            er=df.price.pct_change(fill_method=None).loc[dates]
            etfret[row.code]=er
            rec.update(status="已核实",n=er.notna().sum(),median_amount_cny=df.amount.loc[dates].median(),
                       market_tracking_vol=(er-r.loc[dates,row.code]).std()*np.sqrt(252))
        records.append(rec)
    inst=pd.DataFrame(records)
    csv(inst,"pair_trade_instruments")
    future=[]
    for k,(day,before,code) in enumerate(zip(dates,prev,contracts)):
        q=quotes.get(code)
        if q is None or before not in q.index or day not in q.index:
            future.append(dict(date=day,contract=code,ret=np.nan));continue
        price=q.at[before,"price"]
        roll=k>0 and code!=contracts[k-1]
        gap=np.nan
        if roll and contracts[k-1] in quotes and before in quotes[contracts[k-1]].index:
            gap=price-quotes[contracts[k-1]].at[before,"price"]
        future.append(dict(date=day,contract=code,ret=q.at[day,"price"]/price-1,
                           entry_basis_points=price-p.at[before,"000300"],roll=roll,new_minus_old_points=gap))
    future=pd.DataFrame(future).set_index("date")
    csv(future,"pair_if_roll_daily",index=True)
    benchmark=load("all_pair_results").query("a=='000001' and b=='000300' and spec=='static'").iloc[0]
    h=benchmark.test_mean_h
    rows=[]
    if "000001" in etfret:
        for label,leg in [("ETF/ETF",etfret.get("000300")),("ETF/IF",future.ret)]:
            if leg is None or not (leg.notna()&etfret["000001"].notna()).all():
                gaps.append(label+" 缺少连续共同报价，未拼接收益");continue
            w=np.full(len(dates),1/(1+h)); v=-h*w
            days=(dates-prev).days.to_numpy(float)
            for bp,borrow in [(0,0),(5,.02 if label=="ETF/ETF" else 0)]:
                z=account(etfret["000001"].to_numpy(),leg.to_numpy(),w,v,days,
                          np.ones(len(w),bool),bp/10000,borrow)
                # For IF, replace single net resize at roll by closing old + opening new.
                # Apply the additional matched turnover fee to the daily growth factor.
                rr=z[:,0].copy()
                if label=="ETF/IF" and bp:
                    drift=np.r_[0,abs(v[:-1])*(1+leg.to_numpy()[:-1])/(1+rr[:-1])]
                    extra=2*np.minimum(drift,abs(v))*future["roll"].to_numpy(bool)
                    rr-=extra*bp/10000
                rows.append(dict(implementation=label,bp=bp,borrow=borrow,**metrics(rr)))
    sims=pd.DataFrame(rows)
    csv(sims,"pair_instrument_returns")
    fmt=lambda x:"待核" if pd.isna(x) else f"{x:.2%}"
    lines=["# 配对策略实际可交易性核验","",
           f"行情观察期为 {dates[0]:%Y-%m-%d} 至 {dates[-1]:%Y-%m-%d}。测试收益冠军与验证锁定策略的空头均为上证100（000132）；本次未核实直接跟踪该指数的ETF或股指期货，也没有该腿的历史可借份额与报价，无法给出这两组的实盘净收益。",
           "","|指数|已核实的ETF示例|上市日|日收益数|日成交额中位数（亿元）|相对价格指数的年化跟踪波动|",
           "|---|---|---|---:|---:|---:|"]
    for q in inst.itertuples():
        lines.append(f"|{q.name} {q.code}|{q.etf or '待核'}|{q.listing or '待核'}|{q.n}|{q.median_amount_cny/1e8:.2f}|{fmt(q.market_tracking_vol)}|".replace("nan|","—|"))
    lines+=["","ETF使用东方财富经AKShare取得的前复权场内收盘与成交额，跟踪波动是ETF市场收益减价格指数收益的日标准差乘√252，包含溢折价与分红口径差异。净值对全收益基准的正式跟踪误差、买卖盘深度及逐日券源仍待核。中证1000和国证1000对应不同标的。",
           "","## 有工具可核验的原配对","",
           f"用主实验训练期冻结的 h={h:.4f}，做多510210，分别做空510300或IF。ETF模型允许按收盘成交及借券，IF模型使用单月合约收盘、每日重置名义敞口，允许分数手，资金利息为零。IF手续费按成交额作压力假设，跨月换仓另计双边名义的增量近似费用。",
           "","|工具|成交费（单边bp）|年化借券假设|累计|年化|最大回撤|","|---|---:|---:|---:|---:|---:|"]
    for q in sims.itertuples():
        lines.append(f"|{q.implementation}|{q.bp}|{q.borrow:.0%}|{q.total:.2%}|{q.annual:.2%}|{q.mdd:.2%}|")
    lines+=["",f"IF与沪深300精确对应，合约乘数300元/点。每月第三个周五前七个自然日起切换次月，从当日收盘持有同一合约至次日收盘；使用真实交易日。覆盖 {future.ret.notna().sum()}/{len(dates)} 个收益日。"
           + (f"期初基差（期货减现货）均值 {future.entry_basis_points.mean():.2f} 点，{future.new_minus_old_points.count()} 次换月新减旧合约收盘差均值 {future.new_minus_old_points.mean():.2f} 点。跨合约报价差用于描述远近月结构，不直接扣作单日损失。" if 'entry_basis_points' in future else ""),
           "",f"以合约条款8%最低保证金示例，空头名义占净资产{h/(1+h):.2%}时，最低保证金约占净资产{.08*h/(1+h):.2%}。券商实际保证金、逐日调整、追加保证金、整手约束、冲击成本及实际融资费率均未重建。其他五条指数的同标的期货未获确认。",
           "","数据与依据：[富国上证综指ETF上市公告](https://www.fullgoal.com.cn/contents/2011/3/22-7fbb2a6f15f643659f31245c0a850fa2.html) · [沪深300ETF产品概要](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2023-03-31/510300_20230331_G1XG.pdf) · [广发中证A100ETF产品概要](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2025-06-24/512910_20250624_RX1J.pdf) · [中金所IF合约细则](https://www.cffex.com.cn/u/cms/www/202003/27165436e0mo.pdf)。",
           "","工具行情快照存于 outputs/inputs/pair_instruments，逐日换月明细与可交易性数据表位于 outputs/tables。"]
    if gaps:
        lines+=["","### 报价缺口","",*["- "+s for s in gaps]]
    (REPORTS/"pair_trading_implementation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    instruments()
