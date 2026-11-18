import re

from research_common import db, csv, pd, np

BROAD = {"000001", "000300", "000903", "000132", "399311", "000985"}

def category(code,name):
    if code in BROAD or re.fullmatch(r"(中证(A?(50|100|200|300|500|800|1000|2000|5000)|全指)|沪深300|上证(50|180|380|综指)|深证(成指|100)|国证(2000|A指)|科创(50|100|200)|创业板(指|50)|北证50)(指数)?",name):
        return "宽基"
    if re.search("行业|银行|证券公司|半导体|医药|煤炭|房地产",name):
        return "行业"
    if re.search("主题|人工智能|新能源|机器人",name):
        return "主题"
    if re.search("红利|低波|价值|动量|等权|增强",name):
        return "策略"
    return "待核实类别"


def universe():
    with db() as con:
        data = pd.read_sql_query(
            "SELECT d.index_code,m.index_name,d.provider,d.trade_date,d.close FROM index_daily d "
            "LEFT JOIN index_master m USING(provider,index_code) "
            "ORDER BY d.trade_date,d.index_code", con, parse_dates=["trade_date"])
    providers=(data.groupby(["index_code","provider"]).close.count().reset_index(name="count")
               .assign(preferred=lambda d:d.provider.eq("eastmoney"))
               .sort_values(["count","preferred","provider"],ascending=[False,False,True])
               .drop_duplicates("index_code"))
    data=data.merge(providers[["index_code","provider"]],on=["index_code","provider"])
    prices = data.pivot(index="trade_date", columns="index_code", values="close")
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    rows = []
    for code in prices:
        name = data.loc[data.index_code.eq(code),"index_name"].iloc[0]
        s = prices[code].dropna()
        kind=category(code,str(name))
        broad = kind=="宽基"
        rows.append(dict(code=code,name=name,category=kind,provider=data.loc[data.index_code.eq(code),"provider"].iloc[0],
                         start=s.index.min(),end=s.index.max(),valid_days=len(s),
                         missing_rate=1-len(s)/len(prices),in_main=broad and len(s)>=1260,
                         exclusion="" if broad and len(s)>=1260 else "分类未确认或样本少于1260日"))
    u = pd.DataFrame(rows)
    csv(u,"index_universe")
    return prices[u.loc[u.in_main,"code"]].dropna(),u


if __name__=="__main__":
    universe()
