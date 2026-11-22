import re

from research_common import db, csv, pd, np
from index_pool import INDICES, LEGACY_CODES, PROVIDERS, STAR50_REALTIME_START

BROAD = set(INDICES)

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


def universe(*, legacy=False):
    """Load the fixed nine-index panel, or the six-index replay input."""
    codes = LEGACY_CODES if legacy else tuple(INDICES)
    placeholders = ",".join("?" for _ in codes)
    with db() as con:
        data = pd.read_sql_query(
            "SELECT d.index_code,m.index_name,d.provider,d.trade_date,d.close FROM index_daily d "
            "JOIN index_master m USING(provider,index_code) "
            f"WHERE d.index_code IN ({placeholders}) "
            "ORDER BY d.trade_date,d.index_code", con, params=codes, parse_dates=["trade_date"])
    data = data.loc[data.provider.eq(data.index_code.map(PROVIDERS))]
    missing = set(codes) - set(data.index_code)
    if missing:
        raise ValueError("Missing price series in the fixed index pool: " + ", ".join(sorted(missing)))
    prices = data.pivot(index="trade_date", columns="index_code", values="close")
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    first = prices.apply(pd.Series.first_valid_index)
    last = prices.apply(pd.Series.last_valid_index)
    start, end = first.max(), last.min()
    if not legacy:
        start = max(start, pd.Timestamp(STAR50_REALTIME_START))
    aligned = prices.loc[start:end]
    if aligned.empty or aligned.isna().any().any():
        raise ValueError("Missing closes within the shared index trading period")
    rows = []
    for code in aligned:
        name = data.loc[data.index_code.eq(code),"index_name"].iloc[0]
        s = prices[code].dropna()
        rows.append(dict(code=code,name=name,category=category(code,str(name)),provider=PROVIDERS[code],
                         start=s.index.min(),end=s.index.max(),valid_days=len(s),
                         missing_rate=s.reindex(prices.loc[s.index.min():s.index.max()].index).isna().mean(),
                         in_main=True,exclusion=""))
    u = pd.DataFrame(rows)
    return aligned, u


if __name__=="__main__":
    close, metadata = universe()
    csv(metadata, "index_universe")
    csv(metadata, "index_universe_nine")
    csv(close, "index_closes_nine", index=True)
    print(f"{len(close.columns)} indices; {len(close)} closes; {close.index[0].date()} to {close.index[-1].date()}")
