"""The fixed A-share price-index universe used by new research."""

INDICES = {
    "399311": ("国证1000", "CNI"),
    "000903": ("中证A100", "CSI"),
    "000132": ("上证100", "SSE/CSI"),
    "000300": ("沪深300", "CSI"),
    "000001": ("上证综合指数", "SSE"),
    "000985": ("中证全指", "CSI"),
    "000905": ("中证500", "CSI"),
    "399006": ("创业板指", "SZSE/CNI"),
    "000688": ("科创50", "SSE/CSI"),
}

SINA_SYMBOLS = {
    "000905": "sh000905",
    "399006": "sz399006",
    "000688": "sh000688",
}

PROVIDERS = {code: "sina" if code in SINA_SYMBOLS else "eastmoney" for code in INDICES}
LEGACY_CODES = tuple(code for code in INDICES if code not in SINA_SYMBOLS)
STAR50_REALTIME_START = "2020-07-23"
