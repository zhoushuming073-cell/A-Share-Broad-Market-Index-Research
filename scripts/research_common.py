from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/a_share_indices.sqlite"
TABLES = ROOT / "outputs/tables"
FIGURES = ROOT / "outputs/figures"
INPUTS = ROOT / "outputs/inputs"
LOGS = ROOT / "outputs/logs"
REPORTS = ROOT / "reports"
for folder in (TABLES, FIGURES, INPUTS, LOGS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

TARGETS = ["399311", "000903", "000132", "000300"]
BENCHES = ["000001", "000985"]
CODES = TARGETS + BENCHES
NAMES = dict(zip(CODES, ["CNI 1000", "CSI A100", "SSE 100", "CSI 300", "SSE Composite", "CSI All Share"]))
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .19, "font.size": 10})
HAC_LAGS = 5
SEED = 20261118


def db():
    return sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)


def market():
    with db() as conn:
        q = pd.read_sql_query("SELECT trade_date, index_code, close, turnover_value FROM index_daily "
                              "WHERE provider='eastmoney' ORDER BY trade_date", conn, parse_dates=["trade_date"])
    close = q.pivot(index="trade_date", columns="index_code", values="close")[CODES]
    if close.isna().any().any() or (close <= 0).any().any():
        raise ValueError("Unaligned index calendar or invalid close")
    ret = close.pct_change(fill_method=None).dropna()
    amount = q.pivot(index="trade_date", columns="index_code", values="turnover_value")[CODES]
    return close, ret, amount


def csv(frame, name, index=False):
    frame.to_csv(TABLES / f"{name}.csv", encoding="utf-8-sig", index=index, float_format="%.10g")


def figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def window(index):
    return {"n": len(index), "start": index.min(), "end": index.max()}


def fit(y, x, lags=HAC_LAGS, calendar=None):
    design = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, design).fit()
    if calendar is None:
        return model.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)
    # Preserve actual trading-day gaps for regressions on a subset of dates.
    scores = pd.DataFrame(design.to_numpy() * model.resid.to_numpy()[:, None],
                          index=y.index).reindex(calendar, fill_value=0).to_numpy()
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    bread = model.normalized_cov_params.to_numpy()
    robust = model.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)
    robust.cov_params_default = bread @ meat @ bread * len(y) / model.df_resid
    return robust


def regression_row(y, x, calendar=None):
    f = fit(y, x, calendar=calendar)
    p = np.asarray(f.params)
    ci = np.asarray(f.conf_int())
    return {**window(y.index), "alpha_daily": p[0], "alpha_ann_linear": 252*p[0],
            "alpha_p": np.asarray(f.pvalues)[0], "alpha_daily_lo":ci[0,0], "alpha_daily_hi":ci[0,1],
            "beta": p[1], "beta_se": np.asarray(f.bse)[1],
            "beta_lo": ci[1, 0], "beta_hi": ci[1, 1],
            "beta_eq_one_p": beta_one_p(f),
            "r2": f.rsquared, "resid_vol_ann": np.sqrt(252 * f.ssr / f.df_resid)}


def beta_one_p(f):
    from scipy.stats import t
    return 2 * t.sf(abs((np.asarray(f.params)[1] - 1) / np.asarray(f.bse)[1]), f.df_resid)


def drawdown(returns):
    wealth = np.r_[1., (1 + np.asarray(returns)).cumprod()]
    return np.min(wealth / np.maximum.accumulate(wealth) - 1)


def factors(model="CH3"):
    data = pd.read_excel(INPUTS / f"{model}_factors_daily_202608.xlsx")
    data.index = pd.to_datetime(data.pop("date").astype(str), format="%Y%m%d")
    if not data.index.is_unique or data.isna().any().any() or data.abs().max().max() > 1:
        raise ValueError("Invalid factor series")
    return data.sort_index()
