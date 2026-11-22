"""Replay the first three distinct configurations of the historical return ranking."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1] / "experiments/multidimensional_longshort_search_v1"
sys.path.insert(0, str(MODEL))
from backtest import COLS, simulate
from data import P, OUT


def replay():
    ranks = pd.read_csv(OUT / "top_return.csv", dtype={"id": str}).head(3)
    curves = pd.DataFrame(index=P.index)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
    for row, color in zip(ranks.itertuples(), ["#124f69","#bc6636","#6d5686"]):
        paths = simulate(json.loads(row.params), 0, len(P)-1)
        nav = np.r_[1., paths[:, COLS.index("nav")]]
        drawdown = 1 - nav / np.maximum.accumulate(nav)
        np.testing.assert_allclose([nav[-1]-1, drawdown.max()], [row.net, row.mdd], atol=1e-11)
        curves[row.id] = nav
        ax.plot(P.index, nav, color=color, lw=1.2,
                label=f"第{row.rank}名    {row.net:+.2%}")
    original = pd.read_csv(OUT / "daily_nav.csv")
    np.testing.assert_allclose(curves.iloc[1:,0], original["return"], atol=1e-11)
    ax.axhline(1, lw=.7, color="#66727b", linestyle="--")
    ax.set_ylabel("扣费后净值（期初＝1）")
    ax.set_xlabel("交易日期")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.16)
    ax.legend(loc="upper left",frameon=False)
    curves.index.name = "trade_date"
    curves.to_csv(HERE/"top_three_nav.csv",float_format="%.12f")
    fig.savefig(HERE/"figures"/"top_three.svg")
    plt.close(fig)


if __name__ == "__main__":
    replay()

