"""Render seven rebased paths and verify each panel against the archived V3 series."""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[2] / "experiments"
V2 = EXPERIMENTS / "seven_index_convergence_v2/results"
V3 = EXPERIMENTS / "seven_index_convergence_v3/results"
NAMES = {"000001": "上证综指", "000985": "中证全指", "000300": "沪深300",
         "000903": "中证A100", "000132": "上证100", "399311": "国证1000", "000905": "中证500"}
COLORS = ("#194b80", "#b55c24", "#25846b", "#925985", "#677638", "#855132", "#45789e")
STYLES = ("-", "--", "-", "--", "-", ":", "-.")
METHODS = {"original":"原始行情", "A":"整体分化 A", "U":"并集 U", "R2":"至少两票 R2", "B":"投票前列 B"}


def main(end, destination):
    close = pd.read_csv(V2/"input_closes_main.csv", index_col="date", parse_dates=True)
    assert list(close) == list(NAMES) and close.index.is_unique and close.index.is_monotonic_increasing
    dates = pd.read_csv(V2/"deletion_dates.csv", parse_dates=["date"]).query(
        "sample=='main' and kind=='natural' and K==10")
    archive = pd.read_csv(V3/"key_day_series.csv.gz", parse_dates=["start","end"])
    end = close.index[-1] if end is None else pd.Timestamp(end)
    if end not in close.index or close.index.get_loc(end)<252:
        raise ValueError("Choose an archived trading day with 252 preceding close-to-close returns.")
    end_index = close.index.get_loc(end)
    pairs = list(combinations(range(len(NAMES)), 2))
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,
                         "svg.fonttype":"none","svg.hashsalt":"seven-index-reset-overlay"})
    fig, axes = plt.subplots(5,3,figsize=(12,12.4),sharey="col",sharex="col")
    fig.subplots_adjust(left=.075,right=.985,top=.875,bottom=.052,hspace=.59,wspace=.22)
    for col, w in enumerate((60,120,252)):
        panel_close = close.iloc[end_index-w:end_index+1]
        returns = np.log(panel_close).diff().iloc[1:]
        base_date = panel_close.index[0]
        for row, (method, title) in enumerate(METHODS.items()):
            selected = returns.index.isin(dates.loc[dates.method.eq(method),"date"])
            daily = returns.copy()
            daily.loc[selected] = 0.
            paths = daily.cumsum()
            rms = np.sqrt(np.mean([(paths.iloc[:,i]-paths.iloc[:,j]).pow(2).mean() for i,j in pairs]))
            scenario = "A_K10" if method=="original" else method+"_K10"
            ref = archive.loc[archive.end.eq(end) & archive.window.eq(w) & archive.scenario.eq(scenario)].squeeze()
            assert ref.start == returns.index[0]
            np.testing.assert_allclose(rms,ref.original if method=="original" else ref.modified,atol=1e-12,rtol=0)
            lines = pd.concat([panel_close.iloc[:1]*0,paths])*100
            ax = axes[row,col]
            for code,color,style in zip(NAMES,COLORS,STYLES):
                ax.plot(np.arange(w+1),lines[code],lw=1.02,color=color,ls=style,label=NAMES[code])
            ax.set_title(f"{title}   ·   窗内删 {selected.sum()} 日   ·   RMS {100*rms:.2f}",
                         fontsize=9,loc="left",pad=5)
            if row==0:
                ax.text(.5,1.27,f"{w} 个交易日   |   基日 {base_date:%Y-%m-%d}",
                        transform=ax.transAxes,ha="center",fontsize=11,fontweight="bold")
            ax.axhline(0,color=".5",lw=.5,zorder=0)
            ax.grid(axis="y",alpha=.16)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_xlim(0,w)
            ticks = [0,w//4,w//2,3*w//4,w]
            ax.set_xticks(ticks)
            ax.tick_params(labelsize=8)
            if row==4:
                ax.set_xlabel("距基日的交易日数")
    fig.suptitle(f"窗口重置后的七指数累计路径    |    截至 {end:%Y-%m-%d}",fontsize=16,
                 fontweight="bold",y=.988)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,.957),ncol=7,frameon=False,fontsize=9)
    fig.text(.012,.48,"累计对数收益 × 100（对数点）",va="center",rotation=90,fontsize=11)
    destination.mkdir(parents=True,exist_ok=True)
    stem = "figure_04" if end == close.index[-1] else "reset_overlay_"+end.strftime("%Y%m%d")
    fig.savefig(destination/(stem+".png"),dpi=180,facecolor="white")
    svg = destination/(stem+".svg")
    fig.savefig(svg,metadata={"Date":None})
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    plt.close(fig)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end",help="Optional archived trading date (YYYY-MM-DD); default is the last close.")
    parser.add_argument("--out",type=Path,default=HERE)
    args = parser.parse_args()
    main(args.end,args.out)
