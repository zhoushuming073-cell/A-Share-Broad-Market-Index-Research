import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from data_validation import HERE,OUT,CODES,INDICES,CFG,PAIRS

FIG=HERE/"figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"Microsoft YaHei","svg.fonttype":"none","axes.unicode_minus":False})

def run(r):
    counts=pd.read_csv(OUT/"key_date_overlap.csv",parse_dates=["date"]).sort_values(
        ["top10","date"],ascending=[False,True])
    counts.head(20).to_csv(OUT/"common_key_dates.csv",index=False)
    distances=[]
    first=pd.DataFrame([np.zeros(9)],columns=CODES,index=[pd.Timestamp(CFG["start"])])
    fig,axes=plt.subplots(2,1,figsize=(11.5,8.3),sharey=True,sharex=True,layout="constrained")
    colors=["#4e677f","#438657","#2555a5","#a56f34","#9472ac","#4c9d9b","#756459","#da6822","#a82457"]
    series=[]
    for n in [0,CFG["common_delete"],*CFG["common_sensitivity"]]:
        mask=r.index.isin(counts.head(n).date)
        x=r.copy()
        x.loc[mask]=0
        log_paths=pd.concat([first,x]).cumsum()
        nav=np.exp(log_paths)
        series.append(nav.add_prefix(f"delete{n}_"))
        for a,b in PAIRS:
            gap=log_paths[a]-log_paths[b]
            special=set((a,b)).intersection(CFG["special_indices"])
            group="seven" if not special else ("both" if len(special)==2 else special.pop())
            distances.append(dict(deleted=n,pair=a+"_"+b,group=group,
                rms=np.sqrt(np.mean(gap**2)),terminal_log_gap=abs(gap.iloc[-1]),
                return_corr=r.loc[~mask,a].corr(r.loc[~mask,b])))
        if n not in (0,CFG["common_delete"]): continue
        ax=axes[int(n>0)]
        for code,col in zip(CODES,colors):
            focal=code in CFG["special_indices"]
            ax.plot(nav.index,nav[code]-1,label=INDICES[code][0],color=col,
                linewidth=1.65 if focal else 1.0,ls="--" if code=="399006" else "-",alpha=.95)
        ax.set_title("A　原始累计价格收益" if n==0 else f"B　同步剔除 {n} 个共同关键日后的累计价格收益",loc="left")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("累计收益")
        ax.axhline(0,c=".5",lw=.7)
        ax.grid(axis="y",alpha=.18)
        if n:
            ax.plot(r.index[mask],np.repeat(.02,mask.sum()),"|",transform=ax.get_xaxis_transform(),color="black")
    axes[0].legend(ncol=5,fontsize=9,loc="upper left")
    axes[1].set_xlabel("交易日期　｜　黑色短线标记共同剔除日")
    fig.savefig(FIG/"nine_paths.png",dpi=240)
    fig.savefig(FIG/"nine_paths.svg")
    plt.close(fig)
    pd.concat(series,axis=1).to_csv(OUT/"nine_paths.csv",index_label="date")
    pd.DataFrame(distances).to_csv(OUT/"path_distances.csv",index=False)
