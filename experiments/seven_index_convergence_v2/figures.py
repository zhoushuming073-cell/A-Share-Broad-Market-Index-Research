import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from data_validation import CFG, CODES, HERE, LABELS, NAMES, OUT, PAIRS
from bootstrap_inference import METHODS

FIG = HERE / "figures"
COLORS = ("#235686","#d37624","#3b8968","#9c496b","#63749a","#877038","#168384")
FRIENDLY = dict(A="整体分化 A",U="并集 U",R2="重复集合 R2",B="投票 B",original="原始路径")
PALETTE = dict(A="#256094",U="#ac6138",R2="#368574",B="#946095")

def save(fig,name):
    fig.savefig(FIG/f"{name}.png",dpi=180)
    svg = FIG/f"{name}.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",
                   encoding="utf-8")
    plt.close(fig)


def panel(ax, curves, title):
    for code,c in zip(CODES,COLORS):
        ax.plot(curves.index,curves[code],label=NAMES[code],color=c,lw=1.05)
    ax.set_title(title,loc="left",fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlim(curves.index[0],curves.index[-1])
    ax.axhline(0,color="#89939e",lw=.6)
    ax.grid(axis="y",alpha=.16)


def independent_pair_panels(k):
    closes = pd.read_csv(OUT/"input_closes_main.csv",index_col=0,parse_dates=True)
    returns = np.log(closes).diff().fillna(0.)
    original = closes / closes.iloc[0] - 1
    keys = pd.read_csv(OUT/"pairwise_key_dates.csv",parse_dates=["date"]).query("sample=='main' and K==@k")
    results = pd.read_csv(OUT/"independent_pair_deletion.csv").query("sample=='main' and K==@k").set_index("pair")
    panels = []
    for (a,b),label in zip(PAIRS,LABELS):
        codes = [CODES[a], CODES[b]]
        keep = returns.loc[:,codes].copy()
        keep.loc[keys.loc[keys.pair.eq(label),"date"]] = 0.
        panels.append((codes,label,np.expm1(keep.cumsum())))
    values = np.concatenate([frame.values.ravel() for _,_,frame in panels]+[original.values.ravel()])
    margin = .04*(values.max()-values.min())
    for start in range(0,len(panels),7):
        fig,axes=plt.subplots(4,2,figsize=(12,9.5),sharex=True,sharey=True,layout="constrained")
        for ax,(codes,label,after) in zip(axes.flat,panels[start:start+7]):
            for code,color in zip(codes,("#235686","#b95129")):
                ax.plot(original.index,original[code],ls="--",lw=.8,color=color,alpha=.47)
                ax.plot(after.index,after[code],lw=1.0,color=color,label=NAMES[code])
            delta=results.loc[label,"improvement"]
            ax.set_title(f"{NAMES[codes[0]]} / {NAMES[codes[1]]}    RMS改善率 {delta:+.1%}",
                         fontsize=10,loc="left")
            ax.set_xlim(closes.index[0],closes.index[-1])
            ax.set_ylim(values.min()-margin,values.max()+margin)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.tick_params(labelbottom=True,labelleft=True,labelsize=8)
            ax.axhline(0,color=".55",lw=.6)
            ax.grid(axis="y",alpha=.16)
            ax.legend(loc="upper left",fontsize=7,frameon=False,ncol=2)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(.03,.53,"虚线为原始累计价格收益\n实线为该配对独立剔除关键日后的累计收益\n每个面板的删除日期由本配对独立识别",
                           transform=axes.flat[-1].transAxes,va="center",linespacing=1.7,fontsize=10)
        fig.suptitle(f"21组配对逐一比较  |  每组各删 {k} 日  |  第 {start//7+1} / 3 页",
                     fontweight="bold")
        save(fig,f"09_independent_pairs_{start//7+1}")


def build():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"Microsoft YaHei","svg.fonttype":"none","axes.unicode_minus":False,
                         "axes.spines.top":False,"axes.spines.right":False})
    k=CFG["primary_k"]
    result=pd.read_csv(OUT/"convergence_results.csv").query("sample=='main'")
    primary=result.query("kind=='natural' and K==@k").set_index("method")
    curves=pd.read_csv(OUT/"convergence_paths_main.csv",header=[0,1],index_col=0,parse_dates=True)
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True,sharey=True,layout="constrained")
    for ax,name in zip(axes.flat,["original",*METHODS]):
        count = 0 if name=="original" else primary.loc[name,"deleted"]
        frame=curves["original" if name=="original" else f"{name}_K{k}"]
        panel(ax,frame,f"{FRIENDLY[name]}　删 {count} 日")
        ax.tick_params(labelbottom=True)
    axes.flat[-1].axis("off")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,ncol=2,loc="center",frameon=False)
    fig.suptitle(f"七条指数累计价格收益  |  原始路径与四种自然集合，K={k}",fontweight="bold")
    save(fig,"01_common_paths")
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,layout="constrained")
    for ax,tag in zip(axes,("1pct","5pct")):
        row=result.loc[result.scenario.eq("A_"+tag)].iloc[0]
        panel(ax,curves[row.scenario],f"整体分化前 {tag[:-3]}%　删 {row.deleted} 日")
    axes[0].legend(ncol=2,frameon=False,fontsize=8)
    save(fig,"02_primary_fractions")

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True,layout="constrained")
    budget=result.query("kind=='budget'")
    for ax,k0 in zip(axes,CFG["k"]):
        for name,color in PALETTE.items():
            part=budget.query("K==@k0 and method==@name").sort_values("deleted")
            ax.plot(part.deleted,part.improvement,"o-",label=FRIENDLY[name],lw=1.2,color=color)
        ax.set_title(f"各配对关键日名单 K={k0}",loc="left")
        ax.axhline(0,color=".45",lw=.7)
        ax.set_xlabel("实际删除交易日")
        ax.grid(alpha=.18)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("全池路径RMS改善率")
    axes[0].legend(frameon=False,fontsize=8)
    save(fig,"03_budget_response")

    jac=pd.read_csv(OUT/f"jaccard_main_K{k}.csv",index_col=0)
    fig,ax=plt.subplots(figsize=(9,8),layout="constrained")
    image=ax.imshow(jac,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(len(jac)),[i.replace("_"," / ") for i in LABELS],rotation=65,ha="right",fontsize=8)
    ax.set_yticks(range(len(jac)),[i.replace("_"," / ") for i in LABELS],fontsize=8)
    fig.colorbar(image,ax=ax,shrink=.78,label="Jaccard")
    ax.set_title(f"21组配对关键日期的两两重合  |  K={k}",loc="left")
    save(fig,"04_jaccard_matrix")

    f=pd.read_csv(OUT/"frequency_distribution.csv").query("sample=='main'")
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for k0,g in f.groupby("K"):
        axes[0].plot(range(6),g.days,"o-",label=f"K={k0}",lw=1.4)
    axes[0].set_xticks(range(6),["1","2","3–4","5–9","10–20","21"])
    axes[0].set_xlabel("获票数")
    axes[0].set_ylabel("不同交易日数")
    axes[0].legend(frameon=False)
    calendar=pd.read_csv(OUT/"date_frequency.csv",parse_dates=["date"]).query("sample=='main' and K==@k")
    plot=calendar.assign(year=calendar.date.dt.year,doy=calendar.date.dt.dayofyear).pivot(
        index="year",columns="doy",values="votes").reindex(columns=range(1,367))
    im=axes[1].imshow(plot,aspect="auto",cmap="YlGnBu")
    axes[1].set_xticks(pd.date_range("2024-01-01",periods=12,freq="MS").dayofyear-1,range(1,13))
    axes[1].set_yticks(range(len(plot)),plot.index)
    axes[1].set_xlabel("月份")
    axes[1].set_title(f"逐日票数热力图  |  K={k}")
    fig.colorbar(im,ax=axes[1],label="票数")
    save(fig,"05_vote_frequency")

    o=pd.read_csv(OUT/"method_overlap.csv").query("sample=='main'")
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout="constrained")
    for ax,name in zip(axes,("U","R2","B")):
        part=o[o.other.eq(name)]
        y=np.arange(len(part))
        stack=np.zeros(len(part))
        for col,label,color in zip(("only_A","intersection","only_other"),("A独有","共同",f"{name}独有"),
                                   ("#abbecb","#287b78","#d7a077")):
            ax.barh(y,part[col],left=stack,label=label,color=color)
            stack+=part[col].to_numpy()
        ax.set_yticks(y,[f"K={i}" for i in part.K])
        ax.set_title(f"A 与 {name}")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("不同日期数")
        ax.legend(frameon=False,fontsize=8,loc="lower right")
    save(fig,"06_method_overlap")

    with np.load(OUT/"random_deletion_draws.npz") as draws:
        fig,axes=plt.subplots(2,2,figsize=(10,6),layout="constrained")
        for ax,name in zip(axes.flat,METHODS):
            row=primary.loc[name]
            values=draws["draws"][:,np.searchsorted(draws["budgets"],row.deleted),0]
            ax.hist(values,bins=45,color=PALETTE[name],alpha=.7)
            ax.axvline(row.improvement,color="#202735",lw=2,label="历史观察值")
            ax.set_title(f"{FRIENDLY[name]}　同删 {int(row.deleted)} 日")
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        axes.flat[0].legend(frameon=False)
        fig.supxlabel("全池路径RMS改善率")
        fig.supylabel("随机样本数")
        save(fig,"07_equal_random")

    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for i,length in enumerate(CFG["blocks"],1):
        with np.load(OUT/f"bootstrap_block_{length}.npz") as draws:
            position=CFG["k"].index(k)
            x=draws["effects"][:,position,:,3]
            y=draws["overlap"][:,position,:]
            for ax,values,color in zip(axes.flat,(y[:,1],y[:,3],*x.T),
                                       ("#2b736c","#815173",*(PALETTE[m] for m in METHODS))):
                data=values[np.isfinite(values)]
                b=ax.violinplot(data,positions=[i],widths=.65,showmedians=True,showextrema=False)
                b["bodies"][0].set_facecolor(color)
                b["cmedians"].set_color(color)
    for ax,title in zip(axes.flat,("A 与 R2 的 Jaccard","21组配对平均 Jaccard",
                                  *(m+" 相对同预算随机的改善差" for m in METHODS))):
        ax.set_xticks(range(1,4),[str(i) for i in CFG["blocks"]])
        ax.set_xlabel("循环区块长度（交易日）")
        ax.set_title(title,loc="left")
        ax.grid(axis="y",alpha=.16)
    for ax in axes.flat[2:]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axhline(0,color=".4",lw=.7)
    save(fig,"08_block_bootstrap")
    independent_pair_panels(k)
