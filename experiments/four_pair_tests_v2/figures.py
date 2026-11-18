"""Redraw figures from the revision's exported accounts."""
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from run import ROOT,RES,NAMES,H,S,C,U

FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,
    "font.size":10,"axes.spines.right":False,"axes.spines.top":False,"svg.fonttype":"none"})
METHOD={"one":"等额","static":"静态OLS","rolling":"滚动OLS","ewma":"EWMA","vol":"波动率比",
        "half":"固定0.5","double":"固定2"}
RULE={"units":"固定数量","daily":"每日恢复权重"}
colors=["#18588a","#ae521c","#238276","#6c5796","#9a7222","#59636c","#953b5d"]

def read(name):
    return pd.read_csv(RES/name,dtype={"a":str,"b":str})

def end(f,name):
    titles={"e1_entries":"图1　建仓日期与持有期净收益","e1_distribution":"图2　各建仓日的持有期净收益分布",
            "e2_curves":"图3　六种方向的分阶段账户净值","e3_risk":"图4　五种对冲方法的波动与市场暴露",
            "e3_curves":"图5　不同对冲方法的测试期净值","e4_curves":"图6　同指数与跨指数对冲的测试期净值"}
    f.suptitle(titles[name],fontsize=12)
    f.tight_layout()
    for ext in ("svg","png"):
        f.savefig(FIG/(name+"."+ext),dpi=260,bbox_inches="tight")
    plt.close(f)

def pair(a,b):
    return NAMES[a]+"／"+NAMES[b]

def nav(ax,g,label,color):
    dates=pd.to_datetime([g.execution.iloc[0],*g.date])
    ax.plot(dates,np.r_[1,g.net_nav.values],label=label,color=color,lw=1)
    ax.set(xlabel="日期",ylabel="成本后净值（期初=1）")
    ax.tick_params(axis="x",rotation=25)
    ax.grid(axis="y",alpha=.23)

def main():
    e=read("e1_windows.csv")
    f,ax=plt.subplots(figsize=(9,3.4))
    for k,(rule,lbl) in enumerate(RULE.items()):
        g=e[e.rule==rule]
        ax.plot(pd.to_datetime(g.entry),g.net,label=lbl,lw=1,color=colors[k])
    ax.set(xlabel="建仓日期",ylabel="252日成本后收益");ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(0,color="black",lw=.7);ax.legend();end(f,"e1_entries")
    f,ax=plt.subplots(figsize=(8,3.1))
    bins=np.linspace(e.net.min(),e.net.max(),38)
    for k,(rule,lbl) in enumerate(RULE.items()):
        ax.hist(e[e.rule==rule].net,bins=bins,label=lbl,color=colors[k],alpha=.58)
    ax.set(xlabel="252日成本后收益",ylabel="建仓窗口数（个）");ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend();end(f,"e1_distribution")
    d=read("e2_daily.csv.gz")
    f,axes=plt.subplots(3,1,figsize=(9,8.5))
    for ax,period,title in zip(axes,["train","validation","test"],["训练期","验证期","测试期"]):
        for k,((a,b),g) in enumerate(d[d.period==period].groupby(["a","b"])):
            nav(ax,g,pair(a,b),colors[k])
        ax.set_title(title);ax.tick_params(axis="x",rotation=0)
    axes[0].legend(fontsize=8,ncol=3);end(f,"e2_curves")
    s=read("phase_statistics.csv")
    t=s[(s.experiment==3)&(s.period=="test")&(s.frequency=="D")]
    f,axes=plt.subplots(2,1,figsize=(9,6))
    for ax,k,ylabel in zip(axes,["vol","beta"],["年化净收益波动率","相对沪深300的毛收益Beta"]):
        v=t.pivot(index=["a","b"],columns="method",values=k)[list(METHOD)[:5]]
        v.index=[pair(*i) for i in v.index]
        v.rename(columns=METHOD).plot.bar(ax=ax,rot=0,color=colors[:5],legend=False)
        ax.set(xlabel="多头／空头",ylabel=ylabel)
        ax.tick_params(axis="x",labelsize=8)
    axes[0].legend(ncol=5,fontsize=8);axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    end(f,"e3_risk")
    d=read("e3_daily.csv.gz")
    f,axes=plt.subplots(3,2,figsize=(10,8.4))
    for ax,((a,b),g) in zip(axes.flat,d[(d.period=="test")&(d.frequency=="D")].groupby(["a","b"])):
        for k,m in enumerate(list(METHOD)[:5]):nav(ax,g[g.method==m],METHOD[m],colors[k])
        ax.set_title(pair(a,b));ax.locator_params(axis="y",nbins=4)
    axes[0,0].legend(ncol=3,fontsize=7);end(f,"e3_curves")
    d=read("e4_daily.csv.gz")
    f,axes=plt.subplots(2,1,figsize=(9,6.2))
    for ax,b in zip(axes,(H,U)):
        g=d[(d.period=="test")&(d.b==b)]
        for k,m in enumerate(METHOD):
            if b==H and m not in ("one","half","double"):continue
            nav(ax,g[g.method==m],METHOD[m],colors[k])
        ax.set_title(pair(H,b));ax.legend(ncol=4,fontsize=8)
    end(f,"e4_curves")

if __name__=="__main__":main()
