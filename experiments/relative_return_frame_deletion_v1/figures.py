import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from data import *
from deletion import mask,ORDER
from metrics import path
F=ROOT/"figures";F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,
 "axes.spines.right":False,"axes.spines.top":False,"svg.fonttype":"none","font.size":9})
LABEL={"best":"删最佳","worst":"删最差","absolute":"删最大绝对差","symmetric":"双尾等量"}
def end(i,title,x,y,note):
    ax=plt.gca();f=ax.figure
    ax.set(xlabel=x,ylabel=y);ax.grid(axis="y",alpha=.18)
    handles,labels=ax.get_legend_handles_labels()
    if handles:ax.legend(fontsize=8,ncol=min(3,len(handles)))
    f.suptitle(f"图{i}　{title}");f.text(.09,.01,"2015-01-05 至 2026-11-17　|　"+note,fontsize=8)
    f.tight_layout(rect=(0,.055,1,.95))
    for ext in ("png","svg"):f.savefig(F/f"figure_{i:02}.{ext}",dpi=240,bbox_inches="tight")
    plt.close(f)
def ax():return plt.subplots(figsize=(8.7,3.65))[1]
def make():
    orig=path(np.ones(N,bool));det=read("deterministic_deletion");s=read("random_deletion_summary");s=s[s.reps==100000]
    p=read("daily_relative_returns");annual=read("annual_contributions");yr=annual[annual.rule=="original"]
    q=np.load(OUT/"terminal_Q.npz")
    g=ax();g.plot(P.index,orig.wealth_a,label="上证综指");g.plot(P.index,orig.wealth_b,label="沪深300")
    end(1,"原始价格指数归一化净值","日期","期初=1","实际日收盘价，无删帧。")
    g=ax();g.plot(P.index,orig.D,lw=1);g.axhline(0,color="grey",lw=.6)
    end(2,"沪深300相对上证综指的累计对数差","日期","累计对数收益差","原始路径；正值代表沪深300相对领先。")
    g=ax();g.hist(D*1e4,bins=75,color="#487697")
    for j,c in [(ORDER["best"][0],"tab:red"),(ORDER["worst"][0],"tab:green")]:
        g.axvline(D[j]*1e4,color=c,label=f"{T[j]:%Y-%m-%d}  {D[j]*1e4:+.1f} bp")
    end(3,"每日相对对数收益分布","日对数差（bp）","交易日数","标记相对差的两个极值日期。")
    g=ax()
    for k in (0,1,2,5,10,20,50):
        z=path(mask("best",k) if k else np.ones(N,bool));g.plot(P.index,z.D,lw=.9,label=f"删{k}日")
    end(4,"最有利交易日依次删除后的路径","原始日历日期","累计对数差","两指数同步删除同一天的收益，删去日在日历轴持平。")
    g=ax()
    for rule in ("best","worst","absolute"):
        z=path(mask(rule,20));g.plot(P.index,z.D,lw=.9,label=LABEL[rule])
    end(5,"相同20日删除预算的三种排序","原始日历日期","累计对数差","三条路径均为事后合成。")
    g=ax()
    for rule,sub in det[det.k.gt(0)].groupby("rule"):
        g.plot(sub.k/N*100,sub.D,".-",label=LABEL[rule])
    g.axhline(0,color="grey",lw=.5);end(6,"删除强度与终值","实际删除比例（%）","累计对数差","用对数差统一比较正负两端，完整财富比见统计表。")
    g=ax()
    for prop in (.01,.05,.10,.20):
        k=count(prop);v=q["uniform_"+str(k)]*100
        g.hist(v,bins=70,density=True,histtype="step",label=f"删{prop:.0%}：中位{np.median(v):.2f}%")
        g.plot(np.quantile(v,[.05,.95]),[0,0],"|",ms=9)
    end(7,"随机删帧终点财富优势分布","Q（%）","概率密度","每档10万次，横轴刻线为各档5%与95%分位。")
    g=ax();g.plot(s.fraction*100,s.p_positive*100,"o-");g.fill_between(s.fraction*100,(1-s.p_flip_hi)*100,(1-s.p_flip_lo)*100,alpha=.3)
    end(8,"均匀删帧后仍保持正优势的频率","实际删除比例（%）","Q>0（%）","阴影为95% Wilson蒙特卡洛区间。")
    g=ax();g.plot(s.fraction*100,s.r2_p50,label="随机中位数");g.fill_between(s.fraction*100,s.r2_p05,s.r2_p95,alpha=.2,label="随机5%–95%")
    for rule in ("best","worst","absolute"):
        v=det[(det.rule==rule)&det.k.isin(s.k)];g.plot(v.k/N*100,v.r2,".-",label=LABEL[rule])
    end(9,"保留日收益回归的拟合度","实际删除比例（%）","含截距回归 R²","拟合对象为每日对数收益。")
    g=ax();k=count(.01);dates=read("first_draw_dates");sub=dates[(dates.rule=="uniform")&(dates.k==k)]
    random_keep=~T.isin(pd.to_datetime(sub.date))
    for name,keep,color in [("原始",np.ones(N,bool),"#555555"),("删最佳1%",mask("best",k),"#c65d30"),("随机删1%",random_keep,"#20857b")]:
        z=path(keep)
        for col,ls,lbl in [("wealth_a","--","上证"),("wealth_b","-","沪深300")]:
            g.plot(P.index,z[col],ls=ls,color=color,lw=.9,label=name+" "+lbl)
        if not keep.all():g.plot(T[~keep],np.full((~keep).sum(),.70 if name.startswith("随机") else .67),"|",color=color,ms=3)
    end(10,"相同日历上的双指数合成路径","日期","期初=1","随机示例为预定首抽；底部刻线标示两种规则删去的日期。")
    g=ax();k=count(.10);rules=["uniform","year","month","block5","block10","block20","block60"]
    g.boxplot([q[r+"_"+str(k)]*100 for r in rules],tick_labels=["均匀","年度","月度","5日块","10日块","20日块","60日块"],whis=(5,95),showfliers=False)
    end(11,"同删288日的抽样机制比较","删除机制","终点Q（%）","箱为四分位区间，须为5%–95%。")
    g=ax();g.bar(yr.year,yr.pos,label="正贡献");g.bar(yr.year,yr.neg,label="负贡献")
    top=annual[(annual.rule=="best")&(annual.k==20)].set_index("year")
    for r in yr.itertuples():g.text(r.year,r.pos+.01,str(top.loc[r.year,"removed"]),ha="center",fontsize=8)
    end(12,"各年贡献与最有利20日的分布","年度","对数收益贡献","柱顶数字为该年进入最佳20日的数量。")
    g=ax();o=read("original_metrics").iloc[0];g.bar(["正日合计","负日合计","相抵后净额"],[o.pos,o.neg,o.D],color=["#377d78","#b55738","#34495e"])
    for i,v in enumerate([o.pos,o.neg,o.D]):g.text(i,v,f"{v:+.4f}",ha="center",va="bottom" if v>=0 else "top")
    end(13,"全样本正负贡献抵消","贡献类别","对数收益单位","合计对数贡献可相加，不能将两端指数化后再相加。")
    g=ax();st=read("statistical_tests");v=st[(st.group=="full")&st.lo.notna()]
    for i,r in enumerate(v.itertuples()):g.plot([r.lo*1e4,r.hi*1e4],[i,i],lw=2);g.scatter(r.estimate*1e4,i,color="black",s=12)
    g.set_yticks(range(len(v)),[f"{r.method} {r.lag}" for r in v.itertuples()]);g.axvline(0,color="grey",ls="--")
    end(14,"平均日相对收益的95%区间","均值（bp／日）","标准误或区块规则","原始收益序列；与固定历史删帧分布分开。")
    g=ax();v=read("convergence")
    for k in s.k:
        z=v[v.k==k];g.plot(z.n,z.p_positive*100,".-",label=f"{k}日")
    g.set_xscale("log");end(15,"蒙特卡洛正优势频率收敛","累计抽样次数（对数轴）","Q>0（%）","七档均匀无放回删除，最终各10万次。")
