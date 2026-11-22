import pandas as pd
import numpy as np
from paths import FIG,plt
from data_validation import OUT,CODES,INDICES

def run():
    frame=pd.read_csv(OUT/"pairwise_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout="constrained")
    for ax,metric,title in zip(axes,["kstar","C5"],["删除多少天足以翻转领先","前五日占全部正贡献（%）"]):
        grid=np.full((9,9),np.nan)
        for row in frame.itertuples():
            i,j=map(CODES.index,row.pair.split("_"))
            grid[i,j]=grid[j,i]=getattr(row,metric)*(100 if metric=="C5" else 1)
        ax.imshow(grid,cmap="Blues",vmin=0,vmax=np.nanmax(grid)*1.5)
        for i,j in np.argwhere(np.isfinite(grid)):
            ax.text(j,i,f"{grid[i,j]:.1f}" if metric=="C5" else f"{grid[i,j]:.0f}",ha="center",va="center",size=8)
        ax.set_xticks(range(9),[INDICES[c][0] for c in CODES],rotation=65,ha="right",fontsize=8)
        ax.set_yticks(range(9),[INDICES[c][0] for c in CODES],fontsize=8)
        ax.set_title(title)
    for ext in ["png","svg"]: fig.savefig(FIG/f"attribution.{ext}",dpi=190)
    plt.close(fig)
