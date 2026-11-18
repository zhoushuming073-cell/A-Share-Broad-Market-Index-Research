"""Save official CNI/CSI price-index levels for the snapshot replication dates."""
from research_common import *


def main():
    import akshare as ak
    rows=[]
    for code in CODES:
        is_cni=code=="399311"
        f=ak.index_hist_cni if is_cni else ak.stock_zh_index_hist_csindex
        result=f(code,"20261030","20261102")
        d=pd.DataFrame({"date":pd.to_datetime(result.iloc[:,0]),"close":result.iloc[:,4 if is_cni else 9]})
        d=d[d.date.isin(pd.to_datetime(["2026-10-30","2026-11-02"]))]
        if len(d)!=2 or not d.date.is_unique or not d.close.gt(0).all():
            raise ValueError(f"Invalid official dates or levels for {code}")
        rows.append(d.assign(index_code=code,source="CNI" if is_cni else "CSI",api=f.__name__))
    pd.concat(rows,ignore_index=True).to_csv(INPUTS/"official_index_snapshot.csv",index=False,encoding="utf-8-sig")


if __name__=="__main__":
    main()
