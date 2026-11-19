from data import *
from metrics import describe
from deletion import mask,flip
import json,sqlite3
def verify():
    db=(ROOT.parents[1]/"data/a_share_indices.sqlite").resolve().as_uri()+"?mode=ro"
    with sqlite3.connect(db,uri=True) as con:
        raw=pd.read_sql("select trade_date,index_code,close from index_daily where provider='eastmoney' and index_code in ('000001','000300') and trade_date between '2015-01-05' and '2026-11-17'",con)
    p=raw.pivot(index="trade_date",columns="index_code",values="close")
    np.testing.assert_array_equal(P.index,pd.to_datetime(p.index));np.testing.assert_allclose(P,p)
    d=read("deterministic_deletion")
    ratios=(P/P.shift()).iloc[1:].values
    for r in d.itertuples():
        keep=mask(r.rule,r.k) if r.k else np.ones(N,bool)
        prod=ratios[keep].prod(axis=0)
        np.testing.assert_allclose([prod[0],prod[1],prod[1]/prod[0]-1],[r.wealth_a,r.wealth_b,r.Q],atol=3e-12)
        np.testing.assert_allclose(D.sum()-sum(D[~keep]),r.D,atol=1e-12)
    ana=read("analytic_checks")
    assert ana.z_mean.abs().max()<5 and abs(ana.observed_sd/ana.theory_sd-1).max()<.07
    s=read("random_deletion_summary");rep=read("replica_summary")
    compared=s.merge(rep,on="k",suffixes=("_a","_b"))
    z=(compared.D_mean_a-compared.D_mean_b)/np.hypot(compared.D_mcse_a,compared.D_mcse_b)
    assert abs(z).max()<5
    groups=pd.concat([s,rep,read("stratified_deletion_summary"),read("block_deletion_summary")])
    dates=read("first_draw_dates")
    for (method,k),g in dates.groupby(["rule","k"]):
        assert len(g)==g.date.nunique()==k
        keep=~T.isin(pd.to_datetime(g.date))
        if method in ("month","year"):
            period=T.to_period("M" if method=="month" else "Y")
            for key in period.unique():
                idx=period==key;assert abs(np.sum(idx&~keep)-k*idx.sum()/N)<1+1e-10
        if method.startswith("block"):
            lengths=np.diff(np.where(np.diff(np.r_[False,~keep,False].astype(int)))[0])[::2]
            L=int(method[5:]);assert sum(lengths)==k and sum(lengths%L)==k%L
    trials=int(groups.reps.sum())
    core=s[s.k.isin([count(p) for p in CFG["proportions"]])]
    assert len(core)==7 and (core.reps==100000).all()
    result=dict(status="passed",input_sha256=hashlib.sha256(FILE.read_bytes()).hexdigest(),calendar=N,core_uniform=int(core.reps.sum()),
        total_simulations=trials,analytic_max_abs_mean_z=ana.z_mean.abs().max(),analytic_max_relative_sd_error=abs(ana.observed_sd/ana.theory_sd-1).max(),
        independent_seed_max_abs_z=abs(z).max(),cases_verified=len(d),first_draw_masks_verified=len(groups),flip_days=flip())
    (OUT/"verification.json").write_text(json.dumps(result,indent=2))
    return result
