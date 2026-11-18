import warnings

from research_common import *
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.multitest import multipletests

LABELS = ["advance", "sideways", "decline", "transition"]


def classify(close, length=120, trend=.10, flat=.05):
    prior = close.shift(1)
    change = close.pct_change(length, fill_method=None).shift(1)
    average = close.rolling(length).mean().shift(1)
    state = pd.Series("transition", index=close.index)
    state[change.between(-flat, flat)] = "sideways"
    state[(change > trend) & (prior > average)] = "advance"
    state[(change < -trend) & (prior < average)] = "decline"
    return state.where(change.notna()), change, average


def markov(r):
    """Refit annually on the preceding 756 sessions and predict one day ahead."""
    audit, result = [], []
    for year in range(r.index[0].year+3, r.index[-1].year+1):
        train = r[r.index.year < year].tail(756)
        test = r[r.index.year == year]
        try:
            def model(y):
                return MarkovRegression(y.to_numpy()*100, k_regimes=3, trend="c", switching_variance=False)
            with warnings.catch_warnings(record=True) as caught:
                m = model(train)
                f = m.fit(disp=False, maxiter=500, search_reps=6, rng=SEED+year)
                if not f.mle_retvals["converged"]:
                    f = m.fit(disp=False, maxiter=500, search_reps=6, rng=SEED+year+100)
                if not f.mle_retvals["converged"]:
                    raise ValueError("optimizer did not converge after seeded retry")
                means = [f.params[m.param_names.index(f"const[{i}]")] for i in range(3)]
                order = np.argsort(means)
                probabilities = model(pd.concat([train,test])).filter(f.params).predicted_marginal_probabilities[-len(test):]
                labels = np.array(["decline","sideways","advance"])[np.argsort(order)]
                result.append(pd.DataFrame({"state": labels[probabilities.argmax(axis=1)],
                                            **{f"p_{labels[i]}": probabilities[:, i] for i in range(3)}}, index=test.index))
                audit.append({"year": year, "fit_start": train.index.min(), "fit_end":train.index.max(),
                              "n_train":len(train), "n_test":len(test), "means_pct":str([float(x) for x in means]),
                              "optimizer_warnings":len(caught), "status":"ok"})
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            audit.append({"year":year,"status":str(e),"n_test":len(test)})
    csv(pd.DataFrame(audit), "e2_markov_fit")
    return pd.concat(result) if result else pd.DataFrame(columns=["state"])


def metrics(r, labels, spec):
    rows = []
    for state, d in r.groupby(labels):
        for index in TARGETS:
            y, b = d[index], d["000001"]
            up, down = b > 0, b < 0
            rows.append(dict(spec=spec, state=state, index=index, **window(y.index), mean=y.mean(),
                             cumulative=(1+y).prod()-1, vol_ann=y.std()*np.sqrt(252),
                             beta=regression_row(y,b,r.index)["beta"] if len(y)>10 and b.std()>0 else np.nan,
                             upside_capture=y[up].mean()/b[up].mean(), downside_capture=y[down].mean()/b[down].mean(),
                             stitched_max_drawdown=drawdown(y)))
    table = pd.DataFrame(rows)
    table["rank_by_mean"] = table.groupby(["spec","state"])["mean"].rank(ascending=False, method="min")
    return table


def main():
    close, r, _ = market()
    labels, change, avg = classify(close["000001"])
    states = pd.DataFrame({"state":labels,"trailing_return_tminus1":change,"ma_tminus1":avg,
                           "close_tminus1":close["000001"].shift(1)})
    specs = {"main": labels}
    for length, trend, flat in [(60,.10,.05),(252,.10,.05),(120,.08,.04),(120,.12,.06)]:
        tag = f"w{length}_trend{int(trend*100)}_flat{int(flat*100)}"
        specs[tag] = states[tag] = classify(close["000001"],length,trend,flat)[0]
    alternative = markov(r["000001"])
    if len(alternative):
        specs["markov"] = alternative["state"]
        states = states.join(alternative.add_prefix("markov_"))
    csv(states, "e2_states", True)
    table = pd.concat([metrics(r, s, name) for name,s in specs.items()], ignore_index=True)
    csv(table,"e2_conditional")
    main_r=r.loc[labels.dropna().index]
    flags=pd.get_dummies(labels.loc[main_r.index],dtype=float).drop(columns="transition")
    inference=[]
    for code in TARGETS:
        f=fit(main_r[code],flags)
        for state,b,p,ci in zip(flags.columns,f.params[1:],f.pvalues[1:],f.conf_int()[1:]):
            inference.append(dict(index=code,state=state,reference="transition",**window(main_r.index),
                                  mean_diff=b,p=p,lo=ci[0],hi=ci[1]))
    tests=pd.DataFrame(inference)
    tests["p_holm"]=multipletests(tests.p,method="holm")[1]
    csv(tests,"e2_mean_differences")
    l = labels.reindex(r.index).dropna()
    episodes = []
    for group, s in l.groupby(l.ne(l.shift()).cumsum()):
        for code in TARGETS:
            y = r.loc[s.index,code]
            episodes.append(dict(episode=group,state=s.iloc[0],index=code,**window(s.index),
                                 cumulative=(1+y).prod()-1,max_drawdown=drawdown(y)))
    csv(pd.DataFrame(episodes),"e2_episodes")
    fig, ax = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    d = table[table.spec=="main"]
    for a, col, mult, label in zip(ax,["mean","vol_ann"],[10000,100],["Mean daily return (bp)","Annualized volatility (%)"]):
        p = d.pivot(index="state",columns="index",values=col).reindex(LABELS)[TARGETS]
        (p.rename(columns=NAMES)*mult).plot.bar(ax=a,rot=0,legend=a is ax[0])
        a.set_ylabel(label)
        a.set_xlabel("")
    figure(fig,"e2_conditional")
    fig,ax=plt.subplots(figsize=(11,3.4))
    ax.plot(close.index,close["000001"],c="#30383e",lw=.8)
    for state,color in zip(LABELS,["#2b815e","#5284ae","#bb554c","#888888"]):
        x=close.loc[labels.eq(state),"000001"]
        ax.scatter(x.index,x,s=6,label=state,c=color)
    ax.legend(ncol=4)
    ax.set_ylabel("SSE Composite close")
    figure(fig,"e2_states")


if __name__ == "__main__":
    main()
