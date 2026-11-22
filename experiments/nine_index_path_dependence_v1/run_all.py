from data_validation import OUT,audit
from pairwise_metrics import describe
from key_date_overlap import overlap
import multivariate_bootstrap
import prediction_test
import robustness
import paths
import report_figures

if __name__=="__main__":
    r=audit()
    obs,keys=describe(r)
    obs.to_csv(OUT/"pairwise_summary.csv",index=False)
    keys.to_csv(OUT/"key_days.csv",index=False)
    shared=overlap(keys,r.index,save=True)
    multivariate_bootstrap.run(r,obs,shared)
    robustness.run(r)
    prediction_test.run(r)
    paths.run(r)
    report_figures.run()
