from data import np
from optuna._hypervolume import compute_hypervolume
def front(values):
    v=np.asarray(values);out=[]
    for i in np.argsort(-v[:,0],kind="stable"):
        x=v[i]
        if any(np.all(v[j]>=x) and np.any(v[j]>x) for j in out):continue
        out=[j for j in out if not(np.all(x>=v[j]) and np.any(x>v[j]))]
        out.append(i)
    return out
def volume(values):
    pts=-np.asarray(values);ref=np.array([.5,.75,6.])
    pts=pts[(pts<ref).all(axis=1)]
    return compute_hypervolume(pts,ref) if len(pts) else 0.
