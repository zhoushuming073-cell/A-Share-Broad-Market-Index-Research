from functools import lru_cache
from data import np,pd,R
@lru_cache(None)
def features(k):
    x,y=R.iloc[:,0],R.iloc[:,1]
    spread=x-y
    market=(x+y)/2
    cov=x.rolling(k).cov(y)
    vx=x.rolling(k).var();vy=y.rolling(k).var()
    # Scale directional sums by realized standard deviation on the same historical window.
    def strength(s):
        return s.rolling(k).mean()*np.sqrt(k)/s.rolling(k).std()
    ew=y.ewm(span=k,adjust=False,min_periods=k)
    values=dict(momentum=strength(spread),trend=strength(market),
        shock=market.rolling(5).std()/market.rolling(k).std(),
        vol=spread.rolling(k).std()*np.sqrt(252)/2,
        rolling=cov/vy,ewma=ew.cov(x)/ew.var(),ratio=np.sqrt(vx/vy),
        minimum=(vx+cov)/(vy+cov),beta_match=(vx+cov)/(vy+cov),
        static=x.iloc[:120].cov(y.iloc[:120])/vy.iloc[119])
    # beta-match against the equal-weight pair has the same solution as gross-constrained minimum variance.
    frame=pd.DataFrame(values,index=R.index);frame.iloc[:119,frame.columns.get_loc("static")]=np.nan
    return frame.shift(2)
