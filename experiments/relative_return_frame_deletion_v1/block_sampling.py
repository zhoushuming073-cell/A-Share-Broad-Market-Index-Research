from numba import njit
import numpy as np

@njit(cache=True)
def choose(mark,start,n,k):
    # Floyd sampling. A collision transfers the draw to the previously unavailable j.
    for j in range(n-k,n):
        t=start+np.random.randint(j+1)
        mark[start+j if mark[t] else t]=True

@njit(cache=True)
def fill(mark,k,mode,bounds,length):
    n=len(mark)
    if mode==0:
        choose(mark,0,n,k)
    elif mode==1:
        u=np.random.random()
        for j in range(1,len(bounds)):
            a,b=bounds[j-1],bounds[j]
            quota=int(np.floor(b*k/n+u))-int(np.floor(a*k/n+u))
            choose(mark,a,b-a,quota)
    else:
        # Compress each non-overlapping block to one token, choose token positions,
        # then expand. The partial block, if any, gets a uniformly random token.
        blocks=(k+length-1)//length
        tokens=np.zeros(n-k+blocks,np.bool_);choose(tokens,0,len(tokens),blocks)
        partial=np.random.randint(blocks);seen=0;pos=0
        for j in range(len(tokens)):
            if tokens[j]:
                size=k-length*(blocks-1) if seen==partial else length
                mark[pos:pos+size]=True;pos+=size;seen+=1
            else:pos+=1
