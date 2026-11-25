from hashlib import sha256
import sqlite3
import pandas as pd
import numpy as np
from core import BASE,ROOT,OUT,CFG,load_data
from analysis import generate,oos
import simulation
import residual_check
from presentation import build

def audit(r):
    records=[]
    with sqlite3.connect((ROOT/'data/a_share_indices.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        query="select distinct index_code,raw_file,raw_sha256 from index_daily join ingest_batch using(batch_id) where index_code in ('000300','000001','000985')"
        for code,path,digest in db.execute(query):
            f=ROOT/path
            raw=pd.read_csv(f,index_col='date',parse_dates=True)
            np.testing.assert_allclose(np.log(raw.close).diff().dropna().values,r[{'000300':'a','000001':'b','000985':'market'}[code]],atol=1e-12)
            assert raw.index[1:].equals(r.index)
            records.append([code,path,digest,sha256(f.read_bytes()).hexdigest(),len(raw)])
    pd.DataFrame(records,columns=['index','raw_file','ingest_hash','current_hash','closes']).to_csv(OUT/'source_audit.csv',index=False)

def main():
    r=load_data()
    audit(r)
    r.to_csv(OUT/'inputs.csv')
    generate(r);oos(r)
    simulation.run(r);residual_check.run(r)
    build(r)

if __name__=='__main__':main()
