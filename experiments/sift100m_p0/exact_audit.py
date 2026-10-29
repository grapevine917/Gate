"""Global exact top100 distance audit on eight query IDs fixed before searching."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time
import numpy as np
R=Path(__file__).resolve().parent;D=R.parents[1]/'data/sift100m'
if (R/'EXACT_AUDIT_COMPLETE.json').exists():raise SystemExit(0)
base=np.memmap(D/'base.100M.u8bin','u1','r',offset=8,shape=(100000000,128));queries=np.fromfile(D/'query.public.10K.u8bin','u1',offset=8).reshape(10000,128);qi=np.linspace(0,9999,8,dtype=int);q=queries[qi].astype('f8');best=np.full((8,100),np.inf);qnorm=(q*q).sum(1);start_time=time.time()
for start in range(0,len(base),65536):
 v=base[start:start+65536].astype('f8');dist=(v*v).sum(1)[:,None]+qnorm[None,:]-2*v@q.T
 for j in range(8):
  ix=np.argpartition(dist[:,j],99)[:100];best[j]=np.partition(np.concatenate([best[j],dist[ix,j]]),99)[:100]
 if start%(65536*128)==0:print('EXACT AUDIT',start,len(base),flush=True)
with (D/'groundtruth.public.10K.ibin').open('rb') as f:f.seek(8+10000*100*4);gt=np.fromfile(f,'<f4').reshape(10000,100)[qi]
np.testing.assert_array_equal(np.sort(best,axis=1),gt)
info=dict(passed=True,queries=qi.tolist(),all_base_vectors=100000000,neighbors=100,computation='blocked float64 integer squared L2; compare complete top100 distance multisets (tie ID may differ)',seconds=time.time()-start_time)
(R/'EXACT_AUDIT_COMPLETE.json').write_text(json.dumps(info,indent=2)+'\n');print(info,flush=True)
