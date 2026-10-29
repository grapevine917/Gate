import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,subprocess,time,csv,math,random,shutil
import numpy as np
D=Path(__file__).resolve().parent;ROOT=D.parents[1];SEED=20261102
CATALOG={
 'sift1m':dict(label='SIFT1M',parent='main_comparison',index='experiments/sift1m/index',query_ext='fbin',vectors='data/sift/sift_base.fbin',N=1000000,dim=128,LS=[15,20,30,50,75,100,160,300],calibration='experiments/sift1m_p0a_L100_train_compare/data/calibration.prune.bin'),
 'deep1m':dict(label='DEEP1M',parent='deep_p0',index='experiments/deep1m/index',query_ext='fbin',vectors='data/deep1M_base.fbin',N=1000000,dim=96,LS=[15,20,30,50,75,100,160,300],calibration='experiments/deep_p0/data/validation.prune.bin'),
 'sift100m':dict(label='SIFT100M',parent='sift100m_p0',index='experiments/sift100m_p0/index/ann',query_ext='u8bin',vectors='experiments/sift100m_p0/data/vectors.cvec',N=100000000,dim=128,LS=[15,30,50,100,160,300,500,1000],calibration='experiments/sift100m_p0/data/validation.prune.bin')}
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def catalog(ds):
 c=CATALOG[ds].copy();c.update(out=D/ds,parent_dir=D.parent/c['parent'],index_path=ROOT/c['index'],vector_path=ROOT/c['vectors']);return c
def frontier(rows,metric):
 out=[];best=float('inf')
 for r in sorted(rows,key=lambda r:(-r['total_hits'],-r[metric] if metric=='qps' else r[metric],r['key'])):
  cost=-r[metric] if metric=='qps' else r[metric]
  if cost<best:out.append(r);best=cost
 return sorted(out,key=lambda r:(r['total_hits'],r['key']))
def aggregate(runs,configs):
 out=[]
 for c in configs:
  rs=[r for r in runs if r['key']==c['key']];assert rs
  r=rs[0].copy();r.pop('repeat');r['repeats']=len(rs)
  for k in ['ids_sha256','total_hits','physical_bytes','reads','expansions','skips','model_calls','et_calls','et_stop_fraction','io_rounds']:assert len({a[k] for a in rs})==1,(c['key'],k)
  for k in ['p50_ms','p95_ms','p99_ms','mean_ms','qps','io_ms']:r[k]=float(np.median([a[k] for a in rs]))
  r['p95_min_ms']=min(a['p95_ms'] for a in rs);r['p95_max_ms']=max(a['p95_ms'] for a in rs);out.append(r)
 return out
