import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];E1=R.parent/'e1_similarity';E2=R.parent/'e2_state_reuse';E3=R.parent/'e3_reuse';E4=R.parent/'e4_1_redundant_keep'
SEED=20261020
FEATURES=['retain_logit']+['state_'+str(i) for i in range(17)]+['history_has_skip','log_history_total','log_history_skip','history_skip_fraction','log_nearest_skip_state','log_nearest_skip_query','log_query_at_state_nn','state_close','query_close','joint_close']
METHODS=['M0','M1','M2','M3','M4','M5','M6']
COVERAGES=[.001,.0025,.005,.01,.02,.05,.10,.20]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def csvsave(p,rows):
 if not rows:return
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def log(x):
 line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+x;print(line,flush=True)
 with (R/'experiment.log').open('a') as f:f.write(line+'\n')
def tie_hash(q,n):
 # Exact 32-bit arithmetic, shared with native implementation.
 x=(np.asarray(q,dtype=np.uint64)*1000003+np.asarray(n,dtype=np.uint64)+SEED)&0xffffffff
 x=((x^(x>>16))*0x7feb352d)&0xffffffff;x=((x^(x>>15))*0x846ca68b)&0xffffffff;x=x^(x>>16)
 return x.astype(np.float64)/4294967296.
def load():return dict(np.load(R/'features.npz'))
def evaluate_scores(x,models):
 scores=np.zeros((len(x),7));scores[:,0]=-x[:,0];scores[:,1]=x[:,18];scores[:,2]=x[:,21]
 for k in range(3,7):scores[:,k]=x@np.array(models[k]['weights'])+models[k]['intercept']
 return scores
