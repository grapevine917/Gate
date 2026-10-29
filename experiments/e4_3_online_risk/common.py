import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];E1=R.parent/'e1_similarity';E2=R.parent/'e2_state_reuse';E3=R.parent/'e3_reuse';E41=R.parent/'e4_1_redundant_keep';E42=R.parent/'e4_2_history_skip'
SEED=20261021
FEATURES=['retain_logit']+['state_'+str(i) for i in range(17)]+['history_has_skip','log_history_total','log_history_skip','history_skip_fraction','log_nearest_skip_state','unused_query_distance','unused_query_at_state','state_close','unused_query_close','unused_joint_close']
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
 s=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+x;print(s,flush=True)
 with (R/'experiment.log').open('a') as f:f.write(s+'\n')
