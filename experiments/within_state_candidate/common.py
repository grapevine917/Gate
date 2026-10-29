import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];S=R.parent/'sift1m';DATA=S/'training_data_l100';FG=R.parent/'feature_group_ablation';E1=R.parent/'e1_similarity';E2=R.parent/'e2_state_reuse';SEEDS=[42,20261022,20261023];GROUPS={'Global':[1,2,4,5,6,7,8,11,12,15,16],'Candidate':[0,3,9,10,13,14],'Full17':list(range(17))};SEED=20261027
CPUS=json.loads((FG/'PROTOCOL.json').read_text())['CPU_affinity']
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def csvsave(p,rows):
 if not rows:return
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def log(x):
 s=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+x;print(s,flush=True)
 with (R/'experiment.log').open('a') as f:f.write(s+'\n')
def predict(bundle,states):
 x=np.array(states[:,bundle['feature_indices']],np.float32,order='C');x-=bundle['mean'];x/=bundle['scale'];return np.concatenate([bundle['model'].predict_proba(x[i:i+8192])[:,1] for i in range(0,len(x),8192)]).astype(float)
