import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];S=R.parent/'sift1m';DATA=S/'training_data_l100';E1=R.parent/'e1_similarity'
GROUPS={'Global':[1,2,4,5,6,7,8,11,12,15,16],'Candidate':[0,3,9,10,13,14],'Full17':list(range(17))};SEEDS=[42,20261022,20261023];RMINS=[99.60,99.44,99.20,99.00];PRIMARY=99.44
FNR=[0.,.0001,.0005,.001,.0025,.005,.01,.02,.05,.1,.2]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
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
def cpu_set():
 out=[];seen=set()
 for c in sorted(os.sched_getaffinity(0)):
  p=Path(f'/sys/devices/system/cpu/cpu{c}/topology')
  try:key=((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
  except OSError:key=c
  if key not in seen:out.append(c);seen.add(key)
  if len(out)==4:break
 return out
CPUS=cpu_set()
def predict(model,x):return np.concatenate([model.predict_proba(x[i:i+8192])[:,1] for i in range(0,len(x),8192)]).astype(float)
