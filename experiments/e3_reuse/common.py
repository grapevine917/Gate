import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,datetime,hashlib,csv,subprocess
import numpy as np
R=Path(__file__).resolve().parent;LG=R.parents[1];E1=R.parent/'e1_similarity';E2=R.parent/'e2_state_reuse';S=R.parent/'sift1m';SEED=20261020
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def log(msg):
 line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+msg;print(line,flush=True)
 with (R/'experiment.log').open('a') as f:f.write(line+'\n')
def csvsave(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadcsv(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True))
def run(cmd,p):
 dump(p.with_suffix('.command.json'),{'argv':list(map(str,cmd)),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'threads':4,'MKL_NUM_THREADS':1})
 with p.with_suffix('.log').open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
def summary(a,baseline=None):
 n=a['eligible'].sum();b=baseline['actual_calls'].sum() if baseline is not None else n
 return {'recall_pct':float(a['hits'].mean()*10),'reuse_rate':float(a['reused'].sum()/n) if n else 0.,'model_saving':float(1-a['actual_calls'].sum()/b) if b else 0.,'actual_calls':int(a['actual_calls'].sum()),'eligible':int(n),'reused':int(a['reused'].sum()),'false_skip_rate':float(a['false_skip'].sum()/a['fresh_keep'].sum()) if a['fresh_keep'].sum() else None,'false_skip_given_reuse':float(a['false_skip'].sum()/a['reused'].sum()) if a['reused'].sum() else None,'reads':float(a['reads'].mean()),'expansions':float(a['expansions'].mean()),'latency_us':float(a['total_us'].mean()),'p95_us':float(np.percentile(a['total_us'],95)),'delta_recall_pp':float((a['hits'].mean()-baseline['hits'].mean())*10) if baseline is not None else 0.}
