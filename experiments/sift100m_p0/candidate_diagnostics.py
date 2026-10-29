"""Candidate diagnostics in a short-lived process; release arrays before online search."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,subprocess
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
R=Path(__file__).resolve().parent
SEED=json.loads((R/'PROTOCOL.json').read_text())['seed']
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
if not (R/'native_score_verification.json').exists():
 output=subprocess.check_output([str(R/'native/verify'),str(R/'models'),str(R/'data/vectors.cvec')]);v=json.loads(output.decode().strip().splitlines()[-1]);assert all(v[k]<2e-5 for k in ['LR','XGB','StateMLP','QV_MLP','ET']);dump(R/'native_score_verification.json',v)
risks=[0,.001,.0025,.005,.01,.02,.05,.1];thresholds={};prediction=R/'prediction';prediction.mkdir(exist_ok=True);prediction_metrics=[]
for split in ['validation','test']:
 a=np.fromfile(R/'data'/f'{split}.prune.bin','<f4').reshape(-1,20);mask=a[:,13]==0;miss=a[mask];scores_input=prediction/f'{split}.bin'
 with (R/'data'/f'{split}.fbin').open('rb') as f:n,d=np.fromfile(f,'<u4',2);q=np.fromfile(f,'<f4').reshape(n,d)
 nodes=np.fromfile(R/'data'/f'{split}.prune_nodes.u32','<u4');assert len(nodes)==len(a);raw=np.empty((len(miss),147),'<f4');raw[:,:128]=q[miss[:,0].astype(int)];raw.view('<u4')[:,128]=nodes[mask];raw[:,129:146]=miss[:,3:];raw[:,146]=miss[:,2];raw.tofile(scores_input)
 for model in ['l100','lr','xgb','state','qv']:
  out=prediction/f'{split}_{model}.scores';bench=prediction/f'{split}_{model}.bench.json'
  if not out.exists():subprocess.run([str(R/'native/score'),str(R/'models'/f'{model}.bin'),str(R/'data/vectors.cvec'),str(scores_input),str(out),str(bench)],check=True)
  s=np.fromfile(out,'<f4');assert len(s)==len(miss) and np.isfinite(s).all();y=miss[:,2]
  prediction_metrics.append(dict(split=split,model=model,samples=len(s),retain_auc=float(roc_auc_score(y,s)),safe_skip_ap=float(average_precision_score(1-y,1-s)),logloss=float(log_loss(y,np.clip(s,1e-7,1-1e-7))),native_ns=float(np.median(json.loads(bench.read_text())['ns_per_score'])),model_bytes=(R/'models'/f'{model}.bin').stat().st_size))
  if split=='validation':
   full=np.ones(len(a),'f4');full[mask]=s;positive=np.sort(full[a[:,2]==1]);points=[]
   for risk in risks:
    allowed=int(np.floor(risk*len(positive)));tau=0. if risk==0 else float(positive[allowed]);false=int(np.sum((a[:,2]==1)&(full<tau)));assert false<=allowed
    points.append(dict(risk=risk,tau=tau,false_skips=false,retain_count=len(positive),cache_hits_kept=True))
   thresholds[model]=points
dump(R/'prediction_metrics.json',prediction_metrics);dump(R/'threshold_calibration.json',thresholds)
# Independently check the separate L100 export on held-out native candidate scores.
if not (R/'l100_native_verification.json').exists():
 import torch
 from torch import nn
 torch.set_num_threads(1);checkpoint=torch.load(R/'models/l100.pt',weights_only=False);net=nn.Sequential(nn.Linear(17,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1));net.load_state_dict({k.removeprefix('seq.'):v for k,v in checkpoint['weights'].items()});net.eval();a=np.fromfile(R/'data/validation.prune.bin','<f4').reshape(-1,20);a=a[a[:,13]==0];ix=np.random.default_rng(SEED).choice(len(a),2000,replace=False)
 with torch.no_grad():expected=torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray((a[ix,3:]-checkpoint['mean'])/checkpoint['scale'])))).numpy().ravel()
 actual=np.fromfile(prediction/'validation_l100.scores','<f4')[ix];error=float(np.max(np.abs(expected-actual)));assert error<2e-5;dump(R/'l100_native_verification.json',dict(max_error=error,samples=2000))
