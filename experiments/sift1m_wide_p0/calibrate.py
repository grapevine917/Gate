"""Validation-only wide-L score calibration; process exits before timed search."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json
import numpy as np
import torch,joblib
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score
D=Path(__file__).resolve().parent;P=json.loads((D/'PROTOCOL.json').read_text());torch.set_num_threads(1)
a=np.fromfile(D/'data/calibration.prune.bin','<f4').reshape(-1,20);x=np.ascontiguousarray(a[:,3:]);y=a[:,2].astype(bool)
assert np.all(x[:,15]==300) and np.all(x[:,10]==0)
bundle=joblib.load(D/'models/lr.joblib');lr=bundle['model'].predict_proba((x-bundle['mean'])/bundle['scale'])[:,1]
state=torch.load(D/'models/state.pt',map_location='cpu',weights_only=False)
model=nn.Sequential(nn.Linear(17,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
model.load_state_dict({k.removeprefix('seq.'):v for k,v in state['weights'].items()});model.eval()
with torch.no_grad():mlp=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray((x-state['mean'])/state['scale'])))).numpy().ravel()
out={};metrics={}
for name,scores in [('state',mlp),('lr',lr)]:
    retain=np.sort(scores[y]);points=[]
    for risk in P['risks']:
        tau=0. if risk==0 else float(retain[min(len(retain)-1,int(np.floor(risk*len(retain))))])
        false=int(np.sum(scores[y]<tau));assert false<=int(np.floor(risk*len(retain)))
        points.append(dict(risk=risk,tau=tau,retain_count=len(retain),false_skips=false))
    out[name]=points;metrics[name]=dict(validation_L300_samples=len(x),retain_auc=float(roc_auc_score(y,scores)),safe_skip_ap=float(average_precision_score(~y,1-scores)))
(D/'thresholds.json').write_text(json.dumps(out,indent=2)+'\n');(D/'validation_prediction_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print('CALIBRATED',metrics,flush=True)
