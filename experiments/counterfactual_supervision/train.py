import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,struct,hashlib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,mean_absolute_error,mean_squared_error,r2_score
R=Path(__file__).resolve().parent;E=R.parent/'e4_1_redundant_keep';E1=R.parent/'e1_similarity';S=R.parent/'sift1m';SEED=20261024
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=dict(np.load(E/'keep_positions.npz'));o=dict(np.load(E/'outcomes.npz'));parts=json.loads((R.parent/'e4_2_history_skip/PROTOCOL.json').read_text());qr=o['query_row'];x=p['state_features'];risk=(o['delta_hits']<0).astype(np.float32);gain=o['delta_IO'].astype(np.float32)
assert x.shape==(93410,17) and not p['cache_hit'].any() and not p['model_decision'].any()
for a,b in [('fit','validation'),('fit','test'),('validation','test')]:assert not set(parts[a+'_query_rows'])&set(parts[b+'_query_rows'])
for split in ['fit','validation','test']:
 ids=np.array(parts[split+'_query_rows']);assert len(set(ids))==len(ids)
 mask=np.isin(qr,ids);print(split,'queries',len(ids),'labels',mask.sum(),'harmful',risk[mask].sum(),flush=True)
(R/'data').mkdir(exist_ok=True)
def slice_bin(src,dst,rows):
 with src.open('rb') as f:n,d=np.fromfile(f,'<u4',2);a=np.fromfile(f,'<u4').reshape(n,d)
 with dst.open('wb') as f:np.array([len(rows),d],'<u4').tofile(f);a[rows].tofile(f)
for split in ['validation','test']:
 rows=np.array(parts[split+'_query_rows']);np.save(R/f'data/{split}_source_rows.npy',rows)
 for ext in ['fbin','gt']:slice_bin(E1/f'data/test.{ext}',R/f'data/{split}.{ext}',rows)
 sourceids=np.array(json.loads((E1/'query_splits.json').read_text())['test']);np.save(R/f'data/{split}_query_ids.npy',sourceids[rows])
protocol=dict(source_alias='User E4-B1 mapped to available E4-1 redundant_keep single-node counterfactual artifacts',dataset='SIFT1M',L=100,K=10,beam=4,threads=4,cache_nodes=10000,seed=SEED,query_partition={k:parts[k+'_query_rows'] for k in ['fit','validation','test']},features='exact original 17D pre-I/O state only; no node ID, original MLP score, labels, history or GT as input',targets=dict(risk='delta_hits<0',gain='signed baseline_reads-after_reads; negative values retained'),model='17-128-64 shared ReLU MLP, two linear output heads; sigmoid risk; signed I/O regression',loss='unweighted BCE risk + MSE on fit-standardized signed I/O target',training='fit-only standardization; choose epoch on internal 400/100 query split from fit; refit all500 with chosen epoch count; no test checkpoint selection',deployment='one-stage replacement on every cache-miss candidate; cached candidates protected; skip iff predicted risk<risk_threshold AND predicted gain>gain_threshold',baseline='unchanged frozen original17D MLP; single global threshold; dense validation-only sweep',selection='min validation SSD requests subject to absolute Recall constraint; freeze before test; no-prune included; both families receive same validation queries',recall_targets=[99.60,99.44,99.20,99.00],timing='same isolated native binary; 4 threads, per-run OS/cache warmup; randomized serial configurations; chosen test configs 3 repeats; search-only time includes actual inference',limits=['labels only original MLP Keep miss nodes at fixed original tau; all-candidate deployment is extrapolation','single-node effects conditional on original trajectory; not additive joint gains','reused previously observed 1000-query E4 pool; exploratory within-pool split','E4 single-node harmful label is final hit count reduction, not complete identity/ranking preservation','SSD submitted requests, not physical reads'])
dump(R/'PROTOCOL.json',protocol)
inputs=[E/'keep_positions.npz',E/'outcomes.npz',E/'PROTOCOL.json',E/'verification.json',S/'online_state_mlp/model.bin',R.parent/'e4_2_history_skip/PROTOCOL.json',R.parents[1]/'DiskANN/src/pq_flash_index.cpp',R.parents[1]/'DiskANN/include/online_gate.h']
dump(R/'INPUT_LOCK.json',{str(a):sha(a) for a in inputs})
torch.set_num_threads(1)
class Net(nn.Module):
 def __init__(self):super().__init__();self.layers=nn.Sequential(nn.Linear(17,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,2))
 def forward(self,z):return self.layers(z)
fitrows=np.array(parts['fit_query_rows']);hold=np.random.default_rng(SEED).choice(fitrows,100,replace=False);inner=np.isin(qr,fitrows)&~np.isin(qr,hold);valid=np.isin(qr,hold);fit=np.isin(qr,fitrows)
def normalize(mask):
 mu=x[mask].mean(0,dtype=np.float64).astype('f4');sd=x[mask].std(0,dtype=np.float64).astype('f4');sd[sd<1e-8]=1
 gm=float(gain[mask].mean());gs=float(gain[mask].std());return mu,sd,gm,gs
lossfn=nn.BCEWithLogitsLoss()
def learn(mask,epochs,check=None):
 torch.manual_seed(SEED);model=Net();mu,sd,gm,gs=normalize(mask);z=torch.from_numpy(((x-mu)/sd).astype('f4'));yr=torch.from_numpy(risk);yg=torch.from_numpy(((gain-gm)/gs).astype('f4'));ix=np.flatnonzero(mask);optim=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4);rng=np.random.default_rng(SEED);hist=[];best=None;bestloss=float('inf')
 for epoch in range(1,epochs+1):
  model.train();total=0
  for batch in np.array_split(rng.permutation(ix),max(1,int(np.ceil(len(ix)/1024)))):
   pred=model(z[batch]);loss=lossfn(pred[:,0],yr[batch])+((pred[:,1]-yg[batch])**2).mean();optim.zero_grad();loss.backward();optim.step();total+=float(loss.detach())*len(batch)
  row=dict(epoch=epoch,train_loss=total/len(ix))
  if check is not None:
   model.eval()
   with torch.no_grad():pred=model(z[check]);v=float(lossfn(pred[:,0],yr[check])+((pred[:,1]-yg[check])**2).mean())
   row['internal_validation_loss']=v
   if v<bestloss:bestloss=v;best=epoch
  hist.append(row)
 return model,(mu,sd,gm,gs),hist,best
_,_,hist,best=learn(inner,80,valid);dump(R/'internal_training_history.json',hist)
print('best epoch',best,flush=True);model,norm,hist,_=learn(fit,best);dump(R/'training_history.json',hist);mu,sd,gm,gs=norm
np.savez(R/'normalization.npz',mean=mu,scale=sd,gain_mean=gm,gain_scale=gs)
model.eval()
with torch.no_grad():raw=model(torch.from_numpy(((x-mu)/sd).astype('f4'))).numpy()
probs=1/(1+np.exp(-raw[:,0].astype('f8')));predgain=raw[:,1]*gs+gm
metrics={}
for split in ['fit','validation','test']:
 m=np.isin(qr,parts[split+'_query_rows']);y=risk[m];pr=probs[m];g=gain[m];pg=predgain[m]
 metrics[split]=dict(n=int(m.sum()),harmful=int(y.sum()),risk_AUC=roc_auc_score(y,pr),risk_AP=average_precision_score(y,pr),risk_prevalence=float(y.mean()),risk_Brier=brier_score_loss(y,pr),gain_MAE=mean_absolute_error(g,pg),gain_RMSE=float(np.sqrt(mean_squared_error(g,pg))),gain_R2=r2_score(g,pg),gain_constant_fit_mean_MAE=mean_absolute_error(g,np.full(len(g),gm)))
dump(R/'offline_metrics.json',metrics)
np.savez_compressed(R/'label_predictions.npz',query_row=qr,risk=risk,gain=gain,risk_probability=probs,predicted_gain=predgain)
with (R/'model.bin').open('wb') as f:
 f.write(b'CFGATE01');np.asarray(mu,'<f4').tofile(f);np.asarray(sd,'<f4').tofile(f);np.array([gm,gs],'<f4').tofile(f)
 for layer in model.layers:
  if isinstance(layer,nn.Linear):layer.weight.detach().numpy().astype('<f4').T.copy().tofile(f);layer.bias.detach().numpy().astype('<f4').tofile(f)
torch.save(model.state_dict(),R/'model.pt')
# Native export verification input, all held-out label states, no online decision tapes.
m=np.isin(qr,parts['test_query_rows'])
with (R/'native_check.bin').open('wb') as f:f.write(struct.pack('<I',int(m.sum())));x[m].astype('<f4').tofile(f);np.stack([probs[m],predgain[m]],1).astype('<f4').tofile(f)
dump(R/'TRAIN_COMPLETE.json',dict(best_epoch=best,model_sha256=sha(R/'model.bin'),fit_labels=int(fit.sum()),fit_harmful=int(risk[fit].sum())))
print(json.dumps(metrics),flush=True)
