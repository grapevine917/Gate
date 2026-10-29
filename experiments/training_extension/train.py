"""Longer training with fixed data, query holdout and original architecture."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,struct,hashlib,copy,time,subprocess
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,roc_auc_score,average_precision_score
import joblib
R=Path(__file__).resolve().parent;M=R.parent/'main_comparison';A=R.parent/'et_gate_adaptation';torch.set_num_threads(4)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
P=dict(max_epochs=160,seeds=[20261030,20261022,20261023],adapted_seeds=[20261021,20261022,20261023],learning_rates={'1-40':.001,'41-80':.0003,'81-120':.0001,'121-160':.00003},batch_size=2048,optimizer='Adam weight_decay=1e-4',architecture='17-128-64-1; ReLU hidden; sigmoid output',training='Fixed existing data; internal holdout of 200 disjoint queries; 3 seeds, evaluate all160 epochs; choose seed+epoch on internal loss only; refit all1200 training queries using selected seed, epoch and fixed LR schedule. No external validation/test used for fitting or checkpoint selection.',tasks=['gate','et_unpruned','et_adapted'],limits=['Same finite training examples; longer training does not add query diversity','Gate uses contribution proxy retain labels, not counterfactual recall risk','ET labels are absolute prefix GT recall','Previously observed benchmark queries; exploratory'])
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==P
else:dump(R/'PROTOCOL.json',P)
inputs=[M/'data/mixed.prune.bin',M/'data/mixed.et.bin',A/'data/prefixes.bin',M/'data/train_ids.npy',M/'data/internal_holdout_ids.npy',M/'models/state.bin',M/'models/et.bin',M/'models/lr.bin',M/'models/lr.joblib',A/'models/adapted.bin']
lock={str(p):sha(p) for p in inputs}
if (R/'INPUT_LOCK.json').exists():assert json.loads((R/'INPUT_LOCK.json').read_text())==lock
else:dump(R/'INPUT_LOCK.json',lock)
ids=np.load(M/'data/train_ids.npy');hold=np.load(M/'data/internal_holdout_ids.npy')
class Net(nn.Module):
 def __init__(self):super().__init__();self.seq=nn.Sequential(nn.Linear(17,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.seq(x).squeeze(-1)
def lr_at(epoch):return .001 if epoch<=40 else .0003 if epoch<=80 else .0001 if epoch<=120 else .00003
def normalize(x):
 mu=x.mean(0,dtype='f8').astype('f4');sd=x.std(0,dtype='f8').astype('f4');sd[sd<1e-8]=1;return mu,sd
def lossfn(pred,y,reg):return ((torch.sigmoid(pred)-y)**2).mean() if reg else nn.functional.binary_cross_entropy_with_logits(pred,y)
def fit(name,x,y,mask,seed,epochs,reg,internal):
 mu,sd=normalize(x[mask]);z=torch.from_numpy(np.ascontiguousarray((x-mu)/sd));yt=torch.from_numpy(y);torch.manual_seed(seed);net=Net();opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4);rng=np.random.default_rng(seed);ix=np.flatnonzero(mask);hist=[];best_loss=float('inf');best_epoch=0;best_weights=None
 for ep in range(1,epochs+1):
  for group in opt.param_groups:group['lr']=lr_at(ep)
  net.train();total=0
  for batch in np.array_split(rng.permutation(ix),max(1,int(np.ceil(len(ix)/2048)))):
   loss=lossfn(net(z[batch]),yt[batch],reg);opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch)
  row=dict(epoch=ep,lr=lr_at(ep),train_loss=total/len(ix));net.eval()
  if internal:
   with torch.no_grad():value=float(lossfn(net(z[~mask]),yt[~mask],reg))
   row['holdout_loss']=value
   if value<best_loss:best_loss=value;best_epoch=ep;best_weights=copy.deepcopy(net.state_dict())
  hist.append(row)
  if ep%20==0:log(f'{name} seed={seed} internal={internal} {row}')
  dump(R/'models'/f'{name}_s{seed}_{"inner" if internal else "refit"}.json',hist)
 if internal:net.load_state_dict(best_weights)
 return net,mu,sd,hist,best_epoch,best_loss
def export(name,net,mu,sd,x):
 net.eval()
 with (R/'models'/f'{name}.bin').open('wb') as f:
  f.write(b'STGATE01');mu.astype('<f4').tofile(f);sd.astype('<f4').tofile(f)
  for layer in net.seq:
   if isinstance(layer,nn.Linear):layer.weight.detach().numpy().T.copy().astype('<f4').tofile(f);layer.bias.detach().numpy().astype('<f4').tofile(f)
 torch.save(dict(weights=net.state_dict(),mean=mu,scale=sd),R/'models'/f'{name}.pt')
 ix=np.random.default_rng(20261021).choice(len(x),min(2000,len(x)),replace=False)
 with torch.no_grad():pred=torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray((x[ix]-mu)/sd)))).numpy()
 with (R/'models'/f'{name}_check.bin').open('wb') as f:f.write(struct.pack('<I',len(ix)));np.column_stack([x[ix],pred]).astype('<f4').tofile(f)
 result=subprocess.check_output([str(A/'verify_model'),str(R/'models'/f'{name}.bin'),str(R/'models'/f'{name}_check.bin')]);return float(result)
summaries=[]
for name in P['tasks']:
 if (R/'models'/f'{name}_metrics.json').exists():summaries.append(json.loads((R/'models'/f'{name}_metrics.json').read_text()));continue
 if name=='gate':a=np.fromfile(M/'data/mixed.prune.bin','<f4').reshape(-1,20);x=a[:,3:].copy();y=a[:,2].copy();qid=a[:,0].astype(int)
 elif name=='et_unpruned':a=np.fromfile(M/'data/mixed.et.bin','<f4').reshape(-1,19);x=a[:,2:].copy();y=a[:,1].copy();qid=a[:,0].astype(int)
 else:a=np.fromfile(A/'data/prefixes.bin','<f4').reshape(-1,23);x=a[:,6:].copy();y=a[:,1].copy();qid=a[:,0].astype(int)
 mask=~np.isin(ids[qid],hold);reg=name!='gate';seeds=P['adapted_seeds'] if name=='et_adapted' else P['seeds'];trials=[]
 for seed in seeds:
  path=R/'models'/f'{name}_s{seed}_trial.pt';metric=R/'models'/f'{name}_s{seed}_trial.json'
  if metric.exists():trials.append(json.loads(metric.read_text()));continue
  net,mu,sd,hist,epoch,value=fit(name,x,y,mask,seed,160,reg,True)
  torch.save(dict(weights=net.state_dict(),mean=mu,scale=sd),path)
  original=min(hist[:40],key=lambda r:r['holdout_loss']);trial=dict(seed=seed,best_epoch=epoch,holdout_loss=value,best_first40=original,relative_holdout_improvement_over_first40_pct=100*(1-value/original['holdout_loss']))
  dump(metric,trial);trials.append(trial)
 winner=min(trials,key=lambda r:r['holdout_loss']);seed=winner['seed'];epoch=winner['best_epoch'];net,mu,sd,hist,_,_=fit(name,x,y,np.ones(len(x),bool),seed,epoch,reg,False);err=export(name,net,mu,sd,x)
 # Same-query held-out loss difference: later checkpoint vs <=40-epoch checkpoint.
 chosen=torch.load(R/'models'/f'{name}_s{seed}_trial.pt',weights_only=False);net_inner=Net();net_inner.load_state_dict(chosen['weights']);net_inner.eval();xm=x[~mask];ym=y[~mask]
 with torch.no_grad():pred=torch.sigmoid(net_inner(torch.from_numpy(np.ascontiguousarray((xm-chosen['mean'])/chosen['scale'])))).numpy()
 perquery=[]
 for q in np.unique(qid[~mask]):
  k=qid[~mask]==q;loss=float(np.mean((pred[k]-ym[k])**2)) if reg else float(log_loss(ym[k],pred[k],labels=[0,1]));perquery.append(dict(query=int(q),loss=loss))
 dump(R/'models'/f'{name}_holdout_query_losses.json',perquery)
 metric=dict(task=name,samples=len(x),training_queries=1200,internal_holdout_queries=200,selected=winner,trials=trials,native_max_error=err,model_sha256=sha(R/'models'/f'{name}.bin'))
 dump(R/'models'/f'{name}_metrics.json',metric);summaries.append(metric);log(f'FINISHED {name}: {winner}')
# A stricter optimizer tolerance diagnoses LR convergence without changing labels/C.
if not (R/'LR_CONVERGENCE.json').exists():
 a=np.fromfile(M/'data/mixed.prune.bin','<f4').reshape(-1,20);x=a[:,3:].copy();y=a[:,2].copy();old=joblib.load(M/'models/lr.joblib');mu=old['mean'];sd=old['scale'];z=(x-mu)/sd
 lr=LogisticRegression(C=1,max_iter=10000,tol=1e-9,random_state=20261030).fit(z,y);p0=old['model'].predict_proba(z)[:,1];p1=lr.predict_proba(z)[:,1]
 with (R/'models/lr_tight.bin').open('wb') as f:
  f.write(b'LRGATE01');f.write(struct.pack('<I',0))
  for arr in [mu,sd,-lr.coef_[0],-lr.intercept_]:np.asarray(arr,'<f4').tofile(f)
 joblib.dump(dict(model=lr,mean=mu,scale=sd),R/'models/lr_tight.joblib')
 dump(R/'LR_CONVERGENCE.json',dict(original_iterations=int(old['model'].n_iter_[0]),tight_iterations=int(lr.n_iter_[0]),original_logloss=float(log_loss(y,p0)),tight_logloss=float(log_loss(y,p1)),mean_abs_probability_change=float(np.mean(np.abs(p1-p0))),max_abs_probability_change=float(np.max(np.abs(p1-p0))),source='Training-set convergence diagnosis only; not evidence of test improvement'))
assert all(sha(p)==h for p,h in lock.items())
dump(R/'TRAIN_COMPLETE.json',dict(complete=True,models=summaries,original_inputs_unchanged=True));log('ALL TRAINING COMPLETE')
