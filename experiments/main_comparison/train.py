import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,struct,hashlib,copy
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss,mean_squared_error
from xgboost import XGBClassifier
import joblib
R=Path(__file__).resolve().parent;LG=R.parents[1];M=R/'models';M.mkdir(exist_ok=True);SEED=20261030
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
p=np.fromfile(R/'data/mixed.prune.bin','<f4').reshape(-1,20);e=np.fromfile(R/'data/mixed.et.bin','<f4').reshape(-1,19)
ids=np.load(R/'data/train_ids.npy');hold=np.load(R/'data/internal_holdout_ids.npy');inner=~np.isin(ids[p[:,0].astype(int)],hold);ei=~np.isin(ids[e[:,0].astype(int)],hold)
x=np.ascontiguousarray(p[:,3:]);y=p[:,2].copy();ex=np.ascontiguousarray(e[:,2:]);ey=e[:,1].copy()
assert x.shape[1]==17 and ex.shape[1]==17 and np.isfinite(x).all() and np.isfinite(ex).all()
assert set(np.unique(x[:,15]))==set(json.loads((R/'PROTOCOL.json').read_text())['L_training_and_learned_grid'])
print('PRUNE',len(x),'ET',len(ex),'retain',float(y.mean()),flush=True)
torch.set_num_threads(4)
def normalization(a):
 mu=a.mean(0,dtype=np.float64).astype('f4');sd=a.std(0,dtype=np.float64).astype('f4');sd[sd<1e-8]=1;return mu,sd
mu,sd=normalization(x);z=(x-mu)/sd
lr=LogisticRegression(C=1,max_iter=1500,tol=1e-5,random_state=SEED).fit(z,y)
with (M/'lr.bin').open('wb') as f:
 f.write(b'LRGATE01');f.write(struct.pack('<I',0))
 for a in [mu,sd,-lr.coef_[0],-lr.intercept_]:np.asarray(a,'<f4').tofile(f)
joblib.dump(dict(model=lr,mean=mu,scale=sd),M/'lr.joblib');print('LR DONE',flush=True)
params=dict(objective='binary:logistic',tree_method='hist',n_estimators=300,max_depth=5,learning_rate=.05,min_child_weight=10,subsample=.8,colsample_bytree=.9,reg_lambda=1,n_jobs=4,random_state=SEED,eval_metric='logloss')
trial=XGBClassifier(**params,early_stopping_rounds=20).fit(x[inner],y[inner],eval_set=[(x[~inner],y[~inner])],verbose=False)
params['n_estimators']=int(trial.best_iteration)+1;gb=XGBClassifier(**params).fit(x,y);gb.save_model(M/'xgb.ubj');booster=gb.get_booster();bc=json.loads(booster.save_config())['learner']['learner_model_param']['base_score'];bc=float(str(bc).strip('[]'));trees=booster.get_dump(dump_format='json')
with (M/'xgb.bin').open('wb') as f:
 f.write(b'XGGATE01');f.write(struct.pack('<fI',float(np.log(bc/(1-bc))),len(trees)))
 for raw in trees:
  nodes={}
  def visit(t):
   nodes[t['nodeid']]=(-1,-1,-1,t['leaf']) if 'leaf'in t else(t['yes'],t['no'],int(t['split'][1:]),t['split_condition'])
   for c in t.get('children',[]):visit(c)
  visit(json.loads(raw));assert sorted(nodes)==list(range(len(nodes)));f.write(struct.pack('<I',len(nodes)))
  for j in range(len(nodes)):f.write(struct.pack('<iiif',*nodes[j]))
print('XGB DONE',len(trees),flush=True)
class Net(nn.Module):
 def __init__(self,d):super().__init__();self.seq=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,a):return self.seq(a).squeeze(-1)
def fitnet(name,a,target,mask,reg=False):
 def fit(subset,epochs,validation=None):
  mm,ss=normalization(a[subset]);az=torch.from_numpy(np.ascontiguousarray((a-mm)/ss));yt=torch.from_numpy(target);torch.manual_seed(SEED);net=Net(a.shape[1]);optim=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4);rng=np.random.default_rng(SEED);ix=np.flatnonzero(subset);history=[];best=None;bestloss=float('inf');stale=0
  for ep in range(1,epochs+1):
   net.train();losses=0
   for batch in np.array_split(rng.permutation(ix),max(1,int(np.ceil(len(ix)/2048)))):
    pred=net(az[batch]);loss=((torch.sigmoid(pred)-yt[batch])**2).mean() if reg else nn.functional.binary_cross_entropy_with_logits(pred,yt[batch]);optim.zero_grad();loss.backward();optim.step();losses+=float(loss.detach())*len(batch)
   row=dict(epoch=ep,train_loss=losses/len(ix))
   if validation is not None:
    net.eval()
    with torch.no_grad():pred=net(az[validation]);v=float(((torch.sigmoid(pred)-yt[validation])**2).mean() if reg else nn.functional.binary_cross_entropy_with_logits(pred,yt[validation]))
    row['internal_loss']=v
    if v<bestloss-1e-5:bestloss=v;best=ep;stale=0
    else:stale+=1
   history.append(row)
   if ep%5==0:print(name,ep,row,flush=True)
   if validation is not None and stale>=6:break
  return net,mm,ss,history,best
 _,_,_,hist,best=fit(mask,40,~mask);dump(M/f'{name}_internal_history.json',hist)
 model,mm,ss,hist,_=fit(np.ones(len(a),bool),best);dump(M/f'{name}_history.json',hist);model.eval()
 with torch.no_grad():pred=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray((a-mm)/ss)))).numpy()
 with (M/f'{name}.bin').open('wb') as f:
  f.write(b'M3GATE01' if a.shape[1]==529 else b'STGATE01');mm.astype('<f4').tofile(f);ss.astype('<f4').tofile(f)
  for layer in model.seq:
   if isinstance(layer,nn.Linear):layer.weight.detach().numpy().T.copy().astype('<f4').tofile(f);layer.bias.detach().numpy().astype('<f4').tofile(f)
 torch.save(dict(weights=model.state_dict(),mean=mm,scale=ss),M/f'{name}.pt')
 metric=dict(samples=len(a),input_features=a.shape[1],best_epoch=best,train_MSE=float(mean_squared_error(target,pred)))
 if not reg:metric.update(train_AUC=float(roc_auc_score(target,pred)),train_logloss=float(log_loss(target,pred)))
 dump(M/f'{name}_metrics.json',metric);print(name,'DONE',metric,flush=True);return pred
sp=fitnet('state',x,y,inner);ep=fitnet('et',ex,ey,ei,True)
with (R/'data/train.fbin').open('rb') as f:n,d=np.fromfile(f,'<u4',2);queries=np.fromfile(f,'<f4').reshape(n,d)
base=np.memmap(LG/'data/sift/sift_base.fbin',dtype='<f4',mode='r',offset=8,shape=(1000000,128));q=queries[p[:,0].astype(int)];v=np.asarray(base[p[:,1].astype(int)]);joint=np.empty((len(x),529),'f4');joint[:,:128]=q;joint[:,128:256]=v;joint[:,256:384]=np.abs(q-v);joint[:,384:512]=q*v;joint[:,512:]=x
del v
qp=fitnet('qv',joint,y,inner)
# Independent native export check samples; no search decision tapes.
rng=np.random.default_rng(SEED);ix=rng.choice(len(p),2000,replace=False);expected=np.stack([lr.predict_proba(z[ix])[:,1],gb.predict_proba(x[ix])[:,1],sp[ix],qp[ix]],1)
with (R/'models/native_check.bin').open('wb') as f:
 f.write(struct.pack('<I',len(ix)));np.concatenate([q[ix],p[ix,1:2],x[ix],expected],1).astype('<f4').tofile(f)
ix=rng.choice(len(e),2000,replace=False)
with (M/'et_check.bin').open('wb') as f:f.write(struct.pack('<I',len(ix)));np.concatenate([ex[ix],ep[ix,None]],1).astype('<f4').tofile(f)
dump(R/'TRAIN_COMPLETE.json',dict(complete=True,prune_samples=len(p),et_samples=len(e),L_values=sorted(map(int,np.unique(x[:,15]))),xgb_trees=len(trees),models={n:hashlib.sha256((M/f'{n}.bin').read_bytes()).hexdigest() for n in ['lr','xgb','state','qv','et']},normalization='training queries only; inner stage fits internal training normalization; final refit uses all1200 training queries',training_targets='all pruning models use identical candidate sample rows and retain labels; ET uses offline current GT recall'))
print('TRAINING COMPLETE',flush=True)
