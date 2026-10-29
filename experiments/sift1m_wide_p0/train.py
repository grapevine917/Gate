"""Train matched wide-L LR/MLP plus the shared ET comparator from scratch."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import hashlib,json,struct
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
import joblib
D=Path(__file__).resolve().parent;M=D/'models';P=json.loads((D/'PROTOCOL.json').read_text());SEED=P['seed'];torch.set_num_threads(4)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
pr=np.fromfile(D/'data/mixed.prune.bin','<f4').reshape(-1,20);et=np.fromfile(D/'data/mixed.et.bin','<f4').reshape(-1,19)
ids=np.load(D/'data/train_ids.npy');hold=np.load(D/'data/internal_holdout_ids.npy')
x=np.ascontiguousarray(pr[:,3:]);y=pr[:,2].copy();ex=np.ascontiguousarray(et[:,2:]);ey=et[:,1].copy()
inner=~np.isin(ids[pr[:,0].astype(int)],hold);einner=~np.isin(ids[et[:,0].astype(int)],hold)
assert set(np.unique(x[:,15]))==set(P['L_training_and_learned_grid']) and np.isfinite(x).all() and np.isfinite(ex).all()
print('SAMPLES',len(x),'ET',len(ex),'retain fraction',float(y.mean()),flush=True)
def norm(a):
    mu=a.mean(0,dtype=np.float64).astype('f4');sd=a.std(0,dtype=np.float64).astype('f4');sd[sd<1e-8]=1;return mu,sd
mu,sd=norm(x);z=np.ascontiguousarray((x-mu)/sd)
lr=LogisticRegression(C=1,max_iter=1500,tol=1e-5,random_state=SEED).fit(z,y)
with (M/'lr.bin').open('wb') as f:
    f.write(b'LRGATE01');f.write(struct.pack('<I',0))
    for a in [mu,sd,-lr.coef_[0],-lr.intercept_]:np.asarray(a,'<f4').tofile(f)
joblib.dump(dict(model=lr,mean=mu,scale=sd),M/'lr.joblib')
dump(M/'lr_metrics.json',dict(samples=len(x),train_auc=float(roc_auc_score(y,lr.predict_proba(z)[:,1])),train_logloss=float(log_loss(y,lr.predict_proba(z)[:,1]))))
print('LR TRAINED',flush=True)
class Net(nn.Module):
    def __init__(self):super().__init__();self.seq=nn.Sequential(nn.Linear(17,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,a):return self.seq(a).squeeze(-1)
def fitnet(name,a,target,mask,reg=False):
    def fit(subset,epochs,validation=None):
        mm,ss=norm(a[subset]);az=torch.from_numpy(np.ascontiguousarray((a-mm)/ss));yt=torch.from_numpy(target);torch.manual_seed(SEED)
        net=Net();opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[40,60],gamma=.3)
        rng=np.random.default_rng(SEED);ix=np.flatnonzero(subset);history=[];best_epoch=1;best_loss=float('inf');stale=0
        for epoch in range(1,epochs+1):
            net.train();total=0.
            for batch in np.array_split(rng.permutation(ix),max(1,int(np.ceil(len(ix)/2048)))):
                pred=net(az[batch]);loss=((torch.sigmoid(pred)-yt[batch])**2).mean() if reg else nn.functional.binary_cross_entropy_with_logits(pred,yt[batch])
                opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch)
            sched.step();row=dict(epoch=epoch,train_loss=total/len(ix))
            if validation is not None:
                net.eval()
                with torch.no_grad():
                    pred=net(az[validation]);vl=float(((torch.sigmoid(pred)-yt[validation])**2).mean() if reg else nn.functional.binary_cross_entropy_with_logits(pred,yt[validation]))
                row['internal_loss']=vl
                if vl<best_loss-1e-5:best_loss=vl;best_epoch=epoch;stale=0
                else:stale+=1
            history.append(row)
            if epoch%5==0:print(name,'inner' if validation is not None else 'refit',row,flush=True)
            if validation is not None and stale>=12:break
        return net,mm,ss,history,best_epoch
    _,_,_,history,best=fit(mask,80,~mask);dump(M/f'{name}_internal_history.json',history)
    model,mm,ss,history,_=fit(np.ones(len(a),bool),best);dump(M/f'{name}_history.json',history)
    model.eval()
    with (M/f'{name}.bin').open('wb') as f:
        f.write(b'STGATE01');mm.astype('<f4').tofile(f);ss.astype('<f4').tofile(f)
        for layer in model.seq:
            if isinstance(layer,nn.Linear):layer.weight.detach().numpy().T.copy().astype('<f4').tofile(f);layer.bias.detach().numpy().astype('<f4').tofile(f)
    torch.save(dict(weights=model.state_dict(),mean=mm,scale=ss),M/f'{name}.pt')
    dump(M/f'{name}_metrics.json',dict(samples=len(a),best_epoch=best,input_features=17,regression=reg))
    print(name,'TRAINED epoch',best,flush=True)
    return model,mm,ss
state,sm,ss=fitnet('state',x,y,inner)
estate,em,es=fitnet('et',ex,ey,einner,True)
ix=np.random.default_rng(SEED).choice(len(x),2000,replace=False);features=x[ix]
with torch.no_grad():sp=torch.sigmoid(state(torch.from_numpy(np.ascontiguousarray((features-sm)/ss)))).numpy();ep=torch.sigmoid(estate(torch.from_numpy(np.ascontiguousarray((features-em)/es)))).numpy()
expected=np.column_stack([lr.predict_proba((features-mu)/sd)[:,1],sp,ep]).astype('f4')
with (M/'check.bin').open('wb') as f:f.write(struct.pack('<I',len(ix)));np.column_stack([features,expected]).astype('<f4').tofile(f)
dump(D/'TRAIN_COMPLETE.json',dict(complete=True,prune_samples=len(x),et_samples=len(ex),L_values=sorted(map(int,np.unique(x[:,15]))),same_gate_rows=True,all_weights_from_scratch=True,models={n:sha(M/f'{n}.bin') for n in ['lr','state','et']},normalization='internal1000 queries during epoch selection, all1200 in final refit',max_epochs=80))
print('TRAINING COMPLETE',flush=True)
