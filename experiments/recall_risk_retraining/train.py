from common import *
import torch,joblib
from torch import nn
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(4);torch.set_flush_denormal(True)
assert (D/'LABELS_TRAIN_COMPLETE.json').exists()
a=labels('train');x=np.ascontiguousarray(a['state']);weights=a['weight'].astype('f4');weights/=weights.mean()
ids=np.load(W/'data/train_ids.npy');hold=np.load(W/'data/internal_holdout_ids.npy');inner=~np.isin(ids[a['query']],hold)
assert len(set(a['query'][inner]))==1000 and len(set(a['query'][~inner]))==200
def norm(xx,ww):
    mm=np.average(xx.astype('f8'),axis=0,weights=ww).astype('f4');ss=np.sqrt(np.average((xx.astype('f8')-mm)**2,axis=0,weights=ww)).astype('f4');cc=ss<1e-8;ss[cc]=1;return mm,ss,cc
class Net(nn.Module):
    def __init__(self,constant):
        super().__init__();self.seq=nn.Sequential(nn.Linear(17,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        with torch.no_grad():self.seq[0].weight[:,constant]=0
    def forward(self,xx):return self.seq(xx).squeeze(-1)
results=[]
for target in ['proxy','risk']:
    M=D/'models'/target;M.mkdir(parents=True,exist_ok=True)
    if (M/'COMPLETE.json').exists():results.append(json.loads((M/'COMPLETE.json').read_text()));continue
    y=a['proxy'].astype('f4') if target=='proxy' else (a['keep_hits']>a['skip_hits']).astype('f4')
    mu,sd,cc=norm(x,weights);z=np.ascontiguousarray((x-mu)/sd)
    lr=LogisticRegression(C=1,max_iter=1500,tol=1e-5,random_state=SEED).fit(z,y,sample_weight=weights);lr.coef_[:,cc]=0
    with (M/'lr.bin').open('wb') as f:
        f.write(b'LRGATE01');f.write(struct.pack('<I',0))
        for v in [mu,sd,-lr.coef_[0],-lr.intercept_]:np.asarray(v,'<f4').tofile(f)
    joblib.dump(dict(model=lr,mean=mu,scale=sd,constant=cc),M/'lr.joblib')
    def fit(mask,epochs,validation=None):
        mm,ss,constant=norm(x[mask],weights[mask]);xt=torch.from_numpy(np.ascontiguousarray((x-mm)/ss));yt=torch.from_numpy(y);wt=torch.from_numpy(weights)
        torch.manual_seed(SEED);net=Net(constant);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[40,60],gamma=.3);rng=np.random.default_rng(SEED);ix=np.flatnonzero(mask);history=[];best=1;bestloss=float('inf');stale=0
        for ep in range(1,epochs+1):
            net.train();total=0.
            for batch in np.array_split(rng.permutation(ix),max(1,int(np.ceil(len(ix)/2048)))):
                loss=(nn.functional.binary_cross_entropy_with_logits(net(xt[batch]),yt[batch],reduction='none')*wt[batch]).mean();opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch)
            scheduler.step();r=dict(epoch=ep,train_weighted_loss=total/len(ix))
            if validation is not None:
                net.eval()
                with torch.no_grad():vl=float((nn.functional.binary_cross_entropy_with_logits(net(xt[validation]),yt[validation],reduction='none')*wt[validation]).sum()/wt[validation].sum())
                r['internal_weighted_loss']=vl
                if vl<bestloss-1e-5:bestloss=vl;best=ep;stale=0
                else:stale+=1
            history.append(r)
            if ep%10==0:print(time.strftime('%H:%M:%S'),target,'inner' if validation is not None else 'refit',r,flush=True)
            if validation is not None and stale>=12:break
        return net,mm,ss,constant,history,best
    _,_,_,_,history,best=fit(inner,80,~inner);dump(M/'internal_history.json',history)
    net,mm,ss,constant,history,_=fit(np.ones(len(a),bool),best);dump(M/'history.json',history);net.eval()
    assert torch.count_nonzero(net.seq[0].weight[:,constant]).item()==0
    with (M/'state.bin').open('wb') as f:
        f.write(b'STGATE01');mm.astype('<f4').tofile(f);ss.astype('<f4').tofile(f)
        for layer in net.seq:
            if isinstance(layer,nn.Linear):layer.weight.detach().numpy().T.copy().astype('<f4').tofile(f);layer.bias.detach().numpy().astype('<f4').tofile(f)
    torch.save(dict(weights=net.state_dict(),mean=mm,scale=ss,constant=constant),M/'state.pt')
    # Export verification uses validation states only, never test checkpoint selection.
    v=labels('validation')['state'];ix=np.random.default_rng(SEED).choice(len(v),min(3000,len(v)),replace=False);check=np.ascontiguousarray(v[ix])
    with (M/'check_states.bin').open('wb') as f:f.write(struct.pack('<I',len(check)));check.tofile(f)
    subprocess.run([str(A/'native/score'),str(M),str(M/'check_states.bin'),str(M/'native_check.bin')],check=True)
    with (M/'native_check.bin').open('rb') as f:n=np.fromfile(f,'<u4',1)[0];native=np.fromfile(f,'<f4').reshape(n,2)
    with torch.no_grad():pred=torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray((check-mm)/ss)))).numpy()
    expected=np.column_stack([lr.predict_proba((check-mu)/sd)[:,1],pred]);error=np.max(np.abs(expected-native),axis=0);assert error.max()<2e-5
    record=dict(target=target,training_L=100,samples=len(a),queries=1200,MLP_best_epoch=best,weighted_positive_fraction=float(np.average(y,weights=weights)),native_errors=error.tolist(),constant_columns=np.flatnonzero(cc).tolist(),hashes={str(M/f'{n}.bin'):sha(M/f'{n}.bin') for n in ['state','lr']})
    dump(M/'COMPLETE.json',record);results.append(record);print('TRAINED',record,flush=True)
dump(D/'TRAIN_COMPLETE.json',dict(complete=True,models=results,feature_and_sample_matched=True,labels_source=sha(D/'LABELS_TRAIN_COMPLETE.json')))
