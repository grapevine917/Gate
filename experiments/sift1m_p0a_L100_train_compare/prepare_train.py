"""Matched single-L training for a fixed inference-L100 comparison."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import hashlib,json,struct,subprocess
import numpy as np
import torch,joblib
from torch import nn
from sklearn.linear_model import LogisticRegression
D=Path(__file__).resolve().parent;W=D.parent/'sift1m_wide_p0';SEED=20261024;torch.set_num_threads(4)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
protocol=dict(dataset='SIFT1M',base_vectors=1000000,dimensions=128,inference_L=100,training_L=[100,300],beam=4,K=10,threads=4,MKL_threads=1,cache_nodes=10000,training_queries=1200,internal_holdout_queries=200,validation_queries=1000,test_queries=1000,seed=SEED,training='Train separate fresh LR and17-128-64-1 MLP for single-L100 and single-L300; same query IDs and same sampled candidate count per query for both L values; same labels and rows within each MLP/LR pair',constant_features='Zero first-layer/linear weights for columns constant in training. A single-L model cannot learn L dependence; this prevents arbitrary extrapolation of a constant L column when L300 is evaluated at L100. Applied identically to both training groups.',label='retain iff admitted_neighbors>0 OR in final baseline Top10 at its training L; same contribution proxy',epochs='max80, inner1000/200-query epoch selection then full1200-query refit; same initialization seed and architecture across L',static_threshold='Native scores on full validation baseline L100 cache-miss candidates, retain-FNR1% for each model',dynamic='Only trainL300 MLP/LR;8 presets from wide experiment, recalibrated at inferenceL100; select validation minimumP95 subject to Recall>=matching static trainL300 default; no retraining for dynamic policy',test='8 groups,16 balanced cyclic repeats (8 positions x2); one shared DiskANN and one shared Heuristic pool; no historical timing reuse',families=['DiskANN','Heuristic','MLP_train100','LR_train100','MLP_train300','LR_train300','DynamicMLP_train300','DynamicLR_train300'],heuristic=dict(rank=72,expansions=100),risks=[0,.001,.0025,.005,.01,.02,.05,.1],dynamic_presets=json.loads((W/'PROTOCOL.json').read_text())['dynamic_presets'],limits=['Previously inspected query split; exploratory','Both single-L model groups are freshly matched retrainings, not the historical L100 model weights','Previous wide model was mixed L30..1000, not pureL300; do not relabel it','Proxy FNR is not final Recall loss','Single training seed'])
if (D/'PROTOCOL.json').exists():assert json.loads((D/'PROTOCOL.json').read_text())==protocol
else:dump(D/'PROTOCOL.json',protocol)
dump(D/'INPUT_LOCK.json',{str(p):sha(p) for p in [W/'data/mixed.prune.bin',W/'data/train_ids.npy',W/'data/internal_holdout_ids.npy',W/'data/validation.fbin',W/'data/validation.gt',W/'data/test.fbin',W/'data/test.gt',D/'native/search',D/'native/collect']})
if not (D/'native/score').exists():
    source='''#include "online_gate.h"
#include <iostream>
int main(int argc,char**argv){if(argc!=4)return 2;mkl_set_num_threads_local(1);std::string m=argv[1];diskann::OnlineGate lr(m+"/lr.bin","",0),st(m+"/state.bin","",0);std::ifstream f(argv[2],std::ios::binary);std::ofstream g(argv[3],std::ios::binary);uint32_t n;f.read((char*)&n,4);g.write((char*)&n,4);float x[17],q[128]={},out[2];for(unsigned i=0;i<n;++i){f.read((char*)x,sizeof(x));out[0]=lr.score(q,0,x);out[1]=st.score(q,0,x);g.write((char*)out,sizeof(out));}if(!f)return 3;}
'''
    (D/'score.cpp').write_text(source);cmds=json.loads((W/'build_commands.json').read_text());cmd=next(c for c in cmds if str(W/'verify.cpp') in c);cmd=[str(D/'score.cpp') if x==str(W/'verify.cpp') else str(D/'native/score') if x==str(W/'native/verify') else x for x in cmd]
    dump(D/'score_build_command.json',cmd)
    with (D/'score_build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
raw=np.fromfile(W/'data/mixed.prune.bin','<f4').reshape(-1,20)
sub={L:raw[raw[:,18]==L] for L in [100,300]};chosen={100:[],300:[]}
for qi in range(1200):
    indices={L:np.flatnonzero(sub[L][:,0]==qi) for L in [100,300]};n=min(map(len,indices.values()));rng=np.random.default_rng(SEED+qi)
    for L in [100,300]:chosen[L].extend(np.sort(rng.choice(indices[L],n,replace=False)).tolist())
datasets={L:sub[L][chosen[L]] for L in [100,300]};assert len(datasets[100])==len(datasets[300])
assert np.array_equal(datasets[100][:,0],datasets[300][:,0])
ids=np.load(W/'data/train_ids.npy');hold=np.load(W/'data/internal_holdout_ids.npy')
class Net(nn.Module):
    def __init__(self,constant):
        super().__init__();self.seq=nn.Sequential(nn.Linear(17,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        with torch.no_grad():self.seq[0].weight[:,constant]=0
    def forward(self,a):return self.seq(a).squeeze(-1)
def norm(a):
    mu=a.mean(0,dtype=np.float64).astype('f4');sd=a.std(0,dtype=np.float64).astype('f4');constant=sd<1e-8;sd[constant]=1;return mu,sd,constant
training=[]
for L in [100,300]:
    M=D/f'models_train{L}';M.mkdir(exist_ok=True)
    if (M/'COMPLETE.json').exists():training.append(json.loads((M/'COMPLETE.json').read_text()));continue
    p=datasets[L];x=np.ascontiguousarray(p[:,3:]);y=p[:,2].copy();inner=~np.isin(ids[p[:,0].astype(int)],hold)
    assert set(np.unique(x[:,15]))=={L} and np.isfinite(x).all()
    mu,sd,constant=norm(x);z=np.ascontiguousarray((x-mu)/sd)
    lr=LogisticRegression(C=1,max_iter=1500,tol=1e-5,random_state=SEED).fit(z,y);lr.coef_[:,constant]=0
    with (M/'lr.bin').open('wb') as f:
        f.write(b'LRGATE01');f.write(struct.pack('<I',0))
        for a in [mu,sd,-lr.coef_[0],-lr.intercept_]:np.asarray(a,'<f4').tofile(f)
    joblib.dump(dict(model=lr,mean=mu,scale=sd,constant=constant),M/'lr.joblib');print('TRAIN',L,'LR DONE samples',len(x),flush=True)
    def fit(mask,epochs,validation=None):
        mm,ss,cc=norm(x[mask]);az=torch.from_numpy(np.ascontiguousarray((x-mm)/ss));yt=torch.from_numpy(y);torch.manual_seed(SEED);net=Net(cc);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[40,60],gamma=.3)
        ix=np.flatnonzero(mask);rng=np.random.default_rng(SEED);hist=[];best=1;bestloss=float('inf');stale=0
        for ep in range(1,epochs+1):
            net.train();total=0.
            for batch in np.array_split(rng.permutation(ix),max(1,int(np.ceil(len(ix)/2048)))):
                loss=nn.functional.binary_cross_entropy_with_logits(net(az[batch]),yt[batch]);opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch)
            scheduler.step();row=dict(epoch=ep,train_loss=total/len(ix))
            if validation is not None:
                net.eval()
                with torch.no_grad():vl=float(nn.functional.binary_cross_entropy_with_logits(net(az[validation]),yt[validation]))
                row['internal_loss']=vl
                if vl<bestloss-1e-5:bestloss=vl;best=ep;stale=0
                else:stale+=1
            hist.append(row)
            if ep%10==0:print('TRAIN',L,'inner' if validation is not None else 'refit',row,flush=True)
            if validation is not None and stale>=12:break
        return net,mm,ss,cc,hist,best
    _,_,_,_,hist,best=fit(inner,80,~inner);dump(M/'internal_history.json',hist)
    model,mm,ss,cc,hist,_=fit(np.ones(len(x),bool),best);dump(M/'history.json',hist);model.eval()
    assert torch.count_nonzero(model.seq[0].weight[:,cc]).item()==0
    with (M/'state.bin').open('wb') as f:
        f.write(b'STGATE01');mm.astype('<f4').tofile(f);ss.astype('<f4').tofile(f)
        for layer in model.seq:
            if isinstance(layer,nn.Linear):layer.weight.detach().numpy().T.copy().astype('<f4').tofile(f);layer.bias.detach().numpy().astype('<f4').tofile(f)
    torch.save(dict(weights=model.state_dict(),mean=mm,scale=ss,constant=cc),M/'state.pt')
    # Check both training-domain states and L100 inference states against native math.
    rng=np.random.default_rng(SEED);check=np.concatenate([x[rng.choice(len(x),1000,replace=False)],datasets[100][rng.choice(len(datasets[100]),1000,replace=False),3:]])
    with (M/'check_states.bin').open('wb') as f:f.write(struct.pack('<I',len(check)));check.astype('<f4').tofile(f)
    subprocess.run([str(D/'native/score'),str(M),str(M/'check_states.bin'),str(M/'native_check.bin')],check=True)
    with (M/'native_check.bin').open('rb') as f:n=np.fromfile(f,'<u4',1)[0];native=np.fromfile(f,'<f4').reshape(n,2)
    with torch.no_grad():pred=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray((check-mm)/ss)))).numpy()
    expected=np.column_stack([lr.predict_proba((check-mu)/sd)[:,1],pred]);err=np.max(np.abs(native-expected),axis=0);assert np.max(err)<2e-5,err
    record=dict(training_L=L,samples=len(x),queries=len(np.unique(p[:,0])),retain_fraction=float(y.mean()),MLP_best_epoch=best,constant_columns=np.flatnonzero(cc).tolist(),native_errors=dict(LR=float(err[0]),MLP=float(err[1])),models={n:sha(M/f'{n}.bin') for n in ['lr','state']})
    dump(M/'COMPLETE.json',record);training.append(record);print('TRAIN',L,'COMPLETE',record,flush=True)
dump(D/'TRAIN_COMPLETE.json',dict(complete=True,training=training,query_matched=True,sample_count_matched=True,models_fresh=True))
(D/'data/calibration_L.txt').write_text('100\n')
if not (D/'CALIBRATION_COLLECTION_COMPLETE.json').exists():
    cmd=[D/'native/collect',D.parent/'sift1m/index',W/'data/validation.fbin',W/'data/validation.gt',W/'models/dummy.bin',D/'data/calibration_L.txt',D/'data/calibration',100000]
    with (D/'collect_validation.log').open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
    dump(D/'CALIBRATION_COLLECTION_COMPLETE.json',dict(complete=True,inference_L=100))
print('PREPARE/TRAIN COMPLETE',flush=True)
