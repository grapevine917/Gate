from common import *
import copy,time,joblib,sys,sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits
os.sched_setaffinity(0,set(CPUS));manifest=json.loads((DATA/'manifest.json').read_text());names=manifest['pre_io_state_columns'];assert sorted(GROUPS['Global']+GROUPS['Candidate'])==list(range(17))
protocol=dict(experiment='Global / Candidate / Full17 pruning-MLP ablation',groups={k:dict(indices=v,n_features=len(v),names=[names[j] for j in v]) for k,v in GROUPS.items()},global_candidate_disjoint=True,seed_list=SEEDS,train_queries=8000,validation_queries=1000,test_queries=1000,train_samples=manifest['splits']['train']['expansion_samples'],training_label=manifest['label_definition'],architecture='d-128-64-1, ReLU/ReLU/sigmoid, BCE Adam; first-layer parameter count varies with d',training=dict(batch_size=2048,lr=.001,alpha=.0001,max_epochs=60,patience=8,min_delta=.0001,selection='validation logloss only',normalization='training only'),L=100,K=10,beam=4,threads=4,cache_nodes=10000,cache_policy='same protected static BFS cache; classify miss candidates only',history_and_secondary_risk='disabled for all three groups',native='existing E1 independently instrumented graph library; mapped ABGATE01 models',state_construction='same live 17D reconstruction for every group then gather selected inputs; this ablates model inputs, not separately optimized feature-construction code',primary_R_min_pct=PRIMARY,R_min_sensitivity_pct=RMINS,curve_threshold_source='validation conservative-retain FNR quantiles plus zero; then true validation graph Recall constraint',FNR_grid=FNR,per_group_selection='per seed and R_min: minimum validation SSD requests subject to absolute Recall floor; ties smaller tau; no test threshold selection',primary_comparison='compare real I/O and inference-inclusive latency at the primary-R_min validation-selected I/O-optimal working point for each group/seed; if I/O winner and latency winner differ, report both; do not force a unique overall winner',curve_repeats=1,primary_timing_repeats=5,timing_order_seed=20261024,timing='all model training/audit/profile/curve runs finish before randomized serial timing; each native process warms all queries; 4 fixed distinct physical cores where available; inference, normalization and state feature construction included',CPU_affinity=CPUS,confidence='paired query bootstrap nested within matched training seeds and timing rounds; host noise and previously observed test set limit inference',existing_test_previously_observed=True,sklearn=sklearn.__version__,python=sys.version)
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==protocol
else:dump(R/'PROTOCOL.json',protocol)
ids={s:set(np.load(DATA/s/'query_ids.npy').tolist()) for s in ['train','validation','test']};assert not(ids['train']&ids['validation'] or ids['train']&ids['test'] or ids['test']&ids['validation'])
y={s:np.load(DATA/s/'y.npy') for s in ids};raw={s:np.load(DATA/s/'pre_io_state.npy',mmap_mode='r') for s in ids}
for s in ids:assert np.array_equal(y[s],np.load(DATA/s/'metadata.npy',mmap_mode='r')['label'])
for group,indices in GROUPS.items():
 x={s:np.array(v[:,indices],np.float32,order='C') for s,v in raw.items()};mean=x['train'].mean(0,dtype=np.float64).astype(np.float32);ss=np.zeros(len(indices),float)
 for j in range(0,len(x['train']),8192):ss+=((x['train'][j:j+8192].astype(float)-mean)**2).sum(0)
 scale=np.sqrt(ss/len(x['train'])).astype(np.float32);scale[scale==0]=1
 for s in x:x[s]-=mean;x[s]/=scale
 for seed in SEEDS:
  folder=R/'models'/f'{group}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
  if (folder/'COMPLETE.json').exists():continue
  model=MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',solver='adam',batch_size=2048,learning_rate_init=.001,alpha=.0001,shuffle=True,random_state=seed,early_stopping=False);best=float('inf');stale=0;hist=[];start=time.perf_counter();log('TRAIN '+folder.name)
  with threadpool_limits(limits=4):
   for epoch in range(1,61):
    model.partial_fit(x['train'],y['train'],classes=np.array([0,1]));p=predict(model,x['validation']);loss=float(log_loss(y['validation'],p,labels=[0,1]));hist.append(dict(epoch=epoch,train_loss=float(model.loss_),validation_logloss=loss))
    if loss<best-.0001:best=loss;best_epoch=epoch;bestmodel=copy.deepcopy(model);stale=0
    else:stale+=1
    dump(folder/'training_history.json',hist)
    if epoch%5==0:log(f'{folder.name} epoch={epoch} best_epoch={best_epoch} stale={stale}')
    if stale>=8:break
   model=bestmodel;vp=predict(model,x['validation']);tp=predict(model,x['test'])
  bundle=dict(model=model,mean=mean,scale=scale,feature_indices=indices,feature_names=[names[j] for j in indices]);joblib.dump(bundle,folder/'model.joblib');np.save(folder/'validation_p_retain.npy',vp);np.save(folder/'test_p_retain.npy',tp)
  positive=np.sort(vp[y['validation']==1]);thresholds=[dict(key='tau0',tau=0.,target_FNR=None)]
  for j,risk in enumerate(FNR):thresholds.append(dict(key=f'fnr{j:02d}',tau=float(positive[min(int(np.floor(risk*len(positive))),len(positive)-1)]),target_FNR=risk))
  # Deduplicate equal native float32 thresholds, which are the actual deployed comparisons.
  unique={}
  for t in thresholds:
   t['tau']=float(np.float32(t['tau']));unique.setdefault(t['tau'],t)
  dump(folder/'thresholds.json',list(unique.values()))
  with (folder/'model.bin').open('wb') as f:
   f.write(b'ABGATE01');np.asarray([len(indices)]+indices,'<u4').tofile(f)
   for a in [mean,scale]:np.asarray(a,'<f4').tofile(f)
   for w,b in zip(model.coefs_,model.intercepts_):np.asarray(w,'<f4',order='C').tofile(f);np.asarray(b,'<f4').tofile(f)
  metrics=dict(group=group,seed=seed,features=bundle['feature_names'],indices=indices,architecture=[len(indices),128,64,1],parameter_count=int(sum(v.size for v in model.coefs_+model.intercepts_)),best_epoch=best_epoch,epochs=len(hist),training_seconds=time.perf_counter()-start,validation_logloss=float(log_loss(y['validation'],vp,labels=[0,1])),test_AUROC=float(roc_auc_score(y['test'],tp)),model_bytes=(folder/'model.bin').stat().st_size);dump(folder/'metrics.json',metrics);dump(folder/'COMPLETE.json',dict(complete=True,model_sha256=sha(folder/'model.bin'),thresholds_sha256=sha(folder/'thresholds.json')));log('TRAIN COMPLETE '+folder.name)
log('ALL THREE GROUPS / THREE SEEDS TRAINED')
