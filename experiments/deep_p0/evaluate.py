"""DEEP1M P0-A/B/C: native measurement, frozen validation selection and repeats."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,subprocess,random,csv,math,time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
R=Path(__file__).resolve().parent;ROOT=R.parents[1];D=R.parent/'deep1m';P=json.loads((R/'PROTOCOL.json').read_text());LS=P['L_grid'];TARGETS=P['targets'];METRICS=['physical_bytes','p95_us','qps'];SEED=P['seed']
STUDIES={'p0_b':['DiskANN','Heuristic','MLP_L100','ET','ET_MLP_L100'],'p0_b_lr':['DiskANN','Heuristic','LR','ET','ET_LR'],'p0_c':['DiskANN','LR','XGB','MLP_mixed','QV']}
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def savecsv(p,rows):
 if rows:
  with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert (R/'TRAIN_COMPLETE.json').exists()
lock=json.loads((R/'INPUT_LOCK.json').read_text());assert all(sha(k)==v for k,v in lock.items())
trained=json.loads((R/'TRAIN_COMPLETE.json').read_text());assert all(sha(R/'models'/f'{n}.bin')==h for n,h in trained['models'].items())
if not (R/'native_score_verification.json').exists():
 output=subprocess.check_output([str(R/'native/verify'),str(R/'models'),str(ROOT/'data/deep1M_base.fbin')]);v=json.loads(output.decode().strip().splitlines()[-1]);assert all(v[k]<2e-5 for k in ['LR','XGB','StateMLP','QV_MLP','ET']);dump(R/'native_score_verification.json',v)
risks=[0,.001,.0025,.005,.01,.02,.05,.1];thresholds={};prediction=R/'prediction';prediction.mkdir(exist_ok=True);prediction_metrics=[]
for split in ['validation','test']:
 a=np.fromfile(R/'data'/f'{split}.prune.bin','<f4').reshape(-1,20);mask=a[:,13]==0;miss=a[mask];scores_input=prediction/f'{split}.bin'
 with (R/'data'/f'{split}.fbin').open('rb') as f:n,d=np.fromfile(f,'<u4',2);q=np.fromfile(f,'<f4').reshape(n,d)
 raw=np.empty((len(miss),115),'<f4');raw[:,:96]=q[miss[:,0].astype(int)];raw[:,96]=miss[:,1];raw[:,97:114]=miss[:,3:];raw[:,114]=miss[:,2];raw.tofile(scores_input)
 for model in ['l100','lr','xgb','state','qv']:
  out=prediction/f'{split}_{model}.scores';bench=prediction/f'{split}_{model}.bench.json'
  if not out.exists():subprocess.run([str(R/'native/score'),str(R/'models'/f'{model}.bin'),str(ROOT/'data/deep1M_base.fbin'),str(scores_input),str(out),str(bench)],check=True)
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
BASE=set(LS)
while len(BASE)<64:BASE.add(max((v for v in range(10,301) if v not in BASE),key=lambda v:min(abs(math.log(v/b)) for b in BASE)))
ets=[1.1,.999,.995,.99,.98,.95,.9,.8];pairs=[(0,0),(4,0),(0,3),(2,1),(3,2),(4,3),(5,4),(6,5)];heur=[(1,-1),(.98,1),(.92,1.04),(.72,1),(.64,.92),(.5,.9),(.25,.75),(.1,.5)]
families=['DiskANN','Heuristic','LR','XGB','MLP_L100','MLP_mixed','QV','ET','ET_MLP_L100','ET_LR'];model_names={2:'lr',3:'xgb',4:'l100',5:'state',6:'qv',8:'l100',9:'lr'};configs=[]
def make(kind,L,j=0):
 c=dict(key=f'{families[kind]}_L{L}_p{j}',family=families[kind],kind=kind,L=L,preset=j,tau=0.,et_tau=1.1,rank=0,exp=-1)
 if kind==1:c.update(rank=int(L*heur[j][0]),exp=-1 if heur[j][1]<0 else int(L*heur[j][1]))
 elif kind in [2,3,4,5,6]:c['tau']=thresholds[model_names[kind]][j]['tau']
 elif kind==7:c['et_tau']=ets[j]
 elif kind in [8,9]:pi,ei=pairs[j];c.update(tau=thresholds[model_names[kind]][pi]['tau'],et_tau=ets[ei])
 return c
configs=[make(0,L) for L in sorted(BASE)]+[make(kind,L,j) for kind in range(1,10) for L in LS for j in range(8)];assert len(configs)==640;dump(R/'configs.json',configs)
dump(R/'GRID_LOCK.json',dict(configs=640,per_family=64,baseline_L=sorted(BASE),L=LS,gate_retain_FNR_risks=risks,ET_thresholds=ets,joint_threshold_index_pairs=pairs,heuristic_pairs=heur,threshold_source='Full DEEP validation L100 baseline candidates; cache hits protected; same retain-FNR budgets for all Gates; differs from SIFT P0-C training-quantile grid. Never calibrate using test.',studies=STUDIES,model_hashes=trained['models']))
runlock={str(p):sha(p) for p in [R/'configs.json',R/'GRID_LOCK.json',R/'native/search',*[R/'models'/f'{m}.bin' for m in trained['models']]]}
if (R/'RUN_LOCK.json').exists():assert json.loads((R/'RUN_LOCK.json').read_text())==runlock
else:dump(R/'RUN_LOCK.json',runlock)
def complete(prefix):
 try:return prefix.with_suffix('.ids').stat().st_size==80000 and prefix.with_suffix('.csv').exists() and json.loads(prefix.with_suffix('.meta.json').read_text())['queries']==1000
 except (OSError,ValueError):return False
def load(split,cs,rep=0,study=None,ordered=False):
 folder=(R/study/'runs' if study else R/'runs'/split);folder.mkdir(parents=True,exist_ok=True);todo=[c for c in cs if not complete(folder/f'{c["key"]}_r{rep}')];rng=random.Random(SEED+rep+(split=='test')*100)
 if ordered:batches=[[c] for c in todo]
 else:
  rng.shuffle(todo);batches=[]
  for kind in sorted(set(c['kind'] for c in todo)):
   group=[c for c in todo if c['kind']==kind];batches += [group[j:j+8] for j in range(0,len(group),8)]
  rng.shuffle(batches)
 logs=R/study if study else R/'logs';logs.mkdir(exist_ok=True)
 for i,batch in enumerate(batches):
  digest=hashlib.sha256(' '.join(c['key'] for c in batch).encode()).hexdigest()[:12];tag=f'{split}_r{rep}_{digest}';cp=logs/f'{tag}.configs';cp.write_text(''.join(f'{c["key"]} {c["kind"]} {c["L"]} {c["tau"]:.17g} {c["et_tau"]:.17g} {c["rank"]} {c["exp"]}\n' for c in batch))
  cmd=[str(R/'native/search'),str(D/'index'),str(R/'data'/f'{split}.fbin'),str(R/'data'/f'{split}.gt'),str(R/'models'),str(ROOT/'data/deep1M_base.fbin'),str(cp),str(folder),str(rep)];dump(logs/f'{tag}.command.json',dict(argv=cmd,config_keys=[c['key'] for c in batch]))
  with (logs/f'{tag}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  log(f'{study or split} repeat={rep} batch={i+1}/{len(batches)} configs={len(batch)}')
 with (R/'data'/f'{split}.gt').open('rb') as f:n,d=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4',count=int(n*d)).reshape(n,d)[:,:10]
 rows=[]
 for c in cs:
  prefix=folder/f'{c["key"]}_r{rep}';a=np.genfromtxt(prefix.with_suffix('.csv'),delimiter=',',names=True);meta=json.loads(prefix.with_suffix('.meta.json').read_text());ids=np.fromfile(prefix.with_suffix('.ids'),'<u8').reshape(1000,10);assert np.all(ids<1000000) and np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
  hits=np.array([len(set(v)&set(g)) for v,g in zip(ids,gt)]);np.testing.assert_array_equal(hits,a['hits']);assert len(a)==1000
  assert meta['reader_requests']==int(a['reads'].sum()) and meta['reader_bytes']==meta['reader_requests']*4096 and meta['linux_read_bytes']==meta['reader_bytes'],(c['key'],meta)
  if c['kind'] in [1,2,3,4,5,6,8,9]:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
  rows.append(c|dict(split=split,repeat=rep,total_hits=int(hits.sum()),recall=float(hits.sum()/100),reads=float(a['reads'].mean()),physical_bytes=meta['linux_read_bytes']/1000,expansions=float(a['expansions'].mean()),skips=float(a['skips'].mean()),gate_calls=float(a['model_calls'].mean()),et_calls=float(a['et_evaluated'].mean()),et_stop_fraction=float(a['et_stopped'].mean()),p95_us=float(np.percentile(a['latency_us'],95)),qps=meta['qps'],mean_us=float(a['latency_us'].mean()),ids_sha256=sha(prefix.with_suffix('.ids')),prefix=str(prefix)))
 return rows
def aggregate(runs,cs):
 out=[]
 for c in cs:
  rs=[r for r in runs if r['key']==c['key']];assert len(rs) in [1,3,9],(c['key'],len(rs));r=rs[0].copy();r.pop('repeat')
  for k in ['ids_sha256','total_hits','reads','physical_bytes','expansions','skips','gate_calls','et_calls','et_stop_fraction']:assert len({s[k] for s in rs})==1,(c['key'],k)
  for k in ['p95_us','qps','mean_us']:r[k]=float(np.median([s[k] for s in rs]));r[k+'_min']=min(s[k] for s in rs);r[k+'_max']=max(s[k] for s in rs)
  r['timing_repeats']=len(rs);out.append(r)
 return out
def frontier(rows,metric):
 out=[];cost=float('inf')
 for r in sorted(rows,key=lambda r:(-r['total_hits'],-r[metric] if metric=='qps' else r[metric])):
  v=-r[metric] if metric=='qps' else r[metric]
  if v<cost:out.append(r);cost=v
 return sorted(out,key=lambda r:r['total_hits'])
def parity(rows):
 base={r['L']:r for r in rows if r['kind']==0};checked=[]
 for r in rows:
  if r['kind'] and r['preset']==0:
   for k in ['ids_sha256','total_hits','reads','physical_bytes','expansions']:assert r[k]==base[r['L']][k],(r['key'],k)
   assert r['skips']==0 and r['et_stop_fraction']==0;checked.append(r['key'])
 return checked
# P0-A: two independently timed 3-method experiments, balanced cyclic order.
for name,kind in [('p0_a',4),('p0_a_lr',2)]:
 if (R/name/'COMPLETE.json').exists():continue
 cs=[make(0,100),make(1,100,3),make(kind,100,4)];cs[1].update(rank=72,exp=100)
 for label,c in zip(['Vanilla','Heuristic','Ours'],cs):c['key']=label
 runs=[]
 for rep in range(9):order=cs[rep%3:]+cs[:rep%3];runs+=load('test',order,rep,name,True)
 current=aggregate(runs,cs);dump(R/name/'all_runs.json',runs);dump(R/name/'aggregate.json',current);savecsv(R/name/'summary.csv',current);dump(R/name/'COMPLETE.json',dict(complete=True,runs=27,default_gate_risk=.01,verification='all repeats IDs/counts equal, measured bytes equal reader bytes, recall recomputed from IDs'))
 log(f'{name} COMPLETE')
if (R/'validation.json').exists():validation=json.loads((R/'validation.json').read_text())
else:validation=load('validation',configs);dump(R/'validation.json',validation)
vp=parity(validation);selected=[]
for study,fams in STUDIES.items():
 (R/study).mkdir(exist_ok=True)
 for family in fams:
  for target in TARGETS:
   fs=[r for r in validation if r['family']==family and r['recall']>=target]
   for metric in METRICS:
    c=min(fs,key=lambda r:(-r[metric] if metric=='qps' else r[metric],-r['total_hits'],r['key'])) if fs else None
    selected.append(dict(study=study,family=family,target=target,objective=metric,key=c['key'] if c else None,validation_recall=c['recall'] if c else None))
if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())==selected
else:dump(R/'SELECTION_LOCK.json',selected)
initial=load('test',configs);dump(R/'test_initial.json',initial);tp=parity(initial);allruns=list(initial);rounds=0
while True:
 current=aggregate(allruns,configs);keys={s['key'] for s in selected if s['key']}
 for study,fams in STUDIES.items():
  for family in fams:
   rows=[r for r in current if r['family']==family]
   for metric in METRICS:
    keys.update(r['key'] for r in frontier(rows,metric) if r['recall']>=89)
    for target in TARGETS:
     fs=[r for r in rows if r['recall']>=target]
     if fs:keys.add(min(fs,key=lambda r:-r[metric] if metric=='qps' else r[metric])['key'])
  common=set.intersection(*[{r['total_hits'] for r in current if r['family']==family and r['recall']>=90} for family in fams])
  for h in common:
   for family in fams:
    for metric in METRICS:keys.add(min((r for r in current if r['family']==family and r['total_hits']==h),key=lambda r:-r[metric] if metric=='qps' else r[metric])['key'])
 missing=[c for c in configs if c['key'] in keys and next(r for r in current if r['key']==c['key'])['timing_repeats']<3]
 if not missing:break
 rounds+=1;log(f'FRONTIER REPEAT ROUND={rounds} configs={len(missing)}')
 for rep in [1,2]:allruns+=load('test',missing,rep)
 dump(R/'all_test_runs_checkpoint.json',allruns);assert rounds<=640
dump(R/'all_test_runs.json',allruns);dump(R/'aggregate.json',current);savecsv(R/'all_configurations.csv',current)
for study,fams in STUDIES.items():
 rows=[r for r in current if r['family'] in fams];dump(R/study/'aggregate.json',rows);savecsv(R/study/'all_configurations.csv',rows);fs={metric:{family:frontier([r for r in rows if r['family']==family],metric) for family in fams} for metric in METRICS};dump(R/study/'frontiers.json',fs)
 common=set.intersection(*[{r['total_hits'] for r in rows if r['family']==family and r['recall']>=90} for family in fams]);exact=[];feasible=[];held=[];lookup={r['key']:r for r in rows}
 for h in sorted(common):
  for family in fams:
   for metric in METRICS:
    r=min((r for r in rows if r['family']==family and r['total_hits']==h),key=lambda r:-r[metric] if metric=='qps' else r[metric]);assert r['timing_repeats']==3;exact.append(dict(objective=metric,**r))
 for family in fams:
  for metric in METRICS:
   assert all(r['timing_repeats']==3 for r in fs[metric][family] if r['recall']>=89)
   for target in TARGETS:
    rs=[r for r in rows if r['family']==family and r['recall']>=target]
    if rs:best=min(rs,key=lambda r:-r[metric] if metric=='qps' else r[metric]);assert best['timing_repeats']==3;feasible.append(dict(target=target,objective=metric,**best))
 for s in selected:
  if s['study']!=study:continue
  r=lookup.get(s['key']);held.append(s|dict(test_recall=r['recall'] if r else None,test_meets_target=bool(r and r['recall']>=s['target']),test_physical_bytes=r['physical_bytes'] if r else None,test_p95_us=r['p95_us'] if r else None,test_qps=r['qps'] if r else None,L=r['L'] if r else None,tau=r['tau'] if r else None,et_tau=r['et_tau'] if r else None))
 savecsv(R/study/'strict_equal_recall.csv',exact);savecsv(R/study/'test_feasible_frontier_DESCRIPTIVE.csv',feasible);savecsv(R/study/'validation_locked_test.csv',held);dump(R/study/'COMPLETE.json',dict(complete=True,configs=len(rows),test_runs=sum(len([r for r in allruns if r['key']==c['key']]) for c in rows),common_recall_pct=[h/100 for h in sorted(common)],shared_controls=True))
for name in ['p0_a','p0_a_lr']:
 a=json.loads((R/name/'aggregate.json').read_text());b=next(r for r in current if r['family']=='DiskANN' and r['L']==100);assert a[0]['ids_sha256']==b['ids_sha256'] and a[0]['reads']==b['reads']
cache_hashes={sha(p) for p in R.rglob('cache_nodes.bin')};assert len(cache_hashes)==1
assert all(sha(k)==v for k,v in runlock.items());assert all(sha(k)==v for k,v in lock.items())
dump(R/'verification.json',dict(passed=True,validation_disabled_parity=vp,test_disabled_parity=tp,all_repeated_IDs_counts_equal=True,all_selected_frontier_exact_feasible_points_3_repeats=True,recomputed_recall_from_ids=True,reader_and_linux_bytes_equal=True,request_bytes4096_verified=True,shared_cache_sha256=next(iter(cache_hashes)),P0A_baseline_matches_P0BC=True,inputs_models_unchanged=True))
dump(R/'ONLINE_COMPLETE.json',dict(complete=True,validation_configs=len(validation),test_configs=len(current),test_runs=len(allruns),P0A_runs=54,refinement_rounds=rounds));log('DEEP P0 ONLINE COMPLETE')
