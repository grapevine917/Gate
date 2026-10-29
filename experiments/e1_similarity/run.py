#!/usr/bin/env python3
"""E1: frozen independent search traces, disjoint history/test pair analysis."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,hashlib,subprocess,datetime,time,struct,sys,traceback
import numpy as np
import joblib
R=Path(__file__).resolve().parent;LG=R.parents[1];S=LG/'experiments/sift1m';D=R/'data';D.mkdir(exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def log(s):
 line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+s
 print(line,flush=True)
 with (R/'experiment.log').open('a') as f:f.write(line+'\n')
def mat(p,dtype='<f4'):
 with p.open('rb') as f:n,d=np.frombuffer(f.read(8),'<u4');return np.fromfile(f,dtype,count=int(n)*int(d)).reshape(n,d)
def save(p,x):
 with p.open('wb') as f:np.array(x.shape,dtype='<u4').tofile(f);x.tofile(f)
def execute(key,split,model='-',tau=0,marks='-',mode='time'):
 p=R/'runs_main'/f'{split}_{key}_{mode}';p.parent.mkdir(exist_ok=True)
 cmd=[str(R/'native/collect'),str(S/'index'),str(D/f'{split}.fbin'),str(D/f'{split}.gt'),str(model),str(tau),str(marks),str(p),mode]
 meta={'argv':cmd,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'env':{k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
 if not p.with_suffix('.done.json').exists():
  dump(p.with_suffix('.command.json'),meta);log('START '+p.name)
  with p.with_suffix('.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  dump(p.with_suffix('.done.json'),{'complete':True,'csv_sha256':sha(p.with_suffix('.csv'))});log('DONE '+p.name)
 a=np.atleast_1d(np.genfromtxt(p.with_suffix('.csv'),delimiter=',',names=True))
 return p,a
def summary(a):
 return {'recall_pct':float(a['hits'].mean()*10),'reads':float(a['reads'].mean()),'model_calls':float(a['model_calls'].mean()),'skips':float(a['skips'].mean()),'latency_us':float(a['total_us'].mean()),'p95_us':float(np.percentile(a['total_us'],95)),'p99_us':float(np.percentile(a['total_us'],99))}
def prepare():
 q=mat(LG/'data/sift/sift_query.fbin');gt=mat(LG/'data/sift/gt100.bin','<u4')
 ids={s:np.load(S/f'training_data_l100/{s}/query_ids.npy') for s in ['train','validation','test']}
 assert all(not set(ids[a])&set(ids[b]) for a,b in [('train','test'),('train','validation'),('validation','test')])
 history=np.sort(np.random.default_rng(20261018).choice(ids['train'],1000,replace=False))
 ids['history']=history
 for s in ['history','validation','test']:
  save(D/f'{s}.fbin',q[ids[s]]);save(D/f'{s}.gt',gt[ids[s]]);np.save(D/f'{s}_query_ids.npy',ids[s])
 np.save(D/'query_vectors.npy',q);dump(R/'query_splits.json',{s:ids[s].tolist() for s in ['history','validation','test']})
 # The simplest frozen global mark: observed miss >=1 and ALL observed misses low-contribution.
 meta=np.load(S/'training_data_l100/train/metadata.npy',mmap_mode='r');states=np.load(S/'training_data_l100/train/pre_io_state.npy',mmap_mode='r')
 take=np.isin(meta['query_id'],history)&(states[:,10]==0);m=meta[take]
 n=np.bincount(m['node_id'],minlength=1000000)
 low=(m['oracle_zero_admission']!=0)&(m['oracle_in_final_topk']==0)
 z=np.bincount(m['node_id'],weights=low.astype(int),minlength=1000000)
 mask=(n>0)&(n==z)
 with (D/'global_once.bin').open('wb') as f:f.write(b'NDMARK01');f.write(struct.pack('<I',1000000));f.write(np.packbits(mask,bitorder='little').tobytes())
 dump(R/'global_once.json',{'source':'1000 selected training/history queries, original baseline traces only','rule':'at least one miss; all observed misses admitted=0 and not final top10','marked_nodes':int(mask.sum()),'observed_miss_nodes':int((n>0).sum()),'bytes':125000,'no_test_or_validation_observations':True})
 # Native LR uses inverted sigmoid internally; negate a retain-class LR to preserve its meaning.
 b=joblib.load(S/'state_baselines/logistic_regression/model.joblib')
 with (D/'lr.bin').open('wb') as f:
  f.write(b'LRGATE01');f.write(struct.pack('<I',0))
  for x in [b['mean'],b['scale'],-b['model'].coef_[0],-b['model'].intercept_]:np.asarray(x,dtype='<f4').tofile(f)
 import xgboost as xgb
 x=xgb.XGBClassifier(n_jobs=1);x.load_model(S/'state_baselines/xgboost/model.ubj');booster=x.get_booster()
 config=json.loads(booster.save_config());base=config['learner']['learner_model_param']['base_score'];base=float(str(base).strip('[]'))
 trees=booster.get_dump(dump_format='json')[:x.best_iteration+1]
 with (D/'xgboost.bin').open('wb') as f:
  f.write(b'XGGATE01');f.write(struct.pack('<fI',np.log(base/(1-base)),len(trees)))
  for raw in trees:
   nodes={}
   def visit(t):
    nodes[t['nodeid']]=(-1,-1,-1,t['leaf']) if 'leaf' in t else (t['yes'],t['no'],int(t['split'][1:]),t['split_condition'])
    for c in t.get('children',[]):visit(c)
   visit(json.loads(raw));assert sorted(nodes)==list(range(len(nodes)))
   f.write(struct.pack('<I',len(nodes)))
   for i in range(len(nodes)):f.write(struct.pack('<iiif',*nodes[i]))
 dump(R/'xgboost_export.json',{'trees':len(trees),'best_iteration':x.best_iteration,'all_trees':len(booster.get_dump()),'base_score':base,'missing_values':'forbidden by finite-state verification'})
 return ids
def load_trace(p,split,tau):
 ids=np.load(D/f'{split}_query_ids.npy');rows=[];qids=[];steps=[]
 with p.with_suffix('.trace').open('rb') as f:
  while header:=f.read(8):
   qi,n=struct.unpack('<II',header);a=np.frombuffer(f.read(n*19*4),'<f4').reshape(n,19).copy();rows.append(a);qids.extend([ids[qi]]*n);steps.extend(range(n))
 a=np.concatenate(rows);state=a[:,1:18];node=a[:,0].astype(np.uint32);score=a[:,18];cached=state[:,10]!=0
 assert np.isfinite(a).all();assert all(len(set(r[:,0]))==len(r) for r in rows)
 out={'query_id':np.array(qids,np.uint32),'node_id':node,'state_features':state,'model_score':score,'model_decision':((score<tau)&~cached).astype(np.uint8),'search_step':np.array(steps,np.uint32),'cache_hit':cached,'model_invoked':~cached}
 np.savez_compressed(R/f'{split}_trace.npz',**out)
 return out
def main():
 log('E1 START/RESUME');ids=prepare()
 protocol={'dataset':'SIFT1M','index':str(S/'index'),'L':100,'k':10,'beam':4,'threads':4,'cache_nodes':10000,'cache':'static BFS; identical frozen membership; no dynamic updates','history_queries':1000,'test_queries':1000,'validation_queries':1000,'seed':20261018,'target_recall_pct':99.653,'validation_min_recall_pct':99.60,'selection':'largest existing candidate threshold satisfying validation Recall>=99.60%; fallback tau0 if infeasible; no test selection','main_model':'existing 17-128-64-1 state-only MLP; cached nodes retained','training_label':'legacy conservative retain: skip label iff zero admitted neighbours and not final top10; NOT MS MARCO singleton-removal LR label','reuse':'disabled in all trace and cost runs','pairs':'all 1,000,000 directed history->test pairs; pooled L2 distance quantiles; plus per-target nearest-neighbour ranks and per-target quantile bins','cost_categories':['distance kernels','model scoring including standardization','explicit PQ gather and vector copies','synchronous reader request/wait','all remaining search bookkeeping'],'cost_limit':'exclusive instrumented software regions, not hardware attribution; implicit memory stalls remain within executing region; profile runs separate from timing and trace runs','extensions':['LR independent native trajectory','XGBoost independent native trajectory','MLP threshold sweep on validation','test-test pairs, no history construction from test']}
 dump(R/'PROTOCOL.json',protocol)
 models={'MLP':S/'online_state_mlp/model.bin','LR':D/'lr.bin','XGBoost':D/'xgboost.bin'}
 candidates={'MLP':[0]+[v['threshold'] for v in json.loads((S/'online_state_mlp/thresholds.json').read_text()).values()]}
 for name,folder in [('LR','logistic_regression'),('XGBoost','xgboost')]:
  ts=json.loads((S/f'state_baselines/{folder}/thresholds.json').read_text());candidates[name]=sorted(set([0]+[v['threshold'] for k,v in ts.items() if k!='default_0.5']))
 results={'validation':{},'test':{},'profile':{}};selected={}
 execute('DiskANN','validation')
 for name,model in models.items():
  rows=[]
  for j,tau in enumerate(candidates[name]):
   p,a=execute(f'{name}_candidate{j}','validation',model,tau);rows.append({'tau':tau,**summary(a)})
   if tau==0:
    assert sha(p.with_suffix('.ids'))==sha((R/'runs_main/validation_DiskANN_time.ids')),'tau0 must reproduce DiskANN'
  valid=[r for r in rows if r['recall_pct']>=protocol['validation_min_recall_pct']]
  selected[name]=max(valid,key=lambda r:r['tau'])['tau'] if valid else 0
  results['validation'][name]=rows;log(f'SELECT {name} tau={selected[name]}')
 lock={'selected':selected,'test_used':False,'protocol_sha256':sha(R/'PROTOCOL.json')};dump(R/'SELECTION_LOCK.json',lock)
 files=[S/'index_disk.index',S/'index_pq_compressed.bin',S/'index_pq_pivots.bin',*models.values(),R/'native/collect',R/'run.py',R/'collect.cpp',R/'native/pq_flash_index.cpp',R/'native/online_gate.h',R/'cost_profile.h']
 dump(R/'PROVENANCE.json',{'files':{str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},'python':sys.version,'platform':os.uname()._asdict() if hasattr(os.uname(),'_asdict') else list(os.uname()),'splits_disjoint':True,'existing_test_previously_observed':True})
 # Time without audit/profile in three randomized serial configuration orders.
 methods={'DiskANN':('-',0,'-'),'global_once':('-',0,D/'global_once.bin'),**{n:(m,selected[n],'-') for n,m in models.items()}}
 rng=np.random.default_rng(20261018)
 for rep in range(3):
  for name in rng.permutation(list(methods)):
   model,tau,marks=methods[name];p,a=execute(f'{name}_r{rep}','test',model,tau,marks);results['test'].setdefault(name,[]).append(summary(a))
 # Profile and trace are deliberately distinct; exact IDs/reads must match timed execution.
 checks={}
 baseline_ids=np.fromfile(R/'runs_main/test_DiskANN_r0_time.ids','<u8').reshape(-1,10)
 np.testing.assert_array_equal(baseline_ids,mat(S/'baseline_100_idx_uint32.bin','<u4')[ids['test'],:10])
 checks['baseline_matches_existing_index_search']=True
 for name,(model,tau,marks) in methods.items():
  p,a=execute(name,'test',model,tau,marks,'profile');results['profile'][name]={k:float(a[k].mean()) for k in a.dtype.names if k.endswith('_us')}
  ref=R/'runs_main'/f'test_{name}_r0_time';np.testing.assert_array_equal(np.fromfile(p.with_suffix('.ids'),'<u8'),np.fromfile(ref.with_suffix('.ids'),'<u8'))
  costs=sum(a[k] for k in ['distance_us','model_us','memory_us','ssd_us','bookkeeping_us']);assert np.max(np.abs(costs-a['total_us']))<10
  checks[name]={'profile_ids_equal_timed':True,'timed_repeats_identical_ids':all(sha((R/'runs_main'/f'test_{name}_r{r}_time').with_suffix('.ids'))==sha(ref.with_suffix('.ids')) for r in range(3)),'max_cost_partition_error_us':float(np.max(np.abs(costs-a['total_us'])))}
  assert checks[name]['timed_repeats_identical_ids']
 for name,model in models.items():
  traces={}
  for split in ['history','test']:
   p,a=execute(name,split,model,selected[name],mode='trace');traces[split]=load_trace(p,split,selected[name]);(R/f'{split}_trace.npz').rename(R/f'{name}_{split}_trace.npz')
   if split=='test':
    ref=R/'runs_main'/f'test_{name}_r0_time';assert sha(p.with_suffix('.ids'))==sha(ref.with_suffix('.ids'));checks[name]['trace_ids_equal_timed']=True
  # Verify actual native scores, including exported XGBoost tree ensemble.
  t=traces['test'];take=np.flatnonzero(t['model_invoked'])[:20000];x=t['state_features'][take]
  if name=='MLP':
   b=joblib.load(S/'state_mlp/model.joblib');expected=b['model'].predict_proba((x-b['mean'])/b['scale'])[:,1]
  elif name=='LR':
   b=joblib.load(S/'state_baselines/logistic_regression/model.joblib');expected=b['model'].predict_proba((x-b['mean'])/b['scale'])[:,1]
  else:
   import xgboost as xgb
   b=xgb.XGBClassifier(n_jobs=1);b.load_model(S/'state_baselines/xgboost/model.ubj');expected=b.predict_proba(x)[:,1]
  error=float(np.max(abs(expected-t['model_score'][take])));assert error<2e-4,(name,error);checks[name]['max_native_python_score_error']=error
  # Cached IDs must be identical across independent splits.
  assert sha((R/'runs_main'/f'history_{name}_trace.cache.bin'))==sha((R/'runs_main'/f'test_{name}_trace.cache.bin'))
 dump(R/'results.json',results);dump(R/'verification.json',checks)
 log('SEARCH AND TRACE COMPLETE; starting pair analysis')
 subprocess.run([sys.executable,str(R/'analyze.py')],check=True)
 provenance=json.loads((R/'PROVENANCE.json').read_text())
 for p in [*R.glob('*.py'),R/'PROTOCOL_AMENDMENT.json',R/'data/global_once.bin']:
  provenance['files'][str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
 dump(R/'PROVENANCE.json',provenance)
 artifacts=[p for p in R.rglob('*') if p.is_file() and p.suffix in ['.npz','.npy','.csv','.png','.svg','.pdf','.py','.md','.json','.bin','.fbin','.gt','.trace','.ids'] and p.name not in ['COMPLETE.json','artifact_manifest.json'] and 'pip-cache' not in p.parts]
 dump(R/'artifact_manifest.json',{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(artifacts)})
 dump(R/'COMPLETE.json',{'complete':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':{p.name:sha(p) for p in [R/'results.json',R/'analysis.json',R/'REPORT.md',R/'verification.json',R/'SELECTION_LOCK.json']}})
 log('E1 COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:
  log('FAILED\n'+traceback.format_exc())
  raise
