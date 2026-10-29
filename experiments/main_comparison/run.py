import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,subprocess,random,csv
from collections import Counter
import numpy as np
R=Path(__file__).resolve().parent;LG=R.parents[1];S=R.parent/'sift1m';LS=json.loads((R/'PROTOCOL.json').read_text())['L_training_and_learned_grid'];SEED=20261030
NAMES=['DiskANN','Top3_Heuristic','LR_Gate','XGBoost_Gate','State_MLP','QV_MLP','Learned_ET','ET_State_MLP']
METRICS=['reads','p95_us','qps'];TARGETS=[90.,95.,98.,99.,99.5,99.7]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvsave(p,rows):
 if not rows:return
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Gate thresholds are fit-only quantiles on the SAME representative training-state sample.
with (R/'models/native_check.bin').open('rb') as f:n=np.fromfile(f,'<u4',1)[0];a=np.fromfile(f,'<f4').reshape(n,150)
quantiles=[0.,.01,.03,.08,.18,.35,.55,.75];gate_taus={}
for k,name in enumerate(['LR_Gate','XGBoost_Gate','State_MLP','QV_MLP']):gate_taus[name]=[0.]+[float(np.quantile(a[:,146+k],q)) for q in quantiles[1:]]
et_taus=[1.1,.999,.995,.99,.98,.95,.9,.8]
joint_pairs=[(0,0),(0,4),(3,0),(1,1),(2,2),(3,3),(5,4),(6,5)]
heuristics=[(1.,-1),(.98,1.5),(.9,1.25),(.75,1.),(.5,.9),(.25,.75),(.1,.5),(0.,.25)]
baseLs=set(LS)
while len(baseLs)<64:
 v=max((v for v in range(10,301) if v not in baseLs),key=lambda v:min(abs(np.log(v/b)) for b in baseLs));baseLs.add(v)
configs=[]
def cfg(key,kind,L,tau=0.,et_tau=1.1,rank=0,exp=-1,preset=0):return dict(key=key,family=NAMES[kind],kind=kind,L=int(L),tau=float(tau),et_tau=float(et_tau),rank=int(rank),exp=int(exp),preset=preset)
for L in sorted(baseLs):configs.append(cfg(f'DiskANN_L{L}',0,L))
for kind in range(1,8):
 for L in LS:
  for j in range(8):
   c=cfg(f'{NAMES[kind]}_L{L}_p{j}',kind,L,preset=j)
   if kind==1:
    rank,exp=heuristics[j];c['rank']=int(L*rank);c['exp']=-1 if exp==-1 else int(L*exp)
   elif kind in [2,3,4,5]:c['tau']=gate_taus[NAMES[kind]][j]
   elif kind==6:c['et_tau']=et_taus[j]
   else:pi,ei=joint_pairs[j];c['tau']=gate_taus['State_MLP'][pi];c['et_tau']=et_taus[ei]
   configs.append(c)
assert all(v==64 for v in Counter(c['family'] for c in configs).values());dump(R/'configs.json',configs)
dump(R/'TUNING_PROTOCOL.json',dict(configs_per_family=dict(Counter(c['family'] for c in configs)),baseline_L=sorted(baseLs),gate_thresholds=gate_taus,gate_training_quantiles=quantiles,ET_thresholds=et_taus,joint_index_pairs=joint_pairs,Top3_heuristic='existing rank + progress fixed conjunction; two of Top3-importance features, NOT three-feature MLP',heuristic_fraction_pairs=heuristics,calibration_source='same2000 sampled training candidate states only; no validation/test outcomes for grid creation',objectives='three separately optimal frontiers; SSD/P95/QPS may select different configs',interpretation='best within equal64-configuration budget, not exhaustive global optimum'))
locked={str(p):sha(p) for p in [R/'native/search',R/'configs.json',R/'PROTOCOL.json',R/'TUNING_PROTOCOL.json',*[R/f'models/{n}.bin' for n in ['lr','xgb','state','qv','et']]]}
if (R/'RUN_LOCK.json').exists():assert json.loads((R/'RUN_LOCK.json').read_text())==locked
else:dump(R/'RUN_LOCK.json',locked)

def load(split,cs,rep=0):
 folder=R/'runs'/split;folder.mkdir(parents=True,exist_ok=True);todo=[]
 for c in cs:
  p=folder/f'{c["key"]}_r{rep}'
  if not Path(str(p)+'.meta.json').exists():todo.append(c)
 if todo:
  random.Random(SEED+rep+(0 if split=='validation' else 10)).shuffle(todo);path=R/f'{split}_r{rep}_configs.txt'
  path.write_text(''.join(f'{c["key"]} {c["kind"]} {c["L"]} {c["tau"]:.17g} {c["et_tau"]:.17g} {c["rank"]} {c["exp"]}\n' for c in todo))
  args=[str(R/'native/search'),str(S/'index'),str(R/f'data/{split}.fbin'),str(R/f'data/{split}.gt'),str(R/'models'),str(LG/'data/sift/sift_base.fbin'),str(path),str(folder),str(rep)]
  dump(R/f'{split}_r{rep}.command.json',dict(argv=args,env={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},config_order=[c['key'] for c in todo]));print('START',split,rep,len(todo),flush=True)
  with (R/f'{split}_r{rep}.log').open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
 with (R/f'data/{split}.gt').open('rb') as f:n,d=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4',count=int(n)*int(d)).reshape(n,d)[:,:10]
 rows=[]
 for c in cs:
  prefix=folder/f'{c["key"]}_r{rep}';a=np.atleast_1d(np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True));meta=json.loads(Path(str(prefix)+'.meta.json').read_text());assert len(a)==1000 and meta['queries']==1000
  ids=np.fromfile(str(prefix)+'.ids','<u8').reshape(1000,10);hits=np.array([len(set(v)&set(g)) for v,g in zip(ids,gt)]);np.testing.assert_array_equal(hits,a['hits']);assert np.all(a['reads']<=a['expansions'])
  if c['kind'] in [1,2,3,4,5,7]:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
  rows.append(dict(**c,split=split,repeat=rep,total_hits=int(hits.sum()),recall=float(hits.mean()*10),reads=float(a['reads'].mean()),expansions=float(a['expansions'].mean()),latency_us=float(a['latency_us'].mean()),p95_us=float(np.percentile(a['latency_us'],95)),qps=float(meta['qps']),gate_calls=float(a['model_calls'].mean()),et_calls=float(a['et_evaluated'].mean()),et_stop_fraction=float(a['et_stopped'].mean()),skips=float(a['skips'].mean()),prefix=str(prefix),ids_sha256=sha(Path(str(prefix)+'.ids'))))
 return rows

def frontier(rows,metric):
 reverse=metric=='qps';best={}
 for r in rows:
  hit=r['total_hits'];score=-r[metric] if reverse else r[metric]
  if hit not in best or score<(-best[hit][metric] if reverse else best[hit][metric]):best[hit]=r
 out=[];cost=float('inf')
 for hit in sorted(best,reverse=True):
  r=best[hit];v=-r[metric] if reverse else r[metric]
  if v<cost:out.append(r);cost=v
 return sorted(out,key=lambda r:r['total_hits'])

def parity(rows):
 baseline={r['L']:r for r in rows if r['kind']==0};checks=[]
 for r in rows:
  if r['kind']==0 or r['preset']!=0:continue
  b=baseline[r['L']];assert b['ids_sha256']==r['ids_sha256']
  for k in ['total_hits','reads','expansions']:assert b[k]==r[k]
  assert r['skips']==0 and r['et_stop_fraction']==0;checks.append(r['key'])
 return checks
validation=load('validation',configs);dump(R/'validation.json',validation);vp=parity(validation)
selected=[]
for family in NAMES:
 for target in TARGETS:
  feasible=[r for r in validation if r['family']==family and r['recall']+1e-10>=target]
  for metric in METRICS:
   chosen=min(feasible,key=lambda r:-r[metric] if metric=='qps' else r[metric]) if feasible else None
   selected.append(dict(family=family,target=target,metric=metric,key=chosen['key'] if chosen else None,validation_recall=chosen['recall'] if chosen else None,validation_cost=chosen[metric] if chosen else None))
selection=dict(test_used=False,selected=selected,model_hashes=json.loads((R/'TRAIN_COMPLETE.json').read_text())['models'])
if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())==selection
else:dump(R/'SELECTION_LOCK.json',selection)
test=load('test',configs);dump(R/'test.json',test);tp=parity(test)
# Repeated timing for union of empirical per-objective frontier and validation-locked choices.
repeated={};allruns=list(test);rounds=0
while True:
 current=[]
 for c in configs:
  rs=[a for a in allruns if a['key']==c['key']];row=dict(rs[0])
  if len(rs)>1:
   assert len(rs)==3
   for k in ['ids_sha256','total_hits','reads','expansions','skips']:assert len({a[k] for a in rs})==1
   for metric in ['latency_us','p95_us','qps']:row[metric]=float(np.median([a[metric] for a in rs]))
  row['timing_repeats']=len(rs);current.append(row)
 keys={a['key'] for a in selected if a['key']}
 for family in NAMES:
  familyrows=[r for r in current if r['family']==family]
  for metric in METRICS:keys.update(r['key'] for r in frontier(familyrows,metric) if r['recall']>=89.)
 missing=[c for c in configs if c['key'] in keys and c['key'] not in repeated]
 if not missing:break
 rounds+=1;print('TIMING REFINE',rounds,len(missing),flush=True)
 for rep in [1,2]:allruns+=load('test',missing,rep)
 repeated.update({c['key']:True for c in missing})
 # finite configurations: every selected final frontier has3 repeats; no model/grid changes.
 assert rounds<=len(configs)
dump(R/'all_test_runs.json',allruns);dump(R/'aggregate.json',current);csvsave(R/'all_configurations.csv',current)
frontiers={metric:{family:frontier([r for r in current if r['family']==family],metric) for family in NAMES} for metric in METRICS};dump(R/'frontiers.json',frontiers)
heldout=[];by={r['key']:r for r in current}
for s in selected:
 a=by.get(s['key']);heldout.append(dict(**s,test_recall=a['recall'] if a else None,test_cost=a[s['metric']] if a else None,test_meets_recall=bool(a and a['recall']+1e-10>=s['target'])))
csvsave(R/'validation_locked_test.csv',heldout)
# Exact shared Recall observations, if supported; never fabricate via rounded values.
common=set.intersection(*[{r['total_hits'] for r in current if r['family']==f and r['recall']>=90} for f in NAMES]);dump(R/'exact_common_recall.json',dict(integer_total_hits=sorted(common),recall_pct=[h/100 for h in sorted(common)]))
assert all(sha(Path(p))==h for p,h in locked.items());assert all(sha(Path(p))==h for p,h in json.loads((R/'INPUT_LOCK.json').read_text()).items())
dump(R/'verification.json',dict(disabled_actions_IDs_and_counts_exact_baseline=True,validation_disabled_checks=vp,test_disabled_checks=tp,all_recall_recomputed_from_IDs=True,frontier_timing_repeats3=True,all_repeated_IDs_and_counts_identical=True,equal_grid_budget=dict(Counter(c['family'] for c in configs)),input_models_source_unchanged=True,native_score=json.loads((R/'native_score_verification.json').read_text())))
dump(R/'ONLINE_COMPLETE.json',dict(complete=True,configs=len(configs),repeated_configs=len(repeated),timing_refinement_rounds=rounds,test_runs=len(allruns)))
print('ONLINE COMPLETE',flush=True)
