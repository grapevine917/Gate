import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,subprocess,random,hashlib,csv
import numpy as np
BASE=Path(__file__).resolve().parent;R=BASE/'dense';R.mkdir(exist_ok=True);LG=BASE.parents[1];PRE=BASE.parent/'counterfactual_supervision';S=BASE.parent/'sift1m';DATA=PRE/'data';E1=BASE.parent/'e1_similarity'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvsave(p,rows):
 if not rows:return
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
first=float(json.loads((E1/'SELECTION_LOCK.json').read_text())['selected']['MLP'])
base=[c for c in json.loads((PRE/'configs.json').read_text()) if c['family']=='MLP'];taus=sorted(set([c['tau'] for c in base]+[float(t) for t in np.arange(.02,.2001,.002)]))
configs=[]
for i,t in enumerate(taus):
 configs.append(dict(key=f'MLP_{i:03}',family='MLP',kind=0,tau=t,secondary=0.))
 configs.append(dict(key=f'Score_{i:03}',family='ScoreCascade',kind=2,tau=first,secondary=t))
risks=[0.]+list(np.geomspace(1e-5,.1,101))+[.2,.3,.5,.85,1.01]
for i,r in enumerate(risks):configs.append(dict(key=f'CF_{i:03}',family='CFCascade',kind=1,tau=first,secondary=float(r)))
protocol=dict(stage='Original MLP Keep-only counterfactual-risk cascade',dataset='SIFT1M',L=100,K=10,beam=4,threads=4,cache_nodes=10000,first_stage_threshold=first,first_stage_source='frozen E1 original threshold matching E4-1 label generation',models_frozen=True,new_training=False,grid_refinement='Uniform original-score grid step0.002 in [0.02,0.2] plus previous63; risk log grid101 plus endpoints. Exploratory refinement after seeing coarse curves, before any dense-grid evaluations; not a fresh confirmatory test.',features='same17D state; CF called only after original MLP Keep on cache misses; no I/O-gain gate',families={'MLP':'one-stage p<tau','CFCascade':'p<fixed_tau OR (p>=fixed_tau AND CF_risk<rho)','ScoreCascade':'p<fixed_tau OR (p>=fixed_tau AND p<secondary_tau), equivalent to p<max(fixed_tau,secondary_tau)'},query_split_source=str(PRE/'PROTOCOL.json'),queries_per_split=250,baseline_thresholds=taus,risk_thresholds=risks,curve_comparison='Predeclare grids before evaluation; sweep same grid on validation and test independently; compare policies only at EXACT SAME integer total GT hits across all three families, never same threshold or same Recall lower bound',exact_point_choice='For each empirical hit count in three-way intersection, min mean SSD configuration per family; no interpolation; report all exact common points in Recall range [99,100], with secondary skip counts',interpretation='Matched test-curve configurations use test outcomes to establish empirical operating points. Descriptive same-Recall comparison on fixed previously observed queries, not unbiased held-out deployment selection. Separate validation-locked comparisons also saved; no model training/weights/feature changes using test.',timing='one graph/cache load per batch, warm full queries before every config; randomized serial configs; exact-matched chosen test configs repeated two additional times, three totals; 4 query threads; no cross-query state or mutable policy history',uncertainty='2000 paired-query bootstrap draws on fixed matched configurations; does not rematch Recall per bootstrap or cover curve selection uncertainty',limitations=['E4 pool previously observed; no fresh confirmatory test set','joint trajectories shift despite Keep-only support gate','submitted SSD requests not physical reads','two-stage original-score arm analytically equivalent to one-stage threshold; verify every threshold using final IDs and counts'])
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==protocol
else:dump(R/'PROTOCOL.json',protocol)
dump(R/'configs.json',configs)
inputs=[LG/'DiskANN/src/pq_flash_index.cpp',LG/'DiskANN/include/online_gate.h',PRE/'model.bin',S/'online_state_mlp/model.bin',PRE/'PROTOCOL.json',BASE/'native/search',R/'configs.json',R/'PROTOCOL.json',DATA/'validation.fbin',DATA/'test.fbin',DATA/'validation.gt',DATA/'test.gt']
lock={str(p):sha(p) for p in inputs}
if (R/'INPUT_LOCK.json').exists():assert json.loads((R/'INPUT_LOCK.json').read_text())==lock
else:dump(R/'INPUT_LOCK.json',lock)

def batch(split,cs,repeat=0,name=None):
 name=name or split;folder=R/'runs'/name;folder.mkdir(parents=True,exist_ok=True);configfile=R/f'{name}_configs.txt'
 # Existing complete batches are reused only under identical input lock.
 todo=[c for c in cs if not (folder/f'{c["key"]}_r{repeat}.csv').exists()]
 if todo:
  random.Random(20261025+repeat).shuffle(todo)
  configfile.write_text(''.join(f'{c["key"]} {c["kind"]} {c["tau"]:.17g} {c["secondary"]:.17g}\n' for c in todo))
  args=[str(BASE/'native/search'),str(S/'index'),str(DATA/f'{split}.fbin'),str(DATA/f'{split}.gt'),str(S/'online_state_mlp/model.bin'),str(PRE/'model.bin'),str(configfile),str(folder),str(repeat)]
  dump(R/f'{name}_r{repeat}.command.json',args);print('START',name,repeat,len(todo),flush=True)
  with (R/f'{name}_r{repeat}.log').open('w') as out:subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,check=True)
 rows=[]
 with (DATA/f'{split}.gt').open('rb') as f:n,d=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4').reshape(n,d)[:,:10]
 for c in cs:
  prefix=folder/f'{c["key"]}_r{repeat}';a=np.atleast_1d(np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True));assert len(a)==250
  ids=np.fromfile(str(prefix)+'.ids','<u8').reshape(250,10);hits=np.array([len(set(row)&set(g)) for row,g in zip(ids,gt)])
  np.testing.assert_array_equal(hits,a['hits']);assert np.all(a['reads']<=a['expansions'])
  assert np.array_equal(a['skips'],a['first_skips']+a['secondary_skips'])
  rows.append(dict(**c,split=split,repeat=repeat,total_hits=int(hits.sum()),recall=float(hits.sum()/2500*100),reads=float(a['reads'].mean()),expansions=float(a['expansions'].mean()),secondary_calls=float(a['secondary_calls'].mean()),secondary_skips=float(a['secondary_skips'].mean()),latency_us=float(a['total_us'].mean()),prefix=str(prefix),ids_sha256=sha(Path(str(prefix)+'.ids'))))
 return rows

def verify_equivalence(rows):
 indexed={r['key']:r for r in rows};checks=[]
 for c in configs:
  if c['family']!='ScoreCascade':continue
  effective=max(first,c['secondary']);b=next(b for b in configs if b['family']=='MLP' and b['tau']==effective);a=indexed[c['key']];z=indexed[b['key']]
  assert a['ids_sha256']==z['ids_sha256']
  for k in ['total_hits','reads','expansions']:assert a[k]==z[k]
  checks.append(dict(cascade=c['key'],equivalent_baseline=b['key'],effective_threshold=effective,identical_ids_and_counts=True))
 return checks

def exact_matches(rows):
 families=['MLP','CFCascade','ScoreCascade'];by={f:{} for f in families}
 for r in rows:
  if not 99<=r['recall']<=100:continue
  h=r['total_hits'];f=r['family']
  if h not in by[f] or (r['reads'],r['key'])<(by[f][h]['reads'],by[f][h]['key']):by[f][h]=r
 common=sorted(set(by[families[0]])&set(by[families[1]])&set(by[families[2]]),reverse=True)
 return [dict(total_hits=h,recall=h/2500*100,configurations={f:by[f][h] for f in families}) for h in common]

validation=batch('validation',configs);dump(R/'validation.json',validation);ve=verify_equivalence(validation);vm=exact_matches(validation);dump(R/'validation_exact_matches.json',vm)
# Frozen validation-selected empirical points kept separate from test-curve matching.
dump(R/'VALIDATION_SELECTION_LOCK.json',dict(model_frozen=True,test_used=False,exact_validation_points=[dict(total_hits=m['total_hits'],config_keys={f:r['key'] for f,r in m['configurations'].items()}) for m in vm]))
test=batch('test',configs);dump(R/'test.json',test);te=verify_equivalence(test);matches=exact_matches(test);dump(R/'test_exact_matches_initial.json',matches)
assert matches,'No exact three-way workpoint in predeclared range; do not substitute Recall lower bounds'
# Repeat ALL configurations selected for exact empirical test-curve workpoints.
keys={a['key'] for m in matches for a in m['configurations'].values()};chosen=[c for c in configs if c['key'] in keys];repeats=[]
for rep in [1,2]:repeats+=batch('test',chosen,rep,'matched_repeats')
dump(R/'test_repeats.json',repeats)
alltest=test+repeats;flat=[];comparisons=[];boot=np.random.default_rng(20261025).integers(0,250,(2000,250))
for m in matches:
 group={}
 for family,r in m['configurations'].items():
  rs=[a for a in alltest if a['key']==r['key']];assert len(rs)==3
  for k in ['ids_sha256','total_hits','reads','expansions','secondary_calls','secondary_skips']:assert len({a[k] for a in rs})==1
  g=dict(**{k:v for k,v in r.items() if k not in ['latency_us']},latency_us=float(np.median([a['latency_us'] for a in rs])),latency_min=min(a['latency_us'] for a in rs),latency_max=max(a['latency_us'] for a in rs))
  flat.append(g);group[family]=g
 base=group['MLP'];cf=group['CFCascade'];a=np.genfromtxt(base['prefix']+'.csv',delimiter=',',names=True);b=np.genfromtxt(cf['prefix']+'.csv',delimiter=',',names=True);delta=b['reads']-a['reads'];lo,hi=np.percentile(delta[boot].mean(1),[2.5,97.5])
 assert base['total_hits']==cf['total_hits']==group['ScoreCascade']['total_hits']
 comparisons.append(dict(recall=m['recall'],total_hits=m['total_hits'],baseline=base['key'],cf=cf['key'],score_cascade=group['ScoreCascade']['key'],baseline_reads=base['reads'],cf_reads=cf['reads'],score_cascade_reads=group['ScoreCascade']['reads'],cf_ssd_change_pct=(cf['reads']/base['reads']-1)*100,paired_read_delta=float(delta.mean()),paired_read_delta_ci95=[float(lo),float(hi)],cf_secondary_skips=cf['secondary_skips']))
dump(R/'exact_matched_results.json',flat);csvsave(R/'exact_matched_results.csv',flat);dump(R/'paired_comparisons.json',comparisons);csvsave(R/'paired_comparisons.csv',comparisons)
# Keep validation-selected configs' held-out outcomes available without relabeling them same-Recall.
heldout=[];by={r['key']:r for r in test}
for m in vm:
 for f,c in m['configurations'].items():heldout.append(dict(validation_total_hits=m['total_hits'],validation_recall=m['recall'],family=f,key=c['key'],test_total_hits=by[c['key']]['total_hits'],test_recall=by[c['key']]['recall'],test_reads=by[c['key']]['reads']))
csvsave(R/'validation_locked_test_results.csv',heldout)
# Existing original-program parity at original E4 threshold on exactly the same query rows.
source_rows=np.load(DATA/'test_source_rows.npy');orig=np.fromfile(E1/'runs_main/test_MLP_r0_time.ids','<u8').reshape(1000,10)[source_rows];b=next(r for r in test if r['family']=='MLP' and r['tau']==first)
np.testing.assert_array_equal(orig,np.fromfile(b['prefix']+'.ids','<u8').reshape(250,10))
assert all(sha(Path(p))==h for p,h in lock.items())
legacy=[r for r in json.loads((PRE/'validation.json').read_text()) if r['family']=='MLP']
current={r['tau']:r for r in validation if r['family']=='MLP'}
for original in legacy:
 for metric in ['ids_sha256','reads','expansions']:assert original[metric]==current[original['tau']][metric]
dump(R/'verification.json',dict(original_63_threshold_validation_IDs_counts_equal_previous_original_program=True,validation_score_equivalences=len(ve),test_score_equivalences=len(te),all_final_ids_and_counts_equal=True,stage2_native_keep_only_assertions=True,original_E4_threshold_final_ids_equal=True,all_recalls_recomputed_from_IDs=True,three_repeat_ids_counts_identical=True,exact_common_test_points=len(matches),exact_comparison_uses_integer_hit_equality=True,inputs_unchanged=True,validation_equivalences=ve,test_equivalences=te))
dump(R/'ONLINE_COMPLETE.json',dict(configs=len(configs),exact_common_test_points=len(matches),matched_test_configs=len(chosen)))
print('COMPLETE',len(matches),'exact Recall workpoints',flush=True)
