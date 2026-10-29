import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,subprocess,random,csv
import numpy as np
R=Path(__file__).resolve().parent;P=json.loads((R/'PROTOCOL.json').read_text());CS=json.loads((R/'configs.json').read_text());FAMS=['DiskANN','LR_Gate','XGBoost_Gate','State_MLP','QV_MLP'];METRICS=['p95_us','qps','physical_bytes','reads'];TARGETS=P['targets']
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def savecsv(p,rows):
 if rows:
  with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lock=json.loads((R/'INPUT_LOCK.json').read_text());assert all(sha(Path(p))==h for p,h in lock.items())
def complete(prefix):
 try:
  m=json.loads(Path(str(prefix)+'.meta.json').read_text());return m['queries']==1000 and Path(str(prefix)+'.ids').stat().st_size==80000
 except (OSError,ValueError):return False
def load(split,configs,rep=0):
 out=R/'runs'/split;out.mkdir(parents=True,exist_ok=True);todo=[c for c in configs if not complete(out/f'{c["key"]}_r{rep}')];rng=random.Random(P['seed']+rep);groups={fam:[c for c in todo if c['family']==fam] for fam in FAMS}
 for cs in groups.values():rng.shuffle(cs)
 batches=[]
 while any(groups.values()):
  order=list(FAMS);rng.shuffle(order)
  for fam in order:
   if groups[fam]:batches.append(groups[fam][:16]);groups[fam]=groups[fam][16:]
 done=0
 for batch,cs in enumerate(batches):
  cp=R/f'{split}_r{rep}_b{batch}.configs';cp.write_text(''.join(f'{c["key"]} {c["kind"]} {c["L"]} {c["tau"]:.17g} {c["et_tau"]:.17g} {c["rank"]} {c["exp"]}\n' for c in cs))
  cmd=[str(R/'search'),P['index'],str(Path(P['query_dir'])/f'{split}.fbin'),str(Path(P['query_dir'])/f'{split}.gt'),P['model_dir'],P['base_vectors'],str(cp),str(out),str(rep)]
  with (R/f'{split}_r{rep}_b{batch}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  done+=len(cs);print(f'{split} repeat={rep} completed={done}/{len(todo)} family={cs[0]["family"]}',flush=True)
 with (Path(P['query_dir'])/f'{split}.gt').open('rb') as f:n,d=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4',count=int(n*d)).reshape(n,d)[:,:10]
 rows=[]
 for c in configs:
  prefix=out/f'{c["key"]}_r{rep}';meta=json.loads(Path(str(prefix)+'.meta.json').read_text());a=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True);assert len(a)==1000
  ids=np.fromfile(str(prefix)+'.ids','<u8').reshape(1000,10);hits=np.array([len(set(v)&set(g)) for v,g in zip(ids,gt)]);np.testing.assert_array_equal(hits,a['hits']);assert meta['reader_requests']==int(a['reads'].sum());assert meta['reader_bytes']==meta['reader_requests']*4096;assert meta['linux_read_bytes']==meta['reader_bytes']
  assert np.all(a['reads']<=a['expansions'])
  if c['kind'] in [2,3,4,5]:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
  rows.append(dict(**c,split=split,repeat=rep,total_hits=int(hits.sum()),recall=float(hits.sum()/100),reads=float(a['reads'].mean()),physical_bytes=meta['linux_read_bytes']/1000,resident_kib=meta['resident_kib'],reader_bytes=meta['reader_bytes']/1000,expansions=float(a['expansions'].mean()),p95_us=float(np.percentile(a['latency_us'],95)),qps=meta['qps'],mean_us=float(a['latency_us'].mean()),skips=float(a['skips'].mean()),gate_calls=float(a['model_calls'].mean()),et_calls=float(a['et_evaluated'].mean()),et_stop_fraction=float(a['et_stopped'].mean()),prefix=str(prefix),ids_sha256=sha(Path(str(prefix)+'.ids'))))
 return rows

def frontier(rows,metric):
 best={}
 for r in rows:
  h=r['total_hits'];score=-r[metric] if metric=='qps' else r[metric]
  if h not in best or score<(-best[h][metric] if metric=='qps' else best[h][metric]):best[h]=r
 out=[];cost=float('inf')
 for h in sorted(best,reverse=True):
  r=best[h];v=-r[metric] if metric=='qps' else r[metric]
  if v<cost:out.append(r);cost=v
 return sorted(out,key=lambda r:r['total_hits'])
def parity(rows):
 base={r['L']:r for r in rows if r['kind']==0};checks=[]
 for r in rows:
  if r['kind']==0 or r['preset']!=0:continue
  b=base[r['L']]
  for k in ['ids_sha256','total_hits','reads','expansions','physical_bytes']:assert r[k]==b[k],(r['key'],k)
  assert r['skips']==0 and r['et_stop_fraction']==0;checks.append(r['key'])
 return checks
previous=json.loads((R.parent/'main_comparison/validation.json').read_text());wanted={c['key'] for c in CS};validation=[r for r in previous if r['key'] in wanted];assert len(validation)==320
# No new selection from previously observed validation timing; this is a descriptive model ablation.
dump(R/'validation_reference.json',validation);selected=[];vp=[]
allruns=load('test',CS);tp=parity(allruns)
previous_test={r['key']:r for r in json.loads((R.parent/'main_comparison/aggregate.json').read_text())}
for r in allruns:
 for k in ['ids_sha256','total_hits','reads','expansions','skips','gate_calls']:assert r[k]==previous_test[r['key']][k],(r['key'],k)
dump(R/'test_initial.json',allruns);rounds=0
while True:
 current=[]
 for c in CS:
  rs=[r for r in allruns if r['key']==c['key']];r=dict(rs[0]);assert len(rs) in [1,3]
  for field in ['ids_sha256','total_hits','reads','physical_bytes','expansions','skips','gate_calls','et_calls','et_stop_fraction']:assert len({q[field] for q in rs})==1
  for field in ['p95_us','qps','mean_us']:r[field]=float(np.median([q[field] for q in rs]))
  r['timing_repeats']=len(rs);current.append(r)
 keys={s['key'] for s in selected if s['key']}
 for fam in FAMS:
  for metric in METRICS:keys.update(r['key'] for r in frontier([r for r in current if r['family']==fam],metric) if r['recall']>=89)
 common=set.intersection(*[{r['total_hits'] for r in current if r['family']==fam and r['recall']>=90} for fam in FAMS])
 for h in common:
  for fam in FAMS:
   for metric in METRICS:
    r=min((r for r in current if r['family']==fam and r['total_hits']==h),key=lambda r:-r[metric] if metric=='qps' else r[metric]);keys.add(r['key'])
 missing=[c for c in CS if c['key'] in keys and next(r for r in current if r['key']==c['key'])['timing_repeats']<3]
 if not missing:break
 rounds+=1;print('TIMING REFINEMENT',rounds,'configs',len(missing),flush=True)
 for rep in [1,2]:allruns+=load('test',missing,rep)
 assert rounds<=320
 dump(R/'all_test_runs_checkpoint.json',allruns)
dump(R/'all_test_runs.json',allruns);dump(R/'aggregate.json',current);savecsv(R/'all_configurations.csv',current)
frontiers={metric:{fam:frontier([r for r in current if r['family']==fam],metric) for fam in FAMS} for metric in METRICS};dump(R/'frontiers.json',frontiers)
fr=[]
for metric in METRICS:
 for fam in FAMS:
  for r in frontiers[metric][fam]:
   if r['recall']>=89:assert r['timing_repeats']==3;fr.append(dict(objective=metric,**r))
savecsv(R/'frontier_points.csv',fr)
exact=[]
for h in sorted(common):
 for fam in FAMS:
  for metric in METRICS:
   r=min((r for r in current if r['family']==fam and r['total_hits']==h),key=lambda r:-r[metric] if metric=='qps' else r[metric]);assert r['timing_repeats']==3
   exact.append(dict(objective=metric,frontier_member=r['key'] in {q['key'] for q in frontiers[metric][fam]},**r))
savecsv(R/'strict_equal_recall.csv',exact)
held=[];by={r['key']:r for r in current}
for s in selected:
 r=by.get(s['key']);held.append(dict(**s,test_recall=r['recall'] if r else None,test_cost=r[s['metric']] if r else None,test_meets_target=bool(r and r['recall']>=s['target']),test_L=r['L'] if r else None,test_tau=r['tau'] if r else None,test_et_tau=r['et_tau'] if r else None))
savecsv(R/'validation_locked_test.csv',held)
# Descriptive finite-grid feasible-frontier selections. Always expose actual Recall.
feasible_rows=[]
for target in TARGETS:
 for metric in METRICS:
  for fam in FAMS:
   fs=[r for r in current if r['family']==fam and r['recall']>=target]
   r=min(fs,key=lambda r:-r[metric] if metric=='qps' else r[metric]) if fs else None
   if r:feasible_rows.append(dict(target_recall=target,objective=metric,**r))
savecsv(R/'test_feasible_frontier_DESCRIPTIVE.csv',feasible_rows)
assert all(sha(Path(p))==h for p,h in lock.items())
# Same native-library disabled path matches prior P0-A baseline and original main comparison.
orig=R.parent/'p0_a/runs/Vanilla_r0.ids';test_base=next(r for r in current if r['family']=='DiskANN' and r['L']==100);assert test_base['ids_sha256']==sha(orig)
cache_hashes={sha(p) for p in (R/'runs').glob('*/cache_nodes.bin')};assert len(cache_hashes)==1
v=dict(complete=True,validation_configs=len(validation),test_configs=len(current),test_runs=len(allruns),timing_refinement_rounds=rounds,all_primary_final_frontier_and_exact_points_three_repeats=True,validation_disabled_parity=vp,test_disabled_parity=tp,recomputed_recall_from_ids=True,all_repeats_ids_counts_exact=True,physical_bytes_measured_equal_reader_bytes_all_runs=True,reader_requests_equal_stats_all_runs=True,requests4096_bytes_verified=True,models_queries_index_source_hashes_unchanged=True,baseline_L100_matches_P0A_all_ids=True,cache_nodes_sha256=next(iter(cache_hashes)),exact_common_recall_pct=[h/100 for h in sorted(common)])
dump(R/'verification.json',v);dump(R/'COMPLETE.json',v);print('P0-C ONLINE COMPLETE',json.dumps(v),flush=True)
