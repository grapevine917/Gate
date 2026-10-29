import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,subprocess,hashlib,csv
import numpy as np
R=Path(__file__).resolve().parent;M=R.parent/'main_comparison';S=R.parent/'sift1m';P=json.loads((R/'PROTOCOL.json').read_text());O=R/'runs';O.mkdir(exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
lock=json.loads((R/'INPUT_LOCK.json').read_text());assert all(sha(Path(p))==h for p,h in lock.items())
cs=[('Vanilla',0,0,-1,-1),('Heuristic',1,0,72,100),('Ours',4,P['gate_threshold'],0,-1)]
for rep in range(P['repeats']):
 order=cs[rep%3:]+cs[:rep%3];config=R/f'configs_r{rep}.txt';config.write_text(''.join(f'{k} {kind} 100 {tau:.17g} 1.1 {rank} {exp}\n' for k,kind,tau,rank,exp in order))
 if all((O/f'{c[0]}_r{rep}.meta.json').exists() for c in cs):continue
 args=[str(R/'search'),str(S/'index'),P['query_file'],P['gt_file'],P['model'],'unused',str(config),str(O),str(rep)]
 with (R/f'run_r{rep}.log').open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('REPEAT',rep,'COMPLETE',flush=True)
with Path(P['gt_file']).open('rb') as f:n,d=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4',count=int(n*d)).reshape(n,d)[:,:10]
rows=[];arrays={};ids_by={}
for name,*_ in cs:
 for rep in range(P['repeats']):
  prefix=O/f'{name}_r{rep}';a=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True);meta=json.loads(Path(str(prefix)+'.meta.json').read_text());ids=np.fromfile(str(prefix)+'.ids','<u8').reshape(int(n),10)
  hits=np.array([len(set(r)&set(g)) for r,g in zip(ids,gt)]);np.testing.assert_array_equal(hits,a['hits']);assert meta['reader_requests']==int(a['reads'].sum());assert meta['reader_bytes']==meta['reader_requests']*4096
  if rep:
   np.testing.assert_array_equal(ids,ids_by[name]);
   for field in ['hits','reads','expansions','skips','model_calls']:np.testing.assert_array_equal(a[field],arrays[name][field])
  else:arrays[name]=a;ids_by[name]=ids
  rows.append(dict(method=name,repeat=rep,recall_pct=float(hits.sum()/100),p95_ms=float(np.percentile(a['latency_us'],95)/1000),qps=meta['qps'],requests_per_query=float(a['reads'].mean()),requested_bytes_per_query=meta['reader_bytes']/int(n),linux_read_bytes_per_query=meta['linux_read_bytes']/int(n),expansions_per_query=float(a['expansions'].mean()),skips_per_query=float(a['skips'].mean()),model_calls_per_query=float(a['model_calls'].mean())))
summary=[]
for name,*_ in cs:
 rs=[r for r in rows if r['method']==name];s=dict(method=name,repeats=len(rs))
 for k in rs[0]:
  if k not in ['method','repeat']:s[k]=float(np.median([r[k] for r in rs]))
 for k in ['p95_ms','qps']:s[k+'_min']=min(r[k] for r in rs);s[k+'_max']=max(r[k] for r in rs)
 summary.append(s)
baseline=next(r for r in json.loads((M/'aggregate.json').read_text()) if r['key']=='DiskANN_L100');assert sha(O/'Vanilla_r0.ids')==baseline['ids_sha256'];assert summary[0]['requests_per_query']==baseline['reads'];assert summary[0]['recall_pct']==baseline['recall']
assert np.all(arrays['Vanilla']['model_calls']==0);assert np.all(arrays['Heuristic']['model_calls']==0)
# Paired query bootstrap for recall loss and I/O reduction, conditional on frozen policies.
rng=np.random.default_rng(20261020);ci={}
for name in ['Heuristic','Ours']:
 ix=rng.integers(0,int(n),(5000,int(n)));rec=(arrays['Vanilla']['hits']-arrays[name]['hits'])[ix].mean(axis=1)*10;io=(arrays['Vanilla']['reads']-arrays[name]['reads'])[ix].mean(axis=1)
 ci[name]=dict(recall_loss_pp_95CI=np.percentile(rec,[2.5,97.5]).tolist(),requests_saved_95CI=np.percentile(io,[2.5,97.5]).tolist())
assert all(sha(Path(p))==h for p,h in lock.items())
dump(R/'all_runs.json',rows);dump(R/'summary.json',summary);dump(R/'confidence_intervals.json',ci);dump(R/'verification.json',dict(recomputed_recall_from_ids=True,all9_repeats_identical_ids_and_counts=True,reader_requests_equal_query_stats=True,reader_bytes_equal_requests_times4096=True,vanilla_ids_recall_io_exact_previous_L100=True,inputs_unchanged=True,cache_nodes_sha256=sha(O/'cache_nodes.bin'),linux_read_bytes_equal_reader_bytes_all_runs=all(r['linux_read_bytes_per_query']==r['requested_bytes_per_query'] for r in rows)))
with (R/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print(json.dumps(summary,indent=2));print('CI',json.dumps(ci));dump(R/'COMPLETE.json',dict(complete=True,runs=len(rows)))
