from pathlib import Path
import os,subprocess,json,csv,random,numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent;BIN=LG/'DiskANN/build/apps/search_ram_navigation'
ENV=os.environ.copy();ENV.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='close',OMP_PLACES='cores')
ENV.pop('RAM_USE_ORIGINAL_EDGES',None)
for k in list(ENV):
 if k.startswith('DISKANN_'):ENV.pop(k)
def execute(ds,method,split,budget,label):
 out=ROOT/ds/split/label;out.parent.mkdir(exist_ok=True)
 graph='B1' if method=='B1' else str(ROOT/ds/method)
 data=LG/'experiments/io_predictability'/ds/split
 args=[str(BIN),'run',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),graph,str(data/'query.fbin'),str(data/'gt.bin'),str(out),str(budget),'1']
 (Path(str(out)+'.command.json')).write_text(json.dumps({'args':args,'env':{**{k:ENV[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES']}}},indent=2)+'\n')
 run_env=ENV.copy()
 with open(str(out)+'.log','w') as f:subprocess.run(args,env=run_env,stdout=f,stderr=subprocess.STDOUT,check=True)
 mem=json.loads(Path(str(out)+'.memory.json').read_text());cap=json.loads((ROOT/f'{ds}_bfs.json').read_text())['bfs_native_cache_bytes']
 assert mem['cache_and_navigation_bytes']<=cap,(ds,method,mem,cap)
 rows=list(csv.DictReader(open(str(out)+'.csv')));assert len(rows)==1000
 timings=list(csv.DictReader(open(str(out)+'.timing.csv')))
 return {'prefix':str(out),'memory':mem,'warm_memory':json.loads(Path(str(out)+'.warm_memory.json').read_text()),
         'recall_at_10':sum(int(r['hits']) for r in rows)/10000,'submitted_requests':sum(int(r['submitted_requests']) for r in rows),
         'mean_requests':np.mean([int(r['submitted_requests']) for r in rows]),'mean_latency_us':np.mean([float(r['end_to_end_us']) for r in rows]),
         'latency_p95_us':float(np.quantile([float(r['end_to_end_us']) for r in rows],.95)),
         'mean_navigation_us':np.mean([float(r['nav_us']) for r in rows]),'mean_backend_phase_us':np.mean([float(r['end_to_end_us'])-float(r['nav_us']) for r in rows]),'mean_ssd_search_us':np.mean([float(r['ssd_search_us']) for r in rows]),
         'mean_navigation_distances':np.mean([int(r['nav_distances']) for r in rows]),
         'mean_navigation_expansions':np.mean([int(r['nav_expansions']) for r in rows]),'qps':float(timings[0]['qps'])}
summary={}
for ds in ['sift','deep']:
 # Re-probe the byte cap with the final accounting code; graph construction never depends on validation/test.
 out=ROOT/f'{ds}_bfs';args=[str(BIN),'probe',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(out)]
 with open(ROOT/f'{ds}_probe.log','w') as f:subprocess.run(args,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 validation={'B1/0':execute(ds,'B1','validation',0,'B1_budget0')};selected={}
 for method in ['B2','B3','B4','B5']:
  for budget in [16,32,64]:
   print(ds,'validation',method,budget,flush=True);validation[f'{method}/{budget}']=execute(ds,method,'validation',budget,f'{method}_budget{budget}')
  feasible=[b for b in [16,32,64] if validation[f'{method}/{b}']['recall_at_10']>=validation['B1/0']['recall_at_10']-1e-12]
  selected[method]={'navigation_budget':min(feasible,key=lambda b:validation[f'{method}/{b}']['submitted_requests']) if feasible else 64,'validation_quality_feasible':bool(feasible)}
 print(ds,'selected',selected,flush=True)
 configs=[('B1',0,'B1')]+[(m,selected[m]['navigation_budget'],m) for m in ['B2','B3','B4','B5']]+[(m,0,m+'_cache_only') for m in ['B2','B3','B5']]
 tests={name:[] for _,_,name in configs}
 for repeat in range(3):
  order=configs.copy();random.Random(20261014+repeat).shuffle(order)
  for method,budget,label in order:
   print(ds,'test repeat',repeat,label,flush=True);tests[label].append(execute(ds,method,'test',budget,f'r{repeat}_{label}'))
 aggregate={}
 for label,runs in tests.items():
  assert len(set(r['submitted_requests'] for r in runs))==1 and len(set(r['recall_at_10'] for r in runs))==1
  keys=['mean_latency_us','latency_p95_us','mean_navigation_us','mean_ssd_search_us','mean_navigation_distances','mean_navigation_expansions','mean_backend_phase_us','qps']
  aggregate[label]={k:float(np.median([r[k] for r in runs])) for k in keys}
  aggregate[label].update(recall_at_10=runs[0]['recall_at_10'],mean_requests=runs[0]['mean_requests'],submitted_requests=runs[0]['submitted_requests'],memory=runs[0]['memory'],qps_runs=[r['qps'] for r in runs],latency_runs_us=[r['mean_latency_us'] for r in runs])
 baseline=aggregate['B1']
 for label,r in aggregate.items():
  r.update(net_request_saving_ratio=1-r['submitted_requests']/baseline['submitted_requests'],recall_difference_pp=(r['recall_at_10']-baseline['recall_at_10'])*100,qps_change_ratio=r['qps']/baseline['qps']-1,latency_change_ratio=r['mean_latency_us']/baseline['mean_latency_us']-1)
 # Original baseline parity at every test query.
 prior=LG/'experiments'/('sift1m/feature_ablation/online/r0_baseline.csv' if ds=='sift' else 'deep1m/online/r0_baseline_L100.csv')
 old=list(csv.DictReader(open(prior)));fresh=list(csv.DictReader(open(tests['B1'][0]['prefix']+'.csv')))
 assert all(int(a['ssd_reads'])==int(b['submitted_requests']) and int(a['expansions'])==int(b['expansions']) for a,b in zip(old,fresh))
 hist=np.load(ROOT/ds/'history_query_ids.npy');val=np.load(LG/'experiments/io_predictability'/ds/'validation/query_ids.npy');test=np.load(LG/'experiments/io_predictability'/ds/'test/query_ids.npy')
 assert not (set(hist)&set(val) or set(hist)&set(test) or set(val)&set(test))
 summary[ds]={'budget_bytes':json.loads((ROOT/f'{ds}_bfs.json').read_text())['bfs_native_cache_bytes'],'validation':validation,'selected_navigation_budgets':selected,'test':aggregate,'raw_test_runs':tests,'prior_baseline_query_parity':True,'query_disjoint':True}
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({ds:d['test'] for ds,d in summary.items()},indent=2))
