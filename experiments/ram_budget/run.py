from pathlib import Path
import numpy as np,json,csv,os,subprocess,random,time
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent;TOPO=ROOT.parent/'ram_topology';BIN=LG/'DiskANN/build/apps/search_ram_navigation'
ENV=os.environ.copy();ENV.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='close',OMP_PLACES='cores',RAM_NAV_QUEUE_STORAGE='512');ENV.pop('RAM_USE_ORIGINAL_EDGES',None)
for k in list(ENV):
 if k.startswith('DISKANN_'):ENV.pop(k)
BUDGETS=[0,4,8,16,32,64,96,128,256,512];REPEATS=5
KEYS=['mean_latency_us','latency_p95_us','mean_navigation_us','mean_navigation_cpu_us','mean_backend_us','mean_distances','mean_expansions','qps']
def execute(ds,method,budget,queue,split,label):
 out=ROOT/ds/split/label;out.parent.mkdir(parents=True,exist_ok=True);data=LG/'experiments/io_predictability'/ds/split;env=ENV.copy();env['RAM_NAV_QUEUE_CAPACITY']=str(queue)
 args=[str(BIN),'run',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(TOPO/ds/method),str(data/'query.fbin'),str(data/'gt.bin'),str(out),str(budget),'1'];Path(str(out)+'.command.json').write_text(json.dumps({'args':args,'env':{k:env[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES','RAM_NAV_QUEUE_STORAGE','RAM_NAV_QUEUE_CAPACITY']}},indent=2)+'\n')
 with open(str(out)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 mem=json.loads(Path(str(out)+'.memory.json').read_text());cap=json.loads((ROOT.parent/'ram_navigation'/f'{ds}_bfs.json').read_text())['bfs_native_cache_bytes'];assert mem['cache_and_navigation_bytes']<=cap,(ds,mem,cap)
 rows=list(csv.DictReader(open(str(out)+'.csv')));assert len(rows)==1000;timing=list(csv.DictReader(open(str(out)+'.timing.csv')))[0]
 return {'prefix':str(out),'method':method,'budget':budget,'queue':queue,'memory':mem,'recall_at_10':sum(int(r['hits']) for r in rows)/10000,'mean_requests':float(np.mean([int(r['submitted_requests']) for r in rows])),'submitted_requests':sum(int(r['submitted_requests']) for r in rows),'mean_latency_us':float(np.mean([float(r['end_to_end_us']) for r in rows])),'latency_p95_us':float(np.quantile([float(r['end_to_end_us']) for r in rows],.95)),'mean_navigation_us':float(np.mean([float(r['nav_us']) for r in rows])),'mean_navigation_cpu_us':float(np.mean([float(r['nav_cpu_us']) for r in rows])),'mean_backend_us':float(np.mean([float(r['end_to_end_us'])-float(r['nav_us']) for r in rows])),'mean_distances':float(np.mean([int(r['nav_distances']) for r in rows])),'mean_expansions':float(np.mean([int(r['nav_expansions']) for r in rows])),'queries_budget_exhausted':sum(int(r['nav_expansions'])==budget for r in rows) if budget else 0,'qps':float(timing['qps'])}
def aggregate(rr):
 assert len({r['submitted_requests'] for r in rr})==1 and len({r['recall_at_10'] for r in rr})==1
 out={k:float(np.median([r[k] for r in rr])) for k in KEYS};out.update({k:rr[0][k] for k in ['method','budget','queue','memory','recall_at_10','mean_requests','submitted_requests','queries_budget_exhausted']});out.update(repeats=len(rr),qps_runs=[r['qps'] for r in rr],latency_runs_us=[r['mean_latency_us'] for r in rr],cpu_runs_us=[r['mean_navigation_cpu_us'] for r in rr]);return out
def paired(a,b):
 aa=list(csv.DictReader(open(a['prefix']+'.csv')));bb=list(csv.DictReader(open(b['prefix']+'.csv')));x=np.array([int(r['submitted_requests'])-int(t['submitted_requests']) for r,t in zip(aa,bb)]);h=np.array([int(t['hits'])-int(r['hits']) for r,t in zip(aa,bb)]);rng=np.random.default_rng(20261016);sample=rng.integers(0,1000,(5000,1000));return {'mean_request_saving':float(x.mean()),'request_saving_ci95':np.quantile(x[sample].mean(axis=1),[.025,.975]).tolist(),'recall_delta_pp':float(h.mean()*10),'recall_delta_pp_ci95':(np.quantile(h[sample].mean(axis=1),[.025,.975])*10).tolist(),'queries_recall_lower':int((h<0).sum()),'queries_recall_higher':int((h>0).sum()),'queries_fewer_requests':int((x>0).sum()),'queries_more_requests':int((x<0).sum())}
summary={};top=json.loads((TOPO/'summary.json').read_text())
for ds in ['sift','deep']:
 methods=['Original','Geometry',top[ds]['selected_history']];history=methods[-1];validation={};configs=[(m,b,64) for m in methods for b in BUDGETS];configs +=[(history,b,q) for q in [8,16,32,128,256] for b in [16,64,256]];random.Random(20261016).shuffle(configs)
 for m,b,q in configs:
  key=f'{m}/{b}/{q}';print(ds,'validation',key,flush=True);validation[key]=[execute(ds,m,b,q,'validation',f'{m}_b{b}_q{q}_r0')]
 selected={};shortlisted={}
 for m in methods:
  target=max(validation[f'{m}/64/64'][0]['recall_at_10'],validation[f'{m}/0/64'][0]['recall_at_10']);eligible=[k for k,rr in validation.items() if rr[0]['method']==m and rr[0]['recall_at_10']>=target-1e-12]
  fast=sorted(eligible,key=lambda k:validation[k][0]['mean_latency_us'])[:2];io=min(eligible,key=lambda k:(validation[k][0]['submitted_requests'],validation[k][0]['mean_navigation_cpu_us']));shortlist=list(dict.fromkeys(fast+[f'{m}/64/64',io]));shortlisted[m]={'quality_target':target,'eligible_configs':eligible,'shortlist':shortlist,'minimum_requests_key':io}
 # Interleave two further validation rounds; final selection uses three-repeat median, not the single fastest run.
 for rep in [1,2]:
  keys=list(dict.fromkeys(k for v in shortlisted.values() for k in v['shortlist']));random.Random(20261016+rep).shuffle(keys)
  for key in keys:
   m,b,q=key.split('/');print(ds,'validation repeat',rep,key,flush=True);validation[key].append(execute(ds,m,int(b),int(q),'validation',f'{m}_b{b}_q{q}_r{rep}'))
 for m,v in shortlisted.items():
  feasible=[k for k in v['shortlist'] if aggregate(validation[k])['recall_at_10']>=v['quality_target']-1e-12];best=min(feasible,key=lambda k:aggregate(validation[k])['mean_latency_us']);selected[m]={'key':best,'quality_target':v['quality_target'],'minimum_requests_key':v['minimum_requests_key'],'validation':aggregate(validation[best])}
 (ROOT/ds/'frozen_selection.json').write_text(json.dumps(selected,indent=2)+'\n');print(ds,'frozen', {m:d['key'] for m,d in selected.items()},flush=True)
 keys=list(dict.fromkeys([f'{methods[0]}/0/64']+[f'{m}/64/64' for m in methods]+[d['key'] for d in selected.values()]+[selected[history]['minimum_requests_key']]));tests={k:[] for k in keys}
 for rep in range(REPEATS):
  order=keys.copy();random.Random(20261020+rep).shuffle(order)
  for key in order:
   m,b,q=key.split('/');print(ds,'test',rep,key,flush=True);tests[key].append(execute(ds,m,int(b),int(q),'test',f'{m}_b{b}_q{q}_r{rep}'))
 # Held-out budget curve for the historical graph; points do not change the frozen choice.
 curve={}
 for b in BUDGETS:
  key=f'{history}/{b}/64'
  if key in tests:curve[str(b)]=aggregate(tests[key])
  elif b==0:curve[str(b)]=aggregate(tests[f'{methods[0]}/0/64'])
  else:
   print(ds,'heldout curve',key,flush=True);curve[str(b)]=aggregate([execute(ds,history,b,64,'test',f'curve_{history}_b{b}_q64')])
 result={k:aggregate(rr) for k,rr in tests.items()};reference=tests[f'{methods[0]}/0/64'][0];prior0=ROOT.parent/'ram_navigation'/ds/'test'
 # Queue=64/nav=64 must preserve all per-query counts from topology experiment despite scratch over-allocation and CPU timers.
 parity={}
 for m in methods:
  previous=list(csv.DictReader(open(top[ds]['raw_test_runs'][m][0]['prefix']+'.csv')));fresh=list(csv.DictReader(open(tests[f'{m}/64/64'][0]['prefix']+'.csv')));assert all(int(a['hits'])==int(b['hits']) and int(a['submitted_requests'])==int(b['submitted_requests']) and int(a['nav_distances'])==int(b['nav_distances']) for a,b in zip(previous,fresh));parity[m]=True
 pairs={m:paired(tests[f'{m}/64/64'][0],tests[d['key']][0]) for m,d in selected.items()};assert len({r['memory']['cache_and_navigation_bytes'] for r in result.values()})==1
 summary[ds]={'budgets':BUDGETS,'queue_storage':512,'history_method':history,'validation':{k:aggregate(rr) for k,rr in validation.items()},'raw_validation':validation,'shortlisted':shortlisted,'selected':selected,'test':result,'raw_test_runs':tests,'history_test_curve':curve,'paired_selected_vs64':pairs,'reference64_prior_query_parity':parity,'frozen_before_test':True}
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(ds,'RESULT',result,'PAIRED',pairs,flush=True)
