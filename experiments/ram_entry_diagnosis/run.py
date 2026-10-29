from pathlib import Path
import numpy as np,json,subprocess,os,csv,time
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent;TOPO=ROOT.parent/'ram_topology';BIN=LG/'DiskANN/build/apps/search_ram_entry_diagnosis'
ENV=os.environ.copy();ENV.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='close',OMP_PLACES='cores');ENV.pop('RAM_USE_ORIGINAL_EDGES',None)
for k in list(ENV):
 if k.startswith('DISKANN_'):ENV.pop(k)
def read(p,dtype):
 with open(p,'rb') as f:n,d=np.fromfile(f,np.uint32,2);return np.fromfile(f,dtype,int(n)*int(d)).reshape(int(n),int(d))
def write(p,a):
 with open(p,'wb') as f:np.array(a.shape,np.uint32).tofile(f);a.tofile(f)
summary={};top=json.loads((TOPO/'summary.json').read_text())
for ds in ['sift','deep']:
 directory=ROOT/ds;directory.mkdir(exist_ok=True);old=LG/'experiments/io_predictability'/ds/'test';rows=np.sort(np.random.default_rng(20261015).choice(1000,200,replace=False));np.save(directory/'test_rows.npy',rows);np.save(directory/'query_ids.npy',np.load(old/'query_ids.npy')[rows]);write(directory/'query.fbin',read(old/'query.fbin',np.float32)[rows]);write(directory/'gt.bin',read(old/'gt.bin',np.uint32)[rows]);summary[ds]={}
 for method in ['Original','Geometry',top[ds]['selected_history']]:
  output=directory/method;args=[str(BIN),str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(TOPO/ds/method),str(directory/'query.fbin'),str(directory/'gt.bin'),str(output),'200'];Path(str(output)+'.command.json').write_text(json.dumps({'args':args,'env':{k:ENV[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES']}},indent=2)+'\n');print(ds,method,'start',flush=True)
  with open(str(output)+'.log','w') as f:subprocess.run(args,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
  records=list(csv.DictReader(open(str(output)+'.csv')));groups={}
  for r in records:groups.setdefault(r['method'],[]).append(r)
  aggregate={}
  for m,rr in groups.items():aggregate[m]={'recall_at_10':sum(int(r['hits']) for r in rr)/2000,'mean_requests':np.mean([int(r['submitted_requests']) for r in rr]),'mean_backend_us':np.mean([float(r['backend_us']) for r in rr]),'mean_nav_us':np.mean([float(r['nav_us']) for r in rr]),'mean_selection_us':np.mean([float(r['selection_us']) for r in rr]),'strict_same_results_queries':sum(int(r['strict_same_results']) for r in rr),'seed_counts':sorted(set(int(r['seed_count']) for r in rr))}
  current=groups['current4'];prior=list(csv.DictReader(open(top[ds]['raw_test_runs'][method][0]['prefix']+'.csv')));assert all(int(r['hits'])==int(prior[int(original)]['hits']) and int(r['submitted_requests'])==int(prior[int(original)]['submitted_requests']) for r,original in zip(current,rows))
  pair={};rng=np.random.default_rng(20261015);samples=rng.integers(0,200,(5000,200))
  for m,rr in groups.items():
   if m=='diagnostic_cost':continue
   delta=np.array([int(a['submitted_requests'])-int(b['submitted_requests']) for a,b in zip(current,rr)]);hit=np.array([int(b['hits'])-int(a['hits']) for a,b in zip(current,rr)])
   if m.startswith('oracle'):assert (hit>=0).all() and (delta>=0).all()
   if m in ['oracle_expanded4','oracle_pool4']:assert all(int(r['seed_count'])==4 for r in rr)
   pair[m]={'request_saving_mean':float(delta.mean()),'request_saving_ci95':np.quantile(delta[samples].mean(axis=1),[.025,.975]).tolist(),'queries_fewer_requests':int((delta>0).sum()),'queries_more_requests':int((delta<0).sum()),'queries_recall_lower':int((hit<0).sum()),'queries_recall_higher':int((hit>0).sum())}
  cand=list(csv.DictReader(open(str(output)+'.candidates.csv')));byq={}
  for r in cand:byq.setdefault(int(r['query']),[]).append(r)
  coverage={'candidate_trials':len(cand),'valuable_candidates':sum(int(r['valuable']) for r in cand),'queries_any_valuable_candidate':sum(any(int(r['valuable']) for r in rr) for rr in byq.values()),'queries_valuable_expanded_but_not_selected':sum(any(int(r['valuable']) and int(r['expanded']) and not int(r['selected']) for r in rr) for rr in byq.values()),'queries_valuable_only_outside_expanded':sum(any(int(r['valuable']) for r in rr) and not any(int(r['valuable']) and int(r['expanded']) for r in rr) for rr in byq.values()),'queries_any_selected_valuable':sum(any(int(r['valuable']) and int(r['selected']) for r in rr) for rr in byq.values())}
  summary[ds][method]={'aggregate':aggregate,'paired_vs_current':pair,'candidate_coverage':coverage,'metadata':json.loads(Path(str(output)+'.meta.json').read_text()),'current4_prior_per_query_parity':True}
  (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(ds,method,{m:(r['mean_requests'],r['recall_at_10']) for m,r in aggregate.items()},coverage,flush=True)
