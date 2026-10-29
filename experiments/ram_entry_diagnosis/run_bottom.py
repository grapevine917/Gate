from pathlib import Path
import os,json,subprocess,csv,numpy as np,time
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent;top=json.loads((ROOT.parent/'ram_topology/summary.json').read_text());ENV=os.environ.copy();ENV.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_PROC_BIND='close',OMP_PLACES='cores');ENV.pop('RAM_USE_ORIGINAL_EDGES',None)
for k in list(ENV):
 if k.startswith('DISKANN_'):ENV.pop(k)
summary={}
for ds in ['sift','deep']:
 method=top[ds]['selected_history'];directory=ROOT/ds;out=directory/'bottom';args=[str(LG/'DiskANN/build/apps/search_ram_bottom_entry_diagnosis'),str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(ROOT.parent/'ram_topology'/ds/method),str(directory/'query.fbin'),str(directory/'gt.bin'),str(out)];Path(str(out)+'.command.json').write_text(json.dumps({'args':args,'env':{k:ENV[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES']}},indent=2)+'\n');print(ds,'bottom',method,flush=True);start=time.perf_counter()
 with open(str(out)+'.log','w') as f:subprocess.run(args,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 records=list(csv.DictReader(open(str(out)+'.csv')));groups={}
 for r in records:groups.setdefault(r['method'],[]).append(r)
 previous=[r for r in csv.DictReader(open(directory/(method+'.csv'))) if r['method']=='current4'];current=groups['current4'];assert all(int(a['hits'])==int(b['hits']) and int(a['submitted_requests'])==int(b['submitted_requests']) for a,b in zip(current,previous))
 aggregate={};rng=np.random.default_rng(20261015);samples=rng.integers(0,200,(5000,200))
 for m,rr in groups.items():
  delta=np.array([int(a['submitted_requests'])-int(b['submitted_requests']) for a,b in zip(current,rr)]);hit=np.array([int(b['hits'])-int(a['hits']) for a,b in zip(current,rr)])
  if m.startswith('oracle'):assert (delta>=0).all() and (hit>=0).all()
  if m=='oracle_bottom4':assert all(int(r['seed_count'])==4 for r in rr)
  aggregate[m]={'mean_requests':float(np.mean([int(r['submitted_requests']) for r in rr])),'recall_at_10':sum(int(r['hits']) for r in rr)/2000,'mean_cached_seeds':float(np.mean([int(r['cached_seeds']) for r in rr])),'mean_ram_expanded_seeds':float(np.mean([int(r['ram_expanded_seeds']) for r in rr])),'strict_same_results_queries':sum(int(r['strict_same_results']) for r in rr),'queries_fewer_requests':int((delta>0).sum()),'queries_recall_lower':int((hit<0).sum()),'queries_recall_higher':int((hit>0).sum()),'saving_mean':float(delta.mean()),'saving_ci95':np.quantile(delta[samples].mean(axis=1),[.025,.975]).tolist()}
 cand=list(csv.DictReader(open(str(out)+'.candidates.csv')));byq={}
 for r in cand:byq.setdefault(int(r['query']),[]).append(r)
 coverage={'candidate_trials':len(cand),'valuable_candidates':sum(int(r['valuable']) for r in cand),'valuable_uncached_candidates':sum(int(r['valuable']) and not int(r['cached']) for r in cand),'queries_any_valuable_candidate':sum(any(int(r['valuable']) for r in rr) for rr in byq.values()),'queries_any_uncached_valuable_candidate':sum(any(int(r['valuable']) and not int(r['cached']) for r in rr) for rr in byq.values()),'queries_valuable_only_uncached':sum(any(int(r['valuable']) for r in rr) and not any(int(r['valuable']) and int(r['cached']) for r in rr) for rr in byq.values())}
 summary[ds]={'history_method':method,'queries':200,'wall_seconds':time.perf_counter()-start,'aggregate':aggregate,'coverage':coverage,'current4_repeat_parity':True};(ROOT/'bottom_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(ds,aggregate,coverage,flush=True)
