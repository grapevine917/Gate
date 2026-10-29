#!/usr/bin/env python3
"""Run after original baseline finishes; all outputs stay beside this script."""
import os, sys, json, struct, csv, subprocess, time, hashlib
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent; P=E.parents[1]; D=P/'DiskANN'; LS=[20,50,100,150]
ENV=os.environ.copy(); ENV.update(MKL_NUM_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
EXP=np.dtype([('node','<u4'),('degree','<u4'),('fresh','<u4'),('admitted','<u4'),('exact','<f4'),('cached','<u4')])
EDGE=np.dtype([('parent','<u4'),('child','<u4'),('pq','<f4'),('flags','<u4')])
def run(args,log,extra=None):
 env=ENV.copy(); env.update(extra or {})
 (E/(log+'.command.json')).write_text(json.dumps({'args':args,'env':extra or {}},indent=2))
 with (E/log).open('w') as f: subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def search(name,ls=LS,extra=None):
 args=[str(D/'build/apps/search_disk_index'),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(E/'index'),'--query_file',str(P/'data/sift/sift_query.fbin'),'--gt_file',str(P/'data/sift/gt100.bin'),'-K','10','-L',*map(str,ls),'--result_path',str(E/name),'-W','4','-T','4','--num_nodes_to_cache','10000']
 run(args,name+'.log',extra)
def binmat(path,dtype):
 with open(path,'rb') as f:
  n,d=struct.unpack('<II',f.read(8)); a=np.fromfile(f,dtype=dtype)
 assert a.size==n*d
 return a.reshape(n,d)
def blocks(path):
 with open(path,'rb') as f:
  assert f.read(8)==b'LGTRACE1'
  while True:
   h=f.read(20)
   if not h: break
   assert len(h)==20
   q,L,v,n,m=struct.unpack('<IIIII',h)
   ex=np.frombuffer(f.read(n*24),dtype=EXP); ed=np.frombuffer(f.read(m*16),dtype=EDGE)
   assert len(ex)==n and len(ed)==m
   yield q,L,v,ex,ed
GT=binmat(P/'data/sift/gt100.bin','<u4')
def metrics(name,L):
 a=binmat(E/f'{name}_{L}_idx_uint32.bin','<u4'); b=binmat(E/f'baseline_{L}_idx_uint32.bin','<u4')
 dist=binmat(E/f'{name}_{L}_dists_float.bin','<f4'); bd=binmat(E/f'baseline_{L}_dists_float.bin','<f4')
 hits=np.array([len(set(x)&set(y[:10])) for x,y in zip(a,GT)])
 return {'recall_at_10':float(hits.mean()/10),'identical_queries':int(np.all(a==b,axis=1).sum()),'identical_ids':bool(np.array_equal(a,b)),'identical_distances':bool(np.array_equal(dist,bd)),'queries':len(a)}
def analyze(L):
 results=binmat(E/f'baseline_{L}_idx_uint32.bin','<u4'); rows=[]; seen=set(); total_edges=0
 masks={k:(E/f'{k}_{L}.mask').open('w') for k in ['skip_zero_new','skip_zero_admitted','skip_off_path']}
 for q,l,v,ex,ed in blocks(E/f'trace_{L}.trace'):
  assert l==L and q not in seen; seen.add(q)
  assert len(ex)==len(set(map(int,ex['node'])))
  assert len(ed)==int(ex['degree'].sum())
  assert int(ex['fresh'].sum())+1==v
  assert int(((ed['flags']&1)>0).sum())==int(ex['fresh'].sum())
  assert int(((ed['flags']&2)>0).sum())==int(ex['admitted'].sum())
  assert np.all((ed['flags']&2==0)|((ed['flags']&1)!=0))
  parent={int(x['child']):int(x['parent']) for x in ed if x['flags']&1}
  ancestors=set()
  for node in results[q]:
   node=int(node); chain=set()
   while node in parent:
    node=parent[node]; assert node not in chain; chain.add(node); ancestors.add(node)
  ids=set(map(int,ex['node'])); assert ancestors<=ids
  useful=int((ex['admitted']>0).sum()); useless=len(ex)-useful; zero_new=int((ex['fresh']==0).sum())
  r={'query':q,'L':L,'visited':v,'expanded':len(ex),'useful':useful,'useless':useless,'zero_new':zero_new,'path_useful':len(ancestors),'off_path':len(ex)-len(ancestors),'neighbor_edges':len(ed),'fresh_neighbors':int(ex['fresh'].sum()),'admitted_neighbors':int(ex['admitted'].sum()),'cached_expansions':int(ex['cached'].sum()),'useless_fraction':useless/len(ex),'zero_new_fraction':zero_new/len(ex),'off_path_fraction':(len(ex)-len(ancestors))/len(ex)}
  rows.append(r); total_edges+=len(ed)
  for x in ex:
   node=int(x['node'])
   for k,flag in [('skip_zero_new',x['fresh']==0),('skip_zero_admitted',x['admitted']==0),('skip_off_path',node not in ancestors)]:
    if flag: masks[k].write(f'{q} {node}\n')
 for f in masks.values(): f.close()
 assert seen==set(range(10000))
 rows.sort(key=lambda x:x['query'])
 with (E/f'per_query_{L}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 summary={'L':L,'queries':10000,'neighbor_edges':total_edges,'baseline':metrics('baseline',L),'trace':metrics('trace_results',L)}
 assert summary['trace']['identical_ids'] and summary['trace']['identical_distances']
 for k in list(rows[0])[2:]:
  x=np.array([r[k] for r in rows]); summary[k]={'total':float(x.sum()),'mean':float(x.mean()),'median':float(np.median(x)),'p90':float(np.percentile(x,90)),'p95':float(np.percentile(x,95)),'p99':float(np.percentile(x,99)),'min':float(x.min()),'max':float(x.max())}
 for k in ['useless','zero_new','off_path']:
  summary[k+'_weighted_fraction']=summary[k]['total']/summary['expanded']['total']
 return summary
if __name__=='__main__':
 # The bootstrap process is writing baseline.log; require completed artifacts.
 for L in LS:
  assert (E/f'baseline_{L}_idx_uint32.bin').exists(), 'Baseline has not finished'
 run(['cmake','--build',str(D/'build'),'--target','search_disk_index','-j','4'],'trace_build.log')
 print('Logger built. Running unlogged parity and full trace.',flush=True)
 search('logger_off'); search('trace_results',extra={'DISKANN_TRACE_PREFIX':str(E/'trace')})
 summaries=[]
 for L in LS:
  assert metrics('logger_off',L)['identical_ids'] and metrics('logger_off',L)['identical_distances']
  summaries.append(analyze(L)); print('Analyzed L=',L,flush=True)
 (E/'summary.json').write_text(json.dumps(summaries,indent=2))
 print('Running counterfactual neighborhood skips on every query.',flush=True)
 for mode in ['skip_zero_new','skip_zero_admitted','skip_off_path']:
  search(mode,extra={'DISKANN_SKIP_PREFIX':str(E/mode),'DISKANN_TRACE_PREFIX':str(E/(mode+'_trace'))})
  for s in summaries:
   L=s['L']; m=metrics(mode,L); s[mode]=m
   skipped=sum(1 for _ in (E/f'{mode}_{L}.mask').open())
   actual=0; visited=0; skipped_nodes=0
   for q,l,v,ex,ed in blocks(E/f'{mode}_trace_{L}.trace'):
    actual+=len(ex); visited+=v; skipped_nodes+=int(((ex['degree']>0)&(ex['fresh']==0)&(ex['admitted']==0)).sum())
   m.update(mask_nodes=skipped,expanded_after=actual,visited_after=visited)
   if mode=='skip_zero_new':
    assert m['identical_ids'] and m['identical_distances']
    assert actual==s['expanded']['total'] and visited==s['visited']['total']
  (E/'summary.json').write_text(json.dumps(summaries,indent=2)); print('Validated',mode,flush=True)
 # Verify supplied ground truth and original distances against exact base data on 32 evenly spaced queries.
 base=binmat(P/'data/sift/sift_base.fbin','<f4'); query=binmat(P/'data/sift/sift_query.fbin','<f4')
 norms=(base.astype('float64')**2).sum(axis=1); checked=[]
 for q in np.linspace(0,9999,32,dtype=int):
  ds=norms+(query[q].astype('float64')**2).sum()-2*(base@query[q]).astype('float64')
  kth=np.partition(ds,9)[9]; supplied=ds[GT[q,:10]]
  assert np.all(supplied<=kth), (q,kth,supplied)
  checked.append(int(q))
 (E/'groundtruth_validation.json').write_text(json.dumps({'checked_queries':checked,'method':'Exact float64 squared norms + float32 dot on integer-valued SIFT; top10 tie boundary checked','passed':True},indent=2))
 provenance={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=D,text=True).strip(),'R':64,'Lbuild':100,'search_budget_GB':0.032,'PQ_bytes':int(binmat(E/'index_pq_compressed.bin','u1').shape[1]),'beamwidth':4,'threads':4,'cache_nodes':10000,'base_n':1000000,'dimension':128,'queries':10000,'L':LS}
 (E/'provenance.json').write_text(json.dumps(provenance,indent=2))
 lines=['# SIFT1M DiskANN neighbor expansion experiment','',f"DiskANN commit: `{provenance['commit']}`; R64, Lbuild100, PQ {provenance['PQ_bytes']} bytes, beam4, cache10000, threads4. Base: 1M x 128; queries: 10000. Ground truth: supplied SIFT top100, additionally checked exact top10 boundary for 32 queries.",'','`visited`: distinct medoid + neighbor IDs encountered (including queue rejections). `expanded`: nodes whose adjacency is processed; this differs from visited and from number of beam iterations. `useful`: at least one newly visited neighbor admitted to PQ candidate queue. `useless`: zero admissions. Admission is not equivalent to improving final Recall@10.','', '| L | Recall@10 | visited/query | expanded/query | useful/query | useless/query | useless % | zero-new % | off-path % |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for s in summaries:
  lines.append(f"| {s['L']} | {100*s['baseline']['recall_at_10']:.3f}% | {s['visited']['mean']:.2f} | {s['expanded']['mean']:.2f} | {s['useful']['mean']:.2f} | {s['useless']['mean']:.2f} | {100*s['useless_weighted_fraction']:.2f}% | {100*s['zero_new_weighted_fraction']:.2f}% | {100*s['off_path_weighted_fraction']:.2f}% |")
 lines+=['','Baseline, logger disabled, and logger enabled produce exactly identical ordered top10 IDs and float distances for every query and L. Trace integrity checks cover unique query IDs, 10000-query completeness, edge/degree counts, fresh count, admitted count, expanded uniqueness, and first-discovery ancestry.','', '## Counterfactual validation','', '| L | skip mask | masked nodes/query | Recall@10 | identical top10 queries |','|---:|---|---:|---:|---:|']
 for s in summaries:
  for mode in ['skip_zero_new','skip_zero_admitted','skip_off_path']:
   m=s[mode]; lines.append(f"| {s['L']} | {mode} | {m['mask_nodes']/10000:.2f} | {m['recall_at_10']*100:.3f}% | {m['identical_queries']}/10000 |")
 lines+=['','All skip masks are per-query hindsight oracles generated from the baseline trace. The intervention skips adjacency distance calculations and neighbor insertion AFTER fetching/scoring the parent. It therefore measures removable neighbor scanning, not saved disk reads, not full node expansion, and not a deployable gate.','', '`skip_zero_new` removes only scans with all neighbors already visited; the rerun must preserve IDs, distances, visited counts and expanded counts exactly. `skip_zero_admitted` removes scans that admitted nothing in baseline; reruns reveal consequences of changing the visited set. `skip_off_path` preserves first-discovery ancestors of the baseline returned top10 and removes other scans; ancestry alone does not guarantee identical search dynamics because the bounded queue can change.','', 'The fraction labelled useless is an empirical local no-progress fraction, not an unconditional safe skipping rate. First-discovery off-path fraction is an oracle opportunity estimate rather than a maximum or proof of necessity. Any learned gate needs held-out queries and online features, plus recall versus actual skipped work measurement.','', 'Trace logging stores all neighbor edges and serializes file writes after each query; logged QPS is instrumented overhead and must not be reported as baseline performance. Results are configuration-specific; SIFT1M numbers do not transfer automatically to other datasets or DiskANN versions.','', 'Artifacts: `summary.json` (means/quantiles/counterfactual parity), `per_query_L.csv`, `trace_L.trace` (all real edges), `trace_format.md`, `*.mask`, baseline/logger/skip logs and results, `provenance.json`, `groundtruth_validation.json`.','', 'Re-run: `../../.venv/bin/python run_experiment.py` from this directory after baseline exists. Baseline command is saved in `baseline_command.txt`; index build command in `build_command.txt`.']
 (E/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('COMPLETE',flush=True)
 print(json.dumps([{k:s[k] for k in ['L','baseline','expanded','useful','useless','zero_new_weighted_fraction']} for s in summaries],indent=2),flush=True)
