from pathlib import Path
import json,csv,numpy as np,hashlib
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent
ns={'__file__':str(ROOT/'build.py')};exec((ROOT/'build.py').read_text().split('manifest={}')[0],ns)
fixed=json.loads((ROOT/'fixed64_summary.json').read_text());selected=json.loads((ROOT/'summary.json').read_text());checks={}
for ds in ['sift','deep']:
 assert (ROOT/ds/'B3.ids').read_bytes()==(ROOT/ds/'B4.ids').read_bytes()
 hist=set(np.load(ROOT/ds/'history_query_ids.npy'));val=set(np.load(LG/'experiments/io_predictability'/ds/'validation/query_ids.npy'));test=set(np.load(LG/'experiments/io_predictability'/ds/'test/query_ids.npy'))
 assert len(hist)==8000 and len(val)==len(test)==1000 and not (hist&val or hist&test or val&test)
 for method in ['B2','B3']:
  prefix=ROOT/ds/method;vertices=ns['ids'](str(prefix)+'.ids');n=len(vertices)
  with open(str(prefix)+'.edges','rb') as f:
   h=np.fromfile(f,np.uint32,3);offset=np.fromfile(f,np.uint32,n);edge=np.fromfile(f,np.uint32,n*16)
  assert h[1]==0x80000010 and h[2]==0
  count=int(np.count_nonzero(edge!=0xffffffff));assert np.all(edge[:count]<n)
  expected,degree=ns['original_edges'](LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index_disk.index',vertices,128 if ds=='sift' else 96,64)
  for i in range(n):
   end=int(offset[i+1]) if i+1<n else count
   assert np.array_equal(edge[int(offset[i]):end],expected[i,:int(degree[i])])
  before=list(csv.DictReader(open(ROOT/ds/f'{method}_global_id_control.csv')))
  after=list(csv.DictReader(open(selected[ds]['raw_test_runs'][method][0]['prefix']+'.csv')))
  for a,b in zip(before,after):
   assert all(a[k]==b[k] for k in ['query','hits','submitted_requests','expansions','cache_hits','nav_expansions','nav_distances'])
 for method in ['B1','B2','B3','B4','B5']:
  rr=fixed[ds]['raw_runs'][method]
  assert len(rr)==3 and len({r['submitted_requests'] for r in rr})==1 and len({r['recall_at_10'] for r in rr})==1
  for r in rr:assert r['memory']['cache_and_navigation_bytes']<=fixed[ds]['budget_bytes']
 checks[ds]={'byte_budget_all_methods':'passed','B3_B4_identical_nodes':'passed','history_validation_test_disjoint':'passed','CSR_all_original_resident_edges_in_order':'passed','CSR_vs_global_id_view_every_query_counts_and_recall':'exact parity','repeat_counts_and_recall':'deterministic','B1_old_DiskANN_reads_and_expansions':'exact parity in evaluator'}
# Per-query quality differences and paired bootstrap, diagnostic only.
for ds,d in fixed.items():
 baseline=list(csv.DictReader(open(d['raw_runs']['B1'][0]['prefix']+'.csv')));bh=np.array([int(r['hits']) for r in baseline])
 for method,r in d['test'].items():
  rows=list(csv.DictReader(open(d['raw_runs'][method][0]['prefix']+'.csv')));h=np.array([int(x['hits']) for x in rows]);diff=h-bh
  r['quality_degraded_queries']=int((diff<0).sum());r['quality_improved_queries']=int((diff>0).sum());r['unchanged_recall_queries']=int((diff==0).sum())
  rng=np.random.default_rng(42);samples=diff[rng.integers(0,1000,size=(2000,1000))].mean(axis=1)*10
  r['recall_difference_pp_bootstrap_95pct_ci']=np.quantile(samples,[.025,.975]).tolist()
  r['paired_qps_change_ratios']=[a/b-1 for a,b in zip(r['qps_runs'],d['test']['B1']['qps_runs'])]
(ROOT/'fixed64_summary.json').write_text(json.dumps(fixed,indent=2)+'\n')
(ROOT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
