"""Primary all-method comparison at the same 64-node RAM navigation budget."""
from pathlib import Path
import json,random,numpy as np
ROOT=Path(__file__).resolve().parent
ns={'__file__':str(ROOT/'evaluate.py')};exec((ROOT/'evaluate.py').read_text().split('summary={}')[0],ns)
old=json.loads((ROOT/'summary.json').read_text());out={}
# SIFT's validation-selected methods already all use 64.
for ds in ['sift','deep']:
 if ds=='sift':
  runs={m:old[ds]['raw_test_runs'][m] for m in ['B1','B2','B3','B4','B5']}
 else:
  runs={m:[] for m in ['B1','B2','B3','B4','B5']}
  for rep in range(3):
   order=list(runs);random.Random(20261014+rep).shuffle(order)
   for m in order:
    print(ds,rep,m,flush=True);runs[m].append(ns['execute'](ds,m,'test',0 if m=='B1' else 64,f'fixed64_r{rep}_{m}'))
 result={}
 for m,rr in runs.items():
  assert len(set(r['submitted_requests'] for r in rr))==1 and len(set(r['recall_at_10'] for r in rr))==1
  keys=['mean_latency_us','latency_p95_us','mean_navigation_us','mean_ssd_search_us','mean_navigation_distances','mean_navigation_expansions','mean_backend_phase_us','qps']
  result[m]={k:float(np.median([r[k] for r in rr])) for k in keys}
  result[m].update(recall_at_10=rr[0]['recall_at_10'],mean_requests=rr[0]['mean_requests'],submitted_requests=rr[0]['submitted_requests'],memory=rr[0]['memory'],qps_runs=[r['qps'] for r in rr],latency_runs_us=[r['mean_latency_us'] for r in rr])
 base=result['B1']
 for r in result.values():r.update(net_request_saving_ratio=1-r['submitted_requests']/base['submitted_requests'],recall_difference_pp=(r['recall_at_10']-base['recall_at_10'])*100,qps_change_ratio=r['qps']/base['qps']-1,latency_change_ratio=r['mean_latency_us']/base['mean_latency_us']-1)
 out[ds]={'test':result,'raw_runs':runs,'navigation_budget':64,'budget_bytes':old[ds]['budget_bytes']}
 (ROOT/'fixed64_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({ds:d['test'] for ds,d in out.items()},indent=2))
