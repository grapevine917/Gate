"""Counterfactual diagnosis and delayed-expansion sensitivity, no refitting."""
from pathlib import Path
import csv,json,collections,numpy as np
ROOT=Path(__file__).resolve().parent;EXP=ROOT.parent
load=lambda p:list(csv.DictReader(open(p)))
def stage(n):return '0-9' if n<10 else '10-39' if n<40 else '40-79' if n<80 else '80+'
def ints(rows,k):return sum(int(r[k]) for r in rows)
summary={};all_cases=[]
for ds in ['sift','deep']:
 raw=load(ROOT/f'{ds}_queries.csv');protection=load(ROOT/f'{ds}_protection.csv');skips=load(ROOT/f'{ds}_skips.csv')
 oldlabels={(r['query'],r['node']):r for r in load(EXP/'io_opportunity'/f'{ds}_L100_candidates.csv')}
 qids=np.load(EXP/'io_predictability'/ds/'test/query_ids.npy')
 groups=collections.defaultdict(list)
 for r in raw:groups[int(r['threshold_index']),r['mode']].append(r)
 thresholds={}
 for t in range(3):
  permanent=groups[t,'permanent'];assert len(permanent)==1000
  key=['0.001','0.005','0.01'][t]
  previous=load(EXP/'io_predictability'/ds/'online/test'/f'r0_state_lr__unsafe_fpr_{key}.csv')
  prev={r['query_row']:r for r in previous}
  def read_ids(path):
   with open(path,'rb') as f:n,d=np.fromfile(f,np.uint32,2);return np.fromfile(f,np.uint32,count=int(n)*int(d)).reshape(int(n),int(d))
  truth=read_ids(EXP/'io_predictability'/ds/'test/gt.bin')[:,:10]
  predicted=read_ids(EXP/'io_predictability'/ds/'online/test'/f'r0_state_lr__unsafe_fpr_{key}_100_idx_uint32.bin')
  baseline_ids=read_ids(EXP/'io_predictability'/ds/'online/test/r0_baseline_100_idx_uint32.bin')
  predicted_hits=[len(set(a)&set(g)) for a,g in zip(predicted,truth)]
  baseline_hits=[len(set(a)&set(g)) for a,g in zip(baseline_ids,truth)]
  for r in permanent:
   assert r['reads']==prev[r['query']]['ssd_reads'] and r['expansions']==prev[r['query']]['expansions']
   assert int(r['hits'])==predicted_hits[int(r['query'])] and int(r['baseline_hits'])==baseline_hits[int(r['query'])]
  failed={r['query'] for r in permanent if int(r['hits'])<int(r['baseline_hits'])}
  p=[r for r in protection if int(r['threshold_index'])==t]
  useful=[r for r in p if int(r['protected_hits'])>int(r['permanent_hits'])]
  byq=collections.defaultdict(list)
  for r in p:byq[r['query']].append(r)
  assert set(byq)==failed
  prior_labels=collections.Counter()
  for r in useful:
   prior=oldlabels.get((r['query'],r['node']))
   prior_labels['not_in_baseline' if prior is None else 'singleton_quality_critical' if prior['singleton_recall_ok']=='0' else 'singleton_safe_saving' if int(prior['singleton_delta'])>0 else 'singleton_safe_non_saving']+=1
   all_cases.append({**r,'dataset':ds,'global_query_id':int(qids[int(r['query'])]),'phase':stage(int(r['completed_expansions']))})
  phases=collections.Counter(stage(int(r['completed_expansions'])) for r in useful)
  failed_skip=[r for r in skips if int(r['threshold_index'])==t and r['query'] in failed]
  stats={'failed_queries':len(failed),'failed_global_query_ids':[int(qids[int(q)]) for q in sorted(failed,key=int)],
         'skipped_events_in_failed_queries':len(failed_skip),'single_protection_improving_events':len(useful),
         'unique_improving_node_ids':len({r['node'] for r in useful}),
         'single_node_full_recovery_queries':sum(any(int(r['protected_hits'])>=int(r['baseline_hits']) for r in rs) for rs in byq.values()),
         'queries_without_single_node_improvement':sum(not any(int(r['protected_hits'])>int(r['permanent_hits']) for r in rs) for rs in byq.values()),
         'improving_events_by_completed_expansion_phase':dict(phases),
         'candidate_rank_quantiles':np.quantile([int(r['candidate_rank']) for r in useful],[0,.25,.5,.75,1]).tolist(),
         'direct_baseline_gt_result_events':sum(r['in_baseline_result']=='1' and r['in_gt10']=='1' for r in useful),
         'prior_singleton_labels_of_improving_nodes':dict(prior_labels)}
  thresholds[str(t)]=stats
 profile={}
 for order,suffix in [('pq',''),('retain','_retain'),('rank','_rank')]:
  g=collections.defaultdict(list)
  for r in load(ROOT/f'{ds}{suffix}_queries.csv'):g[int(r['threshold_index']),r['mode']].append(r)
  entries={}
  for (t,mode),rs in g.items():
   assert len(rs)==1000 and len({r['query'] for r in rs})==1000
   base=ints(rs,'baseline_reads');cost=ints(rs,'reads')
   if mode=='permanent':
    assert all(r['hits']==next(x['hits'] for x in groups[t,'permanent'] if x['query']==r['query']) for r in rs)
   entries[f'{t}/{mode}']={'queries':1000,'baseline_submitted_requests':base,'submitted_requests':cost,
      'net_request_saving_ratio':1-cost/base,'recall_at_10':ints(rs,'hits')/10000,'recall_loss_pp':(ints(rs,'baseline_hits')-ints(rs,'hits'))/100,
      'degraded_queries':sum(int(r['hits'])<int(r['baseline_hits']) for r in rs),
      'original_failures_recovered':sum(int(r['permanent_hits'])<int(r['baseline_hits']) and int(r['hits'])>=int(r['baseline_hits']) for r in rs),
      'newly_degraded_queries':sum(int(r['permanent_hits'])>=int(r['baseline_hits']) and int(r['hits'])<int(r['baseline_hits']) for r in rs),
      'strict_result_preserved_queries':sum(int(r['exact_baseline']) for r in rs),'mean_deferred_seeds':ints(rs,'deferred')/1000,
      'mean_rescued_seeds':ints(rs,'rescued')/1000,'mean_expansions':ints(rs,'expansions')/1000}
  profile[order]=entries
 useful_all=[r for r in protection if int(r['protected_hits'])>int(r['permanent_hits'])]
 bynode=collections.defaultdict(set)
 for r in useful_all:bynode[r['node']].add(r['query'])
 unique={(r['query'],r['node']):r for r in useful_all}
 concentration={'unique_failed_queries':len({r['query'] for r in protection}),
                'unique_improving_query_node_pairs':len(unique),'unique_improving_node_ids':len(bynode),
                'node_distinct_query_support_histogram':dict(collections.Counter(len(q) for q in bynode.values())),
                'max_distinct_failed_queries_per_node':max(map(len,bynode.values())),
                'direct_result_unique_pairs':sum(r['in_baseline_result']=='1' and r['in_gt10']=='1' for r in unique.values())}
 joint={}
 for t in range(3):
  rs=[r for r in load(ROOT/f'{ds}_joint_queries.csv') if int(r['threshold_index'])==t and r['mode']=='protect_causal_joint']
  assert len(rs)==thresholds[str(t)]['failed_queries']
  patched={r['query']:r for r in rs};full=[patched.get(r['query'],r) for r in groups[t,'permanent']]
  joint[str(t)]={'diagnostic_failed_queries':len(rs),'full_recovery_queries':sum(int(r['hits'])>=int(r['baseline_hits']) for r in rs),
                'protected_node_count_histogram':dict(collections.Counter(int(r['protected_nodes']) for r in rs)),
                'max_protected_nodes_per_query':max(int(r['protected_nodes']) for r in rs),
                'oracle_patch_degraded_queries':sum(int(r['hits'])<int(r['baseline_hits']) for r in full),
                'oracle_patch_net_request_saving_ratio':1-ints(full,'reads')/ints(full,'baseline_reads')}
 summary[ds]={'thresholds':thresholds,'concentration':concentration,'delayed_policies':profile,'oracle_joint_protection':joint}
with open(ROOT/'improving_protection_nodes.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(all_cases[0]));w.writeheader();w.writerows(all_cases)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
