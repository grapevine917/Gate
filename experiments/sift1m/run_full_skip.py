#!/usr/bin/env python3
"""Additional full-node oracle intervention; requires run_experiment.py completion."""
from run_experiment import *
if __name__=='__main__':
 assert (E/'REPORT.md').exists()
 summaries=json.loads((E/'summary.json').read_text())
 for L in LS:
  results=binmat(E/f'baseline_{L}_idx_uint32.bin','<u4')
  with (E/f'full_skip_{L}.mask').open('w') as f:
   for q,l,v,ex,ed in blocks(E/f'trace_{L}.trace'):
    final=set(map(int,results[q]))
    for x in ex:
     if x['admitted']==0 and int(x['node']) not in final: f.write(f"{q} {int(x['node'])}\n")
 if '--analyze-only' not in sys.argv:
  run(['cmake','--build',str(D/'build'),'--target','search_disk_index','-j','4'],'full_skip_build.log')
  search('full_skip',extra={'DISKANN_SKIP_PREFIX':str(E/'full_skip'),'DISKANN_SKIP_ENTIRE_NODE':'1','DISKANN_TRACE_PREFIX':str(E/'full_skip_trace')})
 rows=[]
 for s in summaries:
  L=s['L']; m=metrics('full_skip',L); assert m['identical_ids'] and m['identical_distances'],m
  count=sum(1 for _ in (E/f'full_skip_{L}.mask').open()); expanded=0; scanned=0; visited=0
  qseen=set()
  for q,l,v,ex,ed in blocks(E/f'full_skip_trace_{L}.trace'):
   assert q not in qseen; qseen.add(q); expanded+=len(ex); scanned+=len(ed); visited+=v
  assert len(qseen)==10000
  assert expanded==int(s['expanded']['total'])-count
  m.update(skipped_nodes=count,skipped_per_query=count/10000,skipped_fraction=count/s['expanded']['total'],expanded_after=expanded,neighbor_edges_after=scanned,visited_after=visited,neighbor_edges_saved=int(s['neighbor_edges']['total'])-scanned)
  s['full_skip']=m; rows.append(f"| {L} | {count/10000:.2f} | {100*m['skipped_fraction']:.2f}% | {expanded/10000:.2f} | {100*m['recall_at_10']:.3f}% | {m['identical_queries']}/10000 |")
 (E/'summary.json').write_text(json.dumps(summaries,indent=2))
 report=(E/'REPORT.md').read_text()
 report+='\n## Full node expansion oracle (actual I/O avoided)\n\nHere the mask contains baseline zero-admission nodes EXCLUDING every baseline final top10 ID. The intervention occurs immediately after popping/marking the candidate, before adjacency cache lookup or disk read, and reserves the original beam slot. Thus both adjacency work and exact node scoring are skipped; uncached masked nodes require no disk reads. Every query returns exactly the same ordered IDs and distances.\n\n| L | full expansions skipped/query | fraction | remaining expansions/query | Recall@10 | identical queries |\n|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\nThis is a validated oracle lower bound on removable complete node expansions for these runs. It is not a claimed maximum, does not provide an online predictor, and requires hindsight knowledge of admissions and final results. A node with zero useful neighbors may itself belong to the final top10, which is why naive full-node skipping of all useless nodes is invalid. Raw I/O and latency measurements are in `full_skip.log`; speedup claims require repeated warm-cache/uncached randomized timing trials.\n\nRe-run this extension with `../../.venv/bin/python run_full_skip.py` after the main experiment.\n'
 (E/'REPORT.md').write_text(report)
 print(json.dumps([{ 'L':s['L'], 'full_skip':s['full_skip']} for s in summaries],indent=2),flush=True)
