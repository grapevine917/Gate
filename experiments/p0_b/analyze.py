from pathlib import Path
import json,csv
R=Path(__file__).resolve().parent;A=json.loads((R/'aggregate.json').read_text());P=json.loads((R/'PROTOCOL.json').read_text());F=['Tuned_DiskANN','Heuristic','Ours','ET','ET_Ours'];M=['p95_us','qps','physical_bytes'];out=[]
for target in P['targets']:
 for metric in M:
  best={}
  for fam in F:
   feasible=[r for r in A if r['family']==fam and r['recall']>=target]
   if feasible:best[fam]=min(feasible,key=lambda r:-r[metric] if metric=='qps' else r[metric])
  b=best.get('Tuned_DiskANN')
  if b:
   for fam,r in best.items():out.append(dict(target=target,objective=metric,family=fam,actual_recall=r['recall'],baseline_actual_recall=b['recall'],cost=r[metric],baseline_cost=b[metric],improvement_pct=(r[metric]/b[metric]-1)*100 if metric=='qps' else (1-r[metric]/b[metric])*100,key=r['key'],L=r['L'],tau=r['tau'],et_tau=r['et_tau'],skips=r['skips'],et_stop_fraction=r['et_stop_fraction'],timing_repeats=r['timing_repeats']))
(R/'quality_requirement_summary.json').write_text(json.dumps(out,indent=2)+'\n')
with (R/'quality_requirement_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('Target objective family actualRecall cost improvement vs tuned baseline:')
for r in out:
 if r['target'] in [95,99,99.5,99.7]:print(r['target'],r['objective'],r['family'],r['actual_recall'],round(r['cost'],3),round(r['improvement_pct'],2),r['key'])
