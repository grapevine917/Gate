"""Secondary historical-graph selection: hold current64 Recall, report stricter no-navigation constraint separately."""
from pathlib import Path
import json,random,numpy as np
ROOT=Path(__file__).resolve().parent;ns={'__file__':str(ROOT/'run.py')};exec((ROOT/'run.py').read_text().split('summary={};top=')[0],ns);main=json.loads((ROOT/'summary.json').read_text());out={}
for ds in ['sift','deep']:
 d=main[ds];h=d['history_method'];refkey=f'{h}/64/64';target=d['validation'][refkey]['recall_at_10'];eligible=[k for k,r in d['validation'].items() if r['method']==h and r['recall_at_10']>=target-1e-12]
 # If both quality constraints coincide, the primary frozen choice already evaluates this question.
 no_nav=d['validation'][f'{h}/0/64']['recall_at_10']
 if target>=no_nav-1e-12:
  key=d['selected'][h]['key'];runs=d['raw_test_runs'][key];refs=d['raw_test_runs'][refkey];reuse=True;validation={k:d['validation'][k] for k in eligible};shortlist=d['shortlisted'][h]['shortlist']
 else:
  fast=sorted(eligible,key=lambda k:d['validation'][k]['mean_latency_us'])[:2];io=min(eligible,key=lambda k:(d['validation'][k]['submitted_requests'],d['validation'][k]['mean_navigation_cpu_us']));shortlist=list(dict.fromkeys(fast+[refkey,io]));vr={k:list(d['raw_validation'][k]) for k in shortlist}
  for rep in [1,2]:
   order=shortlist.copy();random.Random(20261030+rep).shuffle(order)
   for k in order:
    if len(vr[k])>=3:continue
    m,b,q=k.split('/');print(ds,'conditional validation',k,len(vr[k]),flush=True);vr[k].append(ns['execute'](ds,m,int(b),int(q),'validation',f'current_quality_{m}_b{b}_q{q}_r{len(vr[k])}'))
  validation={k:ns['aggregate'](rr) for k,rr in vr.items()};key=min(shortlist,key=lambda k:validation[k]['mean_latency_us']);(ROOT/ds/'frozen_current_recall.json').write_text(json.dumps({'selected_key':key,'quality_target':target,'quality_reference':refkey,'selection_uses_validation_only':True,'validation':validation},indent=2)+'\n');print(ds,'conditional frozen',key,flush=True)
  if key==refkey:runs=d['raw_test_runs'][refkey];refs=runs;reuse=True
  else:
   rr={key:[],refkey:[]}
   for rep in range(5):
    order=list(rr);random.Random(20261040+rep).shuffle(order)
    for k in order:
     m,b,q=k.split('/');print(ds,'conditional test',rep,k,flush=True);rr[k].append(ns['execute'](ds,m,int(b),int(q),'test',f'current_quality_{m}_b{b}_q{q}_r{rep}'))
   runs=rr[key];refs=rr[refkey];reuse=False
 a=ns['aggregate'](refs);b=ns['aggregate'](runs);changes=[c['qps']/r['qps']-1 for r,c in zip(refs,runs)];out[ds]={'selected_key':key,'quality_target_validation':target,'quality_reference':'current64','shortlist':shortlist,'validation':validation,'reference_test':a,'selected_test':b,'raw_reference_test':refs,'raw_selected_test':runs,'paired_vs64':ns['paired'](refs[0],runs[0]),'qps_change_ratio':b['qps']/a['qps']-1,'qps_change_runs':changes,'qps_change_range':[min(changes),max(changes)],'quality_feasible_vs_current_on_test':b['recall_at_10']>=a['recall_at_10']-1e-12,'quality_feasible_vs_no_navigation_on_test':b['recall_at_10']>=next(r for r in d['test'].values() if r['budget']==0)['recall_at_10']-1e-12,'reused_primary_runs':reuse};(ROOT/'current_recall_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(ds,'CONDITIONAL',key,b,out[ds]['paired_vs64'],flush=True)
