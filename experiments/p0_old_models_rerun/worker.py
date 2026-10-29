from measure import *
import sys
E=Experiment(sys.argv[1]);mode=sys.argv[2];O=E.O;P=E.P
E.audit_sources()
if mode=='smoke':
 # All disablements and joint degeneracies must preserve IDs and discrete work.
 cs=[dict(key='SmokeNative',family='Smoke',kind=0,model='none',L=100,tau=0.,et_tau=1.1,rank=100,exp=-1,preset=0)]
 for model in P['models']:
  for kind in [2,4]:cs.append(cs[0]|dict(key=f'SmokeOff_{model}_{kind}',kind=kind,model=model))
 cs.append(cs[0]|dict(key='SmokeETOff',kind=3))
 for model in P['models']:
  tau=json.loads((O/'thresholds.json').read_text())[model][4]['tau']
  for kind in [2,4]:cs.append(cs[0]|dict(key=f'SmokeStatic_{model}_{kind}',kind=kind,model=model,tau=tau))
 rs=E.load('validation',cs,99,True);V={r['key']:r for r in rs};base=V['SmokeNative']
 for r in rs:
  if r['tau']==0:
   for k in ['ids_sha256','total_hits','reads','expansions']:assert r[k]==base[k],(r['key'],k)
 for model in P['models']:
  for k in ['ids_sha256','total_hits','reads','expansions']:assert V[f'SmokeStatic_{model}_2'][k]==V[f'SmokeStatic_{model}_4'][k]
 if E.ds=='sift1m':reference=E.R/'runs/validation/DiskANN_L100_r0.ids'
 else:reference=E.R/'runs/validation/DiskANN_L100_p0_r0.ids'
 assert base['ids_sha256']==sha(reference)
 dump(O/'SMOKE_COMPLETE.json',dict(passed=True,configs=len(cs),no_prune_equals_native=True,joint_ET_off_equals_gate=True,historical_native_IDs_equal=True,rows=rs))
elif mode=='p0a':
 cs=[E.bykey[k] for k in P['P0_A']['keys']];validation=E.load('validation',cs);runs=[]
 for rep in range(15):runs+=E.load('test',cs[rep%5:]+cs[:rep%5],rep,True)
 summary=aggregate(runs,cs);dump(O/'p0a_runs.json',runs);dump(O/'p0a_summary.json',summary);save(O/'p0a_summary.csv',summary);dump(O/'P0A_COMPLETE.json',dict(complete=True,runs=len(runs),methods=5,repeats=15,validation=validation));print('P0A COMPLETE',E.ds,flush=True)
elif mode=='p0b':
 validation=E.load('validation',E.configs);dump(O/'validation.json',validation);save(O/'validation.csv',validation);families=P['P0_B']['families'];objectives=['physical_bytes','p95_ms','qps'];selection=[]
 for family in families:
  for target in P['P0_B']['targets']:
   feasible=[r for r in validation if r['family']==family and r['total_hits']>=int(round(target*100))]
   for metric in objectives:
    if not feasible:selection.append(dict(family=family,target=target,objective=metric,key=None,validation_recall=None));continue
    best=min(feasible,key=lambda r:(-r[metric] if metric=='qps' else r[metric],-r['total_hits'],r['key']));selection.append(dict(family=family,target=target,objective=metric,key=best['key'],validation_recall=best['recall']))
 if (O/'SELECTION_LOCK.json').exists():assert json.loads((O/'SELECTION_LOCK.json').read_text())==selection
 else:dump(O/'SELECTION_LOCK.json',selection)
 allruns={(r['key'],r['repeat']):r for r in json.loads((O/'p0a_runs.json').read_text())};initial=E.load('test',E.configs)
 for r in initial:allruns[(r['key'],r['repeat'])]=r
 dump(O/'test_initial.json',initial)
 iteration=0
 while True:
  iteration+=1;summary=aggregate(list(allruns.values()),E.configs);S={r['key']:r for r in summary};needed={r['key'] for r in selection if r['key']}
  for family in families:
   for metric in objectives:needed.update(r['key'] for r in frontier([r for r in summary if r['family']==family],metric))
  todo=[E.bykey[k] for k in sorted(needed) if S[k]['repeats']<3];dump(O/'repeat_status.json',dict(iteration=iteration,configs_need_repeats=len(todo),total_unique_runs=len(allruns),frontier_and_locked_keys=sorted(needed)))
  dump(O/'all_test_runs_checkpoint.json',list(allruns.values()))
  if not todo:break
  for rep in [1,2]:
   for r in E.load('test',todo,rep):allruns[(r['key'],r['repeat'])]=r
 E.audit_sources();dump(O/'all_test_runs.json',list(allruns.values()));dump(O/'summary.json',summary);save(O/'all_configurations.csv',summary)
 locked=[]
 for row in selection:
  r=S.get(row['key']);locked.append(row|dict(test_available=r is not None,test_meets_target=bool(r and r['recall']>=row['target']),test_recall=r['recall'] if r else None,p95_ms=r['p95_ms'] if r else None,qps=r['qps'] if r else None,physical_bytes=r['physical_bytes'] if r else None,L=r['L'] if r else None,tau=r['tau'] if r else None,et_tau=r['et_tau'] if r else None))
 save(O/'validation_locked_test.csv',locked);dump(O/'P0B_COMPLETE.json',dict(complete=True,grid_configs=len(E.configs),families=families,unique_runs=len(allruns),iterations=iteration));print('P0B COMPLETE',E.ds,flush=True)
else:raise ValueError(mode)
