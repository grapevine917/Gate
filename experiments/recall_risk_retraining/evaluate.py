from common import *
assert (D/'TRAIN_COMPLETE.json').exists()
P=json.loads((D/'PROTOCOL.json').read_text());P['seed']=SEED
for p in [D/'logs',D/'runs/validation',D/'runs/test']:p.mkdir(parents=True,exist_ok=True)
source=(D.parent/'sift1m_mild_dynamic_recall_aligned/measurement.py').read_text()
source=source.replace("models=A/'models_train100'","models=A/'models_train100' if batch[0]['variant']=='old' else D/'models'/batch[0]['variant']")
source=source.replace("(c['training_L'],c['kind'])","(c['variant'],c['kind'])")
(D/'measurement.py').write_text(source);exec(compile(source,str(D/'measurement.py'),'exec'),globals())
configs=[]
def add(key,family,kind,tau=0,variant='old'):
    configs.append(dict(key=key,family=family,kind=kind,variant=variant,training_L=100 if kind>=2 else 0,inference_L=100,tau=float(tau),late_tau=0.,rank=72 if kind==1 else 100,exp=100 if kind==1 else -1,base_scale=1.,multiplier=0.,diagnostic=False))
add('DiskANN','DiskANN',0);add('Heuristic','Heuristic',1)
for variant in ['old','proxy','risk']:
    for family,kind in [('MLP',2),('LR',3)]:
        for i,tau in enumerate(P['threshold_grid']):add(f'{variant}_{family}_t{i:02d}',family,kind,tau,variant)
        if variant=='old':add(f'old_{family}_default',family,kind,TAUS[0 if family=='MLP' else 1],variant)
online=dict(dataset='SIFT1M',configs=configs,selection='minimum validation physical bytes, then higher validation hits then lower tau; same validation Recall floor for all6 learned models; no test selection',timing_repeats=16,amendment='16 instead of12 proposed in label protocol, so all8 cyclic order positions repeat exactly twice; frozen before online validation or test',sources={str(p):sha(p) for p in [D/'PROTOCOL.json',D/'TRAIN_COMPLETE.json',A/'native/search',A/'models_train100/state.bin',A/'models_train100/lr.bin',*[D/'models'/v/f'{m}.bin' for v in ['proxy','risk'] for m in ['state','lr']]]})
if (D/'ONLINE_PROTOCOL.json').exists():assert json.loads((D/'ONLINE_PROTOCOL.json').read_text())==online
else:dump(D/'ONLINE_PROTOCOL.json',online)
validation=load('validation',configs);dump(D/'online_validation.json',validation);save(D/'online_validation.csv',validation);V={r['key']:r for r in validation};C={c['key']:c for c in configs}
for r in validation:
    if r['kind']>=2 and r['tau']==0:assert r['ids_sha256']==V['DiskANN']['ids_sha256'] and r['reads']==V['DiskANN']['reads']
floor=min(V['old_MLP_default']['total_hits'],V['old_LR_default']['total_hits']);selected=[C['DiskANN'],C['Heuristic']]
for variant in ['old','proxy','risk']:
    for family in ['MLP','LR']:
        feasible=[r for r in validation if r['variant']==variant and r['family']==family and r['total_hits']>=floor]
        winner=min(feasible,key=lambda r:(r['physical_bytes'],-r['total_hits'],r['tau'],r['key']));selected.append(C[winner['key']])
selection=dict(validation_floor_hits=floor,validation_floor_recall=floor/100,configs=selected,default_keys=['old_MLP_default','old_LR_default'],validation_results=[V[c['key']] for c in selected])
if (D/'SELECTION_LOCK.json').exists():assert json.loads((D/'SELECTION_LOCK.json').read_text())==selection
else:dump(D/'SELECTION_LOCK.json',selection)
print('SELECTION LOCKED',[(c['key'],V[c['key']]['recall']) for c in selected],flush=True)
allruns={};primary=[]
for rep in range(16):
    order=selected[rep%8:]+selected[:rep%8];rs=load('test',order,rep,True);primary+=rs
    for r in rs:allruns[(r['key'],r['repeat'])]=r
# Keep literal original static defaults as contemporaneous references too.
defaults=[]
for rep in range(16):
    cs=[C[k] for k in selection['default_keys']];cs=cs[rep%2:]+cs[:rep%2];rs=load('test',cs,rep,True);defaults+=rs
    for r in rs:allruns[(r['key'],r['repeat'])]=r
dump(D/'primary_test_runs.json',primary);dump(D/'default_test_runs.json',defaults);dump(D/'all_test_runs.json',list(allruns.values()))
assert all(sha(p)==h for p,h in online['sources'].items());dump(D/'ONLINE_COMPLETE.json',dict(complete=True,primary_runs=len(primary),unique_test_runs=len(allruns),validation_configs=len(validation)));print('ONLINE COMPLETE',flush=True)
