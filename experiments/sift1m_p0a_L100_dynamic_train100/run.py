"""Add dynamic policies for the saved pure-L100 MLP/LR; do not retrain."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,hashlib,json,random,subprocess,time,shutil
import numpy as np
D=Path(__file__).resolve().parent;A=D.parent/'sift1m_p0a_L100_train_compare';W=D.parent/'sift1m_wide_p0';ROOT=D.parents[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
def save(p,rows):
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
old_protocol=json.loads((A/'PROTOCOL.json').read_text());old_validation=json.loads((A/'validation.json').read_text());T=json.loads((A/'thresholds_L100.json').read_text())['100']
P=dict(dataset='SIFT1M',base_vectors=1000000,training_L=100,inference_L=100,beam=4,K=10,threads=4,cache_nodes=10000,validation_queries=1000,test_queries=1000,seed=old_protocol['seed'],retrained=False,model_dir=str(A/'models_train100'),dynamic_presets=old_protocol['dynamic_presets'],selection='8 presets per model; validation minimumP95 subject to Recall>=saved trainL100 static default; lock before test',test='16 repeats each for new dynamicMLP/dynamicLR; balanced alternating order; prior eight groups and shared controls reused, not rerun',limitations=['Existing inspected test split, exploratory','Reused control timings are from the preceding experiment, not contemporaneously interleaved with the two new groups; small timing changes may reflect temporal drift','Proxy FNR is not a final Recall guarantee'],source_hashes={str(p):sha(p) for p in [A/'models_train100/state.bin',A/'models_train100/lr.bin',A/'native/search',A/'thresholds_L100.json',A/'summary.json',A/'validation.json',W/'data/validation.fbin',W/'data/validation.gt',W/'data/test.fbin',W/'data/test.gt']})
if (D/'PROTOCOL.json').exists():assert json.loads((D/'PROTOCOL.json').read_text())==P
else:dump(D/'PROTOCOL.json',P)
def make(name,family,kind,j):
    base,late,r,e=P['dynamic_presets'][j]
    return dict(key=f'{family}_train100_dynamic_p{j}',family=family,kind=kind,training_L=100,inference_L=100,tau=T[name][base]['tau'],late_tau=T[name][late]['tau'],rank=int(100*r),exp=-1 if e<0 else int(100*e),preset=j)
configs=[make(name,family,kind,j) for name,family,kind in [('state','MLP',4),('lr','LR',5)] for j in range(8)]
dump(D/'configs.json',configs)
for p in [D/'logs',D/'runs/validation',D/'runs/test']:p.mkdir(parents=True,exist_ok=True)
# Reuse only helper definitions, not the previous experiment's top-level execution.
exec(compile((D/'measurement.py').read_text(),str(D/'measurement.py'),'exec'),globals())
validation=load('validation',configs);dump(D/'validation.json',validation)
baseline=json.loads((A/'runs/validation/DiskANN_r0.meta.json').read_text())
for r in validation:
    if r['preset']==0:
        for k in ['reads','expansions','ids_sha256','total_hits']:
            expected=next(x[k] for x in old_validation if x['key']=='DiskANN');assert r[k]==expected,(r['key'],k)
selection=[];details=[]
for family,kind in [('MLP',4),('LR',5)]:
    target=next(r['recall'] for r in old_validation if r['family']==family and r['training_L']==100 and r['kind'] in [2,3])
    candidates=[r for r in validation if r['kind']==kind];feasible=[r for r in candidates if r['recall']>=target]
    winner=min(feasible,key=lambda r:(r['p95_ms'],-r['total_hits'],r['key'])) if feasible else min(candidates,key=lambda r:(-r['total_hits'],r['p95_ms'],r['key']))
    selection.append(next(c for c in configs if c['key']==winner['key']))
    details.append(dict(family=family,key=winner['key'],static_validation_recall=target,dynamic_validation_recall=winner['recall'],quality_matched=bool(feasible)))
if (D/'SELECTION_LOCK.json').exists():assert json.loads((D/'SELECTION_LOCK.json').read_text())==selection
else:dump(D/'SELECTION_LOCK.json',selection)
dump(D/'DYNAMIC_SELECTION.json',details);print('SELECTION LOCKED',details,flush=True)
runs=[]
for rep in range(16):runs+=load('test',selection if rep%2==0 else selection[::-1],rep,True)
summary=[]
for c in selection:
    rs=[r for r in runs if r['key']==c['key']];assert len(rs)==16;r=rs[0].copy();r.pop('repeat');r['repeats']=16
    for k in ['total_hits','physical_bytes','reads','expansions','skips','extra_skips','eligible','ids_sha256']:assert len({x[k] for x in rs})==1,(c['key'],k)
    for k in ['p95_ms','qps','mean_ms','io_ms']:r[k]=float(np.median([x[k] for x in rs]))
    r['p95_min_ms']=min(x['p95_ms'] for x in rs);r['p95_max_ms']=max(x['p95_ms'] for x in rs);summary.append(r)
dump(D/'summary.json',summary);save(D/'summary.csv',summary);dump(D/'all_runs.json',runs)
assert all(sha(p)==h for p,h in P['source_hashes'].items())
old=json.loads((A/'summary.json').read_text());combined=old+summary;combined.sort(key=lambda r:(0 if r['kind']==0 else 1 if r['kind']==1 else 2,r['training_L'],r['family'],r['kind']))
assert len(combined)==10 and len([r for r in combined if r['kind'] in [0,1]])==2
dump(D/'combined_summary.json',combined);save(D/'combined_summary.csv',combined)
dump(D/'verification.json',dict(passed=True,new_test_runs=len(runs),all_inference_L100=True,weights_and_source_inputs_unchanged=True,retrained=False,controls_reused=True,recall_recomputed=True,IO_verified=True,no_swap=True,repeats_ID_and_counts_equal=True))
dump(D/'MEASUREMENTS_COMPLETE.json',dict(complete=True,new_groups=2,validation_configs=16,new_test_runs=32,repeats=16,combined_groups=10))
print('MEASUREMENTS COMPLETE',flush=True)
