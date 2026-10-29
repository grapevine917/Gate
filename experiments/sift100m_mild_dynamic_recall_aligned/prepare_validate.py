"""Freeze mild policies and alignment grid, then evaluate validation only."""
from common import *
T=json.loads((A/'thresholds_L100.json').read_text())['100']
protocol=dict(dataset='SIFT100M',base_vectors=100000000,dimensions=128,training_L=100,model_dir=str(A/'models_train100'),retrained=dict(MLP=False,LR=True),beam=4,K=10,threads=4,MKL_threads=1,cache_nodes=10000,seed=20261024,validation_queries=1000,test_queries=1000,static_tau={name:T[name][4]['tau'] for name in ['state','lr']},main='InferenceL100, base tau identical to corresponding static1%FNR threshold; choose active mild policy with validation Recall>=static default, lowest validationP95; lock before test;6methods12balancedcyclicrepeats',multipliers=[1.1,1.25,1.5,2.0],main_progress=[(r,e) for r in [72,85] for e in [100,105,110]],base_scales=[.5,.75,1.,1.1,1.25,1.5,2.],alignment='Same6families. Only DiskANN L may vary85..110; all learned/heuristic inferenceL100. Static threshold and dynamic base may be calibrated, but each static/dynamic pair MUST have identical base_tau. Match integer total_hits, never interpolate.',alignment_validation='Find exact common validation hit count nearest the validation static-reference Recall, lock configs before test; report actual test residual separately',alignment_descriptive_test='Evaluate frozen grid once; match actual test integer total_hits nearest the validation static-reference Recall with same-base static/dynamic pairs and active extra skips. Choose among matching configs using VALIDATION P95 ranking, not test timing. Fresh6timingrepeats12..17. This is descriptive quality matching, not an unbiased deployment result. If exact match unavailable, explicitly label minimal spread approximate.',controls='Fresh shared controls; resident mixed-method process with per-config128-query warmup; identical config/repeat measured once across studies',metrics=['Recall@10','P50','P95','P99','mean latency','QPS','physical bytes','read requests','expansions','skips','model calls','IO time','IO rounds','dynamic eligible','dynamic extra skips'],IO='Strict reader bytes ==Linux read_bytes; reject/archive and retry mismatch, no tolerance relaxation',limits=['Already inspected query split, exploratory','Finite grid and one saved training seed','Same recall does not imply same successful queries','Final timing repeats do not include training variability'],source_hashes={str(p):sha(p) for p in [A/'models_train100/state.bin',A/'models_train100/lr.bin',A/'native/search',A/'thresholds_L100.json',W/'data/validation.u8bin',W/'data/validation.gt',W/'data/test.u8bin',W/'data/test.gt']})
protocol.update(training_provenance=json.loads((D/'TRAIN_COMPLETE.json').read_text()),smoke_verification=json.loads((D/'SMOKE_COMPLETE.json').read_text())['passed'],native_index_dtype='uint8',reference_protocol='SIFT1M mild strategy, same168 configurations and repeats',native_execution='Resident index and both17-feature models; per-config128-query warmup; no query state sharing')
protocol['source_hashes'].update({str(p):sha(p) for p in [D/'TRAIN_COMPLETE.json',D/'SMOKE_COMPLETE.json',W/'data/l100.prune.bin',D/'search.cpp',D/'bootstrap.py']})
if (D/'PROTOCOL.json').exists():assert json.loads((D/'PROTOCOL.json').read_text())==protocol
else:dump(D/'PROTOCOL.json',protocol)
P.update(protocol)
configs=[]
def tok(v):return f'{v:g}'.replace('.','p')
def add(key,family,kind,L=100,tau=0.,late=0.,rank=100,exp=-1,base_scale=0.,multiplier=0.,diagnostic=False):
    configs.append(dict(key=key,family=family,kind=kind,training_L=100 if kind>=2 else 0,inference_L=L,tau=tau,late_tau=late,rank=rank,exp=exp,base_scale=float(base_scale),multiplier=float(multiplier),diagnostic=diagnostic))
for L in range(85,111):add(f'DiskANN_L{L}','DiskANN',0,L)
for rank in [64,72,85,92,98,100]:
    for exp in [90,95,100,105,110]:add(f'Heuristic_r{rank}_e{exp}','Heuristic',1,rank=rank,exp=exp)
for name,family,sk,dk in [('state','MLP',2,4),('lr','LR',3,5)]:
    base=protocol['static_tau'][name]
    for scale in protocol['base_scales']:
        tau=base*scale;add(f'{family}_static_b{tok(scale)}',family,sk,tau=tau,base_scale=scale)
        schedules=protocol['main_progress'] if scale==1 else [(72,100)]
        for rank,exp in schedules:
            for mult in protocol['multipliers']:add(f'Dynamic{family}_b{tok(scale)}_m{tok(mult)}_r{rank}_e{exp}',family,dk,tau=tau,late=tau*mult,rank=rank,exp=exp,base_scale=scale,multiplier=mult)
    add(f'Dynamic{family}_identity',family,dk,tau=base,late=base,rank=72,exp=100,base_scale=1,multiplier=1,diagnostic=True)
assert len(configs)==168 and len({c['key'] for c in configs})==168
dump(D/'configs.json',configs)
lock=dict(protocol=sha(D/'PROTOCOL.json'),configs=sha(D/'configs.json'),sources=protocol['source_hashes'])
if (D/'RUN_LOCK.json').exists():assert json.loads((D/'RUN_LOCK.json').read_text())==lock
else:dump(D/'RUN_LOCK.json',lock)
validation=load('validation',configs);dump(D/'validation.json',validation);V={r['key']:r for r in validation};C={c['key']:c for c in configs}
main=[C['DiskANN_L100'],C['Heuristic_r72_e100'],C['MLP_static_b1'],C['LR_static_b1']];details=[]
for family,kind in [('MLP',4),('LR',5)]:
    static=V[f'{family}_static_b1'];identity=V[f'Dynamic{family}_identity']
    for k in ['total_hits','reads','expansions','skips','ids_sha256']:assert identity[k]==static[k],('identity regression',family,k)
    cand=[r for r in validation if r['kind']==kind and r['base_scale']==1 and r['multiplier']>1 and r['extra_skips']>0]
    assert cand,('no active mild policy',family)
    feasible=[r for r in cand if r['total_hits']>=static['total_hits']]
    winner=min(feasible,key=lambda r:(r['p95_ms'],-r['total_hits'],r['key'])) if feasible else min(cand,key=lambda r:(-r['total_hits'],r['p95_ms'],r['key']))
    main.append(C[winner['key']]);details.append(dict(family=family,key=winner['key'],static_validation_recall=static['recall'],mild_validation_recall=winner['recall'],quality_matched=bool(feasible),base_tau_equals_static=winner['tau']==static['tau']))
for key,old in [('DiskANN_L100','DiskANN_L100_p0'),('Heuristic_r72_e100','Heuristic_L100_p3')]:assert V[key]['ids_sha256']==sha(W/'runs/validation'/f'{old}_r0.ids')
main_lock=dict(configs=main,details=details)
if (D/'MAIN_SELECTION_LOCK.json').exists():assert json.loads((D/'MAIN_SELECTION_LOCK.json').read_text())==main_lock
else:dump(D/'MAIN_SELECTION_LOCK.json',main_lock)
nominal=int(round((V['MLP_static_b1']['total_hits']+V['LR_static_b1']['total_hits'])/2));dump(D/'ALIGNMENT_TARGET_LOCK.json',dict(nominal_hits=nominal,source='mean static MLP/LR validation hits, before test'));aligned=choose_alignment(validation,V,nominal)
if (D/'ALIGNMENT_VALIDATION_LOCK.json').exists():assert json.loads((D/'ALIGNMENT_VALIDATION_LOCK.json').read_text())==aligned
else:dump(D/'ALIGNMENT_VALIDATION_LOCK.json',aligned)
dump(D/'VALIDATION_COMPLETE.json',dict(complete=True,configs=len(configs),identity_checks_passed=True,main=main_lock,alignment=aligned))
print('VALIDATION COMPLETE',main_lock,aligned,flush=True)
