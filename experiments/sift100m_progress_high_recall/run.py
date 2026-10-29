"""Predeclare a finite validation grid, lock policies, then evaluate test once."""
from pathlib import Path
import csv,hashlib,json,os,random,subprocess,time
import numpy as np
D=Path(__file__).resolve().parent;R=D.parent/'sift100m_p0'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,rows):
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
taus=[x['tau'] for x in json.loads((R/'threshold_calibration.json').read_text())['lr']]
original={c['key']:c for c in json.loads((R/'configs.json').read_text())}
configs=[]
def add(key,group,kind=2,tau=0.,late=1.1,rank=-1,exp=-1,original_key=None):
    configs.append(dict(key=key,group=group,kind=kind,L=500,tau=tau,et_tau=late,rank=rank,exp=exp,original_key=original_key))
for j,tau in enumerate(taus):add(f'Static_p{j}','StaticLR',tau=tau,original_key=f'LR_L500_p{j}')
for j in [3,4,5,6]:
    c=original[f'Heuristic_L500_p{j}'];add(f'Heuristic_p{j}','Heuristic',kind=1,rank=c['rank'],exp=c['exp'],original_key=c['key'])
    if j in [5,6]:add(f'Endpoint_p{j}','Endpoint',tau=0.,late=1.01,rank=c['rank'],exp=c['exp'],original_key=c['key'])
add('DiskANN_L500','DiskANN',kind=0,original_key='DiskANN_L500_p0')
for b in [0,1,2]:
    for s,(rank,exp) in enumerate([(125,375),(250,450),(320,460),(360,500)]):
        for t,late in enumerate([taus[4],.35,.95]):add(f'Progress_b{b}_s{s}_t{t}','ProgressLR',tau=taus[b],late=late,rank=rank,exp=exp)
protocol=dict(objective='validation Recall-constrained minimum P95; L500 fixed, LR weights fixed',targets=[99.7,99.75],configs=configs,validation='one initial run of all 51 configs; top 2 feasible ProgressLR and StaticLR per target plus four Heuristic controls and DiskANN remeasured to 3 repeats; select by median P95 within these finalists',test='freeze selections before running test; 3 repeats of unique selected policies and L500 DiskANN; do not retune after test',threshold_rule='base tau until candidate_rank > rank AND completed_expansions > exp, then max(base, late_tau); cache hits always protected',endpoints='not candidates; base tau=0 and late tau=1.01 must reproduce Heuristic IDs/counts exactly',dataset=dict(base_vectors=100000000,dimensions=128,dtype='uint8',validation_queries=1000,test_queries=1000,K=10,L=500,threads=4,cache_nodes=10000),binary_provenance=str(D.parent/'sift100m_progress_gate/native/search'),limitations='follow-up exploration on an already inspected held-out test split, not a new blind external test; finite grid, one model seed, WSL2',source_hashes={str(p):sha(p) for p in [R/'models/lr.bin',R/'native/search',R/'configs.json',R/'threshold_calibration.json',R/'data/validation.u8bin',R/'data/validation.gt',R/'data/test.u8bin',R/'data/test.gt',D/'native/search']})
if (D/'PROTOCOL.json').exists():assert json.loads((D/'PROTOCOL.json').read_text())==protocol
else:dump(D/'PROTOCOL.json',protocol)
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
check_records=[]
def prefix(split,c,rep):return D/'runs'/split/f'{c["key"]}_r{rep}'
def complete(split,c,rep):
    p=prefix(split,c,rep)
    try:return json.loads(p.with_suffix('.meta.json').read_text())['queries']==1000 and p.with_suffix('.ids').stat().st_size==80000 and len(p.with_suffix('.csv').read_text().splitlines())==1001
    except (OSError,ValueError,KeyError):return False
def load(split,cs,rep):
    folder=D/'runs'/split;folder.mkdir(parents=True,exist_ok=True)
    todo=[c for c in cs if not complete(split,c,rep)]
    rng=random.Random(20261024+rep+(100 if split=='test' else 0))
    batches=[]
    for kind in sorted({c['kind'] for c in todo}):
        group=[c for c in todo if c['kind']==kind];rng.shuffle(group)
        # Keep batches small enough to expose progress and resume interrupted work.
        batches.extend(group[i:i+6] for i in range(0,len(group),6))
    rng.shuffle(batches)
    for batch in batches:
        digest=hashlib.sha256(' '.join(c['key'] for c in batch).encode()).hexdigest()[:10]
        tag=f'{split}_r{rep}_{digest}';cp=D/f'{tag}.configs'
        cp.write_text(''.join(f'{c["key"]} {c["kind"]} 500 {c["tau"]:.17g} {c["et_tau"]:.17g} {c["rank"]} {c["exp"]}\n' for c in batch))
        cmd=[str(D/'native/search'),str(R/'index/ann'),str(R/'data'/f'{split}.u8bin'),str(R/'data'/f'{split}.gt'),str(R/'models'),str(R/'data/vectors.cvec'),str(cp),str(folder),str(rep)]
        dump(D/'progress.json',dict(split=split,repeat=rep,keys=[c['key'] for c in batch],argv=cmd,started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        print(time.strftime('%H:%M:%S'),split,rep,[c['key'] for c in batch],flush=True)
        with (D/f'{tag}.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    rows=[]
    with (R/'data'/f'{split}.gt').open('rb') as f:
        n,k=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4',count=int(n*k)).reshape(n,k)[:,:10]
    for c in cs:
        p=prefix(split,c,rep);a=np.genfromtxt(p.with_suffix('.csv'),delimiter=',',names=True);m=json.loads(p.with_suffix('.meta.json').read_text());ids=np.fromfile(p.with_suffix('.ids'),'<u8').reshape(1000,10)
        assert len(a)==1000
        valid=ids<100000000;missing=ids==np.iinfo(np.uint32).max
        assert np.all(valid|missing) and np.all(np.diff(missing.astype(np.int8),axis=1)>=0)
        assert all(len(set(row[mask]))==int(mask.sum()) for row,mask in zip(ids,valid))
        hits=np.array([len(set(x)&set(y)) for x,y in zip(ids,gt)])
        np.testing.assert_array_equal(hits,a['hits'])
        assert m['reader_bytes']==m['linux_read_bytes']==int(a['reads'].sum())*4096
        assert m['reader_requests']==int(a['reads'].sum())
        assert m['swap_before_kib']==m['swap_after_kib']==0
        if c['kind'] in [1,2]:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
        if c['original_key']:
            op=R/'runs'/split/f'{c["original_key"]}_r0'
            assert p.with_suffix('.ids').read_bytes()==op.with_suffix('.ids').read_bytes(),(split,c['key'],'original IDs')
            old=np.genfromtxt(op.with_suffix('.csv'),delimiter=',',names=True)
            for k in ['hits','reads','expansions','skips']:np.testing.assert_array_equal(a[k],old[k],err_msg=c['key'])
        rows.append(c|dict(split=split,repeat=rep,recall=float(hits.sum()/100),total_hits=int(hits.sum()),p95_ms=float(np.percentile(a['latency_us'],95)/1000),mean_ms=float(a['latency_us'].mean()/1000),qps=float(m['qps']),reads=float(a['reads'].mean()),io_rounds=float(a['io_rounds'].mean()),io_ms=float(a['io_us'].mean()/1000),skips=float(a['skips'].mean()),progress_eligible=float(a['progress_eligible'].mean()),progress_extra_skips=float(a['progress_extra_skips'].mean()),ids_sha256=sha(p.with_suffix('.ids'))))
        check_records.append(dict(split=split,key=c['key'],repeat=rep,io_verified=True,no_swap=True,original_parity_checked=bool(c['original_key'])))
    return rows
def aggregate(runs,cs):
    out=[]
    for c in cs:
        rs=[r for r in runs if r['key']==c['key']]
        r=rs[0].copy();r.pop('repeat');r['repeats']=len(rs)
        for k in ['total_hits','reads','io_rounds','skips','progress_eligible','progress_extra_skips','ids_sha256']:assert len({x[k] for x in rs})==1,(c['key'],k)
        for k in ['p95_ms','mean_ms','qps','io_ms']:r[k]=float(np.median([x[k] for x in rs]))
        r['p95_min_ms']=min(x['p95_ms'] for x in rs);r['p95_max_ms']=max(x['p95_ms'] for x in rs)
        out.append(r)
    return out
initial=load('validation',configs,0);dump(D/'validation_initial.json',initial)
finalist_keys=set(c['key'] for c in configs if c['group'] in ['Heuristic','DiskANN'])
for target in protocol['targets']:
    for group in ['ProgressLR','StaticLR']:
        feasible=sorted((r for r in initial if r['group']==group and r['recall']>=target),key=lambda r:(r['p95_ms'],-r['total_hits'],r['key']))
        finalist_keys.update(r['key'] for r in feasible[:2])
finalists=[c for c in configs if c['key'] in finalist_keys]
if (D/'VALIDATION_FINALISTS.json').exists():assert json.loads((D/'VALIDATION_FINALISTS.json').read_text())==finalists
else:dump(D/'VALIDATION_FINALISTS.json',finalists)
vruns=[r for r in initial if r['key'] in finalist_keys]
for rep in [1,2]:vruns+=load('validation',finalists,rep)
validation=aggregate(vruns,finalists);save(D/'validation_finalists.csv',validation);dump(D/'validation_runs.json',vruns)
selection=[]
for target in protocol['targets']:
    for group in ['ProgressLR','StaticLR','Heuristic','DiskANN']:
        feasible=[r for r in validation if r['group']==group and r['recall']>=target]
        winner=min(feasible,key=lambda r:(r['p95_ms'],-r['total_hits'],r['key'])) if feasible else None
        selection.append(dict(target=target,group=group,key=winner['key'] if winner else None,validation_recall=winner['recall'] if winner else None,validation_p95_ms=winner['p95_ms'] if winner else None))
if (D/'SELECTION_LOCK.json').exists():assert json.loads((D/'SELECTION_LOCK.json').read_text())==selection
else:dump(D/'SELECTION_LOCK.json',selection)
print('SELECTION LOCKED',selection,flush=True)
chosen={s['key'] for s in selection if s['key']};test_configs=[c for c in configs if c['key'] in chosen];truns=[]
for rep in range(3):truns+=load('test',test_configs,rep)
test=aggregate(truns,test_configs);save(D/'test_summary.csv',test);dump(D/'test_runs.json',truns)
assert all(sha(p)==h for p,h in protocol['source_hashes'].items())
dump(D/'verification.json',dict(passed=True,checks=check_records,inputs_unchanged=True,validation_selection_locked_before_test=True))
dump(D/'MEASUREMENTS_COMPLETE.json',dict(complete=True,validation_grid=len(configs),validation_finalists=len(finalists),test_configs=len(test_configs),test_runs=len(truns),validation_runs=len(initial)+len(vruns)-len(finalists)))
print('MEASUREMENTS COMPLETE',flush=True)
