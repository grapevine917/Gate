"""Unified P0-A/P0-B pool: one measurement per identical configuration/repeat."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,hashlib,json,math,random,subprocess,time
import numpy as np
D=Path(__file__).resolve().parent;ROOT=D.parents[1];P=json.loads((D/'PROTOCOL.json').read_text());LS=P['L_training_and_learned_grid']
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def save(p,rows):
    if not rows:return
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert (D/'NATIVE_VERIFIED.json').exists()
assert all(sha(p)==h for p,h in json.loads((D/'INPUT_LOCK.json').read_text()).items())
if not (D/'thresholds.json').exists():subprocess.run([str(ROOT/'.venv/bin/python'),str(D/'calibrate.py')],check=True)
T=json.loads((D/'thresholds.json').read_text());families=['DiskANN','Heuristic','MLP','LR','DynamicMLP','DynamicLR','ET','ET_MLP','ET_LR']
heur=[(1,-1),(.98,1),(.92,1.04),(.72,1),(.64,.92),(.5,.9),(.25,.75),(.1,.5)];ets=[1.1,.999,.995,.99,.98,.95,.9,.8];joint=[(0,0),(4,0),(0,3),(2,1),(3,2),(4,3),(5,4),(6,5)]
def make(kind,L,j=0):
    c=dict(key=f'{families[kind]}_L{L}_p{j}',family=families[kind],kind=kind,L=L,preset=j,tau=0.,et_tau=1.1,rank=L,exp=-1,late_tau=0.)
    model='state' if kind in [2,4,7] else 'lr'
    if kind==1:c.update(rank=int(L*heur[j][0]),exp=-1 if heur[j][1]<0 else int(L*heur[j][1]))
    elif kind in [2,3]:c['tau']=T[model][j]['tau']
    elif kind in [4,5]:
        base,late,r,e=P['dynamic_presets'][j];c.update(tau=T[model][base]['tau'],late_tau=T[model][late]['tau'],rank=int(L*r),exp=-1 if e<0 else int(L*e))
    elif kind==6:c['et_tau']=ets[j]
    elif kind in [7,8]:a,b=joint[j];c.update(tau=T[model][a]['tau'],et_tau=ets[b])
    return c
base=set(LS)
while len(base)<64:base.add(max((v for v in range(10,max(LS)+1) if v not in base),key=lambda v:min(abs(math.log(v/b)) for b in base)))
configs=[make(0,L) for L in sorted(base)]+[make(k,L,j) for k in range(1,9) for L in LS for j in range(8)];assert len(configs)==576 and len({c['key'] for c in configs})==576
dump(D/'configs.json',configs)
locks=dict(models={n:sha(D/'models'/f'{n}.bin') for n in ['state','lr','et']},runner=sha(D/'native/search'),grid=sha(D/'configs.json'),thresholds=sha(D/'thresholds.json'))
if (D/'RUN_LOCK.json').exists():assert json.loads((D/'RUN_LOCK.json').read_text())==locks
else:dump(D/'RUN_LOCK.json',locks)
for p in [D/'logs',D/'runs/validation',D/'runs/test']:p.mkdir(parents=True,exist_ok=True)
gt={}
for split in ['validation','test']:
    with (D/'data'/f'{split}.gt').open('rb') as f:n,k=np.fromfile(f,'<u4',2);gt[split]=np.fromfile(f,'<u4',count=int(n*k)).reshape(n,k)[:,:10]
checked={}
def prefix(split,c,rep):return D/'runs'/split/f'{c["key"]}_r{rep}'
def complete(split,c,rep):
    p=prefix(split,c,rep)
    try:return json.loads(p.with_suffix('.meta.json').read_text())['queries']==1000 and p.with_suffix('.ids').stat().st_size==80000 and len(p.with_suffix('.csv').read_text().splitlines())==1001
    except (OSError,ValueError,KeyError):return False
def read(split,c,rep):
    cache=(split,c['key'],rep)
    if cache in checked:return checked[cache]
    p=prefix(split,c,rep);a=np.genfromtxt(p.with_suffix('.csv'),delimiter=',',names=True);m=json.loads(p.with_suffix('.meta.json').read_text());ids=np.fromfile(p.with_suffix('.ids'),'<u8').reshape(1000,10)
    valid=ids<1000000;missing=ids==np.iinfo(np.uint32).max
    assert np.all(valid|missing) and np.all(np.diff(missing.astype(np.int8),axis=1)>=0)
    assert all(len(set(row[mask]))==int(mask.sum()) for row,mask in zip(ids,valid))
    hits=np.array([len(set(row)&set(g)) for row,g in zip(ids,gt[split])]);np.testing.assert_array_equal(hits,a['hits'])
    assert m['reader_bytes']==m['linux_read_bytes']==int(a['reads'].sum())*4096 and m['reader_requests']==int(a['reads'].sum())
    assert m['swap_before_kib']==m['swap_after_kib']==0
    if c['kind'] in [1,2,3,4,5,7,8]:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
    if c['kind']==0 and c['L'] in [100,300]:
        old=D.parent/'p0_b/runs'/split/f'Tuned_DiskANN_L{c["L"]}_p0_r0.ids'
        assert old.read_bytes()==p.with_suffix('.ids').read_bytes(),'baseline ID regression'
    r=c|dict(split=split,repeat=rep,recall=float(hits.sum()/100),total_hits=int(hits.sum()),p95_ms=float(np.percentile(a['latency_us'],95)/1000),mean_ms=float(a['latency_us'].mean()/1000),qps=float(m['qps']),physical_bytes=float(m['linux_read_bytes']/1000),reads=float(a['reads'].mean()),expansions=float(a['expansions'].mean()),skips=float(a['skips'].mean()),model_calls=float(a['model_calls'].mean()),et_stop_fraction=float(a['et_stopped'].mean()),io_ms=float(a['io_us'].mean()/1000),io_rounds=float(a['io_rounds'].mean()),progress_eligible=float(a['progress_eligible'].mean()),progress_extra_skips=float(a['progress_extra_skips'].mean()),ids_sha256=sha(p.with_suffix('.ids')),missing_result_slots=int(missing.sum()))
    checked[cache]=r;return r
def load(split,cs,rep=0,ordered=False,phase='grid'):
    todo=[c for c in cs if not complete(split,c,rep)];rng=random.Random(P['seed']+rep+(100 if split=='test' else 0))
    if ordered:batches=[[c] for c in todo]
    else:
        rng.shuffle(todo);batches=[]
        for kind in sorted({c['kind'] for c in todo}):
            group=[c for c in todo if c['kind']==kind];batches.extend(group[i:i+8] for i in range(0,len(group),8))
        rng.shuffle(batches)
    for i,batch in enumerate(batches):
        digest=hashlib.sha256(' '.join(c['key'] for c in batch).encode()).hexdigest()[:12];tag=f'{split}_r{rep}_{digest}'
        cp=D/'logs'/f'{tag}.configs';cp.write_text(''.join(f'{c["key"]} {c["kind"]} {c["L"]} {c["tau"]:.17g} {c["et_tau"]:.17g} {c["rank"]} {c["exp"]} {c["late_tau"]:.17g}\n' for c in batch))
        cmd=[D/'native/search',Path(P['index']),D/'data'/f'{split}.fbin',D/'data'/f'{split}.gt',D/'models',ROOT/'data/sift/sift_base.fbin',cp,D/'runs'/split,str(rep)]
        dump(D/'STATUS.json',dict(phase=phase,split=split,repeat=rep,batch=i+1,batches=len(batches),keys=[c['key'] for c in batch],complete=False,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        print(time.strftime('%H:%M:%S'),phase,split,rep,i+1,'/',len(batches),[c['key'] for c in batch],flush=True)
        with (D/'logs'/f'{tag}.log').open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
    return [read(split,c,rep) for c in cs]
def aggregate(runs):
    out=[]
    for key in sorted({r['key'] for r in runs}):
        rs=[r for r in runs if r['key']==key];r=rs[0].copy();r.pop('repeat');r['repeats']=len(rs)
        for k in ['total_hits','physical_bytes','reads','expansions','skips','model_calls','et_stop_fraction','progress_eligible','progress_extra_skips','ids_sha256','missing_result_slots']:assert len({s[k] for s in rs})==1,(key,k)
        for k in ['p95_ms','mean_ms','qps','io_ms']:r[k]=float(np.median([s[k] for s in rs]))
        r['p95_min_ms']=min(s['p95_ms'] for s in rs);r['p95_max_ms']=max(s['p95_ms'] for s in rs);out.append(r)
    return out
def parity(rows):
    baseline={r['L']:r for r in rows if r['kind']==0};keys=[]
    for r in rows:
        if r['kind'] and r['preset']==0:
            for k in ['total_hits','reads','expansions','ids_sha256']:assert r[k]==baseline[r['L']][k],(r['key'],k)
            assert r['skips']==0 and r['et_stop_fraction']==0;keys.append(r['key'])
    return keys
def frontier(rows,metric):
    out=[];best=float('inf')
    for r in sorted(rows,key=lambda r:(-r['total_hits'],-r[metric] if metric=='qps' else r[metric])):
        cost=-r[metric] if metric=='qps' else r[metric]
        if cost<best:out.append(r);best=cost
    return sorted(out,key=lambda r:r['total_hits'])
validation=load('validation',configs,phase='P0B_validation');dump(D/'validation.json',validation);vp=parity(validation)
# P0-A: one six-method pool; dynamic versions match each static default's validation quality.
ac=[make(0,300),make(1,300,3),make(2,300,4),make(3,300,4)]
adetail=[]
for static_kind,dynamic_kind in [(2,4),(3,5)]:
    target=next(r['recall'] for r in validation if r['kind']==static_kind and r['L']==300 and r['preset']==4)
    feasible=[r for r in validation if r['kind']==dynamic_kind and r['L']==300 and r['recall']>=target]
    selected=min(feasible,key=lambda r:(r['p95_ms'],-r['total_hits'],r['key'])) if feasible else min((r for r in validation if r['kind']==dynamic_kind and r['L']==300),key=lambda r:(-r['total_hits'],r['p95_ms'],r['key']))
    adetail.append(dict(family=families[dynamic_kind],key=selected['key'],static_validation_recall=target,dynamic_validation_recall=selected['recall'],validation_quality_matched=bool(feasible)))
    ac.append(next(c for c in configs if c['key']==selected['key']))
dump(D/'P0A_DYNAMIC_SELECTION_DETAILS.json',adetail)
if (D/'P0A_SELECTION_LOCK.json').exists():assert json.loads((D/'P0A_SELECTION_LOCK.json').read_text())==ac
else:dump(D/'P0A_SELECTION_LOCK.json',ac)
selection=[]
for family in families:
    for target in P['targets']:
        for metric in ['p95_ms','qps','physical_bytes']:
            feasible=[r for r in validation if r['family']==family and r['recall']>=target]
            win=min(feasible,key=lambda r:(-r[metric] if metric=='qps' else r[metric],-r['total_hits'],r['key'])) if feasible else None
            selection.append(dict(family=family,target=target,objective=metric,key=win['key'] if win else None,validation_recall=win['recall'] if win else None))
if (D/'SELECTION_LOCK.json').exists():assert json.loads((D/'SELECTION_LOCK.json').read_text())==selection
else:dump(D/'SELECTION_LOCK.json',selection)
aruns=[]
for rep in range(12):aruns+=load('test',ac[rep%6:]+ac[:rep%6],rep,True,'P0A_shared_six_methods')
save(D/'p0_a_summary.csv',aggregate(aruns));dump(D/'p0_a_runs.json',aruns)
initial=load('test',configs,phase='P0B_test_grid');tp=parity(initial)
allruns=list(initial)
for c in ac:
    for rep in range(1,12):allruns.append(read('test',c,rep))
rounds=0
while True:
    current=aggregate(allruns);keys={s['key'] for s in selection if s['key']};keys.update(c['key'] for c in ac)
    for family in families:
        rs=[r for r in current if r['family']==family]
        for metric in ['p95_ms','qps','physical_bytes']:keys.update(r['key'] for r in frontier(rs,metric))
    common=set.intersection(*[{r['total_hits'] for r in current if r['family']==f} for f in families])
    for hits in common:
        for family in families:
            rs=[r for r in current if r['family']==family and r['total_hits']==hits]
            for metric in ['p95_ms','qps','physical_bytes']:keys.add(min(rs,key=lambda r:-r[metric] if metric=='qps' else r[metric])['key'])
    todo=[c for c in configs if c['key'] in keys and next(r['repeats'] for r in current if r['key']==c['key'])<3]
    if not todo:break
    rounds+=1
    for rep in [1,2]:allruns+=load('test',todo,rep,phase='P0B_frontier_repeats')
    dump(D/'all_test_runs_checkpoint.json',allruns)
current=aggregate(allruns);dump(D/'all_test_runs.json',allruns);dump(D/'aggregate.json',current);save(D/'all_configurations.csv',current)
assert all(sha(p)==h for p,h in json.loads((D/'INPUT_LOCK.json').read_text()).items())
assert {n:sha(D/'models'/f'{n}.bin') for n in ['state','lr','et']}==locks['models']
dump(D/'verification.json',dict(passed=True,validation_disabled_parity=vp,test_disabled_parity=tp,unique_checked_runs=len(checked),all_recalls_recomputed=True,physical_io_verified=True,no_swap=True,inputs_unchanged=True,shared_controls=True,baseline_L100_L300_ids_equal_old_P0=True))
dump(D/'MEASUREMENTS_COMPLETE.json',dict(complete=True,validation_configs=576,test_configs=576,test_runs=len(allruns),P0A_runs=len(aruns),P0A_repeats=12,refinement_rounds=rounds,shared_controls=True))
print('MEASUREMENTS COMPLETE',flush=True)
