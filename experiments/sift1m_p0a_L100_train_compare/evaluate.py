"""Fixed inference L100 with two single-L training groups and shared controls."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,hashlib,json,random,struct,subprocess,time,shutil
import numpy as np
D=Path(__file__).resolve().parent;W=D.parent/'sift1m_wide_p0';ROOT=D.parents[1];P=json.loads((D/'PROTOCOL.json').read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
def save(p,rows):
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert (D/'TRAIN_COMPLETE.json').exists()
raw=np.fromfile(D/'data/calibration.prune.bin','<f4').reshape(-1,20);x=np.ascontiguousarray(raw[:,3:]);y=raw[:,2].astype(bool)
assert np.all(x[:,15]==100) and np.all(x[:,10]==0)
states=D/'data/validation_states.bin'
with states.open('wb') as f:f.write(struct.pack('<I',len(x)));x.astype('<f4').tofile(f)
thresholds={}
for L in [100,300]:
    M=D/f'models_train{L}';out=M/'validation_scores.bin'
    subprocess.run([str(D/'native/score'),str(M),str(states),str(out)],check=True)
    with out.open('rb') as f:n=np.fromfile(f,'<u4',1)[0];scores=np.fromfile(f,'<f4').reshape(n,2)
    assert n==len(x);thresholds[str(L)]={}
    for j,name in enumerate(['lr','state']):
        keep=np.sort(scores[y,j]);points=[]
        for risk in P['risks']:
            tau=0. if risk==0 else float(keep[min(len(keep)-1,int(np.floor(len(keep)*risk)))])
            false=int(np.sum(scores[y,j]<tau));assert false<=int(np.floor(len(keep)*risk))
            points.append(dict(risk=risk,tau=tau,retain_count=len(keep),false_skips=false))
        thresholds[str(L)][name]=points
dump(D/'thresholds_L100.json',thresholds)
def make(key,family,kind,training_L=0,tau=0.,late=0.,rank=100,exp=-1,preset=-1):return dict(key=key,family=family,kind=kind,training_L=training_L,inference_L=100,tau=tau,late_tau=late,rank=rank,exp=exp,preset=preset)
controls=[make('DiskANN','DiskANN',0),make('Heuristic','Heuristic',1,rank=72,exp=100)]
static=[]
for L in [100,300]:
    for name,family,kind in [('state','MLP',2),('lr','LR',3)]:static.append(make(f'{family}_train{L}_static',family,kind,L,thresholds[str(L)][name][4]['tau']))
dynamic=[]
for name,family,kind in [('state','MLP',4),('lr','LR',5)]:
    for j,(base,late,r,e) in enumerate(P['dynamic_presets']):dynamic.append(make(f'{family}_train300_dynamic_p{j}',family,kind,300,thresholds['300'][name][base]['tau'],thresholds['300'][name][late]['tau'],int(100*r),-1 if e<0 else int(100*e),j))
configs=controls+static+dynamic;dump(D/'configs.json',configs)
lock=dict(protocol=sha(D/'PROTOCOL.json'),thresholds=sha(D/'thresholds_L100.json'),grid=sha(D/'configs.json'),runner=sha(D/'native/search'),weights={str(L):{name:sha(D/f'models_train{L}'/f'{name}.bin') for name in ['lr','state']} for L in [100,300]})
if (D/'RUN_LOCK.json').exists():assert json.loads((D/'RUN_LOCK.json').read_text())==lock
else:dump(D/'RUN_LOCK.json',lock)
for p in [D/'runs/validation',D/'runs/test',D/'logs']:p.mkdir(parents=True,exist_ok=True)
gt={}
for split in ['validation','test']:
    with (W/'data'/f'{split}.gt').open('rb') as f:n,k=np.fromfile(f,'<u4',2);gt[split]=np.fromfile(f,'<u4',count=int(n*k)).reshape(n,k)[:,:10]
cache={}
def prefix(split,c,rep):return D/'runs'/split/f'{c["key"]}_r{rep}'
def complete(split,c,rep):
    p=prefix(split,c,rep)
    try:
        m=json.loads(p.with_suffix('.meta.json').read_text())
        return m['queries']==1000 and m['reader_requests']*4096==m['reader_bytes']==m['linux_read_bytes'] and p.with_suffix('.ids').stat().st_size==80000 and len(p.with_suffix('.csv').read_text().splitlines())==1001
    except (OSError,ValueError,KeyError):return False
def read(split,c,rep):
    ck=(split,c['key'],rep)
    if ck in cache:return cache[ck]
    p=prefix(split,c,rep);a=np.genfromtxt(p.with_suffix('.csv'),delimiter=',',names=True);m=json.loads(p.with_suffix('.meta.json').read_text());ids=np.fromfile(p.with_suffix('.ids'),'<u8').reshape(1000,10)
    valid=ids<1000000;missing=ids==np.iinfo(np.uint32).max
    assert np.all(valid|missing) and np.all(np.diff(missing.astype(np.int8),axis=1)>=0)
    assert all(len(set(row[mask]))==int(mask.sum()) for row,mask in zip(ids,valid))
    hits=np.array([len(set(row)&set(g)) for row,g in zip(ids,gt[split])]);np.testing.assert_array_equal(hits,a['hits'])
    assert m['reader_requests']==int(a['reads'].sum()) and m['reader_bytes']==m['linux_read_bytes']==int(a['reads'].sum())*4096
    assert m['swap_before_kib']==m['swap_after_kib']==0
    if c['kind']:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
    if c['key']=='DiskANN':assert p.with_suffix('.ids').read_bytes()==(W/'runs'/split/'DiskANN_L100_p0_r0.ids').read_bytes()
    if c['key']=='Heuristic':assert p.with_suffix('.ids').read_bytes()==(W/'runs'/split/'Heuristic_L100_p3_r0.ids').read_bytes()
    r=c|dict(split=split,repeat=rep,total_hits=int(hits.sum()),recall=float(hits.sum()/100),p95_ms=float(np.percentile(a['latency_us'],95)/1000),qps=float(m['qps']),mean_ms=float(a['latency_us'].mean()/1000),physical_bytes=float(m['linux_read_bytes']/1000),reads=float(a['reads'].mean()),expansions=float(a['expansions'].mean()),skips=float(a['skips'].mean()),extra_skips=float(a['progress_extra_skips'].mean()),eligible=float(a['progress_eligible'].mean()),io_ms=float(a['io_us'].mean()/1000),io_rounds=float(a['io_rounds'].mean()),ids_sha256=sha(p.with_suffix('.ids')))
    cache[ck]=r;return r
def load(split,cs,rep=0,ordered=False):
    todo=[c for c in cs if not complete(split,c,rep)];rng=random.Random(P['seed']+rep)
    if ordered:batches=[[c] for c in todo]
    else:
        batches=[]
        for group in sorted({(c['training_L'],c['kind']) for c in todo}):
            a=[c for c in todo if (c['training_L'],c['kind'])==group];rng.shuffle(a);batches.append(a)
        rng.shuffle(batches)
    for i,batch in enumerate(batches):
        digest=hashlib.sha256(' '.join(c['key'] for c in batch).encode()).hexdigest()[:10];tag=f'{split}_r{rep}_{digest}';cp=D/'logs'/f'{tag}.configs'
        cp.write_text(''.join(f'{c["key"]} {c["kind"]} 100 {c["tau"]:.17g} 1.1 {c["rank"]} {c["exp"]} {c["late_tau"]:.17g}\n' for c in batch))
        models=D/f'models_train{batch[0]["training_L"] or 100}'
        cmd=[D/'native/search',D.parent/'sift1m/index',W/'data'/f'{split}.fbin',W/'data'/f'{split}.gt',models,ROOT/'data/sift/sift_base.fbin',cp,D/'runs'/split,str(rep)]
        dump(D/'STATUS.json',dict(phase=split,repeat=rep,batch=i+1,batches=len(batches),keys=[c['key'] for c in batch],complete=False))
        print(time.strftime('%H:%M:%S'),split,rep,[c['key'] for c in batch],flush=True)
        with (D/'logs'/f'{tag}.log').open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
        for c in batch:
            p=prefix(split,c,rep);m=json.loads(p.with_suffix('.meta.json').read_text())
            if m['reader_requests']*4096!=m['reader_bytes'] or m['reader_bytes']!=m['linux_read_bytes']:
                journal=json.loads((D/'IO_REJECTIONS.json').read_text()) if (D/'IO_REJECTIONS.json').exists() else []
                attempt=1+sum(r['split']==split and r['key']==c['key'] and r['repeat']==rep for r in journal)
                assert attempt<=3,('repeated IO mismatch',c['key'],m)
                dest=D/'rejected_io'/f'{split}_{c["key"]}_r{rep}_attempt{attempt}';dest.mkdir(parents=True,exist_ok=True)
                for ext in ['csv','ids','meta.json']:shutil.move(str(p.with_suffix('.'+ext)),str(dest/p.with_suffix('.'+ext).name))
                journal.append(dict(split=split,key=c['key'],repeat=rep,reason='kernel/reader IO mismatch',archive=str(dest),metadata=m));dump(D/'IO_REJECTIONS.json',journal)
                load(split,[c],rep,True)
    return [read(split,c,rep) for c in cs]
validation=load('validation',configs);dump(D/'validation.json',validation);vb=validation[0]
for r in validation:
    if r['kind'] in [4,5] and r['preset']==0:
        for k in ['total_hits','reads','expansions','ids_sha256']:assert r[k]==vb[k]
selection=list(controls+static);detail=[]
for family,kind in [('MLP',4),('LR',5)]:
    target=next(r['recall'] for r in validation if r['family']==family and r['training_L']==300 and r['kind'] in [2,3])
    candidates=[r for r in validation if r['kind']==kind];feasible=[r for r in candidates if r['recall']>=target]
    winner=min(feasible,key=lambda r:(r['p95_ms'],-r['total_hits'],r['key'])) if feasible else min(candidates,key=lambda r:(-r['total_hits'],r['p95_ms'],r['key']))
    selection.append(next(c for c in dynamic if c['key']==winner['key']));detail.append(dict(family=family,key=winner['key'],static_validation_recall=target,dynamic_validation_recall=winner['recall'],quality_matched=bool(feasible)))
if (D/'SELECTION_LOCK.json').exists():assert json.loads((D/'SELECTION_LOCK.json').read_text())==selection
else:dump(D/'SELECTION_LOCK.json',selection)
dump(D/'DYNAMIC_SELECTION.json',detail);print('SELECTION LOCKED',detail,flush=True)
runs=[]
for rep in range(16):runs+=load('test',selection[rep%8:]+selection[:rep%8],rep,True)
summary=[]
for c in selection:
    rs=[r for r in runs if r['key']==c['key']];assert len(rs)==16
    r=rs[0].copy();r.pop('repeat');r['repeats']=len(rs)
    for k in ['total_hits','physical_bytes','reads','expansions','skips','extra_skips','eligible','ids_sha256']:assert len({v[k] for v in rs})==1,(c['key'],k)
    for k in ['p95_ms','qps','mean_ms','io_ms']:r[k]=float(np.median([v[k] for v in rs]))
    r['p95_min_ms']=min(v['p95_ms'] for v in rs);r['p95_max_ms']=max(v['p95_ms'] for v in rs);summary.append(r)
save(D/'summary.csv',summary);dump(D/'summary.json',summary);dump(D/'all_runs.json',runs)
assert len({(r['key'],r['repeat']) for r in runs})==len(runs)
assert all(sha(p)==h for p,h in json.loads((D/'INPUT_LOCK.json').read_text()).items())
assert lock['weights']=={str(L):{name:sha(D/f'models_train{L}'/f'{name}.bin') for name in ['lr','state']} for L in [100,300]}
dump(D/'verification.json',dict(passed=True,inference_L_all_100=True,shared_controls=True,unique_test_runs=len(runs),queries_per_run=1000,all_recalls_recomputed=True,IO_verified=True,no_swap=True,repeat_ids_and_counts_equal=True,inputs_and_weights_unchanged=True))
dump(D/'MEASUREMENTS_COMPLETE.json',dict(complete=True,groups=8,repeats=16,test_runs=len(runs),validation_configs=len(configs)))
print('MEASUREMENTS COMPLETE',flush=True)
