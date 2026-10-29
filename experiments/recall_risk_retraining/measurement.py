"""Measurement helpers copied from the verified fixed-L100 runner; globals supplied by run.py."""
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
    r=c|dict(split=split,repeat=rep,total_hits=int(hits.sum()),recall=float(hits.sum()/100),p95_ms=float(np.percentile(a['latency_us'],95)/1000),p50_ms=float(np.percentile(a['latency_us'],50)/1000),p99_ms=float(np.percentile(a['latency_us'],99)/1000),model_calls=float(a['model_calls'].mean()),qps=float(m['qps']),mean_ms=float(a['latency_us'].mean()/1000),physical_bytes=float(m['linux_read_bytes']/1000),reads=float(a['reads'].mean()),expansions=float(a['expansions'].mean()),skips=float(a['skips'].mean()),extra_skips=float(a['progress_extra_skips'].mean()),eligible=float(a['progress_eligible'].mean()),io_ms=float(a['io_us'].mean()/1000),io_rounds=float(a['io_rounds'].mean()),ids_sha256=sha(p.with_suffix('.ids')))
    cache[ck]=r;return r
def load(split,cs,rep=0,ordered=False):
    todo=[c for c in cs if not complete(split,c,rep)];rng=random.Random(P['seed']+rep)
    if ordered:batches=[[c] for c in todo]
    else:
        batches=[]
        for group in sorted({(c['variant'],c['kind']) for c in todo}):
            a=[c for c in todo if (c['variant'],c['kind'])==group];rng.shuffle(a);batches.append(a)
        rng.shuffle(batches)
    for i,batch in enumerate(batches):
        digest=hashlib.sha256(' '.join(c['key'] for c in batch).encode()).hexdigest()[:10];tag=f'{split}_r{rep}_{digest}';cp=D/'logs'/f'{tag}.configs'
        cp.write_text(''.join(f'{c["key"]} {c["kind"]} {c["inference_L"]} {c["tau"]:.17g} 1.1 {c["rank"]} {c["exp"]} {c["late_tau"]:.17g}\n' for c in batch))
        models=A/'models_train100' if batch[0]['variant']=='old' else D/'models'/batch[0]['variant']
        cmd=[A/'native/search',D.parent/'sift1m/index',W/'data'/f'{split}.fbin',W/'data'/f'{split}.gt',models,ROOT/'data/sift/sift_base.fbin',cp,D/'runs'/split,str(rep)]
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
