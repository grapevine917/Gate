from common import *
freeze();source=json.loads((D/'INPUT_LOCK.json').read_text());assert all(sha(p)==h for p,h in source.items())
ids=[np.load(W/'data'/f'{s}_ids.npy') for s in ['train','validation','test']];assert all(not set(ids[i])&set(ids[j]) for i,j in [(0,1),(0,2),(1,2)])
def collect(split,start,end,folder,train):
    folder.mkdir(parents=True,exist_ok=True);p=folder/f'chunk_{start:04d}'
    if p.with_suffix('.json').exists():
        m=json.loads(p.with_suffix('.json').read_text());assert m['complete'] and m['rows']*132==p.with_suffix('.bin').stat().st_size;return
    cmd=[D/'native/collect',D.parent/'sift1m/index',W/'data'/f'{split}.fbin',W/'data'/f'{split}.gt',A/'models_train100',str(TAUS[0]),str(TAUS[1]),str(start),str(end),str(int(train)),p]
    dump(D/'STATUS.json',dict(phase='collect',split=split,start=start,end=end,log=str(p.with_suffix('.log'))))
    print(time.strftime('%H:%M:%S'),'COLLECT',split,start,end,flush=True)
    with p.with_suffix('.log').open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
    assert p.with_suffix('.json').exists()
if not (D/'SMOKE_COMPLETE.json').exists():
    collect('validation',0,4,D/'smoke',False)
    actual=np.fromfile(D/'smoke/chunk_0000.ids','<u8').reshape(4,3,10)
    for i,key in enumerate(['DiskANN','MLP_train100_static','LR_train100_static']):assert np.array_equal(actual[:,i],np.fromfile(A/'runs/validation'/f'{key}_r0.ids','<u8').reshape(1000,10)[:4])
    dump(D/'SMOKE_COMPLETE.json',dict(passed=True,historical_native_MLP_LR_IDs_equal=True,prefix_state_checked=True,shams=12))
for split,n in [('train',1200),('validation',1000),('test',1000)]:
    for start in range(0,n,100):collect(split,start,min(n,start+100),D/'labels'/split,split=='train')
    a=labels(split);assert np.isfinite(a['state']).all() and (a['state'][:,10]==0).all() and (a['state'][:,15]==100).all();assert np.allclose(a['weight'],a['population']/a['sampled'])
    assert set(np.unique(a['query']))==set(range(n))
    dump(D/f'LABELS_{split.upper()}_COMPLETE.json',dict(complete=True,queries=n,samples=len(a),risk_fraction=float(np.mean(a['keep_hits']>a['skip_hits'])),weighted_risk_fraction=float(np.average(a['keep_hits']>a['skip_hits'],weights=a['weight'])),proxy_fraction=float(a['proxy'].mean()),shams=n*3,hashes={str(p):sha(p) for p in sorted((D/'labels'/split).glob('chunk_*.bin'))}))
assert all(sha(p)==h for p,h in source.items());dump(D/'LABELS_COMPLETE.json',dict(complete=True,only_dataset='SIFT1M',rows={s:len(labels(s)) for s in ['train','validation','test']}));print('ALL LABELS COMPLETE',flush=True)
