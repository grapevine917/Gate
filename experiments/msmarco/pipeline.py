"""MS MARCO text -> embeddings -> DiskANN -> frozen marks / LR / fixed-bank reuse."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, os, random, struct, subprocess, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
ROOT = Path(__file__).resolve().parent
LG = ROOT.parents[1]
DATA = LG.parent / 'msmarco-data'
BIN = LG / 'DiskANN/build/apps'


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def matrix(p, dtype='<f4'):
    with open(p, 'rb') as f: n, d = struct.unpack('<II', f.read(8))
    if p.stat().st_size != 8 + n*d*4: raise ValueError(f'Truncated or unsupported matrix: {p}')
    return np.memmap(p, mode='r', offset=8, dtype=dtype, shape=(n, d))


def write(p, a):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'wb') as f:
        f.write(struct.pack('<II', *a.shape)); np.asarray(a).tofile(f)


def run(args, log, env=None):
    print('RUN', ' '.join(map(str, args)), flush=True)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(('DISKANN_', 'CF_'))}
    clean.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    clean.update(env or {})
    with log.open('w') as f:
        subprocess.run(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, env=clean, check=True)


def queries(p):
    seen = set()
    with p.open() as f:
        for line in f:
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) not in (2,3): raise ValueError(f'Query columns: {p}')
            qid, text = fields[0], fields[-1].strip()
            if not qid.isdigit() or not text or qid in seen: raise ValueError(f'Invalid query {qid}')
            seen.add(qid)
            yield {'id':qid, 'text':text}


def ensure_native(a):
    # Let CMake track header dependencies too: an existing executable can still
    # contain an older OnlineGate reader that does not understand LRGATE01.
    targets = {
        'build': ['build_disk_index'],
        'collect': ['collect_io_labels'],
        'evaluate': ['search_disk_index', 'search_query_skip_cache'],
        'all': ['build_disk_index', 'collect_io_labels', 'search_disk_index', 'search_query_skip_cache'],
    }.get(a.stage, [])
    if targets:
        run(['cmake', '--build', BIN.parent, '--target', *targets, '--parallel', '4'],
            a.out/'native_build.log')


def prepare(a):
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    sources = {n:sha(DATA/n) for n in ['train_queries.tsv','dev_queries.tsv','collection.tsv.gz']}
    config = {'sources_sha256':sources, 'seed':20261017, 'base_limit':a.base_limit,
              'model':a.model, 'revision':a.revision, 'split_size':1000,
              'metric':'l2 on unit-normalized float32 embeddings', 'arrival_trace':False}
    path = out/'inputs.json'
    if path.exists():
        if json.loads(path.read_text())['config'] != config: raise ValueError('Existing run configuration differs; use a new --out directory')
        return
    train = list(queries(DATA/'train_queries.tsv')); dev = list(queries(DATA/'dev_queries.tsv'))
    if set(x['id'] for x in train) & set(x['id'] for x in dev): raise ValueError('Overlapping source query IDs')
    # Remove exact text overlap across splits; no fabricated repeats or perturbations.
    rng = random.Random(20261017); rng.shuffle(dev); rng.shuffle(train)
    used = set(); splits = {}
    for name, source in [('validation',dev),('test',dev),('train',train)]:
        selected=[]
        for row in source:
            if row['text'] in used: continue
            used.add(row['text']); selected.append(row)
            if len(selected)==1000: break
        if len(selected)!=1000: raise ValueError('Insufficient disjoint queries')
        splits[name]=selected
    for s, rows in splits.items(): dump(out/s/'texts.json', rows)
    dump(path, {'config':config, 'split_ids':{s:[r['id'] for r in rows] for s,rows in splits.items()},
                'order':'fixed seeded permutation; not chronological traffic',
                'subset_warning':'GT must be recomputed on the actual base; external Web Search vectors are excluded'})


def encode(a):
    from sentence_transformers import SentenceTransformer
    local = a.model_path or (ROOT/'models/all-MiniLM-L6-v2' if (ROOT/'models/all-MiniLM-L6-v2/modules.json').is_file() else None)
    if local is not None:
        local = Path(local).resolve()
        if not (local/'modules.json').is_file():raise ValueError(f'Not a complete SentenceTransformer model directory: {local}')
        print(f'Loading local model: {local}', flush=True)
    try:
        model = SentenceTransformer(str(local) if local else a.model, revision=None if local else a.revision, device=a.device, local_files_only=a.offline or local is not None)
    except OSError as exc:
        raise RuntimeError('无法加载编码模型。sentence-transformers 软件包已安装，但仍需要模型权重。请先在可联网终端运行 experiments/msmarco/download_model.py，或用 --model-path 指定完整模型目录。原始错误: '+str(exc)) from exc
    identity={'model':a.model,'revision':a.revision,'local_files_sha256':{str(p.relative_to(local)):sha(p) for p in sorted(local.rglob('*')) if p.is_file() and '.cache' not in p.parts} if local else None}
    ip=a.out/'encoder_identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Encoder changed; use a new --out to avoid mixing embedding shards')
    dump(ip,identity)
    dim = model.get_sentence_embedding_dimension()
    # Atomic per-shard commits make a long corpus encoding resumable.
    def embed(rows, dest):
        dest.mkdir(parents=True, exist_ok=True); paths=[]; total=0
        def commit(batch):
            nonlocal total
            ids=[r['id'] for r in batch]; digest=hashlib.sha256(json.dumps(batch,ensure_ascii=False).encode()).hexdigest()
            p=dest/f'{len(paths):06d}.fbin'; meta=p.with_suffix('.json')
            if p.exists() and meta.exists():
                if json.loads(meta.read_text()) != {'rows_sha256':digest,'ids':ids}: raise ValueError('Shard input changed')
                arr=matrix(p)
                if arr.shape!=(len(batch),dim): raise ValueError('Invalid shard shape')
            else:
                arr=model.encode([r['text'] for r in batch],batch_size=a.batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                if not np.isfinite(arr).all(): raise ValueError('Nonfinite embeddings')
                tmp=p.with_suffix('.tmp');write(tmp,np.asarray(arr,dtype='<f4'));tmp.replace(p)
                dump(meta,{'rows_sha256':digest,'ids':ids})
            paths.append(p);total+=len(batch);print(dest.name,total,flush=True)
        batch=[]
        for row in rows:
            batch.append(row)
            if len(batch)==4096: commit(batch);batch=[]
        if batch:commit(batch)
        target=dest.parent/('base.fbin' if dest.name=='base_shards' else 'query.fbin')
        tmp=target.with_suffix('.tmp')
        with tmp.open('wb') as f:
            f.write(struct.pack('<II',total,dim))
            for p in paths:
                with p.open('rb') as src:
                    src.read(8)
                    for b in iter(lambda:src.read(8<<20),b''):f.write(b)
        tmp.replace(target)
        return {'rows':total,'dim':dim,'sha256':sha(target)}
    def documents():
        seen=set()
        with gzip.open(DATA/'collection.tsv.gz','rt') as f:
            for i,line in enumerate(f):
                if a.base_limit and i>=a.base_limit:break
                pid,text=line.rstrip('\r\n').split('\t',1)
                if pid in seen:raise ValueError('Duplicate passage ID')
                seen.add(pid)
                yield {'id':pid,'text':text}
    info={'model':a.model,'revision':a.revision,'max_seq_length':model.max_seq_length,'normalized':True, 'local_model_path':str(local) if local else None}
    info['base']=embed(documents(),a.out/'base_shards')
    for s in ['train','validation','test']:info[s]=embed(json.loads((a.out/s/'texts.json').read_text()),a.out/s/'shards')
    dump(a.out/'embeddings.json',info)


def truth(a):
    x=matrix(a.out/'base.fbin')
    if len(x)<10:raise ValueError('Need at least ten base vectors')
    for split in ['train','validation','test']:
        q=matrix(a.out/split/'query.fbin'); result=[]
        for st in range(0,len(q),32):
            z=q[st:st+32].astype(np.float64);best=np.full((len(z),10),np.inf);ids=np.zeros((len(z),10),np.uint32)
            for bs in range(0,len(x),8192):
                b=x[bs:bs+8192].astype(np.float64)
                d=(z*z).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*z@b.T
                # Stable sorting defines equal-distance ties by base row ID.
                ix=np.argsort(d,axis=1,kind='stable')[:,:10]
                v=np.concatenate([best,np.take_along_axis(d,ix,1)],1);ii=np.concatenate([ids,(ix+bs).astype(np.uint32)],1)
                keep=np.argsort(v,axis=1,kind='stable')[:,:10]
                best=np.take_along_axis(v,keep,1);ids=np.take_along_axis(ii,keep,1)
            result.append(ids);print('GT',split,st+len(z),flush=True)
        write(a.out/split/'gt.bin',np.concatenate(result))
    dump(a.out/'gt.json',{'method':'exhaustive float64 squared L2, stable base-row tie ordering','base_sha256':sha(a.out/'base.fbin')})


def build(a):
    run([BIN/'build_disk_index','--data_type','float','--dist_fn','l2','--data_path',a.out/'base.fbin',
         '--index_path_prefix',a.out/'index','-B','0.1','-M','8','-T','4','-R','64','-L','100'],a.out/'build.log')


def collect(a):
    # Zero-threshold state-only LR enables feature auditing without any pruning.
    p=a.out/'audit_model.bin'
    with p.open('wb') as f:
        f.write(b'LRGATE01');f.write(struct.pack('<I',0))
        for arr in [np.zeros(17),np.ones(17),np.zeros(17),np.zeros(1)]:np.asarray(arr,dtype='<f4').tofile(f)
    for s in ['train','validation']:
        run([BIN/'collect_io_labels',a.out/'index',a.out/s/'query.fbin',a.out/s/'gt.bin',p,0,a.out/s/'collected',100,10000],a.out/s/'collect.log')


def fit(a):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    def dataset(s):
        p=a.out/s
        features=np.genfromtxt(p/'collected_features.csv',delimiter=',',names=True)
        labels=np.genfromtxt(p/'collected_candidates.csv',delimiter=',',names=True)
        lookup={(int(r['query']),int(r['node'])):i for i,r in enumerate(features)}
        miss=labels['cache_miss']!=0;lab=labels[miss]
        ix=[lookup[(int(r['query']),int(r['node']))] for r in lab]
        states=np.column_stack([features[f's{i}'][ix] for i in range(17)])
        if not np.isfinite(states).all():raise ValueError('Nonfinite LR states')
        y=((lab['singleton_recall_ok']==1)&(lab['singleton_delta']>0)).astype(int)
        return states,y,labels
    x,y,labels=dataset('train');v,vy,_=dataset('validation')
    scaler=StandardScaler().fit(x);model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=42).fit(scaler.transform(x),y)
    mean=scaler.mean_.astype('<f4');scale=scaler.scale_.astype('<f4');w=model.coef_[0].astype('<f4');bias=model.intercept_.astype('<f4')
    with (a.out/'lr.bin').open('wb') as f:
        f.write(b'LRGATE01');f.write(struct.pack('<I',0))
        for arr in [mean,scale,w,bias]:arr.tofile(f)
    retain=expit(-(((v-mean)/scale)@w+bias[0])).astype(np.float32)
    np.testing.assert_allclose(1-retain,model.predict_proba(scaler.transform(v))[:,1],atol=1e-4)
    unsafe=np.sort(retain[vy==0]);taus=sorted(set([0.]+[float(unsafe[min(int(len(unsafe)*r),len(unsafe)-1)]) for r in [.001,.005,.01,.02]]))
    n=len(matrix(a.out/'base.fbin'));visits=np.zeros(n,np.uint32);low=np.zeros(n,np.uint32)
    for row in labels:
        if row['cache_miss']:
            j=int(row['node']);visits[j]+=1;low[j]+=int(row['admitted']==0 and row['in_result']==0)
    mask=(visits>0)&(low==visits)
    with (a.out/'marks.bin').open('wb') as f:f.write(b'NDMARK01');f.write(struct.pack('<I',n));f.write(np.packbits(mask,bitorder='little').tobytes())
    dump(a.out/'training.json',{'taus':taus,'marked_nodes':int(mask.sum()),'train_events':len(y),'safe_saving_fraction':float(y.mean()),'marks_rule':'observed cache miss at least once; every event admitted=0 and not final top10','model_sha256':sha(a.out/'lr.bin')})


def search(a,s,key,rep,tau=None,marks=False):
    p=a.out/s/f'{key}_r{rep}';env={'DISKANN_QUERY_STATS':str(p)+'.csv'}
    if tau is not None:env.update(DISKANN_M3_MODEL=str(a.out/'lr.bin'),DISKANN_M3_TAU=str(tau))
    if marks:env['DISKANN_NODE_MARKS']=str(a.out/'marks.bin' if marks is True else marks)
    run([BIN/'search_disk_index','--data_type','float','--dist_fn','l2','--index_path_prefix',a.out/'index','--query_file',a.out/s/'query.fbin','--gt_file',a.out/s/'gt.bin','-K',10,'-L',100,'-W',4,'-T',4,'--num_nodes_to_cache',10000,'--result_path',p],Path(str(p)+'.log'),env)
    ids=matrix(Path(str(p)+'_100_idx_uint32.bin'),'<u4');gt=matrix(a.out/s/'gt.bin','<u4')
    stats=np.genfromtxt(str(p)+'.csv',delimiter=',',names=True)
    np.testing.assert_array_equal(stats['query_row'],np.arange(len(gt)))
    if np.any(stats['ssd_reads']>stats['expansions']):raise ValueError('Read accounting failed')
    summary=next(line.split() for line in Path(str(p)+'.log').read_text().splitlines() if line.startswith('ONLINE_SUMMARY '))
    return {'key':key,'rep':rep,'qps':float(summary[2]),'recall_pct':float(np.mean([len(set(i)&set(g)) for i,g in zip(ids,gt)])*10),'reads':float(stats['ssd_reads'].mean()),'latency_us':float(stats['latency_us'].mean()),'p95_us':float(np.percentile(stats['latency_us'],95)),'p99_us':float(np.percentile(stats['latency_us'],99))}


def reuse(a,s,tau,reps):
    identity={'tau':tau,'model':sha(a.out/'lr.bin'),'bank_queries':sha(a.out/'train/query.fbin'),'index':json.loads((a.out/'index_identity.json').read_text())}
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16];p=a.out/s/f'memo_{tag}'
    dump(a.out/f'bank_{tag}.identity.json',identity)
    cfg=Path(str(p)+'.configs');rows=[f'{key} {mode} {rho} {rep}' for rep in range(reps) for key,mode,rho in [('LR',0,0),('reuse',2,.01),('lookup',1,.01),('zero',2,0)]]
    random.Random(20261017).shuffle(rows);cfg.write_text('\n'.join(rows)+'\n')
    run([BIN/'search_query_skip_cache',a.out/'index',a.out/'train/query.fbin',a.out/'lr.bin',tau,a.out/s/'query.fbin',a.out/s/'gt.bin',cfg,p,a.out/f'bank_{tag}.bin'],Path(str(p)+'.log'))
    table=list(csv.DictReader(open(str(p)+'.csv')));out=[]
    throughput={}
    for line in Path(str(p)+'.log').read_text().splitlines():
        if line.startswith('MEMO '):
            v=line.split();throughput[(v[1],int(v[3]))]=float(v[-1])
    for rep in range(reps):
        for key in ['LR','reuse','lookup','zero']:
            rr=[r for r in table if r['key']==key and int(r['rep'])==rep]
            out.append({'key':key,'rep':rep,'qps':throughput[(key,rep)],'recall_pct':float(np.mean([float(r['hits']) for r in rr])*10),'reads':float(np.mean([float(r['reads']) for r in rr])),'latency_us':float(np.mean([float(r['total_us']) for r in rr])),'p95_us':float(np.percentile([float(r['total_us']) for r in rr],95)),'p99_us':float(np.percentile([float(r['total_us']) for r in rr],99)),'reused':float(np.mean([float(r['reused']) for r in rr])),'model_calls':float(np.mean([float(r['model_calls']) for r in rr]))})
        # Lookup-only must preserve the fresh LR search trajectory exactly.
        for ext in ['ids','dists']:
            if sha(Path(f'{p}_lookup_r{rep}.{ext}'))!=sha(Path(f'{p}_LR_r{rep}.{ext}')):raise ValueError('Lookup-only parity failed')
    return out


def evaluate(a):
    taus=json.loads((a.out/'training.json').read_text())['taus'];val=[]
    dump(a.out/'index_identity.json',{suffix:sha(Path(str(a.out/'index')+suffix)) for suffix in ['_disk.index','_pq_compressed.bin','_pq_pivots.bin']})
    configs=[('DiskANN',None,False),('global_once',None,True)]+[(f'LR_{i}',t,False) for i,t in enumerate(taus)]
    rng=random.Random(20261017)
    for rep in range(2):
        order=configs.copy();rng.shuffle(order)
        for k,t,m in order:val.append(search(a,'validation',k,rep,t,m))
    def avg(rows,key,field):return float(np.median([r[field] for r in rows if r['key']==key]))
    baseline=avg(val,'DiskANN','recall_pct');selected={}
    for budget in [.5,1.,2.]:
        feasible=[(k,t) for k,t,m in configs if t is not None and baseline-avg(val,k,'recall_pct')<=budget+1e-9]
        k,t=min(feasible,key=lambda kt:avg(val,kt[0],'reads'))
        selected[str(budget)]={'key':k,'tau':t}
    # Zero LR threshold and an empty mark table must reproduce the baseline.
    empty=a.out/'empty_marks.bin';n=len(matrix(a.out/'base.fbin'))
    with empty.open('wb') as f:f.write(b'NDMARK01');f.write(struct.pack('<I',n));f.write(bytes((n+7)//8))
    search(a,'validation','empty_marks',0,marks=empty)
    for key in ['empty_marks','LR_0']:
        for suffix in ['_100_idx_uint32.bin','_100_dists_float.bin']:
            if sha(Path(str(a.out/'validation'/f'{key}_r0')+suffix))!=sha(Path(str(a.out/'validation'/'DiskANN_r0')+suffix)):raise ValueError(f'Baseline parity failed: {key}')
    # Freeze the LR operating points before any held-out test evaluation.
    dump(a.out/'selection_lock.json',{'budgets_pp':selected,'validation':val,'test_used':False,'rho':.01,'bank':'fixed 1000 training queries; no test insertions'})
    test=[]
    for rep in range(3):
        order=[configs[0],configs[1]]+[(v['key'],v['tau'],False) for v in {x['key']:x for x in selected.values()}.values()];rng.shuffle(order)
        for k,t,m in order:test.append(search(a,'test',k,rep,t,m))
    memo={}
    for tau in sorted(set(v['tau'] for v in selected.values())):memo[str(tau)]={'validation':reuse(a,'validation',tau,2),'test':reuse(a,'test',tau,3)}
    dump(a.out/'results.json',{'correctness':{'zero_LR_and_empty_marks_match_baseline':True,'lookup_only_matches_fresh_LR':True},'validation':val,'selected':selected,'test':test,'fixed_bank_reuse':memo,'rolling_bank':'not implemented; no rolling claim','traffic':'seeded real-query workload, not chronological production trace'})
    lines=['# MS MARCO 三种方法对比','','固定查询划分；非生产时间序列。全局一次标记保持完整定义，不因超出预算而隐藏。LR 和 LR＋复用使用同一验证选中阈值；复用 rho=0.01，固定训练查询库。读取数为提交请求数，不是物理 SSD I/O。','','|方法|Recall@10 %|读取请求/q|平均延迟 μs|p95 μs|p99 μs|QPS|','|---|---:|---:|---:|---:|---:|---:|']
    for key in dict.fromkeys(r['key'] for r in test):lines.append('|'+key+'|'+'|'.join(f'{avg(test,key,f):.4f}' for f in ['recall_pct','reads','latency_us','p95_us','p99_us','qps'])+'|')
    for tau,result in memo.items():
        for key in ['LR','reuse']:lines.append('|'+f'{key} tau={tau}'+'|'+'|'.join(f'{avg(result["test"],key,f):.4f}' for f in ['recall_pct','reads','latency_us','p95_us','p99_us','qps'])+'|')
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','encode','truth','build','collect','fit','evaluate','all'],default='all')
    p.add_argument('--out',type=Path,default=ROOT/'full');p.add_argument('--model',default='sentence-transformers/all-MiniLM-L6-v2');p.add_argument('--revision',default='c9745ed1d9f207416be6d2e6f8de32d1f16199bf')
    p.add_argument('--model-path',type=Path,help='Complete local SentenceTransformer model directory');p.add_argument('--offline',action='store_true',help='Never attempt model network access')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=64);p.add_argument('--base-limit',type=int,default=0,help='0 means all passages; use a separate --out for subsets')
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    stages=[prepare,encode,truth,build,collect,fit,evaluate] if a.stage=='all' else [globals()[a.stage]]
    try:
        if a.stage not in ['all','prepare']:prepare(a)
        ensure_native(a)
        import importlib.metadata
        dump(a.out/'runtime.json',{'python':sys.version,'script_sha256':sha(Path(__file__)),
             'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn']},
             'native_binaries':{name:sha(BIN/name) for name in ['build_disk_index','search_disk_index','collect_io_labels','search_query_skip_cache']}})
        for stage in stages:
            print('STAGE',stage.__name__,flush=True);stage(a)
    except Exception as e:
        dump(a.out/'BLOCKED.json',{'stage':stage.__name__ if 'stage' in locals() else 'preflight','error':f'{type(e).__name__}: {e}','time':time.time(),'completed_experiment':False})
        raise
    (a.out/'BLOCKED.json').unlink(missing_ok=True)
    if a.stage in ['all','evaluate']:dump(a.out/'COMPLETE.json',{'completed_experiment':True,'results_sha256':sha(a.out/'results.json')})

if __name__=='__main__':main()
