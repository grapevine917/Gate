"""P0-A extension: real online skip reuse, validation selection, held-out repeated timing."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,subprocess,csv
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];P=R.parent/'p0_a';D=R.parent/'main_comparison/data';S=R.parent/'sift1m'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def log(s):print(s,flush=True)
protocol=json.loads((P/'PROTOCOL.json').read_text());tau=protocol['gate_threshold'];model=Path(protocol['model'])
# Freeze history to the first 1000 training queries; never use validation/test labels.
ids={s:np.load(D/f'{s}_ids.npy') for s in ['train','validation','test']}
assert not any(set(ids[a])&set(ids[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
with (D/'train.fbin').open('rb') as f:
 n,d=np.fromfile(f,'<u4',2);bank=np.fromfile(f,'<f4',count=1000*int(d)).reshape(1000,int(d))
with (R/'history.fbin').open('wb') as f:np.array(bank.shape,'<u4').tofile(f);bank.tofile(f)
np.save(R/'history_ids.npy',ids['train'][:1000])
pol=dict(base='P0-A',model=str(model),tau=tau,L=100,K=10,beam=4,threads=4,cache_nodes=10000,history_queries=1000,history_source='first 1000 disjoint training queries, real MLP skip only; frozen',rho_definition='squared L2(query,nearest historical query) / max(query squared norm, donor squared norm, 1e-12)',guard='same candidate rank and completed expansion difference <=4',rho_grid=[0,.01,.05,.1,.25,.5,1.0],validation_repeats=3,test_repeats=9,selection='separate max QPS / min P95 among reuse candidates with validation Recall >= heuristic and >= fresh MLP -0.05pp; fresh fallback; test never selects',timing='includes exact 1000-query NN lookup, allocation, membership/guard, inference, real O_DIRECT IO; excludes loading/history construction, 128-query warmup per config, GT/output',order='validation seeded random each round; test cyclic balanced order, nine repeats',limits=['previously observed test set; exploratory','no guarantee of improvement','reader direct requests measured; /proc/self/io unavailable in current environment; no block-layer/NAND measurement claim','finite validation selection and host timing noise'])
if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==pol
else:dump(R/'PROTOCOL.json',pol)
paths=[model,R/'search',R/'search.cpp',R/'run.py',R/'history.fbin',R/'history_ids.npy',ROOT/'experiments/main_comparison/native/libdiskann.a',ROOT/'experiments/main_comparison/native/online_gate.h',S/'index_disk.index',S/'index_pq_pivots.bin',S/'index_pq_compressed.bin']+[D/f'{s}.{ext}' for s in ['validation','test'] for ext in ['fbin','gt']]
lock={str(p):sha(p) for p in paths}
if (R/'INPUT_LOCK.json').exists():assert json.loads((R/'INPUT_LOCK.json').read_text())==lock
else:dump(R/'INPUT_LOCK.json',lock)
base=[dict(key='Vanilla',mode=7,rho=0),dict(key='Heuristic',mode=6,rho=0),dict(key='MLP',mode=0,rho=0)]
grid=[dict(key=f'{name}_{rho:g}',mode=mode,rho=rho) for mode,name in [(2,'Reuse'),(3,'Guard')] for rho in pol['rho_grid']]
def execute(stage,split,configs,reps):
 prefix=R/stage;cfg=R/f'{stage}.configs';rng=np.random.default_rng(20261023);ordered=[]
 for rep in range(reps):
  ix=rng.permutation(len(configs)) if split=='validation' else np.roll(np.arange(len(configs)),-rep)
  ordered.extend((configs[i],rep) for i in ix)
 cfg.write_text(''.join(f"{c['key']} {c['mode']} {c['rho']:.17g} {rep}\n" for c,rep in ordered))
 args=[str(R/'search'),str(S/'index'),str(R/'history.fbin'),str(model),str(tau),str(D/f'{split}.fbin'),str(D/f'{split}.gt'),str(cfg),str(prefix),str(R/'history.bin')]
 done=R/f'{stage}.done.json';signature=dict(args=args,config=sha(cfg),binary=sha(R/'search'))
 if done.exists():assert json.loads(done.read_text())==signature
 else:
  dump(R/f'{stage}.command.json',signature);log(f'{stage} START ({len(ordered)} configurations)')
  with (R/f'{stage}.log').open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
  dump(done,signature);log(f'{stage} DONE')
 with (D/f'{split}.gt').open('rb') as f:
  n,k=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4',count=int(n*k)).reshape(int(n),int(k))[:,:10]
 rows=list(csv.DictReader((R/f'{stage}.csv').open()));assert len(rows)==len(ordered)*int(n)
 summaries=[];runs=[];per={}
 for c in configs:
  ref=None;arr_ref=None
  for rep in range(reps):
   v=[r for r in rows if r['key']==c['key'] and int(r['rep'])==rep];assert [int(r['query']) for r in v]==list(range(int(n)))
   a={k:np.array([float(r[k]) for r in v]) for k in ['hits','reads','expansions','skips','model_calls','reused','total_us','lookup_us','cache_hit']}
   ip=Path(f'{prefix}_{c["key"]}_r{rep}.ids');out=np.fromfile(ip,'<u8').reshape(int(n),10);hits=np.array([len(set(x)&set(y)) for x,y in zip(out,gt)]);np.testing.assert_array_equal(hits,a['hits'])
   if ref is not None:
    np.testing.assert_array_equal(out,ref)
    for k in ['hits','reads','expansions','skips','model_calls','reused','cache_hit']:np.testing.assert_array_equal(a[k],arr_ref[k])
   else:ref=out;arr_ref=a;per[c['key']]=a
   meta=json.loads(Path(f'{prefix}_{c["key"]}_r{rep}.meta.json').read_text());assert meta['reader_requests']==int(a['reads'].sum());assert meta['reader_bytes']==meta['reader_requests']*4096;assert meta['linux_read_bytes']==-1 or meta['reader_bytes']==meta['linux_read_bytes']
   runs.append(dict(key=c['key'],rep=rep,recall_pct=float(hits.mean()*10),p95_ms=float(np.percentile(a['total_us'],95)/1000),qps=meta['qps'],mean_us=float(a['total_us'].mean()),reads=float(a['reads'].mean()),requested_KiB=float(meta['reader_bytes']/int(n)/1024),linux_read_bytes_available=meta['linux_read_bytes']>=0,model_calls=float(a['model_calls'].mean()),reused=float(a['reused'].mean()),lookup_us=float(a['lookup_us'].mean()),cache_hit=float(a['cache_hit'].mean()),ids_sha256=sha(ip)))
  rr=[r for r in runs if r['key']==c['key']];s={**c}
  for k in rr[0]:
   if k not in ['key','rep','ids_sha256']:s[k]=float(np.median([r[k] for r in rr]))
  for k in ['qps','p95_ms']:s[k+'_min']=min(r[k] for r in rr);s[k+'_max']=max(r[k] for r in rr)
  summaries.append(s)
 dump(R/f'{stage}.summary.json',summaries);dump(R/f'{stage}.runs.json',runs)
 return {s['key']:s for s in summaries},per
val,_=execute('validation','validation',base+grid,3)
floor=max(val['Heuristic']['recall_pct'],val['MLP']['recall_pct']-.05)
feasible=[val[c['key']] for c in grid if val[c['key']]['recall_pct']>=floor-1e-9]
# Always expose both validation-selected objectives; neither is chosen on test.
qbest=max(feasible+[val['MLP']],key=lambda s:s['qps']);pbest=min(feasible+[val['MLP']],key=lambda s:s['p95_ms'])
selected={s['key']:next(c for c in base+grid if c['key']==s['key']) for s in [qbest,pbest]}
selection=dict(validation_recall_floor=floor,qps_selected=qbest['key'],p95_selected=pbest['key'],test_used=False)
if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())==selection
else:dump(R/'SELECTION_LOCK.json',selection)
log('SELECTION '+json.dumps(selection))
# Include a predeclared guarded control even if validation chooses fresh for both.
cs=base+[c for c in selected.values() if c['key']!='MLP']
if len(cs)==3:cs.append(next(c for c in grid if c['key']=='Guard_0.25'))
test,per=execute('test','test',cs,9)
for key,old in [('Vanilla','Vanilla'),('Heuristic','Heuristic'),('MLP','Ours')]:
 assert sha(R/f'test_{key}_r0.ids')==sha(P/f'runs/{old}_r0.ids')
 assert sha(R/'test.cache.bin')==sha(P/'runs/cache_nodes.bin')
for path,h in lock.items():assert sha(path)==h
boot={};rng=np.random.default_rng(42);ix=rng.integers(0,1000,(3000,1000))
for c in cs:
 key=c['key'];a=per[key];b=per['Heuristic'];boot[key]=dict(recall_delta_vs_heuristic_pp_95CI=np.percentile((a['hits']-b['hits'])[ix].mean(1)*10,[2.5,97.5]).tolist(),reads_saved_vs_heuristic_95CI=np.percentile((b['reads']-a['reads'])[ix].mean(1),[2.5,97.5]).tolist())
dump(R/'bootstrap.json',boot)
lines=['# P0-A：增加真实在线 Skip 复用','','固定原始 P0-A 模型、阈值、索引、查询、L100/Beam4/四线程/10000节点缓存。历史仅来自互斥训练查询。所有方法本轮同一程序重新计时，在线查找、成员检查、推理和I/O均计入。','',f'验证集 Recall 下限 {floor:.3f}%；QPS 选中 `{qbest["key"]}`，P95 选中 `{pbest["key"]}`。测试集不参与选择。','', '|方法|Recall %|P95 ms|QPS|请求/q|直接读 KiB/q|模型调用/q|复用/q|查找 µs/q|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for c in cs:
 s=test[c['key']];lines.append(f"|{s['key']}|{s['recall_pct']:.3f}|{s['p95_ms']:.3f}|{s['qps']:.1f}|{s['reads']:.3f}|{s['requested_KiB']:.3f}|{s['model_calls']:.3f}|{s['reused']:.3f}|{s['lookup_us']:.2f}|")
lines+=['','9轮中位数及范围：','']
for s in test.values():lines.append(f"- {s['key']}: P95 {s['p95_ms_min']:.3f}–{s['p95_ms_max']:.3f} ms；QPS {s['qps_min']:.1f}–{s['qps_max']:.1f}。")
lines+=['','复用：在1000条历史查询中在线寻找欧氏距离最近者，以归一化平方距离 rho 过滤；历史同节点 Skip 才跳过模型及读盘。Guard 另要求相同候选rank、展开进度差不超过4。命中静态节点缓存始终保护。','', '所有重复均逐ID和离散计数一致；原三方法ID及BFS缓存与原P0-A完全一致；reader请求数与逐query一致，reader直接读字节按实际请求长度累计；当前环境无 /proc/self/io，无法核验块层 read_bytes。','', '自然查询历史重叠可能很低；查找成本可能抵消省模型收益。不能保证复用优于启发式。有限验证选参、既有测试集和主机噪声限制结论；重复范围重叠时不声称稳定加速。本次读取量为reader直接读请求字节，不声称已测得块层或SSD NAND流量。','', '原始文件：validation/test.csv、各配置.ids/.meta.json、test.runs.json、test.summary.json、bootstrap.json、INPUT_LOCK.json、SELECTION_LOCK.json。复现：`.venv/bin/python experiments/p0_a_skip_reuse/run.py`。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
with (R/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(next(iter(test.values()))));w.writeheader();w.writerows(test.values())
dump(R/'verification.json',dict(recall_recomputed=True,nine_repeats_ids_counts_identical=True,original_p0a_ids_cache_match=True,reader_requests_match=True,reader_bytes_equal_linux_read_bytes=all(s['linux_read_bytes_available'] for s in test.values()),inputs_unchanged=True,train_validation_test_disjoint=True,selection_before_test=True))
dump(R/'COMPLETE.json',dict(complete=True,methods=list(test),test_runs=len(cs)*9));log(json.dumps(test,indent=2))
