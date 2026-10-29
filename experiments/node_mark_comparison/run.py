"""Frozen node memory vs state LR. All choices are locked before held-out runs."""
from pathlib import Path
import os,json,csv,hashlib,time,subprocess,struct
import numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];SOURCE=ROOT.parent/'io_predictability';BIN=LG/'DiskANN/build/apps/search_disk_index'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(p,dt):
 with open(p,'rb') as f:n,d=np.fromfile(f,'<u4',2);return np.fromfile(f,dt).reshape(int(n),int(d))
def marks(path,mask):
 with open(path,'wb') as f:f.write(b'NDMARK01');f.write(struct.pack('<I',len(mask)));f.write(np.packbits(mask,bitorder='little').tobytes())
def prepare(ds):
 start=time.perf_counter();dst=ROOT/ds;dst.mkdir(exist_ok=True)
 rows=np.genfromtxt(SOURCE/ds/'train/collected_candidates.csv',delimiter=',',names=True)
 miss=rows['cache_miss']!=0;nodes=rows['node'][miss].astype(int);low=(rows['admitted'][miss]==0)&(rows['in_result'][miss]==0)
 n=np.bincount(nodes,minlength=1000000).astype(np.uint32);z=np.bincount(nodes,weights=low,minlength=1000000).astype(np.uint32)
 assert len(np.unique(rows['query']))==1000 and np.all(z<=n)
 assert len(set(zip(rows['query'],rows['node'])))==len(rows)
 np.savez_compressed(dst/'training_counts.npz',visits=n,low=z)
 cs=[{'key':'DiskANN','method':'DiskANN','marks':None,'tau':None}]
 thresholds=json.loads((SOURCE/ds/'state_lr/thresholds.json').read_text())
 for key,t in thresholds.items():cs.append({'key':'LR_'+key,'method':'LR','marks':None,'tau':t['retain_threshold']})
 rules=[]
 for support in [1,2,5,10]:
  for rate in [.8,.95,1.]:
   mask=(n>=support)&(z>=rate*n);key=f'Mark_m{support}_r{rate:g}';path=dst/(key+'.bin');marks(path,mask)
   c={'key':key,'method':'Mark','marks':str(path),'tau':None,'min_support':support,'low_rate':rate,'marked_nodes':int(mask.sum()),'payload_bytes':125000,'file_bytes':path.stat().st_size,'marks_sha256':sha(path)}
   cs.append(c);rules.append(c)
 marks(dst/'empty.bin',np.zeros(1000000,bool))
 cached=np.zeros(1000000,bool);cached[rows['node'][~miss].astype(int)]=True;marks(dst/'cached_only.bin',cached)
 info={'training_events':len(rows),'miss_events':int(miss.sum()),'low_events':int(low.sum()),'distinct_miss_nodes':int((n>0).sum()),'training_query_count':1000,'training_source_sha256':sha(SOURCE/ds/'train/collected_candidates.csv'),'counter_array_bytes':n.nbytes+z.nbytes,'mark_build_seconds':time.perf_counter()-start,'LR_model_file_bytes':(SOURCE/ds/'state_lr/model.bin').stat().st_size,'rules':rules}
 dump(dst/'training_summary.json',info);dump(dst/'configs.json',cs)
 return cs,info

def run(ds,split,c,rep=0):
 dst=ROOT/ds/split;dst.mkdir(parents=True,exist_ok=True);prefix=dst/f'{c["key"]}_r{rep}';done=Path(str(prefix)+'.json')
 if done.exists():return json.loads(done.read_text())
 env={k:v for k,v in os.environ.items() if not k.startswith('DISKANN_')};env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DISKANN_QUERY_STATS=str(prefix)+'.csv')
 if c['method']=='LR':env.update(DISKANN_M3_MODEL=str(SOURCE/ds/'state_lr/model.bin'),DISKANN_M3_TAU=str(c['tau']))
 if c['method']=='Mark':env.update(DISKANN_NODE_MARKS=c['marks'])
 args=[str(BIN),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),'--query_file',str(SOURCE/ds/split/'query.fbin'),'--gt_file',str(SOURCE/ds/split/'gt.bin'),'-K','10','-L','100','-W','4','-T','4','--num_nodes_to_cache','10000','--result_path',str(prefix)]
 dump(Path(str(prefix)+'.command.json'),{'args':args,'env':{k:v for k,v in env.items() if k.startswith('DISKANN_') or k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}})
 with open(str(prefix)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 ids=matrix(str(prefix)+'_100_idx_uint32.bin','<u4');gt=matrix(SOURCE/ds/split/'gt.bin','<u4')[:,:10];assert ids.shape==(1000,10)
 hits=np.array([len(set(a)&set(b)) for a,b in zip(ids,gt)])
 stats=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True);assert len(stats)==1000;np.testing.assert_array_equal(stats['query_row'],np.arange(1000))
 assert np.all(stats['ssd_reads']<=stats['expansions'])
 if c['method']=='DiskANN':assert not stats['gate_evaluated'].any()
 else:np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped'])
 line=next(l for l in Path(str(prefix)+'.log').read_text().splitlines() if l.startswith('ONLINE_SUMMARY ')).split()
 recall=float(hits.mean()*10);assert abs(recall-float(line[5]))<.0001
 row={**c,'split':split,'repeat':rep,'recall_pct':recall,'reads':float(stats['ssd_reads'].mean()),'expansions':float(stats['expansions'].mean()),'skips':float(stats['gate_skipped'].mean()),'latency_us':float(stats['latency_us'].mean()),'p95_us':float(np.percentile(stats['latency_us'],95)),'qps':float(line[2]),'prefix':str(prefix)}
 dump(done,row);print(ds,split,c['key'],rep,'recall',round(recall,3),'reads',round(row['reads'],3),'us',round(row['latency_us'],1),flush=True)
 return row

def aggregate(rows,configs):
 agg={}
 for c in configs:
  rs=[r for r in rows if r['key']==c['key']];assert len(rs)>=2
  for metric in ['recall_pct','reads','expansions','skips']:assert len({r[metric] for r in rs})==1
  for r in rs[1:]:
   for suffix in ['idx_uint32.bin','dists_float.bin']:assert sha(rs[0]['prefix']+'_100_'+suffix)==sha(r['prefix']+'_100_'+suffix)
  agg[c['key']]={**c,**{m:float(np.median([r[m] for r in rs])) for m in ['recall_pct','reads','expansions','skips','latency_us','p95_us','qps']},'latency_min_us':min(r['latency_us'] for r in rs),'latency_max_us':max(r['latency_us'] for r in rs)}
 return agg

def selection(agg):
 base=agg['DiskANN'];selected=[]
 for budget in [0.,.1,.5]:
  for objective in ['reads','latency_us']:
   for method in ['LR','Mark']:
    feasible=[r for r in agg.values() if r['method']==method and base['recall_pct']-r['recall_pct']<=budget+1e-9 and r[objective]<base[objective]]
    best=min(feasible,key=lambda r:(r[objective],-r['recall_pct'],r['key'])) if feasible else base
    selected.append({'loss_budget_pp':budget,'objective':objective,'method':method,'chosen_key':best['key'],'fallback':not bool(feasible),'validation':best})
 return selected

def report(results):
 lines=['# 节点标记与学习式剪枝性能对照','','Mark不训练分类模型：只由训练查询的低贡献事件生成冻结节点位图；LR按当前17维搜索状态评分。两者仅跳过cache miss，固定L=100/beam=4/K=10/4线程/cache=10000。完整协议见PROTOCOL.md。验证两轮、测试三轮，报告中位数；阈值与回退均先锁定再测测试。','']
 for ds,v in results.items():
  base=v['test']['DiskANN'];t=v['training'];lines += ['## '+ds,'',f"无剪枝测试Recall@10={base['recall_pct']:.3f}%，reads/q={base['reads']:.3f}，平均延迟={base['latency_us']:.2f} μs。训练miss事件{t['miss_events']}，覆盖节点{t['distinct_miss_nodes']}。在线标记位图125000字节，LR模型文件{t['LR_model_file_bytes']}字节；计数数组离线占{t['counter_array_bytes']}字节，运行时不加载。",'','|验证损失预算pp|选参目标|方法|选中配置|验证损失pp|测试Recall%|测试损失pp|净请求节省|延迟变化|p95 μs|QPS|','|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
  for r in v['selected']:
   x=v['test'][r['chosen_key']];vl=v['validation']['DiskANN']['recall_pct']-r['validation']['recall_pct'];tl=base['recall_pct']-x['recall_pct']
   lines.append(f"|{r['loss_budget_pp']:g}|{r['objective']}|{r['method']}|{r['chosen_key']}|{vl:.3f}|{x['recall_pct']:.3f}|{tl:.3f}|{1-x['reads']/base['reads']:.2%}|{1-x['latency_us']/base['latency_us']:.2%}|{x['p95_us']:.1f}|{x['qps']:.1f}|")
  lines += ['','所有冻结配置的测试诊断（主结果仍为上表验证选中项；不能测试重选）：','','|配置|标记节点数|Recall%|损失pp|reads/q|平均延迟μs|延迟范围μs|','|---|---:|---:|---:|---:|---:|---|']
  for x in v['test'].values():lines.append(f"|{x['key']}|{x.get('marked_nodes','—')}|{x['recall_pct']:.3f}|{base['recall_pct']-x['recall_pct']:.3f}|{x['reads']:.3f}|{x['latency_us']:.2f}|{x['latency_min_us']:.2f}–{x['latency_max_us']:.2f}|")
 lines+=['','延迟变化为相对无剪枝的降低比例，负值表示变慢；读取是submitted SSD requests。零平均Recall损失不等于逐查询安全，详见results.json的受损查询统计。此次标记是跨查询全局ID表，未加入查询分区，也未在测试流上更新。计时受当前主机和OS缓存影响，小幅差异不能据三次运行认定稳定优劣。']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 started=time.time();results={};provenance={'binary_sha256':sha(BIN),'run_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'selection':'validation only','L':100,'beam':4,'K':10,'threads':4,'cache_nodes':10000,'validation_repeats':2,'test_repeats':3,'mark_training':'cache miss AND admitted=0 AND not in original query final top10; no singleton labels or GT used','memory':'packed exact bitset; one bit per node; cached candidates always retained'}
 for ds in ['sift','deep']:
  ids=[set(np.load(SOURCE/ds/s/'query_ids.npy').tolist()) for s in ['train','validation','test']];assert all(len(x)==1000 for x in ids);assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
  cs,training=prepare(ds);rng=np.random.default_rng(20261017)
  provenance[ds]={'splits':{s:sha(SOURCE/ds/s/'query_ids.npy') for s in ['train','validation','test']},'LR_model_sha256':sha(SOURCE/ds/'state_lr/model.bin'),'mark_training_sha256':training['training_source_sha256']};dump(ROOT/'PROVENANCE.json',provenance)
  run(ds,'validation',{**cs[0],'key':'warmup'})
  val=[]
  for rep in range(2):
   for ix in rng.permutation(len(cs)):val.append(run(ds,'validation',cs[ix],rep))
  va=aggregate(val,cs);dump(ROOT/ds/'validation.json',va);selected=selection(va)
  dump(ROOT/ds/'SELECTION_LOCK.json',{'test_used_for_selection':False,'selected':selected,'validation_sha256':sha(ROOT/ds/'validation.json')})
  checks={}
  for name in ['empty','cached_only']:
   c={'key':'control_'+name,'method':'Mark','marks':str(ROOT/ds/(name+'.bin')),'tau':None};r=run(ds,'validation',c)
   b=next(x for x in val if x['key']=='DiskANN' and x['repeat']==0)
   for suffix in ['idx_uint32.bin','dists_float.bin']:assert sha(r['prefix']+'_100_'+suffix)==sha(b['prefix']+'_100_'+suffix)
   assert r['reads']==b['reads'] and r['skips']==0;checks[name]={'exact_baseline_parity':True,'skips':0}
  dump(ROOT/ds/'verification.json',checks)
  run(ds,'test',{**cs[0],'key':'warmup'})
  tests=[]
  for rep in range(3):
   for ix in rng.permutation(len(cs)):tests.append(run(ds,'test',cs[ix],rep))
  ta=aggregate(tests,cs)
  # Whole-query quality damage is distinct from aggregate Recall loss.
  base_hits=None
  for c in cs:
   p=ROOT/ds/'test'/f'{c["key"]}_r0';a=matrix(str(p)+'_100_idx_uint32.bin','<u4');gt=matrix(SOURCE/ds/'test/gt.bin','<u4')[:,:10];hits=np.array([len(set(x)&set(g)) for x,g in zip(a,gt)])
   if c['method']=='DiskANN':base_hits=hits
   assert base_hits is not None
   ta[c['key']]['degraded_queries']=int((hits<base_hits).sum());ta[c['key']]['improved_queries']=int((hits>base_hits).sum())
  # Baseline-path encounter rate is diagnostic only, never feeds selection.
  features=np.genfromtxt(SOURCE/ds/'test/collected_features.csv',delimiter=',',names=True);miss=features['s10']==0;node=features['node'][miss].astype(int)
  for c in cs:
   if c['method']=='Mark':
    with open(c['marks'],'rb') as f:f.read(12);bits=np.unpackbits(np.frombuffer(f.read(),np.uint8),bitorder='little')
    ta[c['key']]['baseline_miss_mark_fraction']=float(bits[node].mean())
  results[ds]={'training':training,'validation':va,'selected':selected,'test':ta};dump(ROOT/'results.json',results);report(results)
 provenance['source_sha256']={str(p):sha(p) for p in [LG/'DiskANN/include/online_gate.h',LG/'DiskANN/src/pq_flash_index.cpp',LG/'DiskANN/apps/search_disk_index.cpp']};dump(ROOT/'PROVENANCE.json',provenance)
 dump(ROOT/'COMPLETE.json',{'complete':True,'elapsed_seconds':time.time()-started,'selection_locked_before_test':True,'empty_and_cached_marks_parity':True,'repeat_outputs_deterministic':True,'sha256':{p.name:sha(p) for p in [ROOT/'results.json',ROOT/'REPORT.md',ROOT/'PROVENANCE.json',ROOT/'PROTOCOL.md']}})
if __name__=='__main__':main()
