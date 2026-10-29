from pathlib import Path
import os,json,csv,time,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];SOURCE=ROOT.parent/'io_predictability';MEMO=ROOT.parent/'query_skip_cache';BIN=LG/'DiskANN/build/apps/search_policy_comparison'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(key,kind=0,**kw):return dict(key=key,kind=kind,model='-',tau=0,second='-',tau2=0,mark='-',rank=-1,progress=-1,bank='-',rho=0,guard=0,**kw)
def configurations(ds):
 lr=SOURCE/ds/'state_lr';top=LG/'experiments'/('sift1m/feature_ablation/Top3' if ds=='sift' else 'deep1m/models/L100/pruning');lt=json.loads((lr/'thresholds.json').read_text())['unsafe_fpr_0.005']['retain_threshold'];tt=json.loads((top/'thresholds.json').read_text())['fnr_0.5pct']['threshold']
 # Build dictionaries by updating defaults, so every native config has explicit fields.
 def make(key,kind=0,**kw):
  c=config(key,kind);c.update(kw);return c
 cs=[make('DiskANN'),make('LR',1,model=str(lr/'model.bin'),tau=lt),make('Top3_MLP',1,model=str(top/'model.bin'),tau=tt),make('Joint_LR_Top3',2,model=str(lr/'model.bin'),tau=lt,second=str(top/'model.bin'),tau2=tt)]
 for name,folder,tau,bankname in [('LR',lr,lt,'LR'),('Top3',top,tt,'MLP')]:
  for guard in [0,1]:cs.append(make(name+('_Guard' if guard else '_Reuse'),5,model=str(folder/'model.bin'),tau=tau,bank=str(MEMO/ds/bankname/'bank.bin'),rho=.01,guard=guard))
 for support,rate,label in [(1,1,'Mark_naive'),(5,.8,'Mark_conservative')]:cs.append(make(label,3,mark=str(ROOT.parent/'node_mark_comparison'/ds/f'Mark_m{support}_r{rate:g}.bin')))
 hist=ROOT.parent/'history_predictability';th=json.loads((hist/'results.json').read_text())[ds]['models']['full_44_lr']['thresholds']['coverage_0.25'];cs.append(make('History44_LR',1,model=str(hist/ds/'full_44_lr.bin'),tau=1-th))
 if ds=='sift':
  all17=LG/'experiments/sift1m/feature_ablation/All17';at=json.loads((all17/'thresholds.json').read_text())['fnr_0.5pct']['threshold'];cs.append(make('All17_MLP',1,model=str(all17/'model.bin'),tau=at))
  rules=json.loads((LG/'experiments/sift1m/top3_heuristic/selected_parameters.json').read_text())
  for method,key in [('Rank','Rank'),('Rank+Progress','Rank_Progress')]:
   r=next(x for x in rules if x['method']==method and x['target_fnr_pct']==.5);cs.append(make(key,4,rank=r['r'],progress=r['e']))
 return cs

def execute(ds,workload,cs):
 dst=ROOT/ds/workload;dst.mkdir(parents=True,exist_ok=True);prefix=dst/'test';rng=np.random.default_rng(20261017);lines=[]
 for rep in range(3):
  for ix in rng.permutation(len(cs)):
   c=cs[ix];lines.append(' '.join(str(c[k]) for k in ['key','kind','model','tau','second','tau2','mark','rank','progress','bank','rho','guard'])+' '+str(rep))
 cfg=dst/'configs.txt';cfg.write_text('\n'.join(lines)+'\n')
 inputs=SOURCE/ds/'test' if workload=='natural' else MEMO/ds/'near_test'
 args=[str(BIN),str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(SOURCE/ds/'train/query.fbin'),str(inputs/'query.fbin'),str(inputs/'gt.bin'),str(cfg),str(prefix)]
 signature={'args':args,'binary_sha256':sha(BIN),'query_sha256':sha(inputs/'query.fbin'),'gt_sha256':sha(inputs/'gt.bin'),'config_sha256':sha(cfg)}
 done=dst/'DONE.json'
 if done.exists():assert json.loads(done.read_text())==signature
 else:
  dump(dst/'command.json',signature);print(ds,workload,'START',len(cs),'policies x 3',flush=True)
  with open(dst/'native.log','w') as f:subprocess.run(args,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
  dump(done,signature);print(ds,workload,'DONE',flush=True)
 rows=list(csv.DictReader(open(str(prefix)+'.csv')));nq=1000 if workload=='natural' else 200;assert len(rows)==len(cs)*3*nq
 metrics={};arrays={};throughput={}
 for line in (dst/'native.log').read_text().splitlines():
  if line.startswith('POLICY '):f=line.split();throughput.setdefault(f[1],[]).append(float(f[-1]))
 for c in cs:
  key=c['key'];rep_metrics=[]
  for rep in range(3):
   rs=sorted([r for r in rows if r['key']==key and int(r['rep'])==rep],key=lambda r:int(r['query']));assert [int(r['query']) for r in rs]==list(range(nq))
   a={k:np.array([float(r[k]) for r in rs]) for k in ['hits','reads','expansions','evaluated','skips','calls','reused','cache_hit','total_us','lookup_us']}
   assert np.all(a['reads']<=a['expansions'])
   if key=='DiskANN':assert not a['evaluated'].any() and not a['calls'].any()
   else:np.testing.assert_array_equal(a['expansions'],a['evaluated']-a['skips'])
   if rep==0:arrays[key]=a
   else:
    for k in ['hits','reads','expansions','evaluated','skips','calls','reused','cache_hit']:np.testing.assert_array_equal(a[k],arrays[key][k])
    for ext in ['ids','dists']:assert sha(f'{prefix}_{key}_r0.{ext}')==sha(f'{prefix}_{key}_r{rep}.{ext}')
   rep_metrics.append({'recall_pct':float(a['hits'].mean()*10),'reads':float(a['reads'].mean()),'model_calls':float(a['calls'].mean()),'reused':float(a['reused'].mean()),'cache_hit_fraction':float(a['cache_hit'].mean()),'expansions':float(a['expansions'].mean()),'skips':float(a['skips'].mean()),'latency_us':float(a['total_us'].mean()),'p95_us':float(np.percentile(a['total_us'],95)),'lookup_us':float(a['lookup_us'].mean())})
  metrics[key]={**c,**{m:float(np.median([r[m] for r in rep_metrics])) for m in rep_metrics[0]},'qps':float(np.median(throughput[key])),'latency_min_us':min(r['latency_us'] for r in rep_metrics),'latency_max_us':max(r['latency_us'] for r in rep_metrics)}
 base=metrics['DiskANN'];bh=arrays['DiskANN']['hits']
 for key,r in metrics.items():
  r.update(recall_loss_vs_diskann_pp=base['recall_pct']-r['recall_pct'],request_saving_vs_diskann=1-r['reads']/base['reads'],latency_saving_vs_diskann=1-r['latency_us']/base['latency_us'],degraded_queries_vs_diskann=int((arrays[key]['hits']<bh).sum()))
 # Cross-experiment output consistency; timings are deliberately not imported.
 verified=[]
 for model,oldmodel,policies in [('LR','LR',[('LR','fresh'),('LR_Reuse','reuse_r0.01'),('LR_Guard','guard_r0.01')]),('Top3','MLP',[('Top3_MLP','fresh'),('Top3_Reuse','reuse_r0.01'),('Top3_Guard','guard_r0.01')])]:
  old=MEMO/ds/oldmodel/workload/'test';oldrows=list(csv.DictReader(open(str(old)+'.csv')))
  for newkey,oldkey in policies:
   for ext in ['ids','dists']:assert sha(f'{prefix}_{newkey}_r0.{ext}')==sha(f'{old}_{oldkey}_r0.{ext}')
   oldread=np.array([int(r['reads']) for r in oldrows if r['key']==oldkey and r['rep']=='0']);np.testing.assert_array_equal(arrays[newkey]['reads'],oldread);verified.append(newkey)
 if workload=='natural':
  old=ROOT.parent/'node_mark_comparison'/ds/'test/DiskANN_r0_100_idx_uint32.bin'
  with open(old,'rb') as f:n,d=np.fromfile(f,'<u4',2);oldids=np.fromfile(f,'<u4').reshape(int(n),int(d))
  newids=np.fromfile(str(prefix)+'_DiskANN_r0.ids','<u8').reshape(nq,10);np.testing.assert_array_equal(oldids,newids);verified.append('DiskANN')
 deltas={}
 for model,off,on,guard in [('LR','LR','LR_Reuse','LR_Guard'),('Top3','Top3_MLP','Top3_Reuse','Top3_Guard')]:
  for key in [on,guard]:
   a,b=metrics[key],metrics[off];deltas[key]={'reference':off,'recall_loss_pp':b['recall_pct']-a['recall_pct'],'request_saving':1-a['reads']/b['reads'],'model_call_saving':1-a['model_calls']/b['model_calls'],'latency_saving':1-a['latency_us']/b['latency_us']}
 return {'metrics':metrics,'reuse_increment':deltas,'verification':{'deterministic_three_repetitions':True,'old_outputs_match':verified,'new_timing_only':True},'queries':nq}

def report(results):
 lines=['# 原始DiskANN、既有基线与query复用统一对照','','所有数字来自本轮同程序交错重测，三轮中位数。固定L=100、beam=4、K=10、4线程、BFS cache=10000；每个workload使用同一批query与GT。模型/阈值/标记表均冻结，有复用统一rho<=0.01且不自动回退。Top3_MLP是之前简称的MLP；All17_MLP是另一个全17维MLP。','','自然独立查询1000条；高相似查询200条（1%/5%扰动混合，GT独立重算）。不同workload不可混在一张质量曲线中。不同方法的实际召回不同，这不是matched-recall调参实验。完整参数来源见CONFIG_LOCK.json、方法见PROTOCOL.md。']
 for ds,workloads in results.items():
  for w,v in workloads.items():
   lines+=['',f'## {ds} / {w}','','|方法|Recall@10 %|相对DiskANN损失pp|请求/q|相对DiskANN请求减少|模型调用/q|平均延迟ms|相对DiskANN延迟降低|p95 ms|QPS|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
   for key,r in v['metrics'].items():lines.append(f"|{key}|{r['recall_pct']:.3f}|{r['recall_loss_vs_diskann_pp']:.3f}|{r['reads']:.3f}|{r['request_saving_vs_diskann']:.2%}|{r['model_calls']:.3f}|{r['latency_us']/1000:.4f}|{r['latency_saving_vs_diskann']:.2%}|{r['p95_us']/1000:.4f}|{r['qps']:.1f}|")
   lines+=['','复用的增量收益：以下只相对同模型关闭复用，不以DiskANN为分母。','','|开启策略|无复用参照|额外Recall损失pp|请求减少|模型调用减少|延迟降低|','|---|---|---:|---:|---:|---:|']
   for key,d in v['reuse_increment'].items():lines.append(f"|{key}|{d['reference']}|{d['recall_loss_pp']:.3f}|{d['request_saving']:.2%}|{d['model_call_saving']:.2%}|{d['latency_saving']:.2%}|")
 lines+=['','Rank/Rank_Progress/All17_MLP仅有SIFT既有训练产物，因此DEEP未提供这些基线。History44_LR是此前300条训练query的模型，不与其他模型假设相同训练规模。Global Mark_naive/Mark_conservative均只使用训练查询记录。Joint两次模型评分分别计数；cached候选在全部方法中保留。','','零实际复用配置的小幅延迟变化不能视为复用收益；三轮时间范围在results.json。测试划分此前已观察过；本轮未用测试调参。读取为提交SSD请求，不等于物理盘读取。']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 started=time.time();allcs={ds:configurations(ds) for ds in ['sift','deep']};artifacts={}
 for ds,cs in allcs.items():
  for c in cs:
   for key in ['model','second','mark','bank']:
    if c[key]!='-':artifacts[c[key]]=sha(c[key])
 lock={'configs':allcs,'selection':'frozen previous validation parameters; rho=.01 fixed; no fallback; no test tuning','artifacts_sha256':artifacts,'binary_sha256':sha(BIN),'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'PROTOCOL.md')}
 if (ROOT/'CONFIG_LOCK.json').exists():assert json.loads((ROOT/'CONFIG_LOCK.json').read_text())==lock
 else:dump(ROOT/'CONFIG_LOCK.json',lock)
 results={}
 for ds in ['sift','deep']:
  results[ds]={}
  for w in ['natural','near']:
   results[ds][w]=execute(ds,w,allcs[ds]);dump(ROOT/'results.json',results);report(results)
 sources=[LG/'DiskANN/apps/search_policy_comparison.cpp',LG/'DiskANN/src/pq_flash_index.cpp',LG/'DiskANN/include/online_gate.h']
 dump(ROOT/'COMPLETE.json',{'complete':True,'elapsed_seconds':time.time()-started,'fixed_configs_before_test':True,'all_cross_experiment_output_checks_passed':True,'new_timing_only':True,'sha256':{str(p):sha(p) for p in [ROOT/'REPORT.md',ROOT/'results.json',ROOT/'CONFIG_LOCK.json',*sources]}})
if __name__=='__main__':main()
