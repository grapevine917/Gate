from pathlib import Path
import os,json,hashlib,subprocess,csv,time
import numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];SOURCE=ROOT.parent/'io_predictability';BIN=LG/'DiskANN/build/apps/search_query_skip_cache'
RHOS=[0,.0001,.001,.01,.05,.1,.25]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def model_info(ds,name):
 if name=='LR':
  folder=SOURCE/ds/'state_lr';tau=json.loads((folder/'thresholds.json').read_text())['unsafe_fpr_0.005']['retain_threshold']
 else:
  folder=LG/'experiments'/('sift1m/feature_ablation/Top3' if ds=='sift' else 'deep1m/models/L100/pruning');tau=json.loads((folder/'thresholds.json').read_text())['fnr_0.5pct']['threshold']
 return folder/'model.bin',tau

def configs():
 out=[{'key':'fresh','mode':0,'rho':0},{'key':'lookup_only','mode':1,'rho':.25}]
 for mode,label in [(2,'reuse'),(3,'guard')]:
  for rho in RHOS:out.append({'key':f'{label}_r{rho:g}','mode':mode,'rho':rho})
 return out

def execute(ds,name,workload,split,cs,repeats,stage=None):
 folder=ROOT/ds/name/workload;folder.mkdir(parents=True,exist_ok=True);prefix=folder/(stage or split);config_path=Path(str(prefix)+'.configs');rng=np.random.default_rng(20261017);order=[]
 for rep in range(repeats):
  for ix in rng.permutation(len(cs)):
   c=cs[ix];order.append(f"{c['key']} {c['mode']} {c['rho']:.12g} {rep}")
 config_path.write_text('\n'.join(order)+'\n')
 inputs=SOURCE/ds/split if workload=='natural' else ROOT/ds/(('near_'+split) if workload=='near' else 'repeat')
 model,tau=model_info(ds,name);bank=ROOT/ds/name/'bank.bin'
 args=[str(BIN),str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(SOURCE/ds/'train/query.fbin'),str(model),str(tau),str(inputs/'query.fbin'),str(inputs/'gt.bin'),str(config_path),str(prefix),str(bank)]
 signature={'args':args,'binary_sha256':sha(BIN),'model_sha256':sha(model),'query_sha256':sha(inputs/'query.fbin'),'gt_sha256':sha(inputs/'gt.bin'),'config_sha256':sha(config_path),'bank_query_sha256':sha(SOURCE/ds/'train/query.fbin')}
 done=Path(str(prefix)+'.done.json')
 if done.exists():assert json.loads(done.read_text())==signature
 else:
  dump(Path(str(prefix)+'.command.json'),signature);print(ds,name,workload,split,stage or '', 'START',flush=True)
  with open(str(prefix)+'.log','w') as log:subprocess.run(args,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
  dump(done,signature);print(ds,name,workload,split,stage or '', 'DONE',flush=True)
 rows=list(csv.DictReader(open(str(prefix)+'.csv')));nq=1000 if workload=='natural' else 200 if workload=='near' else 50
 assert len(rows)==nq*len(cs)*repeats
 result={};perquery={}
 for c in cs:
  rr=[];base=[r for r in rows if r['key']=='fresh' and r['rep']=='0']
  # Audit files may contain only auditing policies; callers use the standard test fresh reference.
  for rep in range(repeats):
   v=sorted([r for r in rows if r['key']==c['key'] and int(r['rep'])==rep],key=lambda r:int(r['query']));assert [int(r['query']) for r in v]==list(range(nq))
   a={k:np.array([float(r[k]) for r in v]) for k in ['hits','reads','expansions','skips','model_calls','reused','membership_hits','conflicts','cache_hit','total_us','lookup_us','rho']}
   if rep==0:perquery[c['key']]=a
   else:
    for metric in ['hits','reads','expansions','skips','model_calls','reused','cache_hit']:np.testing.assert_array_equal(a[metric],perquery[c['key']][metric])
    for ext in ['ids','dists']:assert sha(f'{prefix}_{c["key"]}_r0.{ext}')==sha(f'{prefix}_{c["key"]}_r{rep}.{ext}')
   assert np.all(a['reads']<=a['expansions'])
   rr.append({'recall_pct':float(a['hits'].mean()*10),'reads':float(a['reads'].mean()),'model_calls':float(a['model_calls'].mean()),'reused':float(a['reused'].mean()),'membership_hits':float(a['membership_hits'].mean()),'conflicts':float(a['conflicts'].mean()),'cache_hit_fraction':float(a['cache_hit'].mean()),'latency_us':float(a['total_us'].mean()),'lookup_us':float(a['lookup_us'].mean()),'p95_us':float(np.percentile(a['total_us'],95)),'rho_median':float(np.median(a['rho']))})
  result[c['key']]={**c,**{k:float(np.median([r[k] for r in rr])) for k in rr[0]},'latency_min_us':min(r['latency_us'] for r in rr),'latency_max_us':max(r['latency_us'] for r in rr)}
 # Wall-clock throughput from native benchmark, distinct from sum of per-request latency.
 throughput={}
 for line in Path(str(prefix)+'.log').read_text().splitlines():
  if line.startswith('MEMO '):
   fields=line.split();throughput.setdefault(fields[1],[]).append(float(fields[-1]))
 for key in result:result[key]['qps']=float(np.median(throughput[key]))
 if 'fresh' in perquery:
  b=result['fresh'];ref=perquery['fresh']
  for key,r in result.items():
   r['recall_loss_pp']=b['recall_pct']-r['recall_pct'];r['net_request_saving']=1-r['reads']/b['reads'];r['model_call_saving']=1-r['model_calls']/b['model_calls'];r['latency_saving']=1-r['latency_us']/b['latency_us'];r['degraded_queries']=int((perquery[key]['hits']<ref['hits']).sum());r['improved_queries']=int((perquery[key]['hits']>ref['hits']).sum())
   if key in ['lookup_only','reuse_r0','guard_r0'] and (workload=='repeat' or r['reused']==0):
    for ext in ['ids','dists']:assert sha(f'{prefix}_{key}_r0.{ext}')==sha(f'{prefix}_fresh_r0.{ext}')
    np.testing.assert_array_equal(perquery[key]['reads'],ref['reads'])
 return result,perquery,prefix,json.loads(Path(str(prefix)+'.bank.json').read_text())

def report(results):
 lines=['# 相似query的学习式skip决定缓存','','缓存源是1000条历史query的真实模型skip记录。当前query寻找最近历史query，同一节点有skip标记且相似度达标才复用；未命中继续模型评分。Guard额外检查rank及展开进度。缓存近邻查找和成员查表均计入总延迟。模型版本、阈值、L、beam和BFS缓存固定；详见PROTOCOL.md。','','主策略在验证集按最低总平均延迟选择，约束相对Fresh Recall损失<=0.1个百分点，允许回退。所有测试阈值诊断公开，但不用于重选。','', '|数据集|模型|工作负载|验证选中|测试Recall损失pp|受损query|模型调用减少|复用候选/q|query缓存命中率|总延迟变化|lookup μs/q|净请求节省|','|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for ds,m in results.items():
  for name,workloads in m.items():
   for w,v in workloads.items():
    if 'test' not in v:continue
    c=v['selected'];t=v['test'][c['key']];lines.append(f"|{ds}|{name}|{w}|{c['key']}|{t['recall_loss_pp']:.3f}|{t['degraded_queries']}|{t['model_call_saving']:.2%}|{t['reused']:.2f}|{t['cache_hit_fraction']:.2%}|{t['latency_saving']:.2%}|{t['lookup_us']:.2f}|{t['net_request_saving']:.2%}|")
 for ds,m in results.items():
  for name,workloads in m.items():
   for w,v in workloads.items():
    if 'test' not in v:continue
    b=v['bank'];lines+=['',f'## {ds} / {name} / {w}', '',f"历史缓存: {b['cached_queries']} queries，{b['query_node_references']}条query-node成员关系，{b['unique_nodes']}个唯一节点。节点池{b['node_pool_bytes']}B，引用{b['query_reference_bytes']}B，状态检查元数据{b['state_guard_bytes']}B，query向量{b['query_vector_bytes']}B，容器{b['container_bytes']}B，范数{b['norm_bytes']}B；每worker临时lookup数组{b['per_worker_scratch_bytes']}B。仅存未共享节点ID列表需要{b['plain_node_id_lists_bytes']}B，因此共享池本身不保证比直接存ID省内存。",'', '|配置|Recall损失pp|受损query|模型调用减少|query缓存命中|复用候选/q|总平均延迟μs|延迟变化|lookup μs|净请求节省|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in v['test'].values():lines.append(f"|{t['key']}|{t['recall_loss_pp']:.3f}|{t['degraded_queries']}|{t['model_call_saving']:.2%}|{t['cache_hit_fraction']:.2%}|{t['reused']:.2f}|{t['latency_us']:.2f}|{t['latency_saving']:.2%}|{t['lookup_us']:.2f}|{t['net_request_saving']:.2%}|")
 lines+=['','自然查询与人工相似查询的收益不能混为同一工作负载。近似重复查询重新计算了全部1M base的精确GT；精确重复控制另存control结果。成员关系、guard状态和query向量都有成本；仅存skip决定最多省掉遇到的skip部分推理，keep仍需计算。测试缓存不更新。提交读取请求不等同物理读盘，三轮小幅计时差异应结合范围判断。']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 started=time.time();results={};provenance={'binary_sha256':sha(BIN),'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'lookup_in_timing':True,'cache_generation_uses_GT':False,'threshold_grid':RHOS,'validation_repetitions':2,'test_repetitions':3,'workloads':json.loads((ROOT/'workloads.json').read_text())};dump(ROOT/'PROVENANCE.json',provenance)
 for ds in ['sift','deep']:
  results[ds]={}
  for name in ['LR','MLP']:
   results[ds][name]={}
   cs=configs();controls=[{'key':'fresh','mode':0,'rho':0},{'key':'lookup_only','mode':1,'rho':0},{'key':'reuse_r0','mode':2,'rho':0},{'key':'guard_r0','mode':3,'rho':0}]
   cr,_,_,bank=execute(ds,name,'repeat','test',controls,2);dump(ROOT/ds/name/'repeat_control.json',{'results':cr,'exact_result_and_read_parity':True,'bank':bank})
   for workload in ['natural','near']:
    val,_,_,bank=execute(ds,name,workload,'validation',cs,2)
    feasible=[r for r in val.values() if r['mode'] in [2,3] and r['recall_loss_pp']<=.1+1e-9 and r['latency_us']<val['fresh']['latency_us']]
    chosen=min(feasible,key=lambda r:(r['latency_us'],-r['recall_pct'],r['key'])) if feasible else val['fresh'];selected=next(c for c in cs if c['key']==chosen['key'])
    lock=ROOT/ds/name/workload/'SELECTION_LOCK.json';obj={'test_used_for_selection':False,'selected':selected,'validation':val}
    if lock.exists():assert json.loads(lock.read_text())==obj
    else:dump(lock,obj)
    test,_,_,_=execute(ds,name,workload,'test',cs,3)
    audits=[]
    for mode,label in [(4,'audit_reuse'),(5,'audit_guard')]:
     rho=.001 if workload=='near' else .25;audits.append({'key':label,'mode':mode,'rho':rho})
    ar,_,_,_=execute(ds,name,workload,'test',audits,1,stage='audit')
    # Instrumented audits must follow the same trajectory as the corresponding timed policy.
    timed_rho=.001 if workload=='near' else .25
    for audit,label in [('audit_reuse','reuse'),('audit_guard','guard')]:
     key=f'{label}_r{timed_rho:g}'
     for ext in ['ids','dists']:
      folder=ROOT/ds/name/workload;assert sha(folder/f'audit_{audit}_r0.{ext}')==sha(folder/f'test_{key}_r0.{ext}')
     assert ar[audit]['reads']==test[key]['reads'] and ar[audit]['reused']==test[key]['reused']
    results[ds][name][workload]={'validation':val,'selected':selected,'test':test,'bank':bank,'audit':ar};dump(ROOT/'results.json',results);report(results)
 dump(ROOT/'COMPLETE.json',{'complete':True,'elapsed_seconds':time.time()-started,'selection_locked_before_test':True,'repeat_exact_parity':True,'audit_trajectory_parity':True,'deterministic_repetitions':True,'sha256':{n:sha(ROOT/n) for n in ['results.json','REPORT.md','PROTOCOL.md','PROVENANCE.json']}})
if __name__=='__main__':main()
