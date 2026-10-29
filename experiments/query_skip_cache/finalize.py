from pathlib import Path
import json,csv,hashlib,struct,subprocess,os
import numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];SOURCE=ROOT.parent/'io_predictability'
def matrix(path,dt):
 with open(path,'rb') as f:n,d=np.fromfile(f,'<u4',2);return np.fromfile(f,dt,count=int(n)*int(d)).reshape(int(n),int(d))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
results=json.loads((ROOT/'results.json').read_text());checks={};detail={}
for ds,models in results.items():
 checks[ds]={};detail[ds]={};bank_query=matrix(SOURCE/ds/'train/query.fbin','<f4').astype(float)
 for name,workloads in models.items():
  with open(ROOT/ds/name/'bank.bin','rb') as f:
   assert f.read(8)==b'QSMEMO01';n,d,u=struct.unpack('<III',f.read(12));assert n==1000 and d==bank_query.shape[1];pool=np.fromfile(f,'<u4',u);assert len(np.unique(pool))==u and (np.diff(pool.astype(int))>0).all();edges=0
   for _ in range(n):
    count=struct.unpack('<I',f.read(4))[0];r=np.fromfile(f,'<u4',count*3).reshape(count,3);edges+=count;assert np.all(r[:,0]<u) and (np.diff(r[:,0].astype(int))>0).all()
   assert f.read()==b''
  control=json.loads((ROOT/ds/name/'repeat_control.json').read_text())
  for key in ['reuse_r0','guard_r0']:
   r=control['results'][key];b=control['results']['fresh'];assert r['recall_pct']==b['recall_pct'] and r['reads']==b['reads'];assert abs(b['model_calls']-r['model_calls']-r['reused'])<1e-8
  checks[ds][name]={'shared_references_valid':True,'edges':edges,'unique_nodes':u,'exact_repeat_call_accounting':True};detail[ds][name]={}
  for workload,v in workloads.items():
   folder=ROOT/ds/name/workload;lock=json.loads((folder/'SELECTION_LOCK.json').read_text());assert lock['test_used_for_selection'] is False and lock['selected']==v['selected'] and lock['validation']==v['validation']
   assert (folder/'SELECTION_LOCK.json').stat().st_mtime_ns<=(folder/'test.command.json').stat().st_mtime_ns
   feasible=[r for r in v['validation'].values() if r['mode'] in [2,3] and r['recall_loss_pp']<=.1+1e-9 and r['latency_us']<v['validation']['fresh']['latency_us']]
   best=min(feasible,key=lambda r:(r['latency_us'],-r['recall_pct'],r['key'])) if feasible else v['validation']['fresh'];assert best['key']==v['selected']['key']
   rows=list(csv.DictReader(open(folder/'test.csv')))
   inputs=SOURCE/ds/'test' if workload=='natural' else ROOT/ds/'near_test';queries=matrix(inputs/'query.fbin','<f4').astype(float)
   # Independent float64 all-donor check for fixed sample, rather than native float32 dot implementation.
   lookup=sorted([r for r in rows if r['key']=='lookup_only' and r['rep']=='0'],key=lambda r:int(r['query']))
   for i in [0,17,99,len(queries)-1]:
    q=queries[i];dist=((bank_query-q)**2).sum(1);donor=int(lookup[i]['donor']);assert np.isclose(dist[donor],dist.min(),rtol=1e-5,atol=1e-7)
    rho=dist[donor]/max((q*q).sum(),(bank_query[donor]**2).sum(),1e-12);assert np.isclose(rho,float(lookup[i]['rho']),rtol=1e-6,atol=1e-10)
   checks[ds][name][workload]={'validation_only_selection':True,'selection_predates_test':True,'independent_nearest_query_checks':True}
   # Fixed high-similarity operating point: rho=.001 admits 1% perturbations, rejects 5%.
   if workload=='near':
    entries={}
    for key in ['reuse_r0.001','guard_r0.001']:
     entries[key]={}
     for level,lo,hi in [('1pct',0,100),('5pct',100,200)]:
      scores=[]
      for rep in range(3):
       get=lambda k:sorted([r for r in rows if r['key']==k and int(r['rep'])==rep and lo<=int(r['query'])<hi],key=lambda r:int(r['query']))
       a=get(key);b=get('fresh');mean=lambda rr,field:np.mean([float(r[field]) for r in rr]);bh=np.array([int(r['hits']) for r in b]);ah=np.array([int(r['hits']) for r in a]);lat=mean(a,'total_us');lookupus=mean(a,'lookup_us');blat=mean(b,'total_us')
       scores.append({'recall_loss_pp':float((bh.mean()-ah.mean())*10),'degraded_queries':int((ah<bh).sum()),'model_call_saving':float(1-mean(a,'model_calls')/mean(b,'model_calls')),'latency_saving':float(1-lat/blat),'latency_saving_excluding_query_lookup':float(1-(lat-lookupus)/blat),'lookup_us':float(lookupus),'reused_per_query':float(mean(a,'reused')),'query_hit_fraction':float(mean(a,'cache_hit'))})
      entries[key][level]={k:float(np.median([r[k] for r in scores])) for k in scores[0]}
    detail[ds][name]=entries
# Independently verify generated near-query GT on four fixed samples using direct subtraction.
for ds in results:
 p=LG/'data'/('sift/sift_base.fbin' if ds=='sift' else 'deep1M_base.fbin')
 with open(p,'rb') as f:n,d=np.fromfile(f,'<u4',2)
 base=np.memmap(p,mode='r',dtype='<f4',offset=8,shape=(int(n),int(d)));q=matrix(ROOT/ds/'near_test/query.fbin','<f4').astype(float);truth=matrix(ROOT/ds/'near_test/gt.bin','<u4')
 for i in [0,99,100,199]:
  bestd=np.full(10,np.inf);bestid=np.zeros(10,int)
  for start in range(0,len(base),32768):
   dist=((base[start:start+32768].astype(float)-q[i])**2).sum(1);ix=np.argpartition(dist,9)[:10];dv=np.concatenate([bestd,dist[ix]]);di=np.concatenate([bestid,ix+start]);take=np.argsort(dv)[:10];bestd=dv[take];bestid=di[take]
  assert set(bestid)==set(truth[i])
 checks[ds]['near_gt_independent_direct_distance']=True
(ROOT/'verification.json').write_text(json.dumps(checks,indent=2));(ROOT/'near_levels.json').write_text(json.dumps(detail,indent=2))
# CPU-only model scoring benchmark, after all end-to-end measurements have finished.
subprocess.run(['g++','-std=c++17','-O3','-Ofast','-mavx2','-DMKL_ILP64','-I',str(LG/'DiskANN/include'),'-I','/opt/intel/oneapi/mkl/latest/include',str(ROOT/'gate_cost.cpp'),'-o',str(ROOT/'gate_cost'),'-L/opt/intel/oneapi/mkl/latest/lib/intel64','-Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib/intel64','-Wl,--no-as-needed','-lmkl_intel_ilp64','-lmkl_intel_thread','-lmkl_core','-liomp5','-lpthread','-lm','-ldl','-latomic'],check=True)
costs={}
for ds,models in results.items():
 a=np.loadtxt(SOURCE/ds/'train/collected_features.csv',delimiter=',',skiprows=1,dtype=np.float32)[:,2:];a=a[a[:,10]==0];a=a[np.random.default_rng(42).choice(len(a),10000,replace=False)].astype('<f4');inp=ROOT/ds/'score_states.fbin'
 with open(inp,'wb') as f:np.array(a.shape,'<u4').tofile(f);a.tofile(f)
 costs[ds]={}
 for name in models:
  model=SOURCE/ds/'state_lr/model.bin' if name=='LR' else LG/'experiments'/('sift1m/feature_ablation/Top3/model.bin' if ds=='sift' else 'deep1m/models/L100/pruning/model.bin')
  output=subprocess.run([str(ROOT/'gate_cost'),str(model),str(inp)],check=True,capture_output=True,text=True,env={**os.environ,'MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});costs[ds][name]=json.loads(output.stdout)
(ROOT/'model_score_cost.json').write_text(json.dumps(costs,indent=2))
lines=['# 学习式skip的相似query缓存：结果摘要','','历史缓存来自1000条已完成query；新query仍默认执行原学习模型，只对足够相似的历史query、同一节点复用skip。测试不写入缓存；相似query的GT重新对1M base精确计算。内存使用唯一节点池和4字节共享引用；并未把图节点向量复制到每个query。','','## 验证选参后的主结果','','相对同一模型每次重新评分的Fresh；验证Recall损失预算0.1个百分点，测试超预算保留失败结果。延迟含query相似度查找及节点成员查表，正值为变快。','','|数据集|模型|workload|选中配置|额外Recall损失pp|模型调用减少|总延迟降低|复用候选/q|','|---|---|---|---|---:|---:|---:|---:|']
for ds,m in results.items():
 for name,ws in m.items():
  for w,v in ws.items():
   r=v['test'][v['selected']['key']];lines.append(f"|{ds}|{name}|{w}|{r['key']}|{r['recall_loss_pp']:.3f}|{r['model_call_saving']:.2%}|{r['latency_saving']:.2%}|{r['reused']:.2f}|")
lines+=['','## 固定高相似度诊断：1%相对L2扰动，rho阈值0.001','','此表是预先固定阈值的测试诊断，不是测试选参。Guard要求rank一致且已展开数差<=4。','','|数据集|模型|策略|额外Recall损失pp|模型调用减少|总延迟降低|lookup μs/q|复用候选/q|','|---|---|---|---:|---:|---:|---:|---:|']
for ds,m in detail.items():
 for name,policies in m.items():
  for key,levels in policies.items():
   r=levels['1pct'];lines.append(f"|{ds}|{name}|{key}|{r['recall_loss_pp']:.3f}|{r['model_call_saving']:.2%}|{r['latency_saving']:.2%}|{r['lookup_us']:.2f}|{r['reused_per_query']:.2f}|")
lines+=['','## 单次模型评分成本（独立CPU微基准）','','各模型对同一数据集的10000条训练miss状态评分，每轮200000次，共5轮；不包含状态提取、query路由或SSD读取，不替代上面的端到端结果。','','|数据集|模型|每次评分中位μs|最小–最大μs|','|---|---|---:|---|']
for ds,m in costs.items():
 for name,c in m.items():lines.append(f"|{ds}|{name}|{c['median_us_per_score']:.4f}|{c['min_us']:.4f}–{c['max_us']:.4f}|")
lines+=['','## 共享存储的真实开销','','|数据集|模型|唯一节点/成员关系数|池+引用字节|仅ID列表字节|含query向量及状态等的缓存字节|','|---|---|---:|---:|---:|']
for ds,m in results.items():
 for name,ws in m.items():
  b=ws['near']['bank'];total=sum(b[k] for k in ['node_pool_bytes','query_reference_bytes','state_guard_bytes','query_vector_bytes','norm_bytes','container_bytes']);lines.append(f"|{ds}|{name}|{b['unique_nodes']}/{b['query_node_references']}|{b['node_pool_bytes']+b['query_reference_bytes']}|{b['plain_node_id_lists_bytes']}|{total}|")
lines+=['','## 范围与解释','','- 查询相近不代表搜索状态相同；直接复用skip可能覆盖当前模型本来会保留的候选。审计冲突见results.json；它不等价于关键节点误剪次数。','- 缓存中只记录skip，keep仍需模型评分，因此省掉的是部分推理；精确重复的call计数与输出一致性已验证。','- 自然workload与人为高相似workload分开解释；精确重复仅作正确性控制，未与最终结果缓存性能比较。','- 零复用配置的微小延迟变化不能归因于复用收益。三轮计时的小幅差异仍受主机波动影响；模型调用减少不等于端到端同比例加速。','- 搜索阶段减去query lookup的分量另存near_levels.json，仅用于解释匹配开销，不能当作实际端到端加速。','- 共享节点身份可行，但每条query的成员关系不能合并掉。只缓存4字节ID时，增加共享池比直接存ID更大；若已有大块共享节点对象，共享它的收益属于不同存储基线。','- 本轮缓存1000条历史query，固定模型版本/阈值/L/beam/cache；未实现测试流写回、淘汰策略或近似query路由索引。','', '完整参数曲线、Recall、读取、p95、QPS与审计在REPORT.md和results.json；复现用run.sh。']
(ROOT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
complete=json.loads((ROOT/'COMPLETE.json').read_text());complete['independent_verification']=True
for p in [ROOT/'SUMMARY.md',ROOT/'verification.json',ROOT/'near_levels.json',ROOT/'finalize.py',ROOT/'prepare.py',ROOT/'run.sh',ROOT/'gate_cost.cpp',ROOT/'model_score_cost.json',LG/'DiskANN/apps/search_query_skip_cache.cpp',LG/'DiskANN/include/online_gate.h',LG/'DiskANN/src/pq_flash_index.cpp']:
 complete['sha256'][str(p)]=sha(p)
(ROOT/'COMPLETE.json').write_text(json.dumps(complete,indent=2));print((ROOT/'SUMMARY.md').read_text())
