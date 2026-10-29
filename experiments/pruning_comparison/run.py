"""Validation-only configuration selection, followed by locked held-out evaluation."""
from pathlib import Path
import os,json,csv,subprocess,hashlib,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parent; LG=ROOT.parents[1]; SOURCE=LG/'experiments/io_predictability'; BIN=LG/'DiskANN/build/apps/search_disk_index'
def dump(p,x):
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def matrix(p,dtype):
 with open(p,'rb') as f:
  n,d=np.fromfile(f,'<u4',2);return np.fromfile(f,dtype,count=int(n)*int(d)).reshape(int(n),int(d))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def models(ds):
 top=LG/'experiments'/('sift1m/feature_ablation/Top3' if ds=='sift' else 'deep1m/models/L100/pruning')
 return SOURCE/ds/'state_lr',top

def run(ds,split,c,repeat=0):
 folder=ROOT/ds/split;folder.mkdir(parents=True,exist_ok=True);p=folder/f'{c["key"]}_r{repeat}';done=Path(str(p)+'.json')
 if done.exists():return json.loads(done.read_text())
 env={k:v for k,v in os.environ.items() if not k.startswith('DISKANN_')}
 env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',DISKANN_QUERY_STATS=str(p)+'.csv')
 lr,top=models(ds)
 if c['method'] in ['LR','Joint']:env.update(DISKANN_M3_MODEL=str(lr/'model.bin'),DISKANN_M3_TAU=str(c['lr_tau']))
 if c['method']=='Top3':env.update(DISKANN_M3_MODEL=str(top/'model.bin'),DISKANN_M3_TAU=str(c['top_tau']))
 if c['method']=='Joint':env.update(DISKANN_JOINT_MODEL=str(top/'model.bin'),DISKANN_JOINT_TAU=str(c['top_tau']))
 args=[str(BIN),'--data_type','float','--dist_fn','l2','--index_path_prefix',str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),'--query_file',str(SOURCE/ds/split/'query.fbin'),'--gt_file',str(SOURCE/ds/split/'gt.bin'),'-K','10','-L',str(c['L']),'-W',str(c['beam']),'-T','4','--num_nodes_to_cache','10000','--result_path',str(p)]
 dump(Path(str(p)+'.command.json'),{'args':args,'env':{k:v for k,v in env.items() if k.startswith('DISKANN_') or k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}})
 with open(str(p)+'.log','w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 ids=matrix(str(p)+f'_{c["L"]}_idx_uint32.bin','<u4');gt=matrix(SOURCE/ds/split/'gt.bin','<u4')[:,:10]
 assert ids.shape==(len(gt),10)
 hits=np.array([len(set(a)&set(b)) for a,b in zip(ids,gt)])
 s=np.genfromtxt(str(p)+'.csv',delimiter=',',names=True);assert len(s)==len(gt)
 np.testing.assert_array_equal(s['query_row'],np.arange(len(gt)))
 assert np.all(s['ssd_reads']<=s['expansions'])
 if c['method']=='DiskANN':assert not s['gate_evaluated'].any()
 else:np.testing.assert_array_equal(s['expansions'],s['gate_evaluated']-s['gate_skipped'])
 summary=next(x for x in Path(str(p)+'.log').read_text().splitlines() if x.startswith('ONLINE_SUMMARY ')).split()
 recall=float(hits.mean()*10);assert abs(recall-float(summary[5]))<0.0001
 row={**c,'split':split,'repeat':repeat,'recall_pct':recall,'reads':float(s['ssd_reads'].mean()),'expansions':float(s['expansions'].mean()),'latency_us':float(s['latency_us'].mean()),'p95_us':float(np.percentile(s['latency_us'],95)),'qps':float(summary[2]),'skips':float(s['gate_skipped'].mean()),'prefix':str(p)}
 dump(done,row);print(ds,split,c['key'],repeat,'recall',round(recall,3),'reads',round(row['reads'],3),flush=True)
 return row

def configs(ds):
 lr,top=models(ds)
 lt=[x['retain_threshold'] for x in json.loads((lr/'thresholds.json').read_text()).values()]
 tt=[x['threshold'] for k,x in json.loads((top/'thresholds.json').read_text()).items() if k in ['fnr_0.1pct','fnr_0.5pct','fnr_1pct']]
 assert len(lt)==len(tt)==3
 result=[]
 for L in [50,75,100,150,200]:
  for beam in [2,4,8]:
   for method,pairs in [('DiskANN',[(None,None)]),('LR',[(a,None) for a in lt]),('Top3',[(None,b) for b in tt]),('Joint',[(a,b) for a in lt for b in tt])]:
    for i,(a,b) in enumerate(pairs):result.append(dict(key=f'{method}_L{L}_B{beam}_t{i}',method=method,L=L,beam=beam,lr_tau=a,top_tau=b))
 return result

def main():
 started=time.time();all_results={}
 provenance={'binary_sha256':sha(BIN),'script_sha256':sha(Path(__file__)),'L':[50,75,100,150,200],'beam':[2,4,8],'recall_targets_pct':[99.0,99.5,99.7],'objectives':['reads','latency_us'],'validation_repeats':2,'test_repeats':3,'selection':'minimum median objective subject to absolute validation Recall@10 target; infeasible reported explicitly','joint':'skip iff LR score < LR tau OR Top3 score < Top3 tau; both scores evaluated on same live pre-I/O state','threads':4,'cache_nodes':10000,'K':10,'timing':'warm OS cache, randomized sequential configurations; model/index loading excluded; no audit during measurement','io':'submitted SSD requests, not physical device counters','models':'existing frozen models trained only on training query IDs; no retraining for L/beam; LR keeps cached candidates, Top3 retains its original behavior'}
 for ds in ['sift','deep']:
  sets=[set(np.load(SOURCE/ds/s/'query_ids.npy').tolist()) for s in ['train','validation','test']]
  assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
  lr,top=models(ds)
  provenance[ds]={'split_sizes':list(map(len,sets)),'models_sha256':{'LR':sha(lr/'model.bin'),'Top3':sha(top/'model.bin')},'query_ids_sha256':{s:sha(SOURCE/ds/s/'query_ids.npy') for s in ['train','validation','test']}}
 dump(ROOT/'PROVENANCE.json',provenance)
 for ds in ['sift','deep']:
  cs=configs(ds);dump(ROOT/f'{ds}_configs.json',cs)
  rng=np.random.default_rng(20261016);val=[]
  # Baseline warmup is outside reported repetitions.
  run(ds,'validation',{**cs[0],'key':'warmup'})
  for rep in range(2):
   for i in rng.permutation(len(cs)):val.append(run(ds,'validation',cs[i],rep))
  aggregates=[]
  for c in cs:
   rs=[r for r in val if r['key']==c['key']];assert len(rs)==2
   for metric in ['recall_pct','reads','expansions','skips']:assert rs[0][metric]==rs[1][metric]
   aggregates.append({**c,**{m:float(np.median([r[m] for r in rs])) for m in ['recall_pct','reads','expansions','latency_us','p95_us','qps','skips']}})
  dump(ROOT/f'{ds}_validation.json',aggregates)
  selected=[]
  for target in provenance['recall_targets_pct']:
   for objective in provenance['objectives']:
    for method in ['DiskANN','LR','Top3','Joint']:
     feasible=[r for r in aggregates if r['method']==method and r['recall_pct']>=target-1e-9]
     best=min(feasible,key=lambda r:(r[objective],-r['recall_pct'],r['key'])) if feasible else None
     selected.append({'target':target,'objective':objective,'method':method,'config':best})
  # Persist the entire decision before opening test ground truth or running test searches.
  dump(ROOT/f'{ds}_SELECTION_LOCK.json',{'selected':selected,'validation_sha256':sha(ROOT/f'{ds}_validation.json'),'test_used_for_selection':False})
  chosen={s['config']['key']:next(c for c in cs if c['key']==s['config']['key']) for s in selected if s['config']}
  # Joint with either tau=0 must reproduce the corresponding standalone policy.
  verify=[]
  for method in ['LR','Top3']:
   c=next(c for c in cs if c['method']==method and c['L']==100 and c['beam']==4)
   single=run(ds,'validation',c,0)
   joint={**c,'method':'Joint','key':'parity_'+method,'lr_tau':c['lr_tau'] or 0,'top_tau':c['top_tau'] or 0}
   other=run(ds,'validation',joint,0)
   for suffix in ['idx_uint32.bin','dists_float.bin']:
    assert Path(single['prefix']+'_100_'+suffix).read_bytes()==Path(other['prefix']+'_100_'+suffix).read_bytes()
   assert single['reads']==other['reads'];verify.append(method)
  dump(ROOT/f'{ds}_verification.json',{'zero_second_threshold_exact_parity':verify,'disjoint_splits':True,'validation_repetitions_deterministic':True})
  run(ds,'test',{**cs[0],'key':'warmup'})
  tests=[]
  for rep in range(3):
   for key in rng.permutation(list(chosen)):tests.append(run(ds,'test',chosen[key],rep))
  testagg={}
  for key in chosen:
   rows=[r for r in tests if r['key']==key];assert len(rows)==3
   for m in ['recall_pct','reads','expansions','skips']:assert len({r[m] for r in rows})==1
   for r in rows[1:]:
    for suffix in ['idx_uint32.bin','dists_float.bin']:
     assert Path(rows[0]['prefix']+f'_{r["L"]}_'+suffix).read_bytes()==Path(r['prefix']+f'_{r["L"]}_'+suffix).read_bytes()
   testagg[key]={**chosen[key],**{m:float(np.median([r[m] for r in rows])) for m in ['recall_pct','reads','expansions','latency_us','p95_us','qps','skips']},'latency_min_us':min(r['latency_us'] for r in rows),'latency_max_us':max(r['latency_us'] for r in rows)}
  all_results[ds]=[{**s,'test':testagg[s['config']['key']] if s['config'] else None} for s in selected]
  dump(ROOT/'results.json',all_results)
  report(all_results)
 dump(ROOT/'COMPLETE.json',{'complete':True,'elapsed_seconds':time.time()-started,'validation_configs_per_dataset':240,'selection_locked_before_test':True})

def report(results):
 lines=['# DiskANN / LR / Top3 / 联合剪枝对照实验','','训练、验证、测试 query ID 互不相交。现有模型冻结；所有 L、beam、阈值组合只在验证集扫描。分别按 SSD 请求数与平均延迟最小化选参，约束验证集绝对 Recall@10 ≥ 99%、99.5%、99.7%。未达到约束时标为无可行配置。测试结果不用于重选，未达测试召回目标的结果仍如实展示。','','每个数据集 240 组配置，验证两轮、冻结后测试三轮，随机顺序串行运行。延迟和 QPS 报告中位数；缓存预热，4 线程、10000 节点缓存、K=10。LR 为 state_lr；Top3 为 rank、completed_expansions、visited_count 三特征 MLP。Joint 为任一模型判定 skip 就剪枝；包含两个模型的真实推理耗时。SSD 指提交的读取请求，不能等同物理读盘次数。','']
 for ds,rows in results.items():
  lines+=['## '+ds,'','|验证目标 %|选参目标|策略|L / beam|LR τ / Top3 τ|验证 Recall %|测试 Recall %|测试 Reads/q|测试 Exp/q|测试延迟 μs|测试 QPS|','|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|']
  for r in rows:
   c=r['config'];t=r['test']
   if not c:lines.append(f'|{r["target"]}|{r["objective"]}|{r["method"]}|无可行配置|—|—|—|—|—|—|—|');continue
   tau=lambda v:'—' if v is None else f'{v:.5g}'
   lines.append(f'|{r["target"]}|{r["objective"]}|{r["method"]}|{c["L"]} / {c["beam"]}|{tau(c["lr_tau"])} / {tau(c["top_tau"])}|{c["recall_pct"]:.3f}|{t["recall_pct"]:.3f}|{t["reads"]:.3f}|{t["expansions"]:.3f}|{t["latency_us"]:.2f}|{t["qps"]:.1f}|')
  lines+=['','联合互补性（相同验证目标与选参目标，测试仅评价）：','']
  for target in [99.,99.5,99.7]:
   for obj in ['reads','latency_us']:
    group={r['method']:r['test'] for r in rows if r['target']==target and r['objective']==obj}
    if all(group.get(m) for m in ['LR','Top3','Joint']):
     best=min(group[m][obj] for m in ['LR','Top3']);j=group['Joint'];gain=(1-j[obj]/best)*100
     lines.append(f'- 验证目标 {target:g}%，按 {obj} 选参：联合相对测试上较优单模型的 {obj} 改善 {gain:.2f}%；测试召回 LR / Top3 / Joint 为 {group["LR"]["recall_pct"]:.3f}% / {group["Top3"]["recall_pct"]:.3f}% / {j["recall_pct"]:.3f}%。召回不同，不能仅据成本降低断言互补。')
 lines+=['','完整配置与逐查询日志在各数据集目录；验证全表、SELECTION_LOCK、results.json 和 PROVENANCE.json 保存选参依据、模型及二进制哈希。计时受当前主机和 OS 缓存影响。测试集沿用既有保留划分，此前实验曾报告过它，属于与本次选参独立的测试集，并非从未观察过的新测试集。','', '复现：`cd learned-gate && .venv/bin/python experiments/pruning_comparison/run.py`。已完成命令可恢复；修改二进制或协议应使用新输出目录。']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
