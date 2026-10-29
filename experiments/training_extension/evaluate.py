"""Native end-to-end evaluation after training completes; validation locks test."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,subprocess,random,time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
R=Path(__file__).resolve().parent;M=R.parent/'main_comparison';A=R.parent/'et_gate_adaptation';C=R.parent/'p0_c';B=R.parent/'p0_b_lr';S=R.parent/'sift1m';SEED=20261024
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def savecsv(p,rows):
 if rows:
  with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert (R/'TRAIN_COMPLETE.json').exists()
P=dict(L=[30,75,100,160,300],risks=[0,.001,.0025,.005,.01,.02,.05,.1],joint_risks=[0,.02,.05],et_thresholds=[1.1,.999,.995,.99,.98,.95,.9,.8],targets=[95,98,99,99.5,99.7],objectives=['physical_bytes','p95_us','qps'],selection='Internal query holdout selects model/epoch/seed, external validation calibrates Gate thresholds and selects search configs. Lock before test; three independent test repeats. Joint ET extension is paired with frozen LR used to collect its training trajectories; new MLP Gate is evaluated separately, not paired with LR-trajectory ET.',baseline_L=json.loads((B/'PROTOCOL.json').read_text())['baseline_L'],limits=['Finite grid; different search budgets per family','Previously observed benchmark queries; exploratory','Threshold retain-FNR does not guarantee final Recall','Guest O_DIRECT/block reads, not NAND telemetry'])
if (R/'EVALUATION_PROTOCOL.json').exists():assert json.loads((R/'EVALUATION_PROTOCOL.json').read_text())==P
else:dump(R/'EVALUATION_PROTOCOL.json',P)
models={'LR':M/'models/lr.bin','MLP_old':M/'models/state.bin','MLP_new':R/'models/gate.bin','LR_tight':R/'models/lr_tight.bin'}
locks=[B/'search',C/'score',S/'index_disk.index',S/'index_pq_compressed.bin',S/'index_pq_pivots.bin',*models.values(),M/'models/et.bin',R/'models/et_unpruned.bin',A/'models/adapted.bin',R/'models/et_adapted.bin']+[M/'data'/f'{s}.{ext}' for s in ['validation','test'] for ext in ['fbin','gt']]
lock={str(p):sha(p) for p in locks}
if (R/'EVALUATION_INPUT_LOCK.json').exists():assert json.loads((R/'EVALUATION_INPUT_LOCK.json').read_text())==lock
else:dump(R/'EVALUATION_INPUT_LOCK.json',lock)
prediction=R/'prediction';prediction.mkdir(exist_ok=True);metrics=[];thresholds={}
for split in ['validation','test']:
 for name,path in models.items():
  out=prediction/f'{split}_{name}.scores';bench=prediction/f'{split}_{name}.bench.json'
  if not out.exists():subprocess.run([str(C/'score'),str(path),str(R.parents[1]/'data/sift/sift_base.fbin'),str(C/'prediction'/f'{split}.bin'),str(out),str(bench)],check=True)
  score=np.fromfile(out,'<f4');raw=np.memmap(C/'prediction'/f'{split}.bin',dtype='<f4',mode='r',shape=(len(score),147));y=np.asarray(raw[:,146]);assert np.isfinite(score).all()
  metrics.append(dict(split=split,model=name,samples=len(score),retain_auc=float(roc_auc_score(y,score)),safe_skip_ap=float(average_precision_score(1-y,1-score)),logloss=float(log_loss(y,np.clip(score,1e-7,1-1e-7))),native_ns=float(np.median(json.loads(bench.read_text())['ns_per_score']))))
  if split=='validation':
   states=np.load(S/'training_data_l100/validation/pre_io_state.npy',mmap_mode='r');labels=np.load(S/'training_data_l100/validation/y.npy');full=np.ones(len(labels),'f4');full[states[:,10]==0]=score;positive=np.sort(full[labels==1]);points=[]
   for risk in P['risks']:
    allowed=int(np.floor(risk*len(positive)));tau=0. if risk==0 else float(positive[allowed]);false=int(np.sum((labels==1)&(full<tau)));assert false<=allowed
    points.append(dict(risk=risk,tau=tau,false_skips=false,retain_count=len(positive)))
   thresholds[name]=points
dump(R/'prediction_metrics.json',metrics);dump(R/'threshold_calibration.json',thresholds)
configs=[]
def add(family,gate,et,kind,L,risk=0,et_tau=1.1):
 tau=next(x['tau'] for x in thresholds[gate] if x['risk']==risk);key=f'{family}_L{L}_g{P["risks"].index(risk)}_e{P["et_thresholds"].index(et_tau)}'
 configs.append(dict(key=key,family=family,gate_model=str(models[gate]),et_model=str(et),kind=kind,L=L,risk=risk,tau=tau,et_tau=et_tau))
for L in P['baseline_L']:add('DiskANN','LR',M/'models/et.bin',0,L)
for L in P['L']:
 for model in ['LR','MLP_old','MLP_new']:
  for risk in P['risks']:add('Gate_'+model,model,M/'models/et.bin',4,L,risk)
 for label,et in [('old',M/'models/et.bin'),('new',R/'models/et_unpruned.bin')]:
  for tau in P['et_thresholds']:add('ET_'+label,'LR',et,6,L,et_tau=tau)
 for label,et in [('old',A/'models/adapted.bin'),('new',R/'models/et_adapted.bin')]:
  for risk in P['joint_risks']:
   for tau in P['et_thresholds']:add('Joint_LR_'+label,'LR',et,7,L,risk,tau)
dump(R/'configs.json',configs)
def complete(prefix):
 try:return prefix.with_suffix('.ids').stat().st_size==80000 and json.loads(prefix.with_suffix('.meta.json').read_text())['queries']==1000
 except (OSError,ValueError):return False
def evaluate(split,cs,rep):
 out=R/'runs'/split;out.mkdir(parents=True,exist_ok=True);todo=[c for c in cs if not complete(out/f'{c["key"]}_r{rep}')];rng=random.Random(SEED+rep+(split=='test')*100);rng.shuffle(todo);pairs=sorted(set((c['gate_model'],c['et_model']) for c in todo));batches=[]
 for gate,et in pairs:
  group=[c for c in todo if (c['gate_model'],c['et_model'])==(gate,et)];batches += [group[j:j+8] for j in range(0,len(group),8)]
 rng.shuffle(batches)
 for i,batch in enumerate(batches):
  cp=R/f'{split}_r{rep}_b{i}.configs';cp.write_text(''.join(f'{c["key"]} {c["kind"]} {c["L"]} {c["tau"]:.17g} {c["et_tau"]:.17g} 0 0\n' for c in batch));first=batch[0]
  with (R/f'{split}_r{rep}_b{i}.log').open('w') as f:subprocess.run([str(B/'search'),str(S/'index'),str(M/'data'/f'{split}.fbin'),str(M/'data'/f'{split}.gt'),first['gate_model'],first['et_model'],str(cp),str(out),str(rep)],stdout=f,stderr=subprocess.STDOUT,check=True)
  log(f'{split} repeat={rep} batch={i+1}/{len(batches)}')
 with (M/'data'/f'{split}.gt').open('rb') as f:n,d=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4',count=int(n*d)).reshape(n,d)[:,:10]
 rows=[]
 for c in cs:
  prefix=out/f'{c["key"]}_r{rep}';a=np.genfromtxt(prefix.with_suffix('.csv'),delimiter=',',names=True);meta=json.loads(prefix.with_suffix('.meta.json').read_text());ids=np.fromfile(prefix.with_suffix('.ids'),'<u8').reshape(1000,10);hits=np.array([len(set(i)&set(g)) for i,g in zip(ids,gt)]);np.testing.assert_array_equal(hits,a['hits'])
  assert meta['reader_requests']==int(a['reads'].sum()) and meta['reader_bytes']==meta['reader_requests']*4096 and meta['linux_read_bytes']==meta['reader_bytes']
  if c['kind'] in [4,7]:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
  rows.append(c|dict(repeat=rep,total_hits=int(hits.sum()),recall=float(hits.sum()/100),physical_bytes=meta['linux_read_bytes']/1000,p95_us=float(np.percentile(a['latency_us'],95)),qps=meta['qps'],reads=float(a['reads'].mean()),skips=float(a['skips'].mean()),et_stop_fraction=float(a['et_stopped'].mean()),ids_sha256=sha(prefix.with_suffix('.ids'))))
 return rows
if (R/'validation.json').exists():validation=json.loads((R/'validation.json').read_text())
else:validation=evaluate('validation',configs,0);dump(R/'validation.json',validation)
parity=[];baseline={r['L']:r for r in validation if r['family']=='DiskANN'}
for r in validation:
 if r['risk']==0 and r['et_tau']==1.1:
  for k in ['ids_sha256','total_hits','reads','physical_bytes']:assert r[k]==baseline[r['L']][k]
  assert r['skips']==0 and r['et_stop_fraction']==0;parity.append(r['key'])
chosen=[]
for family in sorted(set(c['family'] for c in configs)):
 for target in P['targets']:
  candidates=[v for v in validation if v['family']==family and v['recall']>=target]
  for metric in P['objectives']:
   if candidates:
    c=min(candidates,key=lambda r:(-r[metric] if metric=='qps' else r[metric],-r['total_hits'],r['key']));chosen.append(dict(family=family,target=target,objective=metric,key=c['key'],validation_recall=c['recall']))
if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())==chosen
else:dump(R/'SELECTION_LOCK.json',chosen)
keys={s['key'] for s in chosen};testconfigs=[c for c in configs if c['key'] in keys];dump(R/'test_configs.json',testconfigs);runs=[]
for rep in range(3):runs+=evaluate('test',testconfigs,rep);dump(R/'all_test_runs.json',runs)
summary=[]
for c in testconfigs:
 rs=[r for r in runs if r['key']==c['key']]
 for k in ['total_hits','physical_bytes','reads','skips','et_stop_fraction','ids_sha256']:assert len({r[k] for r in rs})==1
 row=rs[0].copy();row.pop('repeat')
 for k in ['p95_us','qps']:row[k]=float(np.median([r[k] for r in rs]));row[k+'_min']=min(r[k] for r in rs);row[k+'_max']=max(r[k] for r in rs)
 summary.append(row)
dump(R/'test_summary.json',summary);savecsv(R/'test_summary.csv',summary);by={r['key']:r for r in summary};table=[]
for c in chosen:table.append(c|{k:by[c['key']][k] for k in ['L','tau','et_tau','recall','physical_bytes','p95_us','qps','skips','et_stop_fraction']}|dict(test_feasible=by[c['key']]['recall']>=c['target']))
savecsv(R/'validation_locked_test.csv',table)
text=['# 延长训练对照实验','','## 训练是否充分','','固定原有数据和17–128–64–1结构，每个任务3个种子、160轮，40轮后逐步降低学习率。只用训练内部200条查询选种子与轮次，再用1200条训练查询重新拟合。','','|模型|原选定轮次|新选定轮次|新内部损失|同种子前40轮最佳损失|损失降低|','|---|---:|---:|---:|---:|---:|']
old_epochs={'gate':23,'et_unpruned':38,'et_adapted':20}
for t in json.loads((R/'TRAIN_COMPLETE.json').read_text())['models']:
 s=t['selected'];text.append(f'|{t["task"]}|{old_epochs[t["task"]]}|{s["best_epoch"]}|{s["holdout_loss"]:.7f}|{s["best_first40"]["holdout_loss"]:.7f}|{s["relative_holdout_improvement_over_first40_pct"]:.2f}%|')
text+=['','不同任务损失定义不同：Gate为BCE，ET为Recall MSE，不可横向比较。延长训练同时改变学习率，收益不能纯归因于增加轮次。种子与轮次选择带来的不确定性不包含在单个最佳损失中。','','## 独立候选预测诊断','','|划分|模型|Retain AUROC|Safe-skip AP|Logloss|原生推理 ns|','|---|---|---:|---:|---:|---:|']
for r in metrics:text.append(f'|{r["split"]}|{r["model"]}|{r["retain_auc"]:.5f}|{r["safe_skip_ap"]:.5f}|{r["logloss"]:.6f}|{r["native_ns"]:.1f}|')
for target in [99.5,99.7]:
 text+=['',f'## 验证目标{target}%：最低读取量锁定配置','','实际测试Recall不同；未达标者不能算作满足质量要求的收益。每配置3次测试，时间取中位数。','','|方法|验证Recall|测试Recall|达标|KiB/Query|P95 ms|QPS|','|---|---:|---:|---|---:|---:|---:|']
 for r in table:
  if r['target']==target and r['objective']=='physical_bytes':text.append(f'|{r["family"]}|{r["validation_recall"]:.2f}%|{r["recall"]:.2f}%|{r["test_feasible"]}|{r["physical_bytes"]/1024:.3f}|{r["p95_us"]/1000:.3f}|{r["qps"]:.1f}|')
text+=['','## 范围与复现','','- 原模型、原实验均保留；本轮独立计时，不能与历史绝对时间直接比较。','- LR紧容差检查见LR_CONVERGENCE.json；增加优化迭代不等于增加模型容量。','- 新MLP Gate是mixed-L贡献代理标签训练，与原P0-A的L100模型不同。','- ET_old为历史未剪枝ET；ET_new为本次同数据延长训练ET。Joint_LR_old为上一轮剪枝轨迹适配ET；Joint_LR_new为其延长训练版。','- 新MLP Gate没有与LR轨迹训练的ET拼接，避免引入新的轨迹不匹配。','- 标签、查询数量与特征未增加；较低预测误差不保证更好的最终Recall/成本交换。','- 所有目标及P95/QPS独立选参结果见validation_locked_test.csv；重复范围见test_summary.csv。','- 训练与验证/测试按查询隔离；当前基准查询曾被观察，属于探索性实验。','- 包源未提供Matplotlib，保留完整训练曲线JSON与结果CSV。','- 复现：bash experiments/training_extension/run.sh。']
(R/'REPORT.md').write_text('\n'.join(text)+'\n');assert all(sha(p)==h for p,h in lock.items());dump(R/'verification.json',dict(passed=True,disabled_action_parity=parity,original_inputs_unchanged=True,recall_recomputed_from_ids=True,physical_bytes_verified=True,repeated_discrete_results_equal=True))
dump(R/'COMPLETE.json',dict(complete=True,validation_configs=len(configs),test_configs=len(testconfigs),test_runs=len(runs),report='REPORT.md'));log('EVALUATION COMPLETE')
