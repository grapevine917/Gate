"""Training saturation and paired-query comparisons of locked search policies."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def savecsv(p,rows):
 if rows:
  with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
training=json.loads((R/'TRAIN_COMPLETE.json').read_text());stages=[];curves=[]
for task in training['models']:
 for trial in task['trials']:
  name=task['task'];seed=trial['seed'];history=json.loads((R/'models'/f'{name}_s{seed}_inner.json').read_text())
  curves += [dict(task=name,seed=seed,**r) for r in history]
  losses=[min(r['holdout_loss'] for r in history[:limit]) for limit in [40,80,120,160]]
  stages.append(dict(task=name,seed=seed,best_epoch=trial['best_epoch'],best_through_40=losses[0],best_through_80=losses[1],best_through_120=losses[2],best_through_160=losses[3],last40_extra_improvement_pct=100*(1-losses[3]/losses[2]),total_improvement_pct=100*(1-losses[3]/losses[0])))
savecsv(R/'training_curves.csv',curves);savecsv(R/'training_saturation.csv',stages)
selected=json.loads((R/'SELECTION_LOCK.json').read_text());summary={r['key']:r for r in json.loads((R/'test_summary.json').read_text())};by={(r['family'],r['target'],r['objective']):r for r in selected};rng=np.random.default_rng(20261024);weights=rng.multinomial(1000,np.full(1000,.001),size=5000);rows=[]
for newer,older in [('Gate_MLP_new','Gate_MLP_old'),('ET_new','ET_old'),('Joint_LR_new','Joint_LR_old')]:
 for target in [95,98,99,99.5,99.7]:
  for metric in ['physical_bytes','p95_us','qps']:
   n=by.get((newer,target,metric));o=by.get((older,target,metric))
   if not(n and o):continue
   a=summary[n['key']];b=summary[o['key']];qa=np.genfromtxt(R/'runs/test'/f'{n["key"]}_r0.csv',delimiter=',',names=True);qb=np.genfromtxt(R/'runs/test'/f'{o["key"]}_r0.csv',delimiter=',',names=True)
   rec_ci=np.percentile(weights@(qa['hits']-qb['hits'])/100,[2.5,97.5]);read_ci=np.percentile(weights@(qb['reads']-qa['reads'])/1000,[2.5,97.5])
   rows.append(dict(newer=newer,older=older,target=target,objective=metric,new_key=n['key'],old_key=o['key'],new_recall=a['recall'],old_recall=b['recall'],both_test_feasible=a['recall']>=target and b['recall']>=target,recall_delta_pp=a['recall']-b['recall'],recall_delta_ci_low=float(rec_ci[0]),recall_delta_ci_high=float(rec_ci[1]),reads_saved=float(np.mean(qb['reads']-qa['reads'])),reads_saved_ci_low=float(read_ci[0]),reads_saved_ci_high=float(read_ci[1]),io_reduction_pct=100*(1-a['physical_bytes']/b['physical_bytes']),p95_reduction_pct=100*(1-a['p95_us']/b['p95_us']),qps_increase_pct=100*(a['qps']/b['qps']-1)))
savecsv(R/'paired_comparisons.csv',rows)
text=['','## 延长训练后的搜索收益','','以下均为各自验证锁定配置的测试比较。实际Recall不同；若任一方未达到目标，不能作为满足该质量要求的成本优势。','','|目标|比较（新/旧）|实际Recall（新/旧）|双方达标|读取量降低|P95降低|QPS提升|','|---|---|---|---|---:|---:|---:|']
for r in rows:
 if r['target'] in [99.5,99.7] and r['objective']=='physical_bytes':text.append(f'|{r["target"]}%|{r["newer"]}/{r["older"]}|{r["new_recall"]:.2f}% / {r["old_recall"]:.2f}%|{r["both_test_feasible"]}|{r["io_reduction_pct"]:+.2f}%|{r["p95_reduction_pct"]:+.2f}%|{r["qps_increase_pct"]:+.2f}%|')
text+=['','训练各阶段的最佳内部损失见training_saturation.csv，全部曲线见training_curves.csv。逐目标/指标的配对查询bootstrap见paired_comparisons.csv；区间只描述冻结策略的查询抽样，不包含模型/阈值选择或计时的不确定性。']
p=R/'REPORT.md';content=p.read_text().split('\n## 延长训练后的搜索收益\n')[0];p.write_text(content+'\n'.join(text)+'\n')
complete=json.loads((R/'COMPLETE.json').read_text());complete.update(analysis_complete=True,bootstrap_replicates=5000);dump(R/'COMPLETE.json',complete)
manifest={}
for p in R.rglob('*'):
 if p.is_file() and '__pycache__' not in str(p) and p.name not in ['artifact_manifest.json','train.log','evaluation.log','analysis.log']:
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  manifest[str(p.relative_to(R))]=dict(bytes=p.stat().st_size,sha256=h)
dump(R/'artifact_manifest.json',manifest)
print('ANALYSIS COMPLETE')
