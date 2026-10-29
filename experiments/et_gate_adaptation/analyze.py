"""Additional paired-query uncertainty and readable plots after experiment.py."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def savecsv(p,rows):
 if rows:
  with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
p=json.loads((R/'PROTOCOL.json').read_text());selected=json.loads((R/'SELECTION_LOCK.json').read_text());summary={r['key']:r for r in json.loads((R/'test_summary.json').read_text())}
validation=json.loads((R/'validation.json').read_text())
# Paired query bootstrap is descriptive conditional on validation-selected policies.
# Timing intervals are deliberately not bootstrapped from per-query parallel times.
rng=np.random.default_rng(p['seed']);weights=rng.multinomial(1000,np.full(1000,.001),size=5000)
by={(s['family'],s['target'],s['objective']):s for s in selected};comparisons=[]
for target in p['targets']:
 for metric in p['objectives']:
  joint=by.get(('Joint_adapted',target,metric))
  if not joint:continue
  for family in ['DiskANN','Gate','ET_original','ET_unpruned','Joint_original','Joint_unpruned']:
   reference=by.get((family,target,metric))
   if not reference:continue
   a=np.genfromtxt(R/'runs/test'/f'{joint["key"]}_r0.csv',delimiter=',',names=True);b=np.genfromtxt(R/'runs/test'/f'{reference["key"]}_r0.csv',delimiter=',',names=True)
   hitdelta=a['hits']-b['hits'];readsave=b['reads']-a['reads'];recall_ci=np.percentile(weights@hitdelta/100,[2.5,97.5]);read_ci=np.percentile(weights@readsave/1000,[2.5,97.5]);ja=summary[joint['key']];rb=summary[reference['key']]
   comparisons.append(dict(target=target,objective=metric,reference=family,joint_key=joint['key'],reference_key=reference['key'],joint_recall=ja['recall'],reference_recall=rb['recall'],both_test_feasible=ja['recall']>=target and rb['recall']>=target,recall_delta_pp=ja['recall']-rb['recall'],recall_delta_ci_low=float(recall_ci[0]),recall_delta_ci_high=float(recall_ci[1]),reads_saved_per_query=float(readsave.mean()),reads_saved_ci_low=float(read_ci[0]),reads_saved_ci_high=float(read_ci[1]),io_reduction_pct=100*(1-ja['physical_bytes']/rb['physical_bytes']),p95_reduction_pct=100*(1-ja['p95_us']/rb['p95_us']),qps_increase_pct=100*(ja['qps']/rb['qps']-1)))
savecsv(R/'paired_comparisons.csv',comparisons)
# Disabled-action equivalence is checked on discrete results, not noisy timing.
parity=[];base={r['L']:r for r in validation if r['family']=='DiskANN'}
for row in validation:
 if row['tau']==0 and row['et_tau']==1.1:
  for k in ['ids_sha256','total_hits','reads','physical_bytes','expansions']:assert row[k]==base[row['L']][k],(row['key'],k)
  assert row['skips']==0 and row['et_stop_fraction']==0;parity.append(row['key'])
effects=[]
for pair in json.loads((R/'ablation_pairs.json').read_text()):
 a=np.genfromtxt(R/'runs/test'/f'{pair["parent"]}_r0.csv',delimiter=',',names=True);b=np.genfromtxt(R/'runs/test'/f'{pair["control"]}_r0.csv',delimiter=',',names=True)
 delta=b['reads']-a['reads'];ci=np.percentile(weights@delta/1000,[2.5,97.5])
 effects.append(pair|dict(reads_saved_per_query=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),both_actions_queries=int(np.sum((a['skips']>0)&(a['et_stopped']>0))),queries=1000))
savecsv(R/'ablation_paired_queries.csv',effects)
dump(R/'verification.json',dict(disabled_action_parity=parity,passed=True,bootstrap_replicates=5000,bootstrap_scope='paired query sampling conditional on validation-selected frozen policies; no timing CI or selection-adjusted inference'))
import importlib.util
plots_available=importlib.util.find_spec('matplotlib') is not None
if plots_available:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figdir=R/'figures';figdir.mkdir(exist_ok=True)
 families=['DiskANN','Gate','ET_original','ET_unpruned','Joint_original','Joint_unpruned','Joint_adapted'];colors=dict(zip(families,['black','#999999','#d08020','#a05020','#b040a0','#6040a0','#007f70']))
 for metric,label,scale in [('physical_bytes','Measured read KiB / query',1/1024),('p95_us','P95 latency (ms)',.001),('qps','QPS',1)]:
  fig,axes=plt.subplots(1,2,figsize=(13,5))
  for family in families:
   rs=[r for r in validation if r['family']==family];best=[];cost=float('inf')
   for r in sorted(rs,key=lambda r:(-r['total_hits'],-r[metric] if metric=='qps' else r[metric])):
    c=-r[metric] if metric=='qps' else r[metric]
    if c<cost:best.append(r);cost=c
   best.sort(key=lambda r:r['recall']);axes[0].plot([r['recall'] for r in best],[r[metric]*scale for r in best],'.-',label=family,color=colors[family],linewidth=1)
   keys={s['key'] for s in selected if s['family']==family and s['objective']==metric};rs=[summary[k] for k in keys]
   axes[1].scatter([r['recall'] for r in rs],[r[metric]*scale for r in rs],label=family,color=colors[family],s=28)
  axes[0].set_title('Validation grid empirical frontiers');axes[1].set_title('Test: validation-locked configurations (3 repeats)')
  for ax in axes:ax.set_xlabel('Actual Recall@10 (%)');ax.set_ylabel(label);ax.grid(alpha=.2)
  axes[0].set_xlim(94.5,100);axes[1].set_xlim(94.5,100);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/f'recall_{metric}.svg');plt.close(fig)
report=R/'REPORT.md'
addition=['','## 查询配对分析与图表','','- [Recall–读取量](figures/recall_physical_bytes.svg)','- [Recall–P95](figures/recall_p95_us.svg)','- [Recall–QPS](figures/recall_qps.svg)','- paired_comparisons.csv：重训联合组相对各组的逐质量目标、逐成本目标比较，以及配对查询bootstrap区间。','- ablation_paired_queries.csv：关闭实验的读取节省区间和同一查询内两个动作同时发生的数量。','- 区间仅反映已锁定策略的查询抽样，不包含训练、选参或计时的不确定性。']
if not plots_available:
 addition=[line for line in addition if not line.startswith('- [Recall')]
 addition.append('- 当前环境无Matplotlib且包源未提供该依赖，保留完整CSV数据；安装后重跑analyze.py可生成标准图表。')
content=report.read_text();marker='\n## 查询配对分析与图表\n';content=content.split(marker)[0];report.write_text(content+'\n'.join(addition)+'\n')
completion=json.loads((R/'COMPLETE.json').read_text());completion.update(analysis_complete=True,verification='verification.json',figures_generated=plots_available)
if not plots_available:completion['plotting_dependency']='Matplotlib unavailable from configured package sources'
dump(R/'COMPLETE.json',completion)
manifest={}
for path in R.rglob('*'):
 if path.is_file() and '__pycache__' not in str(path) and path.name not in ['artifact_manifest.json','analysis.log','progress.log']:
  with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  manifest[str(path.relative_to(R))]=dict(bytes=path.stat().st_size,sha256=digest)
dump(R/'artifact_manifest.json',manifest)
print('ANALYSIS COMPLETE')
