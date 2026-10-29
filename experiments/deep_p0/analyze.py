"""DEEP P0 reports, paired-query analysis and existing offline P0 plot utility."""
from pathlib import Path
import json,csv,hashlib,subprocess,sys
import numpy as np
R=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def savecsv(p,rows):
 if rows:
  with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pct(a,b):return 100*(1-a/b)
online=json.loads((R/'ONLINE_COMPLETE.json').read_text());verification=json.loads((R/'verification.json').read_text());assert online['complete'] and verification['passed']
metrics=json.loads((R/'prediction_metrics.json').read_text());train=json.loads((R/'TRAIN_COMPLETE.json').read_text());config=json.loads((R/'GRID_LOCK.json').read_text());rng=np.random.default_rng(20261030);weights=rng.multinomial(1000,np.full(1000,.001),size=5000)
label={'DiskANN':'Tuned DiskANN','Heuristic':'Heuristic','MLP_L100':'Ours (L100 MLP)','LR':'Ours (LR)','ET':'ET','ET_MLP_L100':'ET + MLP','ET_LR':'ET + LR','XGB':'XGBoost Gate','MLP_mixed':'State MLP (mixed-L)','QV':'q-v MLP (401D)'}
for name,version in [('p0_a','L100 MLP'),('p0_a_lr','mixed-L LR')]:
 root=R/name;rows=json.loads((root/'aggregate.json').read_text());base=next(r for r in rows if r['key']=='Vanilla');text=[f'# DEEP1M P0-A：Ours使用{version}','','固定原图R64/Lbuild100/PQ34、K10、L100、Beam4、10,000节点BFS缓存；同一1000条测试查询、4查询线程/MKL单线程。三方法各9次，三种循环顺序各重复3次，计时包括状态提取、推理、I/O和搜索；加载、128条热身、GT与输出不计。','','|方法|Recall@10|P95 ms|QPS|直接读请求/Query|实测KiB/Query|展开/Query|','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:text.append(f'|{r["key"]}|{r["recall"]:.2f}%|{r["p95_us"]/1000:.3f}|{r["qps"]:.1f}|{r["reads"]:.3f}|{r["physical_bytes"]/1024:.3f}|{r["expansions"]:.3f}|')
 text+=['','相对本轮Vanilla：','','|方法|Recall损失 pp|读取量降低|P95降低|QPS提升|','|---|---:|---:|---:|---:|'];effects=[];raw_base=np.genfromtxt(root/'runs/Vanilla_r0.csv',delimiter=',',names=True)
 for r in rows:
  if r['key']=='Vanilla':continue
  text.append(f'|{r["key"]}|{base["recall"]-r["recall"]:.2f}|{pct(r["physical_bytes"],base["physical_bytes"]):.2f}%|{pct(r["p95_us"],base["p95_us"]):.2f}%|{100*(r["qps"]/base["qps"]-1):.2f}%|')
  raw=np.genfromtxt(root/'runs'/f'{r["key"]}_r0.csv',delimiter=',',names=True);rec_ci=np.percentile(weights@(raw_base['hits']-raw['hits'])/100,[2.5,97.5]);read_ci=np.percentile(weights@(raw_base['reads']-raw['reads'])/1000,[2.5,97.5]);effects.append(dict(method=r['key'],recall_loss_pp=base['recall']-r['recall'],recall_loss_ci_low=float(rec_ci[0]),recall_loss_ci_high=float(rec_ci[1]),reads_saved=float(np.mean(raw_base['reads']-raw['reads'])),reads_saved_ci_low=float(read_ci[0]),reads_saved_ci_high=float(read_ci[1])))
 savecsv(root/'paired_query_bootstrap.csv',effects);text+=['','## 重复范围与判读','','Ours阈值仅用验证集L100候选轨迹按retain-FNR1%标定，缓存命中固定保留；该代理风险不保证最终Recall损失。Heuristic固定rank>72且completed_expansions>100。','']
 for r in rows:text.append(f'- {r["key"]}：P95范围{r["p95_us_min"]/1000:.3f}–{r["p95_us_max"]/1000:.3f}ms，QPS范围{r["qps_min"]:.1f}–{r["qps_max"]:.1f}。')
 text+=['','逐查询配对bootstrap5000次见paired_query_bootstrap.csv，仅反映冻结策略下的查询抽样，不含训练/选参不确定性，不构成计时显著性检验。','所有重复ID与离散计数一致；Recall从ID与GT复算；实际reader字节与Linux read_bytes一致。I/O为WSL2客户机块层实测，不是SSD内部NAND计数。','固定L的成本交换不等于同Recall最优配置优势；后者看P0-B。两个Gate版本独立计时，且训练来源不同，不是纯模型实现消融。','该DEEP查询集在历史实验中被观察，属于探索性实验。原始结果见all_runs.json、runs/和summary.csv。']
 (root/'REPORT.md').write_text('\n'.join(text)+'\n')
studies={'p0_b':['DiskANN','Heuristic','MLP_L100','ET','ET_MLP_L100'],'p0_b_lr':['DiskANN','Heuristic','LR','ET','ET_LR'],'p0_c':['DiskANN','LR','XGB','MLP_mixed','QV']}
for study,families in studies.items():
 root=R/study;rows=json.loads((root/'aggregate.json').read_text());frontiers=json.loads((root/'frontiers.json').read_text());frontiers['reads']=frontiers['physical_bytes'];dump(root/'frontiers.json',frontiers);held=readcsv(root/'validation_locked_test.csv');feasible=readcsv(root/'test_feasible_frontier_DESCRIPTIVE.csv');exact=readcsv(root/'strict_equal_recall.csv');text=[f'# DEEP1M {study.upper().replace("_","-")}','','同一DEEP1M原图R64/Lbuild100/PQ34、K10、Beam4、10,000节点缓存、4查询线程/MKL单线程；验证/测试各1000条互斥查询。模型仅用DEEP训练查询拟合。','', '## 实验范围','', '每方法64个预先锁定配置：Tuned DiskANN扫描64个L；其余8个L×8档参数。所有Gate按验证L100全候选的相同retain-FNR预算标定，缓存命中保护；不保证最终Recall。与SIFT原P0-C的训练分数量化阈值有所不同。','验证集分别按质量目标和P95/QPS/读取量选参，在测试前锁定。完整固定测试网格的前沿属于描述性结果，不能用测试挑出的点声称无偏部署收益。最终各指标前沿、精确同Recall赢家、质量要求下赢家和验证锁定点均重复3次，时间取中位数。','', '## 三项核心曲线','', '- [Recall–P95](figures/recall_p95.svg)','- [Recall–QPS](figures/recall_qps.svg)','- [Recall–实测块层读取](figures/recall_physical_bytes.svg)','', '标记是实测配置，线只引导阅读；不插值构造可部署点。三个指标分别取前沿，不将各自最优值拼成一行。']
 if study=='p0_c':
  text+=['','## 相同样本上的预测能力与推理成本','','四种Gate使用相同286550个mixed-L训练cache-miss样本和代理retain标签；结构为17维LR、深度5 XGBoost、17–128–64–1状态MLP，以及401–128–64–1 q–v MLP。q–v输入为[q96,v96,abs(q-v),q*v,state17]，额外完整向量表384MB。MLP的轮次在训练内部200条独立查询选择，再用全部1200条训练查询重拟合。','','|划分|模型|Retain AUROC|Safe-skip AP|Logloss|原生score μs/候选|模型KiB|','|---|---|---:|---:|---:|---:|---:|']
  for r in metrics:
   if r['model']=='l100':continue
   text.append(f'|{r["split"]}|{r["model"]}|{r["retain_auc"]:.5f}|{r["safe_skip_ap"]:.5f}|{r["logloss"]:.6f}|{r["native_ns"]/1000:.4f}|{r["model_bytes"]/1024:.3f}|')
  text+=['','原生score为单候选随机状态预热后5次扫描的均值中位数，不含队列状态抽取；真实搜索则包含全部开销。预测诊断使用同一L100 baseline轨迹，不能用分类分数代替最终搜索质量。q–v同时加入候选向量和查询交互，不能单独归因于查询信息。','', '## Recall要求下的最低P95配置（描述性测试网格）','','|Recall要求|方法|实际Recall|P95 ms|QPS|KiB/Query|L|','|---|---|---:|---:|---:|---:|---:|']
  for target in [90,95,98,99,99.5,99.7]:
   for r in feasible:
    if float(r['target'])==target and r['objective']=='p95_us':text.append(f'|{target}%|{label[r["family"]]}|{float(r["recall"]):.2f}%|{float(r["p95_us"])/1000:.3f}|{float(r["qps"]):.1f}|{float(r["physical_bytes"])/1024:.3f}|{r["L"]}|')
 else:
  text+=['','P0-B原MLP版本冻结单独L100训练权重；LR版本冻结mixed-L权重，分别与P0-A各自共用模型。两版本训练来源不同，纯实现消融见P0-C。ET是同一mixed-L17维前缀绝对Recall预测MLP，未声称忠实复现DARTH/LAET。','联合组为8对预声明阈值×8个L，不是完整笛卡尔积；存在关闭动作的配置，必须结合skips和ET停止率判断两种策略是否同时生效。','', '## 相同质量要求的最低读取量配置（描述性测试网格）']
  for target in [99.5,99.7]:
   text+=['',f'实际测试Recall≥{target}%；各行全部指标来自同一配置，实际Recall不同。','','|方法|实际Recall|P95 ms|QPS|KiB/Query|L|Gate τ|ET τ|跳过节点/Query|ET停止率|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
   for r in feasible:
    if float(r['target'])==target and r['objective']=='physical_bytes':text.append(f'|{label[r["family"]]}|{float(r["recall"]):.2f}%|{float(r["p95_us"])/1000:.3f}|{float(r["qps"]):.1f}|{float(r["physical_bytes"])/1024:.3f}|{r["L"]}|{float(r["tau"]):.6g}|{float(r["et_tau"]):.6g}|{float(r["skips"]):.3f}|{float(r["et_stop_fraction"]):.1%}|')
 text+=['','## 验证锁定配置的测试结果','','下表按验证集最低读取量选参；不重新用测试调参。未达标项不能当作满足质量要求的收益。','','|验证目标|方法|验证Recall|测试Recall|达标|P95 ms|QPS|KiB/Query|','|---|---|---:|---:|---|---:|---:|---:|']
 for r in held:
  if float(r['target']) in [99.5,99.7] and r['objective']=='physical_bytes' and r['key']:text.append(f'|{r["target"]}%|{label[r["family"]]}|{float(r["validation_recall"]):.2f}%|{float(r["test_recall"]):.2f}%|{r["test_meets_target"]}|{float(r["test_p95_us"])/1000:.3f}|{float(r["test_qps"]):.1f}|{float(r["test_physical_bytes"])/1024:.3f}|')
 text+=['','## 严格相同实际Recall','','使用相同整数total_hits匹配，不靠四舍五入。各指标分别选本方法该精确Recall下最佳配置；可能存在被更高Recall配置支配的点，应结合完整前沿判读。完整明细见strict_equal_recall.csv。']
 hits=sorted({int(r['total_hits']) for r in exact})
 for metric,title,factor in [('p95_us','P95 ms',.001),('qps','QPS',1),('physical_bytes','实测KiB/Query',1/1024)]:
  text+=['',f'### {title}','','|方法|'+'|'.join(f'{h/100:.2f}%' for h in hits)+'|','|---|'+'|'.join('---:' for _ in hits)+'|']
  for fam in families:
   values=[]
   for h in hits:
    r=next(r for r in exact if r['objective']==metric and r['family']==fam and int(r['total_hits'])==h);values.append(f'{float(r[metric])*factor:.3f}')
   text.append('|'+label[fam]+'|'+'|'.join(values)+'|')
 text+=['','## 验证与解释边界','','所有Recall由返回ID与对应DEEP GT复算；直接读请求和实际reader字节与Linux read_bytes核对；最终重复ID、读取、展开、skip和停止计数一致。P0-A baseline与本实验L100 baseline完整ID一致。','时间包含特征抽取、模型推理、I/O及搜索，排除加载、缓存构建、128条热身、GT与输出。QPS=1000/四线程正式搜索墙钟时间；P95为单轮1000查询P95的跨次中位数。重复范围见all_configurations.csv。','I/O是WSL2客户机块层流量，不是唯一扇区数或SSD内部NAND计数。DEEP查询曾在历史实验中观察，属于探索性实验。模型、索引和查询hash均锁定。','与SIFT原P0系列相比，本次神经网络检查160轮且逐步降学习率，阈值预算也做了DEEP标定；跨数据集变化不能纯归因于数据集难度。ET仅训练于未剪枝路径，联合组可能存在轨迹偏移；8阈值对也不足以排除更充分搜索的收益。','分类误差、模型推理成本、最终检索质量是不同量。有限网格和重复计时不构成普遍优越性或统计显著性证明。','共享的完全相同对照配置只测一次供多个P0实验引用；P0-A两个版本独立计时。完整记录在父目录runs/、all_test_runs.json和INPUT_LOCK.json。']
 (root/'REPORT.md').write_text('\n'.join(text)+'\n')
 # Reuse the repository's existing standalone P0 plotting utility, including its
 # offline SVG backend when Matplotlib is unavailable in the environment.
 source=(R.parent/'p0_b_lr/plot.py').read_text().replace("FAMILIES=['Tuned_DiskANN','Heuristic','Ours','ET','ET_Ours']",'FAMILIES='+repr(families));start=source.index('LABELS=');end=source.index('\n',start);source=source[:start]+'LABELS='+repr([label[f] for f in families])+source[end:];source=source.replace('SIFT1M','DEEP1M').replace('P0-B LR THREE CORE + ONE MECHANISM FIGURES COMPLETE','DEEP P0 FIGURES COMPLETE');(root/'plot.py').write_text(source)
 with (root/'plot.log').open('w') as f:subprocess.run([sys.executable,str(root/'plot.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
# Query-paired sampling intervals for validation-locked policies, not test-picked
# empirical frontier points. Timing uncertainty is reported as repeated ranges.
comparisons=[]
for study,families in studies.items():
 held=readcsv(R/study/'validation_locked_test.csv');lookup={(x['family'],float(x['target']),x['objective']):x for x in held};rows={x['key']:x for x in json.loads((R/study/'aggregate.json').read_text())}
 for (family,target,objective),selection in lookup.items():
  base=lookup.get(('DiskANN',target,objective))
  if family=='DiskANN' or not selection['key'] or not base or not base['key']:continue
  a=rows[selection['key']];b=rows[base['key']];qa=np.genfromtxt(R/'runs/test'/f'{a["key"]}_r0.csv',delimiter=',',names=True);qb=np.genfromtxt(R/'runs/test'/f'{b["key"]}_r0.csv',delimiter=',',names=True);recall_ci=np.percentile(weights@(qa['hits']-qb['hits'])/100,[2.5,97.5]);read_ci=np.percentile(weights@(qb['reads']-qa['reads'])/1000,[2.5,97.5])
  comparisons.append(dict(study=study,family=family,target=target,objective=objective,key=a['key'],baseline_key=b['key'],recall=a['recall'],baseline_recall=b['recall'],both_test_feasible=a['recall']>=target and b['recall']>=target,recall_delta_pp=a['recall']-b['recall'],recall_delta_ci_low=float(recall_ci[0]),recall_delta_ci_high=float(recall_ci[1]),reads_saved=float(np.mean(qb['reads']-qa['reads'])),reads_saved_ci_low=float(read_ci[0]),reads_saved_ci_high=float(read_ci[1]),io_reduction_pct=pct(a['physical_bytes'],b['physical_bytes']),p95_reduction_pct=pct(a['p95_us'],b['p95_us']),qps_increase_pct=100*(a['qps']/b['qps']-1)))
savecsv(R/'paired_validation_locked_comparisons.csv',comparisons)
root_text=['# DEEP1M P0系列实验','','P0-A、P0-B、P0-C全部完成，包括P0-A/B的MLP与LR两版。使用DEEP原图R64/Lbuild100/PQ34，基础向量1M×96；K10、Beam4、10,000节点BFS缓存、4查询线程/MKL单线程。','',f'原生搜索共测量{online["validation_configs"]}个验证配置、{online["test_runs"]}次P0-B/C测试运行，另有P0-A的54次运行。最终前沿、同Recall点、质量要求赢家与验证锁定配置均至少3次测试；P0-A每方法9次。','', '## P0-A：固定L100','','两版本分别独立计时，每行时间为9次中位数。','','|版本|方法|Recall|P95 ms|QPS|KiB/Query|','|---|---|---:|---:|---:|---:|']
for name,version in [('p0_a','MLP'),('p0_a_lr','LR')]:
 for r in json.loads((R/name/'aggregate.json').read_text()):root_text.append(f'|{version}|{r["key"]}|{r["recall"]:.2f}%|{r["p95_us"]/1000:.3f}|{r["qps"]:.1f}|{r["physical_bytes"]/1024:.3f}|')
root_text+=['','## P0-B：Recall≥99.5%的最低读取量（描述性测试网格）','','各行同一配置；实际Recall不同。测试经验网格选出的点不等于验证锁定的部署结果，后者见各分报告。','','|方法|实际Recall|P95 ms|QPS|KiB/Query|L|','|---|---:|---:|---:|---:|---:|'];done=set()
for study in ['p0_b','p0_b_lr']:
 for r in readcsv(R/study/'test_feasible_frontier_DESCRIPTIVE.csv'):
  if float(r['target'])==99.5 and r['objective']=='physical_bytes' and r['family'] not in done:root_text.append(f'|{label[r["family"]]}|{float(r["recall"]):.2f}%|{float(r["p95_us"])/1000:.3f}|{float(r["qps"]):.1f}|{float(r["physical_bytes"])/1024:.3f}|{r["L"]}|');done.add(r['family'])
root_text+=['','## P0-C：相同训练样本上的模型消融','','四种Gate共享286550个mixed-L cache-miss样本、相同17维状态和代理retain标签；q–v额外使用96维查询/候选向量交互。下表为测试L100同一批cache-miss候选上的诊断。','','|模型|Retain AUROC|Safe-skip AP|原生score μs/候选|','|---|---:|---:|---:|']
for r in metrics:
 if r['split']=='test' and r['model']!='l100':root_text.append(f'|{r["model"]}|{r["retain_auc"]:.5f}|{r["safe_skip_ap"]:.5f}|{r["native_ns"]/1000:.4f}|')
root_text+=['','P0-C的主要系统指标是Recall–P95前沿；更高分类分数不保证搜索收益。q–v模型后期明显过拟合，按内部独立查询选择早期最佳点，而非最终160轮权重。','', '## 结果、曲线与复现','','- [P0-A MLP报告](p0_a/REPORT.md)','- [P0-A LR报告](p0_a_lr/REPORT.md)','- [P0-B MLP报告](p0_b/REPORT.md)','- [P0-B LR报告](p0_b_lr/REPORT.md)','- [P0-C消融报告](p0_c/REPORT.md)','- [P0-B MLP Recall–P95](p0_b/figures/recall_p95.svg)','- [P0-B LR Recall–P95](p0_b_lr/figures/recall_p95.svg)','- [P0-C Recall–P95](p0_c/figures/recall_p95.svg)','', '所有分报告均包含实际Recall、对应配置、精确同Recall比较及验证锁定配置的测试达标情况。P0-B/C的三项核心曲线沿用仓库现有P0绘图工具的离线SVG后端，点为实测配置，线仅引导阅读。','训练/内部留出/验证/测试按查询划分；1200条训练查询中200条作轮次选择，随后按选中轮次重拟合全部1200条。没有复用SIFT模型权重，原数据/索引/模型文件未更改。LR、XGB、mixed-L MLP、q–v MLP共享样本；另训单独L100 MLP用于原版P0-A/B。','全部原生导出核验通过；Recall由ID复算，重复离散计数一致，实际reader字节与Linux read_bytes一致。验证锁定的配对查询区间见paired_validation_locked_comparisons.csv，不包含训练、选参和计时不确定性。','解释边界：DEEP基准查询曾在历史实验中观察；本次属于探索性实验。相比SIFT原P0，神经网络训练安排延长、阈值使用DEEP验证数据校准；跨数据集变化不能纯归因于数据集本身。联合组只有8对阈值，没有充分搜索所有组合，ET也未针对剪枝轨迹重训。','I/O为WSL2客户机块层流量，不是SSD内部NAND计数。固定L的收益、同质量下限的收益和严格相同Recall的收益分别报告。','复现：`bash experiments/deep_p0/run.sh`。协议见PROTOCOL.json、GRID_LOCK.json；查询划分见splits.json；验证见verification.json；模型见models/；逐查询结果见runs/。']
(R/'REPORT.md').write_text('\n'.join(root_text)+'\n')
for name in ['p0_a','p0_a_lr','p0_b','p0_b_lr','p0_c']:
 p=R/name/'COMPLETE.json';info=json.loads(p.read_text());info.update(report='REPORT.md',analysis_complete=True);dump(p,info)
dump(R/'COMPLETE.json',online|dict(analysis_complete=True,report='REPORT.md',subexperiments=['p0_a','p0_a_lr','p0_b','p0_b_lr','p0_c'],native_export_verified=True,original_inputs_unchanged=True,bootstrap_replicates=5000))
manifest={}
for p in R.rglob('*'):
 if p.is_file() and '__pycache__' not in str(p) and p.name not in ['artifact_manifest.json','analysis.log','evaluation.log','train.log']:
  manifest[str(p.relative_to(R))]=dict(bytes=p.stat().st_size,sha256=sha(p))
dump(R/'artifact_manifest.json',manifest);print('DEEP P0 REPORTS AND FIGURES COMPLETE')
