from pathlib import Path
import json,csv,html
R=Path(__file__).resolve().parent
P=json.loads((R/'PROTOCOL.json').read_text());V=json.loads((R/'verification.json').read_text());D=json.loads((R/'frontiers.json').read_text());A=json.loads((R/'aggregate.json').read_text());RUNS=json.loads((R/'all_test_runs.json').read_text())
F=['Tuned_DiskANN','Heuristic','Ours','ET','ET_Ours'];LABEL=dict(zip(F,['Tuned DiskANN','DiskANN + Heuristic','DiskANN + Ours','DiskANN + ET','DiskANN + ET + Ours']))
with (R/'strict_equal_recall.csv').open() as f:E=list(csv.DictReader(f))
with (R/'validation_locked_test.csv').open() as f:LOCK=list(csv.DictReader(f))
metrics=[('p95_us','P95 Latency（ms，越低越好）',.001),('qps','QPS（越高越好）',1),('physical_bytes','实测块层读取（KiB/Query，越低越好）',1/1024)]
lines=['# P0-B：允许各方法调参后的端到端成本比较','', '## 实验问题与协议','', '在相同检索质量下，原始17维学习式节点展开剪枝，是否仍优于独立调节L的DiskANN？三项核心指标为P95、QPS和实测块层读取字节。直接读请求曲线只用于机制分析。','', 'SIFT1M，原始R64/Lbuild100/PQ34B图索引，K=10、Beam=4、10,000节点BFS静态缓存、4查询线程/MKL单线程。验证和测试各1,000条互斥查询；权重训练查询与两者互斥。查询集合和缓存与P0-A相同。','', 'Ours沿用P0-A原始冻结17–128–64–1 MLP，未换成mixed-L、CF或q–v模型。模型训练于L100，此处冻结权重并测量不同L上的实际表现。ET使用已有混合L训练的17维前缀Recall预测模型，在Beam边界且至少已有K个精确候选后决定停止；是DiskANN内的可比策略，未声称忠实复现DARTH/LAET，未混用Faiss结果。','', f'五组各64配置，共{V["validation_configs"]}个验证配置和{V["test_configs"]}个测试配置。Tuned DiskANN扫描64个L（10–300），包括所有学习方法的L；其余使用L={P["L_grid"]}×8个阈值/规则档位。联合方法为8对预先锁定的阈值组合，非完整笛卡尔积。所有方法保护缓存命中节点；Gate在读取前决定剪枝。具体阈值见PROTOCOL.json。','', '先锁定全体参数网格，再测验证集；按Recall目标和目标成本用验证集选择并锁定配置，之后才测测试集。完整固定网格在测试集上形成描述性的经验前沿，不重新训练或扩充网格。每项指标分别形成前沿；P95、QPS、I/O最佳配置不一定相同。','', f'测试实际运行{V["test_runs"]}次；最终三项前沿、严格同Recall最优点以及验证锁定点均已重复3次，时间取中位数。每配置先热身128条查询，所有特征提取、推理、直接I/O和搜索计入时间；索引/模型/缓存加载、热身、GT和输出不计入。QPS为1000/四线程搜索墙钟时间，P95为每次1000条查询P95的跨次中位数。','', '## 三张核心曲线','', '- [Recall–P95](figures/recall_p95.svg)','- [Recall–QPS](figures/recall_qps.svg)','- [Recall–实测块层读取字节](figures/recall_physical_bytes.svg)','', '每个标记是实际测量配置，横轴为实际测试Recall；连线仅用于引导阅读，不假设中间Recall存在可部署配置，也不插值延迟/QPS。左右面板分别展示90–100%及99–100%。','', '## 严格相同实测Recall对照','', '按相同整数total_hits匹配，未靠四舍五入匹配。各指标分别取该精确Recall下最优配置。这里可能包含被更高Recall配置支配的点，因而最终方法判断还应看完整前沿；frontier_member已在CSV逐行标记。关闭动作的配置允许入选，Gate/ET的推理开销仍计入。','']
recalls=V['exact_common_recall_pct']
for metric,title,scale in metrics:
 lines+=['### '+title,'', '| 方法 | '+' | '.join(f'{r:.2f}%' for r in recalls)+' |','|---|'+'---:|'*len(recalls)]
 for fam in F:
  vals=[]
  for rc in recalls:
   row=next(r for r in E if r['objective']==metric and r['family']==fam and abs(float(r['recall'])-rc)<1e-8);vals.append(f'{float(row[metric])*scale:.3f}' if metric!='qps' else f'{float(row[metric]):.1f}')
  lines.append('| '+LABEL[fam]+' | '+' | '.join(vals)+' |')
 lines.append('')
lines+=['## 允许更高Recall的可行前沿：描述性结果','', '下表在实际测试Recall≥目标的配置中取最低I/O；每行明确列出实际Recall。它们满足同一质量要求，但不属于严格相同Recall比较；选取自测试经验网格，不是验证锁定的无偏部署评估。','', '| 目标Recall | 方法 | 实际Recall | KiB/Query | L | Gate τ | ET τ |','|---:|---|---:|---:|---:|---:|---:|']
for target in P['targets']:
 for fam in F:
  fs=[r for r in A if r['family']==fam and r['recall']>=target]
  if fs:
   r=min(fs,key=lambda r:r['physical_bytes']);lines.append(f'| {target:g}% | {LABEL[fam]} | {r["recall"]:.2f}% | {r["physical_bytes"]/1024:.3f} | {r["L"]} | {r["tau"]:.6g} | {r["et_tau"]:.6g} |')
lines+=['','## 验证集锁定的部署配置：99.5%目标','', '下表是独立于测试结果选出的配置。是否满足测试质量要求单独判断；同一验证目标并不意味着测试Recall一致。完整90/95/98/99/99.5/99.7目标见validation_locked_test.csv。','', '| 目标指标 | 方法 | 实际测试Recall | 测试成本 | 达到99.5% | L |','|---|---|---:|---:|---|---:|']
for metric,title,scale in metrics:
 for fam in F:
  r=next(r for r in LOCK if r['family']==fam and float(r['target'])==99.5 and r['metric']==metric)
  if r['key']:lines.append(f'| {title} | {LABEL[fam]} | {float(r["test_recall"]):.2f}% | {float(r["test_cost"])*scale:.3f} | {r["test_meets_target"]} | {r["test_L"]} |')
lines+=['','## 物理读取与机制验证','', '核心I/O指标直接取正式搜索窗口前后Linux /proc/self/io read_bytes差值，再除以查询数；并独立在读取器边界累加真实直接读请求长度。所有运行都验证两者完全相等；实际请求为4096字节，累计请求数与逐Query n_ios完全相等。没有用节点展开数估算读取字节。O_DIRECT绕过客户机文件页缓存；静态节点缓存固定不变。','', '环境为WSL2/ext4虚拟块设备；计数代表客户机块层读取流量，不代表唯一扇区数量、宿主机实际SSD指令或NAND读取。底层缓存可能影响延迟，但所有方法共享同一环境并随机交错运行。','', '机制图：[Recall–直接读请求](figures/mechanism_recall_requests.svg)。在本索引的4KiB固定请求下，它与字节曲线呈比例，但物理字节指标独立实测核验。','', f'验证集关闭动作对照{len(V["validation_disabled_parity"])}个，测试集关闭动作对照{len(V["test_disabled_parity"])}个：全部精确匹配对应L的Vanilla完整结果ID、Recall、I/O与展开数。所有重复的ID和离散计数完全相同；L100 Vanilla完整ID匹配P0-A，缓存节点列表哈希一致，模型/索引/查询/源文件哈希未变。','', '## 数据与解释边界','', '这些查询之前已用于探索性评估，结果不能声称全新确认性测试。有限且相等的64配置预算不证明全局最优；联合模型只覆盖预声明阈值对。测试前沿选优和三次计时仍有有限样本噪声，小幅尾延迟差异需要谨慎解释。ET在未剪枝轨迹上训练，联合轨迹存在分布变化；原始Gate在L100外也有分布变化，均在此真实搜索中测量。','', '复现：依次执行prepare.py、run.py、verify.py、plot.py、analyze.py、report.py（项目.venv/bin/python）。已有完整运行可复用；重新计时先归档runs。配置、输入哈希和选参锁分别为configs.json、INPUT_LOCK.json、SELECTION_LOCK.json。完整实测前沿为frontier_points.csv；严格同Recall为strict_equal_recall.csv；完整测试配置为all_configurations.csv；验证锁定测试结果为validation_locked_test.csv。']
# A coherent configuration table complements independently optimized curves.
chosen={fam:min([r for r in A if r['family']==fam and r['recall']>=99.5],key=lambda r:r['physical_bytes']) for fam in F}
b=chosen['Tuned_DiskANN'];o=chosen['Ours']
summary=['## 结果判读','',f'在实际测试Recall至少99.5%的固定网格中，Ours以实际Recall {o["recall"]:.2f}% 对比Tuned DiskANN {b["recall"]:.2f}%，读取量减少{(1-o["physical_bytes"]/b["physical_bytes"])*100:.2f}%，QPS提升{(o["qps"]/b["qps"]-1)*100:.2f}%，但P95增加{(o["p95_us"]/b["p95_us"]-1)*100:.2f}%。这支持该区间I/O和吞吐收益，不支持尾延迟全面改善。该点Ours的Recall更高，因此与此基线配置的收益不依赖牺牲Recall。','', 'ET在99.5%质量要求下的最低I/O和最高QPS优于Ours，但ET最低I/O配置的P95明显高于其自身延迟最优配置。联合方法的最低I/O配置ET阈值为1.1、实际停止率为0，不能据此声称ET与Gate已经产生互补收益。联合只扫描了预声明8对阈值，未排除更细搜索获得改进。','', '90–98%区间未显示Ours有稳定系统优势；99.7%质量要求下当前网格中Ours也不占优。结论局限于原始L100训练Gate和本次有限网格，不能归结为所有学习剪枝策略的上限。严格相同Recall离散点对照见下文；这些点上原始Gate的读取节省大多很小或为零。','', '以下每行采用实际测试Recall≥99.5%时的最低I/O配置；同一行全部数字来自同一配置，并非各指标分别最优。三个指标独立最优配置请看三张前沿图及quality_requirement_summary.csv。','', '| 方法 | 实际Recall | P95 ms | QPS | 实测KiB/Query | L |','|---|---:|---:|---:|---:|---:|']
for fam,r in chosen.items():summary.append(f'| {LABEL[fam]} | {r["recall"]:.2f}% | {r["p95_us"]/1000:.3f} | {r["qps"]:.1f} | {r["physical_bytes"]/1024:.3f} | {r["L"]} |')
summary+=['','注意：这是测试网格的描述性结果。验证集99.5%目标锁定配置在测试集的Recall分别为Tuned DiskANN99.46%、Heuristic99.52%、Ours99.48%、ET99.46%、ET+Ours99.46%，只有Heuristic达到该测试目标；不能把其余配置的成本当作满足99.5%的部署证明。','']
lines[2:2]=summary
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Include timing ranges and action flags for every plotted configuration.
fields=[]
for metric in ['p95_us','qps','physical_bytes']:
 for fam in F:
  for r in D[metric][fam]:
   if r['recall']<89:continue
   rs=[x for x in RUNS if x['key']==r['key']];fields.append(dict(objective=metric,**r,p95_min_us=min(x['p95_us'] for x in rs),p95_max_us=max(x['p95_us'] for x in rs),qps_min=min(x['qps'] for x in rs),qps_max=max(x['qps'] for x in rs),pruning_active=r['skips']>0,termination_active=r['et_stop_fraction']>0))
with (R/'frontier_points_with_ranges.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(fields[0]));w.writeheader();w.writerows(fields)
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P0-B: tuned DiskANN comparison</title><style>body{font:16px system-ui;max-width:1360px;margin:32px auto;padding:0 24px}img{width:100%}a{color:#075cc1}</style><h1>P0-B：允许各方法调参后的系统成本</h1><p>五组各64配置；实际测试Recall经验前沿。标记为实测配置，连线仅引导阅读。Ours沿用P0-A原始17维Gate。</p>'''
for file,label in [('recall_p95','P95 Latency'),('recall_qps','QPS'),('recall_physical_bytes','实测块层读取 KiB / Query')]:page+=f'<h2>{label}</h2><img src="figures/{file}.svg" alt="{label}">'
page+='<p><a href="REPORT.md">报告</a> · <a href="strict_equal_recall.csv">严格相同Recall</a> · <a href="validation_locked_test.csv">验证锁定结果</a> · <a href="figures/mechanism_recall_requests.svg">机制：读请求</a></p></html>'
(R/'index.html').write_text(page);print('REPORT + VIEWER COMPLETE')
