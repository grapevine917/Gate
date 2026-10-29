from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'summary.json').read_text())
lines=['# 质量下降查询：阶段、关键节点和延迟展开','',
'**结论：多数有效误剪发生在搜索中后段；每条失败查询只需保护少数节点，但没有观察到跨失败查询共享的少数关键 Hub。延迟展开能恢复 Recall，但当前简单排序的小预算只能部分恢复；补回全部节点会基本抵消请求节省。**','',
'复用固定 state_lr 模型、原验证阈值、各 1,000 条测试查询。参数为 L=100、k=10、beam=4、固定 cache=10,000，展开不设硬上限。统计口径为 Submitted SSD Read Requests。完整干预定义和实验边界见 [PROTOCOL.md](PROTOCOL.md)。','',
'## 误剪在哪个阶段','',
'在每条失败查询中逐个保护实际被剪节点，并完整重放。下表只计“保护后 Recall 改善”的节点；这些有效干预不必是唯一或必要原因。','',
'| 数据集 | 验证标签 FPR 档 | 失败查询 | 有效保护节点事件 | 展开 0–9 | 展开 10–39 | 展开 40–79 | 展开 ≥80 | 单节点足以完整恢复的查询 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for ds in ['sift','deep']:
 for t in range(3):
  r=s[ds]['thresholds'][str(t)];p=r['improving_events_by_completed_expansion_phase']
  lines.append(f"| {ds.upper()}1M | {[0.1,0.5,1][t]:g}% | {r['failed_queries']} | {r['single_protection_improving_events']} | {p.get('0-9',0)} | {p.get('10-39',0)} | {p.get('40-79',0)} | {p.get('80+',0)} | {r['single_node_full_recovery_queries']}/{r['failed_queries']} |")
lines+=['','在 1% 档，SIFT 39/41（95.12%）、DEEP 26/26 的有效节点在已完成 40 次展开后被剪。候选排名中位数为 SIFT 83、DEEP 77。阶段分桶与排名是不同指标。','',
'SIFT 有早期例外：测试行 894，对应原查询 ID 8986。节点 787150、955665 在仅完成 9 次展开、队列排名 1/2 时被剪，它们都在真实 top-10 及原始返回结果内。该查询原始命中 10/10，永久剪枝命中 8/10；单独保护任一节点恢复到 9/10。这说明质量保护不能只关心尾部候选。','',
'## 是否集中于少量节点','',
'| 数据集 | 去重失败查询 | 有效 (query,node) 对 | 不同节点 ID | 直接属于原始正确 top-10 的节点 | 单节点最大覆盖不同失败查询数 | 联合保护名单大小上限 |',
'|---|---:|---:|---:|---:|---:|---:|']
for ds in ['sift','deep']:
 c=s[ds]['concentration'];j=s[ds]['oracle_joint_protection']['2']
 lines.append(f"| {ds.upper()}1M | {c['unique_failed_queries']} | {c['unique_improving_query_node_pairs']} | {c['unique_improving_node_ids']} | {c['direct_result_unique_pairs']} | {c['max_distinct_failed_queries_per_node']} | {j['max_protected_nodes_per_query']} |")
lines+=['','联合保护名单由逐节点诊断得到，再真实重放验证：SIFT 各失败查询保护名单含 1–4 个节点，DEEP 含 1–3 个节点，均恢复到原始 Recall 或更高；没有证明名单最小。保护对象还包括原始最终结果外的路径节点，不能仅保护预测的 top-10 结果节点。所有有效节点在本失败集内仅涉及一条不同查询，没有共享 Hub 集中的证据。','',
'先前 baseline 单节点标签与当前实际干预存在差异（1% 档）：','']
for ds in ['sift','deep']:
 r=s[ds]['thresholds']['2']['prior_singleton_labels_of_improving_nodes']
 lines.append(f"- {ds.upper()}1M：有效节点中 {r.get('singleton_quality_critical',0)} 个原先单节点质量关键，{r.get('singleton_safe_saving',0)} 个原先单节点安全且省请求，{r.get('singleton_safe_non_saving',0)} 个原先安全但不省请求，{r.get('not_in_baseline',0)} 个不在原始展开记录中。")
lines+=['','这是联合路径依赖的实证：单独跳过安全，不意味着其他节点也被剪之后仍然安全。有效保护既可能直接补回正确结果，也可能改变后续路由；本实验不把所有改善归因于当前节点的直接结果贡献。','',
'## 延迟展开能否恢复 Recall','',
'第一阶段与永久剪枝的评分/决策状态序列逐次相同。队列耗尽后进入恢复阶段，暂存节点直接展开，随后使用原始 best-first 搜索并关闭分类剪枝。预算只限制恢复的种子节点数，所有后代展开/读取均计入成本。以下以 PQ 顺序报告 1% 档，全量 1,000 条查询均启用该机制：','',
'| 数据集 | 策略 | 原失败恢复 | 剩余质量下降查询 | 全量净请求节省 | 全量 Recall 损失（百分点） |',
'|---|---|---:|---:|---:|---:|']
for ds in ['sift','deep']:
 for mode in ['permanent','defer_1','defer_4','defer_8','defer_16','defer_32','defer_all']:
  r=s[ds]['delayed_policies']['pq']['2/'+mode]
  lines.append(f"| {ds.upper()}1M | {mode} | {r['original_failures_recovered']}/{s[ds]['thresholds']['2']['failed_queries']} | {r['degraded_queries']}/1000 | {r['net_request_saving_ratio']:.2%} | {r['recall_loss_pp']:.2f} |")
lines+=['','全部补展开在三档阈值、两个数据集都恢复全部原失败查询，也没有新增质量下降查询；恢复 Recall 不等价于全部返回 ID/距离完全一致。1% 档全量净节省为负，意味着相对于原始搜索请求更多。','',
'同预算下比较延迟排序（1% 档）：','',
'| 数据集 | 排序 | 补展开 8 个：剩余下降 / 净节省 | 补展开 16 个：剩余下降 / 净节省 | 补展开 32 个：剩余下降 / 净节省 |',
'|---|---|---|---|---|']
for ds in ['sift','deep']:
 for order in ['pq','retain','rank']:
  vals=[s[ds]['delayed_policies'][order][f'2/defer_{budget}'] for budget in [8,16,32]]
  lines.append('| '+ds.upper()+'1M | '+order+' | '+' | '.join(f"{r['degraded_queries']}/1000 / {r['net_request_saving_ratio']:.2%}" for r in vals)+' |')
lines+=['','按拒绝时 retain 分数优先、按原队列排名优先，都未稳定改善恢复率。这些排序没有利用测试标签，却还不能从大量冗余候选中可靠找出少数关键节点。','',
'稀疏保护 Oracle 作为诊断对照（测试反馈选择名单，不可当成部署收益）：','']
for ds in ['sift','deep']:
 r=s[ds]['oracle_joint_protection']['2']
 lines.append(f"- {ds.upper()}1M：全量回放只对失败查询保护其有效名单，质量下降查询 {r['oracle_patch_degraded_queries']}/1000，净请求节省 {r['oracle_patch_net_request_saving_ratio']:.2%}。")
lines+=['','这显示关键节点选择仍有研究价值，但“全部延迟后补搜”与简单优先级尚未达到稀疏 Oracle 的收益。该轮利用测试反馈分析失败并扫描预算，没有重新宣称这些机制已在独立测试集验证泛化。','',
'## 验证与产物','',
'- 永久剪枝各查询的读取数和展开数与前轮原生策略一致；补展开阶段未改变第一阶段状态序列。','- 每次回放检查实际补展开数不超预算或暂存节点数；按三个冻结阈值统计，保留恢复失败及请求增加的结果。','- 默认延迟关闭；原生分类评分和旧 Top3 返回结果兼容性检查通过。','',
'原生 runner：`DiskANN/apps/search_delay_diagnosis.cpp`。明细：[improving_protection_nodes.csv](improving_protection_nodes.csv)、`*_skips.csv`、`*_protection.csv`、`*_queries.csv`。汇总：[summary.json](summary.json)，分析脚本：[analyze.py](analyze.py)，复现入口：[run.sh](run.sh)。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('*.csv'))+list(ROOT.glob('*.txt'))+[ROOT/'summary.json',ROOT/'run.sh']
lg=ROOT.parent.parent
for ds in ['sift','deep']:
 paths += [ROOT.parent/'io_predictability'/ds/r for r in ['state_lr/model.bin','state_lr/thresholds.json','test/query.fbin','test/gt.bin','test/query_ids.npy']]
paths += [lg/'DiskANN'/r for r in ['src/pq_flash_index.cpp','include/online_gate.h','include/percentile_stats.h','apps/search_delay_diagnosis.cpp','build/apps/search_delay_diagnosis']]
(ROOT/'provenance.json').write_text(json.dumps({'io_unit':'Submitted SSD Read Requests','model':'frozen state_lr, original validation thresholds','query_partition':'original test 1000 per dataset','stage_bins_completed_expansions':[0,10,40,80],'deferred_seed_budgets':[1,4,8,16,32,'all'],'refinement':'gate disabled, existing visited/retset/full result state retained','oracle_protection':'test-feedback diagnostic, not deployable','files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}},indent=2)+'\n')
