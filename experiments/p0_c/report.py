from pathlib import Path
import json,csv
import numpy as np
R=Path(__file__).resolve().parent;P=json.loads((R/'PROTOCOL.json').read_text());V=json.loads((R/'verification.json').read_text());A=json.loads((R/'aggregate.json').read_text());RUNS=json.loads((R/'all_test_runs.json').read_text());PR=json.loads((R/'prediction_metrics.json').read_text());D=json.loads((R/'frontiers.json').read_text())
F=['DiskANN','LR_Gate','XGBoost_Gate','State_MLP','QV_MLP'];LABEL=dict(zip(F,['Tuned DiskANN','LR','XGBoost','MLP (17D)','q–v MLP (529D)']))
lines=['# P0-C：Gate实现消融','', '## 范围与公平性','', '这是方法实现消融，独立于P0-A固定参数和P0-B原始Gate调参实验。四种Gate使用既有同一组混合L训练样本：1200个训练查询、285754个cache-miss候选。LR/XGBoost/MLP使用同一17维状态；q–v MLP使用[q,v,abs(q-v),q*v,state17]共529维。标签统一为“有新邻居入队或属于本次baseline最终Top10则retain”，是贡献代理标签，不是反事实Recall损害标签。','', 'MLP为17–128–64–1，q–v MLP为529–128–64–1；XGBoost为278棵最大深度5的树；LR为标准化线性逻辑回归。冻结权重，不在本次测试集上重新训练。训练查询与验证/测试互斥，XGB/MLP轮数在训练内部1000/200查询划分上选择后再用1200查询拟合。','', 'P0-C的17D MLP是这组匹配训练的mixed-L权重，不是P0-A/B的原始L100模型。不得将跨实验的MLP差异全部解释为模型实现差异。','', '## 预测能力与单候选推理成本','', '下表在同一批102405个测试集L100 baseline缓存未命中候选上评价；所有四种模型的样本与标签完全一致。AUROC以retain为正类，AP以可剪枝代理标签为正类。单候选耗时是原生C++单线程score，20,000个随机样本、预热后5次扫过的平均耗时的中位数，不使用批量推理。包含归一化和q–v特征拼接/向量访问，不含搜索队列状态抽取；真实搜索时间则全部包含。','', '| Gate | Retain AUROC ↑ | Safe-skip AP ↑ | Logloss ↓ | 原生score μs/候选 ↓ | 模型文件 KiB |','|---|---:|---:|---:|---:|---:|']
for m in [x for x in PR if x['split']=='test']:lines.append(f'| {dict(lr="LR",xgb="XGBoost",state="MLP",qv="q–v MLP")[m["model"]]} | {m["retain_ROC_AUC"]:.4f} | {m["safe_skip_AP"]:.4f} | {m["retain_logloss"]:.4f} | {m["native_ns_per_score_median"]/1000:.4f} | {m["model_file_bytes"]/1024:.3f} |')
lines+=['', '该分类诊断是在baseline轨迹上的代理标签，不能用AUROC代替最终Recall或搜索成本。q–v同时增加了查询和候选向量信息，不能仅据此声称收益或无收益来自“查询信息”本身；单独归因需要q-only/v-only等进一步消融。','', '## 主要结果：Recall–P95 Latency Pareto Frontier','', '![Recall–P95](figures/recall_p95.svg)','', '横轴是每配置真实测试Recall，标记是实测配置；连线仅引导阅读，不表示中间点可部署，不对Latency/QPS插值。左右面板分别为90–100%与99–100%。每个Gate在相同8个L（15/20/30/50/75/100/160/300）和8个训练分数分位阈值下比较，共64配置；另附64个L的Tuned DiskANN参考。网格在原主实验时就已冻结，本次不增加阈值。每条曲线按自己的目标独立取非支配前沿。','', f'新测全部320个测试配置，共{V["test_runs"]}次完整测试运行，最终各指标前沿和精确同Recall最优点均有3次测量，取时间中位数。每个进程只加载对应模型，Vanilla不加载Gate或全量向量表，按family分批交错并随机化每批配置顺序。结果ID、Recall、请求、展开、skip和模型调用数均与既有匹配模型的每个配置逐项核验一致，分类/推理微基准在搜索计时前完成，避免并行争用CPU。','', '固定SIFT1M图索引R64/Lbuild100/PQ34B，K10、Beam4、10,000节点BFS缓存、4查询线程及每查询MKL单线程，1000条相同测试查询。每配置热身128条，搜索计入所有状态抽取、推理、q–v DRAM访问、I/O和邻居处理；加载、热身、GT和结果输出不计入。','', '## 相同质量要求下的P95最优配置','', '每个目标下取实际测试Recall≥目标时的最低P95配置，并明确报告实际Recall。这是测试网格描述性前沿，不能将不同实际Recall叫作严格等Recall。未做本次新的验证选参部署承诺。','', '| Recall要求 | Gate | 实际Recall | P95 ms | L | 阈值 | 相对Tuned DiskANN P95降低 |','|---:|---|---:|---:|---:|---:|---:|']
for target in P['targets']:
 best={f:min([r for r in A if r['family']==f and r['recall']>=target],key=lambda r:r['p95_us']) for f in F};b=best['DiskANN']
 for f,r in best.items():lines.append(f'| {target:g}% | {LABEL[f]} | {r["recall"]:.2f}% | {r["p95_us"]/1000:.3f} | {r["L"]} | {r["tau"]:.6g} | {(1-r["p95_us"]/b["p95_us"])*100:.2f}% |')
lines+=['','## 严格相同实测Recall的P95','', '按相同整数total_hits匹配，不靠Recall四舍五入。部分点可能被更高Recall配置支配；frontier_member在CSV标记。tau=0关闭剪枝动作仍执行推理，不能把无剪枝时的少量计时差异当作剪枝收益。','']
with (R/'strict_equal_recall.csv').open() as f:EX=list(csv.DictReader(f))
recalls=V['exact_common_recall_pct'];lines+=['| Gate | '+' | '.join(f'{v:.2f}%' for v in recalls)+' |','|---|'+'---:|'*len(recalls)]
for fam in F:
 vals=[next(float(r['p95_us'])/1000 for r in EX if r['objective']=='p95_us' and r['family']==fam and abs(float(r['recall'])-v)<1e-8) for v in recalls];lines.append('| '+LABEL[fam]+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
lines+=['','## 不剪枝时的完整Gate开销与内存','', '下面使用tau=0、L100：各组返回结果和搜索读/展开完全一致，时间差包含完整状态抽取与推理的实际系统开销及运行噪声。RSS为预热后、正式搜索前/proc/self/status VmRSS，独立进程只加载对应模型。','', '| 实现 | P95 ms | QPS | Gate调用/Query | RSS MiB |','|---|---:|---:|---:|---:|']
for fam in F:
 r=next(r for r in A if r['family']==fam and r['L']==100 and r['preset']==0);lines.append(f'| {LABEL[fam]} | {r["p95_us"]/1000:.3f} | {r["qps"]:.1f} | {r["gate_calls"]:.3f} | {r["resident_kib"]/1024:.2f} |')
lines+=['', 'q–v额外常驻完整SIFT1M浮点向量表512,000,000字节（488.28MiB），此开销不包含在模型文件大小内。这里没有强行压到相同RAM预算下，因此q–v的成本必须连同额外内存一起解释。','', '## 辅助曲线与验证','', '- [Recall–QPS](figures/recall_qps.svg)','- [Recall–实测块层读取字节](figures/recall_physical_bytes.svg)','- [机制：Recall–直接读请求](figures/mechanism_recall_requests.svg)','', '物理字节是搜索前后Linux read_bytes差值，读取器另记实际请求数量和长度；所有运行验证两者相等。O_DIRECT绕过客户机文件页缓存，但WSL2计数仅代表客户机块层流量，非SSD内部NAND操作。缓存节点列表固定且哈希一致。','', '原生模型分数与Python导出验证误差均小于1e-6。重复运行离散字段和完整ID一致，原模型/索引/查询输入哈希不变。新的计时不与P0-A/B数字混表；旧验证结果仅作为冻结模型的参考标签/离散轨迹检查，未复用其延迟进行本次参数选择。','', '## 解释边界与复现','', '查询曾被观察，属于探索性消融；64配置有限网格不证明全局最优。三次计时中位数仍有选择噪声，数个百分点以内的差异不能自动宣称显著。预测诊断限于L100 baseline状态；真实搜索覆盖全部L与Gate干预轨迹。','', '依次运行prepare.py、predict.py、run.py、verify.py、plot.py、analyze.py、report.py；不能并发运行微基准与搜索计时。已有完整搜索可复用，重新计时先归档runs。原始数据：prediction_metrics.json、prediction/*.bench.json、frontier_points.csv、strict_equal_recall.csv、all_configurations.csv、PROTOCOL.json及INPUT_LOCK.json。']
# Per-objective frontier points with range and action flags.
fr=[]
for metric in ['p95_us','qps','physical_bytes']:
 for fam in F:
  for r in D[metric][fam]:
   if r['recall']<89:continue
   rs=[q for q in RUNS if q['key']==r['key']];fr.append(dict(objective=metric,**r,p95_min_us=min(q['p95_us'] for q in rs),p95_max_us=max(q['p95_us'] for q in rs),qps_min=min(q['qps'] for q in rs),qps_max=max(q['qps'] for q in rs),pruning_active=r['skips']>0))
with (R/'frontier_points_with_ranges.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(fr[0]));w.writeheader();w.writerows(fr)
# Outcome is based on measured search frontiers, not classifier ranking.
winners={fam:0 for fam in F[1:]}
for target in P['targets']:
 best={fam:min([r for r in A if r['family']==fam and r['recall']>=target],key=lambda r:r['p95_us']) for fam in F[1:]};winners[min(best,key=lambda fam:best[fam]['p95_us'])]+=1
summary=['## 选择建议','',f'以Recall–P95为首要目标：当前实现优先LR。6个预声明质量要求下，LR在Gate中有{winners["LR_Gate"]}个最低P95，MLP有{winners["State_MLP"]}个；98%附近MLP仅略低，不能将微小差异视为显著。该计数是有限网格的描述性结果，不是完整连续前沿的统计检验。','', 'MLP的代理标签AUROC/AP最好，原生推理约0.618μs，保留为更强非线性预测候选。但当前测量不支持仅凭AUROC将其设为低延迟默认。XGBoost预测能力略低于MLP、单候选推理约慢13倍，当前278树实现未显示足够补偿。q–v MLP在本次L100诊断中AUROC更低，推理约3.583μs并额外占用512MB向量内存；不建议作为默认，不能据此否定所有查询交互模型。','', 'Tuned DiskANN仍是必要参考。95%到99.7%的各个质量要求下，它的最低P95均低于所有Gate；90%处LR仅有约0.17%的微小表面差异，不能宣称改善。学习剪枝可以改善读取流量或吞吐，但未显示普遍尾延迟收益。','', '例如测试Recall≥99.5%：LR实际Recall99.59%，对比Tuned DiskANN99.58%，P95为7.313ms对7.073ms（增加3.39%），同时QPS由603.0增至697.1（+15.60%）、读取量减少15.50%。因此“采用哪个Gate”和“是否采用Gate”是两个需要分别回答的问题。','']
lines[2:2]=summary
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
(R/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>P0-C Gate implementation ablation</title><style>body{font:16px system-ui;max-width:1360px;margin:30px auto}img{width:100%}</style><h1>P0-C：Gate实现消融</h1><p>匹配训练样本/标签；每种Gate64配置。实际测试Recall经验前沿，每进程只加载对应Gate。</p><h2>主要结果：Recall–P95 Latency</h2><img src="figures/recall_p95.svg"><h2>辅助：QPS</h2><img src="figures/recall_qps.svg"><h2>辅助：实测读取字节</h2><img src="figures/recall_physical_bytes.svg"><p><a href="REPORT.md">报告</a> · <a href="prediction_metrics.json">预测能力与推理成本</a> · <a href="strict_equal_recall.csv">严格同Recall</a></p></html>')
print('P0-C REPORT COMPLETE')
