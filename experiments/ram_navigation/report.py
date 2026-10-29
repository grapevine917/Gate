from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parent.parent
s=json.loads((ROOT/'fixed64_summary.json').read_text());v=json.loads((ROOT/'summary.json').read_text());build=json.loads((ROOT/'build_manifest.json').read_text())
lines=['# 固定预算 RAM 导航图：静态历史分布实验','',
'已实现 B1–B5，复用原有 Vamana 图、历史 8,000 / 验证 1,000 / 测试 1,000 查询，未训练神经网络、未做热点迁移。RAM 导航只读取已驻留向量和独立图，再把最近 4 个全局节点 ID 交给 DiskANN；底层原图没有改边。','',
'**主结果：本轮请求减少约 3%–5%，端到端收益较小。独立重建恢复 RAM 子图可达性，但未证明比热点缓存导航更优；B5 联合启发式相对 B4 的新增收益很小。DEEP 的部分方法仍有 Recall 点估计下降，不能当成严格质量保持的收益。**','',
'主对照均为 L=100、k=10、beam=4，RAM 候选容量 64、最多展开 64 节点。4 线程、固定 CPU 亲和性，三轮交错测试取中位数。Submitted SSD Read Requests 为 O_DIRECT Reader 的提交请求，不是 Physical SSD I/O。详见 [PROTOCOL.md](PROTOCOL.md)。','',
'## 五组主比较','',
'| 数据集 | 方法 | Recall@10 | 请求/查询 | 请求净节省 | 导航 µs | 端到端 ms | QPS | QPS 变化 |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
for ds in ['sift','deep']:
 for method in ['B1','B2','B3','B4','B5']:
  r=s[ds]['test'][method]
  lines.append(f"| {ds.upper()}1M | {method} | {r['recall_at_10']:.2%} | {r['mean_requests']:.3f} | {r['net_request_saving_ratio']:.2%} | {r['mean_navigation_us']:.2f} | {r['mean_latency_us']/1000:.3f} | {r['qps']:.1f} | {r['qps_change_ratio']:+.2%} |")
lines+=['','B1=原始 BFS Cache；B2=热点＋全部原驻留连接；B3=热点＋查询覆盖＋全部原驻留连接；B4=B3 同节点集＋独立几何边；B5=历史导航反馈驱动的节点替换＋独立边更新。','',
'读取计数和 Recall 在三个重复中一致。性能变化来自当前机器的固定亲和性小规模测量，保留各轮区间，未声称生产环境稳定收益：','',
'| 数据集 | 方法 | QPS 三轮范围 | 相对同轮 B1 的 QPS 变化范围 |','|---|---|---:|---:|']
for ds in ['sift','deep']:
 for method in ['B2','B3','B4','B5']:
  r=s[ds]['test'][method];paired=r['paired_qps_change_ratios']
  lines.append(f"| {ds.upper()}1M | {method} | {min(r['qps_runs']):.1f}–{max(r['qps_runs']):.1f} | {min(paired):+.2%}–{max(paired):+.2%} |")
lines+=['','## 质量约束','',
'平均 Recall 保持不代表逐查询保持。以下是同一测试 Query 的配对比较；bootstrap 区间是诊断，不是质量保证：','',
'| 数据集 | 方法 | Recall 差值（百分点） | 95% bootstrap 区间 | 下降 Query / 1000 | 改善 Query / 1000 |',
'|---|---|---:|---|---:|---:|']
for ds in ['sift','deep']:
 for method in ['B2','B3','B4','B5']:
  r=s[ds]['test'][method];ci=r['recall_difference_pp_bootstrap_95pct_ci']
  lines.append(f"| {ds.upper()}1M | {method} | {r['recall_difference_pp']:+.2f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {r['quality_degraded_queries']} | {r['quality_improved_queries']} |")
lines+=['','SIFT 所有方法平均 Recall 略升；DEEP 的 B3 与 B1 平均 Recall 相同，B2/B4/B5 略低。不能将 DEEP B4/B5 的较高吞吐量直接解释为同 Recall 下的确定改进，也不能把平均提升宣传为所有查询均安全。','',
'## RAM 与结构成本','',
'公共底层 DiskANN/PQ/原生四线程配置相同；下表完整计入 Cache 和独立导航的保留分配。向量与 Cache 共享一份，已经收费，没有额外保留全库浮点向量。MiB=2^20 bytes。','',
'| 数据集 | 方法 | 缓存节点数 | 总字节上限 MiB | 实际 Cache＋导航 MiB | 其中共享坐标 MiB | 原生邻接 MiB | 独立边表/ID/元数据 MiB | 导航工作区 MiB |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
for ds in ['sift','deep']:
 for method in ['B1','B2']:
  r=s[ds]['test'][method];m=r['memory'];scale=1048576
  lines.append(f"| {ds.upper()}1M | {'B1' if method=='B1' else 'B2–B5'} | {m['nodes']} | {s[ds]['budget_bytes']/scale:.4f} | {m['cache_and_navigation_bytes']/scale:.4f} | {m['cache_vector_allocated_bytes']/scale:.4f} | {m['cache_original_adjacency_allocated_bytes']/scale:.4f} | {m['navigation_graph_bytes']/scale:.4f} | {m['navigation_thread_scratch_bytes']/scale:.4f} |")
lines+=['','上表原生缓存总分配还包含哈希索引分配及缓存容器对象，因此不能只把坐标＋邻接两列相加当成总 Cache。所有方法在同一字节上限内，B2–B5 预留相同边槽和工作区容量。为了隔离拓扑，B2/B3 未将 CSR 的未使用边槽空间用于更多节点；本轮是同预算受控实现，不是各方法最优内存分配的证明。进程 live heap 与 max RSS 另存在每次运行文件，RSS 包括 allocator 保留页等，不能假称与索引字节预算相同。','',
'| 数据集 | 方法 | 有效导航边 | 从入口可达节点 / 全部节点 |','|---|---|---:|---:|']
for ds in ['sift','deep']:
 for method,r in build[ds]['configurations'].items():
  lines.append(f"| {ds.upper()}1M | {method} | {r['valid_edges']} | {r['reachable_from_root']}/{r['nodes']} |")
lines+=['','B2/B3 保留全部驻留原连接，没有逐点 16 边截断；CSR 的全局边槽容量为 16N，但单节点可有最多 64 条原边。B4/B5 的每点 16 边含两个环连接，全图可达。全图可达不代表有限导航预算下一定找到最优入口，实际效果以底层 Recall/请求/延迟为准。','',
'导航计算量：','',
'| 数据集 | 方法 | RAM 展开/查询 | RAM 向量距离计算/查询 | 导航 µs | 底层完整阶段 µs |',
'|---|---|---:|---:|---:|---:|']
for ds in ['sift','deep']:
 for method in ['B2','B3','B4','B5']:
  r=s[ds]['test'][method];lines.append(f"| {ds.upper()}1M | {method} | {r['mean_navigation_expansions']:.2f} | {r['mean_navigation_distances']:.2f} | {r['mean_navigation_us']:.2f} | {r['mean_backend_phase_us']:.2f} |")
lines+=['','## 分离缓存选点、重建边和导航作用','',
'关闭导航但保留相同缓存节点/内存配置的控制（B4 复用 B3 控制，因为节点与原生邻接相同）：','',
'| 数据集 | 控制 | 请求/查询 | 相对 B1 请求节省 | Recall@10 |','|---|---|---:|---:|---:|']
for ds in ['sift','deep']:
 for method in ['B2_cache_only','B3_cache_only','B5_cache_only']:
  r=v[ds]['test'][method];lines.append(f"| {ds.upper()}1M | {method} | {r['mean_requests']:.3f} | {r['net_request_saving_ratio']:.2%} | {r['recall_at_10']:.2%} |")
lines+=['','固定 B3/B4 节点、导航预算和底层搜索，仅改变导航边：','']
for ds in ['sift','deep']:
 a=s[ds]['test']['B3'];b=s[ds]['test']['B4'];c=s[ds]['test']['B5']
 lines.append(f"- {ds.upper()}1M：B3→B4 多省 {a['mean_requests']-b['mean_requests']:.3f} 请求/查询；导航时间增加 {b['mean_navigation_us']-a['mean_navigation_us']:.2f} µs；平均 Recall 差值 {(b['recall_at_10']-a['recall_at_10'])*100:+.2f} pp。B4→B5 只多省 {b['mean_requests']-c['mean_requests']:.3f} 请求/查询，未观察到明显额外收益。")
lines+=['','B5 是静态联合启发式：SIFT 替换 450 个节点及 8,936 条边；DEEP 替换 442 个节点及 8,838 条边。相对 B4 的改善非常小，不能把这一轮当成联合优化已优于热点缓存导航的证据。','',
'验证选导航预算的敏感性结果另存 [summary.json](summary.json)。DEEP B3/B5 在验证集没有平均质量可行的预算，明确标记并按 64 输出诊断；主固定 64 表不从测试选择最优预算。','',
'## 复现与验证','',
'- 各方法 Cache＋导航保留字节不超过 B1 上限；计入实际分配、共享向量、原始邻接、哈希元数据、导航边和四线程工作区。','- B3/B4 节点文件逐字节一致；历史/验证/测试 query ID 不相交。','- CSR 保留全部原驻留边及其顺序；与直接遍历原 Cache 邻接的实现逐查询 Recall、请求、展开和导航距离计数相同。','- B1 的逐查询请求数和展开数与前轮原始 DiskANN 完全一致，三次计数/Recall 确定性一致。','- 入口交接开关默认关闭，既有分类器与旧 Top3 模型的原生兼容性回归保持通过。','',
'原生实现：`DiskANN/apps/search_ram_navigation.cpp`、`DiskANN/include/ram_entry.h`。复现：[run.sh](run.sh)；构建：[build.py](build.py)；主统一预算：[fixed64_summary.json](fixed64_summary.json)；验证调度敏感性：[summary.json](summary.json)；检查：[verification.json](verification.json)。三轮原始请求/延迟/内存记录保存在各数据集 test 子目录。','',
'静态构建耗时（含历史解析、两次历史 RAM profile 等）约 SIFT 22.94 s、DEEP 20.46 s；此初始化成本不混入已加载索引的查询时间。下一轮若要求严格同 Recall，应先在独立验证上选择底层搜索强度与导航强度，再冻结测试，而不是根据本轮测试结果补调。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
files=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',ROOT/'REPORT.md',ROOT/'build_manifest.json',ROOT/'fixed64_summary.json',ROOT/'summary.json',ROOT/'verification.json',ROOT/'run.sh']
for ds in ['sift','deep']:
 for m in ['B2','B3','B4','B5']:files += [ROOT/ds/(m+'.ids'),ROOT/ds/(m+'.edges')]
 files += [ROOT/ds/'history_query_ids.npy',ROOT/f'{ds}_bfs.json']
files += [LG/'DiskANN'/p for p in ['apps/search_ram_navigation.cpp','include/ram_entry.h','include/pq_flash_index.h','src/pq_flash_index.cpp','build/apps/search_ram_navigation']]
manifest={'io_unit':'Submitted SSD Read Requests','primary_navigation_budget':64,'backend_L':100,'k':10,'beam':4,'threads':4,'history_queries':8000,'validation_queries':1000,'test_queries':1000,'neural_training':False,'online_updates':False,'vector_ownership':'shared native float32 cache; charged once','ram_accounting':'TCMalloc allocated bytes plus container objects','memory_cap_bytes':{ds:d['budget_bytes'] for ds,d in s.items()},'affinity':{'OMP_PROC_BIND':'close','OMP_PLACES':'cores'},'timing_repeats':3,'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}}
(ROOT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
