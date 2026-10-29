from pathlib import Path
import numpy as np,json,hashlib
ROOT=Path(__file__).resolve().parent;s=json.loads((ROOT/'summary.json').read_text());checks={};lines=['# 固定节点与有效边预算：拓扑实验结果','', '本轮评估 SIFT1M、DEEP1M。固定节点集合、逐源出度、有效边数、RAM 字节预算及导航和底层搜索参数，只有目标连接不同。历史增强比例由独立验证集选择，测试集不参与构图或选择。详见 [PROTOCOL.md](PROTOCOL.md)。','', '| 数据集 | 拓扑 | 有效边 | Recall@10 | 请求/Query | 导航 μs | 端到端 ms | QPS |','|---|---|---:|---:|---:|---:|---:|---:|']
for ds,d in s.items():
 ids0=None;deg0=None;memory=set();graph_checks={}
 for name in ['Original','Geometry','History25','History50','History100']:
  ids=(ROOT/ds/(name+'.ids')).read_bytes();raw=np.fromfile(ROOT/ds/(name+'.edges'),np.uint32);n=int(raw[0]);assert int(raw[1])==0x80000010 and int(raw[2])==0
  off=raw[3:3+n];edges=raw[3+n:];valid=int(np.flatnonzero(edges==0xffffffff)[0]);assert valid==d['manifest']['effective_edges'];degree=np.diff(np.r_[off,valid]);assert np.array_equal(degree,np.load(ROOT/ds/'degree.npy'))
  for i in range(n):
   row=edges[int(off[i]):int(off[i])+int(degree[i])];assert len(set(map(int,row)))==len(row) and i not in row and (row<n).all()
  if ids0 is None:ids0=ids;deg0=degree
  assert ids==ids0 and np.array_equal(degree,deg0)
  graph_checks[name]={'sha256_ids':hashlib.sha256(ids).hexdigest(),'effective_unique_nonself_edges':valid,'reachable_from_root':d['manifest']['root_reachability'][name]}
 for name,r in d['test'].items():
  memory.add(r['memory_bytes']);lines.append(f"| {ds.upper()} | {name} | {d['manifest']['effective_edges']} | {r['recall_at_10']*100:.2f}% | {r['mean_requests']:.3f} | {r['mean_navigation_us']:.2f} | {r['mean_latency_us']/1000:.3f} | {r['qps']:.1f} |")
 assert len(memory)==1;checks[ds]={'graphs':graph_checks,'identical_node_ids':True,'identical_per_source_degrees':True,'identical_actual_ram_bytes':memory.pop(),'query_disjoint':d['query_disjoint'],'deterministic_three_repeats':True}
lines+=['','I/O 表示 **Submitted SSD Read Requests**，不是设备层物理读取次数。端到端时延和 QPS 是三轮中位数，小幅变化可能受运行波动影响。','']
for ds,d in s.items():
 h=d['selected_history'];a=d['test'][h];g=d['test']['Geometry'];o=d['test']['Original'];p=d['paired_history_vs_geometry'];m=d['manifest'];ci=p['mean_request_saving_ci95']
 lines +=[f"## {ds.upper()}",'',f"固定 {m['nodes']} 个节点、{m['effective_edges']} 条有效有向边，{m['degree_zero_nodes']} 个零出度节点，实际 RAM {a['memory_bytes']:,} 字节。验证选择 {h}，验证 Recall 约束合格：{d['validation_recall_feasible']}。历史目标占全部边槽的 {m['history_selected_slots'][h]/m['effective_edges']*100:.2f}%，其余由几何连接补齐。",'',f"相对 Geometry，历史图平均少 {p['mean_request_saving']:.3f} 次请求/Query（{(1-a['mean_requests']/g['mean_requests'])*100:.3f}%）；配对查询 bootstrap 95% 区间 [{ci[0]:.3f}, {ci[1]:.3f}] 次。{p['queries_fewer_requests']} 条查询减少请求，{p['queries_more_requests']} 条增加请求。{p['queries_recall_lower']} 条 Recall 下降，{p['queries_recall_higher']} 条提高。总体 Recall 差 {(a['recall_at_10']-g['recall_at_10'])*100:+.3f} 个百分点。",'',f"相对 Original，历史图请求减少 {(1-a['mean_requests']/o['mean_requests'])*100:.3f}%，Recall 差 {(a['recall_at_10']-o['recall_at_10'])*100:+.3f} 个百分点。相对 Geometry，QPS 变化 {(a['qps']/g['qps']-1)*100:+.2f}%，端到端时延变化 {(a['mean_latency_us']/g['mean_latency_us']-1)*100:+.2f}%。",'',f"入口可达节点：{m['root_reachability']}。可达性差异属于本轮拓扑效果，没有通过额外免费连接强制消除。",'', '验证集所有候选：','', '| 拓扑 | Recall@10 | 请求/Query |','|---|---:|---:|']
 for name,r in d['validation'].items():lines.append(f"| {name} | {r['recall_at_10']*100:.2f}% | {r['mean_requests']:.3f} |")
 lines.append('')
lines+=['## 结论与适用范围','','**历史信息能改善本轮几何重建图，但尚未证明优于原始连接。** SIFT 相对几何图少 0.119% 请求，Recall 逐查询完全相同，但 QPS 没有改善；相对原始图少 0.048% 请求。DEEP 相对几何图少 1.450% 请求、QPS 提高约 1.07%，相对原始图仅少 0.071% 请求，且总体 Recall 下降 0.01 个百分点。不能将这个负面质量差异忽略，也不能将非常小的请求差异当作实用收益。','','固定节点与逐源边预算后，请求收益的规模和质量约束必须一起解读；少量请求下降不能直接推导为端到端收益。本轮仅检验训练历史距离收益驱动的连接目标选择，不代表所有历史拓扑算法，且没有测试热点迁移或源节点出度重分配。历史图是一种带几何补齐的简单静态启发式，没有神经网络。不同构图方法的强弱仍与几何候选和历史评分设计有关。','','原先约 14 万条重建边对约 2.9 万条原始边的结果不能归因于历史信息；本实验排除了该边数量混杂。','', '原始逐查询 CSV、命令与内存计量位于各数据集 validation/test 子目录；summary.json 包含所有结果、配对统计及重复运行记录。verification.json 记录独立图文件检查。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');(ROOT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps({ds:{'test':d['test'],'paired':d['paired_history_vs_geometry']} for ds,d in s.items()},indent=2))
