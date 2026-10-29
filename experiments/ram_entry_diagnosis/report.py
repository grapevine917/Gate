from pathlib import Path
import numpy as np,json,csv
ROOT=Path(__file__).resolve().parent;s=json.loads((ROOT/'summary.json').read_text());bottom=json.loads((ROOT/'bottom_summary.json').read_text());top=json.loads((ROOT.parent/'ram_topology/summary.json').read_text());lines=['# RAM 四入口与底层入口价值诊断','', '每个数据集使用独立测试集的固定随机 200 条，三种拓扑使用相同查询。固定缓存、RAM budget=64 和底层 k=10/L=100/beam=4；所有四入口试验均保留原始 medoid 候选。Oracle 使用该查询的反事实结果，保证逐查询 Recall 不低于 current4；它是诊断机会空间，选择成本不免费，也不是最优组合证明。详见 [PROTOCOL.md](PROTOCOL.md)。','', '**结论：当前入口不是最优，但本轮未发现入口排序是主要瓶颈。** 已展开节点内重选四入口仅节省约0.5%请求；扩展缓存候选或纳入底层轨迹/GT反馈后，启发式Oracle约节省1%–1.5%。在固定L=100下，大量剩余成本仍属于底层精细搜索。拓扑影响候选发现，因此不能将入口排序与图结构当作互斥因素，更不能把有限候选的Oracle当作所有入口算法的上界。','', '## 固定四入口：同一候选集重选与扩大候选覆盖','', '| 数据集/拓扑 | 当前四入口 请求/Q | 已展开节点 Oracle4 | 扩展缓存候选 Oracle4 | 全缓存最近四入口 |','|---|---:|---:|---:|---:|'];verification={}
rank_data={};expansion_data={}
for ds in ['sift','deep']:
 name=top[ds]['selected_history'];rows=list(csv.DictReader(open(ROOT/ds/(name+'.candidates.csv'))))
 selected=[int(r['cache_distance_rank']) for r in rows if int(r['selected'])];missed=[int(r['cache_distance_rank']) for r in rows if int(r['valuable']) and int(r['expanded']) and not int(r['selected'])];good=[int(r['cache_distance_rank']) for r in rows if int(r['valuable'])]
 rank_data[ds]={'selected_cache_distance_rank_quantiles':np.quantile(selected,[.25,.5,.75]).tolist(),'valuable_cache_distance_rank_quantiles':np.quantile(good,[.25,.5,.75]).tolist(),'missed_expanded_valuable_cache_distance_rank_quantiles':np.quantile(missed,[.25,.5,.75]).tolist()}
 groups={}
 for r in csv.DictReader(open(ROOT/ds/(name+'.csv'))):groups.setdefault(r['method'],[]).append(r)
 expansion_data[ds]={m:{'mean_expansions':float(np.mean([int(r['expansions']) for r in rr])),'mean_requests':float(np.mean([int(r['submitted_requests']) for r in rr]))} for m,rr in groups.items() if m!='diagnostic_cost'}
 query_ids=set(np.load(ROOT/ds/'query_ids.npy'));lg=ROOT.parent.parent
 assert len(query_ids)==200 and not query_ids.intersection(np.load(ROOT.parent/'ram_navigation'/ds/'history_query_ids.npy')) and not query_ids.intersection(np.load(lg/'experiments/io_predictability'/ds/'validation/query_ids.npy'))
(ROOT/'distance_rank_diagnostic.json').write_text(json.dumps(rank_data,indent=2)+'\n');(ROOT/'expansion_diagnostic.json').write_text(json.dumps(expansion_data,indent=2)+'\n')
for ds,graphs in s.items():
 common=None;common_entries=None
 for name,d in graphs.items():
  a=d['aggregate'];lines.append(f"| {ds.upper()}/{name} | {a['current4']['mean_requests']:.3f} | {a['oracle_expanded4']['mean_requests']:.3f} | {a['oracle_pool4']['mean_requests']:.3f} | {a['cache_nearest4']['mean_requests']:.3f} |")
  rows=list(csv.DictReader(open(ROOT/ds/(name+'.csv'))));fixed=[(int(r['hits']),int(r['submitted_requests'])) for r in rows if r['method']=='cache_nearest4'];assert len(fixed)==200
  if common is None:common=fixed
  assert fixed==common
  entries=[(int(r['query']),int(r['position']),int(r['node'])) for r in csv.DictReader(open(ROOT/ds/(name+'.entries.csv'))) if r['method']=='cache_nearest4']
  if common_entries is None:common_entries=entries
  assert entries==common_entries
  for m in ['oracle_expanded4','oracle_pool4']:
   assert a[m]['seed_counts']==[4];assert d['paired_vs_current'][m]['queries_recall_lower']==0
 verification[ds]={'current4_prior_per_query_parity':all(d['current4_prior_per_query_parity'] for d in graphs.values()),'cache_nearest4_three_graphs_per_query_parity':True,'oracle_four_seeds_and_per_query_recall_and_requests_safe':True,'bottom_current4_repeat_parity':bottom[ds]['current4_repeat_parity']}
lines+=['', '所有请求指标为 Submitted SSD Read Requests；不等同设备物理读取。Oracle4 由单入口评分和八候选贪心组合产生，保留 current4 回退，没有穷举。扩展缓存候选包括 RAM 已展开节点与全缓存距离最近32节点，保证 Oracle_pool4 不差于 Oracle_expanded4。','', '## 当前历史图：入口重选与导航覆盖','']
for ds in ['sift','deep']:
 name=top[ds]['selected_history'];d=s[ds][name];a=d['aggregate'];p=d['paired_vs_current'];coverage=d['candidate_coverage'];now=a['current4']['mean_requests'];expanded=a['oracle_expanded4']['mean_requests'];pool=a['oracle_pool4']['mean_requests'];gain=now-pool;ranking=now-expanded
 lines +=[f"### {ds.upper()} / {name}",'',f"当前四入口：请求 {now:.3f}/Q，Recall {a['current4']['recall_at_10']*100:.2f}%。只在已展开节点中重选四入口：{expanded:.3f}/Q，减少 {ranking:.3f} 次（{ranking/now*100:.2f}%），配对95%区间 {p['oracle_expanded4']['request_saving_ci95']}。扩大至缓存最近32候选后：{pool:.3f}/Q，减少 {gain:.3f} 次（{gain/now*100:.2f}%），区间 {p['oracle_pool4']['request_saving_ci95']}。",'',f"嵌套启发式 Oracle 中，已展开节点重选解释约 {ranking/gain*100 if gain else 0:.1f}% 的请求机会，剩余来自新增候选及组合变化；这不是全图收益的因果分解。{coverage['queries_valuable_expanded_but_not_selected']}/200 条查询存在单入口反事实有益且已展开但未选中的节点；{coverage['queries_valuable_only_outside_expanded']}/200 条查询的候选池中，有益节点全部在已展开集合之外。单入口标签与四入口组合价值不能混同。",'',f"discovered4 请求 {a['discovered4']['mean_requests']:.3f}/Q，Recall {a['discovered4']['recall_at_10']*100:.2f}%；仅将已发现未展开节点纳入最近四个选择并未产生明显收益。全缓存精确最近四入口为 {a['cache_nearest4']['mean_requests']:.3f}/Q，Recall {a['cache_nearest4']['recall_at_10']*100:.2f}%；全缓存扫描平均耗时 {a['cache_nearest4']['mean_selection_us']:.1f} μs，当前 RAM 导航 {a['current4']['mean_nav_us']:.1f} μs。这些诊断计时不用于部署 QPS 比较。",'',f"oracle_expanded4 完整 Top-10 ID 集保持 {a['oracle_expanded4']['strict_same_results_queries']}/200，oracle_pool4 为 {a['oracle_pool4']['strict_same_results_queries']}/200；保证的是逐查询 Recall，不是完整结果相同。缓存候选单入口试验 {coverage['candidate_trials']:,} 次；组合等反事实合计平均 {a['diagnostic_cost']['mean_requests']:,.1f} 次提交请求/Q，不能拿其底层收益当作免费选择的部署收益。",'']
ranks=json.loads((ROOT/'distance_rank_diagnostic.json').read_text())
lines+=['## 距离排序与检索成本不完全一致','']
for ds,d in ranks.items():
 lines.append(f"{ds.upper()}：当前所选四入口的全缓存距离排名中位数为 {d['selected_cache_distance_rank_quantiles'][1]:.1f}；已经展开但没被选、且单入口反事实有益的节点排名中位数为 {d['missed_expanded_valuable_cache_distance_rank_quantiles'][1]:.1f}。这说明部分较远节点仍可降低底层请求，单纯精确距离不能完整表达入口价值。统计按 query-node 记录计数，不代表全图节点排名。")
lines.append('')
lines+=['## 将实际底层展开节点和 GT 节点纳入诊断候选','', '| 数据集 | 当前四入口 请求/Q | 底层 Oracle4 请求/Q | 减少比例 | Oracle4 Recall | Oracle4 平均缓存入口数 |','|---|---:|---:|---:|---:|---:|']
for ds,d in bottom.items():
 a=d['aggregate'];now=a['current4']['mean_requests'];o=a['oracle_bottom4'];lines.append(f"| {ds.upper()} | {now:.3f} | {o['mean_requests']:.3f} | {(1-o['mean_requests']/now)*100:.2f}% | {o['recall_at_10']*100:.2f}% | {o['mean_cached_seeds']:.2f}/4 |")
lines.append('')
for ds,d in bottom.items():
 a=d['aggregate'];o=a['oracle_bottom4'];gt=a['gt_nearest4'];c=d['coverage'];lines +=[f"{ds.upper()}：底层 Oracle4 的请求减少配对95%区间 {o['saving_ci95']} 次/Q，{o['queries_fewer_requests']}/200 条请求下降，{o['queries_recall_lower']} 条 Recall 下降，{o['queries_recall_higher']} 条 Recall 提高。完整结果保持 {o['strict_same_results_queries']}/200。平均入口中仅 {o['mean_ram_expanded_seeds']:.2f}/4 已被 RAM 展开。候选中 {c['queries_any_uncached_valuable_candidate']}/200 条查询存在未缓存有益节点，{c['queries_valuable_only_uncached']}/200 条有益候选全部未缓存。GT 最近四节点直接作入口：{gt['mean_requests']:.3f} 请求/Q、Recall {gt['recall_at_10']*100:.2f}%，其中 {gt['queries_recall_lower']} 条 Recall 低于 current4。GT 邻近不等于请求最省。",'']
expansion=json.loads((ROOT/'expansion_diagnostic.json').read_text())
lines+=['','## 底层精细搜索的剩余成本','']
for ds,d in expansion.items():
 lines.append(f"{ds.upper()}：无 RAM 入口时底层平均展开 {d['no_navigation']['mean_expansions']:.3f} 个节点；当前四入口为 {d['current4']['mean_expansions']:.3f}；缓存候选四入口 Oracle 为 {d['oracle_pool4']['mean_expansions']:.3f}。在 L=100 下，即使替换入口，底层仍展开约103个节点并提交约97–99次读取请求。结果指向精细搜索阶段占据大部分剩余成本；不能把 L=100 当成所有算法的理论最少展开数。")
lines.append('')
lines +=['底层候选包含 current4/no_navigation 实际展开节点、RAM 展开节点和 GT Top-10，并非全图穷举。它和缓存最近32候选池不互相包含，因此不能直接用两个 Oracle 的差当作缓存覆盖缺口。非缓存入口本身的正常底层读取已计入请求；只有离线选择入口的试验成本没有计入每个结果行，另有 diagnostic_cost 记录。','', '## 如何判断瓶颈','', '相同缓存节点及固定四入口交接后，三张 RAM 图的底层结果逐查询完全一致。这说明 RAM 拓扑通过“找到并交接哪些入口”影响底层检索；入口选择与拓扑发现不是互斥解释。已展开集合内 Oracle4 的收益检验入口排序；扩大候选池的收益检验发现与覆盖。只有在同一已展开集合内重选取得主要收益时，才有证据把主要瓶颈归于入口排序。当前数据需要同时考虑排序和覆盖，不能直接宣布图结构已不重要。','','完整原始候选结果与所选入口 ID 在各数据集 *.candidates.csv / *.entries.csv；summary.json 和 bottom_summary.json 提供统计，verification.json 保存对照检查。本轮200查询诊断不代替完整1000查询的上一轮拓扑评估，也不证明 Oracle 规则能在独立部署查询中实现收益。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');(ROOT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps({ds:{'history':s[ds][top[ds]['selected_history']]['aggregate'],'coverage':s[ds][top[ds]['selected_history']]['candidate_coverage'],'bottom':bottom[ds]} for ds in s},indent=2))
