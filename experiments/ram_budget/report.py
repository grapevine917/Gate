from pathlib import Path
import json,csv,numpy as np
ROOT=Path(__file__).resolve().parent;s=json.loads((ROOT/'summary.json').read_text());lines=['# RAM 导航预算：读取请求与 CPU 权衡','', '固定 RAM 节点/有效边/逐源出度与底层图，扫描展开预算及历史图队列容量。验证和独立测试各1,000条；配置在测试前冻结，主测试五轮交错。导航 CPU 使用线程 CPU 时钟，另记录墙钟；读取为 Submitted SSD Read Requests。详见 [PROTOCOL.md](PROTOCOL.md)。','', '## 验证选定配置的独立测试','', '| 数据集 | 拓扑 | 展开预算/队列 | Recall@10 | 请求/Q | 导航CPU μs | 端到端 ms | QPS |','|---|---|---|---:|---:|---:|---:|---:|'];verification={};overview={}
for ds,d in s.items():
 baseline=next(r for k,r in d['test'].items() if r['budget']==0);overview[ds]={}
 lines.append(f"| {ds.upper()} | 同缓存无导航 | 0/64 | {baseline['recall_at_10']*100:.2f}% | {baseline['mean_requests']:.3f} | 0 | {baseline['mean_latency_us']/1000:.3f} | {baseline['qps']:.1f} |")
 for m,selection in d['selected'].items():
  ref=d['test'][f'{m}/64/64'];chosen=d['test'][selection['key']]
  for label,r in [('参考64',ref),('验证选定',chosen)]:lines.append(f"| {ds.upper()} | {m} {label} | {r['budget']}/{r['queue']} | {r['recall_at_10']*100:.2f}% | {r['mean_requests']:.3f} | {r['mean_navigation_cpu_us']:.2f} | {r['mean_latency_us']/1000:.3f} | {r['qps']:.1f} |")
  rr=d['raw_test_runs'][f'{m}/64/64'];cc=d['raw_test_runs'][selection['key']];qps_change=[c['qps']/r['qps']-1 for r,c in zip(rr,cc)];lat_change=[c['mean_latency_us']/r['mean_latency_us']-1 for r,c in zip(rr,cc)];quality_target=max(ref['recall_at_10'],baseline['recall_at_10']);quality_ok=chosen['recall_at_10']>=quality_target-1e-12
  overview[ds][m]={'selected':selection['key'],'recall_quality_feasible_on_test':quality_ok,'quality_target_test':quality_target,'request_saving_ratio_vs64':1-chosen['mean_requests']/ref['mean_requests'],'navigation_cpu_delta_us':chosen['mean_navigation_cpu_us']-ref['mean_navigation_cpu_us'],'backend_latency_delta_us':chosen['mean_backend_us']-ref['mean_backend_us'],'total_latency_delta_us':chosen['mean_latency_us']-ref['mean_latency_us'],'qps_change_ratio':chosen['qps']/ref['qps']-1,'qps_change_range': [min(qps_change),max(qps_change)],'qps_change_runs':qps_change,'latency_change_range':[min(lat_change),max(lat_change)],'paired_queries':d['paired_selected_vs64'][m]}
 memory={r['memory']['cache_and_navigation_bytes'] for r in d['test'].values()};assert len(memory)==1
 verification[ds]={'same_actual_cache_and_navigation_bytes':int(next(iter(memory))),'queue_storage512_all_configs':all(r['memory']['navigation_queue_storage_capacity']==512 for r in d['test'].values()),'five_repeats_deterministic':all(r['repeats']==5 for r in d['test'].values()),'reference64_prior_query_parity':d['reference64_prior_query_parity'],'frozen_before_test':d['frozen_before_test']}
lines+=['','## 相对64/64的收益与质量','', '| 数据集/拓扑 | 请求减少 | CPU变化 μs | 端到端变化 μs | QPS变化 | 同轮QPS变化范围 | 测试总体Recall约束 |','|---|---:|---:|---:|---:|---|---|']
for ds,graphs in overview.items():
 for m,r in graphs.items():lines.append(f"| {ds.upper()}/{m} | {r['request_saving_ratio_vs64']*100:.3f}% | {r['navigation_cpu_delta_us']:+.2f} | {r['total_latency_delta_us']:+.2f} | {r['qps_change_ratio']*100:+.2f}% | [{r['qps_change_range'][0]*100:+.2f}%, {r['qps_change_range'][1]*100:+.2f}%] | {r['recall_quality_feasible_on_test']} |")
lines+=['','总体 Recall 约束为不低于本图64/64及同缓存无导航两者的较高值；不等于每查询Recall不下降。五轮方向范围比仅比较中位数更能揭示小收益的波动。','']
for ds,d in s.items():
 h=d['history_method'];ref=d['test'][f'{h}/64/64'];chosen=d['test'][d['selected'][h]['key']];minimum=d['test'][d['selected'][h]['minimum_requests_key']]
 lines +=[f"## {ds.upper()}：历史图的预算曲线（队列64）",'', '| 最大展开预算 | 实际展开/Q | 距离计算/Q | 导航CPU μs | 请求/Q | Recall@10 | 达预算查询数 |','|---:|---:|---:|---:|---:|---:|---:|']
 for b,r in d['history_test_curve'].items():lines.append(f"| {b} | {r['mean_expansions']:.2f} | {r['mean_distances']:.2f} | {r['mean_navigation_cpu_us']:.2f} | {r['mean_requests']:.3f} | {r['recall_at_10']*100:.2f}% | {r['queries_budget_exhausted']} |")
 lines +=['',f"验证选定历史图配置 {chosen['budget']}/{chosen['queue']}；请求最少的验证合格配置 {minimum['budget']}/{minimum['queue']}。后者测试请求 {minimum['mean_requests']:.3f}/Q、导航CPU {minimum['mean_navigation_cpu_us']:.2f} μs、端到端 {minimum['mean_latency_us']/1000:.3f} ms、QPS {minimum['qps']:.1f}，Recall {minimum['recall_at_10']*100:.2f}%。最低请求不自动等于最短端到端时延。",'', '预算曲线点只用于诊断；除进入主对照的配置外，其CPU计时只有一轮，不依据测试曲线重选配置。','']
 for m,p in d['paired_selected_vs64'].items():lines.append(f"{m}：选定配置相对64/64平均少 {p['mean_request_saving']:.3f} 次请求，配对95%区间 {p['request_saving_ci95']}；Recall 差 {p['recall_delta_pp']:+.3f} 个百分点，区间 {p['recall_delta_pp_ci95']}；{p['queries_recall_lower']}/1000 条下降，{p['queries_recall_higher']}/1000 条提高。")
 lines +=['', '历史图验证集队列容量敏感性：','', '| 展开预算 | 队列 | 导航CPU μs | 请求/Q | Recall@10 |','|---:|---:|---:|---:|---:|']
 for k,r in sorted(d['validation'].items(),key=lambda x:(x[1]['budget'],x[1]['queue'])):
  if r['method']==h and r['budget'] in [16,64,256]:lines.append(f"| {r['budget']} | {r['queue']} | {r['mean_navigation_cpu_us']:.2f} | {r['mean_requests']:.3f} | {r['recall_at_10']*100:.2f}% |")
 lines.append('')
 # Export full curves for plotting/reuse, keeping validation and test measurements separate.
 for split,rows in [('validation',d['validation'].values()),('history_test_curve',d['history_test_curve'].values())]:
  fields=['method','budget','queue','mean_expansions','mean_distances','mean_navigation_cpu_us','mean_navigation_us','mean_requests','recall_at_10','mean_latency_us','qps','repeats'];p=ROOT/ds/(split+'.csv')
  with open(p,'w') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
conditional=json.loads((ROOT/'current_recall_summary.json').read_text())
lines+=['## 保持当前导航Recall的补充权衡','', '主选择还要求达到no-navigation的验证Recall；DEEP某些导航配置在验证上低一个命中，因此主约束可能仅选中关闭导航或很小预算。补充选择仅要求不低于当前64/64验证Recall，仍在测试前冻结并用五轮独立测试；下表同时检查更强no-navigation测试质量，避免混称不同质量目标。','', '| 数据集 | 配置 | 请求/Q | 导航CPU μs | 端到端 ms | QPS | Recall@10 |','|---|---|---:|---:|---:|---:|---:|']
for ds,d in conditional.items():
 for label,r in [('当前',d['reference_test']),('补充验证选择',d['selected_test'])]:lines.append(f"| {ds.upper()} | {label} {r['budget']}/{r['queue']} | {r['mean_requests']:.3f} | {r['mean_navigation_cpu_us']:.2f} | {r['mean_latency_us']/1000:.3f} | {r['qps']:.1f} | {r['recall_at_10']*100:.2f}% |")
lines.append('')
for ds,d in conditional.items():
 a=d['reference_test'];b=d['selected_test'];p=d['paired_vs64'];lines.append(f"{ds.upper()}：导航CPU变化 {(b['mean_navigation_cpu_us']/a['mean_navigation_cpu_us']-1)*100:+.2f}%，请求变化 {(b['mean_requests']/a['mean_requests']-1)*100:+.3f}%；QPS中位数变化 {d['qps_change_ratio']*100:+.2f}%，五轮同轮变化范围 [{d['qps_change_range'][0]*100:+.2f}%, {d['qps_change_range'][1]*100:+.2f}%]。测试相对当前Recall约束满足 {d['quality_feasible_vs_current_on_test']}，相对no-navigation总体Recall约束满足 {d['quality_feasible_vs_no_navigation_on_test']}。逐查询 {p['queries_recall_lower']} 条Recall下降、{p['queries_recall_higher']} 条提高；总体保持不等于逐查询安全。")
 verification[ds]['conditional_five_repeats_deterministic']=d['reference_test']['repeats']==5 and d['selected_test']['repeats']==5
 h=s[ds]['history_method'];common=None;plateau={}
 for budget in [128,256,512]:
  key=f'{h}/{budget}/64';prefix=s[ds]['raw_test_runs'][key][0]['prefix'] if key in s[ds]['raw_test_runs'] else str(ROOT/ds/'test'/f'curve_{h}_b{budget}_q64');rr=list(csv.DictReader(open(prefix+'.csv')));values=[(int(r['hits']),int(r['submitted_requests'])) for r in rr]
  if common is None:common=values
  plateau[str(budget)]=values==common
 verification[ds]['budget128_256_512_per_query_same_recall_and_requests']=plateau
lines+=['', '**判断：目前没有发现稳定优于64/64的端到端配置。** 固定队列64的历史图在128左右饱和，SIFT及DEEP从64增大预算的额外请求收益分别约0.04%、0.11%。提高到256/512不会继续降低同队列请求。DEEP的64/16是CPU更省、SSD请求更多的可选权衡点，尚未出现稳定QPS收益。SIFT队列256虽进一步降低请求，但导航CPU显著增加、端到端更慢且Recall略降，不宜只按最少请求选择。','']
lines+=['## 解释范围','', '有利配置必须同时考虑质量、导航CPU与完整端到端成本。更高展开预算若已被队列耗尽限制，就不意味着实际更多工作；更大队列则可能额外展开大量精确距离很近的节点，但底层请求减少未必补偿导航开销。验证选择受有限短名单和计时波动影响，本轮没有证明全局最优。若五轮收益方向不一致，应报告配置近似持平，而不能宣称稳定加速。','','所有配置保留相同512容量队列存储，以有效容量控制搜索；完整费用计入相同RAM上限。没有把增加工作区当免费RAM，也没有用测试反馈修改配置。原始逐查询CSV/命令/内存计量及五轮时间均保留；tradeoff_summary.json、verification.json提供判断依据。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');(ROOT/'tradeoff_summary.json').write_text(json.dumps(overview,indent=2)+'\n');(ROOT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(overview,indent=2))
