from pathlib import Path
import json
R=Path(__file__).resolve().parent;s=json.loads((R/'summary.json').read_text());p=json.loads((R/'PROTOCOL.json').read_text());v=json.loads((R/'verification.json').read_text());ci=json.loads((R/'confidence_intervals.json').read_text());b=s[0]
L=['# P0-A：Ours替换为LR重跑','',f'固定SIFT1M原图索引、K10、L100、Beam4、10,000节点BFS缓存、同一1000条测试查询、4查询线程/MKL单线程。所有三组重新计时，各9次，顺序平衡交错。Ours为P0-C同一个冻结混合L训练17维LR，τ={p["gate_threshold"]:.17g}，与P0-B LR版本共用权重。','', '阈值仅在原验证集baseline状态标定：缓存命中固定Keep，全部retain标签中最多1%被判Skip；这延续原P0-A的retain-FNR预算口径，不是最终Recall损失保证。Heuristic仍为rank>72且completed_expansions>100，并保护缓存命中。','', '| 方法 | Recall@10 | P95 ms | QPS | 直接请求/Query | 块层读取 KiB/Query | 展开/Query |','|---|---:|---:|---:|---:|---:|---:|']
for r in s:L.append(f'| {r["method"]} | {r["recall_pct"]:.2f}% | {r["p95_ms"]:.3f} | {r["qps"]:.1f} | {r["requests_per_query"]:.3f} | {r["linux_read_bytes_per_query"]/1024:.3f} | {r["expansions_per_query"]:.3f} |')
L+=['','相对本轮Vanilla：','', '| 方法 | Recall损失 pp | I/O减少 | P95降低 | QPS提升 |','|---|---:|---:|---:|---:|']
for r in s[1:]:L.append(f'| {r["method"]} | {b["recall_pct"]-r["recall_pct"]:.2f} | {(1-r["requests_per_query"]/b["requests_per_query"])*100:.2f}% | {(1-r["p95_ms"]/b["p95_ms"])*100:.2f}% | {(r["qps"]/b["qps"]-1)*100:.2f}% |')
L+=['','## 重复范围与核验','']
for r in s:L.append(f'- {r["method"]}：P95范围{r["p95_ms_min"]:.3f}–{r["p95_ms_max"]:.3f}ms；QPS范围{r["qps_min"]:.1f}–{r["qps_max"]:.1f}。')
for name,c in ci.items():L.append(f'- {name}：配对查询bootstrap5000次，Recall损失95%区间{c["recall_loss_pp_95CI"]} pp；请求节省区间{c["requests_saved_95CI"]} /Query。')
L+=['', 'P95是9次单轮P95的中位数；QPS为1000/四线程正式搜索墙钟时间的9次中位数。状态抽取、LR推理、直接读和搜索均计时，加载/缓存构建/128条热身/GT/输出不计。','',f'全部重复ID与离散计数一致，Recall由ID与GT重新计算；请求长度为4096字节。Linux read_bytes与读取器实际字节在全部运行中一致：{v["linux_read_bytes_equal_reader_bytes_all_runs"]}。这是WSL2客户机块层流量，不是SSD内部NAND计数。关闭Gate后结果ID和hits/reads/expansions与Vanilla逐Query一致。原输入文件哈希未变。','', '## 与原MLP版的关系','', '原结果保留在../p0_a。下表来自此前独立计时，不是同时配对计时；LR为P0-C mixed-L权重，原MLP为L100训练权重，因此差异包含模型训练、阈值与运行时噪声，不能视为纯实现消融。纯实现消融见../p0_c。','', '| Ours版本 | Recall | P95 ms | QPS | 读请求/Query |','|---|---:|---:|---:|---:|']
old=next(r for r in json.loads((R.parent/'p0_a/summary.json').read_text()) if r['method']=='Ours');new=s[2]
for name,r in [('原MLP（历史计时）',old),('LR（本轮）',new)]:L.append(f'| {name} | {r["recall_pct"]:.2f}% | {r["p95_ms"]:.3f} | {r["qps"]:.1f} | {r["requests_per_query"]:.3f} |')
L+=['','这是固定参数探索性实验，不是同Recall最优配置比较，后者见../p0_b_lr。查询曾被观察；验证代理风险不保证最终Recall。','', '复现：依次运行prepare.py、run.py、verify.py、report.py。原始记录见runs/、all_runs.json、summary.csv、confidence_intervals.json，配置与校准见PROTOCOL.json和LR_CALIBRATION.json。']
(R/'REPORT.md').write_text('\n'.join(L)+'\n');print('\n'.join(L[:13]))
