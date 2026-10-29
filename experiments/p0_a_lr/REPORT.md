# P0-A：Ours替换为LR重跑

固定SIFT1M原图索引、K10、L100、Beam4、10,000节点BFS缓存、同一1000条测试查询、4查询线程/MKL单线程。所有三组重新计时，各9次，顺序平衡交错。Ours为P0-C同一个冻结混合L训练17维LR，τ=0.0674629807472229，与P0-B LR版本共用权重。

阈值仅在原验证集baseline状态标定：缓存命中固定Keep，全部retain标签中最多1%被判Skip；这延续原P0-A的retain-FNR预算口径，不是最终Recall损失保证。Heuristic仍为rank>72且completed_expansions>100，并保护缓存命中。

| 方法 | Recall@10 | P95 ms | QPS | 直接请求/Query | 块层读取 KiB/Query | 展开/Query |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 99.67% | 7.662 | 558.2 | 102.405 | 409.620 | 111.841 |
| Heuristic | 99.52% | 6.911 | 619.6 | 91.745 | 366.980 | 101.180 |
| Ours | 99.61% | 7.407 | 647.2 | 85.359 | 341.436 | 94.795 |

相对本轮Vanilla：

| 方法 | Recall损失 pp | I/O减少 | P95降低 | QPS提升 |
|---|---:|---:|---:|---:|
| Heuristic | 0.15 | 10.41% | 9.80% | 11.00% |
| Ours | 0.06 | 16.65% | 3.33% | 15.94% |

## 重复范围与核验

- Vanilla：P95范围7.490–7.852ms；QPS范围544.3–568.6。
- Heuristic：P95范围6.652–7.047ms；QPS范围602.8–636.8。
- Ours：P95范围7.274–7.702ms；QPS范围630.9–668.7。
- Heuristic：配对查询bootstrap5000次，Recall损失95%区间[0.06, 0.26] pp；请求节省区间[10.512974999999999, 10.806024999999998] /Query。
- Ours：配对查询bootstrap5000次，Recall损失95%区间[0.0, 0.13] pp；请求节省区间[16.1239, 17.982049999999997] /Query。

P95是9次单轮P95的中位数；QPS为1000/四线程正式搜索墙钟时间的9次中位数。状态抽取、LR推理、直接读和搜索均计时，加载/缓存构建/128条热身/GT/输出不计。

全部重复ID与离散计数一致，Recall由ID与GT重新计算；请求长度为4096字节。Linux read_bytes与读取器实际字节在全部运行中一致：True。这是WSL2客户机块层流量，不是SSD内部NAND计数。关闭Gate后结果ID和hits/reads/expansions与Vanilla逐Query一致。原输入文件哈希未变。

## 与原MLP版的关系

原结果保留在../p0_a。下表来自此前独立计时，不是同时配对计时；LR为P0-C mixed-L权重，原MLP为L100训练权重，因此差异包含模型训练、阈值与运行时噪声，不能视为纯实现消融。纯实现消融见../p0_c。

| Ours版本 | Recall | P95 ms | QPS | 读请求/Query |
|---|---:|---:|---:|---:|
| 原MLP（历史计时） | 99.59% | 7.537 | 652.1 | 83.212 |
| LR（本轮） | 99.61% | 7.407 | 647.2 | 85.359 |

这是固定参数探索性实验，不是同Recall最优配置比较，后者见../p0_b_lr。查询曾被观察；验证代理风险不保证最终Recall。

复现：依次运行prepare.py、run.py、verify.py、report.py。原始记录见runs/、all_runs.json、summary.csv、confidence_intervals.json，配置与校准见PROTOCOL.json和LR_CALIBRATION.json。
