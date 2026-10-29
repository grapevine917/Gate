# 不同 L 的三方法比较

全部实际 C++ 搜索，同一批 1000 test query，K=10、beam=4、threads=4、cache=10000。56 个配置随机交错重复三次，共 168 次；延迟取 median，误差线为 min/max。包含真实模型推理与状态构造。

DiskANN 每个 L 一个点；Learned ET 使用 All17 的 8 档阈值，Top3 pruning 使用 5 档阈值。模型和阈值固定于原 L=100，其他 L 属于泛化实验；名义 unsafe-query/FNR 不代表其他 L 的实际风险保证。

![每个L阈值曲线](recall_cost_by_l.png)

![固定阈值跨L曲线](fixed_threshold_l_sweep.png)

第二张图固定 ET unsafe-query target=0.5%、pruning retain-FNR target=0.5%，点标注 L；两种风险定义不同，不声称风险相等，只展示固定配置跨 L 的行为。

| L | Method | Recall@10 % | SSD Reads/query | Expansions/query | Latency ms |
|---|---|---:|---:|---:|---:|
| 50 | DiskANN | 98.240 | 53.546 | 62.835 | 3.725 |
| 50 | Learned ET | 98.220 | 53.016 | 62.305 | 3.701 |
| 50 | Top3 pruning | 98.240 | 53.154 | 62.442 | 3.780 |
| 100 | DiskANN | 99.670 | 102.405 | 111.841 | 6.853 |
| 100 | Learned ET | 99.550 | 77.986 | 87.380 | 5.391 |
| 100 | Top3 pruning | 99.630 | 89.702 | 99.120 | 6.231 |
| 150 | DiskANN | 99.840 | 151.595 | 161.235 | 10.224 |
| 150 | Learned ET | 99.690 | 87.725 | 97.148 | 5.975 |
| 150 | Top3 pruning | 99.760 | 100.071 | 109.574 | 6.948 |
| 200 | DiskANN | 99.920 | 201.013 | 210.862 | 13.468 |
| 200 | Learned ET | 99.750 | 99.408 | 108.860 | 6.861 |
| 200 | Top3 pruning | 99.780 | 100.411 | 109.919 | 7.195 |

逐query 独立重算 Recall，三次重复的 IDs、读取与展开计数一致；pruning gate 计数恒等式通过。原始数据与原实验保留。SSD reads 为索引请求数，Latency 为当前主机预热状态。

复现：`.venv/bin/python experiments/sift1m/run_l_comparison.py`
