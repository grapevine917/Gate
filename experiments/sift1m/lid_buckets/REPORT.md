# SIFT1M LID 分桶

原始数据完整副本：`data/sift_original/`，SHA256 校验通过；原始实验输出保留。

使用 GT100 的真实邻居向量重算欧氏距离，LID(q) = -100 / Σ log(r_i/r_100)。不能直接将平方距离代入，否则 LID 会缩小一半。GT100 沿用项目现有 ground truth，其既有精确性抽查见 `../groundtruth_validation.json`。

全部 10000 个 query 按 (LID, query_id) 排序三等分；边界 ties 按 query_id 分配。LID 是局部几何难度代理，不保证逐 query 搜索难度单调。

- Q_low: 3334 条，LID [1.7385, 19.1364]，test 340 条。
- Q_mid: 3333 条，LID [19.1409, 24.6508]，test 309 条。
- Q_high: 3333 条，LID [24.6526, 48.1133]，test 351 条。

曲线仅使用现有 1000 条 test query，按桶独立计算 Recall@10 与平均 SSD read requests/query。ET=All17 learned ET，pruning=Top3 expansion pruning；ET+pruning 使用原 validation 选定的组合。保持模型和阈值，未按桶调参。三次原始运行的结果 ID 和 I/O 均一致，桶加权均值复现原整体结果。

![Recall–I/O](recall_io_by_lid.png)

`queries.csv` 保存所有 query 的 LID 和桶；`Q_*/` 保存 query、GT IDs、原始 ID 和 test ID；`test_per_query.csv` 保存逐 query 的 Recall/I/O；`summary.csv` 保存曲线数据。`groundtruth_ids.bin` 是带形状头的 ID 矩阵，不含距离。

复现：`.venv/bin/python experiments/sift1m/analyze_lid_buckets.py`
