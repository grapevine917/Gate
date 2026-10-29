# 按收敛速度比较 ET 与 pruning

全部 10000 query 按基线最终 top10 均完成评分时的展开前缀长度三等分为 fast / mid / slow，ties 按 query ID。逐 query 校验最终 top10 在轨迹中的成员，safe beam boundary 完成数不早于收敛点。原始数据和原实验均保留。

- fast: {'count': 3334, 'test_count': 348, 'convergence_min': 18, 'convergence_max': 33}
- mid: {'count': 3333, 'test_count': 303, 'convergence_min': 33, 'convergence_max': 43}
- slow: {'count': 3333, 'test_count': 349, 'convergence_min': 43, 'convergence_max': 118}

![分桶曲线](recall_io_by_convergence.png)

曲线仅使用原 1000 条 test query 和已冻结 validation 阈值。ET 为 All17，pruning 为 Top3。

![二维分析](lid_delta_cost.png)

二维图每个点是一条 test query，目标 Recall 为该 query 自身的 DiskANN 基线 Recall@10；ET 与 pruning 各自在实测配置中选取 **Recall 完全等于目标** 的最低 I/O。ΔCost = Cost_ET − Cost_Pruning，负数表示 ET 更省，正数表示 pruning 更省。不同 query 的目标 Recall 可以不同，图不表示所有点均为统一 100% Recall。不插值，不外推，不以平均 Recall 相等冒充逐 query Recall 相等。

成功对齐 997/1000，排除 3 条（见 unmatched_queries.json）。最低成本选择使用 query 的评测结果，仅用于离线描述，不是部署策略，不能用于声称无需 oracle 即可获得相同收益。配置网格不同也会影响结果。

- fast: matched=348, mean ΔCost=-26.195 reads/query
- mid: matched=303, mean ΔCost=-25.465 reads/query
- slow: matched=346, mean ΔCost=-21.751 reads/query

`queries.csv` 含全部 query 的收敛点、总展开数、收敛比例和 LID；`summary.csv` 为分桶曲线数据；`recall_matched_per_query.csv` 记录二维图的目标 Recall、两种配置和成本。

复现：`.venv/bin/python experiments/sift1m/analyze_convergence.py`
