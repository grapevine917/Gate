# 17-state-feature importance: XGBoost

使用已训练模型实际预测的前250轮，全部 held-out test（111841 expansion、1000 query），无重新训练/选参。
预测逐项复现保存概率，test AUC=0.930310。Gain 用平均分裂增益归一化；另保存 total_gain 和 split count。
Permutation：逐个特征随机打乱，重复10次，重要性为 baseline AUC − 打乱后AUC。也记录 logloss 增量。
SHAP：XGBoost 原生精确 TreeSHAP（tree-path-dependent），全部测试样本，mean(|SHAP|) 以 retain log-odds 衡量；正SHAP推向保留，负SHAP推向跳过。

| Feature | Gain % | Permutation ΔAUC ± SD | Mean abs SHAP | Gain rank | Perm rank | SHAP rank |
|---|---:|---:|---:|---:|---:|---:|
| candidate_queue_rank | 87.495 | 0.196492 ± 0.001355 | 2.109626 | 1 | 1 | 1 |
| completed_expansions | 4.970 | 0.131016 ± 0.001024 | 1.068108 | 2 | 2 | 2 |
| visited_count | 2.212 | 0.026482 ± 0.000349 | 0.503409 | 3 | 3 | 3 |
| candidate_pq_over_best | 1.076 | 0.000390 ± 0.000043 | 0.128961 | 5 | 6 | 4 |
| unexpanded_count | 0.471 | 0.001177 ± 0.000089 | 0.121303 | 8 | 4 | 5 |
| candidate_pq_over_worst | 0.607 | 0.000586 ± 0.000070 | 0.069906 | 6 | 5 | 6 |
| best_pq_distance | 1.291 | 0.000244 ± 0.000040 | 0.059883 | 4 | 7 | 7 |
| best_exact_before_beam | 0.491 | 0.000199 ± 0.000039 | 0.041497 | 7 | 8 | 8 |
| beam_position | 0.138 | 0.000029 ± 0.000013 | 0.033877 | 14 | 11 | 9 |
| worst_pq_distance | 0.258 | 0.000097 ± 0.000037 | 0.032758 | 10 | 9 | 10 |
| candidate_pq_distance | 0.206 | 0.000058 ± 0.000023 | 0.024296 | 11 | 10 | 11 |
| cached_node | 0.178 | 0.000022 ± 0.000018 | 0.014845 | 12 | 12 | 12 |
| queue_size | 0.458 | 0.000004 ± 0.000003 | 0.003825 | 9 | 13 | 13 |
| beam_step | 0.150 | 0.000000 ± 0.000000 | 0.000008 | 13 | 14 | 14 |
| has_best_exact | 0.000 | 0.000000 ± 0.000000 | 0.000000 | 15 | 15 | 15 |
| search_L | 0.000 | 0.000000 ± 0.000000 | 0.000000 | 15 | 15 | 15 |
| beam_width | 0.000 | 0.000000 ± 0.000000 | 0.000000 | 15 | 15 | 15 |

三种方法衡量不同量，数值不可直接相加或相互比较。Gain 受分裂选择影响；Permutation 打乱相关特征会产生不自然状态；TreeSHAP 的路径依赖背景不是因果解释。
search_L 和 beam_width 在本数据中为常量，零重要性不能说明改变L/beam没有效果。has_best_exact 是否被模型使用可从 split_count 和 SHAP 查看。
query_mean_abs SHAP及按query bootstrap区间保存在CSV；常规mean_abs按expansion加权。Permutation的SD只反映打乱随机性。
历史 feature_importance.csv 使用了导出booster所有280轮；本分析统一250轮，因此gain数值可能略有不同。
完整SHAP（最后一列是bias）、相关系数、10轮permutation数值和有效模型均已保留。
重现：`.venv/bin/python experiments/sift1m/analyze_state_features.py`。

## Interpretation

三种方法一致确认前三位：candidate_queue_rank、completed_expansions、visited_count。队列位置与搜索进度是该模型最主要的信号。
前三位合计占全体 mean(abs SHAP) 数值之和的 87.39%；这不是准确率或因果解释比例。
candidate_queue_rank gain份额87.49%，但completed_expansions打乱后的AUC下降也高达0.1310；仅看gain份额会低估模型对搜索进度的依赖。
candidate_queue_rank 与 completed_expansions 的 Spearman相关为0.9719；completed_expansions 与 beam_step 为0.99999987。逐列打乱会破坏这些关系，因此permutation ΔAUC不能相加，也不能据此断言删掉某特征后会下降同样多。
beam_step几乎冗余；search_L/beam_width因固定L=100/beam=4而是常量；has_best_exact未被当前模型用于任何split。这些结论只适用于本次模型与数据。
本模型中较高candidate_queue_rank和completed_expansions对应负SHAP（倾向skip）；visited_count的方向不同，须结合其他状态解释。
若要确定最小可用特征集合，需要重训删减特征的模型并验证低FNR/在线Recall，而不只是读取当前模型的重要性。
## Figures

![Importance comparison](importance_comparison.png)

![SHAP summary](shap_summary.png)

SHAP图仅抽样显示，每个query两条；排名与mean abs数值来自全部111841条。蓝色代表低特征值，红色代表高特征值。
