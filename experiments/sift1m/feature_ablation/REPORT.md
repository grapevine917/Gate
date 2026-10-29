# State feature ablation

五组重新训练 d→128→64→1 MLP；沿用8000/1000/1000互斥 query、train-only标准化、Adam、seed=42、batch=2048、lr=.001、alpha=.0001。验证logloss早停（最多60 epochs，patience=8，min_delta=.0001）。
Top5按上一实验 mean |SHAP| 排名；All17保持原始列顺序。阈值只在validation选择，测试集报告实际FNR。
特征排名曾在这一测试集上观察，因此本次属于探索性消融；训练和阈值选择均不使用test，但最终确认应使用新query。

| Model | Test AUC | SkipRate@val FNR .1% | actual Test FNR | SkipRate@val FNR 1% | actual Test FNR |
|---|---:|---:|---:|---:|---:|
| Top1 | 0.916151 | 0.866% | 0.0672% | 8.483% | 0.9173% |
| Top2 | 0.917814 | 1.507% | 0.1196% | 9.607% | 0.9656% |
| Top3 | 0.930133 | 4.774% | 0.1050% | 16.495% | 0.9572% |
| Top5 | 0.930182 | 5.319% | 0.1217% | 16.786% | 1.0180% |
| All17 | 0.930416 | 5.372% | 0.1112% | 17.007% | 1.0159% |

## Features

- Top1: candidate_queue_rank
- Top2: candidate_queue_rank;completed_expansions
- Top3: candidate_queue_rank;completed_expansions;visited_count
- Top5: candidate_queue_rank;completed_expansions;visited_count;candidate_pq_over_best;unexpanded_count
- All17: candidate_pq_distance;best_pq_distance;worst_pq_distance;candidate_queue_rank;queue_size;unexpanded_count;visited_count;completed_expansions;beam_step;beam_position;cached_node;best_exact_before_beam;has_best_exact;candidate_pq_over_best;candidate_pq_over_worst;search_L;beam_width

FNR对应保守retain标签误跳过率，不是在线Recall损失保证；SkipRate为样本加权比例。输入维度改变也改变首层参数量，隐藏层与其他训练设置固定。

## Figures

![Offline](offline_ablation.png)

![Online](online/recall_cost_curves.png)

[Online detailed report](online/REPORT.md)

## Interpretation

Top1 AUC=0.916151，但验证FNR目标0.1%时Test SkipRate仅0.866%。Top2仅小幅改善；加入visited_count的Top3达到AUC=0.930133，与All17差0.000283。Top5/All17在低FNR跳过率上仍有小幅增益，不能因gain高度集中就认定一个特征足够。

在线FNR目标1%：Top3 Recall=99.58%、Reads=83.825、Expansions=93.229；All17 Recall=99.59%、Reads=83.209、Expansions=92.623。本次Top3保留了All17大部分搜索收益，但没有稳定的latency优势。高FNR与低FNR结论不能仅从整体AUC推断。

All17本次重新训练；三份split的概率与原state-only MLP逐项一致。单seed、同一数据集的探索性消融不证明信息充分性或统计等效；选择特征时已观察这些test query的重要性排名。

## Reproduce

在learned-gate目录依次运行：

```bash
cmake --build DiskANN/build --target search_disk_index -j 4
.venv/bin/python experiments/sift1m/train_feature_ablation.py
.venv/bin/python experiments/sift1m/run_online_ablation.py
.venv/bin/python experiments/sift1m/plot_feature_ablation.py
```
