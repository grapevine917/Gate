# L=150/200 重新训练与校准

沿用原 8000/1000/1000 query 划分，各 L 分别训练 Top3 pruning 与 All17 ET。结构均为 d→128→64→1，seed=42、batch=2048、Adam lr=.001、alpha=.0001；train-only 标准化，validation logloss 早停（最多60 epochs、patience=8、min_delta=.0001）。

Pruning 标签为基线零 admission 且节点不属于最终 top10 的安全跳过；各 L 全量 oracle mask 实际搜索核验 ID 和距离完全一致。训练状态逐 query 重放原队列验证。ET 使用真实 beam boundary，safe 标签表示全部最终基线 top10 已评分，并用最后 final-top10 beam 独立核验。

ET 的 8 档 unsafe-query 目标、pruning 的 5 档 retain-FNR 目标分别按对应 L 的 validation 概率重新校准（ET 用全量 native float32，pruning 沿用 Python 概率校准并审计首条 validation query 的原生概率）；C++/Python 概率、原生状态和禁用 gate 时的 baseline parity 均核验。两类风险定义不同，不等价于 Recall 损失保证。

原 L100 模型与阈值保留，并与新模型交错重新计时；同一1000 test query，L=150/200、K=10、beam=4、threads=4、cache=10000。54配置×3次=162次实搜，median 延迟含推理与状态构造。三次 ID、距离、读取与展开数一致，Recall 独立重算；冻结 controls 复现此前结果。

![重训比较](recall_cost_retrained.png)

![高召回细节](recall_cost_retrained_high_recall.png)

## 固定名义0.5%风险档

这张表用于查看对应档位变化，ET 与 pruning 风险含义不同，不能当作等 Recall 比较。

| L | Method | Version | Recall % | Reads | Expansions | Latency ms |
|---|---|---|---:|---:|---:|---:|
| 150 | DiskANN | baseline | 99.840 | 151.595 | 161.235 | 10.177 |
| 150 | Learned ET | frozen_L100 | 99.690 | 87.725 | 97.148 | 5.875 |
| 150 | Learned ET | retrained | 99.760 | 94.112 | 103.571 | 6.379 |
| 150 | Top3 pruning | frozen_L100 | 99.760 | 100.071 | 109.574 | 7.252 |
| 150 | Top3 pruning | retrained | 99.820 | 133.216 | 142.833 | 9.223 |
| 200 | DiskANN | baseline | 99.920 | 201.013 | 210.862 | 13.583 |
| 200 | Learned ET | frozen_L100 | 99.750 | 99.408 | 108.860 | 6.777 |
| 200 | Learned ET | retrained | 99.890 | 129.324 | 138.953 | 8.623 |
| 200 | Top3 pruning | frozen_L100 | 99.780 | 100.411 | 109.919 | 7.081 |
| 200 | Top3 pruning | retrained | 99.920 | 173.775 | 183.579 | 12.122 |

## 实测支配关系

| L | Version | Cost | Pruning被ET支配点数 |
|---|---|---|---:|
| 150 | frozen_L100 | ssd_reads_per_query | 2/5 |
| 150 | frozen_L100 | expansions_per_query | 2/5 |
| 150 | frozen_L100 | latency_us | 4/5 |
| 150 | retrained | ssd_reads_per_query | 2/5 |
| 150 | retrained | expansions_per_query | 2/5 |
| 150 | retrained | latency_us | 2/5 |
| 200 | frozen_L100 | ssd_reads_per_query | 0/5 |
| 200 | frozen_L100 | expansions_per_query | 0/5 |
| 200 | frozen_L100 | latency_us | 1/5 |
| 200 | retrained | ssd_reads_per_query | 5/5 |
| 200 | retrained | expansions_per_query | 5/5 |
| 200 | retrained | latency_us | 5/5 |

这里只统计实测点，不插值、不用test选择部署阈值。小幅延迟差异应结合重复范围解释。新实验同时改变训练与校准，不能单独归因于重新训练；未添加仅重新校准的中间消融。原测试集已用于探索，因此结果仍属探索性验证。

复现：`.venv/bin/python experiments/sift1m/retrain_high_l.py`

## 相同 Recall 的实测比较

只比较完全相同的整体 Recall；同 Recall 多个配置取该方法最低 Reads 的点，属于离线描述，不用于选择部署阈值。

| L | Version | Recall % | ET Reads | Pruning Reads | Pruning额外Reads % |
|---|---|---:|---:|---:|---:|
| 150 | frozen_L100 | 99.79 | 104.816 | 108.829 | 3.83 |
| 150 | retrained | 99.81 | 115.314 | 124.498 | 7.96 |
| 200 | frozen_L100 | 99.81 | 114.984 | 110.081 | -4.26 |
| 200 | retrained | 99.89 | 129.324 | 144.184 | 11.49 |
| 200 | retrained | 99.91 | 136.617 | 161.291 | 18.06 |
| 200 | retrained | 99.92 | 164.399 | 173.775 | 5.70 |

这一实验比较“重训+对应L重新校准”与“L100冻结模型+冻结阈值”的完整流程，不能拆出训练和校准各自的因果贡献。
重新校准后，名义风险档的成本/Recall位置明显变化；应按相同Recall和实测曲线比较，而非按同风险百分比直接比较。

## 结果解释

重训并重新校准后，L=200 的5个 pruning 实测点全部被 ET 的实测点在 Recall–Reads、Recall–Expansions 和 Recall–Latency 上支配；冻结模型时，Reads/Expansions 上没有 pruning 点被 ET 支配。上一轮“L增大时pruning相对竞争力增强”的判断仅对冻结L100模型与阈值成立，本补实验不支持将其推广到按L适配的模型。

L=200 相同Recall=99.89%、99.91%、99.92%时，pruning分别比ET多读取约11.49%、18.06%、5.70%。L=150在相同Recall=99.81%时，pruning多读取约7.96%；但pruning最高达到99.83%，超过本次ET最高99.81%，因此L=150仍有更高召回的独有区间，不能称ET全面支配。

重新校准使pruning的原5档FNR点整体更保守、更高Recall，尤其L=200；这既是召回恢复，也是成本增加。不能把变贵直接解释为模型变差，或把风险档位置变化归因于训练本身。后续若要拆因果，需要“冻结模型+对应L仅重新校准”的中间对照。
