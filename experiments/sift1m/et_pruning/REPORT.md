# ET + Expansion Pruning

主组合为冻结的All17 learned ET + 冻结的Top3 expansion-pruning MLP。无需重训；ET在beam边界检查，p_safe>=tau_ET则结束整次搜索；未停止时逐候选使用p_retain<tau_P跳过展开，保留原beam slot。
固定SIFT1M、L=100、K=10、beam=4、threads=4、cache=10000，仍是原1000条test query。各配置全部重新交错随机计时3次，包含两个模型的真实C++推理与state构造，不拼接旧latency。

## Validation calibration and joint selection

Pruning沿用validation FNR targets=0.1%、0.25%、0.5%、1%、2%的五个已定tau_P；这些是原baseline候选标签上的名义FNR，不声称是组合轨迹上的实际FNR。
每个tau_P先在validation完整运行pruning-only，并在ET禁用（tau_ET=1.1）的同一路径上采集ET原生概率。新的safe标签表示已评分节点包含该档完整pruning-only最终top10；这样可校准ET相对pruning-only的额外风险，而不是误用完整baseline的旧状态/阈值。
Conditional ET按误停query targets=0%、0.1%、0.25%、0.5%、1%、2%、5%、10%重新校准，共40个联合配置。安全标签沿同一pruned path单调；原生float32 nextafter阈值排除首个超预算分数及ties。
条件ET安全只保留pruning-only端点，无法撤销pruning损失。两类FNR/误停率分母不同，也不相加；最终以实际Recall和成本评估组合。
40个组合全部真实运行validation。Validation baseline Recall=99.61%；按Recall损失预算 0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2 个百分点，选择最低SSD Reads/query，依次以较少Expansions、较高Recall、较保守风险参数打破并列。Latency不参与选择。
共锁定9个不同组合后才开始test；零Recall损失预算下没有可行联合点，不虚构配置。Validation预算不保证test满足相同Recall。

| selected validation Recall-loss budgets pp | nominal pruning FNR target | conditional ET query target | tau_P | tau_ET | validation Recall | validation Reads |
|---|---:|---:|---:|---:|---:|---:|
| [0.01] | 0.25% | 0% | 0.02587937 | 0.998252034 | 99.600% | 91.683 |
| [0.025] | 0.25% | 0.1% | 0.02587937 | 0.99777627 | 99.590% | 90.753 |
| [0.05] | 0.5% | 0.25% | 0.0366583429 | 0.989821017 | 99.560% | 79.609 |
| [0.1] | 1% | 0.5% | 0.0549282432 | 0.983060718 | 99.510% | 73.926 |
| [0.2] | 1% | 1% | 0.0549282432 | 0.9709993 | 99.460% | 68.242 |
| [0.3] | 1% | 2% | 0.0549282432 | 0.958931684 | 99.350% | 63.842 |
| [0.5] | 2% | 2% | 0.0822836608 | 0.956487477 | 99.290% | 62.600 |
| [0.75, 1] | 2% | 5% | 0.0822836608 | 0.920679629 | 98.970% | 55.074 |
| [2] | 2% | 10% | 0.0822836608 | 0.874577105 | 98.430% | 48.971 |

## Live test

所有36配置交错重复3次，共108次计时；报告median。无模型DiskANN、Budget ET（13档）、All17 ET（8档）、Top3 pruning（5档）均重新测量。

| Method | Configuration | Recall@10 | SSD Reads/query | Expansions/query | Latency ms | QPS | ET stops/query | Candidate skips/query | Additional unsafe ET queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiskANN | no models | 99.670% | 102.405 | 111.841 | 6.895 | 572.9 | 0.000 | 0.000 | — |
| Budget Early Termination | 20 reads | 79.930% | 20.306 | 29.488 | 1.397 | 2781.6 | 0.000 | 0.000 | — |
| Budget Early Termination | 40 reads | 96.090% | 40.411 | 49.676 | 2.722 | 1443.2 | 0.000 | 0.000 | — |
| Budget Early Termination | 60 reads | 98.500% | 60.458 | 69.784 | 4.068 | 968.5 | 0.000 | 0.000 | — |
| Budget Early Termination | 70 reads | 98.990% | 71.804 | 81.168 | 4.783 | 824.1 | 0.000 | 0.000 | — |
| Budget Early Termination | 80 reads | 99.220% | 80.510 | 89.908 | 5.381 | 733.2 | 0.000 | 0.000 | — |
| Budget Early Termination | 85 reads | 99.370% | 87.700 | 97.110 | 5.838 | 675.5 | 0.000 | 0.000 | — |
| Budget Early Termination | 90 reads | 99.470% | 91.765 | 101.186 | 6.132 | 643.9 | 0.000 | 0.000 | — |
| Budget Early Termination | 95 reads | 99.520% | 95.900 | 105.333 | 6.387 | 617.7 | 0.000 | 0.000 | — |
| Budget Early Termination | 100 reads | 99.570% | 99.896 | 109.331 | 6.642 | 594.7 | 0.000 | 0.000 | — |
| Budget Early Termination | 105 reads | 99.670% | 102.373 | 111.809 | 6.872 | 574.7 | 0.000 | 0.000 | — |
| Budget Early Termination | 110 reads | 99.670% | 102.405 | 111.841 | 6.837 | 577.7 | 0.000 | 0.000 | — |
| Budget Early Termination | 120 reads | 99.670% | 102.405 | 111.841 | 6.775 | 582.9 | 0.000 | 0.000 | — |
| Budget Early Termination | 140 reads | 99.670% | 102.405 | 111.841 | 6.884 | 573.7 | 0.000 | 0.000 | — |
| Learned ET All17 | unsafe-query target 0% | 99.660% | 94.448 | 103.872 | 6.363 | 620.6 | 0.474 | 0.000 | — |
| Learned ET All17 | unsafe-query target 0.1% | 99.660% | 92.986 | 102.408 | 6.337 | 623.0 | 0.529 | 0.000 | — |
| Learned ET All17 | unsafe-query target 0.25% | 99.620% | 83.463 | 92.869 | 5.708 | 690.8 | 0.801 | 0.000 | — |
| Learned ET All17 | unsafe-query target 0.5% | 99.550% | 77.986 | 87.380 | 5.326 | 740.6 | 0.889 | 0.000 | — |
| Learned ET All17 | unsafe-query target 1% | 99.320% | 68.661 | 78.032 | 4.635 | 850.3 | 0.968 | 0.000 | — |
| Learned ET All17 | unsafe-query target 2% | 99.230% | 64.006 | 73.355 | 4.342 | 907.1 | 0.984 | 0.000 | — |
| Learned ET All17 | unsafe-query target 5% | 98.850% | 55.422 | 64.742 | 3.888 | 1012.0 | 0.997 | 0.000 | — |
| Learned ET All17 | unsafe-query target 10% | 98.350% | 49.534 | 58.840 | 3.392 | 1158.8 | 1.000 | 0.000 | — |
| Learned Top3 Expansion Pruning | retain FNR target 0.1% | 99.640% | 97.050 | 106.481 | 6.707 | 588.9 | 0.000 | 5.352 | — |
| Learned Top3 Expansion Pruning | retain FNR target 0.25% | 99.640% | 93.286 | 102.710 | 6.475 | 610.1 | 0.000 | 9.116 | — |
| Learned Top3 Expansion Pruning | retain FNR target 0.5% | 99.630% | 89.702 | 99.120 | 6.291 | 627.3 | 0.000 | 12.688 | — |
| Learned Top3 Expansion Pruning | retain FNR target 1% | 99.580% | 83.825 | 93.229 | 5.739 | 687.1 | 0.000 | 18.552 | — |
| Learned Top3 Expansion Pruning | retain FNR target 2% | 99.450% | 75.691 | 85.073 | 5.258 | 749.9 | 0.000 | 26.662 | — |
| ET + Expansion Pruning | P nominal FNR 0.25%, conditional ET query 0% | 99.630% | 91.561 | 100.982 | 6.378 | 618.8 | 0.426 | 3.793 | 1 |
| ET + Expansion Pruning | P nominal FNR 0.25%, conditional ET query 0.1% | 99.630% | 90.606 | 100.026 | 6.257 | 631.2 | 0.488 | 3.188 | 1 |
| ET + Expansion Pruning | P nominal FNR 0.5%, conditional ET query 0.25% | 99.540% | 79.200 | 88.600 | 5.585 | 706.4 | 0.829 | 1.915 | 9 |
| ET + Expansion Pruning | P nominal FNR 1%, conditional ET query 0.5% | 99.440% | 73.443 | 82.823 | 5.145 | 766.7 | 0.893 | 2.395 | 11 |
| ET + Expansion Pruning | P nominal FNR 1%, conditional ET query 1% | 99.310% | 67.440 | 76.809 | 4.772 | 826.3 | 0.957 | 1.397 | 24 |
| ET + Expansion Pruning | P nominal FNR 1%, conditional ET query 2% | 99.220% | 63.333 | 72.680 | 4.349 | 906.2 | 0.978 | 0.905 | 33 |
| ET + Expansion Pruning | P nominal FNR 2%, conditional ET query 2% | 99.160% | 62.091 | 71.433 | 4.344 | 906.6 | 0.975 | 1.756 | 26 |
| ET + Expansion Pruning | P nominal FNR 2%, conditional ET query 5% | 98.780% | 54.442 | 63.760 | 3.826 | 1028.0 | 0.992 | 0.785 | 60 |
| ET + Expansion Pruning | P nominal FNR 2%, conditional ET query 10% | 98.140% | 48.281 | 57.584 | 3.347 | 1174.3 | 0.999 | 0.364 | 114 |

## Curves

![Recall / costs](online/recall_cost_curves.png)

图中展示固定validation选定组合的test曲线；仅连接实测点，无插值拟合或test调参。下排放大Recall>=99%。每个成本维度的实测非支配点及joint-vs-ET支配关系另存JSON，仅供结果描述。

## Interpretation

本轮组合没有明显改善单独All17 ET的整体Recall–成本权衡，多个联合点在高召回区域被单独组件的实测点支配。不能将两类gate的独立节省简单相加。
联合P nominal FNR=0.5% / conditional ET query target=0.25%：Recall=99.54%、Reads=79.200、Expansions=88.600、Latency=5.585ms；单独ET query target=0.5%：Recall=99.55%、Reads=77.986、Expansions=87.380、Latency=5.326ms。
相同Recall=99.63%时，联合P=0.25%/conditional ET=0.1%使用Reads=90.606，单独pruning FNR=0.5%为Reads=89.702；联合并未进一步降低读取。
中低召回段有联合点以较低Recall换取少量额外读取节省，但当前离散测量不能证明相同Recall下的统一优势。两个模型的收益可能重叠，额外candidate推理也会增加成本；本实验未做同tau_ET的完整因果消融，不能把所有差值归因于pruning或其推理开销。
单独ET沿用原baseline validation阈值，联合ET按pruned validation轨迹重校准。结论针对这两个冻结模型和本次校准/选择流程，尚不排除在pruned状态上重新训练ET后有更好组合。

## Verification

两模型同时禁用时全部test逐项复现baseline。关闭pruning时复现standalone ET；关闭ET时各档复现standalone pruning的ID、距离与展开/读取/gate计数。两个模型各自的Python/C++概率已审计。
40次validation joint及全部选定test joint逐query核验第一次停止位置、已完成展开数、ET推理次数和停止计数；safe ET停止逐项保留pruning-only ID与距离。额外unsafe停止恰好对应pruning-only端点变化。
所有计时repeat结果ID一致，Recall独立重算，expansions=gate_evaluated-gate_skipped核验，Budget ET overshoot<=3核验。Standalone controls保持前序搜索行为。
SSD Reads为索引请求数，不是设备物理读取；Latency为当前主机预热状态。测试query此前用于探索性实验，未参与本次阈值校准/组合选择。

## Reproduce

在learned-gate目录运行：

```bash
cmake --build DiskANN/build --target search_disk_index -j 4
.venv/bin/python experiments/sift1m/calibrate_et_pruning.py
.venv/bin/python experiments/sift1m/run_online_et_pruning.py
.venv/bin/python experiments/sift1m/plot_et_pruning.py
```
