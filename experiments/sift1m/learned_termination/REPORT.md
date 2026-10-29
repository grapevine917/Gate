# Learned early termination baseline

SIFT1M，固定L=100、K=10、beam=4、threads=4、cache=10000；与前序相同1000条held-out test query。训练/验证query仍为8000/1000，互不重叠。
Top3: candidate_queue_rank、completed_expansions、visited_count。All17: 原始17维pre_io_state。各模型d→128→64→1，ReLU/ReLU/sigmoid、BCE Adam、batch=2048、lr=.001、alpha=.0001、seed=42；train-only标准化，validation logloss早停（最多60轮、patience=8、min_delta=.0001）。
新数据在完整baseline真实轨迹的beam边界采集：至少K个节点已精确评分且队列仍有未展开候选。safe=1 iff 全部baseline最终top10已精确评分；此时返回当前top10可逐项保留完整baseline结果。用既有展开metadata独立核验标签，全部state与原pre_io数据一致。
y是safe-to-stop，不复用候选级pruning标签；最终节点ID/GT/未来状态不进入模型。ET在新beam选择和SSD请求提交前判定，p_safe>=tau就break整个搜索；未命中阈值则完整处理下一beam。至少K exact-score guard避免不完整topK。

| Model | Validation safe-stop AUC | Test safe-stop AUC | best epoch |
|---|---:|---:|---:|
| ET Top3 | 0.959068 | 0.950089 | 34 |
| ET All17 | 0.963826 | 0.956429 | 27 |

Data boundaries: {"train": {"queries": 8000, "boundaries": 200147, "safe": 138088, "unsafe": 62059}, "validation": {"queries": 1000, "boundaries": 25040, "safe": 17193, "unsafe": 7847}, "test": {"queries": 1000, "boundaries": 25035, "safe": 17072, "unsafe": 7963}}

## Calibration

ET阈值使用C++原生validation概率。每条query的unsafe前缀最大p_safe决定是否会误停；safe标签沿query单调。选择最小float32阈值使validation误停query数不超过目标预算；严格排除首个超预算分数及ties。风险目标为0%、0.1%、0.25%、0.5%、1%、2%、5%、10%，1000 query下0.25%预算向下取整为2条。
Top3 expansion pruning沿用validation候选级retain FNR目标0.1%、0.25%、0.5%、1%、2%。两类风险分母和标签不同，不能将相同百分比解释为相同风险；使用真实Recall–成本曲线比较。验证误停为零不保证测试零误停。
BudgetEarlyTermination使用现有io_limit，预算20、40、60、70、80、85、90、95、100、105、110、120、140个索引读取，在beam边界检查，最多超出3个读取。无模型DiskANN不带任何gate。

## Online measurements

全部35配置交错随机重复3次，共105次测量；所有模型重新计时，未拼接旧latency。报告median mean-query latency/QPS，特征构造、实际native模型推理或预算判断计入；模型加载不计入，主机OS缓存为预热状态。
ET Top3/All17统一构造边界17维state再选择输入；不加载q/v feature表。Expansion pruning统一重建candidate state，每候选实时MLP推理。

| Method | Parameter | tau | Recall@10 | Reads/query | Expansions/query | Latency ms | QPS | Test stopped queries | Test unsafe queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiskANN | — | — | 99.670% | 102.405 | 111.841 | 6.783 | 582.7 | — | — |
| Budget Early Termination | 20 reads | — | 79.930% | 20.306 | 29.488 | 1.425 | 2732.2 | — | — |
| Budget Early Termination | 40 reads | — | 96.090% | 40.411 | 49.676 | 2.773 | 1416.0 | — | — |
| Budget Early Termination | 60 reads | — | 98.500% | 60.458 | 69.784 | 4.079 | 967.3 | — | — |
| Budget Early Termination | 70 reads | — | 98.990% | 71.804 | 81.168 | 4.856 | 811.7 | — | — |
| Budget Early Termination | 80 reads | — | 99.220% | 80.510 | 89.908 | 5.402 | 730.7 | — | — |
| Budget Early Termination | 85 reads | — | 99.370% | 87.700 | 97.110 | 5.805 | 680.1 | — | — |
| Budget Early Termination | 90 reads | — | 99.470% | 91.765 | 101.186 | 6.281 | 628.8 | — | — |
| Budget Early Termination | 95 reads | — | 99.520% | 95.900 | 105.333 | 6.413 | 615.5 | — | — |
| Budget Early Termination | 100 reads | — | 99.570% | 99.896 | 109.331 | 6.724 | 587.6 | — | — |
| Budget Early Termination | 105 reads | — | 99.670% | 102.373 | 111.809 | 6.858 | 576.2 | — | — |
| Budget Early Termination | 110 reads | — | 99.670% | 102.405 | 111.841 | 6.955 | 567.9 | — | — |
| Budget Early Termination | 120 reads | — | 99.670% | 102.405 | 111.841 | 6.901 | 572.3 | — | — |
| Budget Early Termination | 140 reads | — | 99.670% | 102.405 | 111.841 | 6.889 | 573.3 | — | — |
| Learned ET Top3 | 0% unsafe query target | 0.998637319 | 99.650% | 100.085 | 109.521 | 6.884 | 570.3 | 339 | 1 |
| Learned ET Top3 | 0.1% unsafe query target | 0.997519851 | 99.650% | 96.885 | 106.317 | 6.545 | 603.3 | 548 | 1 |
| Learned ET Top3 | 0.25% unsafe query target | 0.99708879 | 99.650% | 95.777 | 105.207 | 6.547 | 603.4 | 604 | 1 |
| Learned ET Top3 | 0.5% unsafe query target | 0.982877135 | 99.450% | 78.552 | 87.942 | 5.342 | 738.0 | 940 | 18 |
| Learned ET Top3 | 1% unsafe query target | 0.975630164 | 99.390% | 74.001 | 83.378 | 5.106 | 771.8 | 967 | 24 |
| Learned ET Top3 | 2% unsafe query target | 0.958117783 | 99.190% | 66.013 | 75.360 | 4.450 | 885.0 | 996 | 42 |
| Learned ET Top3 | 5% unsafe query target | 0.925560832 | 98.870% | 57.033 | 66.356 | 3.904 | 1008.1 | 1000 | 71 |
| Learned ET Top3 | 10% unsafe query target | 0.891305745 | 98.450% | 51.164 | 60.468 | 3.546 | 1108.5 | 1000 | 103 |
| Learned ET All17 | 0% unsafe query target | 0.998252034 | 99.660% | 94.448 | 103.872 | 6.352 | 621.5 | 474 | 1 |
| Learned ET All17 | 0.1% unsafe query target | 0.99777627 | 99.660% | 92.986 | 102.408 | 6.420 | 615.3 | 529 | 1 |
| Learned ET All17 | 0.25% unsafe query target | 0.992542326 | 99.620% | 83.463 | 92.869 | 5.738 | 687.9 | 801 | 5 |
| Learned ET All17 | 0.5% unsafe query target | 0.987092793 | 99.550% | 77.986 | 87.380 | 5.427 | 726.9 | 889 | 12 |
| Learned ET All17 | 1% unsafe query target | 0.972130418 | 99.320% | 68.661 | 78.032 | 4.613 | 854.1 | 968 | 31 |
| Learned ET All17 | 2% unsafe query target | 0.959524274 | 99.230% | 64.006 | 73.355 | 4.350 | 905.1 | 984 | 40 |
| Learned ET All17 | 5% unsafe query target | 0.923555076 | 98.850% | 55.422 | 64.742 | 3.826 | 1028.2 | 997 | 74 |
| Learned ET All17 | 10% unsafe query target | 0.883238673 | 98.350% | 49.534 | 58.840 | 3.374 | 1165.1 | 1000 | 116 |
| Learned Top3 Expansion Pruning | 0.1% retain FNR target | 0.015224196 | 99.640% | 97.050 | 106.481 | 6.709 | 588.9 | — | — |
| Learned Top3 Expansion Pruning | 0.25% retain FNR target | 0.02587937 | 99.640% | 93.286 | 102.710 | 6.326 | 624.3 | — | — |
| Learned Top3 Expansion Pruning | 0.5% retain FNR target | 0.0366583429 | 99.630% | 89.702 | 99.120 | 6.292 | 627.4 | — | — |
| Learned Top3 Expansion Pruning | 1% retain FNR target | 0.0549282432 | 99.580% | 83.825 | 93.229 | 5.870 | 671.9 | — | — |
| Learned Top3 Expansion Pruning | 2% retain FNR target | 0.0822836608 | 99.450% | 75.691 | 85.073 | 5.280 | 746.2 | — | — |

## Curves

![Recall / costs](online/recall_cost_curves.png)

曲线仅连接实测点，不插值调参或平滑拟合；下排放大Recall>=99%。Latency横线显示重复测量min/max。observed_frontiers.json记录各方法每个cost维度的实测非支配点，仅用于描述，不用于test部署参数选择。

## Interpretation

本次safe-stop任务All17的Test AUC为0.956429，高于Top3的0.950089；前序expansion pruning的Top3接近All17结论不能直接移植到终止任务。新增state在本次ET低风险区域仍有收益。
代表实测点：All17 ET（validation unsafe-query目标0.25%）Recall=99.62%、Reads=83.463、Expansions=92.869；Budget100为99.57%、99.896、109.331；Top3 pruning（validation retain FNR目标1%）为99.58%、83.825、93.229。All17 ET这一点在召回和三个cost维度均优于这两个实测配置。
Top3 ET并非在每个区域都优于Top3 pruning：两者Recall=99.45%的实测点中，ET（unsafe-query目标0.5%）Reads=78.552，pruning（retain FNR目标2%）Reads=75.691。结论应按真实Recall–cost而不是统一风险百分比比较。
All17 ET 0.25%目标在test实际误停5/1000=0.5%；验证零误停配置的两个ET在test也各有1条unsafe query。safe-stop是分类预测，不是新的安全保证；验证风险目标和测试实际误停均已报告。

## Verification

关闭ET（tau=1.1）时，全部validation/test的有序ID和距离逐项复现baseline；state、label和Python/C++概率核验通过。所有在线ET的第一次停止位置、展开数、推理次数和停止计数逐query与baseline原生概率预测一致。每个safe停止均逐项保留baseline有序ID和距离；unsafe停止恰好对应baseline结果变化。
全部重复测量结果ID一致，Recall独立重算通过。Budget I/O overshoot<=3核验；baseline和Top3 pruning保持前序搜索行为。实际Recall与SSD/Expansion由真实搜索输出，未复用离线mask。
SSD Reads是索引read请求，不是硬件物理读取；当前时延是预热主机性能。查询此前用于其他探索性分析，因此最终外推应使用新查询/数据集。

## Reproduce

在learned-gate目录运行：

```bash
cmake --build DiskANN/build --target search_disk_index -j 4
.venv/bin/python experiments/sift1m/collect_termination_data.py
.venv/bin/python experiments/sift1m/train_termination.py
.venv/bin/python experiments/sift1m/calibrate_termination.py
.venv/bin/python experiments/sift1m/run_online_termination.py
.venv/bin/python experiments/sift1m/plot_termination_curves.py
```
