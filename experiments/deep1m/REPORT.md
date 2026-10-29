# DEEP1M 四方法核心实验

数据：1M×96 base、10K query、对应1M的GT100。原始数据保留，32条预定query全库float64精确L2核验通过。

L取值固定为[50, 75, 100]，区间与训练前validation baseline pilot确定；K=10、R=64、Lbuild=100、PQ34字节（search RAM预算0.032GB）、beam=4、threads=4、cache=10000。

模型仅两种：All17 Learned ET 与 Top3 expansion pruning。在选定多个L的轨迹上联合训练，query按seed42固定8000/1000/1000互斥划分；归一化只拟合train。结构、优化器与SIFT实验保持一致，未复用SIFT权重。每个L有独立基线标签、阈值校准。全量oracle skip的ID和距离与baseline一致，pre-IO状态逐query重放核验。

Standalone ET按全量原生validation概率的query误停率8档校准；pruning按validation retain-FNR5档校准，原生首query概率审计。ET+pruning的ET按每个固定pruning阈值的实际pruned validation路径重新校准，8×5联合配置各L实际运行，停止路径与safe endpoint逐query核验。

四方法绘图配置在test开始前锁定：DiskANN展示区间内各L；其他三方法分别按validation Recall损失预算选择最低Reads配置，Latency不参与选择，不使用test挑点。只展示实际选定点，未将独立组件节省相加，未插值匹配Recall。

![核心曲线](recall_cost_core.png)

![高召回细节](recall_cost_core_high_recall.png)

| Method | L | Key | Recall@10 % | Reads/query | Latency ms |
|---|---:|---|---:|---:|---:|
| DiskANN | 50 | baseline_L50 | 98.290 | 53.813 | 3.320 |
| DiskANN | 75 | baseline_L75 | 99.160 | 77.886 | 4.796 |
| DiskANN | 100 | baseline_L100 | 99.470 | 102.199 | 6.170 |
| Learned ET | 100 | L100_ET_unsafe_query_0pct | 99.460 | 88.602 | 5.504 |
| Learned ET | 100 | L100_ET_unsafe_query_0.25pct | 99.460 | 86.738 | 5.315 |
| Learned ET | 100 | L100_ET_unsafe_query_0.5pct | 99.460 | 85.061 | 5.156 |
| Learned ET | 100 | L100_ET_unsafe_query_1pct | 99.440 | 79.195 | 5.005 |
| Learned ET | 100 | L100_ET_unsafe_query_2pct | 99.370 | 73.904 | 4.518 |
| Learned ET | 75 | L75_ET_unsafe_query_0.5pct | 99.110 | 67.887 | 4.190 |
| Learned ET | 75 | L75_ET_unsafe_query_5pct | 98.740 | 53.720 | 3.322 |
| Learned ET | 50 | L50_ET_unsafe_query_5pct | 97.870 | 45.216 | 2.854 |
| Learned ET | 50 | L50_ET_unsafe_query_10pct | 97.280 | 41.044 | 2.531 |
| Top3 pruning | 100 | L100_pruning_fnr_0.25pct | 99.410 | 92.526 | 5.818 |
| Top3 pruning | 100 | L100_pruning_fnr_0.5pct | 99.370 | 87.627 | 5.486 |
| Top3 pruning | 100 | L100_pruning_fnr_1pct | 99.280 | 81.382 | 5.115 |
| Top3 pruning | 75 | L75_pruning_fnr_0.25pct | 99.060 | 70.450 | 4.465 |
| Top3 pruning | 75 | L75_pruning_fnr_1pct | 98.870 | 62.611 | 3.943 |
| Top3 pruning | 50 | L50_pruning_fnr_0.5pct | 97.860 | 47.589 | 3.046 |
| Top3 pruning | 50 | L50_pruning_fnr_1pct | 97.460 | 44.682 | 2.900 |
| Top3 pruning | 50 | L50_pruning_fnr_2pct | 96.730 | 40.738 | 2.618 |
| ET+pruning | 100 | L100_joint_fnr_0.25pct_et_0.1pct | 99.400 | 84.663 | 5.321 |
| ET+pruning | 100 | L100_joint_fnr_0.25pct_et_0.5pct | 99.400 | 81.646 | 5.098 |
| ET+pruning | 100 | L100_joint_fnr_0.25pct_et_2pct | 99.330 | 72.088 | 4.525 |
| ET+pruning | 100 | L100_joint_fnr_1pct_et_2pct | 99.110 | 64.981 | 4.051 |
| ET+pruning | 75 | L75_joint_fnr_0.1pct_et_5pct | 98.720 | 53.459 | 3.338 |
| ET+pruning | 75 | L75_joint_fnr_1pct_et_10pct | 97.890 | 45.021 | 2.821 |
| ET+pruning | 50 | L50_joint_fnr_0.5pct_et_10pct | 96.700 | 39.115 | 2.461 |
| ET+pruning | 50 | L50_joint_fnr_2pct_et_10pct | 95.650 | 35.762 | 2.303 |

全部选定配置共同随机交错重复3次，延迟为median且含真实两模型推理与状态构造。图中误差线为min/max，SSDReads为索引请求数。每次Recall独立重算，重复结果ID/距离/读取/展开数一致。test指标不保证满足validation目标。

`online/summary.csv`为曲线数据，`validation/grid.csv`为全部validation配置，`selection.json`为预算选择记录，`L_selection.json`为L选择记录。

复现：`.venv/bin/python experiments/deep1m/check_data.py`；按`build.log`对应参数建索引；`.venv/bin/python experiments/deep1m/run_core.py`。

## 实测结果摘要

相同Recall=99.11%时，ET+pruning的Reads为64.981、ET为67.887，联合少读取约4.28%；median latency分别为4.051ms与4.190ms。相同Recall=99.37%时，ET的Reads为73.904、Top3 pruning为87.627；ET更省读取。这里只描述锁定配置的实测点，不做test调参或插值推广。

DEEP的状态模型入口已支持96维；原SIFT的baseline、ET、pruning、joint四个配置的1000条query回归核验通过，ID、距离和全部gate计数不变（见sift_regression/COMPLETE.json）。
