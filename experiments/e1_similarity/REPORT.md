# E1：Query 相似性与历史 Skip 决策复用潜力

本轮已执行真实原生搜索。主实验 SIFT1M，既有 DiskANN 图，L=100、K=10、beam=4、4线程、固定 BFS 节点缓存10000。主模型为既有17→128→64→1 state-only MLP；LR、XGBoost为扩展对照，均执行各自独立轨迹。所有跨 Query 决策复用关闭。

## 数据与工作点

从既有8000条模型训练查询中，用seed=20261018选1000条作为历史端；验证1000条、测试1000条，ID及历史/测试完整向量不重叠。历史库仅来自训练查询。测试集此前已被历史实验观察，不是全新未见测试集。静态节点缓存的成员ID已保存，历史/测试哈希一致。每条查询的候选池、visited及线程局部逻辑状态独立。

99.653%是旧全10000查询无剪枝工作点，不等于本次1000条测试子集召回。主参数只根据验证集选择：在原有阈值候选中，选满足Recall≥99.60%的最大阈值；没有为凑测试召回调参。候选全表在results.json。最初99.603%的绝对下限会退回MLP不剪枝，随后仅根据验证集0.01pp离散精度将下限对齐为99.60%；修订时间和原因见PROTOCOL_AMENDMENT.json，原pilot运行保存在runs/和pilot_*.json。

|方法|冻结阈值|测试Recall@10 %|请求/q|模型调用/q|平均延迟 µs（三轮中位数）|p95 µs（三轮中位数）|
|---|---:|---:|---:|---:|---:|---:|
|MLP|0.024352464824914932|99.640|93.410|102.391|5452.93|6366.74|
|DiskANN|—|99.670|102.405|0.000|5784.12|6222.55|
|XGBoost|0.01714894361793995|99.650|96.219|102.395|6026.35|6829.00|
|global_once|—|90.550|92.304|0.000|5644.24|6138.90|
|LR|0.004196530601220115|99.640|97.013|102.391|5517.61|6225.67|

全局一次标记保留为baseline：仅统计同一1000条历史查询的原始无剪枝miss；至少观察一次且所有观察均没有新邻居入队、节点不在该查询最终Top10时标记。共56310个节点，位图125000字节。全局标记不通过测试调参，也不隐藏其召回损失。

## 轨迹与指标定义

每条轨迹保存query_id、node_id、state_features[17]、原生retain分数、Skip/Keep决策、search_step（query内候选决策序号，从0开始）、cache_hit、model_invoked。另有原始Query向量和ID映射。缓存命中记录保留作审计，其score=1为保留哨兵值，不代表实际推理；V_i仅包括实际调用模型的cache miss节点；P_i为其中的Skip集合。

对全部1000000个有方向history→test对计算L2和集合交集。A为共同节点上的决策一致率；P_skip为历史Skip在共同节点上仍被目标模型Skip的比例；Coverage=|P_history∩V_test|/|V_test|。共同节点为0不参与A均值；历史Skip共同节点为0不参与P_skip均值，单独报告比例。Coverage不因无重叠而排除，零值参与均值。另报按事件汇总的pooled值，避免与Query对宏平均混淆。

分桶严格使用实际Query对L2的分位数；边界（L2）为：92.698, 303.460, 357.252, 428.520, 559.950, 717.655。阈值相等处用左闭右开，末桶含最大值。另按每条目标Query的历史近邻排名统计，并生成每Query内部距离分位桶。

|距离分位|共同节点/q对|A %|P_skip %|Coverage %|A有效对数|P_skip有效对数|无共同节点 %|无可复用Skip %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0–1%|1.372|88.37|21.30|0.149|3868|619|61.320|93.810|
|1–5%|0.140|93.18|11.29|0.007|3324|186|91.690|99.535|
|5–20%|0.009|93.31|28.42|0.000|1063|55|99.291|99.963|
|20–50%|0.000|93.33|0.00|0.000|15|1|99.995|100.000|
|50–100%|0.000|N/A|N/A|0.000|0|0|100.000|100.000|

![Query相似性与复用潜力](figures/similarity_reuse_potential.png)

![每Query近邻](figures/per_query_neighbours.png)

![距离与交集](figures/distance_and_overlap.png)

## H1的实测解读

最近0–1%桶与最远50–100%桶：A分别为88.37%、N/A；P_skip分别为21.30%、N/A；Coverage分别为0.149%、0.000%。

最近0–1%桶与下一1–5%桶的A差值=-4.810个百分点，目标Query聚类bootstrap95%区间=[-6.104, -3.440]个百分点。

逐目标最近一条历史Query：A=86.90%，P_skip=22.30%，Coverage=0.748%，平均共同节点5.36个。

最远50–100%桶完全没有共同模型决策节点，A和P_skip未定义，报告为N/A，不能填成0或100%。本轮结果不支持“越相似则A越高”的整体单调关系；更近的Query显著增加共享节点，但Skip一致性和覆盖价值仍需独立评估。
这些统计用于检验相似度与决策一致性的关联，不能单凭A高就认定复用有价值。必须同时看Skip精确率、可覆盖的模型调用比例、无重叠比例和错误Skip风险。每桶CSV保存目标Query聚类bootstrap 95%区间；历史端固定，因此区间不包含历史库采样不确定性。Query对并不独立，不使用百万对的独立样本p值宣称显著。H1不是因果结论，当前数据也没有控制共同节点难度等混杂因素。

双方模型同意Skip的交集/目标决策数另列为agreed_skip_coverage；同意不代表反事实安全；这是原独立轨迹上的可复用潜力，不能等同执行复用后的额外I/O节省。已有模型本来会Skip的节点，复用该决策首先省掉的是预测；错误Skip会改变后续路径，必须在后续真正开启复用时单独测量召回和I/O。

## 成本分解

按五项记录 C_total=C_distance+C_model+C_memory+C_SSD+C_bookkeeping。单独profile运行使用互斥软件区域计时，trace写入和性能计时不混在一起；profile与关闭profile计时返回ID一致。关闭profile仍有空指针分支检查的少量开销，所有方法使用同一采集程序。

|方法|distance µs|model µs|memory µs|SSD µs|bookkeeping µs|
|---|---:|---:|---:|---:|---:|
|DiskANN|104.29|0.00|112.74|5359.78|256.92|
|global_once|94.85|0.00|106.22|5260.69|247.97|
|MLP|95.76|101.31|107.13|4913.63|269.87|
|LR|96.56|11.11|108.69|5028.17|273.04|
|XGBoost|94.09|540.77|105.83|5023.79|263.46|

![成本分解](figures/cost_breakdown.png)

distance包含PQ查询表、PQ距离查表和展开节点的完整向量距离内核；model包含标准化和分类器评分；memory仅包含显式PQ编码聚集和磁盘向量到scratch的复制；SSD为同步reader调用（提交与等待）；bookkeeping包含候选队列、visited、特征提取、缓存查找、分配、排序及未单独包裹的小型初始化计算。这里的memory不是全部硬件内存访问时间，模型和距离区域内的隐式内存等待仍归属该区域，不能把硬件内存成本严格分离。SSD请求数也不是物理SSD读盘次数。profile中的计时器会增加开销，绝对延迟采用无profile、无audit三轮结果；不将profile分项直接从无插桩总延迟相减。

## 扩展与核验

LR和XGBoost与MLP共享同一原始训练划分和17维特征，保留其既有权重，单独验证选阈值、独立在线剪枝与采集，非在MLP轨迹上替换分数冒充独立搜索。XGBoost原生导出仅包含best_iteration之前的树，与Python预测核验。

![模型对照](figures/model_controls.png)

另存MLP测试—测试有方向999000对，排除自身Query对，仅作为描述性诊断；不把测试Skip记录用于任何历史库或测试时决策。

核验：历史/验证/测试ID不交叉；历史/测试向量不重复；native/Python分数一致；三轮ID确定性；trace/profile和无插桩返回ID一致；100个随机Query对用独立Python集合公式校验；所有成本分项相加误差检查。详细数值见verification.json和analysis.json。

## 文件与复现

- PROTOCOL.json、SELECTION_LOCK.json：主协议和测试前阈值锁定。
- experiment.log、build.log、runs_main/*.log、*.command.json：阶段日志、编译日志和完整命令。
- *_trace.npz：完整结构化轨迹；data/query_vectors.npy和query_splits.json：向量及映射。
- *_history_test_pairs.npz：逐Query对原始计数和指标；*_distance_bins.csv、*_nearest_rank_bins.csv、*_per_query_quantile_bins.csv：汇总及区间。
- PROVENANCE.json：模型、索引、程序哈希；COMPLETE.json：结束校验。
- figures/：PNG图；主要图同时提供PDF。

```bash
cd /home/lxyue/projects/learned-gate
.venv/bin/python experiments/e1_similarity/build_native.py
.venv/bin/python experiments/e1_similarity/run.py
```

原生插桩库和XGBoost支持仅生成于本实验native/目录，不修改已有DiskANN源码和既有二进制。已完成搜索可按done记录恢复；若改变协议、模型或程序，应使用新目录运行，不能混用旧done记录。

稀疏桶解读：20–50%桶只有15对可计算A、1对可计算P_skip，不能据此解释为稳定的0%复用精确率。最近0–1%桶按共同节点事件加权的A=84.13%，目标Keep比例=89.17%；Keep占多数本身即可造成较高表面一致性。

纯评分成本：MLP的模型区域约101.31µs/query，占该profile五项总和约1.85%。在固定轨迹的分项账本中，消除评分区域本身的空间有限；实际复用还要付出查询匹配、内存查找和维护成本，本E1不宣称端到端复用加速。
