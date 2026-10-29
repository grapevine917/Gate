# DiskANN unified main experiment: mixed-L tuning and cost frontiers

## 三张主图

每张图左侧覆盖Recall 90–100%，右侧放大99–100%。标记为真实测量配置，曲线连接仅帮助阅读；未把不同Recall的配置称为严格同Recall实测。每张图分别求该成本的非支配前沿，而不是把SSD最优配置重复用于P95/QPS。

![Recall–SSD](figures/recall_ssd.svg)

![Recall–P95](figures/recall_p95.svg)

![Recall–QPS](figures/recall_qps.svg)

## 结果观察

- 联合调L后，低Recall区间的SSD前沿较接近；差异在高Recall区间更明显，不能沿用固定L100的收益结论。
- SSD请求、P95和QPS的最优配置不同，省请求并不保证降低尾延迟。原始DiskANN放开L后，P95基线仍有较强竞争力。
- 一个严格同Recall例子：99.95%时，原DiskANN与q–v MLP分别为300.085与257.877请求/query；相应SSD最优配置的三轮P95中位数为21.05与22.52ms，QPS为198.48与201.22。q–v省约14.1%请求，但P95更高，且另需512MB向量内存。这是同Recall的具体实测点，不是所有区间的总体排名。
- ET与ET+State MLP在部分区间的前沿重合，有限64配置预算下没有统一的全目标赢家；所有曲线、配置和范围均保留。

## 公平搜索环境与参数预算

SIFT1M，同一DiskANN图、PQ索引、K10、beam4、4条并发查询线程、10000节点静态BFS缓存。所有候选剪枝方法一致保护缓存命中。Linux reader使用O_DIRECT，绕过OS页缓存；每配置预热128条固定分布查询。每个方法同为64个候选配置，网格在验证/测试扫描之前冻结：

- DiskANN允许64个不同L，范围10–300，且包含所有学习方法使用的L，提供充分调参的强基线。
- 启发式和各Gate：L∈{15,20,30,50,75,100,160,300}×8个阈值/规则预设；不固定L100。
- ET：同一L集合×8个停止阈值。ET+State MLP：同一L集合×8组预声明的二阈值组合；没有额外使用全笛卡尔积预算。
- Gate阈值取自同一批2000个训练状态的分数分位点，不用测试结果生成阈值。ET预测当前Recall，阈值有明确0.8至0.999范围及禁用值1.1。
- Top-3启发式沿用项目已有rank+completed-expansions固定合取规则；它使用Top-3重要特征中的两个，不把三特征MLP冒称启发式。r/e按L缩放的8个预设全部公开在TUNING_PROTOCOL.json。

这里的“最好”限于相同64配置预算内的实测成本前沿，不等于无穷参数空间的全局最优。原始DiskANN源文件、原有模型均未覆盖；主实验使用独立模型及原生库版本。

## 跨L训练与模型定义

原有Gate/ET多只在L100轨迹训练，本轮没有直接把它们生硬外推到不同L。改为在上述8个L采集1200条训练查询的混合轨迹：
剪枝样本285,754条，ET样本168,197条。所有Gate共享同一批非缓存候选和标签；训练查询与1000验证、1000测试查询严格不相交。训练查询内划分1000/200选择epoch/树数，最后全部1200条训练查询重拟合，验证/测试不用于模型权重或checkpoint选择。

|方法|在线输入/行为|
|---|---|
|DiskANN|仅调L，无模型|
|Top-3 heuristic|rank与completed expansions的固定规则|
|LR Gate|17维pre-I/O状态，逻辑回归|
|XGBoost Gate|相同17维状态，GBDT|
|State-only MLP|17→128→64→1，ReLU/ReLU/sigmoid|
|q–v Interaction MLP|[q,v,abs(q-v),q*v,17D state]，529→128→64→1|
|Learned ET|17维beam-boundary状态→当前Recall估计；足够K个精确候选后才可终止|
|ET + Learned Pruning|上述ET与State-only MLP同时实际运行，均基于改变后的现场状态|

Gate监督标签为“本次扩展有邻居入队，或该节点在当前L的最终基线Top10”；这是保守贡献代理，不是前几轮的单节点反事实风险标签。ET标签是当前已精确评分前缀的Top10与真实GT的Recall@10；GT只用于离线训练标签和最终评分，不进入在线输入。ET在无剪枝混合L轨迹上训练，联合策略存在轨迹偏移，但结果通过完整真实搜索测量，不做oracle回放。

q–v模型需要额外驻留全部1M×128 float32向量，约512MB；这个内存代价明确保留，不能把它称为与state-only方法等内存。所有方法使用相同10000节点图缓存，模型加载不计入查询时延，所有推理和q–v特征组装/访存都计入查询耗时。

## Recall覆盖与配置数量

|方法|配置数|最小实测Recall%|最大实测Recall%|SSD/P95/QPS前沿点数（≥90%）|
|---|---:|---:|---:|---|
|DiskANN (L sweep)|64|75.980|99.950|46/46/46|
|Top-3 heuristic (rank+progress)|64|12.850|99.950|31/27/27|
|LR Gate|64|29.020|99.950|25/26/25|
|XGBoost Gate|64|43.420|99.950|29/25/27|
|State-only MLP|64|42.500|99.950|27/22/25|
|q-v Interaction MLP|64|39.550|99.950|26/20/25|
|Learned ET (DiskANN)|64|84.230|99.950|26/21/25|
|ET + State-only MLP|64|76.530|99.950|31/24/30|

## 相同Recall如何比较

主图横坐标全部为**实际测试Recall**，不是验证目标或阈值。每种方法从其相同预算的完整冻结网格中求出各自非支配前沿；因此不会拿任意固定配置代表方法。主图的前沿是固定测试查询上的描述性“可达到成本”曲线，不是完全独立的部署参数选择。

另将每个验证Recall目标90/95/98/99/99.5/99.7%、每个成本目标分别选择的配置在测试前冻结，保存为SELECTION_LOCK.json；它们的实际测试Recall、成本和是否达标位于validation_locked_test.csv。不同测试Recall没有仅因共用验证下限而被宣称相同。

**严格同Recall审核表** `strict_equal_recall_ssd.csv` 只在八组原始配置都实际具有相同整数GT命中数时生成，同一1000查询、分母10000，不用四舍五入匹配；每个方法取该精确Recall的最低请求配置。禁用模型动作可提供共同基线工作点，表中同时记录是否真的Skip或ET，避免把禁用点包装为学习收益。该表可能包含被更高Recall、较低成本配置支配的离散点；CSV另标记read_frontier_member，不能用这类点取代主图非支配前沿。配对query bootstrap区间位于strict_equal_recall_ssd_CI.csv，条件于已选择配置，不覆盖训练或曲线选点不确定性。

本次共同实测Recall点：90.86%, 95.39%, 98.24%, 99.26%, 99.67%, 99.87%, 99.95%。

|相同实测Recall%|DiskANN (L sweep)|Top-3 heuristic (rank+progress)|LR Gate|XGBoost Gate|State-only MLP|q-v Interaction MLP|Learned ET (DiskANN)|ET + State-only MLP|
|---|---|---|---|---|---|---|---|---|
|90.86|25.566|25.566|25.566|25.566|25.566|25.562|25.566|25.566|
|95.39|34.615|34.615|34.615|34.615|34.530|34.407|34.593|34.521|
|98.24|53.546|53.546|53.494|53.186|53.282|52.206|53.535|53.535|
|99.26|77.862|77.862|77.862|77.459|77.691|77.775|76.154|76.154|
|99.67|102.405|102.405|102.381|101.258|102.405|101.981|102.405|102.405|
|99.87|161.472|161.472|157.546|158.466|151.330|152.488|161.472|161.472|
|99.95|300.085|290.836|300.085|290.174|282.154|257.877|300.085|300.085|

为方便读取90/95/98/99/99.5/99.7标准位置，另提供 `ssd_frontier_interpolation_ESTIMATES.csv`：只对有相邻实际前沿点包围的SSD曲线做线性读数，标为估计、无外推，不冒充单配置实测。P95分位数和并发QPS不能直接由两配置线性混合解释，本实验不输出这类可部署性能的插值承诺。

## 计时与吞吐定义

单query延迟计时包住完整cached_beam_search，包含I/O等待、现场特征、所有启发式/模型推理。P95来自1000条query的单次延迟分布；每个最终前沿配置与验证锁定配置均重复三轮，图中采用三轮P95的中位数。QPS为1000条查询实际4线程并发搜索的wall-clock吞吐，并非1/平均延迟，也不含加载、GT评分和CSV写盘。配置顺序固定种子随机串行执行，不同时跑多个配置。三轮ID、Recall、请求及展开计数必须一致；具体时间范围记录在frontier_points.csv。

禁用值0/1.1关闭动作但仍执行相应模型，便于核验输出一致性并保留实际启用模型的成本；纯DiskANN不执行任何模型。图中不同成本可以选择不同的(L,threshold)；在frontier_points.csv可追溯每个点。少量模型概率或GT命中无法排除有限样本误差，本实验不宣称每条query的Recall保证。

## 相关研究：同环境终止对照

[DARTH官方实现](https://github.com/MChatzakis/DARTH)集成于Faiss，并提供HNSW/IVF实验入口；[Learned Adaptive Early Termination官方实现](https://github.com/efficient/faiss-learned-termination)同样以Faiss为基础。它们的公开HNSW/IVF性能不进入本实验任何数值表或主图。

本轮ET是在相同DiskANN搜索环境内实现的可比学习终止控制，受相关策略类别启发，但不是DARTH/LAET的严格复现，也不在图例中冒用其名称。具体差异见RELATED_WORK.md。

## 核验、限制与复现

每个模型2000个状态的原生/Python概率最大误差：{'LR': 2.08616e-07, 'XGB': 1.19209e-07, 'StateMLP': 2.38419e-07, 'QV_MLP': 3.57628e-07, 'ET': 9.53674e-07, 'samples_each': 2000}。所有八个L的禁用动作对照，最终ID、Recall、请求及展开与原DiskANN完全一致。全部Recall从保存ID和GT独立重算；原有源码、模型哈希不变；最终前沿配置计时三轮且ID/计数一致。

共512个验证配置、512个测试配置；310个配置补做三轮计时，共1132次测试配置运行。不同方法预算相同；重复计时不引入新配置或新模型。

这是现有SIFT1M查询池上的探索性主比较；测试查询此前已被多个实验观察，不是全新确认性测试集。训练使用固定1200-query共同预算，未穷举模型结构或训练数据规模；结论限于公开模型/调参预算与硬件设置。q–v的额外内存代价及ET+pruning的轨迹偏移均不可省略。

复现：在项目根目录执行 `bash experiments/main_comparison/run.sh`。独立SVG图可直接下载编辑；若环境有Matplotlib，plot.py同时生成PNG/PDF。当前绘图后端记录在figures/plot_backend.json。

主要数据：all_configurations.csv、frontier_points.csv、strict_equal_recall_ssd.csv、coverage.csv、validation_locked_test.csv、configs.json、PROTOCOL.json、TUNING_PROTOCOL.json、verification.json。
