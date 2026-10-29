# E4-2：历史Skip是否识别遗漏的冗余Keep

## 核心结论

历史Skip标记本身没有富集A类：评估集Conflict Keep的A比例72.90%，Normal Keep为83.94%，差值95%区间完全小于零。状态相似筛选提升Conflict Keep精确率至87.23%，但覆盖很小；在本轮嵌套Logistic对照中，没有证实Query相似度的独立增益，也没有证实历史统计在当前状态之外的独立增益。验证选中的完整搜索点中，M0仅按当前分数筛选获得最大请求节省22.34%，评估Recall下降0.20pp；该点刚好在预算边界，区间不保证总体损失≤0.2pp。

## 数据、模型与冻结顺序

1000条E1训练历史Query构成唯一历史库，静态且不在线更新。1000条E4-1 Query按seed=20261020分成500拟合、250验证、250评估；ID完全隔离。它们来自已经观察过的E4-1数据池，不是全新确认性测试集。U沿用E4-1逐位置真实单节点反事实标签。测试标签不参与拟合、参数选择或部署阈值设定。

标准化沿用E2训练统计；状态阈值为 2.894251，Query阈值为 277.456303。H要求同节点任意历史实际模型调用为Skip；频率分母是该节点全部历史实际模型调用。距离只匹配历史Skip记录。联合条件要求同一条历史Skip记录同时满足State和Query限制，不把两个不同来源的近邻拼成一条证据。

|方法|筛选依据|可筛范围|
|---|---|---|
|M0|当前retain分数低者优先|全部Keep|
|M1|曾有历史Skip；并列按与标签无关的固定hash处理|Conflict Keep|
|M2|历史Skip占比高者优先|Conflict Keep|
|M3|Logistic：最近历史Skip的状态距离|Conflict Keep|
|M4|Logistic：最近历史Skip的Query距离|Conflict Keep|
|M5|Logistic：M3状态距离 + 最近Skip Query距离 + 状态最近Skip的Query距离|Conflict Keep|
|M6|Logistic：当前分数、17维当前状态、历史频率统计、最近Skip状态距离/状态近指示；不使用Query距离|全部Keep|

Logistic固定C=1、无类别重加权，特征scaler只拟合500条拟合Query。M3–M5只在Conflict Keep上拟合，M6在全部Keep上拟合。M0–M2为简单规则。分数表示排序，原生retain分数未校准为检索必要性概率。

## 分组对照

|范围|分组|Keep数|Query数|A比例 %|95%区间 %|
|---|---|---:|---:|---:|---|
|all|All Keep|93410|1000|84.17|[83.39, 84.89]|
|all|Conflict Keep|1359|433|76.67|[72.61, 80.18]|
|all|Normal Keep|92051|1000|84.28|[83.51, 84.99]|
|all|Conflict + State close|453|236|87.20|[82.84, 91.13]|
|all|Conflict + Query close|911|240|73.98|[68.47, 78.49]|
|all|Conflict + joint close same record|305|137|86.56|[80.11, 91.67]|
|test|All Keep|23233|250|83.74|[82.22, 85.06]|
|test|Conflict Keep|417|108|72.90|[65.48, 79.59]|
|test|Normal Keep|22816|250|83.94|[82.45, 85.19]|
|test|Conflict + State close|141|66|87.23|[79.61, 94.40]|
|test|Conflict + Query close|304|71|72.04|[62.89, 80.51]|
|test|Conflict + joint close same record|102|44|86.27|[75.53, 95.61]|

|范围|差值对照|A比例差 pp|95%区间 pp|
|---|---|---:|---|
|all|Conflict minus Normal|-7.60|[-11.39, -4.43]|
|all|State close minus Normal|2.92|[-1.32, 6.59]|
|all|State close minus State far within H|15.78|[11.43, 20.24]|
|all|State close minus All Conflict|10.52|[7.61, 13.46]|
|all|Query close minus Query far within H|-8.16|[-14.82, -2.18]|
|all|Joint close minus State-close without joint|-1.96|[-10.13, 6.04]|
|test|Conflict minus Normal|-11.04|[-18.07, -4.82]|
|test|State close minus Normal|3.30|[-4.12, 10.01]|
|test|State close minus State far within H|21.65|[14.16, 28.24]|
|test|State close minus All Conflict|14.33|[9.36, 18.63]|
|test|Query close minus Query far within H|-3.18|[-17.41, 9.80]|
|test|Joint close minus State-close without joint|-3.47|[-17.20, 9.37]|

所有区间按Query聚类bootstrap1000次，历史库固定；覆盖率低的条件不能只看较高比例。state close是min状态距离<阈值，query close是min Query距离<阈值；联合近邻必须来自同一条Skip。全量结果是探索性描述，250条评估Query结果用于本轮方法比较。

## Precision@Coverage

Coverage分母始终为评估集的全部原始Keep。M1–M5不向Normal Keep补选；超过历史标记容量时为N/A。边界同分样本按分数相同事件的期望精确率计算，避免人为打破并列造成M1虚假优势；实际固定hash排序精确率另存CSV。这是常规排序评估，部署图搜索阈值则只用验证分数分布冻结。

|方法|0.1%覆盖 Precision %|0.25%|0.5%|1%|2%|5%|10%|20%|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|M0|100.00|98.31|99.15|96.57|96.13|93.80|93.63|94.06|
|M1|72.90|72.90|72.90|72.90|N/A|N/A|N/A|N/A|
|M2|72.62|72.62|72.62|72.62|N/A|N/A|N/A|N/A|
|M3|95.83|96.61|89.74|84.12|N/A|N/A|N/A|N/A|
|M4|87.50|77.97|76.07|77.68|N/A|N/A|N/A|N/A|
|M5|95.83|94.92|93.16|86.70|N/A|N/A|N/A|N/A|
|M6|100.00|98.31|97.44|97.00|96.99|97.76|97.76|97.89|

评估集A类基准比例=83.739%；历史标记容量=1.795%（417/23233）。完整区间、样本数、lift与实际hash选择结果见 `precision_at_coverage.csv`。

## Query是否有独立预测增益

M5与M3在完全相同的评估Conflict Keep上比较；另用完整当前状态的辅助嵌套对照，检验Query信息是否在当前状态与历史统计之外继续提供信息。正的Log loss改善表示增加特征后更好；这检验当前设定的预测增益，不是因果独立性。

|嵌套对照|n|AUROC（新增 / 原）|AP（新增 / 原）|Log loss改善|95%区间|
|---|---:|---|---|---:|---|
|M5 minus M3: Query added to Skip-state distance|417|0.7418 / 0.7190|0.8770 / 0.8649|0.02142|[-0.01757, 0.07239]|
|M6 minus current_state_only: historical statistics added|23233|0.8509 / 0.8508|0.9588 / 0.9587|0.00028|[-0.00040, 0.00096]|
|current_state_history_query minus M6: Query added after full current state/history|23233|0.8506 / 0.8509|0.9587 / 0.9588|-0.00020|[-0.00045, 0.00001]|

## 完整搜索的Recall–SSD曲线

策略真实运行在原生完整搜索中：保留原MLP，先判定当前是否Keep，再由策略额外Skip；后续路径重新计算当前状态和策略特征。没有复用E4-1的旧位置标签或静态决策记录。DRAM命中沿用原冻结策略，直接保留、不参与额外剪枝。

对每种方法，先用250条验证Query在冻结coverage菜单中选SSD节省最大且额外Recall损失≤0.2pp的点，无可行改善则回退原始搜索。随后冻结选择，评估集全部曲线点如实报告，不按评估结果挑点。验证预算不是评估预算保证。

|方法|验证选中配置|评估Recall %|ΔRecall pp|SSD请求/q|请求节省 %|评估点是否在0.2pp内|
|---|---|---:|---:|---:|---:|---|
|原始MLP|baseline|99.640|0.000|92.932|0.000|是|
|M0|M0_c2000|99.440|-0.200|72.172|22.339|是|
|M1|M1_c0010|99.280|-0.360|92.776|0.168|否|
|M2|M2_c0010|99.280|-0.360|92.792|0.151|否|
|M3|M3_c0050|99.560|-0.080|92.208|0.779|是|
|M4|M4_c0025|99.560|-0.080|92.808|0.133|是|
|M5|M5_c0050|99.560|-0.080|92.280|0.702|是|
|M6|M6_c0200|99.520|-0.120|91.268|1.791|是|

M0所选点的ΔRecall 95%区间为[-0.40, -0.04]pp；M6为[-0.48, 0.00]pp。点估计满足预算不代表已证明总体质量损失在预算内。M1/M2在验证通过后，评估点仍下降0.36pp，应如实视为预算未通过。

每个点的配对Query置信区间和全部冻结曲线见 `recall_ssd_curve.csv`；被选点区间见 `selected_system_results.csv`。SSD统计为提交请求数，不是物理设备读盘次数。本实验不宣称端到端延迟收益：Query距离预先从向量计算，且没有独立计量在线查询匹配和维护成本；真正图搜索路径、最终Recall与请求数均为实测。

## 核验和边界

原生无额外剪枝的全部93,410个Keep位置均与Python逐位置核对28维特征和7个策略分数。原始Top10、SSD请求、展开数、邻居距离比较数及固定缓存成员与E1/E4-1一致。全部验证和评估曲线点的最终Recall均从保存Top10与GT重新计算。历史库与E3序列化记录哈希相同。

同一个Query的多个Keep不作为独立样本；所有关键比例差和质量/I/O区间都按Query聚类。H仅覆盖很少的Keep；M1–M5的高精确率即便成立，也不能自动外推成较高系统节省。U仅证明单节点干预性质，累计剪枝必须看本轮真实Recall曲线。对现有数据池的查询分割降低拟合泄漏，但不能消除先前观察数据带来的探索性限制。

## 文件与复现

`PROTOCOL.json`：拟合/验证/评估Query划分与规则；`models.json`：权重；`SELECTION_LOCK.json`：验证选择；`history_group_comparison.csv`、`group_contrasts.csv`：分组检验；`precision_at_coverage.csv`、`prediction_metrics.csv`、`nested_feature_comparisons.csv`：识别性能；`position_predictions.csv`：逐位置历史特征、预测和标签；`graph/*.csv,*.ids`：原生完整搜索统计和最终结果；`native_parity.json`：特征核验；`figures/*.svg`：独立可导出曲线。

复现：`.venv/bin/python experiments/e4_2_history_skip/run.py`。新协议、模型或原生程序必须使用新输出目录；既有E1–E4-1及DiskANN源文件不修改。
