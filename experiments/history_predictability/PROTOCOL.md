# 历史信息与关键候选保护：预先固定协议

本实验检验动态剪枝轨迹上，读取前历史是否改善关键候选保护，不复用原始路径的单节点标签来冒充剪枝路径标签。

SIFT1M/DEEP1M 独立运行，沿用 io_predictability 的互斥 query 划分；每 split 用 seed=20261017 随机取 300 queries，训练、验证、测试绝不混用。L=100, beam=4, K=10, BFS cache=10000。轨迹生成策略固定为原 17 维 state_lr 的验证 unsafe_fpr_0.005 阈值，不能用本轮测试结果修改。

在每条真实 LR 剪枝路径的每个 cache-miss 候选处记录 44 维 pre-I/O 状态（包括实际跳过的候选）。分别强制保留和强制跳过该候选，其后继续执行同一个冻结策略。关键标签为 skip_hits < keep_hits；安全省请求标签为 skip_hits >= keep_hits 且 skip_reads < keep_reads。不假定原本跳过的候选安全。标签只用于监督和评价，不进入历史特征。

特征：原 17 维；此前已判定候选数、实际跳过数/比例、连续跳过数、距最近跳过的候选数、有无跳过；此前被跳过候选 PQ 距离均值/标准差/极值/最近值、当前相对该均值差和比值；前一候选以来 candidate/best/worst PQ、rank、queue/unexpanded、visited/expanded、beam round、best exact、两个距离比值的变化、前一候选存在标志、跨轮 visited 增量/expanded 增量。历史按 query 初始化，并在当前决定后更新；cached 候选也更新序列但始终保留。没有被跳过候选的 exact 距离，只有读取前 PQ 距离。

对照：原模型直接迁移；在本轮训练标签上重训 17 维 LR；17+剪枝历史；17+距离历史；17+搜索变化；完整 44 维 LR。均 StandardScaler + LogisticRegression(C=1,class_weight=balanced,max_iter=1000)，仅训练集拟合，随机种子42。另设完整 44 维非线性模型 HistGradientBoosting(max_iter=150,max_leaf_nodes=15,l2_regularization=1,min_samples_leaf=100,random_state=42)，检查非线性可预测性；不据测试指标选模型。

以验证集安全省请求正类覆盖率 10%/25%/50% 冻结各模型阈值，比较测试关键候选误剪率和请求减少标签覆盖率，避免模型仅靠少剪枝获得安全优势。另设验证关键误剪预算 0.1%/0.5%/1%，报告测试实际误剪和覆盖。相同验证覆盖并不保证相同测试覆盖，必须一起展示。AP/AUC作为补充。用 query cluster bootstrap（1000次、seed42）报告主比较：完整 LR 相对重训17维 LR 在25%验证覆盖处的测试关键误剪率差的95%区间。

这是冻结 LR 策略分布上的条件可预测性对照。增强模型将改变后续历史和轨迹，故不能仅由本实验宣称部署后 Recall 或净读取收益。标签中的逐候选请求差不能相加当作查询净收益。所有读取计数均为 submitted SSD requests。

补充诊断：单独展示原冻结LR已误剪的关键候选，在各冻结阈值下还有多少被增强分类器判为跳过，同时展示原LR剪枝决定的接受比例。该子组样本很少，不能替代全体关键事件和query bootstrap主比较。
