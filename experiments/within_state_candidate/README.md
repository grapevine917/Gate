# Within-State Candidate Discrimination

本实验的结果与逐样本记录已保存，原始模型不重新训练。

- `primary_summary.csv`：实际近似同状态、原生概率分数的主指标。
- `within_state_metrics.csv`：实际近似状态与固定Global对照、原生概率与未饱和logit、全部阶段及目标的完整指标和区间。
- `paired_model_differences.csv`：Full17/Global/Candidate及简单排序基线的配对差值。
- `candidate_phase_metrics.csv`、`phase_definitions.json`：Candidate-only分阶段结果及训练集确定的边界。
- `pair_support.csv`、`natural_exact_support.csv`：每项指标有效候选对、Query与Beam数量，目标相同而排除的对数。
- `decision_records.csv`、`records.npz`：全部候选的17维状态、模型分数和实际邻居贡献。
- `candidate_pairs.csv`、`pairs.npz`：全部同Query、同Beam非缓存候选对，未经结果筛选。
- `contributions.npz`：单节点反事实的命中损失、请求节省、展开变化与A类标签。
- `intervention.csv`、`intervention.ids`、`intervention.baseline.ids`：完整反事实计数与最终Top10；IDs为uint64，逐请求10个。
- `sham.*`：1000次Keep→Keep对照。
- `figures/phase_*.svg`：分阶段条件AUC图。
- `verification.json`：字段、贡献、前缀、结果和候选配对核验。

同一Beam内，实际Global状态只有unexpanded_count会随候选选择递减，其余10维完全一致。故实际数据属于“近似同状态”；完全相同Global状态的实际候选对没有支持，指标记NA。另将Global固定为真实Beam起始快照后离线重评分，构造严格同Global对照；它不是实际执行的新搜索。Global-only在该对照中平分是构造的必然结果，不是独立的模型劣势证据。完全相同Global输入每Beam只取一次评分并广播，避免BLAS分块浮点舍入产生虚假的候选排序；不使用标签进行此数值控制。原生float32概率可能饱和，额外保存未饱和logit排序，主指标仍用实际原生概率。

贡献包含真实处理顺序下入队邻居数，以及全部非缓存候选的独立单节点Skip最终Recall/I/O效应。前者受同Beam邻域重叠与处理次序影响，不能等同检索必要性；后者是单次删除的边际效应，不等同联合剪枝效果。缓存节点被原策略保护，仅记录、不进入主候选对或反事实分母。旧E4-1标签来自不同的已剪枝路径，本实验没有直接复用这些标签。

Pairwise Accuracy只比较贡献数不同的候选；条件AUC只比较同Beam正负标签候选，分数并列计0.5。条件AUC等于这类二值候选对的Pairwise Accuracy，不是额外独立证据。所有目标并列、无可比较候选的Beam/阶段均报告支持数量，不填造精确率。置信区间按1000个Query聚类，并配对重采样3个训练seed，不把候选对当作独立样本。Query宏平均与Beam宏平均另列，避免只有大Beam主导。

这些查询先前已被观察，本轮是探索性诊断；反事实GT和未来贡献不输入模型，不参与训练或匹配边界选择。模型采用上一特征消融实验的9个冻结权重，标签在所有模型共享的无剪枝轨迹上定义，不混合各自已剪枝后的轨迹。

复现：`.venv/bin/python experiments/within_state_candidate/run.py`。协议/模型/原生输入变化须使用新目录。原E1–E4及特征组消融产物不修改。
