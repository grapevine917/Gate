# Global / Candidate / Full17 特征消融

主约束：Recall@10 ≥ 99.44%。每组3个训练seed、每个选中工作点5轮随机交错计时。模型训练、审计和profile均先于正式计时结束。

I/O最低的组（要求3个seed的评估点都满足约束）：Full17。计入推理后实测平均延迟最低的组：Full17。
延迟最低组与其他可行组的配对95%区间是否均排除零：False。若为False，只能报告实测最低，不能证明稳定延迟优势。

|组|维数|参数量|平均Recall %|最低seed Recall %|SSD请求/q|profile模型开销 µs/q|含推理平均延迟 µs|p95 µs|全部seed达标|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Global|11|9857|99.463|99.460|76.444|106.421|6256.889|8643.714|True|
|Candidate|6|9217|99.437|99.410|77.701|102.428|6467.160|8515.639|False|
|Full17|17|10625|99.470|99.450|75.309|110.092|6178.151|8638.814|True|

## 特征定义

两组输入列互不重叠，合并覆盖全部17维。Candidate中的相对距离仍按原定义使用全局best/worst作分母，列互斥不等于信息统计独立。beam_position是候选在当前beam内的位置，放入Candidate组；beam_step是搜索进度，放入Global组。search_L和beam_width是固定上下文，在此配置中为常量。缓存标志作为Candidate输入，但命中节点沿用既有保护策略，不执行模型。

- Global: best_pq_distance, worst_pq_distance, queue_size, unexpanded_count, visited_count, completed_expansions, beam_step, best_exact_before_beam, has_best_exact, search_L, beam_width
- Candidate: candidate_pq_distance, candidate_queue_rank, beam_position, cached_node, candidate_pq_over_best, candidate_pq_over_worst
- Full17: candidate_pq_distance, best_pq_distance, worst_pq_distance, candidate_queue_rank, queue_size, unexpanded_count, visited_count, completed_expansions, beam_step, beam_position, cached_node, best_exact_before_beam, has_best_exact, candidate_pq_over_best, candidate_pq_over_worst, search_L, beam_width

## 公平性与计时

三组重训 d→128→64→1 MLP，训练894077条状态、8000条Query；验证和评估各1000条Query，互不交叉。标签沿用原始保守retain标签；本次消融原始剪枝MLP，关闭历史复用及二次风险评估。输入宽度改变首层参数量，隐藏层和优化器完全相同。标准化只拟合训练集，早停只看验证logloss。

阈值候选只由验证标签分数确定，再通过真实验证图搜索选择满足绝对Recall下限的最低请求点。评估集不调阈值，是否仍满足约束逐seed记录。主选点以I/O最小化选择，报告该点的真实端到端延迟；这不是对所有可能阈值的延迟全局最优证明。

相同SIFT1M索引、L=100、K=10、beam=4、线程4、BFS缓存10000。CPU固定到4个可用的不同物理核心；不能排除其他宿主任务干扰。所有方法使用同一原生程序。验证和曲线扫描以4进程并行，只比较确定性Recall与I/O，扫描阶段的延迟不用于性能比较；profile和正式计时均串行执行。完整17维live state仍统一构建，再映射取子集输入；未对每组单独改写特征构造。这使结果反映模型输入消融，而非各组手工系统优化。

无profile的正式计时包含state构造、输入映射、标准化、MLP评分、距离计算、实际读取和搜索bookkeeping；不包含模型/索引加载、预热和结果文件写出。单独profile用于记录模型区域开销，存在计时器扰动，不能与无profile总延迟直接相减。SSD请求是reader提交请求数，不是物理设备读盘计数。

三组3个seed按seed配对，5轮计时按轮配对、相同Query按ID配对；2000次嵌套bootstrap区间保留这些层级，配对区间未作多重比较校正。只有3个训练seed，且评估集以前已被观察，本轮为探索性比较，不是全新确认性实验。

## 产物

`primary_group_comparison.csv`：主约束三组对比；`paired_group_differences.csv`：配对差值区间；`selected_recall_constraints.csv`：所有Recall下限的验证选点及评估达标情况；`recall_io_curves.csv`：全部真实曲线；`all_run_summary.csv`：所有原始运行汇总；`profile/`：单独推理成本profile；`timing_r0`至`timing_r4`：正式计时及Top10；`native_audit.json`：概率映射核验；`models/`：每组每seed的权重、训练过程、阈值。

复现：`.venv/bin/python experiments/feature_group_ablation/run.py`。输入改变时使用新目录。
