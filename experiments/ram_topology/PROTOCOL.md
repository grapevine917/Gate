# 固定节点与有效边预算的拓扑对照

复用 ram_navigation 的 B3 节点集合，底层 DiskANN Vamana 索引保持不变。SIFT1M 与 DEEP1M；本轮没有 GloVe 结果。每个图保持完全相同的节点 ID、原始入口、逐节点出度、有效非自环且唯一的有向边数、CSR 容量与实际 RAM 占用。不存在不计预算的环或补充边。原始缓存子图的零出度节点也保留零出度；因此这是对连接目标的严格对照，不评价重新分配源节点出度的收益。

- Original：底层 Vamana 图在固定缓存节点集合上的全部已有连接。
- Geometry：缓存向量精确近邻候选 128 个；按 alpha=1.2 几何多样性剪枝，不足源节点原始出度时按距离补齐，不添加额外连接骨架。
- History：从 Geometry 的 8,000 条独立历史查询 RAM 导航反馈，统计实际展开的源节点到该查询全缓存集合最近节点的候选连接。分数累加相对距离改进 `(source_distance-target_distance)/source_distance`，要求源距离超过目标距离的 1.05 倍。每个源最多使用其出度的 25%、50%、100%（向上取整）历史目标，其余使用 Geometry 连接补齐，去重且无自环。候选队列的连续展开顺序不视为连续图路径。

历史反馈仅用于训练，验证 1,000 条和测试 1,000 条与历史及彼此均不重叠。验证选择 Recall@10 不低于 Geometry 且 Submitted SSD Read Requests 最少的历史比例；若没有合格比例，报告不合格并仅保留 Recall 优先的诊断配置。最终三种拓扑在测试集上固定导航展开预算 64、队列容量 64、交接最近四个实际展开节点，保留原始 medoid 候选。底层 k=10、L=100、beam=4，无剪枝或早停。缓存中的向量和原始邻接信息、导航 CSR 表、元数据、四线程导航 scratch 均计入已有同一 RAM 上限。

每种测试配置串行交错运行三轮，固定四线程及 CPU affinity 环境，延迟/QPS 使用三轮中位数。读取请求数及 Recall 要求三轮完全一致。统计按查询配对的请求数、Recall 差异，5,000 次查询 bootstrap；时间波动不能仅凭三轮当作显著差异。I/O 指标准确称为 Submitted SSD Read Requests，并非设备实际物理读取次数。

历史边中没有充足反馈的源使用几何补齐，因此 History 是历史增强拓扑，并非每条边都由历史决定。报告实际历史选择槽位数、有效边数、入口可达节点数与展开成本。验证质量达标不保证测试质量达标；总体 Recall 不下降也不等于每条查询结果保持。

复现：从 projects 目录运行 `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 learned-gate/.venv/bin/python learned-gate/experiments/ram_topology/run.py`。依赖已构建的 `DiskANN/build/apps/search_ram_navigation` 与现有数据/索引，生成图不会改写底层索引。
