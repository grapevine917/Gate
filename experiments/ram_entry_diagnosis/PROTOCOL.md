# RAM 入口价值反事实诊断

使用上一轮固定节点、逐源出度、有效边预算的 Original、Geometry 和验证选择的 History 图，保持底层 Vamana 图及缓存节点集合不变。每个数据集从独立测试集 1,000 条中，用 seed=20261015 无放回随机取 200 条，三张 RAM 图使用相同查询。SIFT1M 和 DEEP1M，无 GloVe 结果。

当前 RAM 导航 budget=64、queue=64；当前入口为实际展开节点中精确距离最近四个。底层仍 k=10、L=100、beam=4，所有试验保留原始 medoid 候选，SSD 路径无剪枝/早停。一次 trial 都重新执行完整原始 best-first 搜索，静态缓存状态一致。节点价值不是与查询距离或是否 GT 节点，而是该候选作底层入口时的实际反事实读取请求与最终 Recall。

候选池为该查询最多 64 个 RAM 已展开节点，合并全缓存节点中精确距离最近 32 个；后者需要全缓存距离扫描，本轮只作诊断。没有读取候选完整 SSD 邻接表作为免费特征，也没有改变缓存驻留节点或底层邻接表。

对照：
- no_navigation：同一缓存集合，仅原始 medoid 入口。
- current4：目前实际展开节点中距离最近四个。
- discovered4：RAM 已计算过距离的所有发现节点中距离最近四个，包括尚未展开的节点。诊断程序全扫描重建排序，没有将该排序实现成可部署计时版本。
- cache_nearest4：全缓存最近四个，扫描 9,000/8,850 个向量的选择耗时单独记录。这组在三张图中应有完全相同的底层结果，因相同节点集合与固定四入口。
- oracle_expanded_single / oracle_pool_single：评估候选分别作为唯一新增入口的 bottom search，选择不降低 current4 的逐查询 Recall 且请求更少的候选；无收益时回退 current4。因此 seed 数可能为 1 或 4。这两组是候选价值诊断，不是固定四入口的主比较。
- oracle_expanded4：从已展开节点候选中，先按单入口的质量合格/请求数排序；取最好的八个，用反事实搜索逐步添加节点，构成恰好四个入口。每步保持该查询 Recall>=current4，四入口组合不改善则回退 current4。
- oracle_pool4：在扩展候选池中用同一启发式寻找四入口，并允许回退 oracle_expanded4 和 current4，因此候选空间嵌套。不是穷举组合，也不是理论上界。

“有价值候选”定义为单入口反事实 Recall>=current4 且 Submitted SSD Read Requests<current4。组合可能存在协同/替代，单入口价值不是充分必要条件，最终四入口必须直接重放确认。

质量约束是每个查询 Recall@10 不低于其 current4，原始 medoid 保留且新增入口数固定四个。它不意味着完整 Top-10 ID 集保持，也不保证超过 no_navigation 的逐查询 Recall。分别报告 Recall、完整结果保持比例与更少请求查询数。

Oracle 显式使用测试反馈，仅回答当前候选和入口选择的机会空间；不训练、不调阈值，不可把其收益当作独立部署效果。所有反事实试验的请求数和次数记录在 diagnostic_cost，选择开销不包含在 Oracle 底层搜索结果中。计时受大量反事实重复及暖机顺序影响，不用于 QPS 或端到端部署收益声明。提交请求与设备物理 I/O 不等价。

验证：current4 逐查询请求数和 Recall 与上一轮完整测试记录完全一致；cache_nearest4 三图逐查询底层结果一致；Oracle 四入口种子数恒为四、逐查询 Recall 不下降且请求不增加；保存种子 ID、候选距离排名/是否展开/是否被选及每次单入口结果。查询配对 bootstrap 5,000 次评估均值差的区间。

复现：先 `cmake -S learned-gate/DiskANN -B learned-gate/DiskANN/build`，再 `cmake --build learned-gate/DiskANN/build --target search_ram_entry_diagnosis -j4`，随后运行 `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 learned-gate/.venv/bin/python learned-gate/experiments/ram_entry_diagnosis/run.py`，最后运行同目录 report.py。

补充底层候选诊断：在同一 200 条查询的历史图上，候选扩展为 current4 底层搜索展开节点、no_navigation 底层展开节点、RAM 展开节点及精确 GT Top-10 的并集。逐节点单入口反事实，再按同样八候选贪心规则构造四入口并直接重放。GT 仅作离线诊断，不是任何部署输入。报告这些高价值入口是否缓存驻留、是否已被 RAM 展开。`gt_nearest4` 使用 GT 最近四节点独立重放，质量不保证；`oracle_bottom4` 回退 current4，逐查询 Recall 保持且请求不增加。此底层候选池与全缓存最近32池并不互相包含，两个启发式 Oracle 不能相减作严格因果分解，也不代表全图最优入口。另运行 `run_bottom.py` 生成补充数据。
