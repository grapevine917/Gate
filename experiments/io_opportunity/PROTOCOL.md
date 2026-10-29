# 候选访问价值与 Cache Miss 联合反事实实验

本实验估计限定搜索条件下的机会空间，不是算法无关的理论上界，也不是可部署选择器的预测效果。测试查询本身参与反事实决策，因此不能将 Oracle 的收益作为独立测试泛化收益。

固定既有 SIFT1M / DEEP1M 索引与原有 1,000 条测试查询；L=100、k=10、beam=4、四线程、固定 BFS 缓存 10,000 节点，不使用重排和 early termination。L 是候选队列容量，**不是节点展开预算**；本轮展开节点数和读取次数不设硬上限，沿用原始终止条件。跳过消耗原来的 beam 槽位，发生在缓存查询与 exact scoring 之前，保留原始 PQ best-first 策略。每个查询从空 scratch 状态重放，缓存不在线更新。

两个参考搜索：原始 DiskANN，以及已有 Top3 pruning（验证集冻结 FNR=0.5% 阈值：SIFT 0.03665834292769432，DEEP 0.02848522551357746）。后者的机会空间相对于 pruning 自身的返回质量，不代表恢复到原始搜索质量。

质量约束为逐查询 Recall@10 不低于对应参考结果；采用 GT 前十个 ID 的集合命中数，不扩展距离并列。附加严格约束为前十返回 ID 顺序及 float 距离完全相同。结果完全一致是比 Recall 相同更强的条件。

对每个参考展开节点分别执行完整的单节点跳过重放，包括缓存命中与未命中、最终结果节点。节点访问价值以 Recall 变化、结果一致性和 I/O 差值联合定义；zero admission 只是一种候选标签，不能代替反事实价值。随后重放两种联合掩码：zero-admission 且非最终 top-k 的整组跳过；按原展开顺序逐个加入单节点质量安全且减少读取的 miss 节点，每次完整重放，只接受逐查询 Recall 不下降且读取不增加的组合。严格对照使用 zero-admission 联合掩码，若结果不完全一致则回退参考搜索。节点跳过预算为该查询原始展开集合大小，不做全局最优组合证明。

I/O 口径为原生 O_DIRECT reader 提交的索引读取请求数（QueryStats.n_ios）；Cache Miss 指固定 DiskANN 节点缓存未命中。它不是硬件 SSD miss、NVMe 内部缓存 miss 或实际设备计数。当前不开启 reorder，每个未缓存展开对应一个请求，代码逐查询断言该计数关系。缓存构建、索引加载与模型加载不计入查询分母。多个节点同一 sector 的请求仍分别计数，未按 sector 去重；这是 reader 的实际请求口径。

Avoidable I/O Ratio = 质量可行联合重放中未再访问的参考 miss 节点对应读取请求数 / 参考总读取请求数。同时报告新增替代读取以及 **净 I/O 节省率 = (参考请求数 - 反事实请求数) / 参考请求数**。只有遗漏请求、没有新增请求时两者相等。不可行查询按参考搜索回退，贡献零收益；原始重放质量也单独保留。不能把所有单节点安全跳过或单节点 I/O 差值相加。

程序：DiskANN/apps/search_io_opportunity.cpp。原生搜索算法没有被改写，复用已有 gate_trace 和 gate_skip_entire_node。CSV 包含完整单节点重放与联合重放；summarize.py 校验查询数、展开计数和净请求守恒。本实验多次反事实执行会影响墙钟缓存与设备负载，因此不把此次并发反事实耗时作为部署延迟或吞吐量结果。

## Cache Miss Expansion Utility Distribution

统计单位为 original 参考搜索中的查询—节点 Cache Miss 展开事件，而不是去重节点数。使用同一查询的完整单节点跳过重放，令 ΔR = 参考 Submitted SSD Read Requests − 跳过后的 Submitted SSD Read Requests。

- 冗余节点：Recall@10 不下降且 ΔR > 0，优先剪枝候选。
- 质量关键节点：Recall@10 下降，不论 ΔR 符号均归入此类。
- 替代读取节点：Recall@10 不下降且 ΔR ≤ 0，仅为减少请求而剪枝没有收益。
- 高收益诊断子类：替代读取节点中 ΔR < 0，保留该节点在当前策略下避免了部分后续请求。它与替代读取类重叠；不将四项相加。提前收敛和优先调度收益需要额外的顺序干预实验，目前没有测量。

前三类互斥且完整覆盖所有 Cache Miss 展开。输出整体事件加权比例、逐查询计数和逐节点标签。Pruning 参考目前只有联合重放结果，未保存其单节点明细，因此本轮不生成 pruning 后节点效用分布。

论文中 I/O 口径统一写作 **Submitted SSD Read Requests**，不要写作 Physical SSD I/O。Avoidable I/O Ratio 是联合重放省略的参考请求比例；Net I/O Saving 是扣除替代请求后的净节省数及比例；Strict Result-preserving Saving 是在返回 ID 顺序及距离完全一致条件下的净请求节省。Pruning 后剩余优化空间采用 pruning 自身的分母和质量基准，不能当成相对 original 的额外百分点直接相加。
