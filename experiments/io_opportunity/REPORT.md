# DiskANN + Pruning：访问价值与可避免读取空间

固定现有索引、各 1,000 条测试查询，L=100、k=10、beam=4、BFS 缓存 10,000 节点。展开预算不设硬上限；联合跳过最多使用参考展开集合内的节点。所有质量约束均逐查询执行。完整口径见 [PROTOCOL.md](PROTOCOL.md)。

以下选取的是已经实际重放的可行组合；测试反馈用于事后选掩码，收益不代表可部署模型的泛化能力或算法无关上界。Avoidable I/O Ratio 按省掉的参考读取计算；净节省率进一步扣除新增读取。

| 数据集 | 参考搜索 | 质量约束 | Submitted SSD Read Requests/查询 | Avoidable I/O Ratio | Net I/O Saving | 受益查询 |
|---|---|---|---:|---:|---:|---:|
| SIFT1M | original | 返回 ID 顺序及距离相同 | 102.405 | 80.73% | 69.51% | 1000/1000 |
| SIFT1M | original | 逐查询 Recall@10 不下降 | 102.405 | 81.43% | 69.80% | 1000/1000 |
| SIFT1M | pruning | 返回 ID 顺序及距离相同 | 89.702 | 66.91% | 51.99% | 999/1000 |
| SIFT1M | pruning | 逐查询 Recall@10 不下降 | 89.702 | 67.70% | 52.34% | 1000/1000 |
| DEEP1M | original | 返回 ID 顺序及距离相同 | 102.199 | 81.38% | 70.83% | 1000/1000 |
| DEEP1M | original | 逐查询 Recall@10 不下降 | 102.199 | 82.31% | 71.26% | 1000/1000 |
| DEEP1M | pruning | 返回 ID 顺序及距离相同 | 87.627 | 68.73% | 59.13% | 997/1000 |
| DEEP1M | pruning | 逐查询 Recall@10 不下降 | 87.627 | 69.83% | 59.67% | 1000/1000 |

I/O 计量单位统一为 **Submitted SSD Read Requests**：O_DIRECT Reader 提交的索引读取请求，不是 Physical SSD I/O。历史 CSV 的 `reads` / `ssd_reads` 字段保留兼容，但含义同此口径。

Cache Miss Expansion Utility Distribution（原始搜索的查询—节点展开事件）：

| 数据集 | 冗余节点 | 质量关键节点 | 替代读取节点 | 高收益诊断子类 |
|---|---:|---:|---:|---:|
| SIFT1M | 91150 (89.01%) | 9960 (9.73%) | 1295 (1.26%) | 380 (0.37%) |
| DEEP1M | 90849 (88.89%) | 9925 (9.71%) | 1425 (1.39%) | 452 (0.44%) |

冗余：跳过后 Recall 不下降且请求数减少；质量关键：Recall 下降；替代读取：Recall 不下降且请求数不减少。这三类互斥且覆盖全部 Cache Miss 展开。高收益诊断子类是替代读取中请求数严格增加的节点，说明保留它可能避免后续请求；它不是第四个可相加类别，且尚未验证优先调度或提前收敛。

该分布按查询—节点事件加权，另保存逐查询分布；目前只对 original 参考保存了逐节点反事实明细，pruning 的剩余空间来自联合重放，不能据此声称已获得 pruning 后的节点效用分布。

Strict Result-preserving Saving 定义为返回 ID 顺序及距离完全一致时的净请求节省率：

- SIFT1M：原始 69.51%；pruning 后 51.99%。
- DEEP1M：原始 70.83%；pruning 后 59.13%。

原始与已有 pruning 的参考 Recall@10（GT 前十 ID，不扩展距离并列）：

- SIFT1M：原始 99.67%，pruning 99.63%。Pruning 剩余空间相对于自身质量成立。
- DEEP1M：原始 99.47%，pruning 99.37%。Pruning 剩余空间相对于自身质量成立。

单节点访问价值与缓存状态（原始参考）：

| 数据集 | 缓存状态 | 展开次数 | 单节点跳过质量安全 | 安全且读取减少 | 安全但读取增加 |
|---|---|---:|---:|---:|---:|
| SIFT1M | cache_hit | 9436 | 8381 | 402 | 1715 |
| SIFT1M | cache_miss | 102405 | 92445 | 91150 | 380 |
| DEEP1M | cache_hit | 9991 | 8856 | 434 | 1659 |
| DEEP1M | cache_miss | 102199 | 92274 | 90849 | 452 |

这些单节点统计不能相加得到联合收益；候选被跳过后，队列、模型决策和后续展开都会改变。联合结果以完整重放为准。

不加质量回退的 zero-admission 联合掩码诊断：

- SIFT1M / original：Recall 下降 0 条，读取增加 0 条，返回完全一致 1000/1000。
- SIFT1M / pruning：Recall 下降 0 条，读取增加 7 条，返回完全一致 999/1000。
- DEEP1M / original：Recall 下降 0 条，读取增加 0 条，返回完全一致 1000/1000。
- DEEP1M / pruning：Recall 下降 3 条，读取增加 0 条，返回完全一致 993/1000。

验证：两数据集原始逐查询读取数、展开数与已有 r0 baseline 一致；每次原生重放均验证未缓存展开数等于 n_ios；汇总验证联合省略请求减新增请求等于净节省。

产物：[summary.json](summary.json)、单节点 candidates.csv、联合 queries.csv、[provenance.json](provenance.json)。[run.sh](run.sh) 可重跑；原生 runner 为 `DiskANN/apps/search_io_opportunity.cpp`。

该轮没有测量物理 SSD 计数，也没有将事后反事实计算开销计入部署收益；未证明最优联合掩码。进一步落地须在独立验证集学习可用特征，并在独立测试集评估选择开销、质量与真实净 I/O。
