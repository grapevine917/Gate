# decision_trace 补齐与验收记录

记录时间：2026-10-18T16:40:26.159552+00:00。代码、功能验收、E1 补充分析、原生入口独立分析均完成；**端到端延迟 ±5% 的严格验收未通过，不能标记为“性能完全等价”**。

## 实现与开关

标准 `DiskANN/apps/search_disk_index.cpp` 新增 `--experiment_mode decision_trace`，使用 app 层 `decision_trace.h` 接入已有被动审计 hook。保留核心搜索源代码和静态库 SHA256 不变；保留模型、阈值、缓存和实际决策逻辑。默认 `search` 与显式关闭日志均进入 `DecisionTraceEnabled=false` 模板实例。采集代码受 `if constexpr` 控制。

支持日志启停、输出前缀、全局 Query ID 映射和显式覆盖开关。记录 Query/Node ID、17 维状态、原生 retain 概率、实际 Skip/Expand、候选决策序号、原生 hop 计数、缓存命中和模型调用标志。Query 向量与 ID 映射单独保存；每个 L 独立日志。默认拒绝覆盖，未完成 manifest 不允许转换。使用方法见 [README](../decision_trace/README.md)。

当前原生模式支持 STGATE01 17 维 MLP、float、无过滤 L2、无重排。LR/XGBoost、全局一次标记和 R0–R5/Oracle 对照继续保留于原 E1–E3 实验，未替换或重算。

## 功能验收

固定 SIFT1M、现有 DiskANN 图、L=100、k=10、beam=4、线程=4、固定缓存 10000 节点，tau=0.024352464824914932。历史和测试各 1000 个 Query，ID 与向量均不重叠，顺序沿用 E1。

- 原始保存的二进制、从原始源代码重编译的对照、默认入口、显式关闭日志、开启日志：结果 ID、距离、展开数、SSD reads、模型决策/Skip 计数完全一致。
- 测试集原生工作点 Recall@10=99.640%，每 Query SSD reads=93.410；开关日志不改变这些指标。
- 历史/测试日志各 111,802/111,828 条。全局 ID 和独立 Query 向量映射完全核对。
- 每个分区抽样 2000 条，用 Python 独立 MLP 前向计算：最大分数误差分别 1.79e-07/2.98e-07。全部记录的决策与阈值比较一致。
- 多 L=75/100、默认行号 ID、覆盖保护、显式覆盖、重复/数量错误 ID、无模型、非法模式均验收通过。
- 关闭日志且传入不存在的 ID 文件及输出目录，运行成功且不创建目录，确认没有日志路径访问。
- 编译后查询 OMP 热循环均为 5547 字节、1018 条指令；规范化地址重定位和 ELF TLS 符号位置后指令完全一致。源码和核心库哈希也与改动前一致。详见 `hotpath_verification.json` 和两份汇编。

## 延迟验收的限制

先预热，再以固定随机种子进行 9 轮随机顺序配对，逐次独立启动进程；此期间不编译或运行分析任务。每次搜索仍为相同 1000 个测试 Query。以配对比值的中位数统计，bootstrap 95% CI 以轮为单位：

| 模式相对原版重编译对照 | Mean latency 比值 | 95% CI | p95 比值 | 95% CI |
|---|---:|---|---:|---|
| 默认 search | 1.0761 | 0.9706–1.3549 | 1.1517 | 0.9315–1.3510 |
| decision_trace 显式关闭 | 1.0453 | 0.9680–1.0901 | 1.0859 | 0.9364–1.1670 |

四个 CI 均包含 1，但不处于 ±5% 范围内；默认模式均值及两种模式的 p95 中位数超过预设门槛。故结论是**关闭日志后的结果与热循环无新增采集指令已证实；端到端延迟非劣性未证实**。不能用“不显著”代替等价。原始逐 Query CSV、完整轮次与命令保留在 `runs/` 和 `verification.json`，没有删除慢轮次。当前同一二进制的默认和显式关闭模式调用同一个函数，二者仍出现较大墙钟波动，进一步说明该指标不能仅归因于日志实现。尚未满足的严格性能门槛需要在更稳定、可控制 I/O 的环境复测。

## State Coverage 与联合分析

`experiments/decision_trace/analyze.py` 支持原 E1 NPZ 和原生转换后的 NPZ，冻结 E2 训练集 mean/std（894077 条训练状态），不使用测试集拟合标准化。State 阈值来自各日志历史 Query 的同节点状态对分位数。Query 分桶来自实际全体历史→测试距离分布；同时提供每测试 Query 的近邻排名和分位数。

两套分析各覆盖 1000000 个有向历史→测试 Query 对。所有对进入 Coverage 分母，包括无共同节点、无历史 Skip 的零覆盖对。Agreement 和 Skip Precision 的零分母排除并单独报告比例。无限状态阈值下，新增 State Coverage 必须精确恢复原 Query Coverage；并用 100 个独立 Python 集合计算交叉核验。

| 日志语义 | 同节点事件对 | 历史校准状态对 | 历史状态中位阈值 |
|---|---:|---:|---:|
| E1：cache-miss-only | 20,740 | 21,656 | 2.894251 |
| 原生入口：实际全部 MLP 调用 | 1,713,487 | 1,698,734 | 1.686364 |

E1 补充结果：状态距离不超过训练中位阈值 2.894251 时，Query 对平均 Skip Precision=38.292%，平均 Coverage=0.000812%；不设状态限制时分别为 19.567% 与 0.001824%。这些是**所有 Query 对等权平均**，不等于 E3 选择最近历史 Query 后的模型节省率。收紧状态筛选提高精确率并减少覆盖率，不能只报告一致率。

缓存语义必须分开：原生 MLP 对缓存命中也实际推理，历史/测试各有 9442/9437 个缓存命中决策，其中 4/5 个被 Skip；E1 采集器则显式采用 miss-only 并保留缓存节点。因此原生日志共同缓存节点中的大量 Keep 会抬高 Agreement；该模式的模型调用集合和 SSD 读取集合也不相同。新入口没有为了匹配 E1 而修改原生逻辑。两套输出分开保存，不混合分母。

每套输出包括：

- `query_distance_bins.csv`、`query_nearest_rank_bins.csv`、`query_per_target_quantiles.csv`。
- `joint_groups.csv`：同节点 G1–G4、一致率、Skip Precision、模型参考 False Skip Rate、分数差异及目标 Query 聚类置信区间。
- `state_coverage_cumulative.csv`、`query_state_coverage.csv`：状态累计阈值与 Query×State 分桶，含零覆盖对及零分母比例。
- `summary.json` 与 `figures/{query_similarity,state_coverage,joint_groups}.{png,svg,pdf}`。已检查图像，极低 Coverage 使用可辨识的独立纵轴尺度。

跨节点控制和最终 Recall 干预继续见 [E2 报告](../e2_state_reuse/REPORT.md)；六策略、两类 Oracle、静态回放、真图搜索、五类成本继续见 [E3 报告](../e3_reuse/REPORT.md)。本次未把离线同决策的一致性当作最终 Recall 安全性证明。

## 复现与记录

`build.py` 编译 app 和原始源代码对照；`accept.py` 执行功能与性能对照；`verify_fields.py` 独立验证分数/字段/多 L/保护；`verify_hotpath.py` 对比编译后热循环。所有命令及搜索环境变量逐次写入 `runs/*.command.json`。

验收脚本开发中修正过一次测试命令重复追加 `--search_list` 的问题（原 CLI 会合并该选项，导致重复 L 被正确拒绝），随后多 L 验证通过；没有修改搜索以绕过校验。`artifact_manifest.json` 记录最终源代码、二进制和本轮产物 SHA256。`STATUS.json` 明确区分已完成实现/功能/分析与未通过的严格端到端性能验收。
