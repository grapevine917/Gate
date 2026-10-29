# DiskANN、既有剪枝基线与query级决定复用统一对照

目标：统一重测同一查询、同一索引和同一运行框架下的绝对Recall、提交SSD请求、模型调用数与端到端延迟，不拼接旧实验的计时结果。

数据：SIFT1M/DEEP1M，自然独立测试各1000查询，来自io_predictability/test；高相似测试各200查询，复用query_skip_cache/near_test的1%/5%扰动混合及独立1M全库float64 GT。两种workload分开报告。相似query缓存使用此前1000训练query生成的bank.bin，测试不写回。

共同配置：L=100, beam=4, K=10, 4线程, BFS cache=10000，不使用ET、reorder、保护Oracle；所有方法仅跳过cache miss，cached候选保留。query复用阈值统一rho<=0.01，直接复用和Guard分别展示，不自动回退；模型、阈值、全局节点位图及query缓存均使用既有冻结产物，不根据本轮测试选参。

预先固定方法：DiskANN无剪枝；17维LR（旧unsafe_fpr_0.005阈值）；Top3 MLP（旧fnr_0.5pct阈值）；LR+Top3 Joint（任一决定skip则skip，双模型真实评分）；LR+Reuse与LR+Guard；Top3+Reuse与Top3+Guard；全局Mark(m=1,r=1)朴素规则和Mark(m=5,r=0.8)保守规则；历史44维LR（旧coverage_0.25阈值）。SIFT另包括旧Rank、Rank+Progress、All17 MLP，均用旧0.5%验证标签FNR档。DEEP没有对应旧Rank/Progress/All17模型，标为未提供，不转用SIFT参数。

历史44维模型此前只用300条训练query且训练于LR剪枝路径；其他旧模型的训练标签/规模也不同。本轮比较既有固定策略，不能把不同Recall的成本点宣称为同等质量胜出。没有进行本轮matched-recall重调参；全部参数来源在CONFIG_LOCK.json保存，并在运行前落盘。

程序一次加载索引、共同BFS缓存、所有模型和query缓存资源；每workload先用完整DiskANN预热，然后每轮配置随机顺序串行运行，各配置内部4线程，共三轮。所有方法使用同一外层计时，含候选处理、特征提取、模型评分及query查找/成员检查，排除模型/索引/缓存初始化和结果写出。报告每轮平均延迟的中位数、p95、中位QPS、三轮范围、确定性Recall/reads/calls，直接列每条策略的两侧绝对值。每条cache-miss的实际模型评分计数，Joint两次评分分别计入；DiskANN/启发式/Mark的模型调用为0。

验证：模型和训练产物SHA256；三轮结果ID/距离逐字节一致、reads/expansions/calls/reused逐query一致；LR/Top3及复用的输出与上轮相同配置逐query精确一致；自然DiskANN结果与前轮原始baseline一致。Mark只来自训练低贡献记录，query缓存只来自历史真实skip，测试没有反馈更新。

收益分母：统一以本轮同workload的DiskANN为reads和延迟基线；复用增量另对同模型的无复用策略计算。两种收益不能混淆。reads为submitted SSD requests，不等同物理设备读盘。此前已经看过这些测试划分，本轮是固定参数下的统一复测，不声称全新未见测试集。
