# 原生 decision_trace 与离线分析

新增的是 `DiskANN/build/apps/search_disk_index` 的可选实验模式，默认仍为 `search`。
核心 `pq_flash_index.cpp`、模型推理、阈值及节点缓存均未修改。此入口不进行跨 Query 决策复用；每次搜索的逻辑状态由原有函数重新构造。

```bash
export DISKANN_M3_MODEL="$PWD/experiments/sift1m/online_state_mlp/model.bin"
export DISKANN_M3_TAU=0.024352464824914932
DiskANN/build/apps/search_disk_index \
  --data_type float --dist_fn l2 --index_path_prefix experiments/sift1m/index \
  --query_file experiments/e1_similarity/data/test.fbin \
  --gt_file experiments/e1_similarity/data/test.gt \
  --result_path /tmp/test_results --recall_at 10 --search_list 100 \
  --beamwidth 4 --num_threads 4 --num_nodes_to_cache 10000 \
  --experiment_mode decision_trace --decision_trace_enabled true \
  --decision_trace_prefix /tmp/test_trace \
  --decision_trace_query_ids experiments/decision_trace_acceptance/test_ids.txt
.venv/bin/python experiments/decision_trace/convert.py /tmp/test_trace.manifest.json /tmp/test_npz
```

`--decision_trace_query_ids` 是文本，每行一个唯一 uint32 ID，顺序与输入向量行一致。省略时使用本文件行号；历史与测试应显式传入全局 ID，避免两个文件的行号冲突。向量单独保存为 fbin，并保存行号至全局 ID 的映射。

`--experiment_mode search` 或 `--decision_trace_enabled false` 均走原有模板实例；不读 Query ID 文件，不创建日志目录。追踪开启仅支持当前 STGATE01 17 维 MLP、float、无过滤 L2，无重排。其他剪枝/审计 hook 同时开启会报错。原先 LR/XGBoost 和全局一次标记的 E1–E3 对照保留在原目录。

输出默认拒绝覆盖，只有显式 `--decision_trace_overwrite true` 才覆盖。每个 L 一份记录；manifest 的 `complete` 只有全部搜索完成才变为 true。不要让多个进程使用同一输出前缀。

记录包含 Query ID、Node ID、17 维状态、retain 概率、实际 Skip/Expand、`decision_step`（候选决策序号）、`search_step`（原生 hop 计数，state[8]）、`cache_hit`（state[10]）和 `model_invoked`。1=Skip，0=Expand；阈值以下为 Skip。缓存固定为原搜索入口构建的节点缓存，记录按输入 Query 顺序导出，OMP 调度不改变文件排列。追踪时使用额外内存和复制，文件写出在搜索计时结束后，追踪开启的时间不能作为原模型性能结论。

**缓存语义区别：**原生 STGATE01 对缓存命中节点也执行 MLP；此前 E1 采集器显式启用了 cache-miss-only。新入口保留原生行为，不能把两者的 `model_invoked` 分母直接混合。分析使用每份日志自己的实际模型决策集合；两类结果分别保存。

离线分析可以直接读取转换后的 NPZ，或已有 E1 trace：

```bash
.venv/bin/python experiments/decision_trace/analyze.py \
  --history experiments/e1_similarity/MLP_history_trace.npz \
  --test experiments/e1_similarity/MLP_test_trace.npz \
  --query-vectors experiments/e1_similarity/data/query_vectors.npy \
  --normalization experiments/e2_state_reuse/normalization.npz \
  --output /tmp/reuse_analysis
```

`query-vectors` 为全局 ID 可直接索引的 N×D `.npy`。标准化 mean/std 必须来自训练集；主实验沿用 E2 已冻结的训练统计。状态距离使用 `(s_i-s_j)/(std+1e-8)`，与先减 mean 再相减完全等价。状态阈值从历史 Query 的同节点状态对校准，不用测试集拟合。Query 距离分位数仅作描述性分桶，同时报告每个测试 Query 的邻居排名，避免远距离对支配结果。

输出 Query 分桶、每 Query 邻居排名、G1–G4、累计 State Coverage 及 Query×State 联合 CSV、summary.json、PNG/SVG/PDF。Coverage 的分母为测试 Query 全部实际模型决策；无共同节点的 Query 对贡献零覆盖。Agreement、Skip Precision 的零分母不进入均值，另报缺失比例。False Skip Rate 指相对于当前 MLP 的误剪率，实际检索风险仍以 E2 干预及 E3 真正图搜索结果为准。

验收源代码：`build.py`、`accept.py`、`verify_fields.py`。验收结果与日志在 `../decision_trace_acceptance/`。`build.py` 使用现有核心静态库，仅编译搜索 app；保留的原始源代码与原始二进制用于对照。重新完整运行验收前，应将已有验收输出归档，避免覆盖保护中止。
