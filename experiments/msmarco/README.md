# MS MARCO 全局一次标记、LR pruning、LR＋复用

用户确认的实验目标：使用 MS MARCO 的真实搜索 Query 和文档文本，以同一个 Embedding 模型编码，构建向量索引并对实际索引语料计算精确 Top-10；比较全局一次标记、LR + Pruning、LR + Pruning + 相似查询 skip 复用，并以 DiskANN 为基线。这里的真实查询指数据集中的真实搜索 Query，不要求生产请求时间戳或滚动缓存。

当前实际状态：`out_1m` 已完成百万文档、训练/验证/测试各 1000 条查询的实验，结果见 `out_1m/REPORT.md`；`full` 未完成，不能将其旧阻塞记录解释为百万实验的状态。用户已确认使用现有 `collection.tsv.gz`、`train_queries.tsv`、`dev_queries.tsv`，保留当前百万规模数据与结果。这些文件与 `download.py` 所列 Web Search SimANS 预编码资源是不同输入，本轮不切换数据源，也不将现有输入标为已核实的 Web Search SimANS 版本。

## 从本机终端运行

在能够联网的终端安装依赖、下载编码模型，然后启动全流程（默认 CPU）：

```bash
cd /home/lxyue/projects/learned-gate
PIP_CACHE_DIR="$PWD/experiments/msmarco/pip-cache" .venv/bin/python -m pip install -r experiments/msmarco/requirements.txt
.venv/bin/python experiments/msmarco/download_model.py
bash experiments/msmarco/run.sh --stage all --offline
```

若终端中的 GPU 可用，可追加 `--device cuda`。当前受限环境的 GPU 不可访问。下载脚本将固定版本模型保存到 experiments/msmarco/models/all-MiniLM-L6-v2，流水线自动离线加载。也可用 --model-path 指定已有完整模型目录；只安装 sentence-transformers 软件包并不包含模型权重。启动脚本保留已有 HF_HOME 和默认缓存。

全流程处理全部 8,841,823 段文档；精确 GT 需要扫描全部文档向量。编码和 GT 都是大计算任务，脚本前台运行，退出终端不会自动成为后台服务。

也可按阶段执行 `--stage prepare|encode|truth|build|collect|fit|evaluate`。编码已完成的 4096 条分片会校验文本后复用；GT、建索引和评估阶段重跑会重新计算。中断后可从未完成的阶段继续，不必重做之前阶段。更改模型或语料规模应使用新的 `--out`。

`build`、`collect`、`evaluate` 和 `all` 会先用 CMake 增量编译对应原生程序，日志为输出目录的 `native_build.log`。这需要已有的 `DiskANN/build` CMake 配置与编译依赖。旧版 `collect_io_labels` 不支持 `LRGATE01`，会对有效的 LR 文件报 `Invalid MLP export`；更新源码后必须同步编译。2026-10-18 已重新构建，并在 `out_test` 的一个真实查询上通过特征采集与 singleton 删除验证；这不代表完整实验已完成。

## 实验定义

- 默认模型：[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，固定 revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`。同一模型编码文档与查询，384 维 float32，单位归一化后使用平方 L2。模型默认截断长文本，实际 max_seq_length 写入 embeddings.json。
- 输入来自 msmarco-data/collection.tsv.gz、train_queries.tsv、dev_queries.tsv。旧 Web Search vectors.bin 不参与。
- train 从 train_queries 选 1000 条，validation/test 从 dev_queries 各选 1000 条。种子 20261017，跨段排除相同文本。剩余查询未纳入本轮；这是查询集实验，不宣称完整真实请求流。复用库不包含验证/测试查询。
- GT：对实际 base 做 float64 穷举 L2，K=10；同距离按 base 行号确定顺序。不是 qrels 相关性评价，也不使用旧 Web Search GT。
- DiskANN：R=64、Lbuild=100，建索引内存预算 8 GB，搜索压缩预算 0.1 GB。评估 L=100、beam=4、K=10、4 线程、BFS cache=10000，所有方法共享索引。
- 全局一次标记：训练原始检索中，节点至少有一次 cache miss，且所有已观测 miss 都 admitted=0 且不在该查询最终 top10 才标记。完整位图冻结；不拿 singleton 标签或 GT 构造标记。
- LR：与现有实验相同的 17 维搜索状态；训练标签为 singleton 删除后 Recall 不降且提交读取数减少。训练仅使用 train，阈值候选仅来自 validation。阈值网格包括 0。
- 验证 2 轮，在 0.5/1/2pp Recall 损失预算内按读取数选 LR 阈值；测试前写 selection_lock.json。测试 3 轮。完整全局一次标记单独报告，不为满足预算修改定义。
- LR＋复用：采用相同 LR 模型、阈值与固定 1000 条训练查询库，rho<=0.01；未命中仍调用 LR。使用已有 search_query_skip_cache 实现；库路径绑定模型、阈值、索引和库查询的哈希。没有滚动更新、到达时间或排队实验。
- 空标记、LR 阈值 0 必须与 DiskANN 结果一致；lookup-only 必须与 fresh LR 结果逐字节一致。模型导出时检查 float32 原生参数与 sklearn 概率一致。
- 报告 Recall@10、提交读取请求数、平均/p95/p99 延迟和 QPS。复用 CSV 另存模型调用、复用次数和查找耗时；`.bank.json` 保存增量缓存内存明细。提交读取数不代表物理 SSD I/O。
- 现有 native benchmark 为预热后的测量；未实现冷启动、滚动库或相同总内存预算比较，不能据此作这些结论。普通 LR 与复用在不同 native harness 中运行，主复用耗时收益应比较 memo 文件中的 fresh LR 与 reuse。

`full/inputs.json` 记录源文件 SHA256 和划分；texts.json 保存实际查询及 ID；base_shards/*.json 保存向量行与 passage ID 映射。runtime.json 保存代码、native binary 哈希及环境。最终产物为 results.json、REPORT.md、COMPLETE.json；只有成功结束评估才写 COMPLETE.json。

## 快速代码验证

```bash
cd /home/lxyue/projects/learned-gate
.venv/bin/python experiments/msmarco/test_pipeline.py
```

测试覆盖三列/两列查询解析、重复 ID 拒绝、截断向量拒绝、GT 与独立 SciPy 距离结果（含 ties）一致、全局标记对高贡献/最终结果/cache hit 的处理，以及 LR 原生文件格式。尚未完成 MS MARCO native 端到端验证。
