# SIFT100M P0-A/B/C

2026-10-23 结果汇总恢复：640 个验证配置（含全部 64 个 q–v 配置）均无换页、reader 与内核字节一致。`Heuristic_L15_p7` 有 1 条查询仅返回 9 个邻居；底层按约定以 UINT32_MAX 填充不足位置。汇总现仅允许尾部该占位值，仍检查有效 ID 范围和唯一性；不足邻居按未命中计算，Recall 分母保持固定 Top-10。全部 640 配置重新核验通过，证据见 `RESULT_PADDING_AUDIT.json`。无需重跑搜索计时，流水线已重新启动，后续状态以 STATUS.json / PROGRESS.json 为准。

2026-10-23 驻留加载修正：仅分离 Python 进程仍在第二个 q–v 批次触发换页。新加载器保留原缓冲区类型和评分实现，在计时前按 16 MiB 分块复制完整压缩向量，并通过 `posix_fadvise(DONTNEED)` 释放已复制的冗余文件页缓存，缓解加载期内存压力。两个曾失败的批次连续三轮共 24 个配置通过，采样进程换页均为 0，reader 与内核字节相等；有旧输出的配置逐查询 ID 和除延迟外的计数一致。证据为 `resident_probe/VERIFIED.json`。正式计时统一使用新二进制重跑，旧计时归档在 `backups/resident_loader/formal_before_amendment/`，输入锁变更仅涉及搜索二进制与加载器头文件，审计见 `RESIDENT_LOADER_AMENDMENT.json`。训练、索引、阈值及配置网格不变。

`PROGRESS.json` 记录当前批次；失败时流水线立即写入 `ALERT.json` 并输出 `EXPERIMENT ALERT`，保留错误配置、时间及日志。当前会话不具备跨轮主动向聊天发送通知的工具；磁盘告警不等同于聊天推送。

2026-10-23 内存恢复：候选评分、校准及可选 Torch 校验迁移至 `candidate_diagnostics.py` 独立进程，退出后由 `evaluate.py` 执行在线搜索，避免保留候选数组及诊断依赖。原失败的 8 个 q–v 配置独立复跑全部通过，采样峰值 RSS 12,647,808 KiB、进程 VmSwap 全程为 0，各配置 reader 与内核读取字节相等；阈值和评分指标文件逐字节保持一致。证据见 `memory_fix_probe/VERIFIED.json` 和 `memory_samples.json`。诊断复跑结果单独保存，不计入正式重复次数。随后通过原恢复入口续跑正式流水线，仍保留输入/模型锁及换页/I/O 断言。

2026-10-23 恢复：运行内核为 `6.6.87.2-microsoft-standard-WSL2`，`kernel_fix/resume.sh` 的 I/O 与关闭 Gate 一致性核验已通过。两组各读取 11,575,296 字节，reader 与 `/proc/self/io` 统计相等，ID/计数一致且无换页；证据见 `IO_ACCOUNTING_VERIFIED.json`。已复用冻结数据和训练模型进入正式 P0-A/B/C 评测，续跑日志为 `resume.log`、`evaluation.log`。最终完成仍以 `COMPLETE.json` 为准。

本目录运行完整 SIFT100M 实验：P0-A、P0-B 的 MLP/LR 两版，以及 P0-C 的 LR/XGBoost/状态 MLP/q–v MLP 消融。当前状态以 `STATUS.json` 为准；只有 `COMPLETE.json` 出现且 `complete=true` 才表示结果与报告全部完成。

数据为本地完整 100M×128 uint8 SIFT，独立索引 R64/Lbuild100/PQ34。查询划分：1200 训练（其中200内部留出选择轮次后重拟合）、1000验证、1000测试。原始向量和既有 SIFT1M/DEEP 实验不会修改。

已完成的数据预检见 `DATA_COMPLETE.json`、`EXACT_AUDIT_COMPLETE.json`；后者对8条预定查询扫描全部1亿条基础向量，核对 Top-100 的精确距离。原生 uint8 采集和关闭 Gate 后的搜索一致性见 `PREFLIGHT_COMPLETE.json`。

q–v 额外使用9,498,691,061字节的无损压缩驻内存向量文件，采用32向量独立块，逐次评分解码，解码计入推理和搜索时间。与1M完整 float 表的 CPU、内存实现不同。`COMPRESSION_COMPLETE.json` 记录了逐字节解压验证。所有节点 ID 都以 uint32/uint64 保存；采集数据的节点 ID 为同序 `.prune_nodes.u32` 文件，20-float 行的第二列为0占位。

`run_pipeline.py` 等待当前索引任务完成后，自动进行原生向量访问核验、验证集 L 范围预检及锁定、轨迹采集、模型训练、640配置验证、测试前选参锁定、P0-A/B/C 正式测试、重复计时、报告和 SVG 曲线生成。P0-A每方法9次；P0-B/C最终各指标前沿及锁定配置至少3次。未达到的 Recall 目标会如实报告。

查看进度：

```bash
cat experiments/sift100m_p0/STATUS.json
tail -c 2000 experiments/sift100m_p0/index_build.log
tail experiments/sift100m_p0/pipeline.log
```

后续重入命令（同一时间只允许运行一个流水线）：

```bash
bash experiments/sift100m_p0/run.sh
```

原始索引构建命令记录在 `index_build.command.json`。该重入命令负责索引完成后的阶段；不会重启正在运行的建图任务。任何正式搜索计时若发生进程换页将失败，额外读取也须与实际 reader 字节核对。

最终结果：`REPORT.md`、`p0_a/`、`p0_a_lr/`、`p0_b/`、`p0_b_lr/`、`p0_c/`。完整协议与日志均保留在本目录。

## 2026-10-22 索引恢复

17 个已完成分片的文件头大小和 ID 映射长度已通过恢复预检。原合并为每片分配 1 GiB 缓存，超过本机 15 GiB 内存；日志停止位置与内存不足吻合，但没有保存明确 OOM 证据。

`recover_index.py` 从原 DiskANN 源码提取同一合并算法，只将 I/O 缓存改为每流 16 MiB；继续调用原库的磁盘布局及预热样本生成，保留所有分片。恢复构建命令与源码摘要见 `index_recovery.command.json`，运行日志见 `index_recovery.log`。合并仍使用原实现的随机邻居截断，恢复后的索引将在采集前冻结。

索引恢复成功后校验磁盘索引头并生成 `INDEX_COMPLETE.json`，随后自动运行完整流水线。恢复入口（锁文件防止重复运行）：

```bash
OMP_NUM_THREADS=12 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/sift100m_p0/recover_index.py >> experiments/sift100m_p0/index_recovery.log 2>&1
```

2026-10-22 22:50：当前内核不提供 `/proc/self/io`，`getrusage(RUSAGE_SELF).ru_inblock*512` 在 O_DIRECT 搜索中亦返回 0。原生搜索增加有来源标记的备用统计；小图实测未通过物理字节一致性，故不能视为验证成功。参数预跑仅使用 Recall 选择 L，日志及锁定记录将物理 I/O 校验与正式计时有效性显式标为 false。采集和训练继续，流水线在训练完成后标记 `blocked_evaluation_io_accounting`；正式评测仍保留读取量一致性断言，不生成伪造的物理 I/O 结果。诊断见 `IO_ACCOUNTING_UNAVAILABLE.json`。
