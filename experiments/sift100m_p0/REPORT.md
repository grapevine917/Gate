# SIFT100M P0系列实验

P0-A、P0-B、P0-C全部完成，包括P0-A/B的MLP与LR两版。使用SIFT100M原图R64/Lbuild100/PQ34，基础向量100M×128；K10、Beam4、10,000节点BFS缓存、4查询线程/MKL单线程。

原生搜索共测量640个验证配置、1490次P0-B/C测试运行，另有P0-A的54次运行。最终前沿、同Recall点、质量要求赢家与验证锁定配置均至少3次测试；P0-A每方法9次。

## P0-A：固定L100

两版本分别独立计时，每行时间为9次中位数。

|版本|方法|Recall|P95 ms|QPS|KiB/Query|
|---|---|---:|---:|---:|---:|
|MLP|Vanilla|96.95%|8.711|499.0|442.976|
|MLP|Heuristic|96.08%|7.500|573.0|384.604|
|MLP|Ours|96.77%|8.843|532.3|401.504|
|LR|Vanilla|96.95%|8.633|501.4|442.976|
|LR|Heuristic|96.08%|7.554|574.9|384.604|
|LR|Ours|96.68%|8.523|543.9|407.796|

## P0-B：Recall≥99.5%的最低读取量（描述性测试网格）

各行同一配置；实际Recall不同。测试经验网格选出的点不等于验证锁定的部署结果，后者见各分报告。

|方法|实际Recall|P95 ms|QPS|KiB/Query|L|
|---|---:|---:|---:|---:|---:|
|Tuned DiskANN|99.52%|23.375|181.1|1233.748|300|
|Heuristic|99.51%|22.726|183.1|1213.884|300|
|Ours (L100 MLP)|99.52%|23.452|176.9|1233.748|300|
|ET|99.58%|33.389|182.1|1208.240|500|
|ET + MLP|99.52%|23.679|175.5|1233.748|300|
|Ours (LR)|99.50%|28.409|202.0|1064.256|500|
|ET + LR|99.58%|35.311|172.9|1207.288|500|

## P0-C：相同训练样本上的模型消融

四种Gate共享304031个mixed-L cache-miss样本、相同17维状态和代理retain标签；q–v额外使用128维查询/候选向量交互。下表为测试L100同一批cache-miss候选上的诊断。

|模型|Retain AUROC|Safe-skip AP|原生score μs/候选|
|---|---:|---:|---:|
|lr|0.88223|0.83208|0.0414|
|xgb|0.89021|0.84531|8.6132|
|state|0.89260|0.85003|0.6773|
|qv|0.87866|0.83088|13.0090|

P0-C的主要系统指标是Recall–P95前沿；更高分类分数不保证搜索收益。q–v模型按内部独立查询的最小损失选择轮次，再使用全部训练查询重拟合。

## 结果、曲线与复现

- [P0-A MLP报告](p0_a/REPORT.md)
- [P0-A LR报告](p0_a_lr/REPORT.md)
- [P0-B MLP报告](p0_b/REPORT.md)
- [P0-B LR报告](p0_b_lr/REPORT.md)
- [P0-C消融报告](p0_c/REPORT.md)
- [P0-B MLP Recall–P95](p0_b/figures/recall_p95.svg)
- [P0-B LR Recall–P95](p0_b_lr/figures/recall_p95.svg)
- [P0-C Recall–P95](p0_c/figures/recall_p95.svg)

所有分报告均包含实际Recall、对应配置、精确同Recall比较及验证锁定配置的测试达标情况。P0-B/C的三项核心曲线沿用仓库现有P0绘图工具的离线SVG后端，点为实测配置，线仅引导阅读。
训练/内部留出/验证/测试按查询划分；1200条训练查询中200条作轮次选择，随后按选中轮次重拟合全部1200条。没有复用SIFT1M或DEEP模型权重，原数据/索引/模型文件未更改。LR、XGB、mixed-L MLP、q–v MLP共享样本；另训单独L100 MLP用于原版P0-A/B。
全部原生导出核验通过；Recall由ID复算，重复离散计数一致，实际reader字节与Linux read_bytes一致。验证锁定的配对查询区间见paired_validation_locked_comparisons.csv，不包含训练、选参和计时不确定性。
解释边界：本次为预先冻结查询划分上的有限网格实验。相比SIFT原P0，神经网络训练安排延长、阈值使用SIFT100M验证数据校准；跨数据集变化不能纯归因于数据集本身。联合组只有8对阈值，没有充分搜索所有组合，ET也未针对剪枝轨迹重训。
I/O为WSL2客户机块层流量，不是SSD内部NAND计数。固定L的收益、同质量下限的收益和严格相同Recall的收益分别报告。
复现：`bash experiments/sift100m_p0/run.sh`。协议见PROTOCOL.json、GRID_LOCK.json；查询划分见splits.json；验证见verification.json；模型见models/；逐查询结果见runs/。

100M资源差异：采用uint8原始向量索引；q–v使用额外驻内存的无损压缩原始向量表，解码计入推理和搜索。PQ34本体约3.4GB；向量表实际大小见COMPRESSION_COMPLETE.json。节点ID始终以uint32/uint64保存，未经过float32转换。

不足Top-10的结果按底层约定以尾部UINT32_MAX占位；汇总仅允许该占位值，检查其余ID合法且唯一，缺少位置计作未命中，Recall仍使用固定Top-10分母。逐配置缺少位置和查询数见all_configurations.csv的missing_result_slots / short_result_queries；验证审计见RESULT_PADDING_AUDIT.json。

驻留加载修正：候选诊断在独立进程退出后开始在线搜索。q–v在计时前按16 MiB块加载全部压缩向量，并释放已复制的冗余文件页缓存；搜索时仍从进程内存解码。所有正式计时统一使用修正后的搜索二进制；旧计时归档保留，不混入本报告。数据、索引、训练权重、阈值及配置网格不变。输入锁修订及连续失败批次复测见[加载修订记录](RESIDENT_LOADER_AMENDMENT.json)和[驻留验证](resident_probe/VERIFIED.json)。
