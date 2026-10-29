# DEEP1M P0系列实验

P0-A、P0-B、P0-C全部完成，包括P0-A/B的MLP与LR两版。使用DEEP原图R64/Lbuild100/PQ34，基础向量1M×96；K10、Beam4、10,000节点BFS缓存、4查询线程/MKL单线程。

原生搜索共测量640个验证配置、1518次P0-B/C测试运行，另有P0-A的54次运行。最终前沿、同Recall点、质量要求赢家与验证锁定配置均至少3次测试；P0-A每方法9次。

## P0-A：固定L100

两版本分别独立计时，每行时间为9次中位数。

|版本|方法|Recall|P95 ms|QPS|KiB/Query|
|---|---|---:|---:|---:|---:|
|MLP|Vanilla|99.47%|7.579|567.5|408.796|
|MLP|Heuristic|99.38%|6.625|639.6|364.924|
|MLP|Ours|99.34%|7.409|684.3|323.812|
|LR|Vanilla|99.47%|7.580|563.4|408.796|
|LR|Heuristic|99.38%|6.694|641.0|364.924|
|LR|Ours|99.37%|7.320|665.3|337.896|

## P0-B：Recall≥99.5%的最低读取量（描述性测试网格）

各行同一配置；实际Recall不同。测试经验网格选出的点不等于验证锁定的部署结果，后者见各分报告。

|方法|实际Recall|P95 ms|QPS|KiB/Query|L|
|---|---:|---:|---:|---:|---:|
|Tuned DiskANN|99.53%|7.959|542.6|431.980|106|
|Heuristic|99.60%|8.160|524.3|443.664|160|
|Ours (L100 MLP)|99.51%|8.210|645.4|337.500|300|
|ET|99.58%|9.685|701.3|324.424|160|
|ET + MLP|99.52%|10.086|675.1|319.332|160|
|Ours (LR)|99.62%|10.324|601.7|381.312|160|
|ET + LR|99.58%|10.090|674.0|324.424|160|

## P0-C：相同训练样本上的模型消融

四种Gate共享286550个mixed-L cache-miss样本、相同17维状态和代理retain标签；q–v额外使用96维查询/候选向量交互。下表为测试L100同一批cache-miss候选上的诊断。

|模型|Retain AUROC|Safe-skip AP|原生score μs/候选|
|---|---:|---:|---:|
|lr|0.91058|0.93630|0.0401|
|xgb|0.92018|0.94373|9.3656|
|state|0.92216|0.94504|0.6516|
|qv|0.90646|0.93436|3.0007|

P0-C的主要系统指标是Recall–P95前沿；更高分类分数不保证搜索收益。q–v模型后期明显过拟合，按内部独立查询选择早期最佳点，而非最终160轮权重。

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
训练/内部留出/验证/测试按查询划分；1200条训练查询中200条作轮次选择，随后按选中轮次重拟合全部1200条。没有复用SIFT模型权重，原数据/索引/模型文件未更改。LR、XGB、mixed-L MLP、q–v MLP共享样本；另训单独L100 MLP用于原版P0-A/B。
全部原生导出核验通过；Recall由ID复算，重复离散计数一致，实际reader字节与Linux read_bytes一致。验证锁定的配对查询区间见paired_validation_locked_comparisons.csv，不包含训练、选参和计时不确定性。
解释边界：DEEP基准查询曾在历史实验中观察；本次属于探索性实验。相比SIFT原P0，神经网络训练安排延长、阈值使用DEEP验证数据校准；跨数据集变化不能纯归因于数据集本身。联合组只有8对阈值，没有充分搜索所有组合，ET也未针对剪枝轨迹重训。
I/O为WSL2客户机块层流量，不是SSD内部NAND计数。固定L的收益、同质量下限的收益和严格相同Recall的收益分别报告。
复现：`bash experiments/deep_p0/run.sh`。协议见PROTOCOL.json、GRID_LOCK.json；查询划分见splits.json；验证见verification.json；模型见models/；逐查询结果见runs/。
