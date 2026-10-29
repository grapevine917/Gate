最终结果见 [加密网格报告](dense/REPORT.md)、[精确同Recall配对](dense/paired_comparisons.csv) 和 [各组实测配置](dense/exact_matched_results.csv)。

保留冻结原始17维MLP为第一阶段；CF仅处理第一阶段Keep的非缓存节点，使用风险概率，不使用收益门槛。原分数二次剪枝作为策略等价性对照。最终加密网格包含154个单阶段原分数阈值、154个二阶段原分数阈值和107个CF风险阈值。

同Recall比较按同一250条评估查询的整数GT命中总数精确匹配，不使用相同阈值或相同验证Recall下限替代。每个点采用本组扫描网格中最少SSD请求的配置；这属于固定评估集上的描述性曲线比较，不是独立部署阈值测试。

粗网格中间结果保留在 [REPORT.md](REPORT.md)。加密是在看过粗网格后进行的统一探索性精化，未重训或修改任何模型。最终结论使用dense目录，不混用两个网格。

复现：在项目根目录运行 `bash experiments/cf_cascade/run.sh`。已有完成输出在输入哈希锁相同时复用。
