主报告：[REPORT.md](REPORT.md)；三图总览：[index.html](index.html)。

八种方法，每种64配置，联合调L与阈值。混合L模型在models目录；旧模型和DiskANN源码不修改。

Gate统一使用保守贡献代理监督，ET使用当前已评分前缀的真实Recall监督；不是上一轮CF风险模型的直接复用。DARTH/LAET列为相关研究，主图只含同一DiskANN环境内的实测结果。

主要数据：frontier_points.csv、strict_equal_recall_ssd.csv、strict_equal_recall_ssd_CI.csv、all_configurations.csv、validation_locked_test.csv。

复现：在项目根目录运行 `bash experiments/main_comparison/run.sh`。模型/网格哈希锁不一致时需要归档旧结果后重新运行。
