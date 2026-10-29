# LR / MLP 的统一搜索状态特征分析补充

本任务在扩大P0-B根COMPLETE=true且run.lock释放后自动开始。等待时只读取小型状态文件，不加载模型、数据或执行分析，避免干扰验证选参和正式性能计时。

## 范围

先分析SIFT1M。使用 `experiments/main_comparison/models/` 中当前主实验使用的旧混合L模型：LR、17维State MLP；在相同状态上重新分析对应XGBoost，作为可比较参照。不重训，不覆盖旧权重，不启动任何图搜索。

评估输入是既有SIFT1M基线L100的1000条验证查询状态，筛选未缓存节点。查询ID与主实验validation_ids逐项核对。模型训练是1200查询的混合L，分析切片是部署L100，两者分别标注。原来的纯L100/8000训练查询XGB重要性结果保留，不直接与本次数值合并。

## 方法

- 三模型使用相同样本、标签、特征和逐轮相同的随机排列。
- 对17个单特征各做10轮permutation，并对Global11、Candidate6、RankProgress、Distance和All17做组内联合排列；组内部的列保持同一个行排列。
- 报告AUC下降、logloss增加，以及既有冻结验证阈值（retain-FNR预算0.1%/1%）下的误跳过率和跳过率变化。排列后不重新选阈值。
- 单独给出LR的标准化系数；这不是与MLP/XGB直接比较的统一重要性度量。
- 在每次排列的变化按query汇总后，按query做1000次bootstrap；排列SD与query置信区间明确区分。没有训练种子不确定性估计。
- 用原始2000条native_check预期概率核验二进制权重解析及三模型预测，再进行全量分析。

这是固定模型的依赖分析，不是删除特征后重训消融，也不提供最终Recall或在线IO/延迟收益。相关特征的排列可能产生不自然状态，组排列也不消除所有分布偏移。L/Beam/cache在此切片是常量，其零重要性仅对该切片成立。后续如果据排名选择特征子集，需要训练和真实在线验证。

## 运行与结果

后台入口：`.venv/bin/python experiments/model_feature_analysis/launch.py`；有进程检查和运行锁，不重复启动。

进度：`STATUS.json`，等待阶段为 `waiting_for_expanded_P0B`；日志 `pipeline.log`。异常看 `FAILED.json`，完成以根 `COMPLETE.json` 为准。启动记录BACKGROUND_STATE不代表实时心跳。

产物：`REPORT.md`、`importance_summary.csv`、`permutation_runs.csv`、`lr_coefficients.csv`、`ranking_agreement.csv`、PNG/SVG/PDF图、`SOURCE_LOCK.json`和`baseline.json`。完整协议在实际分析启动后写入 `PROTOCOL.json`。
