# 旧模型严格召回匹配补充实验

本目录独立保存新实验，不修改原始 P0-A/P0-B 或权重。

- 范围：SIFT1M、DEEP1M、SIFT100M；原生、启发式、LR/XGB/State MLP、ET、ET+三个模型，共九族。
- 质量锚点：各数据集原生 L100，以及1M原生L160 / 100M原生L500的总命中数。
- **描述性严格匹配**：在已观察过的原测试查询上细化参数并匹配整数命中数，冻结后在同一查询上独立计时九轮；不是独立未见查询上的盲测，也不表示逐查询命中完全相同。另保留原实验验证选参结果，二者不能混为一谈。
- 搜索：先查原576配置网格；无有效剪枝的严格匹配时，对较大L的两个值做有上限的二分和邻域探测。Recall可能不单调、命中数离散，因此不保证九族都精确命中。完整预算见PROTOCOL.json。
- 优先选择实际剪枝/终止生效的配置。仅关闭策略而等同原生的精确点单独计数；联合组要求剪枝和ET都生效。
- 未命中目标的项打印命中差及严格匹配=False，不插值、不混用配置、不删返回结果。
- 复测：冻结所选配置，九轮循环顺序交错计时；保存完整延迟、QPS、实际物理字节及搜索工作量。继承原runner的数据、GT、IO和换页核验。

进度：`STATUS.json`、`pipeline.log`、各数据集 `progress_L*.json`；参数冻结：`SELECTION_LOCK.json`；失败：`FAILED.json`；最终完成：根 `COMPLETE.json`。`BACKGROUND_STATE.json`是启动记录，不能单独视为实时运行证明。

后台入口：`.venv/bin/python experiments/p0_old_models_recall_matched/launch.py`，有进程检查和文件锁，勿重复启动。最终报告 `REPORT.md`、汇总 `results.csv`。
