# E4-3 运行产物

实验已执行，参数与数据定义见 `PROTOCOL.json`；本文件仅说明保存位置与复现方式。

- `selected_results.csv`：各绝对 Recall 下限下，由验证集选择的配置及评估集指标。
- `recall_ssd_curve.csv`、`figures/recall_ssd_curve.svg`：验证/评估完整 Recall–SSD 请求曲线。
- `all_run_summary.csv`：全部顺序、初始化和配置的统计。
- `order_initialization_robustness.csv`：验证选点在三种固定重排、暖历史与冷启动下的实际搜索结果。
- `graph/<运行名>/<配置>.csv`：逐 Query 指标、模型/二次评估计数、历史版本和历史更新计数。
- `graph/<运行名>/<配置>.ids`：最终 Top10，按逐 Query CSV 顺序保存的 uint64，形状为查询数×10。
- `*.observations.bin`：每次实际原始 MLP 输出及17维状态，顺序由 CSV 的 model_calls 划分；记录类型为 node:uint32, skip:uint32, fresh:float32, state:float32[17]，每条80字节。
- `*.decisions.bin`：二次评估完整输入、来源历史版本、最近历史状态来源、风险值及动作；格式见 `collect.py` 的 ddtype，每条296字节。
- `*.updates.bin`：Query完成后实际提交的原始 MLP Skip 状态；格式见 `collect.py` 的 udtype，每条88字节。
- `risk_models.json`：二次风险模型权重；原始 MLP 没有重新训练。
- `SELECTION_LOCK.json`：测试前冻结的验证选点。
- `verification.json`、`run_verification.csv`：逐流因果历史重放、风险特征/动作复算、最终 Recall 重算及确定性重复核验。
- `*.command.json`、`*.log`、`INPUT_LOCK.json`：完整命令、阶段日志与输入哈希。

主约束默认 R_min=99.44%，另保存99.60%、99.20%、99.00%的选择结果。验证可行不保证评估仍可行，是否达标单列记录。数据来自此前已观察的E4-1查询池，属于探索性实验；使用E4-2的500/250/250 Query划分。顺序为固定基准顺序及可复现重排，不能称为生产请求流。

搜索严格串行读取历史，只在当前 Query 完成后更新；二次 Skip 不成为历史 Skip。缓存命中沿用E1–E4保护策略。历史对有限流保留全部记录，无淘汰；内存字段是负载/容器下界估计，不是RSS。置信区间条件于已实现的历史轨迹，不能替代不同流顺序的真实重放。

复现：`.venv/bin/python experiments/e4_3_online_risk/run.py`。程序按已完成配置恢复；协议、模型或原生输入变化必须使用新目录。
