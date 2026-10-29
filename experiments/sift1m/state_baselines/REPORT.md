# State-only baseline: Logistic Regression / XGBoost

使用 L=100 expansion 数据；train/validation/test 分别为 8000/1000/1000 个互斥 query，894077/111715/111841 个样本。仅使用 17 维 pre_io_state，不读取 q/v 向量，不使用 query/node ID、邻居观测或事后标签作为输入。L 和 beam_width 是常量特征。

y=0 为已验证 oracle safe-to-skip，y=1 为保守 retain 标签。模型输出 p_retain；p_retain < tau 才跳过。LR 使用 train-only 标准化、C=1、L-BFGS；XGBoost 使用 CPU hist、depth=6、最多500棵树、learning_rate=.05，validation logloss early stopping=30。没有超参数搜索。

## 测试集：默认阈值 0.5

| 模型 | ROC-AUC | AP(safe) | Accuracy | Skip rate | Retain 误跳过率 | Skip precision |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.928034 | 0.935392 | 85.846% | 60.649% | 20.424% | 85.655% |
| xgboost | 0.930310 | 0.937498 | 86.189% | 62.925% | 22.693% | 84.638% |

## 验证集选阈值，固定后评估测试集

| 模型 | 验证集 retain FNR 上限 | tau | Test skip rate | Test skipped/query | Test retain 误跳过数 | Test retain FNR | Test skip precision | 有误跳过的 test query 数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.0% | 0.0002051228 | 0.391% | 0.437 | 1 | 0.0021% | 99.771% | 1/1000 |
| logistic_regression | 0.1% | 0.0041965306 | 4.810% | 5.379 | 56 | 0.1175% | 98.959% | 43/1000 |
| logistic_regression | 1.0% | 0.030894061 | 16.346% | 18.282 | 482 | 1.0117% | 97.364% | 320/1000 |
| xgboost | 0.0% | 0.002795211 | 0.649% | 0.726 | 3 | 0.0063% | 99.587% | 3/1000 |
| xgboost | 0.1% | 0.017148944 | 5.613% | 6.278 | 58 | 0.1217% | 99.076% | 48/1000 |
| xgboost | 1.0% | 0.053024102 | 16.667% | 18.641 | 461 | 0.9677% | 97.527% | 308/1000 |

Retain FNR = 错误跳过的 y=1 数 / 全部 y=1 数。Skip precision = 正确跳过的 y=0 数 / 所有预测跳过数。两者分母不同。验证集上的风险上限不是测试集安全保证；尤其 validation zero-error 不能证明新 query 零风险。

这些结果是原始 baseline 状态上的离线分类结果，尚未把模型放入 DiskANN 搜索环，不能将 skip rate 当作真实在线节省率，也不能推断 Recall@10 不变。y=1 是保守标签，误跳过它表示违背 oracle 保留规则，不一定等于召回损失；真正影响需下一步在线 gate 重跑验证。

## 特征与 artifact

LR 的 standardized coefficients、XGBoost 的 feature importance 分别保存在模型子目录 CSV。每个模型有 train/validation/test p_retain、阈值、完整 metrics、每-query 统计与 threshold curve。保存的模型已加载回验，与测试概率逐项一致。requirements.lock.txt 固定本次依赖；provenance.json 记录 split、标签口径和版本。

运行：在 learned-gate 中执行 `.venv/bin/python experiments/sift1m/train_state_baselines.py`。仅重新运行推理时使用 `predict_state.py`，输入必须按 provenance 中的17维顺序；LR bundle 含标准化参数，XGBoost 直接用 raw state。

实现文档：[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)、[XGBoost sklearn interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)。本报告的全部数值来自本地训练与测试。
