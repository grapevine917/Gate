# Model1: query/candidate pair only

输入仅为 x=[q;v]，v 为待展开候选节点。256→128→64→1，ReLU/ReLU/sigmoid，BCE + Adam，batch=2048，lr=0.001。
沿用 L=100、seed=42、8000/1000/1000 个互斥 query。只用训练集拟合标准化；验证集 logloss 早停选模型，验证集选阈值；测试集仅最终评估。

最佳 epoch=12，运行 20 epochs，参数量=41217。

| 模型 | Train AUC | Validation AUC | Test AUC | Test logloss |
|---|---:|---:|---:|---:|
| model1 q-v MLP | 0.755154 | 0.706941 | 0.705288 | 0.594161 |
| logistic_regression | 0.928011 | 0.927090 | 0.928034 | 0.325636 |
| xgboost | 0.932129 | 0.929551 | 0.930310 | 0.313126 |

| 模型 | 验证目标 FNR | Test skip rate | Test retain FNR | 误跳过数 | 受影响 query |
|---|---|---:|---:|---:|---:|
| model1 q-v MLP | default_0.5 | 66.793% | 50.6916% | 24150 | 999/1000 |
| model1 q-v MLP | validation_zero_false_skip | 0.023% | 0.0000% | 0 | 0/1000 |
| model1 q-v MLP | validation_fnr_0.1pct | 0.469% | 0.1091% | 52 | 34/1000 |
| model1 q-v MLP | validation_fnr_1pct | 2.608% | 0.9194% | 438 | 198/1000 |
| logistic_regression | default_0.5 | 60.649% | 20.4236% | 9730 | 989/1000 |
| logistic_regression | validation_zero_false_skip | 0.391% | 0.0021% | 1 | 1/1000 |
| logistic_regression | validation_fnr_0.1pct | 4.810% | 0.1175% | 56 | 43/1000 |
| logistic_regression | validation_fnr_1pct | 16.346% | 1.0117% | 482 | 320/1000 |
| xgboost | default_0.5 | 62.925% | 22.6926% | 10811 | 1000/1000 |
| xgboost | validation_zero_false_skip | 0.649% | 0.0063% | 3 | 3/1000 |
| xgboost | validation_fnr_0.1pct | 5.613% | 0.1217% | 58 | 48/1000 |
| xgboost | validation_fnr_1pct | 16.667% | 0.9677% | 461 | 308/1000 |

结论：q-v pair 有预测信号（test AUC=0.7053），但本次指定 MLP 未达到 state-only 基线的低风险跳过能力。验证目标 FNR=0.1%/1% 时，测试跳过比例仅 0.469%/2.608%。

这里的 retain FNR 是保守保留标签误跳过率，不是 Recall@10 损失。验证零错误不保证测试零错误。
本实验检验指定 MLP 是否能从 q-v 学到可用信号；单一结构的失败不能证明 q-v 在信息论意义上不充分，也可能受模型容量、优化或样本量限制。
skip rate 是 baseline 轨迹上的离线比例，需在线搜索验证召回、I/O 和延迟；全精度 v 的读取成本尚未计入。

重现：在 learned-gate 目录执行 `.venv/bin/python experiments/sift1m/train_model1.py`。推理脚本：`experiments/sift1m/predict_model1.py`。
