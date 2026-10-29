# M2 interaction MLP

输入为原始向量构造的 [q;v;|q-v|;q⊙v]，512→128→64→1。ReLU/ReLU/sigmoid，BCE + Adam，batch=2048，lr=0.001。
沿用 L=100、seed=42、8000/1000/1000 个互斥 query。只用训练集拟合标准化；验证集 logloss 早停选模型，验证集选阈值；测试集仅最终评估。

最佳 epoch=3，运行 11 epochs，参数量=73985。

| 模型 | Train AUC | Validation AUC | Test AUC | Test logloss |
|---|---:|---:|---:|---:|
| M2 interaction MLP | 0.795790 | 0.755460 | 0.752680 | 0.557306 |
| logistic_regression | 0.928011 | 0.927090 | 0.928034 | 0.325636 |
| xgboost | 0.932129 | 0.929551 | 0.930310 | 0.313126 |
| q-v MLP | 0.755154 | 0.706941 | 0.705288 | 0.594161 |
| state-only MLP | 0.931270 | 0.929768 | 0.930416 | 0.313044 |

| 模型 | 验证目标 FNR | Test skip rate | Test retain FNR | 误跳过数 | 受影响 query |
|---|---|---:|---:|---:|---:|
| M2 interaction MLP | default_0.5 | 68.571% | 47.6606% | 22706 | 1000/1000 |
| M2 interaction MLP | validation_zero_false_skip | 0.031% | 0.0021% | 1 | 1/1000 |
| M2 interaction MLP | validation_fnr_0.1pct | 0.528% | 0.0861% | 41 | 25/1000 |
| M2 interaction MLP | validation_fnr_1pct | 3.342% | 0.9005% | 429 | 148/1000 |
| logistic_regression | default_0.5 | 60.649% | 20.4236% | 9730 | 989/1000 |
| logistic_regression | validation_zero_false_skip | 0.391% | 0.0021% | 1 | 1/1000 |
| logistic_regression | validation_fnr_0.1pct | 4.810% | 0.1175% | 56 | 43/1000 |
| logistic_regression | validation_fnr_1pct | 16.346% | 1.0117% | 482 | 320/1000 |
| xgboost | default_0.5 | 62.925% | 22.6926% | 10811 | 1000/1000 |
| xgboost | validation_zero_false_skip | 0.649% | 0.0063% | 3 | 3/1000 |
| xgboost | validation_fnr_0.1pct | 5.613% | 0.1217% | 58 | 48/1000 |
| xgboost | validation_fnr_1pct | 16.667% | 0.9677% | 461 | 308/1000 |
| q-v MLP | default_0.5 | 66.793% | 50.6916% | 24150 | 999/1000 |
| q-v MLP | validation_zero_false_skip | 0.023% | 0.0000% | 0 | 0/1000 |
| q-v MLP | validation_fnr_0.1pct | 0.469% | 0.1091% | 52 | 34/1000 |
| q-v MLP | validation_fnr_1pct | 2.608% | 0.9194% | 438 | 198/1000 |
| state-only MLP | default_0.5 | 62.868% | 22.6255% | 10779 | 1000/1000 |
| state-only MLP | validation_zero_false_skip | 0.733% | 0.0042% | 2 | 2/1000 |
| state-only MLP | validation_fnr_0.1pct | 5.372% | 0.1112% | 53 | 42/1000 |
| state-only MLP | validation_fnr_1pct | 17.007% | 1.0159% | 484 | 317/1000 |

interaction MLP：test AUC=0.7527。验证目标 FNR=0.1%/1% 时，测试跳过比例为 0.528%/3.342%。
这里的 retain FNR 是保守保留标签误跳过率，不是 Recall@10 损失。验证零错误不保证测试零错误。
各 MLP 的隐藏层、优化器、batch、seed、早停和数据划分一致，输入维度/特征和对应训练标准化统计不同。state MLP 与 LR/XGBoost 使用相同原始状态特征。
M2 的交互特征是 q/v 的确定性函数，没有添加新的外部信息；但输入扩展也增加首层参数，M1/M2 不是参数量严格相等的对照。
单一结构的失败不能证明 q-v 在信息论意义上不充分，也可能受模型容量、优化或样本量限制。
skip rate 是 baseline 轨迹上的离线比例，需在线搜索验证召回、I/O 和延迟。

重现：在 learned-gate 目录执行 `.venv/bin/python experiments/sift1m/train_model2.py`.

## M1 vs M2 interpretation

M2-M1 AUC 差为 0.047391；按 query 配对 bootstrap 的 95% 区间为 [0.04292491402374997, 0.051905237549958294]。
M2 弥补 M1 到 state MLP 的 AUC 差距约 21.1%（描述性比例）。
交互项不引入新信息；提升意味着这种表示使关系更容易被当前 MLP 学到。剩余差距不能单凭本次实验归因于信息不足。
M1 参数量 41217，M2 参数量 73985；输入扩展与首层容量变化共同构成此次对照。单个训练 seed 的结果尚不覆盖训练随机性。
bootstrap 区间只描述固定模型在新 query 上的采样不确定性。
独立验证：全部 111841 原始测试 q/v 对复现导出概率，特征顺序及公式核对通过，三个 MLP 的数据源与训练配置一致。
重现分析：`.venv/bin/python experiments/sift1m/analyze_model2.py`。
