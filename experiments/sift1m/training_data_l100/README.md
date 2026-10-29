# L=100 expansion 训练数据

一个训练样本对应一次候选节点展开：`x=(q,v,state)`，`y=0` 为经过 oracle 验证的 safe-to-skip，`y=1` 为保守 retain。完整数据共 **1,117,633** 个 expansion：642,134 个 y=0，475,499 个 y=1。对应 **71,528,512** 条完整邻居观测；其中 **41,096,576** 条属于 safe expansion，全部 parent_y=0，另单独导出。4109 万条负类邻居边不能单独用于训练双类别分类器，也不能当作 4109 万个独立 expansion 样本。

## 标签与分组

标签 y=0 当且仅当原始 expansion 没有新邻居入队，并且当前节点不属于原始最终 top10；标签与 `full_skip_100.mask` 严格一致。这组掩码已实际在读盘前整体跳过，10000 个 query 的有序 top10 ID 和距离完全不变。y=1 表示未被这条充分安全准则证明可以跳过，保守保留；不声称每一个 y=1 都是数学上绝对必要的展开。单条邻居边继承 `parent_y` 仅表示父 expansion 的标签，不是独立验证的边删除因果标签。

固定 seed=42，以 query ID 随机划分 train/validation/test=80%/10%/10%：8000/1000/1000 个 query。各文件按 query ID、expansion step 排序。每个 query 的全部 expansion 与邻居观测进入同一集合；query 集合两两不相交。每个 split 的样本数及类别统计在 `manifest.json`。

## 每个 split 的文件

- `metadata.npy`：sample_id、query_id、query_row、step_id、beam_step、node_id、label；另有仅供审计的 oracle_zero_admission/oracle_in_final_topk。
- `y.npy`：uint8 二分类标签，0=skip，1=retain。
- `pre_io_state.npy`：17 维读盘前状态。通过重放真实有界队列和原始 beam 选择顺序重建，每个 query 均与原 trace 逐次展开核对。
- `post_score_state.npy`：12 维已读取/评分当前节点、但尚未扫描邻居时的状态，来自该 expansion 的第一条边快照。
- `query_ids.npy`、`query_vectors.npy`：本集合 query ID 与 128 维 q 向量。
- `neighbors.npy`：该集合全部邻居观测，包含 v2 的全部 22 个字段，加 expansion_sample_id、expansion_row 和 parent_y。
- `safe_skip_neighbors.npy`：所需的 4109 万条观测在该集合中的部分，parent_y 全为 0。
- `query_*_offsets.npy`：每个 query 的 expansion、完整邻居和 safe 邻居连续区间；末尾元素为总行数。

v 向量通过 node_id 从原始 `sift_base.fbin` 以只读 memmap 解析，避免重复复制 100 万个向量。所有 NPY 文件支持 memory mapping；无需把几千万行一次性读入内存，也无需 pyarrow/pandas。

## 特征时刻与避免未来信息泄漏

默认使用 pre_io 状态：candidate PQ distance、队列 best/worst PQ、candidate queue rank、queue_size、unexpanded_count、visited_count、completed_expansions、beam_step、beam_position、cached_node、之前已评分节点的 best exact、has_best_exact、两个 PQ 比值及 L/beam 参数。整个 beam 的候选节点在任何新邻居处理之前选定；因此不能直接用处理中后续节点的第一条边快照冒充选择时刻状态。这里已重建并验证真实选择时刻。

第一轮尚无 exact incumbent 时以 0 + has_best_exact=0 编码，避免 NaN/Inf。默认状态不使用当前候选的 exact distance，更不使用当前邻居入队情况、邻居改善、最终 topk 等未来标签。post_score 状态仅用于读盘后邻居扫描 gate。邻居表和 metadata 的 oracle 字段用于监督/审计，不能整表直接作为 X 输入；`dataset.py` 的 batch() 仅返回显式状态矩阵、q/v 和 y。

`v` 的全精度向量供离线训练按需读取。如果部署时 v 仅在 SSD 上，读出 v 本身有 I/O 成本，不能声称是免费读盘前特征；纯读盘前 gate 可先使用 q、v_id、缓存/队列状态，或在 RAM 中预存可用 v 表示。提供两种 state view 不等于解决该部署成本。

仅用 train 拟合状态均值/标准差，保存在 `*_state_train_normalization.npz`，validation/test 复用 train 的统计。原始矩阵未归一化，loader 默认归一化。q/v 保留原始 SIFT float32 数值。

## 使用

```python
import sys
sys.path.insert(0, "/home/lxyue/projects/learned-gate/experiments/sift1m/training_data_l100")
from dataset import ExpansionDataset
train = ExpansionDataset(split="train", state_view="pre_io")
batch = train.batch([0, 1, 2])
# batch['q']: [3,128]; batch['v']: [3,128]; batch['state']: [3,17]; batch['y']: [3]
# include_vectors=False 可省去加载 v，仍返回 q、node_id 和 state。
validation = ExpansionDataset(split="validation")
test = ExpansionDataset(split="test")
for batch in train.iter_batches(batch_size=4096, shuffle=True, seed=42):
    pass  # 接入模型
# 查询级邻居上下文（包含未来标签，不是直接输入特征）：
context = train.neighbors(int(train.query_ids[0]), safe_only=True)
```

生成脚本：`../build_training_data.py`。完成与审计结果：`COMPLETE.json`、`manifest.json`。所有输出仅对应 L=100，未混入其他 L 的相同 query。
