# Gate: Search-State-Aware Learned Pruning for SSD Graph ANN Search

Gate is a research codebase for learned pruning in SSD-backed graph approximate nearest-neighbor (ANN) search. It uses search-state features to decide whether a candidate node should be read from storage and expanded. The project studies recall, tail latency, throughput, and physical I/O, including comparisons with heuristic pruning and early termination.

本文研究 SSD 图近邻搜索中的学习型剪枝方法 Gate：根据搜索过程中的状态特征，判断候选节点是否值得继续读取和扩展。实验关注 Recall@10、尾延迟、吞吐量和实际物理读取量，并与启发式剪枝和提前终止进行比较。

## Repository contents

- `DiskANN/` contains the DiskANN source snapshot used by this project, including local changes and the upstream MIT license.
- `experiments/` contains experiment drivers, analysis and report sources, protocols, and selected compact results.
- Large datasets, graph indexes, trained model binaries, per-query traces, virtual environments, and temporary run outputs are excluded. Obtain datasets from their original providers and follow their terms. Follow each model's and dependency's license when redistributing it.
- `experiments/p0_old_models_expanded/` contains the expanded tuning experiment. Its scripts and protocol are included; generated run state is excluded.

## Reproducibility

Experiments require the matching dataset, graph index, model weights, and native dependencies described by each experiment's protocol. Install dependencies in a separate environment, prepare the required artifacts, then use the experiment's documented build and run scripts. Do not run every historical experiment directory as one pipeline; each has its own protocol and prerequisites.

The completed old-model comparison is documented in `experiments/p0_old_models_rerun/REPORT.md`. The strict-recall descriptive comparison is documented in `experiments/p0_old_models_recall_matched/REPORT.md`. These reports state the measured configurations and limitations. The expanded tuning experiment is still running and does not yet have final results.

## DiskANN source and license

`DiskANN/` is based on Microsoft's [DiskANN](https://github.com/microsoft/DiskANN) and includes project-specific modifications. The upstream MIT license is retained in `DiskANN/LICENSE`. This repository does not relicense third-party components; consult the license files and notices for each dependency.

## Status

This is an active research project. Results are specific to the datasets, indexes, model weights, hardware, and protocols recorded with each experiment. Curves and tables should be interpreted together with their reported recall and measurement conditions.
