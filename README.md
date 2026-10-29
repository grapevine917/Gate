# Gate: Search-State-Aware Learned Pruning for SSD Graph ANN Search

Gate is a research codebase for learned pruning in SSD-backed graph approximate nearest-neighbor (ANN) search. It uses search-state features to decide whether a candidate node should be read from storage and expanded. The project studies recall, tail latency, throughput, and physical I/O, including comparisons with heuristic pruning and early termination.

本文研究 SSD 图近邻搜索中的学习型剪枝方法 Gate：根据搜索过程中的状态特征，判断候选节点是否值得继续读取和扩展。实验关注 Recall@10、尾延迟、吞吐量和实际物理读取量，并与启发式剪枝和提前终止进行比较。

## Repository contents

- DiskANN source is maintained separately in [grapevine917/DiskANN](https://github.com/grapevine917/DiskANN), branch `research/learned-gate`. Clone it into `DiskANN/`; it is not embedded in this repository.
- `experiments/` contains experiment drivers, analysis and report sources, protocols, and selected compact results.
- Large datasets, graph indexes, trained model binaries, per-query traces, virtual environments, and temporary run outputs are excluded. Obtain datasets from their original providers and follow their terms. Follow each model's and dependency's license when redistributing it.
- `experiments/p0_old_models_expanded/` contains the completed expanded tuning experiment. Generated run state is excluded.
- `experiments/server/` provides an isolated server-rerun entry point, an asset manifest, and a native frozen-model verifier. Start with the [server runbook](docs/deployment/SERVER_RUNBOOK.md).

## Reproducibility

Experiments require the matching dataset, graph index, model weights, and native dependencies described by each experiment's protocol. Large artifacts are transferred separately, not downloaded automatically. The server entry point checks the asset list, creates a new workspace, rebuilds native runners, verifies frozen model outputs, and reruns validation before expanded tuning. Do not copy historical result caches into a new server run or run every historical experiment directory as one pipeline.

The completed old-model comparison is documented in `experiments/p0_old_models_rerun/REPORT.md`. The strict-recall descriptive comparison is documented in `experiments/p0_old_models_recall_matched/REPORT.md`. The expanded tuning experiment completed on 2026-10-28; its report is in `experiments/p0_old_models_expanded/REPORT.md`. These are historical measurements in the recorded WSL environment, not evidence that a new server build has been validated. Validation-selected results and descriptive test-grid frontiers are reported separately.

## DiskANN source and license

The separate DiskANN fork is based on Microsoft's [DiskANN](https://github.com/microsoft/DiskANN) and retains the upstream MIT license. Frozen experiment overlays copied from DiskANN are accompanied by `experiments/DISKANN_LICENSE`. This repository does not relicense third-party components; consult their license files and notices.

## Status

This is an active research project. Results are specific to the datasets, indexes, model weights, hardware, and protocols recorded with each experiment. Curves and tables should be interpreted together with their reported recall and measurement conditions.
