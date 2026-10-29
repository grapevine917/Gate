# Project Context

## Project Goal
研究 DiskANN 中基于学习模型的节点展开剪枝（learned gate），
探索利用候选节点的动态搜索状态预测其展开价值，并评估对搜索性能的影响。

## Repositories
- Research repository: https://github.com/grapevine917/Gate
- DiskANN fork: https://github.com/grapevine917/DiskANN
- Current DiskANN research branch: research/learned-gate

## Development Environment
- OS: WSL Ubuntu
- Main tool: Codex CLI
- Build and experiment environment: local Ubuntu filesystem

## Current Git Checkpoints
- Gate master: 56b8c3d
- DiskANN research branch: c9e6122
- Earlier DiskANN commit: 46aacd0

## Important Constraints
- Do not upload datasets, indexes, large traces, model binaries, or generated run artifacts by default.
- Preserve existing local experiment data and build outputs.
- Distinguish implemented code from code that has been compiled or experimentally validated.
- Do not claim performance improvements without verifiable experimental evidence.

## Current Validation Status
The research implementation has been committed and pushed to GitHub.
Compilation correctness, algorithm correctness, Recall, latency, and I/O performance
must be tracked separately and must not be assumed validated from Git commits alone.
