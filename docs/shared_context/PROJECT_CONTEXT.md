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
- Gate master: use `git rev-parse HEAD`; the server-preparation publication follows 7254d3b.
- DiskANN research branch: 73583a23cd6fab84b63beb2d95ee72ef2cc43eb1
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

Historical three-dataset P0-A and expanded P0-B experiments completed in WSL.
The 2026-10-29 publication adds an isolated server entry point and verifies a
fresh local native build, frozen scores, calibration and 42 smoke configurations.
It does not mean an SSH server has been tested. See
`docs/deployment/PUBLICATION_VALIDATION.md` and `SERVER_RUNBOOK.md`.
