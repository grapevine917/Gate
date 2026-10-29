# Task Tracker

## Local publication checks (2026-10-29)

- [x] Fresh DiskANN static-library build and XGBoost export support checked locally.
- [x] Three dataset-specific runners, native scores, calibration and smoke checked.
- [x] Publish server entry point, asset list and manual runbook (see publication commit).

## Pending target-server work

- [ ] T1: Verify the current DiskANN research branch builds on the target SSH server.
- [ ] T2: Review learned-gate integration and candidate-expansion logic.
- [ ] T3: Verify baseline search behavior without the gate on that server.
- [ ] T4: Evaluate gated search using Recall, latency, and I/O metrics on that server.
- [ ] T5: Record reproducible experiment commands, configurations, and results.
- [ ] T6: Establish a reliable ChatGPT-GitHub context retrieval workflow.

## Rules
- Mark a task complete only when its acceptance criteria have been checked.
- Record actual test commands and observed results.
- Do not treat planned experiments as completed experiments.
