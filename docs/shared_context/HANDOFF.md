# Latest Handoff

Updated: 2026-10-29.

## Current State

- User authorized publishing missing experiment code for manual SSH-server work.
- DiskANN research/learned-gate now includes XGGATE01 model support, commit 73583a23cd6fab84b63beb2d95ee72ef2cc43eb1.
- Gate publication includes experiment source/protocols/reports, a fresh-workspace server runner, asset manifest and manual runbook. Large assets remain local.
- Historical WSL experiments are completed; new server experiments have not run.

## Validation

- Fresh local CMake static-library build passed.
- Three frozen dataset-specific runners and score checks passed.
- New calibration/configuration preparation and 42 smoke configurations passed.
- 6 workspace/input safeguard tests passed.
- Details and limits: docs/deployment/PUBLICATION_VALIDATION.md.

## Next Actions

1. User connects to the SSH server, clones both repositories as in SERVER_RUNBOOK.
2. Install native/Python dependencies and transfer the 59 listed assets (~42.81GiB).
3. Create a fresh workspace and build/prepare/smoke on that server.
4. Run server P0-A/expanded P0-B only after checks pass; do not import WSL timings.

## Reporting Requirements

Record commit/branch, changed files, actual commands and results, hardware/storage, input hashes and unresolved limits. Do not treat code publication as server validation.
