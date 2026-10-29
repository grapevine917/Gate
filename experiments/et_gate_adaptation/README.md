# Gate / ET trajectory adaptation

Run from the repository root:

```bash
bash experiments/et_gate_adaptation/run.sh
```

The experiment resumes completed search configurations and completed training.
Keep the protocol, inputs and trained models fixed while resuming. Archive the
directory before a fresh timing run. Existing P0-A/P0-B results are preserved.

The frozen LR Gate is the primary pruning policy. The ET architecture remains
17–128–64–1, with sigmoid output and MSE on actual prefix Recall@10. Compare:

| Family | Node pruning | Termination model |
|---|---|---|
| DiskANN | None | None |
| Gate | Frozen LR | None |
| ET_original | None | Historical ET |
| ET_unpruned | None | ET refit on this run's unpruned trajectories |
| Joint_original | Frozen LR | Historical ET |
| Joint_unpruned | Frozen LR | ET refit on this run's unpruned trajectories |
| Joint_adapted | Frozen LR | ET trained on this run's pooled pruning trajectories |

Train on 1,200 queries, use the existing 200-query internal holdout to select
epochs, then refit on all training queries. Validation and test each contain
1,000 disjoint queries. The historical test benchmark has already been observed,
so these are exploratory results rather than a new confirmatory evaluation.

Collect five L values and four Gate thresholds, sampling at most 32 boundary
states per query and configuration. The adapted model receives the zero-threshold
trajectories as well as three active-pruning settings. The unpruned refit uses
only zero-threshold trajectories, hence has fewer distinct training states;
this controls the training procedure but does not equalize the number of rows.

Each joint family evaluates the full 5×4×8 Cartesian grid. This explicitly gives
joint policies more tuning configurations than single policies. A 64-L DiskANN
baseline is retained. Select independently for P95, QPS and measured read bytes at
each validation quality target; lock before testing. Test locked policies and
fixed-parameter disable-Gate/disable-ET controls, three repeats each.

All ET features are live search state. Ground truth, final recall and final
read counts are offline labels/diagnostics only. This experiment adapts the
existing absolute-Recall ET model; it does not introduce a separate learned
incremental-benefit stopping rule.

`REPORT.md`, `validation_locked_test.csv`, `test_summary.csv`, optional figures and paired
query bootstrap comparisons are written after completion. `COMPLETE.json`
records completion of the core experiment; `verification.json` records the
additional analysis checks. Timing medians and ranges describe repeated runs;
query bootstrap intervals do not establish timing significance.
Standard figures are generated when Matplotlib is available; CSV analysis and
verification complete without it.
