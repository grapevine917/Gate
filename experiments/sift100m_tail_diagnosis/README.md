# SIFT100M tail diagnosis

This is an isolated diagnostic experiment. It does not change the completed
`sift100m_p0` experiment, its model, index, thresholds, runner, or formal results.

The six policies are frozen validation selections minimizing physical reads for
Recall targets 99.5% and 99.7%, for DiskANN, Heuristic, and LR. DiskANN L500 is an
additional same-L control. No policy is selected using this diagnostic test run.
Each configuration runs the same 1,000 test queries with four threads, three
repeats, and the original warm-up protocol.

Within each process the profiler first records per-query binary pruning
decisions. Full and replay modes then alternate their order across repeats.
Replay consumes the exact recorded decisions instead of queue-state extraction,
feature construction, and model scoring. The full mode does not collect
per-candidate timings or record tapes. All modes must produce byte-identical IDs
and identical discrete counts to the original experiment, query by query.

Profiling includes queue rank scanning, feature construction, scoring, original
DiskANN synchronous reader elapsed time, I/O rounds, outer search rounds, empty
I/O rounds, comparisons, and candidate selections. Synchronous reader time
includes submission/completion overhead; it is not pure device service time.
The existing `cpu_us` only times portions of computation and is not an additive
CPU total. `latency - io_us` is the full non-reader residual.

Per-candidate clocks and trace recording perturb the profile mode, particularly
the very short LR scorer. Use full-versus-replay for the end-to-end ablation,
and profile components for descriptive attribution. Replay itself has tape
lookup overhead, and removes the entire gate decision path rather than only the
dot product. It does not represent a deployable no-cost classifier.

Own-tail cohorts are exactly 50 query IDs selected by per-query median full-mode
latency over three repeats. Paired cohorts compare every method on those same
IDs, avoiding comparison of unrelated difficult queries. Global P95 is computed
per repeat and then median-aggregated, matching the formal protocol.

The selected/skip node trace is collected only in profile mode, repeat zero.
Comparing it to DiskANN L500 can establish alternate routes and lost hits.
It cannot establish that a particular skipped node was a necessary bridge:
that would require a separate node intervention.

Reproduce:

```sh
python3 experiments/sift100m_tail_diagnosis/build.py
.venv/bin/python experiments/sift100m_tail_diagnosis/run.py
.venv/bin/python experiments/sift100m_tail_diagnosis/analyze.py
```
