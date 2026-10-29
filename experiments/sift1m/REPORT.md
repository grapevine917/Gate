# SIFT1M DiskANN neighbor expansion experiment

DiskANN commit: `78256bbab4685e1774e78d331e081a153be26823`; R64, Lbuild100, PQ 34 bytes, beam4, cache10000, threads4. Base: 1M x 128; queries: 10000. Ground truth: supplied SIFT top100, additionally checked exact top10 boundary for 32 queries.

`visited`: distinct medoid + neighbor IDs encountered (including queue rejections). `expanded`: nodes whose adjacency is processed; this differs from visited and from number of beam iterations. `useful`: at least one newly visited neighbor admitted to PQ candidate queue. `useless`: zero admissions. Admission is not equivalent to improving final Recall@10.

| L | Recall@10 | visited/query | expanded/query | useful/query | useless/query | useless % | zero-new % | off-path % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 91.027% | 1693.68 | 34.74 | 22.08 | 12.66 | 36.45% | 0.02% | 67.02% |
| 50 | 98.395% | 2618.94 | 62.79 | 32.37 | 30.42 | 48.45% | 0.07% | 80.84% |
| 100 | 99.653% | 4008.23 | 111.76 | 46.60 | 65.17 | 58.31% | 0.17% | 89.08% |
| 150 | 99.835% | 5249.20 | 161.18 | 59.64 | 101.55 | 63.00% | 0.28% | 92.41% |

Baseline, logger disabled, and logger enabled produce exactly identical ordered top10 IDs and float distances for every query and L. Trace integrity checks cover unique query IDs, 10000-query completeness, edge/degree counts, fresh count, admitted count, expanded uniqueness, and first-discovery ancestry.

## Counterfactual validation

| L | skip mask | masked nodes/query | Recall@10 | identical top10 queries |
|---:|---|---:|---:|---:|
| 20 | skip_zero_new | 0.01 | 91.027% | 10000/10000 |
| 20 | skip_zero_admitted | 12.66 | 91.027% | 10000/10000 |
| 20 | skip_off_path | 23.28 | 92.591% | 7635/10000 |
| 50 | skip_zero_new | 0.04 | 98.395% | 10000/10000 |
| 50 | skip_zero_admitted | 30.42 | 98.395% | 10000/10000 |
| 50 | skip_off_path | 50.77 | 98.880% | 9326/10000 |
| 100 | skip_zero_new | 0.19 | 99.653% | 10000/10000 |
| 100 | skip_zero_admitted | 65.17 | 99.653% | 10000/10000 |
| 100 | skip_off_path | 99.56 | 99.734% | 9815/10000 |
| 150 | skip_zero_new | 0.45 | 99.835% | 10000/10000 |
| 150 | skip_zero_admitted | 101.55 | 99.835% | 10000/10000 |
| 150 | skip_off_path | 148.95 | 99.837% | 9914/10000 |

All skip masks are per-query hindsight oracles generated from the baseline trace. The intervention skips adjacency distance calculations and neighbor insertion AFTER fetching/scoring the parent. It therefore measures removable neighbor scanning, not saved disk reads, not full node expansion, and not a deployable gate.

`skip_zero_new` removes only scans with all neighbors already visited; the rerun must preserve IDs, distances, visited counts and expanded counts exactly. `skip_zero_admitted` removes scans that admitted nothing in baseline; reruns reveal consequences of changing the visited set. `skip_off_path` preserves first-discovery ancestors of the baseline returned top10 and removes other scans; ancestry alone does not guarantee identical search dynamics because the bounded queue can change.

The fraction labelled useless is an empirical local no-progress fraction, not an unconditional safe skipping rate. First-discovery off-path fraction is an oracle opportunity estimate rather than a maximum or proof of necessity. Any learned gate needs held-out queries and online features, plus recall versus actual skipped work measurement.

Trace logging stores all neighbor edges and serializes file writes after each query; logged QPS is instrumented overhead and must not be reported as baseline performance. Results are configuration-specific; SIFT1M numbers do not transfer automatically to other datasets or DiskANN versions.

Artifacts: `summary.json` (means/quantiles/counterfactual parity), `per_query_L.csv`, `trace_L.trace` (all real edges), `trace_format.md`, `*.mask`, baseline/logger/skip logs and results, `provenance.json`, `groundtruth_validation.json`.

Re-run: `../../.venv/bin/python run_experiment.py` from this directory after baseline exists. Baseline command is saved in `baseline_command.txt`; index build command in `build_command.txt`.

## Full node expansion oracle (actual I/O avoided)

Here the mask contains baseline zero-admission nodes EXCLUDING every baseline final top10 ID. The intervention occurs immediately after popping/marking the candidate, before adjacency cache lookup or disk read, and reserves the original beam slot. Thus both adjacency work and exact node scoring are skipped; uncached masked nodes require no disk reads. Every query returns exactly the same ordered IDs and distances.

| L | full expansions skipped/query | fraction | remaining expansions/query | Recall@10 | identical queries |
|---:|---:|---:|---:|---:|---:|
| 20 | 8.44 | 24.30% | 26.30 | 91.027% | 10000/10000 |
| 50 | 28.22 | 44.95% | 34.57 | 98.395% | 10000/10000 |
| 100 | 64.21 | 57.45% | 47.55 | 99.653% | 10000/10000 |
| 150 | 101.06 | 62.70% | 60.12 | 99.835% | 10000/10000 |

This is a validated oracle lower bound on removable complete node expansions for these runs. It is not a claimed maximum, does not provide an online predictor, and requires hindsight knowledge of admissions and final results. A node with zero useful neighbors may itself belong to the final top10, which is why naive full-node skipping of all useless nodes is invalid. Raw I/O and latency measurements are in `full_skip.log`; speedup claims require repeated warm-cache/uncached randomized timing trials.

Re-run this extension with `../../.venv/bin/python run_full_skip.py` after the main experiment.
