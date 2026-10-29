# Training sufficiency / longer-training experiment

Run from the project root:

```bash
bash experiments/training_extension/run.sh
```

The workflow preserves existing models and results, resumes completed training
trials and search runs, and uses a fresh directory for all new artifacts.

The three neural tasks are the mixed-L state-only Gate, the original unpruned
prefix-Recall ET, and the previous LR-pruned-trajectory ET. Architecture, labels,
data, batch size and optimizer family are fixed. Each task trains three seeds for
160 epochs. The learning rate is 0.001 for epochs 1–40, 0.0003 for 41–80,
0.0001 for 81–120 and 0.00003 for 121–160. There is no short-patience early stop.
The existing 200-query internal holdout selects seed and epoch. Refit on all
1,200 training queries using the selected seed, epoch and fixed schedule.

The first 40 epochs provide a same-seed, same-data reference. This comparison
tests longer training plus a lower learning rate, not epoch count alone.
Evaluating many checkpoints and seeds can itself overfit the internal holdout;
native search on external validation/test remains necessary. Existing benchmark
test queries have been observed in historical experiments, so the findings are
exploratory rather than a new confirmatory evaluation.

The LR diagnosis retains C=1 and the same normalization/data, while tightening
tolerance from 1e-5 to 1e-9 and raising the iteration cap from 1,500 to 10,000.
This checks optimization convergence; it does not add model capacity.

Evaluation separately compares old/new Gate MLP with the frozen LR reference,
old/new unpruned ET, and old/new adapted ET with the same frozen LR Gate whose
trajectories trained that ET. It does not combine the new MLP Gate with an ET
trained on LR trajectories. All Gate models recalibrate the same eight
retain-FNR budgets on validation baseline states. These budgets are not final
Recall guarantees. Joint ET scans three of those budgets crossed with eight ET
thresholds and five L values. A 64-L DiskANN baseline is included.

Validation alone locks configurations for each quality target and objective.
Only locked configurations run on test, three times each. Recall is recomputed
from returned IDs; request counts and actual Linux read bytes are checked.
Timing includes feature extraction, model inference, I/O and search. Compare
timings within this run, not against historical measurements.

Training histories are under `models/`. `LR_CONVERGENCE.json` records the LR
diagnosis. Final prediction diagnostics, locked search results, verification and
interpretation are in `prediction_metrics.json`, `validation_locked_test.csv`,
`verification.json` and `REPORT.md`. Standard plots can be produced from the
CSV/JSON artifacts when a plotting package is available.
