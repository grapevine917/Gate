# Top3 heuristic baselines

两条请求的规则：rank>r；rank>r AND completed_expansions>e。只使用Top3中的rank和completed，不使用visited_count。
rank为zero-based队列位置，在标记当前候选expanded之前取得。completed为beam开始前已完成精确评分的展开数；beam内未完成的读取不计入。
r=-1..99、e=-1..validation最大completed，穷举整数网格。在每档validation retain FNR预算下最大化样本加权SkipRate；并列依次选择较少误跳过、较大r、较大e。e=-1等价于只用rank。
所有参数先保存锁定，然后才评估test。未根据test选择参数；learned Top3/All17使用原有validation校准tau。目标FNR不是在线Recall损失保证。

| Rule | val FNR target | r | e | Validation SkipRate | Validation actual FNR | Test SkipRate | Test actual FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rank | 0.1% | 98 | — | 0.862% | 0.0940% | 0.866% | 0.0672% |
| Rank | 0.25% | 97 | — | 1.719% | 0.1879% | 1.717% | 0.1259% |
| Rank | 0.5% | 94 | — | 4.240% | 0.4468% | 4.237% | 0.4072% |
| Rank | 1% | 89 | — | 8.475% | 0.9333% | 8.483% | 0.9173% |
| Rank | 2% | 80 | — | 16.103% | 1.9960% | 16.100% | 1.9710% |
| Rank+Progress | 0.1% | 98 | 100 | 0.862% | 0.0940% | 0.866% | 0.0672% |
| Rank+Progress | 0.25% | 2 | 106 | 2.511% | 0.2485% | 2.597% | 0.2666% |
| Rank+Progress | 0.5% | 92 | 104 | 4.970% | 0.4781% | 5.044% | 0.4660% |
| Rank+Progress | 1% | 72 | 100 | 9.193% | 0.9980% | 9.277% | 0.9886% |
| Rank+Progress | 2% | 64 | 92 | 16.213% | 1.9918% | 16.251% | 1.9521% |

## Live curves

![Recall / cost](online/recall_cost_curves.png)

两条heuristic和Learned Top3、Learned All17及无模型DiskANN重新交错测量。固定L=100、K=10、beam=4；每个配置1000 test query、3 repeats，中位数，含实际决策开销。
[Full online measurements](online/REPORT.md) · [Online CSV](online/summary.csv) · [Offline CSV](offline_summary.csv)

## Interpretation

验证FNR目标1%时，Rank规则r=89，Rank+Progress规则r=72/e=100；两者test Recall均为99.52%，Reads为92.497/91.744。Learned Top3为Recall 99.58%、Reads 83.825，Learned All17为99.59%、83.209；这两条简单规则在这一档位均被learned的召回/读取点支配。
相同Recall=99.64%的实测点中，Rank最少Reads=100.411，Rank+Progress最少Reads=101.389，Learned Top3最少Reads=93.286，Learned All17最少Reads=89.327。该比较使用各曲线已有实测点，不插值或重新根据test选择部署参数。
Rank+Progress的各r/e组合分别在validation优化，规则集合不一定嵌套；在线Recall不必随FNR目标单调。例如0.25%档在test上被0.5%档支配，图中保留原实测点。参数优化目标是离线validation标签SkipRate，而不是在线validation Recall/I/O；结论仅针对这两条规则及这一调参流程。

## Reproduce

在learned-gate目录依次运行：

```bash
cmake --build DiskANN/build --target search_disk_index -j 4
.venv/bin/python experiments/sift1m/calibrate_heuristics.py
.venv/bin/python experiments/sift1m/run_online_heuristics.py
.venv/bin/python experiments/sift1m/plot_heuristic_curves.py
```
