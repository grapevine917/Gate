#!/usr/bin/env python3
"""Measure varying-L DiskANN on the same held-out query set as online M3."""
import json
import numpy as np
from run_online_m3 import E,execute
OUT=E/'online_curves'; OUT.mkdir(exist_ok=True)
q=E/'online_m3/test_query.fbin';gt=E/'online_m3/test_gt.bin'
execute('varying_l_warmup',q,gt,{},output_dir=OUT)
rng=np.random.default_rng(42);runs=[]
for repeat in range(3):
    for l in rng.permutation([20,40,50,60,70,80,90,100,120,150]):
        row=execute(f'baseline_r{repeat}_L{l}',q,gt,{},search_l=int(l),output_dir=OUT)
        row.update(L=int(l),repeat=repeat);runs.append(row)
        (OUT/'varying_l_runs.json').write_text(json.dumps(runs,indent=2))
        print('L',l,'repeat',repeat,'recall',row['recall_pct'],'reads',row['ssd_reads_per_query'],flush=True)
summary=[]
for l in sorted({r['L'] for r in runs}):
    rows=[r for r in runs if r['L']==l]
    assert len({r['recall_pct'] for r in rows})==1
    summary.append(dict(L=l,**{k:float(np.median([r[k] for r in rows])) for k in
        ['recall_pct','ssd_reads_per_query','expansions_per_query','latency_us','qps']}))
(OUT/'varying_l_summary.json').write_text(json.dumps(summary,indent=2))
print('VARYING L COMPLETE',flush=True)
