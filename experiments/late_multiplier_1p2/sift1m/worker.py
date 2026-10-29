from common import *
configs={c['key']:c for c in P['configs']};groups=P['groups']
assert all(sha(p)==h for p,h in P['source_hashes'].items())
v=load('validation',list(configs.values()),0,True);dump(D/'validation.json',v)
allruns={}
for study,repeats in [('main',12),('recall_reference',6)]:
    keys=groups[study];rs=[]
    for rep in range(repeats):
        order=keys[rep%6:]+keys[:rep%6]
        measured=load('test',[configs[k] for k in order],rep,True)
        for r in measured:allruns[(r['key'],r['repeat'])]=r
        rs+=measured
    dump(D/(study+'_runs.json'),rs)
assert all(sha(p)==h for p,h in P['source_hashes'].items())
dump(D/'all_test_runs.json',list(allruns.values()))
dump(D/'MEASUREMENTS_COMPLETE.json',dict(complete=True,unique_test_runs=len(allruns),validation_configs=len(configs)))
print('DATASET COMPLETE',P['dataset'],flush=True)
