"""Fixed 1.2x late threshold, two datasets, no training or test-time selection."""
from pathlib import Path
import os,json,hashlib,csv,subprocess,time,copy
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,rows):
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare(ds):
    out=D/ds;out.mkdir(exist_ok=True);old=ROOT/'experiments'/f'{ds}_mild_dynamic_recall_aligned'
    A=old if ds=='sift100m' else ROOT/'experiments/sift1m_p0a_L100_train_compare'
    W=ROOT/'experiments'/('sift100m_p0' if ds=='sift100m' else 'sift1m_wide_p0')
    lock=json.loads((old/'MAIN_SELECTION_LOCK.json').read_text());base=json.loads((old/'configs.json').read_text());C={c['key']:c for c in base}
    match=json.loads((old/'ALIGNMENT_TEST_MATCH.json').read_text());groups={};configs={}
    for study,cs in [('main',lock['configs']),('recall_reference',[C[k] for k in match['keys']])]:
        group=[]
        for c in copy.deepcopy(cs):
            if c['kind'] in [4,5]:
                c['multiplier']=1.2;c['late_tau']=1.2*c['tau'];base_token=f"{c['base_scale']:g}".replace('.','p');c['key']=f"Dynamic{c['family']}_b{base_token}_m1p2_r{c['rank']}_e{c['exp']}"
            if c['key'] in configs:assert configs[c['key']]==c
            configs[c['key']]=c;group.append(c['key'])
        groups[study]=group
    sources={str(p):sha(p) for p in [A/'native/search',A/'models_train100/state.bin',A/'models_train100/lr.bin',old/'MAIN_SELECTION_LOCK.json',old/'ALIGNMENT_TEST_MATCH.json',old/'configs.json',W/'data/validation.gt',W/'data/test.gt',W/'data'/('validation.u8bin' if ds=='sift100m' else 'validation.fbin'),W/'data'/('test.u8bin' if ds=='sift100m' else 'test.fbin')]}
    protocol=dict(dataset=ds,training_L=100,retrained=False,multiplier=1.2,definition='late_tau=1.2*base_tau; same base threshold and rank/exp triggers as previous corresponding main or Recall-reference study',main_L=100,K=10,beam=4,threads=4,cache_nodes=10000,seed=20261024,main_repeats=12,recall_reference_repeats=6,validation_queries=1000,test_queries=1000,groups=groups,configs=list(configs.values()),source_hashes=sources,selection='Fixed by user; no grid search or parameter selection. Evaluate validation then test; Recall-reference configs borrowed from previous descriptive test matching, explicitly exploratory and NOT guaranteed still aligned.',timing='Sequential datasets; cyclic6-method ordering; shared config/repeat measured once; contemporaneous controls',IO='Strict reader bytes == Linux read_bytes ==4096*reads; reject/archive and retry mismatch; no swap')
    if (out/'PROTOCOL.json').exists():assert json.loads((out/'PROTOCOL.json').read_text())==protocol
    else:dump(out/'PROTOCOL.json',protocol)
    # Reuse verified readers, with all output redirected to this new directory.
    source=(old/'measurement.py').read_text()
    (out/'measurement.py').write_text(source)
    common='''import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,hashlib,json,random,subprocess,time,shutil
import numpy as np
'''+f'D=Path({str(out)!r});A=Path({str(A)!r});W=Path({str(W)!r});ROOT=Path({str(ROOT)!r})\n'+'''def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
P=json.loads((D/'PROTOCOL.json').read_text())
for p in [D/'logs',D/'runs/validation',D/'runs/test']:p.mkdir(parents=True,exist_ok=True)
exec(compile((D/'measurement.py').read_text(),str(D/'measurement.py'),'exec'),globals())
'''
    # The 1M helper's index path was relative to the previous experiment parent.
    source=source.replace("D.parent/'sift1m/index'",f"Path({str(ROOT/'experiments/sift1m/index')!r})")
    (out/'measurement.py').write_text(source);(out/'common.py').write_text(common)
    (out/'worker.py').write_text('''from common import *
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
''')
    return out
if __name__=='__main__':
    folders=[prepare(ds) for ds in ['sift1m','sift100m']]
    dump(D/'RUN_LOCK.json',dict(protocols={str(p/'PROTOCOL.json'):sha(p/'PROTOCOL.json') for p in folders},coordinator=sha(Path(__file__))))
    for out in folders:
        print('DATASET',out.name,flush=True)
        subprocess.run([str(ROOT/'.venv/bin/python'),str(out/'worker.py')],check=True)
    subprocess.run([str(ROOT/'.venv/bin/python'),str(D/'report.py')],check=True)
