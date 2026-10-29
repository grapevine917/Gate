"""Independent per-query repeat checks and original-program parity."""
from pathlib import Path
import json,hashlib,platform,os,importlib.metadata
from collections import defaultdict,Counter
import numpy as np
R=Path(__file__).resolve().parent
runs=json.loads((R/'all_test_runs.json').read_text());groups=defaultdict(list)
for r in runs:groups[r['key']].append(r)
checks=0
fields=['hits','reads','expansions','skips','gate_evaluated','et_evaluated','et_stopped','model_calls']
for key,rows in groups.items():
 for r in rows:
  meta=json.loads(Path(r['prefix']+'.meta.json').read_text());assert np.isclose(meta['qps'],1000/meta['wall_seconds'],rtol=1e-10)
 if len(rows)==1:continue
 assert len(rows)==3;original=np.genfromtxt(rows[0]['prefix']+'.csv',delimiter=',',names=True)
 for r in rows[1:]:
  current=np.genfromtxt(r['prefix']+'.csv',delimiter=',',names=True)
  for k in fields:np.testing.assert_array_equal(original[k],current[k],err_msg=key+' '+k)
  checks+=1
# Exact old native DiskANN L100 reference on identical1000 held-out query IDs.
E=R.parent/'e1_similarity';np.testing.assert_array_equal(np.load(R/'data/test_ids.npy'),np.load(E/'data/test_query_ids.npy'))
old=np.fromfile(E/'runs_main/test_DiskANN_r0_time.ids','<u8');new=np.fromfile(R/'runs/test/DiskANN_L100_r0.ids','<u8');np.testing.assert_array_equal(old,new)
a=np.genfromtxt(E/'runs_main/test_DiskANN_r0_time.csv',delimiter=',',names=True);b=np.genfromtxt(R/'runs/test/DiskANN_L100_r0.csv',delimiter=',',names=True)
for k in ['hits','reads','expansions']:np.testing.assert_array_equal(a[k],b[k])
parts={s:set(np.load(R/f'data/{s}_ids.npy').tolist()) for s in ['train','validation','test']}
assert all(not parts[a]&parts[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
assert set(Counter(r['family'] for r in json.loads((R/'aggregate.json').read_text())).values())=={64}
v=json.loads((R/'verification.json').read_text());v.update(per_query_repeat_fields=fields,per_query_repeated_pair_checks=checks,original_E1_DiskANN_L100_full_IDs_and_per_query_counts_equal=True,train_validation_test_query_disjoint=True);(R/'verification.json').write_text(json.dumps(v,indent=2)+'\n')
cpu={}
for line in Path('/proc/cpuinfo').read_text().splitlines():
 if line.startswith(('model name','vendor_id','cpu family','model\t')):
  k,value=line.split(':',1);cpu[k.strip()]=value.strip()
versions={}
for package in ['numpy','scipy','torch','scikit-learn','xgboost-cpu']:
 try:versions[package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError:pass
env=dict(platform=platform.platform(),cpu=cpu,logical_cpus=os.cpu_count(),query_threads=4,MKL_threads_per_query=1,RAM_memtotal=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:')),versions=versions)
(R/'environment.json').write_text(json.dumps(env,indent=2)+'\n');print('PER QUERY AND ORIGINAL PROGRAM VERIFICATION PASSED',checks)
