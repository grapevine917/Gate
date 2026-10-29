import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import subprocess,json,hashlib,random
import numpy as np
R=Path(__file__).resolve().parent;LG=R.parents[1];S=LG/'experiments/sift1m';DATA=LG/'experiments/e1_similarity/data'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def matrix(p):
 with p.open('rb') as f:n,d=np.fromfile(f,'<u4',2);return np.fromfile(f,'<f4').reshape(n,d)
(R/'data').mkdir(exist_ok=True)
for split in ['validation','test']:
 for ext in ['fbin','gt']:
  a=matrix(DATA/f'{split}.{ext}')[:300]
  with (R/f'data/{split}.{ext}').open('wb') as f:np.array(a.shape,'<u4').tofile(f);a.tofile(f)
 ids=np.load(DATA/f'{split}_query_ids.npy')[:300];np.save(R/f'data/{split}_ids.npy',ids)
assert not set(np.load(R/'data/validation_ids.npy'))&set(np.load(R/'data/test_ids.npy'))
taus=[0,.016,.024,.037,.056,.082,.12,.18,.27,.4,.6,.85,1.01]
names=['MLP','max_skip_1','max_skip_2','keep_best_1']
source_files=[LG/'DiskANN/src/pq_flash_index.cpp',LG/'DiskANN/include/online_gate.h',S/'online_state_mlp/model.bin']
locked={str(p):sha(p) for p in source_files}
dump(R/'PROTOCOL.json',dict(dataset='SIFT1M',queries_per_split=300,L=100,K=10,beam=4,threads=4,cache_nodes=10000,taus=taus,policies=names,selection='validation min reads under absolute Recall targets; test frozen thresholds; same-recall curves are interpolation estimates only',targets=[99.,99.3,99.5,99.6],timing='warmup per run; randomized serial runs; selected test configs 3 repeats; model loading excluded; beam preview copy overhead included',keep_best='highest retain score among original next <=4 candidates; cached candidates protected and count toward minimum Keep; no extra model calls',original_hashes=locked,limitations=['previously observed test set; exploratory','300 queries; finite recall resolution','warm OS cache; submitted SSD requests, not physical device reads']))
(R/'runs').mkdir(exist_ok=True)
def run(split,p,t,rep=0,reference=False):
 key=f'{split}_p{p}_t{t}_r{rep}'.replace('.','p')+('_ref' if reference else '');prefix=R/'runs'/key
 if not prefix.with_suffix('.csv').exists():
  binary=LG/'experiments/e1_similarity/native/collect' if reference else R/'native/collect'
  args=[str(binary),str(S/'index'),str(R/f'data/{split}.fbin'),str(R/f'data/{split}.gt'),str(S/'online_state_mlp/model.bin'),str(t),'-' if reference else str(p),str(prefix),'time']
  dump(prefix.with_suffix('.command.json'),args)
  with prefix.with_suffix('.log').open('w') as out:subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,check=True)
 a=np.atleast_1d(np.genfromtxt(prefix.with_suffix('.csv'),delimiter=',',names=True))
 assert len(a)==300 and np.all(a['reads']<=a['expansions'])
 return dict(split=split,policy=names[p],p=p,tau=t,repeat=rep,recall=float(a['hits'].mean()*10),reads=float(a['reads'].mean()),expansions=float(a['expansions'].mean()),latency_us=float(a['total_us'].mean()),p95_us=float(np.percentile(a['total_us'],95)),skips=float(a['skips'].mean()),prefix=str(prefix))
# Verify original baseline parity, and all policies reduce to no pruning at tau0.
checks={}
for t in [0,.056,.27]:
 b=run('validation',0,t);ref=run('validation',0,t,reference=True)
 assert Path(b['prefix']+'.ids').read_bytes()==Path(ref['prefix']+'.ids').read_bytes()
 for k in ['recall','reads','expansions','skips']:assert b[k]==ref[k]
 checks[str(t)]='original frozen E1 collector IDs and cost counts identical'
b=run('validation',0,0)
for p in [1,2,3]:
 x=run('validation',p,0)
 assert Path(x['prefix']+'.ids').read_bytes()==Path(b['prefix']+'.ids').read_bytes()
 for k in ['reads','expansions','skips']:assert x[k]==b[k]
dump(R/'verification.json',checks)
val=[];jobs=[(p,t) for p in range(4) for t in taus];random.Random(20261020).shuffle(jobs)
for p,t in jobs:
 x=run('validation',p,t);val.append(x);print('validation',names[p],t,x['recall'],x['reads'],flush=True)
dump(R/'validation.json',val)
selected=[]
for target in [99.,99.3,99.5,99.6]:
 for p in range(4):
  feasible=[x for x in val if x['p']==p and x['recall']>=target]
  selected.append(dict(target=target,p=p,tau=min(feasible,key=lambda x:x['reads'])['tau'] if feasible else None))
dump(R/'SELECTION_LOCK.json',selected)
# Entire predeclared test grid for descriptive same-recall curves, never reselect configurations from it.
test=[];random.Random(23).shuffle(jobs)
for p,t in jobs:
 x=run('test',p,t);test.append(x);print('test',names[p],t,x['recall'],x['reads'],flush=True)
configs=sorted({(x['p'],x['tau']) for x in selected if x['tau'] is not None})
for rep in [1,2]:
 random.Random(rep).shuffle(configs)
 for p,t in configs:test.append(run('test',p,t,rep))
dump(R/'test.json',test)
assert all(sha(Path(p))==v for p,v in locked.items())
lines=['# 同 Beam 剪枝约束小实验','','冻结原17维MLP，SIFT1M，验证/测试各300条，L100、beam4、K10、4线程、10000节点缓存。原DiskANN源文件及模型哈希未变。独立实验库加入约束；Strategy 3复制队列预评分，每个非缓存候选只推理一次，复制及选择开销计入延迟。缓存节点始终保留并计入至少Keep一个。','','原MLP与既有原生程序在三个阈值的最终ID、SSD请求、展开及Skip数完全相同；所有约束在tau0返回相同ID和计数。','','## 验证集选参后的真实测试配置','','Recall目标为验证集下限，不意味着各行测试Recall严格相同。SSD为每query提交请求数。选定配置测试三轮，成本取中位数；单query延迟含并发调度，暖OS缓存。','','|验证Recall下限%|策略|阈值|测试Recall%|SSD请求/q|展开/q|延迟µs/q|','|---|---|---|---|---|---|---|']
for x in selected:
 if x['tau'] is None:lines.append(f"|{x['target']}|{names[x['p']]}|不可行|||||");continue
 rows=[r for r in test if r['p']==x['p'] and r['tau']==x['tau']]
 assert len(rows)==3
 for k in ['recall','reads','expansions','skips']:assert len({r[k] for r in rows})==1
 m={k:float(np.median([r[k] for r in rows])) for k in ['recall','reads','expansions','latency_us']}
 lines.append(f"|{x['target']}|{names[x['p']]}|{x['tau']}|{m['recall']:.3f}|{m['reads']:.3f}|{m['expansions']:.3f}|{m['latency_us']:.2f}|")
lines+=['','## 相同测试Recall的曲线插值估计','','仅用预先声明网格的相邻实测Recall点做线性插值；不是单配置实测，不用于重新选择阈值。曲线扫描点仅一轮计时，延迟只作探索诊断。相同Recall重复点取最少SSD请求，非单调区间不强行单调化。','','|Recall%|策略|SSD请求/q估计|展开/q估计|延迟µs/q估计|','|---|---|---|---|---|']
for target in [99.,99.3,99.5,99.6]:
 for p in range(4):
  rows=[x for x in test if x['p']==p and x['repeat']==0];unique={}
  for x in sorted(rows,key=lambda x:-x['reads']):unique[x['recall']]=x
  rs=sorted(unique.values(),key=lambda x:x['recall']);rec=[x['recall'] for x in rs]
  if not rec[0]<=target<=rec[-1]:lines.append(f'|{target}|{names[p]}|N/A|N/A|N/A|');continue
  vals=[np.interp(target,rec,[x[k] for x in rs]) for k in ['reads','expansions','latency_us']]
  lines.append(f'|{target}|{names[p]}|{vals[0]:.3f}|{vals[1]:.3f}|{vals[2]:.2f}|')
lines+=['','本实验仅检验beam级保护是否有收益，不能证明模型识别了替代或互补关系。测试集此前已被其他实验观察；300条查询且关键损失稀少，结论为探索性。限制Skip可能只让剪枝变保守，必须结合相同Recall曲线判断。','', '协议、完整验证/测试网格、冻结阈值和逐query结果保存在同目录。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n');dump(R/'COMPLETE.json',dict(original_files_unchanged=True,validation=len(val),test_runs=len(test),verified=True))
print('COMPLETE',flush=True)
