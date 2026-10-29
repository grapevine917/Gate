"""Resumable, isolated Gate/ET trajectory-adaptation experiment."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='4'
from pathlib import Path
import json, hashlib, subprocess, random, csv, struct, sys, time
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parents[1]; M=R.parent/'main_comparison'; B=R.parent/'p0_b_lr'
SEED=20261021
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def log(s):print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def run(cmd,path):
 with path.open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
def csvout(p,rows):
 if rows:
  with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare():
 p0=json.loads((B/'PROTOCOL.json').read_text())
 protocol=dict(seed=SEED,dataset='SIFT1M',gate='Frozen P0-C mixed-L 17D LR; no Gate retraining',L=[30,75,100,160,300],gate_thresholds=[0,p0['gate_thresholds'][4],p0['gate_thresholds'][5],p0['gate_thresholds'][6]],et_thresholds=p0['et_thresholds'],baseline_L=p0['baseline_L'],targets=[95,98,99,99.5,99.7],objectives=['physical_bytes','p95_us','qps'],train_queries=1200,internal_holdout_queries=200,validation_queries=1000,test_queries=1000,beam=4,K=10,cache_nodes=10000,threads=4,training='Same 17-128-64-1 sigmoid MSE ET architecture/optimizer as original. Train unpruned control and pooled Gate-trajectory ET with query-level inner holdout; early stop then refit all training queries. Inputs contain only 17 live features; Gate threshold and future outcomes are not inputs.',label='Actual GT Recall@10 of exact-scored prefix; final Gate-only recall recorded for diagnosis, not substituted for absolute quality.',tuning='Full Cartesian L x Gate threshold x ET threshold for each joint family; ET-only uses all ET thresholds; baseline uses existing 64 L. Greater tuning budget for joint families is explicit. Select on validation only, freeze before any test. Test only locked configurations plus their fixed-parameter disable-Gate / disable-ET controls; 3 repeats.',timing='Native search, feature extraction, inference and I/O included; load and 128-query warmup excluded. Sequential workloads. Interleave model/family batches.',limits=['Previously observed benchmark queries; exploratory, not a new confirmatory sample','Finite grid, joint search has more configurations than single strategies','WSL2 guest O_DIRECT/block read traffic, not NAND telemetry','Only LR Gate evaluated; MLP Gate adaptation remains outside this run','No promise that average-recall validation constraint transfers to test'])
 if (R/'PROTOCOL.json').exists():assert json.loads((R/'PROTOCOL.json').read_text())==protocol
 else:dump(R/'PROTOCOL.json',protocol)
 inputs=[M/'models/lr.bin',M/'models/et.bin',M/'native/libdiskann.a',M/'native/pq_flash_index.cpp',R.parent/'sift1m/index_disk.index',R.parent/'sift1m/index_pq_compressed.bin',R.parent/'sift1m/index_pq_pivots.bin']
 inputs += list((M/'data').glob('*ids.npy'))+[M/'data'/f'{s}.{ext}' for s in ['train','validation','test'] for ext in ['fbin','gt']]
 lock={str(p):sha(p) for p in inputs}
 if (R/'INPUT_LOCK.json').exists():assert json.loads((R/'INPUT_LOCK.json').read_text())==lock
 else:dump(R/'INPUT_LOCK.json',lock)
 ids={s:set(np.load(M/'data'/f'{s}_ids.npy').tolist()) for s in ['train','validation','test']}
 assert not(ids['train']&ids['validation'] or ids['train']&ids['test'] or ids['validation']&ids['test'])
 assert set(np.load(M/'data/internal_holdout_ids.npy'))<=ids['train']
 dump(R/'split_verification.json',{s:len(v) for s,v in ids.items()}|{'query_disjoint':True})
 template=json.loads((B/'build.command.json').read_text());source=template.index(str(B/'search.cpp'));out=template.index('-o')+1
 cmds=[]
 for name in ['collect','verify_model']:
  cmd=template.copy();cmd[source]=str(R/f'{name}.cpp');cmd[out]=str(R/name);cmds.append(cmd)
  if not (R/name).exists() or (R/name).stat().st_mtime<(R/f'{name}.cpp').stat().st_mtime:run(cmd,R/f'build_{name}.log')
 dump(R/'build.commands.json',cmds)
 return protocol
def collect(p):
 if (R/'COLLECT_COMPLETE.json').exists():return
 configs=R/'collection.configs';configs.write_text(''.join(f'{L} {t:.17g}\n' for L in p['L'] for t in p['gate_thresholds']))
 run([R/'collect',R.parent/'sift1m/index',M/'data/train.fbin',M/'data/train.gt',M/'models/lr.bin',configs,R/'data/prefixes.bin',32],R/'collect.log')
 a=np.fromfile(R/'data/prefixes.bin','<f4').reshape(-1,23)
 assert np.isfinite(a).all() and set(a[:,0].astype(int))==set(range(1200))
 dump(R/'COLLECT_COMPLETE.json',dict(rows=len(a),columns=23,sha256=sha(R/'data/prefixes.bin')));log(f'COLLECT COMPLETE {len(a)} boundaries')
def train(p):
 if (R/'TRAIN_COMPLETE.json').exists():return
 import torch
 from torch import nn
 torch.set_num_threads(4)
 a=np.fromfile(R/'data/prefixes.bin','<f4').reshape(-1,23);hold=np.load(M/'data/internal_holdout_ids.npy');ids=np.load(M/'data/train_ids.npy');inner=~np.isin(ids[a[:,0].astype(int)],hold)
 class Net(nn.Module):
  def __init__(self):super().__init__();self.seq=nn.Sequential(nn.Linear(17,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
  def forward(self,x):return self.seq(x).squeeze(-1)
 for name,use in [('unpruned',a[:,2]==0),('adapted',np.ones(len(a),bool))]:
  x=a[use,6:].copy();y=a[use,1].copy();mask=inner[use]
  def fit(ix,epochs,valid=None):
   mu=x[ix].mean(0,dtype='f8').astype('f4');sd=x[ix].std(0,dtype='f8').astype('f4');sd[sd<1e-8]=1;z=torch.from_numpy(np.ascontiguousarray((x-mu)/sd));yt=torch.from_numpy(y)
   torch.manual_seed(SEED);net=Net();opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4);rng=np.random.default_rng(SEED);indices=np.flatnonzero(ix);hist=[];best=1;bestloss=float('inf');stale=0
   for ep in range(1,epochs+1):
    net.train();total=0
    for batch in np.array_split(rng.permutation(indices),max(1,int(np.ceil(len(indices)/2048)))):
     loss=((torch.sigmoid(net(z[batch]))-yt[batch])**2).mean();opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch)
    row=dict(epoch=ep,train_mse=total/len(indices));net.eval()
    if valid is not None:
     with torch.no_grad():v=float(((torch.sigmoid(net(z[valid]))-yt[valid])**2).mean())
     row['holdout_mse']=v
     if v<bestloss-1e-5:bestloss=v;best=ep;stale=0
     else:stale+=1
    hist.append(row)
    if ep%5==0:log(f'TRAIN {name} {row}')
    if valid is not None and stale>=6:break
   return net,mu,sd,hist,best
  _,_,_,hist,best=fit(mask,40,~mask);dump(R/'models'/f'{name}_inner.json',hist)
  net,mu,sd,hist,_=fit(np.ones(len(x),bool),best);dump(R/'models'/f'{name}_history.json',hist)
  with (R/'models'/f'{name}.bin').open('wb') as f:
   f.write(b'STGATE01');mu.astype('<f4').tofile(f);sd.astype('<f4').tofile(f)
   for layer in net.seq:
    if isinstance(layer,nn.Linear):layer.weight.detach().numpy().T.copy().astype('<f4').tofile(f);layer.bias.detach().numpy().astype('<f4').tofile(f)
  torch.save(dict(weights=net.state_dict(),mean=mu,scale=sd),R/'models'/f'{name}.pt')
  ix=np.random.default_rng(SEED).choice(len(x),min(2000,len(x)),replace=False)
  with torch.no_grad():pred=torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray((x[ix]-mu)/sd)))).numpy()
  with (R/'models'/f'{name}_check.bin').open('wb') as f:f.write(struct.pack('<I',len(ix)));np.column_stack([x[ix],pred]).astype('<f4').tofile(f)
  run([R/'verify_model',R/'models'/f'{name}.bin',R/'models'/f'{name}_check.bin'],R/'models'/f'{name}_native_check.log')
  dump(R/'models'/f'{name}_metrics.json',dict(samples=len(x),best_epoch=best,model_sha256=sha(R/'models'/f'{name}.bin'),native_max_error=float((R/'models'/f'{name}_native_check.log').read_text()),inner_best_mse=min(h['holdout_mse'] for h in json.loads((R/'models'/f'{name}_inner.json').read_text()))))
 dump(R/'TRAIN_COMPLETE.json',dict(complete=True,models=['unpruned','adapted']));log('TRAIN COMPLETE')
def configurations(p):
 cs=[]
 def add(family,model,kind,L,tau=0,et=1.1):
  key=f'{family}_L{L}_g{p["gate_thresholds"].index(tau)}_e{p["et_thresholds"].index(et)}'
  cs.append(dict(key=key,family=family,model=model,kind=kind,L=L,tau=tau,et_tau=et,rank=0,exp=0))
 for L in p['baseline_L']:add('DiskANN','original',0,L)
 for L in p['L']:
  for t in p['gate_thresholds']:add('Gate','original',4,L,t)
  for model,fam in [('original','ET_original'),('unpruned','ET_unpruned')]:
   for et in p['et_thresholds']:add(fam,model,6,L,et=et)
  for model,fam in [('original','Joint_original'),('unpruned','Joint_unpruned'),('adapted','Joint_adapted')]:
   for t in p['gate_thresholds']:
    for et in p['et_thresholds']:add(fam,model,7,L,t,et)
 dump(R/'configs.json',cs);return cs
def modelpath(name):return M/'models/et.bin' if name=='original' else R/'models'/f'{name}.bin'
def valid(prefix):
 try:return json.loads(prefix.with_suffix('.meta.json').read_text())['queries']==1000 and prefix.with_suffix('.ids').stat().st_size==80000
 except (OSError,ValueError):return False
def evaluate(split,configs,repeat):
 out=R/'runs'/split;out.mkdir(parents=True,exist_ok=True);todo=[c for c in configs if not valid(out/f'{c["key"]}_r{repeat}')]
 rng=random.Random(SEED+repeat+100*(split=='test'));rng.shuffle(todo)
 # Small, independently shuffled model-homogeneous batches avoid concurrent timing.
 batches=[]
 for model in ['original','unpruned','adapted']:
  group=[c for c in todo if c['model']==model]
  batches += [group[j:j+8] for j in range(0,len(group),8)]
 rng.shuffle(batches)
 for i,cs in enumerate(batches):
  cp=R/f'{split}_r{repeat}_b{i}.configs';cp.write_text(''.join(f'{c["key"]} {c["kind"]} {c["L"]} {c["tau"]:.17g} {c["et_tau"]:.17g} 0 0\n' for c in cs))
  run([B/'search',R.parent/'sift1m/index',M/'data'/f'{split}.fbin',M/'data'/f'{split}.gt',M/'models/lr.bin',modelpath(cs[0]['model']),cp,out,repeat],R/f'{split}_r{repeat}_b{i}.log')
  log(f'{split} repeat={repeat} batch={i+1}/{len(batches)} ({len(cs)} configs)')
 rows=[]
 with (M/'data'/f'{split}.gt').open('rb') as f:n,d=np.fromfile(f,'<u4',2);gt=np.fromfile(f,'<u4',count=int(n*d)).reshape(n,d)[:,:10]
 for c in configs:
  prefix=out/f'{c["key"]}_r{repeat}';a=np.genfromtxt(prefix.with_suffix('.csv'),delimiter=',',names=True);m=json.loads(prefix.with_suffix('.meta.json').read_text());ids=np.fromfile(prefix.with_suffix('.ids'),'<u8').reshape(1000,10)
  hits=np.array([len(set(v)&set(g)) for v,g in zip(ids,gt)]);np.testing.assert_array_equal(hits,a['hits'])
  assert m['reader_requests']==int(a['reads'].sum()) and m['reader_bytes']==m['reader_requests']*4096
  assert m['linux_read_bytes']==m['reader_bytes'],(c['key'],m)
  if c['kind'] in [4,7]:np.testing.assert_array_equal(a['expansions'],a['gate_evaluated']-a['skips'])
  rows.append(dict(**c,repeat=repeat,total_hits=int(hits.sum()),recall=float(hits.sum()/100),p95_us=float(np.percentile(a['latency_us'],95)),qps=m['qps'],physical_bytes=m['linux_read_bytes']/1000,reads=float(a['reads'].mean()),skips=float(a['skips'].mean()),et_stop_fraction=float(a['et_stopped'].mean()),gate_calls=float(a['model_calls'].mean()),et_calls=float(a['et_evaluated'].mean()),expansions=float(a['expansions'].mean()),ids_sha256=sha(prefix.with_suffix('.ids'))))
 return rows
def select(p,rows):
 chosen=[]
 for fam in sorted(set(c['family'] for c in rows)):
  for target in p['targets']:
   feasible=[r for r in rows if r['family']==fam and r['recall']>=target]
   for metric in p['objectives']:
    if feasible:
     r=min(feasible,key=lambda r:(-r[metric] if metric=='qps' else r[metric],-r['total_hits'],r['key']))
     chosen.append(dict(family=fam,target=target,objective=metric,key=r['key'],validation_recall=r['recall']))
 dump(R/'SELECTION_LOCK.json',chosen);return chosen
def test(p,cs,chosen):
 by={c['key']:c for c in cs};configs={s['key']:by[s['key']] for s in chosen};ablations=[]
 for key,c in list(configs.items()):
  if c['family']!='Joint_adapted':continue
  for suffix,kind,tau,et in [('no_gate',6,0,c['et_tau']),('no_et',4,c['tau'],1.1)]:
   a=dict(c,key=key+'_'+suffix,family='Ablation_'+suffix,kind=kind,tau=tau,et_tau=et);configs[a['key']]=a;ablations.append(dict(parent=key,control=a['key'],disabled=suffix))
 dump(R/'ablation_pairs.json',ablations);dump(R/'test_configs.json',list(configs.values()))
 runs=[]
 for rep in range(3):runs+=evaluate('test',list(configs.values()),rep);dump(R/'all_test_runs.json',runs)
 rows=[]
 for key,c in configs.items():
  rs=[r for r in runs if r['key']==key]
  for field in ['total_hits','reads','physical_bytes','skips','et_stop_fraction','gate_calls','et_calls','expansions','ids_sha256']:assert len({r[field] for r in rs})==1,(key,field)
  r=rs[0].copy();r.pop('repeat')
  for field in ['p95_us','qps']:r[field]=float(np.median([s[field] for s in rs]));r[field+'_min']=min(s[field] for s in rs);r[field+'_max']=max(s[field] for s in rs)
  rows.append(r)
 dump(R/'test_summary.json',rows);return rows
def report(p,chosen,rows):
 by={r['key']:r for r in rows};table=[]
 for s in chosen:table.append(s|{k:by[s['key']][k] for k in ['recall','p95_us','qps','physical_bytes','skips','et_stop_fraction','L','tau','et_tau']}|{'test_feasible':by[s['key']]['recall']>=s['target']})
 csvout(R/'validation_locked_test.csv',table);csvout(R/'test_summary.csv',rows)
 text=['# LR Gate + ET：剪枝轨迹适配实验','', '验证集锁定配置后测试，每配置独立运行3次，P95/QPS取中位数。Recall和I/O跨重复完全一致。原始结果保留。','', '## 99.5% 验证质量目标：最低读取量配置','', '|方法|验证Recall|测试Recall|测试达标|P95 ms|QPS|KiB/Query|跳过节点/Query|ET停止率|','|---|---:|---:|---|---:|---:|---:|---:|---:|']
 for r in table:
  if r['target']==99.5 and r['objective']=='physical_bytes':text.append(f'|{r["family"]}|{r["validation_recall"]:.2f}%|{r["recall"]:.2f}%|{r["test_feasible"]}|{r["p95_us"]/1000:.3f}|{r["qps"]:.1f}|{r["physical_bytes"]/1024:.3f}|{r["skips"]:.3f}|{r["et_stop_fraction"]:.1%}|')
 text+=['','## 99.7% 验证质量目标：最低读取量配置','','同一质量下限，各行实际Recall不同；不是严格等Recall比较。','','|方法|验证Recall|测试Recall|测试达标|P95 ms|QPS|KiB/Query|跳过节点/Query|ET停止率|','|---|---:|---:|---|---:|---:|---:|---:|---:|']
 for r in table:
  if r['target']==99.7 and r['objective']=='physical_bytes':text.append(f'|{r["family"]}|{r["validation_recall"]:.2f}%|{r["recall"]:.2f}%|{r["test_feasible"]}|{r["p95_us"]/1000:.3f}|{r["qps"]:.1f}|{r["physical_bytes"]/1024:.3f}|{r["skips"]:.3f}|{r["et_stop_fraction"]:.1%}|')
 lookup={(r['family'],r['target'],r['objective']):r for r in table}
 joint=lookup['Joint_adapted',99.7,'physical_bytes'];gate=lookup['Gate',99.7,'physical_bytes'];old=lookup['Joint_original',99.7,'physical_bytes']
 text+=['','## 结果判读','',f'- 在99.7%质量要求下，重训联合组实际Recall {joint["recall"]:.2f}%，Gate为{gate["recall"]:.2f}%；联合组读取量减少{100*(1-joint["physical_bytes"]/gate["physical_bytes"]):.2f}%，QPS提升{100*(joint["qps"]/gate["qps"]-1):.2f}%，P95增加{100*(joint["p95_us"]/gate["p95_us"]-1):.2f}%。有有限的I/O与吞吐收益，未显示尾延迟收益。',f'- 原ET联合组与重训联合组在该点实际Recall均为{joint["recall"]:.2f}%，重训组读取量反而增加{100*(joint["physical_bytes"]/old["physical_bytes"]-1):.2f}%。原ET联合组也实际触发了两种策略，因此本次不支持“必须重训才能互补”或“重训明显优于原ET联合组”。','- 99.5%验证目标的最低读取量配置中，三个联合组测试Recall均未达到99.5%，不能把其读取节省当作满足99.5%质量要求的收益。','- 原P0-B只扫8对预设阈值，本次穷举联合网格后原ET也有两动作同时生效的配置；阈值搜索预算是历史结果差异的一项因素。','- 结论限定于冻结LR Gate、现有17维绝对Recall预测ET和本次有限网格。未训练独立的“继续搜索边际收益”模型。']
 text+=['','## 重训联合组的固定参数关闭实验','','下表保持L和剩余策略阈值不变。关闭策略可能提高Recall，因此这是机制归因，最终同质量比较仍需看各方法独立验证选参结果。','','|联合配置|关闭策略|Recall变化 pp|读取量变化|P95变化|QPS变化|','|---|---|---:|---:|---:|---:|']
 effects=[]
 for pair in json.loads((R/'ablation_pairs.json').read_text()):
  parent=by[pair['parent']];control=by[pair['control']];e=pair|dict(recall_delta=control['recall']-parent['recall'],read_delta_pct=100*(control['physical_bytes']/parent['physical_bytes']-1),p95_delta_pct=100*(control['p95_us']/parent['p95_us']-1),qps_delta_pct=100*(control['qps']/parent['qps']-1));effects.append(e)
  text.append(f'|{pair["parent"]}|{pair["disabled"]}|{e["recall_delta"]:+.2f}|{e["read_delta_pct"]:+.2f}%|{e["p95_delta_pct"]:+.2f}%|{e["qps_delta_pct"]:+.2f}%|')
 csvout(R/'ablation_effects.csv',effects)
 text+=['','## 方法与限制','','- Gate为已有冻结LR；ET_original为已有未剪枝轨迹模型；ET_unpruned为本次同训练流程重新训练的未剪枝对照；Joint_unpruned控制训练流程变化；Joint_adapted使用混合Gate阈值的剪枝轨迹训练。','- 未剪枝重训ET使用139,518个边界样本；适配ET使用557,450个边界样本，包含零阈值轨迹。两者训练查询相同，但训练行数不相等，不能将差异纯归因于轨迹分布。','- ET预测当前绝对GT Recall，不把“达到剪枝后最终结果”替代为达到质量目标。','- 训练/内部holdout/验证/测试按查询隔离。用于推理的17维状态不含GT、未来收益或Gate阈值。','- 联合组穷举5个L×4个Gate阈值×8个ET阈值；Tuned DiskANN扫描64个L。各方法预算不同，结论受有限网格限制。','- 同时触发不等于互补；需结合验证锁定后的测试质量、成本和关闭实验判断。P95/QPS重复范围见test_summary.csv。','- 当前查询曾在历史实验中观察，属于探索性实验；I/O为WSL2客户机块层实测读取。','- 本轮独立计时，性能变化只在本轮内比较，不与历史P0-B的绝对延迟直接归因。','- 所有目标及三个成本指标见validation_locked_test.csv；训练与导出核验见models/。']
 (R/'REPORT.md').write_text('\n'.join(text)+'\n')
 lock=json.loads((R/'INPUT_LOCK.json').read_text());assert all(sha(k)==v for k,v in lock.items())
 dump(R/'COMPLETE.json',dict(complete=True,validation_configs=len(json.loads((R/'configs.json').read_text())),test_configs=len(rows),test_runs=3*len(rows),original_inputs_unchanged=True,report='REPORT.md'))
 log('EXPERIMENT COMPLETE')
if __name__=='__main__':
 p=prepare();collect(p);train(p);cs=configurations(p)
 if (R/'validation.json').exists():v=json.loads((R/'validation.json').read_text())
 else:v=evaluate('validation',cs,0);dump(R/'validation.json',v)
 chosen=json.loads((R/'SELECTION_LOCK.json').read_text()) if (R/'SELECTION_LOCK.json').exists() else select(p,v)
 rows=test(p,cs,chosen);report(p,chosen,rows)
