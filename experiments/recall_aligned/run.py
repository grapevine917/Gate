from pathlib import Path
import os,json,csv,hashlib,struct,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parent;LG=ROOT.parents[1];SOURCE=ROOT.parent/'io_predictability';MEMO=ROOT.parent/'query_skip_cache';BIN=LG/'DiskANN/build/apps/search_policy_comparison';BUILDER=LG/'DiskANN/build/apps/search_query_skip_cache'
TAUS=[0,.005,.01,.02,.04,.06,.08,.1,.15,.2,.3,.5,.8,1.];FRACTIONS=[0,.05,.1,.2,.4,.6,.8,1.];BUDGETS=[.5,1.,2.]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x):return float(np.float32(x))
def token(x):return struct.pack('<f',f32(x)).hex()
def config(key,family,parameter=0,**kw):
 d=dict(key=key,family=family,parameter=float(parameter),kind=0,model='-',tau=0,second='-',tau2=0,mark='-',rank=-1,progress=-1,bank='-',rho=0,guard=0);d.update(kw);return d

def marks(ds,fraction):
 fraction=float(fraction);folder=ROOT/ds/'marks';folder.mkdir(parents=True,exist_ok=True);p=folder/f'{float(fraction).hex()}.bin'
 if p.exists():return p
 raw=(ROOT.parent/'node_mark_comparison'/ds/'Mark_m1_r1.bin').read_bytes();assert raw[:8]==b'NDMARK01';bits=np.unpackbits(np.frombuffer(raw[12:],np.uint8),bitorder='little').astype(bool)
 x=np.arange(1000000,dtype=np.uint64)+np.uint64(20261017)
 with np.errstate(over='ignore'):
  x=x+np.uint64(0x9e3779b97f4a7c15);x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);x=x^(x>>np.uint64(31))
 uniform=(x>>np.uint64(11)).astype(float)/float(2**53);mask=bits&(uniform<fraction)
 p.write_bytes(b'NDMARK01'+struct.pack('<I',1000000)+np.packbits(mask,bitorder='little').tobytes())
 if fraction==1:assert p.read_bytes()==raw
 if fraction==0:assert not mask.any()
 return p

def bank(ds,tau):
 tau=f32(tau);folder=ROOT/ds/'banks'/token(tau);folder.mkdir(parents=True,exist_ok=True);p=folder/'bank.bin';model=SOURCE/ds/'state_lr/model.bin'
 signature={'tau':tau,'model_sha256':sha(model),'queries_sha256':sha(SOURCE/ds/'train/query.fbin'),'builder_sha256':sha(BUILDER)}
 if (folder/'DONE.json').exists():assert json.loads((folder/'DONE.json').read_text())==signature;return p
 cfg=folder/'build.config';cfg.write_text('fresh 0 0 0\n');out=folder/'build'
 args=[str(BUILDER),str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(SOURCE/ds/'train/query.fbin'),str(model),str(tau),str(MEMO/ds/'repeat/query.fbin'),str(MEMO/ds/'repeat/gt.bin'),str(cfg),str(out),str(p)]
 print(ds,'build bank tau',tau,flush=True)
 dump(folder/'command.json',{'args':args,**signature})
 with open(folder/'native.log','w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
 dump(folder/'DONE.json',signature);return p

def lr_configs(ds,tau):
 tau=f32(tau);model=str(SOURCE/ds/'state_lr/model.bin');p=bank(ds,tau)
 return [config('LR_'+token(tau),'LR',tau,kind=1,model=model,tau=tau),config('Reuse_'+token(tau),'Reuse',tau,kind=5,model=model,tau=tau,bank=str(p),rho=.01)]
def mark_config(ds,f):return config('Mark_'+float(f).hex().replace('+','p'),'Mark',f,kind=3,mark=str(marks(ds,f)))

def execute(ds,workload,split,cs,reps,stage):
 folder=ROOT/ds/workload/split/stage;folder.mkdir(parents=True,exist_ok=True);prefix=folder/'run';cfg=folder/'configs.txt';rng=np.random.default_rng(20261017);lines=[]
 for rep in range(reps):
  for ix in rng.permutation(len(cs)):
   c=cs[ix];lines.append(' '.join(str(c[k]) for k in ['key','kind','model','tau','second','tau2','mark','rank','progress','bank','rho','guard'])+' '+str(rep))
 cfg.write_text('\n'.join(lines)+'\n');inputs=SOURCE/ds/split if workload=='natural' else MEMO/ds/('near_'+split)
 args=[str(BIN),str(LG/'experiments'/('sift1m' if ds=='sift' else 'deep1m')/'index'),str(SOURCE/ds/'train/query.fbin'),str(inputs/'query.fbin'),str(inputs/'gt.bin'),str(cfg),str(prefix)]
 sig={'args':args,'binary_sha256':sha(BIN),'config_sha256':sha(cfg),'query_sha256':sha(inputs/'query.fbin'),'gt_sha256':sha(inputs/'gt.bin')}
 done=folder/'DONE.json'
 if done.exists():assert json.loads(done.read_text())==sig
 else:
  dump(folder/'command.json',sig);print(ds,workload,split,stage,len(cs),'configs START',flush=True)
  with open(folder/'native.log','w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  dump(done,sig);print(ds,workload,split,stage,'DONE',flush=True)
 rows=list(csv.DictReader(open(str(prefix)+'.csv')));nq=1000 if workload=='natural' else 200;assert len(rows)==nq*len(cs)*reps
 qps={}
 for line in (folder/'native.log').read_text().splitlines():
  if line.startswith('POLICY '):a=line.split();qps.setdefault(a[1],[]).append(float(a[-1]))
 results={}
 for c in cs:
  rr=[];first=None
  for rep in range(reps):
   rs=sorted([r for r in rows if r['key']==c['key'] and int(r['rep'])==rep],key=lambda r:int(r['query']));assert [int(r['query']) for r in rs]==list(range(nq))
   a={k:np.array([float(r[k]) for r in rs]) for k in ['hits','reads','expansions','calls','reused','cache_hit','total_us','lookup_us']}
   if first is None:first=a
   else:
    for k in ['hits','reads','expansions','calls','reused','cache_hit']:np.testing.assert_array_equal(a[k],first[k])
    for ext in ['ids','dists']:assert sha(f'{prefix}_{c["key"]}_r0.{ext}')==sha(f'{prefix}_{c["key"]}_r{rep}.{ext}')
   rr.append({'recall_pct':float(a['hits'].mean()*10),'reads':float(a['reads'].mean()),'calls':float(a['calls'].mean()),'reused':float(a['reused'].mean()),'latency_us':float(a['total_us'].mean()),'p95_us':float(np.percentile(a['total_us'],95)),'lookup_us':float(a['lookup_us'].mean()),'cache_hit_fraction':float(a['cache_hit'].mean())})
  results[c['key']]={**c,**{k:float(np.median([r[k] for r in rr])) for k in rr[0]},'qps':float(np.median(qps[c['key']])),'latency_min_us':min(r['latency_us'] for r in rr),'latency_max_us':max(r['latency_us'] for r in rr),'prefix':str(prefix)}
 return results

def proposal(points,family,target):
 pts=sorted([r for r in points.values() if r['family']==family],key=lambda r:r['parameter']);matched=[r for r in pts if target-1e-8<=r['recall_pct']<=target+.05000001]
 if matched:return None
 brackets=[(a,b) for a,b in zip(pts,pts[1:]) if (a['recall_pct']-target)*(b['recall_pct']-target)<=0 and a['recall_pct']!=b['recall_pct']]
 if not brackets:return None
 a,b=min(brackets,key=lambda ab:abs(ab[1]['parameter']-ab[0]['parameter']));x=(a['parameter']+b['parameter'])/2
 if family!='Mark':x=f32(x)
 return None if any(abs(x-r['parameter'])<1e-12 for r in pts) else x

def select(points,family,target):
 pts=[r for r in points.values() if r['family']==family];matched=[r for r in pts if target-1e-8<=r['recall_pct']<=target+.05000001]
 if matched:r=min(matched,key=lambda r:(r['reads'],-r['recall_pct'],r['key']))
 else:
  above=[r for r in pts if r['recall_pct']>=target-1e-8];r=min(above,key=lambda r:(r['recall_pct']-target,r['reads'])) if above else max(pts,key=lambda r:r['recall_pct'])
 return {'key':r['key'],'target_recall_pct':target,'validation_recall_pct':r['recall_pct'],'matching_gap_pp':r['recall_pct']-target,'within_0.05pp':bool(matched),'parameter':r['parameter']}

def interpolate(points,family,target):
 pts=[r for r in points.values() if r['family']==family];candidates=[]
 for a in pts:
  if abs(a['recall_pct']-target)<1e-8:candidates.append((a,a,1.))
  for b in pts:
   if a['recall_pct']<target<b['recall_pct']:
    weight=(b['recall_pct']-target)/(b['recall_pct']-a['recall_pct']);candidates.append((a,b,weight))
 if not candidates:return None
 a,b,w=min(candidates,key=lambda z:z[2]*z[0]['reads']+(1-z[2])*z[1]['reads'])
 return {'recall_pct':target,'interpolated':a['key']!=b['key'],'endpoint_low':a['key'],'endpoint_high':b['key'],'low_weight':w,**{m:w*a[m]+(1-w)*b[m] for m in ['reads','calls','reused','latency_us']}}

def report(results):
 lines=['# 同Recall对照：全局一次标记 / LR / LR+Reuse','','原模型、历史数据和L100/beam4/K10/cache10000不变。Reuse固定rho<=0.01，每个LR阈值建立独立历史skip缓存。标记f=1是原完整一次标记；f<1为固定节点哈希启用比例，并非改用学习分类器。','','下表为同一测试Recall上的成本曲线比较：成本来自预先锁定实测点之间的线性插值估计，不冒充实测单配置，也不用于测试集重新选上线参数。三轮时间中位数。验证选参后真实测试结果另列。']
 for ds,ws in results.items():
  for work,v in ws.items():
   if 'test' not in v:continue
   base=v['test']['DiskANN'];lines+=['',f'## {ds} / {work}', '',f"原始DiskANN Recall={base['recall_pct']:.3f}%，reads/q={base['reads']:.3f}，平均延迟={base['latency_us']/1000:.4f}ms。完整一次标记Recall={v['full_mark']['recall_pct']:.3f}%。",'', '|期望损失预算pp|共同实际Recall%|方法|请求/q估计|相对DiskANN请求减少|平均延迟ms估计|相对DiskANN延迟降低|插值|','|---:|---:|---|---:|---:|---:|---:|---|']
   for item in v['aligned']:
    for family,a in item['methods'].items():
     if a is None:lines.append(f"|{item['budget_pp']:g}|{item['target_recall_pct']:.3f}|{family}|未覆盖目标|—|—|—|—|");continue
     lines.append(f"|{item['budget_pp']:g}|{item['target_recall_pct']:.3f}|{family}|{a['reads']:.3f}|{1-a['reads']/base['reads']:.2%}|{a['latency_us']/1000:.4f}|{1-a['latency_us']/base['latency_us']:.2%}|{'是' if a['interpolated'] else '否'}|")
   lines+=['','验证集选出的单配置：以下均为真实测试结果，不用插值掩盖测试Recall偏差。','','|预算pp|方法|参数|验证Recall%|测试Recall%|请求/q实测|平均延迟ms实测|','|---:|---|---:|---:|---:|---:|---:|']
   for budget,choices in v['selected'].items():
    for family,c in choices.items():
     a=v['test'][c['key']];lines.append(f"|{budget}|{family}|{c['parameter']:.7g}|{c['validation_recall_pct']:.3f}|{a['recall_pct']:.3f}|{a['reads']:.3f}|{a['latency_us']/1000:.4f}|")
 lines+=['','完整Mark在高相似场景可能达不到2pp损失，主表自动使用共同可达的更高Recall，不能误称为2pp恰好用满。自然查询2pp目标下Mark通常只启用一部分原标记节点；原完整方案另报。插值依据全部锁定实测点的I/O成本下包络，延迟是对应同一组合的均值估计，不独立挑最小时延。p95/QPS不插值。高相似查询父query位于历史缓存中，不能等同自然独立query。']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 started=time.time();results={};dump(ROOT/'RUN_PROTOCOL.json',{'tau_grid':TAUS,'fraction_grid':FRACTIONS,'budgets_pp':BUDGETS,'rho':.01,'refinement_rounds':8,'validation_matching_tolerance_pp':.05,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'binary_sha256':sha(BIN),'script_sha256':sha(Path(__file__))})
 for ds in ['sift','deep']:
  results[ds]={};old=json.loads((SOURCE/ds/'state_lr/thresholds.json').read_text())['unsafe_fpr_0.005']['retain_threshold'];taus=sorted(set(f32(t) for t in TAUS+[old]));common=[config('DiskANN','DiskANN')]+[mark_config(ds,f) for f in FRACTIONS]
  for tau in taus:common.extend(lr_configs(ds,tau))
  for workload in ['natural','near']:
   points=execute(ds,workload,'validation',common,1,'coarse');used={c['key']:c for c in common};base=points['DiskANN'];full=points[mark_config(ds,1)['key']];max_loss=max(0,base['recall_pct']-full['recall_pct']);targets={str(b):base['recall_pct']-min(b,max_loss) for b in BUDGETS}
   for round in range(8):
    pending={}
    for target in targets.values():
     for family in ['Mark','LR','Reuse']:
      x=proposal(points,family,target)
      if x is None:continue
      generated=[mark_config(ds,x)] if family=='Mark' else lr_configs(ds,x)
      for c in generated:
       if c['key'] not in points:pending[c['key']]=c
    if not pending:break
    new=execute(ds,workload,'validation',list(pending.values()),1,f'refine{round}');points.update(new);used.update(pending)
   selected={b:{family:select(points,family,target) for family in ['Mark','LR','Reuse']} for b,target in targets.items()}
   lock=ROOT/ds/workload/'SELECTION_LOCK.json';obj={'selected':selected,'targets':targets,'validation_points':points,'test_used_for_selection':False,'configs':list(used.values())}
   if lock.exists():assert json.loads(lock.read_text())==obj
   else:dump(lock,obj)
   test=execute(ds,workload,'test',list(used.values()),3,'all_locked_points');full=test[mark_config(ds,1)['key']];base=test['DiskANN'];max_loss=max(0,base['recall_pct']-full['recall_pct'])
   aligned=[{'budget_pp':b,'target_recall_pct':base['recall_pct']-min(b,max_loss),'methods':{family:interpolate(test,family,base['recall_pct']-min(b,max_loss)) for family in ['Mark','LR','Reuse']}} for b in BUDGETS]
   anchor={'target_recall_pct':full['recall_pct'],'methods':{family:interpolate(test,family,full['recall_pct']) for family in ['Mark','LR','Reuse']}}
   results[ds][workload]={'validation':points,'selected':selected,'test':test,'full_mark':full,'aligned':aligned,'full_mark_anchor':anchor};dump(ROOT/'results.json',results);report(results)
 dump(ROOT/'COMPLETE.json',{'complete':True,'elapsed_seconds':time.time()-started,'validation_only_deployment_selection':True,'descriptive_interpolation_separate':True,'sha256':{n:sha(ROOT/n) for n in ['results.json','REPORT.md','PROTOCOL.md','RUN_PROTOCOL.json']}})
if __name__=='__main__':main()
