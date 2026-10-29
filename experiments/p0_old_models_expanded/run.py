"""Expanded validation tuning + separately labelled empirical exact-recall study."""
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'p0_old_models_rerun'
sys.path.insert(0,str(OLD))
import measure
from measure import Experiment
from common import *
from index_lock import snapshot
measure.D=BASE
FAMILIES=['DiskANN','Heuristic','Ours_lr','Ours_xgb','Ours_state','ET','Joint_lr','Joint_xgb','Joint_state']
METRICS=['p95_ms','qps','physical_bytes']
TARGETS=[95.,98.,99.,99.5,99.8,99.9]
FIELDS=['key','family','kind','model','L','tau','et_tau','rank','exp','preset']
PROTOCOL=dict(version=1,datasets=list(CATALOG),families=FAMILIES,models='old matched mixed-L proxy LR/XGB/State and ET; no retraining or dynamic thresholds',controls='unchanged graph, queries, weights, K10, beam4, threads4, cache10000, warm128; inherited Reader/Linux IO equality and no swap',primary=dict(budget_per_family='64 original configurations +96 new coarse +32 new validation-local =192 unique configs; same count for every family',L_range_1m=[10,1600],L_range_100m=[10,4000],gate_tau=[0.,1.],ET_tau=[.8,1.],heuristic='rank [0,L], completed-expansions [0,2L]',joint='L, gate tau and ET tau all varied',local='32 perturbations around validation winners of 6 targets x3 independent objectives',selection='validation only; freeze all objective/target selections before new test grid; previously observed benchmark test queries, not a fresh unseen dataset',targets=TARGETS),strict=dict(label='descriptive test-quality calibration; NOT held-out selection',nominal_targets=[98.,99.5,99.8],target_choice='native actual hit count nearest nominal target from primary grid; each within0.15pp or reported unavailable; reference need not L100',budget='up to72 additional probes per non-native family per target: six branches x(2 endpoints+10 bisection steps), no stop at first exact hit; integer duplicate heuristic probes reused',branches='six L factors 1.0,1.2,1.5,2,3,4; heuristic six completed-expansion settings; joint three gate-threshold sweeps at different ET and three ET sweeps at different gate thresholds',exact='equal integer GT total_hits out of10000; no interpolation, no query-dependent policy mixtures, no result truncation',active='Gate/Heuristic skips>0; ET stopped_fraction>0; Joint BOTH; inactive exacts counted separately',objective='select P95/QPS/bytes independently among exact active candidates; no exact =>explicit unmatched nearest record',repeats=9),frontiers='three independent empirical measured frontiers; every final displayed point at least3 repeats; validation-selected and exact-objective winners9 repeats',timing='datasets serial; configs randomized in blocks16; repetitions cyclic ordered; all results from one true config, with full metrics')

def locked(path,value):
 if path.exists():assert json.loads(path.read_text())==value,str(path)
 else:dump(path,value)
 return value

def cfg(row):return {k:row[k] for k in FIELDS}
def active(r):
 if r['family']=='DiskANN':return True
 if r['family']=='ET':return r['et_stop_fraction']>0
 if r['family'].startswith('Joint'):return r['skips']>0 and r['et_stop_fraction']>0
 return r['skips']>0

def cost(r,m):return -r[m] if m=='qps' else r[m]
def signature(c):return tuple(c[k] for k in FIELDS if k not in ['key','preset'])
def normalize(c,maxL):
 c=c.copy();c['L']=max(10,min(maxL,int(round(c['L']))));c['rank']=max(0,min(c['L'],int(round(c['rank']))));c['exp']=max(-1,min(2*c['L'],int(round(c['exp']))));c['tau']=max(0.,min(1.,float(c['tau'])));c['et_tau']=max(.8,min(1.1,float(c['et_tau'])))
 if c['kind']==0:c.update(rank=100,exp=-1,tau=0.,et_tau=1.1)
 c['key']='Expand_'+hashlib.sha256(repr(signature(c)).encode()).hexdigest()[:20]
 return c

def coarse_configs(E):
 maxL=4000 if E.ds=='sift100m' else 1600;out=[]
 for fi,f in enumerate(FAMILIES):
  rng=random.Random(81013+fi);template=next(c for c in E.configs if c['family']==f);seen={signature(c) for c in E.configs if c['family']==f};cs=[];attempt=0
  while len(cs)<96:
   i=attempt;attempt+=1
   # Alternate logarithmic exploration and more densely sampled useful L ranges.
   L=round(math.exp(rng.uniform(math.log(10),math.log(maxL)))) if i%2 else rng.choice([30,50,75,100,125,160,200,250,300,400,500,750,1000])*rng.uniform(.85,1.15)
   c=template|dict(L=L,tau=0.,et_tau=1.1,rank=round(L),exp=-1)
   if f=='Heuristic':c.update(rank=round(L*rng.uniform(.05,1.)),exp=round(L*rng.uniform(0.,2.)))
   if f.startswith(('Ours','Joint')):c['tau']=10**rng.uniform(-4,-.03) if i%3 else rng.uniform(0.,1.)
   if f=='ET' or f.startswith('Joint'):c['et_tau']=1.-10**rng.uniform(-6,math.log10(.2))
   c=normalize(c,maxL)
   if signature(c) not in seen:seen.add(signature(c));cs.append(c)
  out+=cs
 return out

def select_validation(rows):
 selections=[]
 for f in FAMILIES:
  for target in TARGETS:
   feasible=[r for r in rows if r['family']==f and r['total_hits']>=round(target*100)]
   for m in METRICS:
    best=min(feasible,key=lambda r:(cost(r,m),-r['total_hits'],r['key'])) if feasible else None
    selections.append(dict(family=f,target=target,objective=m,key=best['key'] if best else None,validation_recall=best['recall'] if best else None))
 return selections

def local_configs(E,rows,coarse):
 maxL=4000 if E.ds=='sift100m' else 1600;selected=select_validation(rows);out=[]
 for fi,f in enumerate(FAMILIES):
  rng=random.Random(91031+fi);seen={signature(c) for c in E.configs+coarse if c['family']==f};rs={r['key']:r for r in rows};parents=[rs[s['key']] for s in selected if s['family']==f and s['key']];assert parents
  cs=[];i=0
  while len(cs)<32:
   c=cfg(parents[i%len(parents)]);i+=1;c['L']*=rng.uniform(.75,1.3)
   if f=='Heuristic':c['rank']+=rng.randint(-15,15);c['exp']+=rng.randint(-30,30)
   if f.startswith(('Ours','Joint')):c['tau']+=rng.uniform(-.03,.03)
   if f=='ET' or f.startswith('Joint'):c['et_tau']=1.-max(1e-7,(1.-min(1.,c['et_tau'])))*10**rng.uniform(-.6,.6)
   c=normalize(c,maxL)
   if signature(c) not in seen:seen.add(signature(c));cs.append(c)
  out+=cs
 return out

def retime(E,configs,initial,mandatory,frontier_configs=None,repeat_base=1):
 allruns={(r['key'],r['repeat']):r for r in initial};bykey={c['key']:c for c in configs}
 # 9 repeats for locked comparison points, separate from calibration repeat0.
 cs=[bykey[k] for k in sorted(mandatory)]
 for rep in range(repeat_base,repeat_base+9):
  shift=(rep-repeat_base)%max(1,len(cs))
  for r in E.load('test',cs[shift:]+cs[:shift],rep,True):allruns[(r['key'],r['repeat'])]=r
 while True:
  measured=[r for r in allruns.values() if r['key'] not in mandatory or repeat_base<=r['repeat']<repeat_base+9]
  summary=aggregate(measured,configs);S={r['key']:r for r in summary};needed=set()
  if frontier_configs:
   allowed={c['key'] for c in frontier_configs}
   for f in FAMILIES:
    for m in METRICS:needed.update(r['key'] for r in frontier([r for r in summary if r['family']==f and r['key'] in allowed],m))
  todo=[bykey[k] for k in sorted(needed) if S[k]['repeats']<3]
  dump(E.O/'repeat_status.json',dict(frontier_need_repeats=len(todo),unique_runs=len(allruns),mandatory9=len(mandatory)))
  if not todo:break
  for rep in [1,2]:
   for r in E.load('test',todo,rep):allruns[(r['key'],r['repeat'])]=r
 dump(E.O/'all_runs.json',list(allruns.values()))
 return {r['key']:r for r in summary}

def strict_search(E,primary):
 O=E.O;maxL=4000 if E.ds=='sift100m' else 1600;pool={r['key']:r for r in primary};choices=[];references=[]
 for nominal in [98.,99.5,99.8]:
  reference=min([r for r in primary if r['family']=='DiskANN'],key=lambda r:(abs(r['recall']-nominal),r['p95_ms']))
  references.append(dict(nominal=nominal,target_hits=reference['total_hits'],target_recall=reference['recall'],reference_key=reference['key'],available=abs(reference['recall']-nominal)<=.15))
 locked(O/'STRICT_TARGETS.json',references)
 for ref in references:
  if not ref['available']:continue
  target=ref['target_hits'];reference=pool[ref['reference_key']]
  for f in FAMILIES:
   if f!='DiskANN':
    template=cfg(next(r for r in primary if r['family']==f))
    for branch,factor in enumerate([1.,1.2,1.5,2.,3.,4.]):
     L=min(maxL,round(reference['L']*factor));lo,hi=(0.,float(L)) if f=='Heuristic' else (.8,1.) if f=='ET' or (f.startswith('Joint') and branch>=3) else (0.,1.)
     increasing=f in ['Heuristic','ET'] or (f.startswith('Joint') and branch>=3)
     seen=set()
     for step in range(12):
      x=lo if step==0 else hi if step==1 else (lo+hi)/2
      if f=='Heuristic':x=float(round(x))
      if x in seen:break
      seen.add(x);c=template|dict(L=L,tau=0.,et_tau=1.1,rank=L,exp=-1)
      if f=='Heuristic':c.update(rank=round(x),exp=round(reference['L']*[0.,.25,.5,.75,1.,1.5][branch]))
      elif f=='ET':c['et_tau']=x
      elif f.startswith('Joint'):
       if branch<3:c.update(tau=x,et_tau=[.98,.999,.99999][branch])
       else:c.update(tau=[.005,.04,.15][branch-3],et_tau=x)
      else:c['tau']=x
      c=normalize(c,maxL)
      if c['key'] not in pool:pool[c['key']]=E.load('test',[c],0)[0]
      r=pool[c['key']]
      if step>=2:
       if (r['total_hits']<target)==increasing:lo=x
       else:hi=x
     # All branches evaluated even after an exact match: objectives may differ.
   family_rows=[r for r in pool.values() if r['family']==f];valid=[r for r in family_rows if active(r)];exact=[r for r in valid if r['total_hits']==target]
   for m in METRICS:
    best=min(exact,key=lambda r:(cost(r,m),r['key'])) if exact else min(valid or family_rows,key=lambda r:(abs(r['total_hits']-target),cost(r,m),r['key']))
    choices.append(dict(nominal=ref['nominal'],target_hits=target,target_recall=target/100,family=f,objective=m,key=best['key'],strict_exact=best['total_hits']==target,active=active(best),hit_gap=best['total_hits']-target,exact_active_candidates=len(exact),inactive_exact_candidates=sum(r['total_hits']==target and not active(r) for r in family_rows)))
   dump(O/'strict_progress.json',choices)
 locked(O/'STRICT_SELECTION_LOCK.json',choices)
 return pool,choices

def dataset(ds):
 O=BASE/ds;O.mkdir(exist_ok=True)
 if (O/'COMPLETE.json').exists():return
 E=Experiment(ds);E.audit_sources();oldO=E.O;E.O=O
 for p in [O/'logs',O/'runs/test',O/'runs/validation']:p.mkdir(parents=True,exist_ok=True)
 if not (O/'native').exists():(O/'native').symlink_to(oldO/'native',target_is_directory=True)
 locked(O/'SOURCE_LOCK.json',dict(source_hashes=E.P['source_hashes'],old_protocol_sha256=sha(oldO/'PROTOCOL.json'),index_lock_sha256=sha(OLD/'INDEX_LOCK.json')))
 coarse=coarse_configs(E);locked(O/'coarse_configs.json',coarse)
 oldval=json.loads((oldO/'validation.json').read_text());coarseval=E.load('validation',coarse)
 localpath=O/'local_configs.json'
 local=json.loads(localpath.read_text()) if localpath.exists() else local_configs(E,oldval+coarseval,coarse)
 locked(localpath,local);localval=E.load('validation',local)
 configs=E.configs+coarse+local
 assert all(sum(c['family']==f for c in configs)==192 for f in FAMILIES)
 assert len({signature(c) for c in configs})==len(configs)
 validation=oldval+coarseval+localval;save(O/'validation.csv',validation)
 selections=select_validation(validation);locked(O/'VALIDATION_SELECTION_LOCK.json',selections)
 # No primary test evaluation precedes above lock.
 test=E.load('test',configs);dump(O/'primary_initial.json',test)
 primary_mandatory={s['key'] for s in selections if s['key']}
 primary_summary=retime(E,configs,test,primary_mandatory,configs)
 save(O/'primary_configurations.csv',list(primary_summary.values()))
 mainrows=[]
 for s in selections:
  r=primary_summary.get(s['key'])
  mainrows.append(s|dict(available=r is not None,test_meets_target=bool(r and r['total_hits']>=round(s['target']*100)))|({k:v for k,v in r.items() if k not in ['family']} if r else {}))
 save(O/'validation_selected_test.csv',mainrows)
 dump(O/'PRIMARY_COMPLETE.json',dict(complete=True,configurations=len(configs)))
 pool,strict=strict_search(E,test)
 allconfigs=[cfg(r) for r in pool.values()]
 strict_summary=retime(E,allconfigs,list(pool.values()),{s['key'] for s in strict},repeat_base=11)
 strictrows=[s|{k:v for k,v in strict_summary[s['key']].items() if k not in ['family','key']} for s in strict]
 save(O/'strict_results.csv',strictrows)
 save(O/'all_strict_candidates.csv',list(strict_summary.values()))
 E.audit_sources();assert snapshot()==json.loads((OLD/'INDEX_LOCK.json').read_text())
 dump(O/'COMPLETE.json',dict(complete=True,primary_configs=len(configs),strict_rows=len(strictrows),strict_matched=sum(r['strict_exact'] for r in strictrows),finished=time.strftime('%Y-%m-%d %H:%M:%S')))
 print('DATASET COMPLETE',ds,flush=True)

def report():
 os.environ['MPLCONFIGDIR']=str(BASE/'mpl_cache')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figures=BASE/'figures';figures.mkdir(exist_ok=True)
 lines=['# 扩大搜索的旧模型补充实验','','主表：仅用验证集选参，在新测试扫描前冻结。测试查询曾被历史实验观察过，不是全新盲测。严格匹配表：额外使用测试质量调参，属于描述性分析。两者分开报告。','','每方法192个主实验候选（原64+新128）；L/阈值可调，环境/权重固定。P95、QPS和物理字节各自独立选择真实配置。严格匹配与未匹配分开，退化策略不冒充有效剪枝。','']
 for ds in CATALOG:
  rows=list(csv.DictReader((BASE/ds/'strict_results.csv').open()));main=list(csv.DictReader((BASE/ds/'validation_selected_test.csv').open()))
  lines += [f'## {ds}','','完整指标及所有目标见 `'+ds+'/validation_selected_test.csv` 和 `'+ds+'/strict_results.csv`。','']
  for label,rs in [('验证Recall下限99.5%的选参结果',[r for r in main if float(r['target'])==99.5]),('严格匹配结果',rows)]:
   lines += [f'### {label}','','|目标%|方法|优化指标|L|测试Recall%|达标/严格匹配|P95ms|QPS|字节/Query|','|---:|---|---|---:|---:|---|---:|---:|---:|']
   for r in rs:
    if not r.get('recall'):continue
    lines.append(f'| {r.get("target",r.get("target_recall"))} | {r["family"]} | {r["objective"]} | {r["L"]} | {float(r["recall"]):.2f} | {r.get("strict_exact",r.get("test_meets_target"))} | {float(r["p95_ms"]):.3f} | {float(r["qps"]):.1f} | {float(r["physical_bytes"]):.1f} |')
   lines.append('')
 for metric in METRICS:
  fig,axs=plt.subplots(3,3,figsize=(16,12),squeeze=False)
  for i,ds in enumerate(CATALOG):
   rows=list(csv.DictReader((BASE/ds/'primary_configurations.csv').open()))
   for r in rows:
    for k in ['total_hits',metric,'recall']:r[k]=float(r[k])
   for j,model in enumerate(['lr','xgb','state']):
    ax=axs[i,j]
    for f in ['DiskANN','Heuristic','Ours_'+model,'ET','Joint_'+model]:
     pts=frontier([r for r in rows if r['family']==f],metric);ax.plot([r['recall'] for r in pts],[r[metric] for r in pts],'.-',label=f)
    ax.set(xlim=(95,100),title=ds+' / '+model,xlabel='Recall@10 (%)',ylabel=metric);ax.grid(alpha=.2);ax.legend(fontsize=7)
  fig.tight_layout()
  for ext in ['png','svg','pdf']:fig.savefig(figures/f'{metric}.{ext}',dpi=160)
  plt.close(fig)
  lines += [f'![{metric}](figures/{metric}.png)','']
 (BASE/'REPORT.md').write_text('\n'.join(lines))
 dump(BASE/'COMPLETE.json',dict(complete=True,report_sha256=sha(BASE/'REPORT.md'),figures=9,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
 dump(BASE/'STATUS.json',dict(complete=True,phase='complete'))

def main():
 if (BASE/'COMPLETE.json').exists():return
 locked(BASE/'PROTOCOL.json',PROTOCOL)
 assert snapshot()==json.loads((OLD/'INDEX_LOCK.json').read_text())
 for ds in CATALOG:dataset(ds)
 report()

if __name__=='__main__':
 import fcntl
 lock=(BASE/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 except Exception as error:
  dump(BASE/'FAILED.json',dict(error=repr(error),at=time.strftime('%Y-%m-%d %H:%M:%S')))
  raise
