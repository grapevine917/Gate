"""Descriptive, empirical exact-recall matching; no claims of held-out tuning."""
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
PROTOCOL=dict(version=1,datasets=list(CATALOG),families=FAMILIES,weights='unchanged old matched mixed-L proxy models',selection='Descriptive: tune on already-observed test-query Recall, freeze, independently retime SAME queries. NOT held-out generalization.',targets='native L100 and native L160 for 1M / L500 for SIFT100M',exact='equal integer total_hits / 10000; no interpolation, no mixed policies, no output truncation',search='existing grid first; then two larger L: ceil(1.5*referenceL), 2*referenceL; 12 bounded scalar bisection evaluations per L, plus up to 8 local probes per family/target; nonmonotonicity possible; no guarantee exact feasible',selection_objective='exact and active preferred, then absolute hit gap, then measured P95; inactive exact shown separately',active='Heuristic/Ours: skips>0; ET: stopped fraction>0; Joint: skips>0 AND stopped fraction>0',repeats=9,metrics='Recall,P50,P95,P99,mean,QPS,physical_bytes,reads,expansions,skips,model_calls,ET_calls,IO; median nine cyclic-order timing repeats',controls='same binary/index/model/query/K10/beam4/threads4/cache10000/warm128 as original; Linux/Reader/4096 equality and zero swap checks inherited',budget='at most32 NEW probes per family/target; 2 targets x8 families x3 datasets; old grid reused read-only')

def active(r):
 f=r['family']
 if f=='DiskANN':return True
 if f=='ET':return r['et_stop_fraction']>0
 if f.startswith('Joint'):return r['skips']>0 and r['et_stop_fraction']>0
 return r['skips']>0

def main():
 if (BASE/'COMPLETE.json').exists():return
 if (BASE/'PROTOCOL.json').exists():assert json.loads((BASE/'PROTOCOL.json').read_text())==PROTOCOL
 else:dump(BASE/'PROTOCOL.json',PROTOCOL)
 assert snapshot()==json.loads((OLD/'INDEX_LOCK.json').read_text())
 results=[]
 for ds in CATALOG:
  E=Experiment(ds);E.audit_sources();oldO=E.O;O=BASE/ds;O.mkdir(exist_ok=True)
  E.O=O
  for p in [O/'logs',O/'runs/test',O/'runs/validation']:p.mkdir(parents=True,exist_ok=True)
  if not (O/'native').exists():(O/'native').symlink_to(oldO/'native',target_is_directory=True)
  history=json.loads((oldO/'summary.json').read_text())
  dump(O/'SOURCE_LOCK.json',dict(original_protocol_sha256=sha(oldO/'PROTOCOL.json'),original_summary_sha256=sha(oldO/'summary.json'),source_hashes=E.P['source_hashes']))
  selected=[]
  for refL in [100,500 if ds=='sift100m' else 160]:
   ref=next(r for r in history if r['family']=='DiskANN' and r['L']==refL);target=ref['total_hits'];rows=[]
   for family in FAMILIES:
    pool=[r for r in history if r['family']==family];template=E.bykey[pool[0]['key']].copy();tested=[]
    def evaluate(L,x,exp=None):
     c=template|dict(L=L,rank=L,exp=-1,tau=0.,et_tau=1.1)
     if family=='Heuristic':c.update(rank=int(round(x)),exp=refL//2 if exp is None else exp)
     elif family=='ET':c['et_tau']=x
     else:
      c['tau']=x
      if family.startswith('Joint'):c['et_tau']=.999
     signature={k:c[k] for k in ['family','kind','model','L','tau','et_tau','rank','exp']}
     c['key']='Match_'+hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:18]
     r=E.load('test',[c],0)[0];tested.append(r);return r
    exact=[r for r in pool if r['total_hits']==target and active(r)]
    if family!='DiskANN' and not exact:
     for L in [math.ceil(refL*1.5),refL*2]:
      lo,hi=(0.,float(L)) if family=='Heuristic' else (0.,1.)
      seen=set()
      for _ in range(12):
       x=(lo+hi)/2
       if family=='Heuristic':x=float(round(x))
       if x in seen:break
       seen.add(x);r=evaluate(L,x)
       if r['total_hits']==target and active(r):break
       # higher rank or ET score keeps more; higher gate threshold keeps fewer.
       increasing=family in ['Heuristic','ET']
       if (r['total_hits']<target)==increasing:lo=x
       else:hi=x
      if any(r['total_hits']==target and active(r) for r in tested):break
     if not any(r['total_hits']==target and active(r) for r in tested):
      near=sorted(tested,key=lambda r:abs(r['total_hits']-target))[:2]
      for r in near:
       center=r['rank'] if family=='Heuristic' else r['et_tau'] if family=='ET' else r['tau']
       for delta in [-2,-1,1,2]:
        x=center+delta*(1 if family=='Heuristic' else .0005)
        if 0<=x<=(r['L'] if family=='Heuristic' else 1):evaluate(r['L'],x)
        if any(q['total_hits']==target and active(q) for q in tested):break
       if any(q['total_hits']==target and active(q) for q in tested):break
    pool+=tested
    if family=='DiskANN':best=ref
    else:
     meaningful=[r for r in pool if active(r)]
     best=min(meaningful or pool,key=lambda r:(abs(r['total_hits']-target),r['p95_ms'],r['key']))
    inactive=[r for r in pool if r['total_hits']==target and not active(r)]
    row=dict(reference_L=refL,target_hits=target,target_recall=target/100,family=family,selected=best,strict_exact=best['total_hits']==target,active=active(best),inactive_exact_keys=[r['key'] for r in inactive],new_probe_count=len(tested))
    rows.append(row);dump(O/f'progress_L{refL}.json',rows)
   selected+=rows
  lock=O/'SELECTION_LOCK.json'
  if lock.exists():assert json.loads(lock.read_text())==selected
  else:dump(lock,selected)
  configs={}
  for row in selected:
   r=row['selected'];c={k:r[k] for k in ['key','family','kind','model','L','tau','et_tau','rank','exp','preset']};configs[c['key']]=c
  cs=list(configs.values());runs=[]
  for rep in range(1,10):
   shift=(rep-1)%len(cs);runs+=E.load('test',cs[shift:]+cs[:shift],rep,True)
  summaries={r['key']:r for r in aggregate(runs,cs)}
  for row in selected:
   r=summaries[row['selected']['key']];assert r['total_hits']==row['selected']['total_hits']
   results.append({k:v for k,v in row.items() if k not in ['selected','inactive_exact_keys']}|dict(dataset=ds,inactive_exact_count=len(row['inactive_exact_keys']))|r)
  E.audit_sources();save(O/'results.csv',[r for r in results if r['dataset']==ds]);dump(O/'COMPLETE.json',dict(complete=True,configs=len(cs),repeats=9))
 assert snapshot()==json.loads((OLD/'INDEX_LOCK.json').read_text())
 save(BASE/'results.csv',results)
 lines=['# 旧模型严格 Recall 匹配补充实验','',PROTOCOL['selection'],'','同Recall仅指当前1000条查询总命中数一致，不表示逐查询命中相同。参数搜索用到了该查询集，属于描述性分析。未匹配项保留命中差；只关闭剪枝的匹配单独计数，不冒充有效剪枝结果。','']
 for ds in CATALOG:
  for refL in [100,500 if ds=='sift100m' else 160]:
   subset=[r for r in results if r['dataset']==ds and r['reference_L']==refL]
   lines += [f'## {ds} / 原生L={refL} / 目标Recall={subset[0]["target_recall"]:.2f}%','', '| 方法 | L | Recall% | 命中差 | 严格匹配 | 有效剪枝/终止 | P50ms | P95ms | P99ms | QPS | 字节/Query |','|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|']
   for r in subset:lines.append(f'| {r["family"]} | {r["L"]} | {r["recall"]:.2f} | {r["total_hits"]-r["target_hits"]:+d} | {r["strict_exact"]} | {r["active"]} | {r["p50_ms"]:.3f} | {r["p95_ms"]:.3f} | {r["p99_ms"]:.3f} | {r["qps"]:.1f} | {r["physical_bytes"]:.1f} |')
   lines.append('')
 (BASE/'REPORT.md').write_text('\n'.join(lines))
 dump(BASE/'COMPLETE.json',dict(complete=True,rows=len(results),strict_active_rows=sum(r['strict_exact'] and r['active'] for r in results),report_sha256=sha(BASE/'REPORT.md')))
 dump(BASE/'STATUS.json',dict(complete=True,phase='complete'))
if __name__=='__main__':
 import fcntl
 lock=(BASE/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 except Exception as error:
  dump(BASE/'FAILED.json',dict(error=repr(error),at=time.strftime('%Y-%m-%d %H:%M:%S')))
  raise
