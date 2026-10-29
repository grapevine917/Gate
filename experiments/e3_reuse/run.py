from common import *
import sys,traceback
from static import tapes,write_tapes,static_analysis
def execute(split,c,mode='plan',tape=None,tag=None):
 folder=R/('plans' if mode=='plan' else 'timing');folder.mkdir(exist_ok=True);p=folder/f'{split}_{tag or c["key"]}_{mode}'
 cmd=[R/'native/search',S/'index',E1/f'data/{split}.fbin',E1/f'data/{split}.gt',S/'online_state_mlp/model.bin',json.loads((R/'PROTOCOL.json').read_text())['tau'],R/'history.bin',R/f'{split}_neighbours.txt',c['kind'],c['qcut'],c['scut'],c['votes'],c['oracle'],mode,tape or '-',p]
 signature={'argv':list(map(str,cmd)),'binary':sha(R/'native/search'),'history':sha(R/'history.bin')}
 done=p.with_suffix('.done.json')
 if done.exists():assert json.loads(done.read_text())['signature']==signature,'Changed run signature'
 else:
  log('START '+p.name);run(cmd,p);dump(done,{'signature':signature,'csv_sha256':sha(p.with_suffix('.csv'))});log('DONE '+p.name)
 return p,loadcsv(p.with_suffix('.csv'))
def oracle_b(configs,arrays):
 base=arrays['R0'];names=[c['key'] for c in configs];n=len(base)
 loss=np.column_stack([base['hits']-arrays[k]['hits'] for k in names]).astype(int);gains=np.column_stack([base['actual_calls']-arrays[k]['actual_calls'] for k in names]).astype(int)
 minimum=loss.min(1);weights=loss-minimum[:,None];budget=int(round(.2/100*n*10));capacity=int(budget-minimum.sum())
 dp=np.full(capacity+1,-1e12);dp[0]=0;choice=np.full((n,capacity+1),-1,np.int16)
 for i in range(n):
  nxt=np.full_like(dp,-1e12)
  for j in range(len(names)):
   w=weights[i,j]
   if w>capacity:continue
   candidate=dp[:capacity+1-w]+gains[i,j];view=nxt[w:];take=candidate>view;view[take]=candidate[take];cv=choice[i,w:];cv[take]=j
  dp=nxt
 weight=int(np.argmax(dp));value=int(dp[weight]);selected=np.zeros(n,int)
 for i in range(n-1,-1,-1):j=int(choice[i,weight]);assert j>=0;selected[i]=j;weight-=weights[i,j]
 assert weight==0
 selected_loss=int(loss[np.arange(n),selected].sum());selected_gain=int(gains[np.arange(n),selected].sum());assert selected_loss<=budget and selected_gain==value
 all_tapes={k:tapes(R/'plans'/f'test_{k}_plan.tape') for k in names};all_ids={k:np.fromfile(R/'plans'/f'test_{k}_plan.ids','<u8').reshape(n,10) for k in names}
 write_tapes(R/'plans/test_OracleB_plan.tape',[all_tapes[names[j]][i] for i,j in enumerate(selected)])
 b=np.array([arrays[names[j]][i] for i,j in enumerate(selected)],dtype=base.dtype);ids=np.array([all_ids[names[j]][i] for i,j in enumerate(selected)],dtype='<u8');ids.tofile(R/'plans/test_OracleB_plan.ids')
 csvsave(R/'plans/test_OracleB_plan.csv',[{k:float(row[k]) for k in b.dtype.names} for row in b]);csvsave(R/'OracleB_choices.csv',[{'query_row':i,'trajectory':names[j],'lost_hits':int(loss[i,j]),'saved_calls':int(gains[i,j])} for i,j in enumerate(selected)])
 dump(R/'OracleB_certificate.json',{'scope':'exact optimum over fixed replayed trajectory menu; not unrestricted per-decision oracle','menu':names,'queries':n,'budget_lost_hits':budget,'actual_lost_hits':selected_loss,'saved_calls':selected_gain,'minimum_possible_loss':int(minimum.sum()),'knapsack_capacity':capacity,'test_GT_used':True,'deployable':False})
 return b
def main():
 log('E3 START/RESUME')
 if not (R/'PREPARED.json').exists():subprocess.run([sys.executable,str(R/'prepare.py')],check=True)
 if not (R/'native/search').exists():subprocess.run([sys.executable,str(R/'build_native.py')],check=True)
 configs=json.loads((R/'configs.json').read_text());bykey={c['key']:c for c in configs};valid={}
 for c in configs:_,valid[c['key']]=execute('validation',c)
 validation_static=static_analysis('validation',configs)
 valrows=[{'key':c['key'],**summary(valid[c['key']],valid['R0'])} for c in configs];dump(R/'validation_results.json',valrows);csvsave(R/'validation_results.csv',valrows)
 # Planner physically scores shadows; logical calls are confirmed later by a true bypass tape pass.
 selected={'R0':bykey['R0'],'R1':bykey['R1'],'R2':bykey['R2']}
 for family in ['R3','R4','R5']:
  candidates=[r for r in valrows if r['key'].startswith(family+'_') and r['delta_recall_pp']>=-.2-1e-9]
  if candidates:
   winner=max(candidates,key=lambda r:(r['model_saving'],-(r['false_skip_rate'] or 0),-bykey[r['key']]['scut'],-bykey[r['key']]['qcut']));selected[family]=bykey[winner['key']]
  else:selected[family]=dict(bykey['R0'],key=family+'_fallback')
 selected['OracleA']=bykey['OracleA']
 if (R/'SELECTION_LOCK.json').exists():assert json.loads((R/'SELECTION_LOCK.json').read_text())['selected']==selected,'Frozen selection changed'
 else:dump(R/'SELECTION_LOCK.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected':selected,'test_used':False,'recall_budget_pp':.2,'protocol_sha256':sha(R/'PROTOCOL.json')})
 log('LOCKED '+json.dumps({k:v['key'] for k,v in selected.items()}))
 test={}
 for c in configs:_,test[c['key']]=execute('test',c)
 assert sha(R/'plans/test_R0_plan.ids')==sha(E1/'runs_main/test_MLP_r0_time.ids')
 assert np.all(test['OracleA']['hits']==test['R0']['hits'])
 assert sha(R/'plans/test_OracleA_plan.ids')==sha(R/'plans/test_R0_plan.ids')
 test_static=static_analysis('test',configs)
 testrows=[{'key':c['key'],**summary(test[c['key']],test['R0'])} for c in configs];dump(R/'test_all_results.json',testrows);csvsave(R/'test_all_results.csv',testrows)
 test['OracleB']=oracle_b(configs,test);selected['OracleB']=dict(bykey['R0'],key='OracleB')
 # The B schedule is a quality-oracle-only test result, never a deployable retuning.
 timing={};rng=np.random.default_rng(SEED)
 for rep in range(3):
  for family in rng.permutation(list(selected)):
   c=selected[family];key=c['key'];tape=R/'plans'/f'test_{key}_plan.tape'
   if key.endswith('_fallback'):
    import shutil
    for ext in ['tape','ids','csv']:shutil.copy(R/f'plans/test_R0_plan.{ext}',R/f'plans/test_{key}_plan.{ext}')
   p,a=execute('test',c,'replay',tape,f'{family}_r{rep}')
   assert sha(p.with_suffix('.ids'))==sha(R/'plans'/f'test_{key}_plan.ids')
   plan=test.get(key,test['R0']);
   for field in ['hits','reads','expansions','eligible','reused','actual_calls','false_skip','fresh_keep']:np.testing.assert_array_equal(a[field],plan[field])
   np.testing.assert_array_equal(a['physical_calls'],a['actual_calls'])
   timing.setdefault(family,[]).append(summary(a,test['R0']))
 # Five cost regions are a distinct profiling run, not a latency claim.
 profile={}
 for family in ['R0','R4','OracleA','OracleB']:
  c=selected[family];p,a=execute('test',c,'profile',R/'plans'/f'test_{c["key"]}_plan.tape',family)
  assert sha(p.with_suffix('.ids'))==sha(R/'plans'/f'test_{c["key"]}_plan.ids')
  profile[family]={k:float(a[k].mean()) for k in ['distance_us','model_us','memory_us','ssd_us','bookkeeping_us']}
 # Retain the OLD contribution mark as an extra, separately labelled baseline.
 legacy=[]
 for rep in range(3):
  p=R/'timing'/f'legacy_global_r{rep}'
  if not p.with_suffix('.csv').exists():
   log('LEGACY GLOBAL MARK repeat '+str(rep));run([E1/'native/collect',S/'index',E1/'data/test.fbin',E1/'data/test.gt','-',0,E1/'data/global_once.bin',p,'time'],p)
  a=loadcsv(p.with_suffix('.csv'));legacy.append({'recall_pct':float(a['hits'].mean()*10),'reads':float(a['reads'].mean()),'latency_us':float(a['total_us'].mean()),'p95_us':float(np.percentile(a['total_us'],95))})
 dump(R/'results.json',{'timing':timing,'profile':profile,'legacy_global':legacy,'selected':selected})
 dump(R/'verification.json',{'R0_equals_e1':True,'OracleA_exact_ids_equal_R0':True,'all_tapes_exact_node_state_score_checks':True,'plan_tape_ids_reads_expansions_calls_parity':True,'physical_model_calls_equal_reported_actual':True,'OracleB_budget_passed':summary(test['OracleB'],test['R0'])['delta_recall_pp']>=-.2-1e-9,'legacy_not_renamed_to_R1':True})
 log('ALL SEARCHES COMPLETE; final analysis')
 subprocess.run([sys.executable,str(R/'analyze.py')],check=True)
 files=[p for p in R.rglob('*') if p.is_file() and p.suffix in ['.py','.cpp','.h','.csv','.json','.npz','.png','.svg','.pdf','.md','.bin','.tape','.ids'] and p.name not in ['artifact_manifest.json','COMPLETE.json']]
 dump(R/'artifact_manifest.json',{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)})
 dump(R/'COMPLETE.json',{'complete':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':{p.name:sha(p) for p in [R/'results.json',R/'REPORT.md',R/'verification.json',R/'artifact_manifest.json']}});log('E3 COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:log('FAILED\n'+traceback.format_exc());raise
