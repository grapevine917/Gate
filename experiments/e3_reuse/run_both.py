from common import *
from static import tapes,write_tapes
import traceback
def execute(split,c,mode='plan',tag=None,tape=None):
 folder=R/('both_plans' if mode=='plan' else 'both_timing');folder.mkdir(exist_ok=True);p=folder/f'{split}_{tag or c["key"]}_{mode}'
 cmd=[R/'both_native/search',S/'index',E1/f'data/{split}.fbin',E1/f'data/{split}.gt',S/'online_state_mlp/model.bin',json.loads((R/'PROTOCOL.json').read_text())['tau'],R/'history.bin',R/f'{split}_neighbours.txt',c['kind'],c['qcut'],c['scut'],c['votes'],c['oracle'],mode,tape or '-',p]
 signature={'argv':list(map(str,cmd)),'binary':sha(R/'both_native/search')};done=p.with_suffix('.done.json')
 if done.exists():assert json.loads(done.read_text())['signature']==signature
 else:log('START '+p.name);run(cmd,p);dump(done,{'signature':signature});log('DONE '+p.name)
 return p,loadcsv(p.with_suffix('.csv'))
def main():
 log('BOTH-DECISION EXTENSION START')
 configs=json.loads((R/'both_configs.json').read_text());valid={}
 for c in configs:_,valid[c['key']]=execute('validation',c)
 selected={'D0':configs[0]};rows=[{'key':c['key'],**summary(valid[c['key']],valid['D0'])} for c in configs];dump(R/'both_validation_results.json',rows)
 for family in ['D3','D4']:
  good=[r for r in rows if r['key'].startswith(family+'_') and r['delta_recall_pp']>=-.2-1e-9]
  winner=max(good,key=lambda r:(r['model_saving'],-(r['false_skip_rate'] or 0))) if good else next(r for r in rows if r['key']=='D0');selected[family]=next(c for c in configs if c['key']==winner['key'])
 for key in ['D5','OracleA_both']:selected[key]=next(c for c in configs if c['key']==key)
 dump(R/'BOTH_SELECTION_LOCK.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected':selected,'test_used':False});log('BOTH LOCKED '+json.dumps({k:v['key'] for k,v in selected.items()}))
 test={}
 for c in {c['key']:c for c in selected.values()}.values():_,test[c['key']]=execute('test',c)
 assert sha(R/'both_plans/test_D0_plan.ids')==sha(E1/'runs_main/test_MLP_r0_time.ids')
 assert sha(R/'both_plans/test_OracleA_both_plan.ids')==sha(R/'both_plans/test_D0_plan.ids')
 # Exact menu-limited quality oracle, now allowing both decisions as well as Skip-only trajectories.
 menu=[(c['key'],R/'plans'/f'test_{c["key"]}_plan') for c in json.loads((R/'configs.json').read_text())]+[(k,R/'both_plans'/f'test_{k}_plan') for k in test]
 arrays=[loadcsv(p.with_suffix('.csv')) for _,p in menu];base=test['D0'];n=len(base);loss=np.column_stack([base['hits']-a['hits'] for a in arrays]).astype(int);gain=np.column_stack([base['actual_calls']-a['actual_calls'] for a in arrays]).astype(int);minimum=loss.min(1);weights=loss-minimum[:,None];capacity=int(20-minimum.sum());dp=np.full(capacity+1,-1e12);dp[0]=0;choices=np.full((n,capacity+1),-1,np.int16)
 for i in range(n):
  nxt=np.full_like(dp,-1e12)
  for j in range(len(menu)):
   w=weights[i,j]
   if w>capacity:continue
   v=dp[:capacity+1-w]+gain[i,j];mask=v>nxt[w:];nxt[w:][mask]=v[mask];choices[i,w:][mask]=j
  dp=nxt
 w=int(np.argmax(dp));optimum=int(dp[w]);schedule=[]
 for i in range(n-1,-1,-1):j=int(choices[i,w]);assert j>=0;schedule.append(j);w-=weights[i,j]
 schedule=np.array(schedule[::-1]);assert w==0 and gain[np.arange(n),schedule].sum()==optimum and loss[np.arange(n),schedule].sum()<=20
 tt=[tapes(p.with_suffix('.tape')) for _,p in menu];ii=[np.fromfile(p.with_suffix('.ids'),'<u8').reshape(n,10) for _,p in menu]
 write_tapes(R/'both_plans/test_OracleB_both_plan.tape',[tt[j][i] for i,j in enumerate(schedule)])
 chosen=np.array([arrays[j][i] for i,j in enumerate(schedule)],dtype=base.dtype);np.array([ii[j][i] for i,j in enumerate(schedule)],'<u8').tofile(R/'both_plans/test_OracleB_both_plan.ids');csvsave(R/'both_plans/test_OracleB_both_plan.csv',[{k:float(row[k]) for k in base.dtype.names} for row in chosen]);test['OracleB_both']=chosen
 selected['OracleB_both']=dict(configs[0],key='OracleB_both');dump(R/'OracleB_both_certificate.json',{'scope':'exact optimum in evaluated full-trajectory menu, not arbitrary individual-decision optimum','menu':[k for k,_ in menu],'actual_lost_hits':int(loss[np.arange(n),schedule].sum()),'budget_lost_hits':20,'saved_calls':optimum,'test_GT_used':True,'deployable':False});csvsave(R/'OracleB_both_choices.csv',[{'query_row':i,'trajectory':menu[j][0],'lost_hits':int(loss[i,j]),'saved_calls':int(gain[i,j])} for i,j in enumerate(schedule)])
 timing={};rng=np.random.default_rng(SEED+1)
 for rep in range(3):
  for family in rng.permutation(list(selected)):
   c=selected[family];key=c['key'];p,a=execute('test',c,'replay',f'{family}_r{rep}',R/'both_plans'/f'test_{key}_plan.tape');plan=test[key]
   assert sha(p.with_suffix('.ids'))==sha(R/'both_plans'/f'test_{key}_plan.ids')
   for field in ['hits','reads','expansions','eligible','reused','actual_calls','false_skip']:np.testing.assert_array_equal(a[field],plan[field])
   np.testing.assert_array_equal(a['actual_calls'],a['physical_calls']);timing.setdefault(family,[]).append(summary(a,base))
 diagnostics={}
 for family,c in selected.items():
  allrecords=np.concatenate(tapes(R/'both_plans'/f'test_{c["key"]}_plan.tape'));skip=allrecords['reused']==1;keep=allrecords['reused']==2;freshskip=allrecords['fresh']<json.loads((R/'PROTOCOL.json').read_text())['tau']
  diagnostics[family]={'reused_skip':int(skip.sum()),'reused_keep':int(keep.sum()),'false_skip':int((skip&~freshskip).sum()),'false_keep':int((keep&freshskip).sum()),'skip_reuse_precision':float(freshskip[skip].mean()) if skip.any() else None,'keep_reuse_precision':float((~freshskip[keep]).mean()) if keep.any() else None}
 dump(R/'both_results.json',{'timing':timing,'selected':selected,'diagnostics':diagnostics,'OracleA_both_exact_R0_ids':True,'all_replays_exact_plan_ids_and_counts':True,'physical_calls_verified':True})
 rows=[]
 for family,reps in timing.items():rows.append({'strategy':family,'selected_config':selected[family]['key'],**{k:None if reps[0][k] is None else float(np.median([r[k] for r in reps])) for k in reps[0]}})
 csvsave(R/'both_summary.csv',rows);dump(R/'both_summary.json',rows);log('BOTH-DECISION EXTENSION COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:log('BOTH FAILED\n'+traceback.format_exc());raise
