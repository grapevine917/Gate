from common import *
MODELS=['lr','xgb','state']
RISKS=[0,.001,.0025,.005,.01,.02,.05,.1];ETS=[1.1,.999,.995,.99,.98,.95,.9,.8]
HEUR=[(1.,-1),(.98,1.),(.92,1.04),(.72,1.),(.64,.92),(.5,.9),(.25,.75),(.1,.5)]
PAIRS=[(0,0),(4,0),(0,3),(2,1),(3,2),(4,3),(5,4),(6,5)]
for ds in CATALOG:
 C=catalog(ds);O=C['out'];R=C['parent_dir'];O.mkdir(exist_ok=True)
 commands=json.loads((R/'build.commands.json').read_text());cmd=next(c for c in commands if str(R/'search.cpp') in c)
 cmd=[str(D/'calibrate.cpp') if x==str(R/'search.cpp') else x for x in cmd];cmd[cmd.index('-o')+1]=str(O/'native/calibrate')
 with (O/'calibrate_build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 calibration=ROOT/C['calibration'];a=np.fromfile(calibration,'<f4').reshape(-1,20);miss=a[:,13]==0;positive=(a[:,2]==1)&miss;assert positive.any();T={}
 for model in MODELS:
  scorefile=O/f'calibration_{model}.scores';subprocess.run([str(O/'native/calibrate'),str(R/'models'/f'{model}.bin'),str(calibration),str(scorefile)],check=True)
  scores=np.fromfile(scorefile,'<f4');assert len(scores)==len(a) and np.isfinite(scores).all();pos=np.sort(scores[positive]);T[model]=[]
  for risk in RISKS:
   allowed=int(np.floor(risk*len(pos)));tau=0. if risk==0 else float(pos[allowed]);false=int(np.sum(scores[positive]<tau));assert false<=allowed;T[model].append(dict(risk=risk,tau=tau,false_skips=false,cache_miss_retains=len(pos)))
 dump(O/'thresholds.json',T)
 configs=[]
 def add(key,family,kind,model='none',L=100,tau=0.,et=1.1,rank=100,exp=-1,preset=0):configs.append(dict(key=key,family=family,kind=kind,model=model,L=int(L),tau=float(tau),et_tau=float(et),rank=int(rank),exp=int(exp),preset=preset))
 ls=C['LS'];base=set(ls)
 while len(base)<64:base.add(max((l for l in range(10,max(ls)+1) if l not in base),key=lambda l:(min(abs(math.log(l/b)) for b in base),-l)))
 for L in sorted(base):add(f'DiskANN_L{L}','DiskANN',0,L=L)
 for L in ls:
  for j in range(8):
   r,e=HEUR[j];add(f'Heuristic_L{L}_p{j}','Heuristic',1,L=L,rank=int(L*r),exp=-1 if e<0 else int(L*e),preset=j)
   add(f'ET_L{L}_p{j}','ET',3,L=L,et=ETS[j],preset=j)
   for model in MODELS:
    add(f'Ours_{model}_L{L}_p{j}',f'Ours_{model}',2,model,L,T[model][j]['tau'],preset=j)
    pi,ei=PAIRS[j];add(f'Joint_{model}_L{L}_p{j}',f'Joint_{model}',4,model,L,T[model][pi]['tau'],ETS[ei],preset=j)
 assert len(configs)==576 and len({c['key'] for c in configs})==576
 p0a=['DiskANN_L100','Heuristic_L100_p3']+[f'Ours_{m}_L100_p4' for m in MODELS]
 files=[O/'native/search',D/'search.cpp',R/'TRAIN_COMPLETE.json',calibration,*[R/'models'/f'{m}.bin' for m in MODELS+['et']],*[R/'data'/f'{s}.{e}' for s in ['validation','test'] for e in [C['query_ext'],'gt']],R/'data/train_ids.npy',R/'data/validation_ids.npy',R/'data/test_ids.npy']
 protocol=dict(dataset=C['label'],N=C['N'],dimensions=C['dim'],index=str(C['index_path']),old_model_parent=str(R),models=MODELS,model_training='reuse existing matched mixed-L proxy-label LR/XGB/state MLP and matching ET; no weights trained, edited, or replaced',training_L=ls,beam=4,K=10,threads=4,MKL_threads=1,cache_nodes=10000,seed=SEED,validation_queries=1000,test_queries=1000,P0_A=dict(L=100,keys=p0a,repeats=15,ordering='5 cyclic positions x3',tau='native validation L100 cache-miss retain-FNR1%, each model independently calibrated',heuristic='rank>72 AND completed>100; cache misses only'),P0_B=dict(families=sorted({c['family'] for c in configs}),per_family_configs=64,L=ls,baseline_L=sorted(base),risks=RISKS,ET_thresholds=ETS,heuristic_fraction_pairs=HEUR,joint_pairs=PAIRS,targets=[90.,95.,98.,99.,99.5,99.7,99.9],selection='Validation-only per-family Recall constraint and per-objective costs; lock before test; full frozen test-grid frontiers descriptive; never interpolate actual measured costs',repeat='all test configs once; union of every objective frontier and validation-locked choices to3 until all displayed frontier configs have >=3 repeats; P0A pool shared',baseline='64 distinct L incl every learned L; tuned strong native baseline'),timing='datasets serial; configs shuffled into small mixed-family batches; shared controls across Ours versions;128-query warmup; model/index loads excluded; scoring/features/search/IO included',physical_IO='Actual Linux proc_self_io read_bytes in timed search equals metered reader bytes and4096*query_reads; no tolerance; archive and retry anomalies; reject swapping',calibration='same full L100 validation cache-miss candidate proxy labels per dataset, no test outcomes',limits=['previously observed benchmark queries, exploratory','single existing seed; timing repeats omit model-training variation','equal finite64-config budgets do not prove global policy optimality','8 declared joint pairs, not exhaustive Cartesian sweep','ET trained on unpruned prefixes; joint policy can shift trajectory','float1M vs uint8 compressed100M implementations differ; cross-dataset is not pure size scaling'],source_hashes={str(p):sha(p) for p in files})
 if (O/'PROTOCOL.json').exists():assert json.loads((O/'PROTOCOL.json').read_text())==protocol
 else:dump(O/'PROTOCOL.json',protocol)
 if (O/'configs.json').exists():assert json.loads((O/'configs.json').read_text())==configs
 else:dump(O/'configs.json',configs)
 ids=[np.load(R/'data'/f'{s}_ids.npy') for s in ['train','validation','test']];assert all(not set(ids[i])&set(ids[j]) for i,j in [(0,1),(0,2),(1,2)])
 dump(O/'PREPARED.json',dict(complete=True,configurations=576,old_models_unchanged=True));print('PREPARED',ds,flush=True)
dump(D/'PREPARED.json',dict(complete=True,datasets=list(CATALOG),models=MODELS))
