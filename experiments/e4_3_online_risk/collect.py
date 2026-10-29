from common import *
from scipy.special import expit
import html,struct
log('VERIFY all real-search traces, causal histories and risk decisions; collect artifacts')
proto=json.loads((R/'PROTOCOL.json').read_text());configs=json.loads((R/'configs.json').read_text());bykey={c['key']:c for c in configs};models=json.loads((R/'risk_models.json').read_text());lock=json.loads((R/'SELECTION_LOCK.json').read_text());tau=proto['original_tau'];norm=np.load(E2/'normalization.npz');mean=norm['mean'];denom=norm['std']+1e-8;scut=json.loads((E2/'PROTOCOL.json').read_text())['state_high_if_normalized_l2_le']
with (E1/'data/test.gt').open('rb') as f:shape=np.fromfile(f,np.uint32,2);gt=np.fromfile(f,np.uint32,int(shape.prod())).reshape(tuple(shape))
baseline_ids=np.fromfile(E41/'intervention.baseline.ids',np.uint64).reshape(1000,10);e1=np.genfromtxt(E1/'runs_main/test_MLP_r0_time.csv',delimiter=',',names=True)
h=dict(np.load(E1/'MLP_history_trace.npz'));hi=np.flatnonzero(h['model_invoked']);hs=hi[h['model_decision'][hi].astype(bool)];seedtot=np.bincount(h['node_id'][hi].astype(int),minlength=1000000).astype(np.uint64);seedskip=np.bincount(h['node_id'][hs].astype(int),minlength=1000000).astype(np.uint64);seedz=(h['state_features'][hs].astype(float)-mean)/denom;seedby={}
for j,node in enumerate(h['node_id'][hs]):seedby.setdefault(int(node),[]).append((seedz[j],0))
pdtype=np.dtype([('node','<u4'),('skip','<u4'),('fresh','<f4'),('state','<f4',(17,))]);udtype=np.dtype([('seq','<u4'),('row','<u4'),('node','<u4'),('pad','<u4'),('fresh','<f4'),('state','<f4',(17,))]);ddtype=np.dtype([('seq','<u4'),('row','<u4'),('node','<u4'),('ordinal','<u4'),('history_version','<u4'),('nearest_origin','<u4'),('action','<u4'),('kind','<u4'),('fresh','<f8'),('risk','<f8'),('distance','<f8'),('node_total','<f8'),('node_skips','<f8'),('x','<f8',(28,))]);assert pdtype.itemsize==80 and udtype.itemsize==88 and ddtype.itemsize==296
rng=np.random.default_rng(SEED);draws=rng.integers(0,250,(1000,250));summaries=[];all_data={};nobs=0;ndec=0;nupdate=0;maxfeature=0.;maxrisk=0.;original_samples=[];original_scores=[];sample_rng=np.random.default_rng(SEED+100)
checked=[]
for directory in sorted((R/'graph').iterdir()):
 if not directory.is_dir():continue
 tag=directory.name;cold=tag.endswith('_cold');audit=tag=='audit';prefixes=sorted(directory.glob('*.complete'))
 if not prefixes:continue
 # Every policy stream is compared with an original MLP stream in the same request order.
 bg=np.genfromtxt(directory/'MLP.csv',delimiter=',',names=True);base_rows=bg['query_row'].astype(int);base_hits=bg['hits'];base_reads=bg['reads'];bids=np.fromfile(directory/'MLP.ids',np.uint64).reshape(len(bg),10);assert np.array_equal(bids,baseline_ids[base_rows])
 for done in prefixes:
  key=done.stem;cfg=bykey[key];prefix=directory/key;g=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True);rows=g['query_row'].astype(int);seqs=g['seq'].astype(int);assert np.array_equal(rows,base_rows) and np.array_equal(seqs,np.arange(len(g)))
  ids=np.fromfile(str(prefix)+'.ids',np.uint64).reshape(len(g),10);hits=(ids[:,:,None]==gt[rows,None,:10]).any(2).sum(1);assert np.array_equal(hits,g['hits']);assert sha(Path(str(prefix)+'.cache.bin'))==sha(E1/'runs_main/test_MLP_trace.cache.bin')
  obs=np.fromfile(str(prefix)+'.observations.bin',pdtype);dec=np.fromfile(str(prefix)+'.decisions.bin',ddtype);updates=np.fromfile(str(prefix)+'.updates.bin',udtype);nobs+=len(obs);ndec+=len(dec);nupdate+=len(updates)
  assert len(obs)==int(g['model_calls'].sum()) and len(dec)==int(g['secondary_evaluations'].sum()) and len(updates)==int(g['committed_skips'].sum())
  assert np.all(obs['state'][:,10]==0);assert np.array_equal(obs['skip'].astype(bool),obs['fresh']<tau);assert np.all(updates['fresh']<tau)
  if len(obs):
   sample=sample_rng.choice(len(obs),min(24,len(obs)),replace=False);original_samples.append(obs['state'][sample]);original_scores.append(obs['fresh'][sample])
  total=np.zeros_like(seedtot) if cold else seedtot.copy();skip=np.zeros_like(seedskip) if cold else seedskip.copy();added={};global_total=int(total.sum());global_skip=int(skip.sum());marked=int(np.sum(skip>0));op=dp=up=0;updating=not audit and cfg['kind']!=3
  for j,line in enumerate(g):
   calls=int(line['model_calls']);ds_count=int(line['secondary_evaluations']);un=int(line['committed_skips']);ps=obs[op:op+calls];decs=dec[dp:dp+ds_count];us=updates[up:up+un];op+=calls;dp+=ds_count;up+=un
   assert int(line['history_version'])==j and int(line['observations_before'])==global_total and int(line['skip_records_before'])==global_skip and int(line['marked_nodes_before'])==marked
   assert int(line['original_keeps'])==int(np.sum(ps['skip']==0)) and int(line['original_skips'])==int(np.sum(ps['skip']==1))
   assert len(np.unique(ps['node']))==len(ps);pn=ps['node'].astype(int);H=skip[pn]>0;assert int(line['conflicts'])==int(np.sum(H&(ps['skip']==0)))
   assert int(line['secondary_skips'])==int(decs['action'].sum());assert int(line['model_calls'])==int(line['original_keeps']+line['original_skips'])
   if cfg['kind']!=6:assert np.all(decs['node_skips']>0)
   if cold and j==0 and cfg['kind']!=6:assert ds_count==0
   if len(decs):
    assert np.all(decs['seq']==j) and np.all(decs['row']==rows[j]) and np.all(decs['history_version']==j)
    nodepos={int(n):k for k,n in enumerate(ps['node'])};pi=np.array([nodepos[int(n)] for n in decs['node']]);assert np.all(np.diff(decs['ordinal'].astype(np.int64))>0) and np.all(np.diff(pi)>0);assert np.array_equal(ps['node'][pi],decs['node']);assert np.all(ps['skip'][pi]==0);assert np.array_equal(ps['fresh'][pi].astype(float),decs['fresh'])
    for d,p in zip(decs,ps[pi]):
     node=int(d['node']);assert d['node_total']==total[node] and d['node_skips']==skip[node];x=np.zeros(28);fresh=float(p['fresh']);prob=np.clip(fresh,1e-7,1-1e-7);x[0]=np.log(prob/(1-prob));x[1:18]=(p['state'].astype(float)-mean)/denom;x[18]=skip[node]>0;x[19]=np.log1p(float(total[node]));x[20]=np.log1p(float(skip[node]));x[21]=float(skip[node])/max(float(total[node]),1.)
     if skip[node]>0:
      records=([] if cold else seedby.get(node,[]))+added.get(node,[]);assert len(records)==int(skip[node]);zz=np.array([v[0] for v in records]);distance=np.linalg.norm(zz-x[1:18],axis=1);nn=int(np.argmin(distance));origin=records[nn][1];dist=float(distance[nn]);valid_origins={records[k][1] for k in np.flatnonzero(distance<=dist+1e-9)};assert int(d['nearest_origin'])<j+1 and int(d['nearest_origin']) in valid_origins;assert abs(dist-d['distance'])<1e-7;x[22]=np.log1p(dist);x[25]=dist<scut
     else:assert int(d['nearest_origin'])==4294967295
     err=float(np.max(np.abs(x-d['x'])));maxfeature=max(maxfeature,err);assert err<1e-7
     if cfg['kind']==1:risk=0.;action=True
     elif 2<=cfg['kind']<=4:
      m=models[0 if cfg['kind']==2 else 1];risk=float(expit(x@np.array(m['weights'])+m['intercept']));action=cfg['threshold']>0 and risk<=cfg['threshold']
     else:risk=fresh;action=risk<=cfg['threshold']
     maxrisk=max(maxrisk,abs(risk-float(d['risk'])));assert abs(risk-float(d['risk']))<2e-7;assert action==bool(d['action'])
   if updating:
    os=ps[ps['skip']!=0];assert len(us)==len(os) and np.all(us['seq']==j) and np.all(us['row']==rows[j]);assert np.array_equal(us['node'],os['node']) and np.array_equal(us['fresh'],os['fresh']) and np.array_equal(us['state'],os['state'])
    sn=os['node'].astype(int);marked+=int(np.sum(skip[np.unique(sn)]==0));np.add.at(total,pn,1);np.add.at(skip,sn,1);global_total+=len(ps);global_skip+=len(os)
    for p in os:added.setdefault(int(p['node']),[]).append(((p['state'].astype(float)-mean)/denom,j+1))
    assert int(line['committed_observations'])==len(ps)
   else:assert len(us)==0 and int(line['committed_observations'])==0
   assert int(line['observations_after'])==global_total and int(line['skip_records_after'])==global_skip and int(line['marked_nodes_after'])==marked
  assert op==len(obs) and dp==len(dec) and up==len(updates)
  if cfg['kind']==0 or (cfg['kind'] in [2,3,4] and cfg['threshold']==0):
   assert np.array_equal(ids,baseline_ids[rows]);assert np.array_equal(g['reads'],e1['reads'][rows]);assert np.array_equal(g['expanded'],e1['expansions'][rows])
  if not audit:
   dr=(hits-base_hits)*10.;di=base_reads-g['reads'];brec=dr[draws].mean(1);bio=di[draws].mean(1);entry=dict(tag=tag,key=key,family=cfg['family'],kind=cfg['kind'],threshold=cfg['threshold'],initialization='cold' if cold else 'warm',queries=len(g),recall_pct=float(hits.mean()*10),delta_recall_pp=float(dr.mean()),delta_recall_lo=float(np.percentile(brec,2.5)),delta_recall_hi=float(np.percentile(brec,97.5)),reads_per_query=float(g['reads'].mean()),reads_saved_per_query=float(di.mean()),reads_saved_lo=float(np.percentile(bio,2.5)),reads_saved_hi=float(np.percentile(bio,97.5)),reads_saved_fraction=float(di.sum()/base_reads.sum()),expanded_per_query=float(g['expanded'].mean()),neighbor_cmps_per_query=float(g['neighbor_cmps'].mean()),original_MLP_calls_per_query=float(g['model_calls'].mean()),original_skips_per_query=float(g['original_skips'].mean()),conflicts_per_query=float(g['conflicts'].mean()),risk_LR_calls_per_query=float(g['risk_LR_calls'].mean()),secondary_skips_per_query=float(g['secondary_skips'].mean()),search_us_mean=float(g['search_us'].mean()),update_us_mean=float(g['update_us'].mean()),service_us_mean=float(g['service_us'].mean()),service_us_p50=float(np.percentile(g['service_us'],50)),service_us_p95=float(np.percentile(g['service_us'],95)),service_us_p99=float(np.percentile(g['service_us'],99)),final_history_observations=int(g['observations_after'][-1]),final_history_skip_records=int(g['skip_records_after'][-1]),final_history_marked_nodes=int(g['marked_nodes_after'][-1]),final_history_payload_bytes=int(g['history_payload_bytes'][-1]),final_history_container_lower_bound_bytes=int(g['history_container_lower_bound_bytes'][-1]))
   summaries.append(entry);all_data[(tag,key)]=g
  checked.append(dict(tag=tag,key=key,queries=len(g),original_observations=len(obs),secondary_decisions=len(dec),original_Skip_updates=len(updates),all_final_hits_recomputed=True,history_counts_and_source_origins_replayed=True,all_live_risk_features_recomputed=True))
 log('VERIFIED '+tag)
# Original native MLP checked on a reproducible sample of actually visited states from all runs.
states=np.concatenate(original_samples);expected=np.concatenate(original_scores)
with (R.parent/'sift1m/online_state_mlp/model.bin').open('rb') as f:
 assert f.read(8)==b'STGATE01';m=np.fromfile(f,np.float32,17);scale=np.fromfile(f,np.float32,17);params=[]
 for ni,no in [(17,128),(128,64),(64,1)]:params.append((np.fromfile(f,np.float32,ni*no).reshape(ni,no),np.fromfile(f,np.float32,no)))
z=(states-m)/scale
for i,(w,bias) in enumerate(params):z=z@w+bias;z=np.maximum(z,0) if i<2 else z
pred=expit(z[:,0]);mlp_error=float(np.max(np.abs(pred-expected)));assert mlp_error<3e-6,mlp_error
csvsave(R/'all_run_summary.csv',summaries);csvsave(R/'recall_ssd_curve.csv',[s for s in summaries if s['tag'] in ['validation','test']]);csvsave(R/'run_verification.csv',checked)
# Deterministic repeat checks use exact integer/text fields and binary histories, excluding clocks.
repeatkeys=[p.stem for p in (R/'graph/test_repeat').glob('*.complete')]
for key in repeatkeys:
 left=R/'graph/test'/key;right=R/'graph/test_repeat'/key
 for ext in ['ids','observations.bin','updates.bin','decisions.bin','complete']:assert sha(Path(str(left)+'.'+ext))==sha(Path(str(right)+'.'+ext)),(key,ext)
 with Path(str(left)+'.csv').open() as f:l=list(csv.DictReader(f))
 with Path(str(right)+'.csv').open() as f:r=list(csv.DictReader(f))
 for a,b in zip(l,r):assert all(a[k]==b[k] for k in a if k not in ['search_us','update_us','service_us']),key
# Probability-one dynamic risk reproduces unconditional historical-mark pruning exactly.
for tag in ['validation','test']:
 cfg=next(c for c in configs if c['family']=='Risk_dynamic_history' and c['threshold']==1.)
 for ext in ['ids','observations.bin','updates.bin','complete']:assert sha(R/'graph'/tag/(cfg['key']+'.'+ext))==sha(R/'graph'/tag/('History_unconditional.'+ext)),(tag,ext)
selected=[]
for item in lock['selection']:
 out=dict(R_min_pct=item['R_min_pct'],family=item['family'],feasible_on_validation=item['feasible_on_validation'],selected_key=item['selected']['key'] if item['selected'] else None,validation_recall_pct=None,validation_reads_per_query=None,test_recall_pct=None,test_meets_R_min=None,test_reads_per_query=None,test_reads_saved_fraction=None,test_delta_recall_pp=None,test_delta_recall_lo=None,test_delta_recall_hi=None)
 if item['selected']:
  key=item['selected']['key'];vs=next(s for s in summaries if s['tag']=='validation' and s['key']==key);ts=next(s for s in summaries if s['tag']=='test' and s['key']==key);out.update(validation_recall_pct=vs['recall_pct'],validation_reads_per_query=vs['reads_per_query'],test_recall_pct=ts['recall_pct'],test_meets_R_min=ts['recall_pct']>=item['R_min_pct']-1e-9,test_reads_per_query=ts['reads_per_query'],test_reads_saved_fraction=ts['reads_saved_fraction'],test_delta_recall_pp=ts['delta_recall_pp'],test_delta_recall_lo=ts['delta_recall_lo'],test_delta_recall_hi=ts['delta_recall_hi'])
 selected.append(out)
csvsave(R/'selected_results.csv',selected)
robust=[]
for item in lock['selection']:
 if item['R_min_pct']!=proto['primary_R_min_pct'] or not item['selected']:continue
 key=item['selected']['key']
 for s in summaries:
  if s['key']==key and s['tag'].startswith('test_shuffle'):robust.append(dict(selected_family=item['family'],R_min_pct=item['R_min_pct'],meets_R_min=s['recall_pct']>=item['R_min_pct']-1e-9,**s))
csvsave(R/'order_initialization_robustness.csv',robust)
# Simple exportable vector curve, without an optional plotting dependency.
F=R/'figures';F.mkdir(exist_ok=True);colors=['#64748b','#dc2626','#16a34a','#9333ea','#2563eb','#d97706','#0f172a'];families=list(dict.fromkeys(c['family'] for c in configs));test=[s for s in summaries if s['tag']=='test'];xs=[s['reads_per_query'] for s in test];ys=[s['recall_pct'] for s in test];xmin,xmax=min(xs)-1,max(xs)+1;ymin,ymax=min(ys)-.1,max(ys)+.1
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="560"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12" fill="#1e293b"><text x="80" y="28" font-size="18">E4-3: sequential dynamic-history graph search (held-out Queries)</text>']
def point(x,y):return 80+(x-xmin)/(xmax-xmin)*640,470-(y-ymin)/(ymax-ymin)*410
for x in np.linspace(xmin,xmax,6):
 xx,_=point(x,ymin);svg.append(f'<path d="M{xx} 60V470" stroke="#e2e8f0"/><text x="{xx-15}" y="495">{x:.2f}</text>')
for y in np.linspace(ymin,ymax,6):
 _,yy=point(xmin,y);svg.append(f'<path d="M80 {yy}H720" stroke="#e2e8f0"/><text x="25" y="{yy+4}">{y:.2f}</text>')
for i,family in enumerate(families):
 pts=sorted((s['reads_per_query'],s['recall_pct']) for s in test if s['family']==family);positions=[point(*p) for p in pts];color=colors[i%len(colors)];svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in positions)+'"/>')
 for xx,yy in positions:svg.append(f'<circle cx="{xx}" cy="{yy}" r="3" fill="{color}"/>')
 svg.append(f'<path d="M750 {80+i*27}h20" stroke="{color}" stroke-width="3"/><text x="780" y="{84+i*27}">{html.escape(family)}</text>')
_,ry=point(xmin,proto['primary_R_min_pct']);svg.append(f'<path d="M80 {ry}H720" stroke="#ef4444" stroke-dasharray="5 4"/><text x="750" y="320">R_min = {proto["primary_R_min_pct"]:.2f}%</text><text x="220" y="535">SSD requests / Query (lower is better)</text><text x="80" y="50">Recall@10 (%)</text></g></svg>');(F/'recall_ssd_curve.svg').write_text(''.join(svg))
verification=dict(complete=True,streams_verified=len(checked),complete_graph_queries=sum(c['queries'] for c in checked),original_observations_replayed=nobs,secondary_decisions_recomputed=ndec,original_Skip_updates_replayed=nupdate,max_live_secondary_feature_error=maxfeature,max_risk_probability_error=maxrisk,original_MLP_sampled_states=len(states),max_original_MLP_score_error=mlp_error,baseline_native_parity=json.loads((R/'native_parity.json').read_text()),all_final_hits_recomputed_from_ids_GT=True,all_history_sources_seed_or_previously_completed_query=True,secondary_Skip_never_inserted_as_history_Skip=True,all_updates_original_MLP_Skip_only=True,all_history_observation_counters_include_original_Keeps=True,no_history_updates_during_query=True,static_history_immutable=True,cold_first_query_no_conflicts=True,zero_risk_threshold_equals_original_MLP=True,probability_one_dynamic_risk_equals_unconditional_mark=True,selected_policy_and_history_exact_repeat_verified=repeatkeys,fit_validation_test_queries_disjoint=True,selection_uses_validation_only=True,original_MLP_model_sha256=sha(R.parent/'sift1m/online_state_mlp/model.bin'),source_inputs=json.loads((R/'INPUT_LOCK.json').read_text()))
dump(R/'verification.json',verification)
dump(R/'results.json',dict(primary_R_min_pct=proto['primary_R_min_pct'],selected_results=selected,main_curve_points=len([s for s in summaries if s['tag'] in ['validation','test']]),robustness_rows=len(robust),verified_streams=len(checked),verified_complete_queries=verification['complete_graph_queries'],objective=proto['objective'],metric_definitions={'SSD Reads':'submitted reader requests, not physical device reads','Recall':'Top10 hit fraction against exact GT','history bytes':'payload and container lower-bound estimates, not total RSS','bootstrap':'paired Query bootstrap conditional on realized stream histories; order sensitivity measured by actual replays','service_us':'actual search plus commit, excluding trace serialization, history-size accounting, index loading and BFS cache initialization'}))
readme='''# E4-3 运行产物

实验已执行，参数与数据定义见 `PROTOCOL.json`；本文件仅说明保存位置与复现方式。

- `selected_results.csv`：各绝对 Recall 下限下，由验证集选择的配置及评估集指标。
- `recall_ssd_curve.csv`、`figures/recall_ssd_curve.svg`：验证/评估完整 Recall–SSD 请求曲线。
- `all_run_summary.csv`：全部顺序、初始化和配置的统计。
- `order_initialization_robustness.csv`：验证选点在三种固定重排、暖历史与冷启动下的实际搜索结果。
- `graph/<运行名>/<配置>.csv`：逐 Query 指标、模型/二次评估计数、历史版本和历史更新计数。
- `graph/<运行名>/<配置>.ids`：最终 Top10，按逐 Query CSV 顺序保存的 uint64，形状为查询数×10。
- `*.observations.bin`：每次实际原始 MLP 输出及17维状态，顺序由 CSV 的 model_calls 划分；记录类型为 node:uint32, skip:uint32, fresh:float32, state:float32[17]，每条80字节。
- `*.decisions.bin`：二次评估完整输入、来源历史版本、最近历史状态来源、风险值及动作；格式见 `collect.py` 的 ddtype，每条296字节。
- `*.updates.bin`：Query完成后实际提交的原始 MLP Skip 状态；格式见 `collect.py` 的 udtype，每条88字节。
- `risk_models.json`：二次风险模型权重；原始 MLP 没有重新训练。
- `SELECTION_LOCK.json`：测试前冻结的验证选点。
- `verification.json`、`run_verification.csv`：逐流因果历史重放、风险特征/动作复算、最终 Recall 重算及确定性重复核验。
- `*.command.json`、`*.log`、`INPUT_LOCK.json`：完整命令、阶段日志与输入哈希。

主约束默认 R_min=99.44%，另保存99.60%、99.20%、99.00%的选择结果。验证可行不保证评估仍可行，是否达标单列记录。数据来自此前已观察的E4-1查询池，属于探索性实验；使用E4-2的500/250/250 Query划分。顺序为固定基准顺序及可复现重排，不能称为生产请求流。

搜索严格串行读取历史，只在当前 Query 完成后更新；二次 Skip 不成为历史 Skip。缓存命中沿用E1–E4保护策略。历史对有限流保留全部记录，无淘汰；内存字段是负载/容器下界估计，不是RSS。置信区间条件于已实现的历史轨迹，不能替代不同流顺序的真实重放。

复现：`.venv/bin/python experiments/e4_3_online_risk/run.py`。程序按已完成配置恢复；协议、模型或原生输入变化必须使用新目录。
'''
(R/'README.md').write_text(readme)
# Preserve records and hashes of all previous E4 inputs; no mutation of older experiments.
source_provenance={str(f):sha(f) for f in [E41/'COMPLETE.json',E41/'keep_positions.npz',E41/'outcomes.npz',E42/'COMPLETE.json',E42/'features.npz',E42/'PROTOCOL.json',E2/'normalization.npz',E1/'SELECTION_LOCK.json']};dump(R/'PROVENANCE.json',source_provenance)
artifacts={str(f.relative_to(R)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(R.rglob('*')) if f.is_file() and '__pycache__' not in str(f) and f.name not in ['artifact_manifest.json','COMPLETE.json','run.log','experiment.log']};dump(R/'artifact_manifest.json',artifacts);dump(R/'COMPLETE.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_streams=len(checked),verified_complete_queries=verification['complete_graph_queries'],verification_sha256=sha(R/'verification.json'),manifest_sha256=sha(R/'artifact_manifest.json')))
log('E4-3 COMPLETE; artifacts: '+str(R))
