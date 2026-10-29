from common import *
from scipy.special import expit
import itertools,html,joblib
log('VERIFY native curve, inference-profile and repeated timing results')
proto=json.loads((R/'PROTOCOL.json').read_text());configs=json.loads((R/'configs.json').read_text());bykey={c['key']:c for c in configs};lock=json.loads((R/'SELECTION_LOCK.json').read_text());chosen=json.loads((R/'timing_configs.json').read_text());curves=[];runrows=[];data={};verification=[]
def truth(split):
 with (E1/'data'/f'{split}.gt').open('rb') as f:shape=np.fromfile(f,np.uint32,2);return np.fromfile(f,np.uint32,int(shape.prod())).reshape(tuple(shape))
gts={s:truth(s) for s in ['validation','test']};baseids={};expectedids={}
for audit_key in json.loads((R/'native_audit.json').read_text()):assert sha(R/'audit'/(audit_key+'.cache.bin'))==sha(R/'audit/DiskANN.cache.bin')
for stage in ['validation','test_curve','profile']+[f'timing_r{i}' for i in range(5)]:
 split='validation' if stage=='validation' else 'test';cfgs=configs if stage in ['validation','test_curve'] else chosen
 for c in cfgs:
  p=R/stage/c['key'];a=np.genfromtxt(str(p)+'.csv',delimiter=',',names=True);ids=np.fromfile(str(p)+'.ids',np.uint64).reshape(1000,10);hits=(ids[:,:,None]==gts[split][:,None,:10]).any(2).sum(1);assert np.array_equal(hits,a['hits']);assert len(a)==1000
  idhash=sha(Path(str(p)+'.ids'))
  if stage in ['validation','test_curve']:
   if c['group']=='DiskANN':baseids[split]=idhash
   if c['tau']==0:assert idhash==baseids[split],(stage,c['key'])
  if split=='test':
   if c['key'] not in expectedids:expectedids[c['key']]=idhash
   assert expectedids[c['key']]==idhash,(stage,c['key'])
  if c['group']=='DiskANN':assert np.all(a['model_calls']==0)
  else:assert np.array_equal(a['model_calls']-a['skips'],a['reads'])
  if stage!='profile':assert np.all(a['model_us']==0)
  else:
   components=sum(a[k] for k in ['distance_us','model_us','memory_us','ssd_us','bookkeeping_us']);assert np.max(np.abs(components-a['total_us']))<1e-5
  if stage.startswith('timing_') or stage=='profile':data[(stage,c['key'])]=a
  row=dict(stage=stage,dedicated_latency_benchmark=stage.startswith('timing_'),instrumented_profile=stage=='profile',key=c['key'],group=c['group'],seed=c['seed'],tau=c['tau'],recall_pct=float(hits.mean()*10),reads_per_query=float(a['reads'].mean()),expanded_per_query=float(a['expansions'].mean()),model_calls_per_query=float(a['model_calls'].mean()),skips_per_query=float(a['skips'].mean()),latency_us=float(a['total_us'].mean()),p50_us=float(np.percentile(a['total_us'],50)),p95_us=float(np.percentile(a['total_us'],95)),p99_us=float(np.percentile(a['total_us'],99)),model_profile_us=float(a['model_us'].mean()),model_profile_us_per_call=float(a['model_us'].sum()/max(a['model_calls'].sum(),1)),target_FNR=c['target_FNR'])
  runrows.append(row)
  if stage in ['validation','test_curve']:curves.append(row)
  verification.append(dict(stage=stage,key=c['key'],all_hits_recomputed=True,final_ids_sha256=idhash))
 csvsave(R/'all_run_summary.csv',runrows);log('VERIFIED '+stage)
csvsave(R/'recall_io_curves.csv',curves);csvsave(R/'run_verification.csv',verification)
# Full17 seed42 is a fresh training, not a copied old model; all groups are handled identically.
selected=[]
for item in lock['selection']:
 c=item['selected'];row=dict(R_min_pct=item['R_min_pct'],group=item['group'],seed=item['seed'],selected_key=c['key'] if c else None,tau=c['tau'] if c else None,validation_recall_pct=None,validation_reads=None,test_recall_pct=None,test_reads=None,test_meets_R_min=False)
 if c:
  v=next(x for x in curves if x['stage']=='validation' and x['key']==c['key']);t=next(x for x in curves if x['stage']=='test_curve' and x['key']==c['key']);row.update(validation_recall_pct=v['recall_pct'],validation_reads=v['reads_per_query'],test_recall_pct=t['recall_pct'],test_reads=t['reads_per_query'],test_meets_R_min=t['recall_pct']>=item['R_min_pct']-1e-9)
 selected.append(row)
csvsave(R/'selected_recall_constraints.csv',selected)
# Balanced arrays pair the same seed, timing round and Query across groups.
T={};IO={};REC={};groupsummary=[];rng=np.random.default_rng(20261025);B=2000
resample=[(rng.integers(0,3,3),rng.integers(0,5,5),rng.integers(0,1000,1000)) for _ in range(B)]
for group in GROUPS:
 items=[s for s in selected if s['group']==group and s['R_min_pct']==PRIMARY];assert len(items)==3 and all(s['selected_key'] for s in items)
 items=sorted(items,key=lambda s:SEEDS.index(s['seed']));keys=[s['selected_key'] for s in items];T[group]=np.array([[data[(f'timing_r{r}',k)]['total_us'] for r in range(5)] for k in keys]);IO[group]=np.array([data[('timing_r0',k)]['reads'] for k in keys]);REC[group]=np.array([data[('timing_r0',k)]['hits']*10 for k in keys]);modelprof=[data[('profile',k)] for k in keys]
 for key in keys:
  first=data[('timing_r0',key)]
  for r in range(1,5):
   other=data[(f'timing_r{r}',key)]
   for metric in ['hits','reads','expansions','model_calls','skips','pq_comparisons']:assert np.array_equal(first[metric],other[metric]),(key,r,metric)
 ci=np.percentile([T[group][np.ix_(si,ri,qi)].mean() for si,ri,qi in resample],[2.5,97.5]);params=[json.loads((R/'models'/f'{group}_s{s}/metrics.json').read_text())['parameter_count'] for s in SEEDS]
 groupsummary.append(dict(group=group,n_features=len(GROUPS[group]),parameters=params[0],R_min_pct=PRIMARY,mean_test_recall_pct=float(REC[group].mean()),min_seed_test_recall_pct=float(REC[group].mean(1).min()),all_seeds_test_meet_R_min=all(s['test_meets_R_min'] for s in items),reads_per_query=float(IO[group].mean()),reads_seed_min=float(IO[group].mean(1).min()),reads_seed_max=float(IO[group].mean(1).max()),model_calls_per_query=float(np.mean([a['model_calls'].mean() for a in modelprof])),model_profile_us_per_query=float(np.mean([a['model_us'].mean() for a in modelprof])),model_profile_us_per_call=float(sum(a['model_us'].sum() for a in modelprof)/sum(a['model_calls'].sum() for a in modelprof)),end_to_end_mean_us=float(T[group].mean()),end_to_end_mean_lo=float(ci[0]),end_to_end_mean_hi=float(ci[1]),end_to_end_p50_us=float(np.percentile(T[group],50)),end_to_end_p95_us=float(np.percentile(T[group],95)),end_to_end_p99_us=float(np.percentile(T[group],99)),timing_run_mean_min=float(T[group].mean(2).min()),timing_run_mean_max=float(T[group].mean(2).max()),seeds=3,timing_repeats=5,selected_keys=';'.join(keys)))
csvsave(R/'primary_group_comparison.csv',groupsummary)
pairs=[]
for left,right in itertools.combinations(GROUPS,2):
 d=T[left]-T[right];dio=IO[left]-IO[right];drec=REC[left]-REC[right];lv=np.array([d[np.ix_(si,ri,qi)].mean() for si,ri,qi in resample]);iv=np.array([dio[np.ix_(si,qi)].mean() for si,ri,qi in resample]);rv=np.array([drec[np.ix_(si,qi)].mean() for si,ri,qi in resample]);pairs.append(dict(left=left,right=right,latency_difference_us=float(d.mean()),latency_lo=float(np.percentile(lv,2.5)),latency_hi=float(np.percentile(lv,97.5)),IO_difference=float(dio.mean()),IO_lo=float(np.percentile(iv,2.5)),IO_hi=float(np.percentile(iv,97.5)),recall_difference_pp=float(drec.mean()),recall_lo=float(np.percentile(rv,2.5)),recall_hi=float(np.percentile(rv,97.5))))
csvsave(R/'paired_group_differences.csv',pairs)
feasible=[s for s in groupsummary if s['all_seeds_test_meet_R_min']];bestio=min(feasible,key=lambda x:x['reads_per_query'])['group'] if feasible else None;bestlat=min(feasible,key=lambda x:x['end_to_end_mean_us'])['group'] if feasible else None
latency_supported=bool(bestlat)
if bestlat:
 for s in feasible:
  if s['group']==bestlat:continue
  p=next(p for p in pairs if set([p['left'],p['right']])==set([bestlat,s['group']]))
  if (p['latency_hi'] if p['left']==bestlat else -p['latency_lo'])>=0:latency_supported=False
result=dict(primary_R_min_pct=PRIMARY,group_comparison=groupsummary,best_IO_group_among_all_seeds_test_feasible=bestio,fastest_observed_group_among_all_seeds_test_feasible=bestlat,fastest_group_pairwise_95pct_intervals_exclude_zero_against_other_feasible_groups=latency_supported,overall_interpretation='I/O and inference-inclusive latency are separate outcomes; no forced overall winner if their winners differ or latency differences are uncertain',paired_comparisons=pairs,existing_test_previously_observed=True,selection='validation graph minimum reads under common absolute Recall bound; no test threshold retuning',uncertainty='2000 paired nested bootstrap draws over 3 matched training seeds, 5 randomized timing rounds and 1000 Query IDs; pairwise intervals unadjusted for multiple comparisons')
dump(R/'results.json',result)
# Standalone vector figures for actual curve points; no test-interpolated configuration is deployed.
F=R/'figures';F.mkdir(exist_ok=True);palette={'Global':'#2563eb','Candidate':'#16a34a','Full17':'#dc2626'}
for metric,filename,xlabel in [('reads_per_query','recall_io.svg','SSD requests / Query')]:
 vals=[v for v in curves if v['stage']=='test_curve'];xs=[v[metric] for v in vals];ys=[v['recall_pct'] for v in vals];xmin,xmax=min(xs)-.02*(max(xs)-min(xs)),max(xs)+.02*(max(xs)-min(xs));ymin,ymax=min(ys)-.1,max(ys)+.1
 def pos(x,y):return 85+(x-xmin)/(xmax-xmin)*630,465-(y-ymin)/(ymax-ymin)*405
 svg=['<svg xmlns="http://www.w3.org/2000/svg" width="970" height="550"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12" fill="#1e293b"><text x="85" y="28" font-size="18">Global / Candidate / Full17 - live held-out graph search</text>']
 for x in np.linspace(xmin,xmax,6):
  xx,_=pos(x,ymin);svg.append(f'<path d="M{xx} 60V465" stroke="#e2e8f0"/><text x="{xx-18}" y="490">{x:.2f}</text>')
 for y in np.linspace(ymin,ymax,6):
  _,yy=pos(xmin,y);svg.append(f'<path d="M85 {yy}H715" stroke="#e2e8f0"/><text x="25" y="{yy+4}">{y:.2f}</text>')
 for i,group in enumerate(GROUPS):
  for seed in SEEDS:
   points=sorted([(v[metric],v['recall_pct']) for v in vals if v['group']==group and v['seed']==seed]);points=[pos(x,y) for x,y in points];svg.append(f'<polyline fill="none" stroke="{palette[group]}" stroke-opacity="0.65" stroke-width="1.7" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'"/>')
  svg.append(f'<path d="M750 {85+i*28}h20" stroke="{palette[group]}" stroke-width="3"/><text x="780" y="{89+i*28}">{group} (3 seeds)</text>')
 _,yy=pos(xmin,PRIMARY);svg.append(f'<path d="M85 {yy}H715" stroke="#475569" stroke-dasharray="5 4"/><text x="750" y="200">R_min={PRIMARY:.2f}%</text><text x="85" y="50">Recall@10 (%)</text><text x="85" y="530">{html.escape(xlabel)}</text></g></svg>');(F/filename).write_text(''.join(svg))
lines=['# Global / Candidate / Full17 特征消融','',f'主约束：Recall@10 ≥ {PRIMARY:.2f}%。每组3个训练seed、每个选中工作点5轮随机交错计时。模型训练、审计和profile均先于正式计时结束。','',f'I/O最低的组（要求3个seed的评估点都满足约束）：{bestio or "无满足条件的组"}。计入推理后实测平均延迟最低的组：{bestlat or "无满足条件的组"}。',f'延迟最低组与其他可行组的配对95%区间是否均排除零：{latency_supported}。若为False，只能报告实测最低，不能证明稳定延迟优势。','','|组|维数|参数量|平均Recall %|最低seed Recall %|SSD请求/q|profile模型开销 µs/q|含推理平均延迟 µs|p95 µs|全部seed达标|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for g in groupsummary:lines.append(f'|{g["group"]}|{g["n_features"]}|{g["parameters"]}|{g["mean_test_recall_pct"]:.3f}|{g["min_seed_test_recall_pct"]:.3f}|{g["reads_per_query"]:.3f}|{g["model_profile_us_per_query"]:.3f}|{g["end_to_end_mean_us"]:.3f}|{g["end_to_end_p95_us"]:.3f}|{g["all_seeds_test_meet_R_min"]}|')
lines+=['','## 特征定义','','两组输入列互不重叠，合并覆盖全部17维。Candidate中的相对距离仍按原定义使用全局best/worst作分母，列互斥不等于信息统计独立。beam_position是候选在当前beam内的位置，放入Candidate组；beam_step是搜索进度，放入Global组。search_L和beam_width是固定上下文，在此配置中为常量。缓存标志作为Candidate输入，但命中节点沿用既有保护策略，不执行模型。','']
for group,cols in proto['groups'].items():lines.append('- '+group+': '+', '.join(cols['names']))
lines+=['','## 公平性与计时','','三组重训 d→128→64→1 MLP，训练894077条状态、8000条Query；验证和评估各1000条Query，互不交叉。标签沿用原始保守retain标签；本次消融原始剪枝MLP，关闭历史复用及二次风险评估。输入宽度改变首层参数量，隐藏层和优化器完全相同。标准化只拟合训练集，早停只看验证logloss。','','阈值候选只由验证标签分数确定，再通过真实验证图搜索选择满足绝对Recall下限的最低请求点。评估集不调阈值，是否仍满足约束逐seed记录。主选点以I/O最小化选择，报告该点的真实端到端延迟；这不是对所有可能阈值的延迟全局最优证明。','','相同SIFT1M索引、L=100、K=10、beam=4、线程4、BFS缓存10000。CPU固定到4个可用的不同物理核心；不能排除其他宿主任务干扰。所有方法使用同一原生程序。验证和曲线扫描以4进程并行，只比较确定性Recall与I/O，扫描阶段的延迟不用于性能比较；profile和正式计时均串行执行。完整17维live state仍统一构建，再映射取子集输入；未对每组单独改写特征构造。这使结果反映模型输入消融，而非各组手工系统优化。','','无profile的正式计时包含state构造、输入映射、标准化、MLP评分、距离计算、实际读取和搜索bookkeeping；不包含模型/索引加载、预热和结果文件写出。单独profile用于记录模型区域开销，存在计时器扰动，不能与无profile总延迟直接相减。SSD请求是reader提交请求数，不是物理设备读盘计数。','','三组3个seed按seed配对，5轮计时按轮配对、相同Query按ID配对；2000次嵌套bootstrap区间保留这些层级，配对区间未作多重比较校正。只有3个训练seed，且评估集以前已被观察，本轮为探索性比较，不是全新确认性实验。','','## 产物','','`primary_group_comparison.csv`：主约束三组对比；`paired_group_differences.csv`：配对差值区间；`selected_recall_constraints.csv`：所有Recall下限的验证选点及评估达标情况；`recall_io_curves.csv`：全部真实曲线；`all_run_summary.csv`：所有原始运行汇总；`profile/`：单独推理成本profile；`timing_r0`至`timing_r4`：正式计时及Top10；`native_audit.json`：概率映射核验；`models/`：每组每seed的权重、训练过程、阈值。','','复现：`.venv/bin/python experiments/feature_group_ablation/run.py`。输入改变时使用新目录。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
dump(R/'verification.json',dict(complete=True,native_audit=json.loads((R/'native_audit.json').read_text()),all_final_Recall_independently_recomputed=True,tau0_ids_counts_equal_DiskANN=True,all_models_same_BFS_cache_members=True,profile_timing_ids_match_uninstrumented=True,all_repeated_ids_and_search_counts_deterministic=True,model_calls_minus_skips_equals_reads=True,exclusive_profile_components_sum_correct=True,curve_runs=len(curves),dedicated_timing_runs=len(chosen)*5,profile_runs=len(chosen),training_seeds=SEEDS,source_inputs=json.loads((R/'INPUT_LOCK.json').read_text())))
dump(R/'PROVENANCE.json',{str(p):sha(p) for p in [DATA/'manifest.json',DATA/'train/pre_io_state.npy',DATA/'train/y.npy',DATA/'train/query_ids.npy',DATA/'validation/query_ids.npy',DATA/'test/query_ids.npy',E1/'native/collect',E1/'PROVENANCE.json']})
artifacts={str(f.relative_to(R)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(R.rglob('*')) if f.is_file() and '__pycache__' not in str(f) and f.name not in ['artifact_manifest.json','COMPLETE.json','run.log','experiment.log','collect.log']};dump(R/'artifact_manifest.json',artifacts);dump(R/'COMPLETE.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results_sha256=sha(R/'results.json'),verification_sha256=sha(R/'verification.json'),manifest_sha256=sha(R/'artifact_manifest.json')))
log('FEATURE GROUP ABLATION COMPLETE')
