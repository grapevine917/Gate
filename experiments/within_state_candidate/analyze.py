from common import *
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import html
log('VERIFY exhaustive single-node contributions and compute within-state rankings')
r=dict(np.load(R/'records.npz'));p=dict(np.load(R/'pairs.npz'));l=p['left'].astype(int);rr=p['right'].astype(int);n=len(r['node_id']);qrow=r['query_row'].astype(int);proto=json.loads((R/'PROTOCOL.json').read_text());idx=np.load(R/'intervention_records.npy');a=np.genfromtxt(R/'intervention.csv',delimiter=',',names=True);sham=np.genfromtxt(R/'sham.csv',delimiter=',',names=True);shidx=np.load(R/'sham_records.npy')
assert len(a)==len(idx) and np.array_equal(a['request'],np.arange(len(idx)));assert np.array_equal(a['query_row'],qrow[idx]) and np.array_equal(a['node'],r['node_id'][idx]) and np.array_equal(a['step'],r['carrier_step'][idx]);assert np.all(a['applied']==1) and np.all(a['original_skip']==0) and np.all(a['replacement_skip']==1);assert np.allclose(a['original_score'],r['native_scores'][idx,2,0],rtol=0,atol=2e-6)
with (E1/'data/test.gt').open('rb') as f:shape=np.fromfile(f,np.uint32,2);gt=np.fromfile(f,np.uint32,int(shape.prod())).reshape(tuple(shape))
base=np.fromfile(R/'intervention.baseline.ids',np.uint64).reshape(1000,10);after=np.fromfile(R/'intervention.ids',np.uint64).reshape(len(a),10);hits=lambda ids,g:(ids[:,:,None]==g[:,None,:10]).any(2).sum(1);bh=hits(base,gt);ah=hits(after,gt[a['query_row'].astype(int)]);assert np.array_equal(ah,a['after_hits']) and np.array_equal(bh[qrow[idx]],a['baseline_hits']);assert sha(R/'intervention.baseline.ids')==sha(FG/'audit/DiskANN.ids')
assert np.array_equal(sham['node'],r['node_id'][shidx]) and np.all(sham['applied']==1) and np.all(sham['same_ids']==1)
for x,y in [('baseline_hits','after_hits'),('baseline_reads','after_reads'),('baseline_expanded','after_expanded'),('baseline_neighbor_cmps','after_neighbor_cmps')]:assert np.array_equal(sham[x],sham[y]),x
assert np.array_equal(np.fromfile(R/'sham.ids',np.uint64).reshape(-1,10),base[sham['query_row'].astype(int)])
for name in ['sham','intervention']:assert sha(R/(name+'.cache.bin'))==sha(FG/'audit/DiskANN.cache.bin')
lost=np.zeros(n,int);saving=np.zeros(n,int);exp_saved=np.zeros(n,int);cmps_saved=np.zeros(n,int);lost[idx]=bh[qrow[idx]]-ah;saving[idx]=a['baseline_reads'].astype(int)-a['after_reads'].astype(int);exp_saved[idx]=a['baseline_expanded'].astype(int)-a['after_expanded'].astype(int);cmps_saved[idx]=a['baseline_neighbor_cmps'].astype(int)-a['after_neighbor_cmps'].astype(int);harm=(lost>0)&~r['cached'];safe=(lost<=0)&(saving>0)&~r['cached']
np.savez_compressed(R/'contributions.npz',intervention_records=idx,lost_hits=lost,SSD_saved=saving,expanded_saved=exp_saved,neighbor_cmps_saved=cmps_saved,harmful_skip=harm,safe_positive_IO=safe,has_intervention=~r['cached'])
assert np.array_equal(qrow[l],qrow[rr]) and np.array_equal(r['beam_step'][l],r['beam_step'][rr]);assert np.all(~r['cached'][l]&~r['cached'][rr]);assert np.array_equal(r['state'][l,7],r['state'][rr,7])
gcols=GROUPS['Global'];assert np.array_equal(r['canonical_state'][l][:,gcols],r['canonical_state'][rr][:,gcols]);assert not p['global_exact'].any()
logits=dict(np.load(R/'logit_scores.npz'));assert np.array_equal(logits['natural'][:,1,:],logits['canonical'][:,1,:]);assert np.array_equal(r['canonical_scores'][l,0,:],r['canonical_scores'][rr,0,:]);assert np.array_equal(logits['canonical'][l,0,:],logits['canonical'][rr,0,:])
# Only 11 Global columns are matched, without matching away the Candidate information under test.
other=[j for j in gcols if j!=5];assert np.array_equal(r['state'][l][:,other],r['state'][rr][:,other]);assert p['global_distance'].max()<=proto['global_distance_limit']+1e-8
query=p['query_row'].astype(int);beamkey=query.astype(np.uint64)*100000+r['beam_step'][l];_,beamix=np.unique(beamkey,return_inverse=True);nbeams=int(beamix.max()+1);P=len(l)
targets={'admitted_count':r['admitted'].astype(float),'admission_positive':(r['admitted']>0).astype(float),'conservative_retain':r['conservative_retain'].astype(float),'lost_hits':lost.astype(float),'harmful_skip':harm.astype(float),'safe_positive_IO':safe.astype(float)}
types={k:('graded_pairwise_accuracy' if k in ['admitted_count','lost_hits'] else 'within_beam_conditional_AUC') for k in targets}
# Query and matched seed resampling is shared by every model/target comparison.
rng=np.random.default_rng(SEED);B=1000;qw=np.empty((B,1000),float);sw=np.empty((B,3),float)
for b in range(B):qw[b]=np.bincount(rng.integers(0,1000,1000),minlength=1000);sw[b]=np.bincount(rng.integers(0,3,3),minlength=3)
phases=[('all',np.ones(P,bool))]+[(f'phase{i}',p['phase']==i) for i in range(5)];stage_ranges=[];cuts=proto['phase_cuts_train_only']
for i in range(5):stage_ranges.append(dict(phase=f'phase{i}',lower_exclusive=None if i==0 else cuts[i-1],upper_inclusive=cuts[i] if i<4 else None))
dump(R/'phase_definitions.json',stage_ranges)
model_names=list(GROUPS)+['PQ_distance','Queue_rank'];metrics=[];seedmetrics=[];differences=[];support=[];auc_unconditional=[]
# Pair support is determined by contribution labels, never by model scores.
for phase,pm in phases:
 for target,y in targets.items():
  dy=y[l]-y[rr];eligible=pm&(dy!=0);denq=np.bincount(query,weights=eligible,minlength=1000);denbeam=np.bincount(beamix,weights=eligible,minlength=nbeams);bootden=3*(qw@denq);valid=bootden>0;count=int(eligible.sum());support.append(dict(phase=phase,target=target,total_pairs=int(pm.sum()),eligible_pairs=count,target_tie_pairs=int(np.sum(pm&~eligible)),queries=int(np.sum(denq>0)),beams=int(np.sum(denbeam>0)),pair_weighting='all unequal-target within-beam pairs; not pooled cross-state AUC'))
  for semantics in ['natural_near','canonical_same_global']:
   for basis in ['native_probability','unsaturated_logit']:
    scores=(r['native_scores'] if semantics=='natural_near' else r['canonical_scores']) if basis=='native_probability' else logits['natural' if semantics=='natural_near' else 'canonical'];numerators={};bootvalues={};points={}
    for gi,model in enumerate(model_names):
     if model=='PQ_distance':s=np.repeat((-r['state'][:,0,None]),3,axis=1)
     elif model=='Queue_rank':s=np.repeat((-r['state'][:,3,None]),3,axis=1)
     else:s=scores[:,gi,:]
     if target=='safe_positive_IO':s=-s
     ds=s[l]-s[rr];success=((ds*np.sign(dy[:,None]))>0).astype(float)+.5*(ds==0);num=np.array([np.bincount(query,weights=success[:,seed]*eligible,minlength=1000) for seed in range(3)]);numerators[model]=num
     if count:
      bootnum=((qw@num.T)*sw).sum(1);bv=bootnum[valid]/bootden[valid];value=float(num.sum()/(3*count));lo,hi=map(float,np.percentile(bv,[2.5,97.5]));qmacro=float(np.mean(num[:,denq>0]/denq[denq>0]));bm=[]
      for seed in range(3):
       nb=np.bincount(beamix,weights=success[:,seed]*eligible,minlength=nbeams);bm.append(float(np.mean(nb[denbeam>0]/denbeam[denbeam>0])))
      bmacro=float(np.mean(bm));tie=float(np.sum((ds==0)*eligible[:,None])/(3*count))
     else:value=lo=hi=qmacro=bmacro=tie=None;bv=None
     points[model]=value;bootvalues[model]=bv
     metrics.append(dict(semantics=semantics,score_basis=basis,phase=phase,target=target,metric=types[target],model=model,pairs=count,queries=int(np.sum(denq>0)),beams=int(np.sum(denbeam>0)),value=value,lo=lo,hi=hi,query_macro=qmacro,beam_macro=bmacro,prediction_tie_fraction=tie,score_direction='lower retain favors U' if target=='safe_positive_IO' else 'higher retain favors contribution'))
     if model in GROUPS:
      for seed in range(3):seedmetrics.append(dict(semantics=semantics,score_basis=basis,phase=phase,target=target,model=model,seed=SEEDS[seed],pairs=count,value=float(num[seed].sum()/count) if count else None,prediction_tie_fraction=float(np.sum((ds[:,seed]==0)&eligible)/count) if count else None))
    for first,second in [('Full17','Global'),('Full17','Candidate'),('Candidate','Global'),('Full17','PQ_distance'),('Full17','Queue_rank')]:
     if count:
      dv=bootvalues[first]-bootvalues[second];v=float(points[first]-points[second]);lo,hi=map(float,np.percentile(dv,[2.5,97.5]))
     else:v=lo=hi=None
     differences.append(dict(semantics=semantics,score_basis=basis,phase=phase,target=target,metric=types[target],first=first,second=second,pairs=count,queries=int(np.sum(denq>0)),difference=v,lo=lo,hi=hi))
log('WITHIN-STATE METRICS AND MATCHED MODEL DIFFERENCES COMPUTED')
csvsave(R/'pair_support.csv',support);csvsave(R/'within_state_metrics.csv',metrics);csvsave(R/'per_seed_metrics.csv',seedmetrics);csvsave(R/'paired_model_differences.csv',differences)
csvsave(R/'candidate_phase_metrics.csv',[v for v in metrics if v['model']=='Candidate']);csvsave(R/'natural_exact_support.csv',[dict(phase=phase,pairs=int(np.sum(mask&p['global_exact'])),metric=None,status='no actual exact Global-state pairs; NA, not 0 or .5') for phase,mask in phases])
# A conventional cross-state AUC is saved only as a diagnostic, never mislabeled within-state.
for phase,_ in phases:
 m=~r['cached']
 if phase!='all':m=m&(np.searchsorted(cuts,r['state'][:,7],side='left')==int(phase[-1]))
 for target in ['admission_positive','conservative_retain','harmful_skip','safe_positive_IO']:
  y=targets[target][m]
  for gi,model in enumerate(GROUPS):
   for si,seed in enumerate(SEEDS):
    s=r['native_scores'][m,gi,si];s=-s if target=='safe_positive_IO' else s;auc_unconditional.append(dict(phase=phase,target=target,model=model,seed=seed,candidates=int(m.sum()),AUROC=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,interpretation='cross-state descriptive AUROC; not main within-state metric'))
csvsave(R/'unconditional_auc_diagnostic.csv',auc_unconditional)
# Reviewable full records retain all 17 raw columns and all nine actual model scores.
rows=[]
for i in range(n):
 v=dict(record=i,query_id=int(r['query_id'][i]),query_row=int(qrow[i]),beam_step=int(r['beam_step'][i]),node_id=int(r['node_id'][i]),cached=int(r['cached'][i]),candidate_ordinal=int(r['carrier_step'][i]),phase=int(np.searchsorted(cuts,r['state'][i,7],side='left')),admitted_neighbors=int(r['admitted'][i]),fresh_neighbors=int(r['fresh'][i]),in_final_top10=int(r['in_final_top10'][i]),conservative_retain=int(r['conservative_retain'][i]),has_intervention=int(not r['cached'][i]),lost_hits=int(lost[i]) if not r['cached'][i] else None,SSD_saved=int(saving[i]) if not r['cached'][i] else None,harmful_skip=int(harm[i]) if not r['cached'][i] else None,safe_positive_IO=int(safe[i]) if not r['cached'][i] else None)
 v.update({f'state_{j}':float(r['state'][i,j]) for j in range(17)});v.update({f'{g}_s{seed}':float(r['native_scores'][i,gi,si]) for gi,g in enumerate(GROUPS) for si,seed in enumerate(SEEDS)});rows.append(v)
csvsave(R/'decision_records.csv',rows)
# Distances and candidate IDs suffice to reconstruct every conditional pair without arbitrary subsampling.
pairrows=[dict(pair=i,left_record=int(a),right_record=int(b),query_id=int(p['query_id'][i]),beam_step=int(p['beam_step'][i]),phase=int(p['phase'][i]),global_distance=float(p['global_distance'][i]),beam_position_gap=int(p['beam_gap'][i]),admitted_left=int(r['admitted'][a]),admitted_right=int(r['admitted'][b]),lost_hits_left=int(lost[a]),lost_hits_right=int(lost[b]),safe_IO_left=int(safe[a]),safe_IO_right=int(safe[b])) for i,(a,b) in enumerate(zip(l,rr))];csvsave(R/'candidate_pairs.csv',pairrows)
# Standalone phase plot of actual native-probability conditional AUC.
F=R/'figures';F.mkdir(exist_ok=True);colors={'Global':'#2563eb','Candidate':'#16a34a','Full17':'#dc2626'}
for target in ['admission_positive','harmful_skip','safe_positive_IO']:
 svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="460"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13" fill="#1e293b"><text x="80" y="28" font-size="17">Within-beam conditional AUC: {html.escape(target)}</text>']
 for y in [0,.25,.5,.75,1]:
  yy=370-y*300;svg.append(f'<path d="M80 {yy}H650" stroke="#e2e8f0"/><text x="28" y="{yy+4}">{y:.2f}</text>')
 for j in range(5):svg.append(f'<text x="{90+j*130}" y="395">Phase {j}</text>')
 for gi,model in enumerate(GROUPS):
  mm=[next(v for v in metrics if v['semantics']=='natural_near' and v['score_basis']=='native_probability' and v['phase']==f'phase{j}' and v['target']==target and v['model']==model) for j in range(5)];points=[]
  for j,v in enumerate(mm):
   if v['value'] is None:continue
   xx=110+j*130;yy=370-v['value']*300;points.append((xx,yy));lo=370-v['lo']*300;hi=370-v['hi']*300;svg.append(f'<path d="M{xx} {lo}V{hi}" stroke="{colors[model]}"/><circle cx="{xx}" cy="{yy}" r="4" fill="{colors[model]}"/>')
  svg.append(f'<polyline fill="none" stroke="{colors[model]}" stroke-width="2" points="'+' '.join(f'{x},{y}' for x,y in points)+'"/>');svg.append(f'<path d="M700 {90+gi*28}h20" stroke="{colors[model]}" stroke-width="3"/><text x="730" y="{94+gi*28}">{model}</text>')
 svg.append('<text x="80" y="430">Stage boundaries: training-only completed-expansion quintiles</text></g></svg>');(F/f'phase_{target}.svg').write_text(''.join(svg))
summary=[v for v in metrics if v['semantics']=='natural_near' and v['score_basis']=='native_probability' and v['phase']=='all' and v['model'] in GROUPS];csvsave(R/'primary_summary.csv',summary)
dump(R/'results.json',dict(primary_summary=summary,primary_differences=[v for v in differences if v['semantics']=='natural_near' and v['score_basis']=='native_probability' and v['phase']=='all'],queries=1000,records=n,beams=json.loads((R/'PREPARED.json').read_text())['beams'],interventions=len(a),sham_searches=len(sham),paired_miss_candidates=len(l),actual_exact_pairs=0,canonical_control='fixed true beam-start Global state, offline rescoring only',contribution_labels='direct admissions plus exhaustive true single-node Recall/IO effects on common all-Keep baseline',all_metrics_csv='within_state_metrics.csv',phase_metrics_csv='candidate_phase_metrics.csv'))
verification=dict(complete=True,all_native_scores_and_17D_states_joined_to_actual_baseline=True,all_models_share_same_query_node_beam_records=True,all_64_neighbor_edge_observations_joined_to_correct_parent=True,direct_admission_counts_match_original_zero_admission_labels=True,all_counterfactual_prefix_node_state_and_score_checked_by_native=True,every_single_intervention_applied_exactly_once=True,all_original_decisions_Keep=True,all_recall_independently_recomputed_from_saved_IDs_and_GT=True,sham_controls_identical_IDs_reads_expansions_neighbor_cmps=True,same_cache_members=True,pairs_never_cross_query_or_beam=True,all_Global_columns_except_unexpanded_count_exactly_matched=True,no_true_exact_Global_pairs_reported_as_NA=True,canonical_Global_inputs_and_scores_exactly_identical_within_beam=True,Candidate_unsaturated_scores_unchanged_by_Global_freeze=True,model_training_not_changed=True,phase_bins_fit_on_train_only=True,independent_queries_used_for_bootstrap=1000,matched_training_seeds=SEEDS,source_inputs=json.loads((R/'INPUT_LOCK.json').read_text()))
dump(R/'verification.json',verification)
readme='''# Within-State Candidate Discrimination

本实验的结果与逐样本记录已保存，原始模型不重新训练。

- `primary_summary.csv`：实际近似同状态、原生概率分数的主指标。
- `within_state_metrics.csv`：实际近似状态与固定Global对照、原生概率与未饱和logit、全部阶段及目标的完整指标和区间。
- `paired_model_differences.csv`：Full17/Global/Candidate及简单排序基线的配对差值。
- `candidate_phase_metrics.csv`、`phase_definitions.json`：Candidate-only分阶段结果及训练集确定的边界。
- `pair_support.csv`、`natural_exact_support.csv`：每项指标有效候选对、Query与Beam数量，目标相同而排除的对数。
- `decision_records.csv`、`records.npz`：全部候选的17维状态、模型分数和实际邻居贡献。
- `candidate_pairs.csv`、`pairs.npz`：全部同Query、同Beam非缓存候选对，未经结果筛选。
- `contributions.npz`：单节点反事实的命中损失、请求节省、展开变化与A类标签。
- `intervention.csv`、`intervention.ids`、`intervention.baseline.ids`：完整反事实计数与最终Top10；IDs为uint64，逐请求10个。
- `sham.*`：1000次Keep→Keep对照。
- `figures/phase_*.svg`：分阶段条件AUC图。
- `verification.json`：字段、贡献、前缀、结果和候选配对核验。

同一Beam内，实际Global状态只有unexpanded_count会随候选选择递减，其余10维完全一致。故实际数据属于“近似同状态”；完全相同Global状态的实际候选对没有支持，指标记NA。另将Global固定为真实Beam起始快照后离线重评分，构造严格同Global对照；它不是实际执行的新搜索。Global-only在该对照中平分是构造的必然结果，不是独立的模型劣势证据。完全相同Global输入每Beam只取一次评分并广播，避免BLAS分块浮点舍入产生虚假的候选排序；不使用标签进行此数值控制。原生float32概率可能饱和，额外保存未饱和logit排序，主指标仍用实际原生概率。

贡献包含真实处理顺序下入队邻居数，以及全部非缓存候选的独立单节点Skip最终Recall/I/O效应。前者受同Beam邻域重叠与处理次序影响，不能等同检索必要性；后者是单次删除的边际效应，不等同联合剪枝效果。缓存节点被原策略保护，仅记录、不进入主候选对或反事实分母。旧E4-1标签来自不同的已剪枝路径，本实验没有直接复用这些标签。

Pairwise Accuracy只比较贡献数不同的候选；条件AUC只比较同Beam正负标签候选，分数并列计0.5。条件AUC等于这类二值候选对的Pairwise Accuracy，不是额外独立证据。所有目标并列、无可比较候选的Beam/阶段均报告支持数量，不填造精确率。置信区间按1000个Query聚类，并配对重采样3个训练seed，不把候选对当作独立样本。Query宏平均与Beam宏平均另列，避免只有大Beam主导。

这些查询先前已被观察，本轮是探索性诊断；反事实GT和未来贡献不输入模型，不参与训练或匹配边界选择。模型采用上一特征消融实验的9个冻结权重，标签在所有模型共享的无剪枝轨迹上定义，不混合各自已剪枝后的轨迹。

复现：`.venv/bin/python experiments/within_state_candidate/run.py`。协议/模型/原生输入变化须使用新目录。原E1–E4及特征组消融产物不修改。
'''
(R/'README.md').write_text(readme)
dump(R/'PROVENANCE.json',{str(f):sha(f) for f in [DATA/'manifest.json',DATA/'test/metadata.npy',DATA/'test/pre_io_state.npy',DATA/'test/neighbors.npy',FG/'COMPLETE.json',FG/'PROTOCOL.json',*list((FG/'models').glob('*/model.bin'))]})
artifacts={str(f.relative_to(R)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(R.rglob('*')) if f.is_file() and '__pycache__' not in str(f) and f.name not in ['artifact_manifest.json','COMPLETE.json','run.log','experiment.log','analyze.log']};dump(R/'artifact_manifest.json',artifacts);dump(R/'COMPLETE.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results_sha256=sha(R/'results.json'),verification_sha256=sha(R/'verification.json'),manifest_sha256=sha(R/'artifact_manifest.json')))
log('WITHIN-STATE CANDIDATE DISCRIMINATION COMPLETE')
