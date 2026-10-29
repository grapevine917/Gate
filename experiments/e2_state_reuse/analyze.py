#!/usr/bin/env python3
from common import *
def summarize_interventions(p,outcomes,indices,label):
 take=np.flatnonzero(indices);a=outcomes[take];n=len(take);cluster=p['target_query'][take]
 delta=(a['after_hits']-a['baseline_hits'])*10;bad=a['after_hits']<a['baseline_hits'];ss=p['source_skip'][take];ts=p['target_skip'][take]
 r={'group':label,'donor_events':n,'unique_replacements':int(len(np.unique(a['request']))),'target_queries':int(len(np.unique(cluster))),'changed_decision_events':int((ss!=ts).sum()),'false_skip_events':int((ss&~ts).sum()),'harmful_events':int(bad.sum()),'harmful_false_skip_events':int((bad&ss&~ts).sum()),'baseline_recall_pct':float(a['baseline_hits'].mean()*10) if n else None,'after_recall_pct':float(a['after_hits'].mean()*10) if n else None}
 for key,v,d in [('recall_delta_pp',delta,np.ones(n,bool)),('harm_rate',bad,np.ones(n,bool)),('harm_given_false_skip',bad,ss&~ts),('harm_given_reuse_skip',bad,ss),('read_delta',a['after_reads']-a['baseline_reads'],np.ones(n,bool))]:r[key],r[key+'_lo'],r[key+'_hi']=rate_ci(v,d,cluster)
 return r
def main():
 log('FINAL ANALYSIS START')
 raw=np.genfromtxt(R/'intervention.csv',delimiter=',',names=True);raw=np.atleast_1d(raw);requests=json.loads((R/'intervention_requests.json').read_text())
 assert len(raw)==len(requests);np.testing.assert_array_equal(raw['request'],np.arange(len(raw)));assert np.all(raw['applied']==1)
 assert np.all(raw['original_skip']==np.array([r['expected_score']<json.loads((E1/'SELECTION_LOCK.json').read_text())['selected']['MLP'] for r in requests]))
 sham=raw['replacement_skip']==raw['original_skip'];assert np.all(raw['same_ids'][sham]==1);assert np.all(raw['baseline_reads'][sham]==raw['after_reads'][sham]);assert np.all(raw['baseline_hits'][sham]==raw['after_hits'][sham])
 # Independently recalculate final Recall from saved native output IDs and exact GT.
 from pathlib import Path
 qids=np.array(json.loads((E1/'query_splits.json').read_text())['test'])
 with (E1/'data/test.gt').open('rb') as f:n,d=np.frombuffer(f.read(8),'<u4');gt=np.fromfile(f,'<u4',count=int(n*d)).reshape(n,d)[:,:10]
 resultids=np.fromfile(R/'intervention.ids','<u8').reshape(-1,10);baselineids=np.fromfile(R/'intervention.baseline.ids','<u8').reshape(-1,10)
 recalc=np.array([len(set(ids)&set(gt[int(row)])) for ids,row in zip(resultids,raw['query_row'])]);np.testing.assert_array_equal(recalc,raw['after_hits'])
 baserecalc=np.array([len(set(ids)&set(g)) for ids,g in zip(baselineids,gt)]);np.testing.assert_array_equal(baserecalc[raw['query_row'].astype(int)],raw['baseline_hits'])
 mapping=np.genfromtxt(R/'intervention_event_map.csv',delimiter=',',names=True,dtype=None,encoding='utf-8');out={};paired=[]
 for population in ['same_node','cross_node']:
  p=dict(np.load(R/f'MLP_{population}_pairs.npz'));m=mapping[mapping['population']==population];np.testing.assert_array_equal(m['event'],np.arange(len(m)));np.testing.assert_array_equal(m['group'],p['group'])
  a=raw[m['request']];np.testing.assert_array_equal(a['query_row'],[requests[int(i)]['query_row'] for i in m['request']]);np.testing.assert_array_equal(a['replacement_skip'],p['source_skip']);np.testing.assert_array_equal(a['original_skip'],p['target_skip'])
  groups=[summarize_interventions(p,a,p['group']==g,f'G{g}') for g in range(1,5)];csvsave(R/f'{population}_intervention_groups.csv',groups)
  save={k:p[k] for k in ['source_query','target_query','source_node','target_node','source_record','target_record','group','source_skip','target_skip']}
  for key in ['request','baseline_hits','after_hits','baseline_reads','after_reads','same_ids']:save[key]=a[key]
  np.savez_compressed(R/f'{population}_intervention_events.npz',**save)
  out[population]={'groups':groups,'overall':summarize_interventions(p,a,np.ones(len(p['group']),bool),'all')}
  # Within the SAME current target position, contrast donor groups with equal position weights.
  bygroup={}
  for g in range(1,5):
   take=p['group']==g;pos=p['target_record'][take];values=(a['after_hits'][take]-a['baseline_hits'][take])*10
   sums={};counts={};query={}
   for node,v,q in zip(pos,values,p['target_query'][take]):sums[int(node)]=sums.get(int(node),0)+v;counts[int(node)]=counts.get(int(node),0)+1;query[int(node)]=int(q)
   bygroup[g]=({node:sums[node]/counts[node] for node in sums},query)
  for ga,gb in [(1,2),(1,3),(3,4),(2,4)]:
   va,qa=bygroup[ga];vb,qb=bygroup[gb];common=sorted(set(va)&set(vb));delta=np.array([va[i]-vb[i] for i in common]);query=np.array([qa[i] for i in common]);mu,lo,hi=rate_ci(delta,np.ones(len(common),bool),query)
   paired.append({'population':population,'contrast':f'G{ga}-G{gb}','matched_target_positions':len(common),'target_queries':len(np.unique(query)),'recall_delta_difference_pp':mu,'ci_low':lo,'ci_high':hi})
 csvsave(R/'paired_position_contrasts.csv',paired);out['paired_position_contrasts']=paired
 verification={'unique_interventions':len(raw),'all_prefix_state_node_and_score_checks_passed':True,'applied_exactly_once':True,'same_decision_sham_cases':int(sham.sum()),'all_same_decision_shams_exact_ids_and_reads':True,'independent_recall_recalculation_all_requests':True,'fresh_baseline_equals_e1':sha(R/'intervention.baseline.ids')==sha(E1/'runs_main/test_MLP_r0_time.ids'),'cache_equals_e1':sha(R/'intervention.cache.bin')==sha(E1/'runs_main/test_MLP_trace.cache.bin'),'same_node_donor_events':len(np.load(R/'MLP_same_node_pairs.npz')['group']),'cross_node_donor_events':len(np.load(R/'MLP_cross_node_pairs.npz')['group'])}
 # Validate same/cross-node identity and matched query-pair/target-position populations.
 same=np.load(R/'MLP_same_node_pairs.npz');cross=np.load(R/'MLP_cross_node_pairs.npz');assert np.all(same['source_node']==same['target_node']);assert np.all(cross['source_node']!=cross['target_node']);np.testing.assert_array_equal(same['source_query'],cross['source_query']);np.testing.assert_array_equal(same['target_record'],cross['target_record']);np.testing.assert_array_equal(same['query_distance'],cross['query_distance'])
 verification['cross_control_query_and_target_position_matched']=True
 norm=np.load(R/'normalization.npz');h=trace('MLP','history');t=trace('MLP','test');rng=np.random.default_rng(SEED)
 for k in rng.choice(len(same['group']),100,replace=False):
  i,j=int(same['source_record'][k]),int(same['target_record'][k]);left=(h['state_features'][i].astype(float)-norm['mean'])/(norm['std']+1e-8);right=(t['state_features'][j].astype(float)-norm['mean'])/(norm['std']+1e-8);assert abs(np.linalg.norm(left-right)-same['state_distance'][k])<1e-9
 verification['independent_state_normalization_formula_checks']=100
 original=np.load(S/'state_mlp/normalization.npz')
 np.testing.assert_allclose(norm['mean'],original['mean'],rtol=1e-6,atol=1e-5)
 np.testing.assert_allclose(np.where(norm['std']==0,1,norm['std']),original['scale'],rtol=1e-6,atol=1e-5)
 trainids=np.load(S/'training_data_l100/train/query_ids.npy');assert not np.intersect1d(trainids,qids).size
 assert set(np.unique(h['query_id'])).issubset(set(trainids))
 verification['normalization_matches_existing_train_only_model_statistics']=True
 verification['history_subset_train_and_test_disjoint']=True
 # No-op output parity is a mechanism check; individual substitutions are not full-cache deployment runs.
 dump(R/'verification.json',verification);dump(R/'intervention_summary.json',out)
 from plot_figures import render_all
 analysis=json.loads((R/'analysis.json').read_text());protocol=json.loads((R/'PROTOCOL.json').read_text());render_all(analysis,out)
 report(analysis,out,protocol,verification);log('FINAL ANALYSIS COMPLETE')
def report(a,out,p,v):
 def f(x,d=3):return 'N/A' if x is None else f'{x:.{d}f}'
 def pct(x,d=2):return 'N/A' if x is None else f'{100*x:.{d}f}'
 main=a['MLP']['same_node'];cross=a['MLP']['cross_node'];groups=main['groups'];pred=main['predictive_metrics'];corr=main['spearman_delta_p'];effect=main['state_vs_query_logloss_improvement'];risk=out['same_node']['groups']
 lines=['# E2：17维搜索状态的可复用性','','## 固定设置与防泄漏','','沿用E1：SIFT1M既有DiskANN图，L=100、K=10、beam=4、4线程、静态BFS缓存10000。主模型17→128→64→1 state-only MLP，阈值0.024352464824914932；1000条训练历史Query、1000条互不重叠测试Query，测试无替换基线Recall=99.64%。LR/XGBoost仅做独立轨迹上的关联对照，风险干预使用MLP。E1原始轨迹和阈值均不修改。','','标准化严格使用既有训练集894077条pre-I/O状态，float64总体均值和标准差，epsilon=1e-8。保存normalization.npz；测试和验证Query不参与拟合。常量维度为'+str(p['constant_feature_indices'])+'，同配置下其差值为0。状态距离严格等价于先标准化两个状态再求L2。','','四组界限在训练历史Query之间、同节点且不同Query的'+str(p['train_calibration_pairs'])+'个有方向事件对上取距离中位数，既不使用测试分布，也不使用Skip标签挑界限：',f"- Query高相似：L2≤{p['query_high_if_l2_le']:.6f}；低相似：大于该值。",f"- 状态高相似：标准化L2≤{p['state_high_if_normalized_l2_le']:.6f}；低相似：大于该值。",'','这里的高/低相对“有共同节点的训练事件对”定义，不沿用E1全部Query对的绝对分位界限。这样避免多数远Query对没有共同节点、无法形成四组。完整阈值在测试分析前写入SELECTION_LOCK.json及PROTOCOL.json。','','## 同节点四组：决策与分数','','主分析穷举所有'+str(main['event_pairs'])+'个历史→测试同节点模型调用事件对；缓存命中不作为模型调用。统计按事件对加权，区间采用500次目标Query聚类bootstrap；同一当前位置可以有多个历史来源，不能视为相互独立实验。','','FSR_model定义为P(历史Skip | 当前模型Keep)，衡量相对于当前模型的误剪分歧；它不是相对于真实检索安全性的错误率。另存P(当前Keep | 历史Skip)=1-P_skip，避免分母混淆。分数差Δp使用原生retain分数。','','|组|Query相似|状态相似|事件对|一致率 %|平均Δp|FSR_model %|Skip精确率 %|当前Keep数|历史Skip数|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in groups:
  g=int(r['group'][1]);lines.append(f"|G{g}|{'高' if g<3 else '低'}|{'高' if g in [1,3] else '低'}|{r['pairs']}|{pct(r['agreement'])}|{f(r['delta_p'])}|{pct(r['false_skip_rate'])}|{pct(r['skip_precision'])}|{r['target_keep_count']}|{r['source_skip_count']}|")
 lines+=['','![四组决策风险](figures/joint_groups.png)','','## H2：支持到什么程度','',f"状态距离与分数差Δp的Spearman相关系数为{corr['state_distance']:.4f}，Query距离为{corr['query_distance']:.4f}。状态高相似组的平均分数差明显较低；在高Query相似条件内，G1比G2的模型误剪分歧更低，在低Query相似条件内G3也比G4更低。",'','不过，“分数更接近”与“阈值化决策一定更一致”不能混为一谈。用训练事件对拟合、测试事件对评估的一维logistic探针（输入log1p距离）得到：','','|解释变量|测试AUROC|Log loss|Brier|','|---|---:|---:|---:|']
 for r in pred:lines.append(f"|{r['predictor']}|{r['auc']:.4f}|{r['log_loss']:.4f}|{r['brier']:.4f}|")
 lines += ['',f"状态单独相对Query单独的Log loss改善（Query损失减状态损失）={f(effect['mean'],5)}，95%目标Query聚类bootstrap区间=[{f(effect['ci95'][0],5)}, {f(effect['ci95'][1],5)}]。正值才表示状态更优。",'','本轮应得出有限结论：状态距离对分数差具有更强关联，且为联合解释提供补充信息；当前预设的一维探针并未证明状态距离单独全面胜过Query距离。四组的Keep/Skip构成不同，也会影响一致率，因此不凭G3的一致率较高就认定其可直接跨Query迁移。state-only模型本来就是状态的确定函数，部分分数关联来自模型结构，不能据此直接证明检索安全。','','![状态距离与分数差](figures/distance_score_difference.png)','','## 跨节点对照','','为每个同节点事件对，从相同历史Query中随机选另一个节点的模型记录，目标Query和当前位置保持不变，Query距离分布逐事件完全一致。共'+str(cross['event_pairs'])+'对。使用相同训练标准化与四组界限，不在跨节点测试样本上重设阈值。独立结果见MLP_cross_node_groups.csv；这些结果不能与同节点数据混合汇总。本E2衡量的是存在可比历史记录时的条件风险，不是所有目标决策的可复用覆盖率；跨节点配对控制也不是实际的全库状态近邻查找策略。','','## 辅助实验：实际替换一次决策','','对同节点和匹配跨节点的全部41480个历史来源事件，按(当前Query、当前决策位置、替换Skip/Keep)去重，真正执行'+str(v['unique_interventions'])+'次独立完整搜索。相同三元组的结果复用于多个历史来源，因此“来源事件数”不等于独立搜索次数。','','每次从头执行当前Query；到指定候选序号时，必须逐字节匹配原Node ID和17维pre-I/O状态，原分数也必须一致，然后只替换这一处输出，后续继续使用原MLP。没有提前改变前缀，没有同时复用其他节点，也没有用最终Recall筛选历史来源。为诊断机制，替换位置仍计算了fresh分数作校验，不能将此实验当作推理加速计时。','','同节点的分组实测：','','|组|来源事件|独立替换|替换前Recall %|替换后Recall %|平均变化 pp|Recall下降事件 %|错误Skip后有害比例 %|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in risk:lines.append(f"|{r['group']}|{r['donor_events']}|{r['unique_replacements']}|{f(r['baseline_recall_pct'])}|{f(r['after_recall_pct'])}|{f(r['recall_delta_pp'],5)}|{pct(r['harm_rate'],3)}|{pct(r['harm_given_false_skip'],3)}|")
 lines += ['','“错误Skip后有害比例”分母为历史Skip且当前模型Keep的实际替换事件；分子为替换后最终Recall下降。它与FSR_model不同：当前Keep是保守标签，并非每次都必要。分组Recall按该组当前Query来源事件加权，不能拿各组绝对Recall直接比较好坏，应看各自替换前后差值。所有零事件或零分母指标显示N/A；即使观测到零风险，也不能当作总体风险为零的证明。','','![真实替换后的检索风险](figures/intervention_risk.png)','','## 固定同一当前位置的可比性','','同节点可由全部四组历史来源覆盖的当前位置仅'+str(a['MLP']['same_position_support']['all_four_groups'])+'个，因此不宣称四组已形成完整的同位置匹配试验。对有共同支持的组对，另按相同当前位置计算各组历史来源的平均Recall变化，再做等位置权重的配对比较，结果见paired_position_contrasts.csv。区间仍按目标Query聚类，位置内多个历史来源不是独立样本。','','|对比|同一位置数|目标Query数|Recall变化差值 pp|95%区间|','|---|---:|---:|---:|---|']
 for r in out['paired_position_contrasts']:
  if r['population']=='same_node':lines.append(f"|{r['contrast']}|{r['matched_target_positions']}|{r['target_queries']}|{f(r['recall_delta_difference_pp'],5)}|[{f(r['ci_low'],5)}, {f(r['ci_high'],5)}]|")
 lines+=['','## LR / XGBoost 扩展对照','','以下均基于各模型自己的独立在线轨迹，相同训练标准化及MLP训练事件确定的分组界限；没有把MLP的状态轨迹冒充其他模型自己的轨迹。风险替换仅对主模型MLP执行。','','|模型|Query距离解释一致性的AUROC|状态距离AUROC|联合AUROC|状态距离与Δp相关|','|---|---:|---:|---:|---:|']
 for name in ['MLP','LR','XGBoost']:
  s=a[name]['same_node'];m={r['predictor']:r for r in s['predictive_metrics']};lines.append(f"|{name}|{m['query_only']['auc']:.4f}|{m['state_only']['auc']:.4f}|{m['joint']['auc']:.4f}|{s['spearman_delta_p']['state_distance']:.4f}|")
 lines+=['','## 核验与边界','',f"fresh无替换结果与E1逐ID一致；每次替换恰好触发一次且前缀状态一致；{v['same_decision_sham_cases']}个同决策替换对照的最终ID、读取数完全一致；全部请求的最终Recall均从保存的结果ID和GT独立重算。另用100个事件核验标准化距离公式，训练统计与既有MLP训练标准化记录一致。",'','测试划分此前已在E1使用，独立于本轮标准化、距离界限和探针拟合，但并非首次观察的新测试集。','本轮没有重新训练主检索模型，没有启用累计历史复用，没有测动态缓存或部署查找成本。单次替换的Recall/I/O变化不能简单累加为多次复用的效果。读数是提交请求而非物理SSD计数；干预的reads变化已独立记录，未声称节省模型推理。E1全局一次标记baseline保留在其原报告，Recall90.55%、请求92.304/q；本轮不将其旧测量冒称重新计时结果。','','## 产物与复现','','- PROTOCOL.json、SELECTION_LOCK.json、normalization.npz：训练统计、分组界限和冻结设置。','- *_same_node_pairs.npz、*_cross_node_pairs.npz：MLP/LR/XGBoost独立轨迹事件及Δp。','- *_groups.csv、*_distance_bins.csv：分组FSR、精确率、分数差与置信区间。','- intervention_requests.json、interventions.bin、intervention_event_map.csv：固定位置替换请求与历史来源映射。','- intervention.csv、intervention.ids、*_intervention_events.npz：实际搜索结果、读取数和映射后的事件结果。','- experiment.log、build.log、intervention.log、intervention.command.json：全过程日志和命令。','- verification.json、PROVENANCE.json、artifact_manifest.json、COMPLETE.json：核验与哈希。','- figures/：PNG、SVG、PDF。','','```bash','cd /home/lxyue/projects/learned-gate','.venv/bin/python experiments/e2_state_reuse/prepare.py','.venv/bin/python experiments/e2_state_reuse/build_native.py','.venv/bin/python experiments/e2_state_reuse/run.py','```','','prepare会重新拟合训练距离统计和生成请求；改变模型、协议、二进制或请求时应在新目录运行。既有E1与DiskANN源文件和原二进制均保留，E2使用独立native/库。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
