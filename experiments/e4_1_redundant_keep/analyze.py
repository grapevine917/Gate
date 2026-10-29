from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import json,csv,hashlib,datetime
R=Path(__file__).resolve().parent;E1=R.parent/'e1_similarity'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def savecsv(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
a=np.genfromtxt(R/'intervention.csv',delimiter=',',names=True,dtype=None,encoding=None);s=np.genfromtxt(R/'sham.csv',delimiter=',',names=True,dtype=None,encoding=None)
p=dict(np.load(R/'keep_positions.npz'));h=dict(np.load(E1/'MLP_history_trace.npz'));t=dict(np.load(E1/'MLP_test_trace.npz'));n=len(a);qr=a['query_row'].astype(int)
assert n==len(p['record']) and np.array_equal(a['request'],np.arange(n))
assert np.all(a['applied']==1) and np.all(a['original_skip']==0) and np.all(a['replacement_skip']==1)
assert np.array_equal(a['node'],p['node_id']) and np.array_equal(a['step'],p['search_step'])
assert np.allclose(a['original_score'],p['model_score'],rtol=0,atol=2e-6)
assert np.all(p['cache_hit']==0) and np.all(p['model_invoked'])
e1=np.genfromtxt(E1/'runs_main/test_MLP_r0_time.csv',delimiter=',',names=True)
for current,original in [('baseline_hits','hits'),('baseline_reads','reads'),('baseline_expanded','expansions'),('baseline_neighbor_cmps','pq_comparisons')]:
 assert np.array_equal(a[current],e1[original][qr]),current

ids=np.fromfile(R/'intervention.ids',np.uint64).reshape(n,10);base=np.fromfile(R/'intervention.baseline.ids',np.uint64).reshape(1000,10)
with (E1/'data/test.gt').open('rb') as f:
 shape=np.fromfile(f,np.uint32,2);gt=np.fromfile(f,np.uint32,int(shape.prod())).reshape(tuple(shape))
def hit(x,g):return (x[:,:,None]==g[:,None,:10]).any(2).sum(1)
bh=hit(base,gt);ah=hit(ids,gt[qr]);assert np.array_equal(bh[qr],a['baseline_hits']) and np.array_equal(ah,a['after_hits'])
assert np.all(s['same_ids']==1) and np.all(s['applied']==1)
for x,y in [('baseline_hits','after_hits'),('baseline_reads','after_reads'),('baseline_expanded','after_expanded'),('baseline_neighbor_cmps','after_neighbor_cmps')]:assert np.array_equal(s[x],s[y])
sids=np.fromfile(R/'sham.ids',np.uint64).reshape(-1,10);assert np.array_equal(sids,base[s['query_row'].astype(int)])
dh=ah-bh[qr];di=a['baseline_reads'].astype(int)-a['after_reads'].astype(int);de=a['baseline_expanded'].astype(int)-a['after_expanded'].astype(int);dc=a['baseline_neighbor_cmps'].astype(int)-a['after_neighbor_cmps'].astype(int)
cat=np.where(dh>=0,np.where(di>0,'A','B'),np.where(di>0,'C','D'));u=cat=='A';saving=np.maximum(0,di)*(dh>=0)
rng=np.random.default_rng(20261019);boot=rng.integers(0,1000,(1000,1000))
def metric(v,mask):
 den=np.bincount(qr[mask],minlength=1000);num=np.bincount(qr[mask],weights=np.asarray(v)[mask],minlength=1000);bd=den[boot].sum(1);bn=num[boot].sum(1);val=bn[bd>0]/bd[bd>0]
 return (float(num.sum()/den.sum()),*map(float,np.percentile(val,[2.5,97.5]))) if den.sum() else (None,None,None)
def summary(mask):
 row={'n':int(mask.sum()),'queries':int(len(np.unique(qr[mask])))}
 for k in 'ABCD':row[k]=int(np.sum(mask&(cat==k)));row[k+'_pct']=float(np.mean(cat[mask]==k)*100) if mask.any() else None
 for key,v in [('redundant_rate',u),('S_IO',saving),('mean_delta_IO',di),('mean_delta_expanded',de),('mean_delta_neighbor_cmps',dc),('mean_delta_recall_pp',dh*10)]:row[key],row[key+'_lo'],row[key+'_hi']=metric(v,mask)
 return row
allmask=np.ones(n,bool);overall=summary(allmask)
# Per-query values retain queries with zero Keep in the output and bootstrap frame.
query_rows=[]
for row in range(1000):
 m=qr==row;query_rows.append(dict(query_row=row,query_id=int(json.loads((E1/'query_splits.json').read_text())['test'][row]),baseline_hits=int(bh[row]),keep=int(m.sum()),A=int(u[m].sum()),S_IO=float(saving[m].mean()) if m.any() else None,redundant_rate=float(u[m].mean()) if m.any() else None))
savecsv(R/'per_query.csv',query_rows)
# History frequency counts do not use target-query outcomes.
frequency=np.bincount(h['node_id'][h['model_invoked']].astype(int),minlength=1000000);freq=frequency[p['node_id'].astype(int)]
hkeep=h['model_invoked'] & ~h['model_decision'].astype(bool)
names=['candidate_pq','best_pq','worst_pq','candidate_rank','queue_size','unexpanded_count','visited_count','completed_expansions','hops','candidate_ordinal','cache_hit','best_exact','has_exact','candidate_best_ratio','candidate_worst_ratio','L','beam']
bins=[]
def add(key,labels,masks,boundaries):
 for label,mask in zip(labels,masks):
  row=dict(variable=key,bin=str(label),boundaries=boundaries,**summary(mask))
  # Descriptive replication across fixed query halves; neither half is a fresh test set.
  for tag,half in [('half0',qr<500),('half1',qr>=500)]:
   m=mask&half;row[tag+'_n']=int(m.sum());row[tag+'_A_rate']=float(u[m].mean()) if m.any() else None
  bins.append(row)
def quantile_bins(key,x,cal):
 cuts=np.unique(np.quantile(cal,[.2,.4,.6,.8]));idx=np.searchsorted(cuts,x,side='left');add(key,range(len(cuts)+1),[idx==i for i in range(len(cuts)+1)],json.dumps(cuts.tolist()))
quantile_bins('retain_score',p['model_score'],h['model_score'][hkeep])
for j,name in enumerate(names):
 if j!=10:quantile_bins(name,p['state_features'][:,j],h['state_features'][hkeep,j])
add('history_node_frequency',['0','1','2-4','5+'],[freq==0,freq==1,(freq>=2)&(freq<=4),freq>=5],'fixed counts of history actual model invocations')
add('baseline_query_hits',['10/10','9/10','<=8/10'],[bh[qr]==10,bh[qr]==9,bh[qr]<=8],'post-hoc baseline quality, not online observable')
add('DRAM_cache_hit',['miss','hit'],[p['cache_hit']==0,p['cache_hit']!=0],'hits excluded by frozen miss-only model policy')
savecsv(R/'state_bins.csv',bins)
rows=[]
for i in range(n):rows.append(dict(request=i,query_row=int(qr[i]),query_id=int(p['query_id'][i]),step=int(p['search_step'][i]),node=int(p['node_id'][i]),retain_score=float(p['model_score'][i]),cache_hit=int(p['cache_hit'][i]),history_frequency=int(freq[i]),baseline_hits=int(bh[qr[i]]),after_hits=int(ah[i]),delta_recall_pp=int(dh[i])*10,delta_IO=int(di[i]),delta_expanded=int(de[i]),delta_neighbor_cmps=int(dc[i]),category=str(cat[i]),U=int(u[i])))
savecsv(R/'classified_interventions.csv',rows)
np.savez_compressed(R/'outcomes.npz',query_row=qr,category=cat,U=u,delta_hits=dh,delta_IO=di,delta_expanded=de,delta_neighbor_cmps=dc,baseline_ids=base,after_ids=ids)
# Candidate concentration factors are exploratory, with sample-size filters and query-half consistency.
candidates=[b for b in bins if b['variable'] not in ['L','beam','DRAM_cache_hit','baseline_query_hits'] and b['n']>=1000 and b['queries']>=100 and b['half0_n']>=300 and b['half1_n']>=300]
candidates.sort(key=lambda b:b['redundant_rate'],reverse=True)
result=dict(overall=overall,baseline_recall_pct=float(bh.mean()*10),baseline_reads_per_query=float(np.mean([a['baseline_reads'][np.flatnonzero(qr==i)[0]] for i in np.unique(qr)])),keep_positions=n,queries_with_A=int(np.sum(np.bincount(qr,weights=u,minlength=1000)>0)),total_positive_safe_IO_single_interventions=int(saving.sum()),delta_IO_range=[int(di.min()),int(di.max())],delta_hits_range=[int(dh.min()),int(dh.max())],top_exploratory_bins=candidates[:10],mean_IO_saved_given_A=float(di[u].mean()) if u.any() else None,mean_expanded_saved_given_A=float(de[u].mean()) if u.any() else None,mean_neighbor_cmps_saved_given_A=float(dc[u].mean()) if u.any() else None,cache_hit_candidates_not_scored=int(np.sum(t['cache_hit'])),keep_confidence='native retain score; not calibrated probability of retrieval necessity')
dump(R/'results.json',result)
verification=dict(all_keep_positions_exhaustive=True,baseline_hits_reads_expansions_comparisons_equal_e1=True,positions=n,all_original_keep=True,all_forced_skip=True,prefix_node_state_score_checked_by_native=True,exactly_one_intervention=True,all_final_recall_recomputed_from_ids_gt=True,baseline_ids_equal_e1=sha(R/'intervention.baseline.ids')==sha(E1/'runs_main/test_MLP_r0_time.ids'),sham_controls=len(s),sham_ids_reads_expansion_neighbor_cmps_identical=True,all_four_categories_sum_to_keep=sum(overall[k] for k in 'ABCD')==n,source_hashes={str(f):sha(f) for f in [E1/'MLP_test_trace.npz',E1/'MLP_history_trace.npz',E1/'SELECTION_LOCK.json',E1/'data/test.gt',R/'intervene.cpp',R/'intervene',R/'interventions.bin']})
assert verification['baseline_ids_equal_e1'];dump(R/'verification.json',verification)
F=R/'figures';F.mkdir(exist_ok=True)
# Standalone SVG chart, requiring no optional plotting package.
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="380" viewBox="0 0 1100 380"><rect width="1100" height="380" fill="white"/><g font-family="sans-serif" font-size="13" fill="#1e293b">']
panels=[('Outcome (% of original Keeps)',[(k,overall[k+'_pct']) for k in 'ABCD']),('A rate by retain score (low to high)',[(b['bin'],b['redundant_rate']*100) for b in bins if b['variable']=='retain_score']),('A rate by completed expansions',[(b['bin'],b['redundant_rate']*100) for b in bins if b['variable']=='completed_expansions'])]
for pi,(title,vals) in enumerate(panels):
 x0=pi*365+35;svg.append(f'<text x="{x0}" y="30">{title}</text>')
 for level in [0,25,50,75,100]:
  y=320-level*2.5;svg.append(f'<path d="M{x0} {y}h300" stroke="#e2e8f0"/><text x="{x0-27}" y="{y+4}" font-size="10">{level}</text>')
 for j,(label,value) in enumerate(vals):
  x=x0+15+j*(280/len(vals));w=190/len(vals);height=value*2.5
  svg.append(f'<rect x="{x}" y="{320-height}" width="{w}" height="{height}" fill="#2563eb"/><text x="{x}" y="{310-height}" font-size="11">{value:.1f}%</text><text x="{x}" y="340">{label}</text>')
svg.append('</g></svg>');(F/'redundant_keep.svg').write_text(''.join(svg))
lines=['# E4-1：寻找模型遗漏的可剪枝节点','',f'固定 SIFT1M / L=100 / K=10 / beam=4 / 4线程 / BFS缓存10000，沿用E1冻结MLP与阈值；测试1000条Query，穷举全部 {n:,} 个实际模型Keep。原始Recall@10={bh.mean()*10:.3f}%。','',f'**冗余Keep比例 R_redundant={overall["redundant_rate"]*100:.3f}%**，目标Query聚类bootstrap 95%区间 [{overall["redundant_rate_lo"]*100:.3f}, {overall["redundant_rate_hi"]*100:.3f}]%。',f'**S_IO={overall["S_IO"]:.6f} 请求/原始Keep**，95%区间 [{overall["S_IO_lo"]:.6f}, {overall["S_IO_hi"]:.6f}]。它是所有单节点实验中无Recall下降的正向请求节省之和除以Keep总数；不是同时剪枝后的节省。','', '|类别|节点数|占Keep %|含义|','|---|---:|---:|---|']
for k,meaning in zip('ABCD',['Recall无下降、请求减少','Recall无下降、请求不减少','Recall下降、请求减少','Recall下降、请求不减少']):lines.append(f'|{k}|{overall[k]:,}|{overall[k+"_pct"]:.3f}|{meaning}|')
# Interpret representative pre-specified axes before the exploratory ranking.
lines+=['','A类节点平均单次减少 '+f'{di[u].mean():.4f}'+' 个SSD请求、'+f'{de[u].mean():.4f}'+' 个展开节点，以及 '+f'{dc[u].mean():.2f}'+' 次原生邻居距离比较。1000条Query均存在A类Keep。','','## 模型是否系统性过于保守','','当前数据支持：后期搜索、候选队列靠后和未展开候选较少的状态，存在集中的冗余Keep。以训练历史定义边界后，在测试中已完成展开数>85的桶，A类为97.980%；候选rank>72的桶，A类为97.964%；未展开候选≤26的桶，A类为97.727%。这些模式在固定Query前后两个半区都出现，但仍需独立数据与累计干预验证。','','Keep分数仍包含风险区分信息，不能解释成模型完全失效：最高分桶A类仅51.714%，C+D=43.442%；最低分Keep桶A类93.901%，C+D=0.063%。原阈值偏保守的现象，尤其值得在低分Keep及后期队列状态下继续挖掘。','','|原生retain分数范围|Keep数|A比例 %|Recall下降比例 C+D %|S_IO|','|---|---:|---:|---:|---:|']
score_bins=[b for b in bins if b['variable']=='retain_score'];edges=json.loads(score_bins[0]['boundaries']);tau=json.loads((R/'PROTOCOL.json').read_text())['tau'];edges=[tau]+edges+[1.0]
for j,b in enumerate(score_bins):lines.append(f'|{edges[j]:.5f}–{edges[j+1]:.5f}|{b["n"]}|{b["A_pct"]:.3f}|{b["C_pct"]+b["D_pct"]:.3f}|{b["S_IO"]:.4f}|')
lines+=['','|已完成展开数|Keep数|A比例 %|Recall下降比例 C+D %|S_IO|','|---|---:|---:|---:|---:|']
for label,b in zip(['≤25','26–45','46–65','66–85','>85'],[b for b in bins if b['variable']=='completed_expansions']):lines.append(f'|{label}|{b["n"]}|{b["A_pct"]:.3f}|{b["C_pct"]+b["D_pct"]:.3f}|{b["S_IO"]:.4f}|')
lines+=['','节点历史访问频率的0次、1次、2–4次组A类分别为84.069%、84.796%、83.754%，没有明显单调关系；≥5次组仅75个Keep，不能据此作稳定高频结论。原始Recall难度≤8/10组仅164个Keep，样本很少，其差异也不能作为稳定难度效应。','','## 状态集中性（探索性）','','分桶边界来自1000条训练历史Query的实际Keep状态五分位；频率使用历史实际模型调用次数。保留全部17维状态分桶，包括常量维度。置信区间按目标Query聚类，不将同Query干预视为独立样本。以下列出样本≥1000、Query≥100且两个固定Query半区样本均≥300的高A比例桶；多重探索未作显著性校正，不能直接解释为因果或安全部署规则。','','|变量|桶（从0、低到高）|Keep数|Query数|A比例 %|S_IO|半区0 / 半区1 A比例 %|','|---|---:|---:|---:|---:|---:|---|']
for b in candidates[:10]:lines.append(f'|{b["variable"]}|{b["bin"]}|{b["n"]}|{b["queries"]}|{b["redundant_rate"]*100:.3f}|{b["S_IO"]:.4f}|{b["half0_A_rate"]*100:.2f} / {b["half1_A_rate"]*100:.2f}|')
lines+=['','具体边界、四类占比、平均Recall变化、展开节点变化与邻居距离计算变化见 `state_bins.csv`。`retain_score`为原生retain分数，不是“该节点检索必要性”的已校准概率。搜索进度采用当前已完成展开数和候选序号；队列状态涵盖候选rank、队列长度、未展开数、visited数和距离比值。','', '## 实验语义与核验','','每次从头运行原始Query，指定候选序号处逐字节核对17维pre-I/O状态及Node ID，并核对fresh分数，再只将该位置Keep改成Skip；后续照原模型完成整个图搜索。正常搜索和每次反事实Top10均保存，逐条从GT独立重算命中数。原始结果及固定缓存成员与E1完全一致。另执行1000次Keep→Keep sham，Top10、SSD请求、展开数和邻居距离计算数均完全一致。','', '展开节点数使用原生 n_expanded；邻居距离计算数使用 n_cmps（包含原生实现计入的距离比较，非唯一邻居数），不是把hop当成展开节点。请求数为提交的SSD请求，不代表设备物理读取。','', f'DRAM缓存命中候选共 {int(np.sum(t["cache_hit"])):,} 个，在冻结E1/E2策略中直接保留且不评分，因此不是“模型预测Keep”，不进入本实验分母；DRAM hit桶显示无样本，不能据此推断其冗余率。','', '当前Recall难度按原始Query的10/10、9/10、≤8/10命中分组，仅为事后诊断；不可作为在线输入。节点频率也只衡量固定历史库中的调用次数，不是实际线上流量频率。','', '这是已经观察过的测试集上的穷举描述性实验；没有训练或改变模型。U=1仅代表本次单节点干预没有降低Recall，不代表最终结果ID不变，也不证明在其他Query或累计剪枝时安全。各单节点节省不能直接叠加成新策略的收益。下一步若据分桶训练纠错策略，应冻结规则并在独立Query集做真实累计剪枝验证。','','## 产物与复现','','`classified_interventions.csv`：逐位置四类标签、Recall/IO/展开变化；`intervention.csv`：原始统计；`intervention.ids`与`intervention.baseline.ids`：完整最终Top10；`keep_positions.npz`：原始状态与分数；`per_query.csv`：逐Query统计；`results.json`、`verification.json`：汇总与核验；`figures/`：独立SVG图。','','运行：`.venv/bin/python experiments/e4_1_redundant_keep/run.py`。独立链接E2已核验的干预静态库，不修改E1/E2产物或DiskANN源代码。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
artifacts={str(f.relative_to(R)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(R.rglob('*')) if f.is_file() and f.name not in ['artifact_manifest.json','COMPLETE.json','analysis.log']};dump(R/'artifact_manifest.json',artifacts);dump(R/'COMPLETE.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verification_sha256=sha(R/'verification.json'),report_sha256=sha(R/'REPORT.md')))
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
