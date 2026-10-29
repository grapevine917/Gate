#!/usr/bin/env python3
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv
import numpy as np
from scipy import sparse,stats
R=Path(__file__).resolve().parent;D=R/'data';OUT=R/'figures';OUT.mkdir(exist_ok=True)
SEED=20261018
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def csvsave(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csr(t,ids,skip=False):
 lookup={int(q):i for i,q in enumerate(ids)};take=t['model_invoked'].astype(bool)
 if skip:take&=t['model_decision']!=0
 rows=np.array([lookup[int(q)] for q in t['query_id'][take]])
 return sparse.csr_matrix((np.ones(take.sum(),np.int32),(rows,t['node_id'][take])),shape=(len(ids),1000000))
def metrics(h,t,hids,tids):
 hv,hs=csr(h,hids),csr(h,hids,True);tv,ts=csr(t,tids),csr(t,tids,True)
 common=(hv@tv.T).toarray();source_skip=(hs@tv.T).toarray();target_skip=(hv@ts.T).toarray();both=(hs@ts.T).toarray()
 equal=common-source_skip-target_skip+2*both;target_count=np.asarray(tv.sum(axis=1)).ravel()
 assert np.all(equal>=0)&np.all(equal<=common)&np.all(both<=source_skip)
 coverage=np.divide(source_skip,target_count[None,:],out=np.zeros_like(source_skip,dtype=float),where=target_count[None,:]>0)
 agreement=np.divide(equal,common,out=np.full_like(common,np.nan,dtype=float),where=common>0)
 precision=np.divide(both,source_skip,out=np.full_like(common,np.nan,dtype=float),where=source_skip>0)
 agreed_skip_coverage=both/np.maximum(target_count[None,:],1)
 return dict(common=common,source_skip=source_skip,target_skip=target_skip,both_skip=both,equal=equal,target_count=target_count,coverage=coverage,agreement=agreement,skip_precision=precision,agreed_skip_coverage=agreed_skip_coverage)
def l2(a,b):return np.sqrt(np.maximum((a.astype(float)**2).sum(1)[:,None]+(b.astype(float)**2).sum(1)[None,:]-2*a.astype(float)@b.astype(float).T,0))
def boot_ci(values,mask):
 good=mask&np.isfinite(values);s=np.where(good,values,0).sum(axis=0);n=good.sum(axis=0)
 rng=np.random.default_rng(SEED);idx=rng.integers(0,len(s),(500,len(s)))
 den=n[idx].sum(1);v=s[idx].sum(1)/np.maximum(den,1)
 return np.percentile(v[den>0],[2.5,97.5]).tolist() if np.any(den>0) else [None,None]
def aggregate(m,mask,label,distance):
 n=int(mask.sum());common=m['common'];ss=m['source_skip']
 row={'group':label,'pairs':n,'mean_l2':float(distance[mask].mean()),'no_common_fraction':float(np.mean(common[mask]==0)),'no_reusable_skip_fraction':float(np.mean(ss[mask]==0)),'mean_common_nodes':float(common[mask].mean()),'mean_source_skip_common':float(ss[mask].mean())}
 for key in ['agreement','skip_precision','coverage','agreed_skip_coverage']:
  v=m[key];valid=mask&np.isfinite(v);row[key]=float(v[valid].mean()) if valid.any() else None
  row[key+'_valid_pairs']=int(valid.sum());ci=boot_ci(v,mask);row[key+'_ci_low'],row[key+'_ci_high']=ci
 denom=common[mask].sum();row['pooled_agreement']=float(m['equal'][mask].sum()/denom) if denom else None
 denom=ss[mask].sum();row['pooled_skip_precision']=float(m['both_skip'][mask].sum()/denom) if denom else None
 row['all_retain_agreement']=float(((common-m['target_skip'])[mask].sum())/common[mask].sum()) if common[mask].sum() else None
 return row
def groups(m,dist):
 edges=np.quantile(dist,[0,.01,.05,.20,.50,1]);labels=['0–1%','1–5%','5–20%','20–50%','50–100%'];rows=[]
 for j,label in enumerate(labels):
  mask=(dist>=edges[j])&((dist<edges[j+1]) if j<4 else (dist<=edges[j+1]));rows.append(aggregate(m,mask,label,dist))
 ranks=np.argsort(np.argsort(dist,axis=0,kind='stable'),axis=0,kind='stable')+1
 near=[]
 for lo,hi in [(1,1),(2,5),(6,10),(11,50),(51,200),(201,len(dist))]:
  mask=(ranks>=lo)&(ranks<=hi);near.append(aggregate(m,mask,f'rank {lo}–{hi}',dist))
 perquery=[]
 # Exactly the requested distance-percentile bins within EACH target query.
 n=len(dist)
 for lo,hi,label in [(0,.01,labels[0]),(.01,.05,labels[1]),(.05,.20,labels[2]),(.20,.50,labels[3]),(.50,1,labels[4])]:
  mask=(ranks>round(lo*n))&(ranks<=round(hi*n));perquery.append(aggregate(m,mask,label,dist))
 return edges,rows,near,perquery
def main():
 splits=json.loads((R/'query_splits.json').read_text());q=np.load(D/'query_vectors.npy');hids=np.array(splits['history']);tids=np.array(splits['test']);dist=l2(q[hids],q[tids])
 assert not set(map(bytes,q[hids]))&set(map(bytes,q[tids])),'Exact vector overlap across history/test'
 allout={};mainm=None;mainrows=None;mainnear=None
 for name in ['MLP','LR','XGBoost']:
  h=np.load(R/f'{name}_history_trace.npz');t=np.load(R/f'{name}_test_trace.npz');m=metrics(h,t,hids,tids)
  edges,rows,near,perquery=groups(m,dist)
  csvsave(R/f'{name}_distance_bins.csv',rows);csvsave(R/f'{name}_nearest_rank_bins.csv',near);csvsave(R/f'{name}_per_query_quantile_bins.csv',perquery)
  np.savez_compressed(R/f'{name}_history_test_pairs.npz',history_query_ids=hids,test_query_ids=tids,l2=dist,**m)
  # Transparent pair-level correlations; descriptive, not independent-pair significance tests.
  corr={}
  for key in ['agreement','skip_precision','coverage','agreed_skip_coverage']:
   valid=np.isfinite(m[key]);corr[key]=float(stats.spearmanr(dist[valid],m[key][valid]).statistic)
  allout[name]={'distance_edges':edges.tolist(),'distance_bins':rows,'nearest_rank_bins':near,'per_query_quantile_bins':perquery,'spearman_l2':corr,'history_model_decisions':int(h['model_invoked'].sum()),'test_model_decisions':int(t['model_invoked'].sum())}
  if name=='MLP':
   order=np.argsort(dist,axis=0,kind='stable')[:50]
   nearest=[]
   for j,qid in enumerate(tids):
    for rank,i in enumerate(order[:,j],1):
     nearest.append({'test_query_id':int(qid),'history_query_id':int(hids[i]),'neighbour_rank':rank,'l2':float(dist[i,j]),'common_nodes':int(m['common'][i,j]),'historical_skip_common_nodes':int(m['source_skip'][i,j]),'agreement':float(m['agreement'][i,j]),'skip_precision':float(m['skip_precision'][i,j]),'coverage':float(m['coverage'][i,j])})
   csvsave(R/'MLP_per_target_nearest_50.csv',nearest)
   close=dist<edges[1];far=(dist>=edges[1])&(dist<edges[2]);rng=np.random.default_rng(SEED);indices=rng.integers(0,len(tids),(500,len(tids)))
   effects={}
   for key in ['agreement','skip_precision','coverage']:
    v=m[key];means=[];boots=[]
    for mask in [close,far]:
     valid=mask&np.isfinite(v);s=np.where(valid,v,0).sum(0);n=valid.sum(0);means.append(float(s.sum()/n.sum()));boots.append(s[indices].sum(1)/np.maximum(n[indices].sum(1),1))
    delta=boots[0]-boots[1];effects[key]={'nearest_minus_next_bin':means[0]-means[1],'ci95':np.percentile(delta,[2.5,97.5]).tolist()}
   allout[name]['near_next_effects']=effects
   allout[name]['farthest_agreement_defined']=rows[-1]['agreement'] is not None
  if name=='MLP':mainm=m;mainrows=rows;mainnear=near
 # Test-test diagnostic only: both directions, diagonals excluded. Never build a deployment bank.
 t=np.load(R/'MLP_test_trace.npz');m=metrics(t,t,tids,tids);td=l2(q[tids],q[tids]);mask=~np.eye(len(tids),dtype=bool)
 edges=np.quantile(td[mask],[0,.01,.05,.20,.5,1]);tt=[]
 for j,label in enumerate(['0–1%','1–5%','5–20%','20–50%','50–100%']):
  take=mask&(td>=edges[j])&((td<edges[j+1]) if j<4 else td<=edges[j+1]);tt.append(aggregate(m,take,label,td))
 csvsave(R/'MLP_test_test_diagnostic.csv',tt);allout['test_test_diagnostic']={'directed_pairs':int(mask.sum()),'distance_bins':tt,'history_use':False}
 # Pair-formula validation against independent Python sets for 100 seeded pairs.
 h=np.load(R/'MLP_history_trace.npz');t=np.load(R/'MLP_test_trace.npz');rng=np.random.default_rng(SEED)
 def sets(trace,qi):
  ix=(trace['query_id']==qi)&trace['model_invoked'];return set(trace['node_id'][ix].tolist()),set(trace['node_id'][ix&(trace['model_decision']!=0)].tolist())
 for i,j in rng.integers(0,1000,(100,2)):
  hv,hs=sets(h,hids[i]);tv,ts=sets(t,tids[j]);assert mainm['common'][i,j]==len(hv&tv);assert mainm['source_skip'][i,j]==len(hs&tv);assert mainm['both_skip'][i,j]==len(hs&ts);assert mainm['equal'][i,j]==len((hv&tv)-(hs^ts))
 allout['verification']={'independent_set_formula_checks':100,'history_test_id_disjoint':True,'history_test_vector_disjoint':True,'pairs':1000000,'ci':'500 target-query cluster bootstrap resamples; descriptive conditional on frozen history; no pair-independent significance claim'}
 dump(R/'analysis.json',allout)
 from plot_figures import render_all
 results=json.loads((R/'results.json').read_text())
 render_all(allout,results,dist)
 report(allout,results)
 print('Pair analysis and figures complete',flush=True)
def report(analysis,results):
 def pct(x,d=2):return 'N/A' if x is None else f'{100*x:.{d}f}'
 lock=json.loads((R/'SELECTION_LOCK.json').read_text());a=analysis['MLP'];b=a['distance_bins'];near=a['nearest_rank_bins'];g=json.loads((R/'global_once.json').read_text())
 lines=['# E1：Query 相似性与历史 Skip 决策复用潜力','','本轮已执行真实原生搜索。主实验 SIFT1M，既有 DiskANN 图，L=100、K=10、beam=4、4线程、固定 BFS 节点缓存10000。主模型为既有17→128→64→1 state-only MLP；LR、XGBoost为扩展对照，均执行各自独立轨迹。所有跨 Query 决策复用关闭。','','## 数据与工作点','','从既有8000条模型训练查询中，用seed=20261018选1000条作为历史端；验证1000条、测试1000条，ID及历史/测试完整向量不重叠。历史库仅来自训练查询。测试集此前已被历史实验观察，不是全新未见测试集。静态节点缓存的成员ID已保存，历史/测试哈希一致。每条查询的候选池、visited及线程局部逻辑状态独立。','','99.653%是旧全10000查询无剪枝工作点，不等于本次1000条测试子集召回。主参数只根据验证集选择：在原有阈值候选中，选满足Recall≥99.60%的最大阈值；没有为凑测试召回调参。候选全表在results.json。最初99.603%的绝对下限会退回MLP不剪枝，随后仅根据验证集0.01pp离散精度将下限对齐为99.60%；修订时间和原因见PROTOCOL_AMENDMENT.json，原pilot运行保存在runs/和pilot_*.json。','','|方法|冻结阈值|测试Recall@10 %|请求/q|模型调用/q|平均延迟 µs（三轮中位数）|p95 µs（三轮中位数）|','','|---|---:|---:|---:|---:|---:|---:|']
 for name,rows in results['test'].items():
  v={k:float(np.median([x[k] for x in rows])) for k in rows[0]};lines.append(f"|{name}|{lock['selected'].get(name,'—')}|{v['recall_pct']:.3f}|{v['reads']:.3f}|{v['model_calls']:.3f}|{v['latency_us']:.2f}|{v['p95_us']:.2f}|")
 lines+=['',f"全局一次标记保留为baseline：仅统计同一1000条历史查询的原始无剪枝miss；至少观察一次且所有观察均没有新邻居入队、节点不在该查询最终Top10时标记。共{g['marked_nodes']}个节点，位图125000字节。全局标记不通过测试调参，也不隐藏其召回损失。",'','## 轨迹与指标定义','','每条轨迹保存query_id、node_id、state_features[17]、原生retain分数、Skip/Keep决策、search_step（query内候选决策序号，从0开始）、cache_hit、model_invoked。另有原始Query向量和ID映射。缓存命中记录保留作审计，其score=1为保留哨兵值，不代表实际推理；V_i仅包括实际调用模型的cache miss节点；P_i为其中的Skip集合。','','对全部1000000个有方向history→test对计算L2和集合交集。A为共同节点上的决策一致率；P_skip为历史Skip在共同节点上仍被目标模型Skip的比例；Coverage=|P_history∩V_test|/|V_test|。共同节点为0不参与A均值；历史Skip共同节点为0不参与P_skip均值，单独报告比例。Coverage不因无重叠而排除，零值参与均值。另报按事件汇总的pooled值，避免与Query对宏平均混淆。','','分桶严格使用实际Query对L2的分位数；边界（L2）为：'+', '.join(f'{x:.3f}' for x in a['distance_edges'])+'。阈值相等处用左闭右开，末桶含最大值。另按每条目标Query的历史近邻排名统计，并生成每Query内部距离分位桶。','','|距离分位|共同节点/q对|A %|P_skip %|Coverage %|A有效对数|P_skip有效对数|无共同节点 %|无可复用Skip %|','','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in b:lines.append(f"|{r['group']}|{r['mean_common_nodes']:.3f}|{pct(r['agreement'])}|{pct(r['skip_precision'])}|{pct(r['coverage'],3)}|{r['agreement_valid_pairs']}|{r['skip_precision_valid_pairs']}|{pct(r['no_common_fraction'],3)}|{pct(r['no_reusable_skip_fraction'],3)}|")
 effect=a['near_next_effects']['agreement'];ci=effect['ci95']
 lines+=['','![Query相似性与复用潜力](figures/similarity_reuse_potential.png)','','![每Query近邻](figures/per_query_neighbours.png)','','![距离与交集](figures/distance_and_overlap.png)','','## H1的实测解读','',f"最近0–1%桶与最远50–100%桶：A分别为{100*b[0]['agreement']:.2f}%、{pct(b[-1]['agreement'])}；P_skip分别为{100*b[0]['skip_precision']:.2f}%、{pct(b[-1]['skip_precision'])}；Coverage分别为{100*b[0]['coverage']:.3f}%、{100*b[-1]['coverage']:.3f}%。",'',f"最近0–1%桶与下一1–5%桶的A差值={100*effect['nearest_minus_next_bin']:.3f}个百分点，目标Query聚类bootstrap95%区间=[{100*ci[0]:.3f}, {100*ci[1]:.3f}]个百分点。",'',f"逐目标最近一条历史Query：A={100*near[0]['agreement']:.2f}%，P_skip={100*near[0]['skip_precision']:.2f}%，Coverage={100*near[0]['coverage']:.3f}%，平均共同节点{near[0]['mean_common_nodes']:.2f}个。",'','最远50–100%桶完全没有共同模型决策节点，A和P_skip未定义，报告为N/A，不能填成0或100%。本轮结果不支持“越相似则A越高”的整体单调关系；更近的Query显著增加共享节点，但Skip一致性和覆盖价值仍需独立评估。','这些统计用于检验相似度与决策一致性的关联，不能单凭A高就认定复用有价值。必须同时看Skip精确率、可覆盖的模型调用比例、无重叠比例和错误Skip风险。每桶CSV保存目标Query聚类bootstrap 95%区间；历史端固定，因此区间不包含历史库采样不确定性。Query对并不独立，不使用百万对的独立样本p值宣称显著。H1不是因果结论，当前数据也没有控制共同节点难度等混杂因素。','','双方模型同意Skip的交集/目标决策数另列为agreed_skip_coverage；同意不代表反事实安全；这是原独立轨迹上的可复用潜力，不能等同执行复用后的额外I/O节省。已有模型本来会Skip的节点，复用该决策首先省掉的是预测；错误Skip会改变后续路径，必须在后续真正开启复用时单独测量召回和I/O。','','## 成本分解','','按五项记录 C_total=C_distance+C_model+C_memory+C_SSD+C_bookkeeping。单独profile运行使用互斥软件区域计时，trace写入和性能计时不混在一起；profile与关闭profile计时返回ID一致。关闭profile仍有空指针分支检查的少量开销，所有方法使用同一采集程序。','','|方法|distance µs|model µs|memory µs|SSD µs|bookkeeping µs|','','|---|---:|---:|---:|---:|---:|']
 for name,p in results['profile'].items():lines.append('|'+name+'|'+ '|'.join(f'{p[k]:.2f}' for k in ['distance_us','model_us','memory_us','ssd_us','bookkeeping_us'])+'|')
 lines+=['','![成本分解](figures/cost_breakdown.png)','','distance包含PQ查询表、PQ距离查表和展开节点的完整向量距离内核；model包含标准化和分类器评分；memory仅包含显式PQ编码聚集和磁盘向量到scratch的复制；SSD为同步reader调用（提交与等待）；bookkeeping包含候选队列、visited、特征提取、缓存查找、分配、排序及未单独包裹的小型初始化计算。这里的memory不是全部硬件内存访问时间，模型和距离区域内的隐式内存等待仍归属该区域，不能把硬件内存成本严格分离。SSD请求数也不是物理SSD读盘次数。profile中的计时器会增加开销，绝对延迟采用无profile、无audit三轮结果；不将profile分项直接从无插桩总延迟相减。','','## 扩展与核验','','LR和XGBoost与MLP共享同一原始训练划分和17维特征，保留其既有权重，单独验证选阈值、独立在线剪枝与采集，非在MLP轨迹上替换分数冒充独立搜索。XGBoost原生导出仅包含best_iteration之前的树，与Python预测核验。','','![模型对照](figures/model_controls.png)','','另存MLP测试—测试有方向999000对，排除自身Query对，仅作为描述性诊断；不把测试Skip记录用于任何历史库或测试时决策。','','核验：历史/验证/测试ID不交叉；历史/测试向量不重复；native/Python分数一致；三轮ID确定性；trace/profile和无插桩返回ID一致；100个随机Query对用独立Python集合公式校验；所有成本分项相加误差检查。详细数值见verification.json和analysis.json。','','## 文件与复现','','- PROTOCOL.json、SELECTION_LOCK.json：主协议和测试前阈值锁定。','- experiment.log、build.log、runs_main/*.log、*.command.json：阶段日志、编译日志和完整命令。','- *_trace.npz：完整结构化轨迹；data/query_vectors.npy和query_splits.json：向量及映射。','- *_history_test_pairs.npz：逐Query对原始计数和指标；*_distance_bins.csv、*_nearest_rank_bins.csv、*_per_query_quantile_bins.csv：汇总及区间。','- PROVENANCE.json：模型、索引、程序哈希；COMPLETE.json：结束校验。','- figures/：PNG图；主要图同时提供PDF。','','```bash','cd /home/lxyue/projects/learned-gate','.venv/bin/python experiments/e1_similarity/build_native.py','.venv/bin/python experiments/e1_similarity/run.py','```','','原生插桩库和XGBoost支持仅生成于本实验native/目录，不修改已有DiskANN源码和既有二进制。已完成搜索可按done记录恢复；若改变协议、模型或程序，应使用新目录运行，不能混用旧done记录。']
 lines += ['',f"稀疏桶解读：20–50%桶只有{b[3]['agreement_valid_pairs']}对可计算A、{b[3]['skip_precision_valid_pairs']}对可计算P_skip，不能据此解释为稳定的0%复用精确率。最近0–1%桶按共同节点事件加权的A={100*b[0]['pooled_agreement']:.2f}%，目标Keep比例={100*b[0]['all_retain_agreement']:.2f}%；Keep占多数本身即可造成较高表面一致性。",'',f"纯评分成本：MLP的模型区域约{results['profile']['MLP']['model_us']:.2f}µs/query，占该profile五项总和约{100*results['profile']['MLP']['model_us']/sum(results['profile']['MLP'][k] for k in ['distance_us','model_us','memory_us','ssd_us','bookkeeping_us']):.2f}%。在固定轨迹的分项账本中，消除评分区域本身的空间有限；实际复用还要付出查询匹配、内存查找和维护成本，本E1不宣称端到端复用加速。"]
 lines=[line for i,line in enumerate(lines) if not (line=='' and i>0 and i+1<len(lines) and lines[i-1].startswith('|') and lines[i+1].startswith('|---'))]
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
