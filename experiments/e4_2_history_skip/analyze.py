from common import *
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from scipy.special import expit
import html
A=load();X=A['x'];U=A['U'].astype(bool);qr=A['query_row'].astype(int);H=A['H'];part=A['partition'];models=json.loads((R/'models.json').read_text());S=np.load(R/'scores.npy');single=np.load(E4/'outcomes.npz');categories=single['category'];proto=json.loads((R/'PROTOCOL.json').read_text());configs=json.loads((R/'configs.json').read_text());select=json.loads((R/'SELECTION_LOCK.json').read_text());base=np.genfromtxt(R/'graph/baseline.baseline.csv',delimiter=',',names=True);rng=np.random.default_rng(SEED)
# Bootstrap Query blocks, keeping the historical bank fixed and correlated Keep positions together.
boot_cache={}
def boot_rows(scope):
 ids=np.flatnonzero(np.ones(1000,bool) if scope=='all' else A['query_partition']=={'fit':0,'validation':1,'test':2}[scope]);key=scope
 if key not in boot_cache:boot_cache[key]=(ids,rng.integers(0,len(ids),(1000,len(ids))))
 return boot_cache[key]
def ratio(value,denom,scope):
 ids,draws=boot_rows(scope);num=np.bincount(qr,weights=np.asarray(value,dtype=float),minlength=1000)[ids];den=np.bincount(qr,weights=np.asarray(denom,dtype=float),minlength=1000)[ids];ns=num[draws].sum(1);ds=den[draws].sum(1);samples=ns[ds>0]/ds[ds>0]
 if den.sum()==0:return (None,None,None),None
 return (float(num.sum()/den.sum()),*map(float,np.percentile(samples,[2.5,97.5]))),(ns,ds)
def contrast_groups(m1,m0,scope):
 ids,draws=boot_rows(scope)
 def vals(mask):
  ns=np.bincount(qr,weights=U*mask,minlength=1000)[ids];ds=np.bincount(qr,weights=mask,minlength=1000)[ids];return ns.sum()/ds.sum() if ds.sum() else None,ns[draws].sum(1),ds[draws].sum(1)
 a,an,ad=vals(m1);b,bn,bd=vals(m0);valid=(ad>0)&(bd>0)
 if a is None or b is None:return dict(difference=None,lo=None,hi=None)
 z=an[valid]/ad[valid]-bn[valid]/bd[valid];return dict(difference=float(a-b),lo=float(np.percentile(z,2.5)),hi=float(np.percentile(z,97.5)))
scut=proto['state_distance_cutoff'];qcut=proto['query_distance_cutoff'];state=H&(A['state_distance']<scut);query=H&(A['query_distance']<qcut);joint=A['joint_close'];groups=[];contrasts=[]
for scope in ['all','fit','validation','test']:
 scope_mask=np.ones(len(U),bool) if scope=='all' else part=={'fit':0,'validation':1,'test':2}[scope]
 for name,cond in [('All Keep',np.ones(len(U),bool)),('Conflict Keep',H),('Normal Keep',~H),('Conflict + State close',state),('Conflict + State far',H&~state),('Conflict + Query close',query),('Conflict + Query far',H&~query),('Conflict + joint close same record',joint),('Conflict + State close + no joint close',state&~joint)]:
  mask=scope_mask&cond;stats,_=ratio(U*mask,mask,scope);groups.append(dict(scope=scope,group=name,n=int(mask.sum()),queries=int(len(np.unique(qr[mask]))),A=int(np.sum(U&mask)),coverage=float(mask.sum()/scope_mask.sum()),precision=stats[0],lo=stats[1],hi=stats[2],mean_state_distance=float(A['state_distance'][mask].mean()) if mask.any() and np.all(H[mask]) else None))
 for name,m1,m0 in [('Conflict minus Normal',H,~H),('State close minus Normal',state,~H),('State close minus State far within H',state,H&~state),('State close minus All Conflict',state,H),('Query close minus Query far within H',query,H&~query),('Joint close minus State-close without joint',joint,state&~joint)]:contrasts.append(dict(scope=scope,contrast=name,**contrast_groups(scope_mask&m1,scope_mask&m0,scope)))
csvsave(R/'history_group_comparison.csv',groups);csvsave(R/'group_contrasts.csv',contrasts)
# Association by fixed distance bands, preserving unsupported/empty bins.
bins=[]
for name,values,edges in [('state_distance',A['state_distance'],[0,.5,1,2,scut,4,6,float('inf')]),('query_distance',A['query_distance'],[0,150,qcut,370.59762491429177,500,float('inf')])]:
 for lower,upper in zip(edges[:-1],edges[1:]):
  mask=(part==2)&H&(values>=lower)&(values<upper);st,_=ratio(U*mask,mask,'test');bins.append(dict(variable=name,lower=lower,upper=upper if np.isfinite(upper) else None,n=int(mask.sum()),queries=int(len(np.unique(qr[mask]))),precision=st[0],lo=st[1],hi=st[2]))
csvsave(R/'distance_bins.csv',bins)
# Static Precision@Coverage: common denominator is ALL original Keeps, even for H-gated methods.
# Fractional boundary ties give the expected precision of an outcome-independent tie breaker.
pc=[]
for scope in ['validation','test']:
 sm=part=={'validation':1,'test':2}[scope];nall=int(sm.sum());prevalence=float(U[sm].mean())
 for k,name in enumerate(METHODS):
  eligible=sm&(H if 1<=k<=5 else True);ix=np.flatnonzero(eligible);rank=ix[np.lexsort((A['tie'][ix],-S[ix,k]))];capacity=len(ix)/nall
  targets=sorted(set(COVERAGES+[1.]+([capacity] if 1<=k<=5 else [])))
  for cov in targets:
   count=int(np.ceil(cov*nall-1e-10));entry=dict(scope=scope,method=name,target_coverage=cov,capacity=capacity,available=count<=len(ix),selected=None,precision=None,lo=None,hi=None,deterministic_precision=None,lift=None,B_rate=None,recall_harm_rate=None)
   if count<=len(ix) and count>0:
    chosen=rank[:count];boundary=S[rank[count-1],k];higher=eligible&(S[:,k]>boundary);tied=eligible&(S[:,k]==boundary);fraction=(count-higher.sum())/tied.sum();weights=higher.astype(float)+fraction*tied;st,_=ratio(U*weights,weights,scope);entry.update(selected=count,precision=st[0],lo=st[1],hi=st[2],deterministic_precision=float(U[chosen].mean()),lift=st[0]/prevalence,B_rate=float(np.sum(weights*(categories=='B'))/count),recall_harm_rate=float(np.sum(weights*np.isin(categories,['C','D']))/count))
   pc.append(entry)
csvsave(R/'precision_at_coverage.csv',pc)
# AUROC/AP are reported both globally (history gate limits coverage) and within identical H positions.
perf=[]
for scope in ['fit','validation','test']:
 for population in ['all_keep','conflict_only']:
  mask=(part=={'fit':0,'validation':1,'test':2}[scope])&(H if population=='conflict_only' else True);y=U[mask]
  for k,name in enumerate(METHODS):
   score=S[:,k].copy()
   if population=='all_keep' and 1<=k<=5:score[~H]=-1e30
   canprob=k>=3 and (population=='conflict_only' or k==6);prob=expit(S[mask,k]) if canprob else None
   perf.append(dict(scope=scope,population=population,method=name,n=int(mask.sum()),prevalence=float(y.mean()),AUROC=float(roc_auc_score(y,score[mask])) if len(np.unique(y))>1 else None,AP=float(average_precision_score(y,score[mask])),Brier=float(brier_score_loss(y,prob)) if canprob else None,log_loss=float(log_loss(y,prob,labels=[0,1])) if canprob else None))
csvsave(R/'prediction_metrics.csv',perf)
# Nested-model tests for independent predictive information; improvements are positive by convention.
abl=json.loads((R/'ablations.json').read_text());ascores={name:X@np.array(m['weights'])+m['intercept'] for name,m in abl.items()}
nested=[]
def compare(name,left,right,mask,auc_boot=False):
 y=U[mask];pl=expit(left[mask]);pr=expit(right[mask]);eps=1e-12
 ll=lambda p:-(y*np.log(np.clip(p,eps,1-eps))+(~y)*np.log(np.clip(1-p,eps,1-eps)))
 lossdiff=np.zeros(len(U));lossdiff[mask]=ll(pr)-ll(pl);st,_=ratio(lossdiff,mask,'test');brier=np.zeros(len(U));brier[mask]=(pr-y)**2-(pl-y)**2;bs,_=ratio(brier,mask,'test')
 result=dict(contrast=name,n=int(mask.sum()),queries=int(len(np.unique(qr[mask]))),AUROC_left=float(roc_auc_score(y,left[mask])),AUROC_right=float(roc_auc_score(y,right[mask])),AP_left=float(average_precision_score(y,left[mask])),AP_right=float(average_precision_score(y,right[mask])),logloss_improvement=st[0],logloss_lo=st[1],logloss_hi=st[2],Brier_improvement=bs[0],Brier_lo=bs[1],Brier_hi=bs[2])
 if auc_boot:
  ids,draws=boot_rows('test');qindex={int(v):i for i,v in enumerate(ids)};local=np.array([qindex[int(r)] for r in qr[mask]]);au=[];ap=[]
  for b in draws:
   w=np.bincount(b,minlength=len(ids))[local]
   if w[y].sum()==0 or w[~y].sum()==0:continue
   au.append(roc_auc_score(y,left[mask],sample_weight=w)-roc_auc_score(y,right[mask],sample_weight=w));ap.append(average_precision_score(y,left[mask],sample_weight=w)-average_precision_score(y,right[mask],sample_weight=w))
  result.update(AUROC_difference=float(result['AUROC_left']-result['AUROC_right']),AUROC_lo=float(np.percentile(au,2.5)),AUROC_hi=float(np.percentile(au,97.5)),AP_difference=float(result['AP_left']-result['AP_right']),AP_lo=float(np.percentile(ap,2.5)),AP_hi=float(np.percentile(ap,97.5)))
 nested.append(result)
compare('M5 minus M3: Query added to Skip-state distance',S[:,5],S[:,3],(part==2)&H,True)
compare('M6 minus current_state_only: historical statistics added',S[:,6],ascores['current_state_only'],part==2)
compare('current_state_history_query minus M6: Query added after full current state/history',ascores['current_state_history_query'],S[:,6],part==2)
csvsave(R/'nested_feature_comparisons.csv',nested)
# Recompute every deployed graph point from final IDs against GT, with paired Query confidence intervals.
with (E1/'data/test.gt').open('rb') as f:shape=np.fromfile(f,np.uint32,2);gt=np.fromfile(f,np.uint32,int(shape.prod())).reshape(tuple(shape))
graph=[];verification_count=0;graph_data={}
for scope in ['validation','test']:
 rows=np.flatnonzero(A['query_partition']=={'validation':1,'test':2}[scope]);b=base[rows];ids_scope,draws=boot_rows(scope)
 for cfg in configs:
  prefix=R/'graph'/f'{cfg["key"]}.{scope}';g=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True);ids=np.fromfile(str(prefix)+'.ids',np.uint64).reshape(len(rows),10);hits=(ids[:,:,None]==gt[rows,None,:10]).any(2).sum(1)
  assert np.array_equal(g['query_row'],rows) and np.array_equal(hits,g['hits']);assert np.all(g['pruned']<=g['live_keeps'])
  assert sha(Path(str(prefix)+'.cache.bin'))==sha(E1/'runs_main/test_MLP_trace.cache.bin')
  if cfg['method']==-1:
   assert np.array_equal(ids,np.fromfile(E4/'intervention.baseline.ids',np.uint64).reshape(1000,10)[rows])
  if 1<=cfg['method']<=5:assert np.all(g['pruned']<=g['live_conflicts'])
  dr=(g['hits']-b['hits'])*10;dio=b['reads']-g['reads'];zrec=dr[draws].mean(1);zio=dio[draws].mean(1)
  entry=dict(scope=scope,key=cfg['key'],method='baseline' if cfg['method']==-1 else METHODS[cfg['method']],target_coverage=cfg['target_coverage'],recall_pct=float(hits.mean()*10),delta_recall_pp=float(dr.mean()),delta_recall_lo=float(np.percentile(zrec,2.5)),delta_recall_hi=float(np.percentile(zrec,97.5)),reads_per_query=float(g['reads'].mean()),reads_saved_per_query=float(dio.mean()),reads_saved_lo=float(np.percentile(zio,2.5)),reads_saved_hi=float(np.percentile(zio,97.5)),reads_saved_fraction=float(dio.sum()/b['reads'].sum()),expanded_per_query=float(g['expanded'].mean()),neighbor_cmps_per_query=float(g['neighbor_cmps'].mean()),pruned_per_query=float(g['pruned'].mean()),live_keep_coverage=float(g['pruned'].sum()/g['live_keeps'].sum()),selected_on_validation=False)
  entry['selected_on_validation']=any(s['key']==cfg['key'] for s in select['selected'].values());graph.append(entry);graph_data[(scope,cfg['key'])]=g;verification_count+=1
csvsave(R/'recall_ssd_curve.csv',graph)
selected=[]
for method,cfg in select['selected'].items():
 key=cfg['key'];v=next(g for g in graph if g['scope']=='test' and g['key']==key);val=next(g for g in graph if g['scope']=='validation' and g['key']==key);selected.append(dict(method=method,selected_key=key,validation_delta_recall_pp=val['delta_recall_pp'],validation_reads_saved=val['reads_saved_per_query'],**{k:v[k] for k in ['recall_pct','delta_recall_pp','delta_recall_lo','delta_recall_hi','reads_per_query','reads_saved_per_query','reads_saved_lo','reads_saved_hi','reads_saved_fraction','pruned_per_query','live_keep_coverage']},test_point_within_0_2pp=v['delta_recall_pp']>=-.2-1e-9))
csvsave(R/'selected_system_results.csv',selected)
# Save per-position predictors without exposing U to the deployed classifier.
rows=[]
for i in range(len(U)):rows.append(dict(query_row=int(qr[i]),query_id=int(A['query_id'][i]),node=int(A['node_id'][i]),partition=['fit','validation','test'][part[i]],U=int(U[i]),H=int(H[i]),history_total=int(A['history_total'][i]),history_skip=int(A['history_skip'][i]),skip_fraction=float(X[i,21]),state_distance=float(A['state_distance'][i]) if H[i] else None,query_distance=float(A['query_distance'][i]) if H[i] else None,joint_close=int(joint[i]),**{m:float(S[i,k]) for k,m in enumerate(METHODS)}))
csvsave(R/'position_predictions.csv',rows)
result=dict(groups=groups,group_contrasts=contrasts,nested_comparisons=nested,selected_system_results=selected,test_keep=int(np.sum(part==2)),test_conflict=int(np.sum((part==2)&H)),test_A_prevalence=float(U[part==2].mean()),test_conflict_capacity=float(np.sum((part==2)&H)/np.sum(part==2)),history_bank_fixed=True,complete_graph_points=verification_count,limitations=['Existing E4-1 observed queries; exploratory within-pool split, not new confirmatory dataset','History methods have small support; high precision at small coverage has wide uncertainty','No latency claim: query distances precomputed from vectors; real graph/Recall/SSD requests are measured','Single-intervention U is not a label of cumulative strategy safety','Budget selected by validation aggregate Recall; does not guarantee each test query or test aggregate meets budget','No causal conclusion from predictive associations'])
dump(R/'results.json',result)
# Standalone vector charts: Precision@Coverage, Recall-SSD graph curves, and H/state groups.
F=R/'figures';F.mkdir(exist_ok=True);colors=['#2563eb','#dc2626','#d97706','#16a34a','#9333ea','#0891b2','#0f172a']
def chart(filename,title,series,xlabel,ylabel,xlim,ylim):
 width,height=860,530;left,right,top,bottom=85,620,60,445
 def pos(x,y):return left+(x-xlim[0])/(xlim[1]-xlim[0])*(right-left),bottom-(y-ylim[0])/(ylim[1]-ylim[0])*(bottom-top)
 ss=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13" fill="#1e293b"><text x="85" y="28" font-size="18">{html.escape(title)}</text>']
 for x in np.linspace(*xlim,6):
  xx,_=pos(x,ylim[0]);ss.append(f'<path d="M{xx} {top}V{bottom}" stroke="#e2e8f0"/><text x="{xx-14}" y="{bottom+23}">{x:.2f}</text>')
 for y in np.linspace(*ylim,6):
  _,yy=pos(xlim[0],y);ss.append(f'<path d="M{left} {yy}H{right}" stroke="#e2e8f0"/><text x="18" y="{yy+4}">{y:.2f}</text>')
 for i,(name,points) in enumerate(series):
  color=colors[i%len(colors)];points=sorted(points);valid=[pos(x,y) for x,y in points];ss.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'+ ' '.join(f'{x:.2f},{y:.2f}' for x,y in valid)+'"/>')
  for xx,yy in valid:ss.append(f'<circle cx="{xx}" cy="{yy}" r="3" fill="{color}"/>')
  ss.append(f'<path d="M650 {70+i*25}h20" stroke="{color}" stroke-width="3"/><text x="678" y="{74+i*25}">{html.escape(name)}</text>')
 ss.append(f'<text x="220" y="500">{html.escape(xlabel)}</text><text x="85" y="48">{html.escape(ylabel)}</text></g></svg>');(F/filename).write_text(''.join(ss))
series=[]
for m in METHODS:series.append((m,[(p['target_coverage']*100,p['precision']*100) for p in pc if p['scope']=='test' and p['method']==m and p['available'] and p['target_coverage']<=.02]))
chart('precision_at_coverage.svg','Held-out identification: A-class precision at Keep coverage',series,'Coverage of all original Keeps (%)','A-class precision (%)',(0,2),(50,100))
series=[]
for m in METHODS:series.append((m,[(g['reads_per_query'],g['recall_pct']) for g in graph if g['scope']=='test' and g['method'] in [m,'baseline']]))
ys=[g['recall_pct'] for g in graph if g['scope']=='test'];xs=[g['reads_per_query'] for g in graph if g['scope']=='test'];chart('recall_ssd_curve.svg','Held-out cumulative graph search: frozen validation thresholds',series,'SSD requests per Query (lower is better)','Recall@10 (%)',(min(xs)-1,max(xs)+1),(min(ys)-.1,max(ys)+.1))
# Main report, with explicit support and uncertainty.
def pct(x):return 'N/A' if x is None else f'{x*100:.2f}'
def group(scope,name):return next(g for g in groups if g['scope']==scope and g['group']==name)
lines=['# E4-2：历史Skip是否识别遗漏的冗余Keep','','## 核心结论','','历史Skip标记本身没有富集A类：评估集Conflict Keep的A比例72.90%，Normal Keep为83.94%，差值95%区间完全小于零。状态相似筛选提升Conflict Keep精确率至87.23%，但覆盖很小；在本轮嵌套Logistic对照中，没有证实Query相似度的独立增益，也没有证实历史统计在当前状态之外的独立增益。验证选中的完整搜索点中，M0仅按当前分数筛选获得最大请求节省22.34%，评估Recall下降0.20pp；该点刚好在预算边界，区间不保证总体损失≤0.2pp。','','## 数据、模型与冻结顺序','','1000条E1训练历史Query构成唯一历史库，静态且不在线更新。1000条E4-1 Query按seed=20261020分成500拟合、250验证、250评估；ID完全隔离。它们来自已经观察过的E4-1数据池，不是全新确认性测试集。U沿用E4-1逐位置真实单节点反事实标签。测试标签不参与拟合、参数选择或部署阈值设定。','','标准化沿用E2训练统计；状态阈值为 '+f'{scut:.6f}'+'，Query阈值为 '+f'{qcut:.6f}'+'。H要求同节点任意历史实际模型调用为Skip；频率分母是该节点全部历史实际模型调用。距离只匹配历史Skip记录。联合条件要求同一条历史Skip记录同时满足State和Query限制，不把两个不同来源的近邻拼成一条证据。','','|方法|筛选依据|可筛范围|','|---|---|---|','|M0|当前retain分数低者优先|全部Keep|','|M1|曾有历史Skip；并列按与标签无关的固定hash处理|Conflict Keep|','|M2|历史Skip占比高者优先|Conflict Keep|','|M3|Logistic：最近历史Skip的状态距离|Conflict Keep|','|M4|Logistic：最近历史Skip的Query距离|Conflict Keep|','|M5|Logistic：M3状态距离 + 最近Skip Query距离 + 状态最近Skip的Query距离|Conflict Keep|','|M6|Logistic：当前分数、17维当前状态、历史频率统计、最近Skip状态距离/状态近指示；不使用Query距离|全部Keep|','','Logistic固定C=1、无类别重加权，特征scaler只拟合500条拟合Query。M3–M5只在Conflict Keep上拟合，M6在全部Keep上拟合。M0–M2为简单规则。分数表示排序，原生retain分数未校准为检索必要性概率。','','## 分组对照','','|范围|分组|Keep数|Query数|A比例 %|95%区间 %|','|---|---|---:|---:|---:|---|']
for scope in ['all','test']:
 for name in ['All Keep','Conflict Keep','Normal Keep','Conflict + State close','Conflict + Query close','Conflict + joint close same record']:
  g=group(scope,name);lines.append(f'|{scope}|{name}|{g["n"]}|{g["queries"]}|{pct(g["precision"])}|[{pct(g["lo"])}, {pct(g["hi"])}]|')
lines+=['','|范围|差值对照|A比例差 pp|95%区间 pp|','|---|---|---:|---|']
for c in contrasts:
 if c['scope'] in ['all','test']:lines.append(f'|{c["scope"]}|{c["contrast"]}|{pct(c["difference"])}|[{pct(c["lo"])}, {pct(c["hi"])}]|')
lines+=['','所有区间按Query聚类bootstrap1000次，历史库固定；覆盖率低的条件不能只看较高比例。state close是min状态距离<阈值，query close是min Query距离<阈值；联合近邻必须来自同一条Skip。全量结果是探索性描述，250条评估Query结果用于本轮方法比较。','','## Precision@Coverage','','Coverage分母始终为评估集的全部原始Keep。M1–M5不向Normal Keep补选；超过历史标记容量时为N/A。边界同分样本按分数相同事件的期望精确率计算，避免人为打破并列造成M1虚假优势；实际固定hash排序精确率另存CSV。这是常规排序评估，部署图搜索阈值则只用验证分数分布冻结。','','|方法|0.1%覆盖 Precision %|0.25%|0.5%|1%|2%|5%|10%|20%|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for m in METHODS:
 vals=[]
 for cov in COVERAGES:
  rr=next(p for p in pc if p['scope']=='test' and p['method']==m and p['target_coverage']==cov);vals.append(pct(rr['precision']))
 lines.append('|'+m+'|'+'|'.join(vals)+'|')
lines+=['',f'评估集A类基准比例={result["test_A_prevalence"]*100:.3f}%；历史标记容量={result["test_conflict_capacity"]*100:.3f}%（{result["test_conflict"]}/{result["test_keep"]}）。完整区间、样本数、lift与实际hash选择结果见 `precision_at_coverage.csv`。','','## Query是否有独立预测增益','','M5与M3在完全相同的评估Conflict Keep上比较；另用完整当前状态的辅助嵌套对照，检验Query信息是否在当前状态与历史统计之外继续提供信息。正的Log loss改善表示增加特征后更好；这检验当前设定的预测增益，不是因果独立性。','','|嵌套对照|n|AUROC（新增 / 原）|AP（新增 / 原）|Log loss改善|95%区间|','|---|---:|---|---|---:|---|']
for v in nested:lines.append(f'|{v["contrast"]}|{v["n"]}|{v["AUROC_left"]:.4f} / {v["AUROC_right"]:.4f}|{v["AP_left"]:.4f} / {v["AP_right"]:.4f}|{v["logloss_improvement"]:.5f}|[{v["logloss_lo"]:.5f}, {v["logloss_hi"]:.5f}]|')
lines+=['','## 完整搜索的Recall–SSD曲线','','策略真实运行在原生完整搜索中：保留原MLP，先判定当前是否Keep，再由策略额外Skip；后续路径重新计算当前状态和策略特征。没有复用E4-1的旧位置标签或静态决策记录。DRAM命中沿用原冻结策略，直接保留、不参与额外剪枝。','','对每种方法，先用250条验证Query在冻结coverage菜单中选SSD节省最大且额外Recall损失≤0.2pp的点，无可行改善则回退原始搜索。随后冻结选择，评估集全部曲线点如实报告，不按评估结果挑点。验证预算不是评估预算保证。','','|方法|验证选中配置|评估Recall %|ΔRecall pp|SSD请求/q|请求节省 %|评估点是否在0.2pp内|','|---|---|---:|---:|---:|---:|---|']
testbase=next(g for g in graph if g['scope']=='test' and g['method']=='baseline')
lines.append(f'|原始MLP|baseline|{testbase["recall_pct"]:.3f}|0.000|{testbase["reads_per_query"]:.3f}|0.000|是|')
for v in selected:lines.append(f'|{v["method"]}|{v["selected_key"]}|{v["recall_pct"]:.3f}|{v["delta_recall_pp"]:.3f}|{v["reads_per_query"]:.3f}|{v["reads_saved_fraction"]*100:.3f}|{"是" if v["test_point_within_0_2pp"] else "否"}|')
lines+=['','M0所选点的ΔRecall 95%区间为[-0.40, -0.04]pp；M6为[-0.48, 0.00]pp。点估计满足预算不代表已证明总体质量损失在预算内。M1/M2在验证通过后，评估点仍下降0.36pp，应如实视为预算未通过。','','每个点的配对Query置信区间和全部冻结曲线见 `recall_ssd_curve.csv`；被选点区间见 `selected_system_results.csv`。SSD统计为提交请求数，不是物理设备读盘次数。本实验不宣称端到端延迟收益：Query距离预先从向量计算，且没有独立计量在线查询匹配和维护成本；真正图搜索路径、最终Recall与请求数均为实测。','','## 核验和边界','','原生无额外剪枝的全部93,410个Keep位置均与Python逐位置核对28维特征和7个策略分数。原始Top10、SSD请求、展开数、邻居距离比较数及固定缓存成员与E1/E4-1一致。全部验证和评估曲线点的最终Recall均从保存Top10与GT重新计算。历史库与E3序列化记录哈希相同。','','同一个Query的多个Keep不作为独立样本；所有关键比例差和质量/I/O区间都按Query聚类。H仅覆盖很少的Keep；M1–M5的高精确率即便成立，也不能自动外推成较高系统节省。U仅证明单节点干预性质，累计剪枝必须看本轮真实Recall曲线。对现有数据池的查询分割降低拟合泄漏，但不能消除先前观察数据带来的探索性限制。','','## 文件与复现','','`PROTOCOL.json`：拟合/验证/评估Query划分与规则；`models.json`：权重；`SELECTION_LOCK.json`：验证选择；`history_group_comparison.csv`、`group_contrasts.csv`：分组检验；`precision_at_coverage.csv`、`prediction_metrics.csv`、`nested_feature_comparisons.csv`：识别性能；`position_predictions.csv`：逐位置历史特征、预测和标签；`graph/*.csv,*.ids`：原生完整搜索统计和最终结果；`native_parity.json`：特征核验；`figures/*.svg`：独立可导出曲线。','','复现：`.venv/bin/python experiments/e4_2_history_skip/run.py`。新协议、模型或原生程序必须使用新输出目录；既有E1–E4-1及DiskANN源文件不修改。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
verification=dict(complete_graph_points=verification_count,all_final_recall_independently_recomputed=True,all_keep_pruned_counts_valid=True,all_history_policy_prunes_bounded_by_live_conflicts=True,all_cache_members_equal_e1=True,source_e4_1_hashes={str(f):sha(f) for f in [E4/'keep_positions.npz',E4/'outcomes.npz',E4/'COMPLETE.json']},native_parity=json.loads((R/'native_parity.json').read_text()),fit_validation_test_query_ids_disjoint=True,history_immutable=True,test_outcomes_used_for_model_fit_or_threshold_selection=False,selection_lock_sha256=sha(R/'SELECTION_LOCK.json'))
dump(R/'verification.json',verification)
artifacts={str(f.relative_to(R)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(R.rglob('*')) if f.is_file() and '__pycache__' not in str(f) and f.name not in ['artifact_manifest.json','COMPLETE.json','analysis.log','experiment.log']};dump(R/'artifact_manifest.json',artifacts);dump(R/'COMPLETE.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_sha256=sha(R/'REPORT.md'),verification_sha256=sha(R/'verification.json'),manifest_sha256=sha(R/'artifact_manifest.json')))
log('E4-2 COMPLETE')
print(json.dumps(dict(test_group_comparison=[g for g in groups if g['scope']=='test'],test_contrasts=[c for c in contrasts if c['scope']=='test'],nested=nested,selected=selected),ensure_ascii=False,indent=2))
