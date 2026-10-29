"""Shared-control wide-queue P0 reports and standalone SVG objective frontiers."""
from pathlib import Path
import csv,hashlib,json,subprocess,sys
import numpy as np
D=Path(__file__).resolve().parent
def load(name):return json.loads((D/name).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def save(p,rows):
    if not rows:return
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]
def fmt(x,n=3):return '—' if x is None else f'{float(x):.{n}f}'
LABEL={'DiskANN':'DiskANN','Heuristic':'Heuristic','MLP':'MLP','LR':'LR','DynamicMLP':'动态 MLP','DynamicLR':'动态 LR','ET':'ET','ET_MLP':'ET + MLP','ET_LR':'ET + LR'}
P=load('PROTOCOL.json');train=load('TRAIN_COMPLETE.json');models=['MLP','LR','DynamicMLP','DynamicLR'];families=list(LABEL)
pre=['# SIFT1M：扩大候选队列后重新训练与P0对比','',
'本轮停止SIFT100M实验，在完整100万条128维SIFT1M索引上重新训练并测量。搜索候选队列L从P0-A原100扩大到300；P0-B上限从300扩大到1000。Beam仍为4，K10、4查询线程、10,000节点缓存。','',
f'MLP和LR从头训练，共享{train["prune_samples"]}个候选样本、相同17维特征、retain代理标签与查询划分。训练覆盖L={P["L_training_and_learned_grid"]}；静态/动态版本共用各自同一份权重。ET也基于新范围的前缀轨迹重新训练，供P0-B及联合组共用。',
'训练查询1200条，其中200条仅用于内部训练轮次选择，再按选定轮次拟合全部1200条；验证和测试各1000条互斥查询。没有复用旧模型权重；没有使用验证/测试查询拟合权重。',
'本轮使用已经分析过的旧留出查询，属于探索性复测。队列范围与训练样本同时变化，不能将新旧差异纯归因于某一个因素。','',
'## 统一对照与运行池','',
'所有方法放在一张对比表中。DiskANN、Heuristic与ET各只有一套配置和原始结果；不再按MLP版/LR版重复运行。同一配置/重复编号只有一份记录。P0-A六组以六种循环顺序各重复2次，共12次，全部可被P0-B复用；其余P0-B最终前沿、验证锁定点和精确同Recall赢家至少3次。',
'计时包含特征、推理、I/O与搜索，排除加载、缓存构建、128条热身、GT和输出。I/O同时核对reader请求长度与Linux read_bytes，表示WSL2客户机块层流量。','']
if not (D/'MEASUREMENTS_COMPLETE.json').exists():
    (D/'REPORT.md').write_text('\n'.join(pre+['实验仍在运行，尚未生成最终对比结论。当前阶段见STATUS.json。','']))
    print('TRAINING REPORT WRITTEN; measurement still running');sys.exit(0)
assert load('verification.json')['passed']
rows=load('aggregate.json');bykey={r['key']:r for r in rows};aselection=load('P0A_SELECTION_LOCK.json');aselect_details=load('P0A_DYNAMIC_SELECTION_DETAILS.json')
arows=[bykey[c['key']] for c in aselection]
text=pre+['## P0-A：固定L300，六组共享对照','',
'静态阈值按验证L300候选的retain-FNR1%标定。动态配置仅用验证集选择：要求最终验证Recall不低于对应静态默认，再选最低P95；若无达标配置则明确标记。该条件不保证测试Recall相等。启发式固定rank>216且completed_expansions>300，缓存命中保护。','']
text+=table(['方法','Recall %','P95 ms','P95范围 ms','QPS','KiB/查询','读取/查询','跳过/查询','动态额外跳过/查询'],[[LABEL[r['family']],fmt(r['recall'],2),fmt(r['p95_ms']),fmt(r['p95_min_ms'])+'–'+fmt(r['p95_max_ms']),fmt(r['qps'],1),fmt(r['physical_bytes']/1024,2),fmt(r['reads'],2),fmt(r['skips'],2),fmt(r['progress_extra_skips'],2)] for r in arows])
base=arows[0];aeffects=[]
for r in arows[1:]:aeffects.append(dict(family=r['family'],key=r['key'],recall_loss_pp=base['recall']-r['recall'],p95_reduction_pct=100*(1-r['p95_ms']/base['p95_ms']),qps_increase_pct=100*(r['qps']/base['qps']-1),io_reduction_pct=100*(1-r['physical_bytes']/base['physical_bytes'])))
save(D/'p0_a_effects.csv',aeffects)
text+=['','相对同一份DiskANN对照：','']+table(['方法','Recall损失 pp','P95降低 %','QPS提升 %','读取量降低 %'],[[LABEL[r['family']],fmt(r['recall_loss_pp'],2),fmt(r['p95_reduction_pct'],2),fmt(r['qps_increase_pct'],2),fmt(r['io_reduction_pct'],2)] for r in aeffects])
text+=['','动态阈值参数与验证质量：','']+table(['方法','配置','基础τ','晚期τ','rank门槛','展开门槛','静态验证Recall','动态验证Recall','验证达标'],[[LABEL[a['family']],a['key'],fmt(bykey[a['key']]['tau'],6),fmt(bykey[a['key']]['late_tau'],6),bykey[a['key']]['rank'],bykey[a['key']]['exp'],fmt(a['static_validation_recall'],2),fmt(a['dynamic_validation_recall'],2),a['validation_quality_matched']] for a in aselect_details])
def frontier(rs,metric):
    out=[];cost=float('inf')
    for r in sorted(rs,key=lambda r:(-r['total_hits'],-r[metric] if metric=='qps' else r[metric])):
        v=-r[metric] if metric=='qps' else r[metric]
        if v<cost:out.append(r|dict(timing_repeats=r['repeats']));cost=v
    return sorted(out,key=lambda r:r['total_hits'])
frontiers={metric:{f:frontier([r for r in rows if r['family']==f],metric) for f in families} for metric in ['p95_ms','qps','physical_bytes','reads']};dump(D/'frontiers.json',frontiers)
save(D/'frontier_points.csv',[r|dict(frontier_metric=metric) for metric,fs in frontiers.items() for group in fs.values() for r in group])
selection=load('SELECTION_LOCK.json');validation={r['key']:r for r in load('validation.json')};selected=[]
for s in selection:
    r=bykey.get(s['key']);v=validation.get(s['key'])
    selected.append(s|dict(test_recall=r['recall'] if r else None,test_feasible=bool(r and r['recall']>=s['target']),p95_ms=r['p95_ms'] if r else None,qps=r['qps'] if r else None,KiB_per_query=r['physical_bytes']/1024 if r else None,L=r['L'] if r else None,tau=r['tau'] if r else None,late_tau=r['late_tau'] if r else None,rank=r['rank'] if r else None,exp=r['exp'] if r else None,skips=r['skips'] if r else None,extra_skips=r['progress_extra_skips'] if r else None,et_stopped=r['et_stop_fraction'] if r else None))
save(D/'validation_locked_test.csv',selected)
text+=['','## P0-B：扩大L范围的Recall—成本比较','',
'每组64个预先冻结配置。DiskANN扫描64个L，其余为8个L×8档参数。三条曲线分别取实测非支配前沿，不将不同配置的最优值拼在一行；完整测试网格前沿是描述性结果。',
'验证集按目标Recall和P95/QPS/实测字节分别选参，测试前锁定。下表只展示按验证最低P95选出的高质量配置；不能用测试不达标的配置声称满足质量要求。完整各目标/指标见validation_locked_test.csv。','']
for target in [99.5,99.7,99.9]:
    group=[s for s in selected if s['target']==target and s['objective']=='p95_ms']
    text += [f'### Recall要求 ≥{target}%（验证选参，测试实测）','']
    text+=table(['方法','实际Recall %','达标','P95 ms','QPS','KiB/查询','L','动态额外跳过/查询'],[[LABEL[r['family']],fmt(r['test_recall'],2),r['test_feasible'],fmt(r['p95_ms']),fmt(r['qps'],1),fmt(r['KiB_per_query'],2),r['L'] or '—',fmt(r['extra_skips'],2)] for r in group])+['']
common=set.intersection(*[{r['total_hits'] for r in rows if r['family']==f} for f in families]);exact=[]
for hits in sorted(common):
    for family in families:
        rs=[r for r in rows if r['family']==family and r['total_hits']==hits]
        for metric in ['p95_ms','qps','physical_bytes']:
            r=min(rs,key=lambda r:-r[metric] if metric=='qps' else r[metric]);exact.append(dict(total_hits=hits,recall=hits/100,family=family,metric=metric,value=r[metric],key=r['key'],L=r['L'],skips=r['skips'],extra_skips=r['progress_extra_skips'],et_stop_fraction=r['et_stop_fraction'],repeats=r['repeats']))
save(D/'strict_equal_recall.csv',exact)
text+=['## 动态策略是否实际生效','',
'动态曲线包括关闭动作的配置。需要结合skips、progress_extra_skips确认：模型虽然被调用，但没有额外跳过时，不能将结果解释为动态策略收益。τbase=0且触发候选全部跳过时，还需检查是否退化为纯启发式。','']
equivalence=[]
for r in [r for r in arows if r['family'] in ['DynamicMLP','DynamicLR']]:
    a=np.genfromtxt(D/'runs/test'/f'{r["key"]}_r0.csv',delimiter=',',names=True)
    for h in [h for h in rows if h['family']=='Heuristic' and h['L']==300]:
        b=np.genfromtxt(D/'runs/test'/f'{h["key"]}_r0.csv',delimiter=',',names=True)
        same_ids=r['ids_sha256']==h['ids_sha256'];same_counts=all(np.array_equal(a[k],b[k]) for k in ['reads','expansions','skips'])
        equivalence.append(dict(dynamic=r['key'],heuristic=h['key'],same_ids=same_ids,same_counts=same_counts,behavior_equivalent=same_ids and same_counts))
save(D/'P0A_behavior_equivalence.csv',equivalence)
eq=[r for r in equivalence if r['behavior_equivalent']]
text+=table(['动态配置','行为等价的启发式'],[[r['dynamic'],r['heuristic']] for r in eq]) if eq else ['P0-A获选动态配置没有与本轮8档L300启发式完全等价；这本身不证明模型带来性能收益。']
tail_rows=[]
for static_family,dynamic_family in [('MLP','DynamicMLP'),('LR','DynamicLR')]:
    static=next(r for r in arows if r['family']==static_family);dynamic=next(r for r in arows if r['family']==dynamic_family)
    cache={}
    for r in [static,dynamic,arows[0],arows[1]]:
        aa=[np.genfromtxt(D/'runs/test'/f'{r["key"]}_r{rep}.csv',delimiter=',',names=True) for rep in range(12)]
        cache[r['key']]={k:np.median(np.stack([a[k] for a in aa]),axis=0) for k in ['latency_us','io_us','io_rounds','reads','hits','progress_extra_skips']}
    tail=np.argsort(cache[static['key']]['latency_us'])[-50:]
    for r in [static,dynamic,arows[0],arows[1]]:
        c=cache[r['key']];tail_rows.append(dict(tail_defined_by=static_family,family=r['family'],key=r['key'],queries=50,mean_ms=float(c['latency_us'][tail].mean()/1000),io_ms=float(c['io_us'][tail].mean()/1000),reads=float(c['reads'][tail].mean()),io_rounds=float(c['io_rounds'][tail].mean()),hits=int(c['hits'][tail].sum()),extra_skips=float(c['progress_extra_skips'][tail].mean())))
save(D/'P0A_paired_tail.csv',tail_rows)
text+=['','P0-A在对应静态模型最慢50条查询上的配对结果，所有方法使用相同查询ID；不是比较各自不同的最慢集合：','']
text+=table(['尾部依据','方法','平均延迟 ms','Reader ms','读取/查询','I/O轮/查询','50条总命中'],[[r['tail_defined_by'],LABEL[r['family']],fmt(r['mean_ms']),fmt(r['io_ms']),fmt(r['reads'],2),fmt(r['io_rounds'],2),r['hits']] for r in tail_rows])
text+=['','## 曲线、数据和复现','',
'- [Recall–P95](figures/recall_p95.svg)',
'- [Recall–QPS](figures/recall_qps.svg)',
'- [Recall–读取量](figures/recall_physical_bytes.svg)',
'- [全部配置](all_configurations.csv) 与 [验证锁定结果](validation_locked_test.csv)',
'- [严格相同实际Recall比较](strict_equal_recall.csv)，匹配整数命中数，不插值；动作关闭的点明确保留计数。',
'- [训练信息](TRAIN_COMPLETE.json)、[原生核验](NATIVE_VERIFIED.json)、[运行核验](verification.json)',
'复现依次运行bootstrap.py、evaluate.py、report.py。原SIFT100M、原SIFT1M模型和原DiskANN源码保持不变。','',
'解释边界：本次仅一个训练随机种子，计时重复不包含训练变异；retain标签是贡献代理，不是反事实Recall损害；动态参数只有8档、联合策略只有8对阈值；ET仍基于未剪枝轨迹训练。测试结果不代表跨设备或全局最优，也不把有限计时差异当作统计显著性证明。','']
(D/'REPORT.md').write_text('\n'.join(text))
# Reuse the established offline plotting backend; all files are standalone SVG.
source=(D.parent/'p0_b_lr/plot.py').read_text()
replacements={"FAMILIES=['Tuned_DiskANN','Heuristic','Ours','ET','ET_Ours']":'FAMILIES='+repr(families),"LABELS=['Tuned DiskANN (L sweep)','Heuristic (rank + progress)','Ours (17D LR Gate)','ET (native DiskANN)','ET + LR']":'LABELS='+repr([LABEL[f] for f in families]),"COLORS=['#111111','#E69F00','#D55E00','#0072B2','#009E73']":"COLORS=['#111111','#E69F00','#0072B2','#009E73','#0072B2','#009E73','#777777','#CC79A7','#D55E00']","DASH=['','9 4','','3 3','9 3 2 3'];MARKERS=['o','s','^','D','v']":"DASH=['','9 4','','','7 3','7 3','3 3','9 3 2 3','9 3 2 3'];MARKERS=['o','s','^','D','v','+','x','s','D']","('p95_us','recall_p95','P95 query latency (ms)',.001)":"('p95_ms','recall_p95','P95 query latency (ms)',1.)",'y=665+(i//4)*45':'y=650+(i//4)*32','Each panel uses its own empirical objective frontier; final frontier timing points use medians of three runs.':'Separate empirical frontiers; at least 3 repeats, shared P0-A policies use 12 repeats.'}
for a,b in replacements.items():
    assert a in source,a;source=source.replace(a,b)
(D/'plot.py').write_text(source);subprocess.run([sys.executable,str(D/'plot.py')],check=True)
done=load('MEASUREMENTS_COMPLETE.json')|dict(analysis_complete=True,report='REPORT.md',models_new=True,plots_complete=True)
dump(D/'COMPLETE.json',done);dump(D/'STATUS.json',dict(phase='complete',complete=True,report='REPORT.md'))
print('REPORT COMPLETE',flush=True)
