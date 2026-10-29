"""Summarize frozen validation selections and their held-out test outcomes."""
from pathlib import Path
import csv,json
import numpy as np
D=Path(__file__).resolve().parent
def load(n):return json.loads((D/n).read_text())
def save(name,rows):
    with (D/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]
protocol=load('PROTOCOL.json');selection=load('SELECTION_LOCK.json');initial=load('validation_initial.json');runs=load('test_runs.json')
configs={c['key']:c for c in protocol['configs']}
summary={r['key']:r for r in csv.DictReader((D/'test_summary.csv').open())}
results=[]
for s in selection:
    r=summary.get(s['key'])
    results.append(s|dict(test_recall=float(r['recall']) if r else None,test_feasible=bool(r and float(r['recall'])>=s['target']),test_p95_ms=float(r['p95_ms']) if r else None,test_qps=float(r['qps']) if r else None,test_reads=float(r['reads']) if r else None,p95_min_ms=float(r['p95_min_ms']) if r else None,p95_max_ms=float(r['p95_max_ms']) if r else None))
save('selected_results.csv',results)
comparisons=[]
for target in protocol['targets']:
    group={r['group']:r for r in results if r['target']==target}
    dynamic=group['ProgressLR']
    for base in ['StaticLR','Heuristic']:
        control=group[base];ok=dynamic['test_feasible'] and control['test_feasible']
        comparisons.append(dict(target=target,baseline=base,dynamic_key=dynamic['key'],baseline_key=control['key'],both_test_feasible=ok,p95_reduction_pct=100*(1-dynamic['test_p95_ms']/control['test_p95_ms']) if ok else None,qps_increase_pct=100*(dynamic['test_qps']/control['test_qps']-1) if ok else None,read_reduction_pct=100*(1-dynamic['test_reads']/control['test_reads']) if ok else None))
save('comparisons.csv',comparisons)
# Distinguish useful scheduling from a classifier that merely reproduces a rule.
R=D.parent/'sift100m_p0'
heuristics=[c for c in json.loads((R/'configs.json').read_text()) if c['family']=='Heuristic' and c['L']==500]
equivalence=[]
for key in sorted({s['key'] for s in selection if s['group']=='ProgressLR' and s['key']}):
    for split in ['validation','test']:
        p=D/'runs'/split/f'{key}_r0'
        a=np.genfromtxt(p.with_suffix('.csv'),delimiter=',',names=True)
        for h in heuristics:
            op=R/'runs'/split/f'{h["key"]}_r0'
            b=np.genfromtxt(op.with_suffix('.csv'),delimiter=',',names=True)
            same_ids=p.with_suffix('.ids').read_bytes()==op.with_suffix('.ids').read_bytes()
            same_counts=all(np.array_equal(a[c],b[c]) for c in ['reads','expansions','skips'])
            equivalence.append(dict(key=key,split=split,heuristic=h['key'],same_ids=same_ids,same_search_counts=same_counts,behavior_equivalent=same_ids and same_counts))
if equivalence:save('behavior_equivalence.csv',equivalence)
def f(x,n=3):return '不可用' if x is None else f'{x:.{n}f}'
text=['# SIFT100M：更高Recall目标的进度阈值验证','',
'搜索完整1亿条128维uint8向量的原SIFT100M索引；固定原LR权重和L500，仅改变阈值调度；使用原1000条验证查询选参，提前锁定后对原1000条测试查询重复3次。结果用于验证有限策略，不能声称已经找到全局最优策略。','',
'## 策略与选择过程','',
'跳过条件仍为score < tau。通常使用基础阈值；当候选队列零起始排名rank > r，且已完成展开数completed > e时，改用max(基础阈值, 晚期阈值)。缓存命中始终保留。策略只使用在线当前状态，不使用该查询最终耗时或GT。',
'预先冻结36个动态配置：基础阈值取0、0.08935668、0.10672785，(r,e)取(125,375)、(250,450)、(320,460)、(360,500)，晚期阈值取0.15341286、0.35、0.95。对照含全部8档固定LR、4档启发式、未剪枝L500，以及两个纯启发式退化检查，共51个验证配置。目标为99.7%与99.75%。',
'验证初测后，每个质量目标/动态或固定LR组选P95最小的两个达标候选复测到3次；四档启发式和DiskANN也复测到3次。最终仅在这些预声明候选中按三次中位P95选择，锁定后测试，不按测试结果改参。','',
'## 锁定参数','']
for target in protocol['targets']:
    outcome=next(r for r in results if r['target']==target and r['group']=='ProgressLR')
    contrast=next(r for r in comparisons if r['target']==target and r['baseline']=='StaticLR')
    if contrast['both_test_feasible']:
        text[2:2]=[f'目标{target}%：动态策略测试Recall为{outcome["test_recall"]:.2f}%，P95为{outcome["test_p95_ms"]:.3f} ms，相对固定LR下降{contrast["p95_reduction_pct"]:.2f}%。这是按质量下限选参的结果；各方法实际Recall是否相同需结合下表判断。','']
    else:
        text[2:2]=[f'目标{target}%：本轮没有同时通过验证选择与测试Recall要求的动态策略，不能声明满足该目标的性能收益。','']
text[2:2]=['两个目标均未显示优于所选启发式的系统性能。99.7%动态配置与原另一档启发式在验证及测试上逐查询行为等价；99.75%配置保留少量晚期例外，但测试总体Recall没有超过启发式。因此，相对固定LR的小幅P95改善不能归因于模型判断优于规则。','']
dynamic_grid=[r for r in initial if r['group']=='ProgressLR']
text[-2:-2]=[f'动态网格的最高验证Recall为{max(r["recall"] for r in dynamic_grid):.2f}%。本轮已包含早期完全不剪枝的基础阈值0，但仅测试4种晚期触发条件与3档晚期阈值；结论限于这个固定网格和L500。','']
by_base=[]
for tau in sorted({r['tau'] for r in dynamic_grid}):
    rs=[r for r in dynamic_grid if r['tau']==tau]
    by_base.append(dict(base_tau=tau,configs=len(rs),max_validation_recall=max(r['recall'] for r in rs),feasible_99_7=sum(r['recall']>=99.7 for r in rs),feasible_99_75=sum(r['recall']>=99.75 for r in rs)))
save('validation_by_base.csv',by_base)
text[-2:-2]=table(['早期阈值','配置数','最高验证Recall %','≥99.7%配置数','≥99.75%配置数'],[[f(r['base_tau'],6),r['configs'],f(r['max_validation_recall'],2),r['feasible_99_7'],r['feasible_99_75']] for r in by_base])+['']
text+=table(['目标','组','配置','基础τ','晚期τ','排名门槛r','展开门槛e','验证Recall %','验证P95 ms'],[[s['target'],s['group'],s['key'] or '无达标配置',f(configs[s['key']]['tau'],6) if s['key'] else '—',f(configs[s['key']]['et_tau'],3) if s['key'] and s['group']=='ProgressLR' else '—',configs[s['key']]['rank'] if s['key'] else '—',configs[s['key']]['exp'] if s['key'] else '—',f(s['validation_recall'],2),f(s['validation_p95_ms'])] for s in selection])
text+=['','## 测试结果','',
'每行指标来自同一锁定配置，P95/QPS取三次中位数。实际Recall并非精确相等；未达标配置不能声明满足目标的性能收益。','']
text+=table(['目标','组','测试Recall %','达标','P95 ms','P95范围 ms','QPS','读取/查询'],[[r['target'],r['group'],f(r['test_recall'],2),r['test_feasible'],f(r['test_p95_ms']),f(r['p95_min_ms'])+'–'+f(r['p95_max_ms']),f(r['test_qps'],1),f(r['test_reads'],2)] for r in results])
text+=['','## 动态策略相对对照','']
text+=table(['目标','对照','双方测试达标','P95降低 %','QPS提升 %','读取减少 %'],[[r['target'],r['baseline'],r['both_test_feasible'],f(r['p95_reduction_pct'],2),f(r['qps_increase_pct'],2),f(r['read_reduction_pct'],2)] for r in comparisons])
text+=['','## 模型贡献与规则等价检查','',
'获选动态策略在验证、测试上分别与原8档L500启发式逐查询比较。只有返回ID、读取、展开、跳过数全部相同，才标记行为等价。即使计时有波动，行为等价的配置也不能被解释成模型判断优于规则。','']
eq=[r for r in equivalence if r['behavior_equivalent']]
if eq:text+=table(['动态配置','划分','行为等价的启发式'],[[r['key'],r['split'],r['heuristic']] for r in eq])
else:text+=['没有获选动态策略与上述8档启发式在返回ID和搜索计数上完全等价；这本身不证明模型有独立性能贡献。']
text+=['']+table(['动态配置','早期阈值','晚期触发候选/查询','晚期额外跳过/查询'],[[k,f(configs[k]['tau'],6),f(float(summary[k]['progress_eligible']),3),f(float(summary[k]['progress_extra_skips']),3)] for k in sorted({s['key'] for s in selection if s['group']=='ProgressLR' and s['key']})])
text+=['','当早期阈值为0时，早期所有候选都保留；若晚期触发数与额外跳过数相等，则晚期所有触发的未缓存候选都被跳过，没有模型保留的例外。这种情形应解释为规则调度的收益。','']
# Compare fixed query IDs selected by the static LR tail, not each policy's own different tail.
paired=[]
for target in protocol['targets']:
    ss={s['group']:s for s in selection if s['target']==target}
    if not ss['StaticLR']['key']:continue
    def arrays(key):return [np.genfromtxt(D/'runs/test'/f'{key}_r{rep}.csv',delimiter=',',names=True) for rep in range(3)]
    static=arrays(ss['StaticLR']['key']);tail=np.argsort(np.median(np.stack([a['latency_us'] for a in static]),axis=0))[-50:]
    query_rows=[dict(query=q,in_static_tail=bool(q in tail)) for q in range(1000)]
    for group in ['StaticLR','ProgressLR','Heuristic','DiskANN']:
        key=ss[group]['key']
        if not key:continue
        a=arrays(key)
        for col in ['latency_us','hits','reads','io_rounds','io_us','progress_extra_skips']:
            values=np.median(np.stack([x[col] for x in a]),axis=0)
            for q in range(1000):query_rows[q][f'{group}_{col}']=float(values[q])
        med=lambda col:np.median(np.stack([x[col] for x in a]),axis=0)[tail]
        paired.append(dict(target=target,group=group,key=key,queries=50,mean_ms=float(med('latency_us').mean()/1000),io_ms=float(med('io_us').mean()/1000),reads=float(med('reads').mean()),io_rounds=float(med('io_rounds').mean()),hits=float(med('hits').sum()),progress_eligible=float(med('progress_eligible').mean()),progress_extra_skips=float(med('progress_extra_skips').mean())))
    save(f'paired_queries_{target}.csv',query_rows)
save('paired_static_tail.csv',paired)
text+=['','## 固定LR最慢5%查询上的配对比较','',
'每个目标按固定阈值LR三次逐查询中位耗时选出50条查询，所有方法在同一批ID上比较。该表只作测试后诊断，没有参与策略选择。','']
text+=table(['目标','组','平均延迟 ms','Reader ms','读取/查询','I/O轮/查询','50条总命中','晚期额外跳过/查询'],[[r['target'],r['group'],f(r['mean_ms']),f(r['io_ms']),f(r['reads'],2),f(r['io_rounds'],2),f(r['hits'],0),f(r['progress_extra_skips'],2)] for r in paired])
text+=['','晚期额外跳过指在该时刻基础阈值会保留、提高阈值后被跳过的候选数，不是离线推算的独立节点因果收益。','',
'## 实现与核验','',
'修改仅在本目录的独立runner和DiskANN副本中，原正式实验二进制与LR权重不变。固定LR关闭策略时，返回ID、读取、展开、跳过数与原实验逐查询一致。两个退化检查使用base_tau=0、late_tau=1.01，返回ID及上述计数必须与相应原启发式完全一致。全部重复核对返回ID/离散计数，Recall由返回ID和GT重算；reader字节与Linux read_bytes一致，且无交换内存。详见verification.json。','',
'## 解释边界','',
'此次是已经分析过的原留出测试集上的后续探索，未使用新的盲测集。参数选择只读取验证结果，但测试结果不能作为独立外部复现。固定L500的比较用于隔离策略效果，不等于跨L完整前沿。有限网格与三次计时不能证明统计显著性或模型普遍优越。若动态策略失去测试Recall，则应明确判作未验证成功；不进一步用测试集调参。','',
'复现：依次运行build.py、run.py、report.py。完整网格和源文件hash见PROTOCOL.json；测试前选择锁见SELECTION_LOCK.json；validation_initial.json、validation_finalists.csv与test_summary.csv保留全部选择依据。','']
(D/'REPORT.md').write_text('\n'.join(text))
assert load('verification.json')['passed']
dump=load('MEASUREMENTS_COMPLETE.json')|dict(report='REPORT.md',analysis_complete=True)
(D/'COMPLETE.json').write_text(json.dumps(dump,indent=2)+'\n')
print('REPORT COMPLETE')
for r in results:print(r)
