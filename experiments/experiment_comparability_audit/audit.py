"""Audit existing records only; no new training or timed experiments."""
from pathlib import Path
import csv,json,hashlib
E=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def csvrows(p):return list(csv.DictReader(p.open()))
def table(h,rows):return ['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
heur=[];latest=E/'sift1m_p0a_L100_train_compare';new_h=latest/'runs/test/Heuristic_r0.ids'
for folder,label in [('p0_a','早期MLP版'),('p0_a_lr','早期LR版')]:
    rows=csvrows(E/folder/'summary.csv');h=next(r for r in rows if r['method']=='Heuristic');b=next(r for r in rows if r['method']=='Vanilla')
    ids=list((E/folder/'runs').glob('*Heuristic*.ids'));assert ids and all(p.read_bytes()==new_h.read_bytes() for p in ids)
    heur.append(dict(run=label,L=100,recall=float(h['recall_pct']),reads=float(h['requests_per_query']),expansions=float(h['expansions_per_query']),qps=float(h['qps']),qps_min=float(h['qps_min']),qps_max=float(h['qps_max']),baseline_qps=float(b['qps']),ids_equal_latest=True))
s=load(latest/'summary.json');h=next(r for r in s if r['key']=='Heuristic');b=next(r for r in s if r['key']=='DiskANN');rr=[r for r in load(latest/'all_runs.json') if r['key']=='Heuristic']
heur.append(dict(run='最近共享对照',L=100,recall=h['recall'],reads=h['reads'],expansions=h['expansions'],qps=h['qps'],qps_min=min(r['qps'] for r in rr),qps_max=max(r['qps'] for r in rr),baseline_qps=b['qps'],ids_equal_latest=True))
append=E/'sift1m_p0a_L100_dynamic_train100';p=load(append/'PROTOCOL.json');source=str((latest/'summary.json').resolve());assert p['source_hashes'][source]==sha(latest/'summary.json')
combined=load(append/'combined_summary.json');assert next(r for r in combined if r['key']=='Heuristic')==h
cases=[]
def add(name,rows,static_key,dynamic_key,L,base_field='tau',late_field='late_tau'):
    a=next(r for r in rows if r['key']==static_key);b=next(r for r in rows if r['key']==dynamic_key)
    n=lambda r,k:float(r[k])
    cases.append(dict(case=name,L=L,static_tau=n(a,'tau'),dynamic_early_tau=n(b,'tau'),dynamic_late_tau=n(b,late_field),rank=int(b['rank']),exp=int(b['exp']),static_recall=n(a,'recall'),dynamic_recall=n(b,'recall'),recall_delta_pp=n(b,'recall')-n(a,'recall'),static_p95_ms=n(a,'p95_ms'),dynamic_p95_ms=n(b,'p95_ms'),p95_reduction_pct=100*(1-n(b,'p95_ms')/n(a,'p95_ms')),read_increase_pct=100*(n(b,'reads')/n(a,'reads')-1)))
add('100M mixed-L，99.5%质量目标',csvrows(E/'sift100m_progress_gate/test_summary.csv'),'Static_p5','Progress_b5_s1_t0',500,late_field='et_tau')
add('1M mixed-wide-L，推理L300',load(E/'sift1m_wide_p0/aggregate.json'),'LR_L300_p4','DynamicLR_L300_p7',300)
add('1M pure-L100，推理L100',combined,'LR_train100_static','LR_train100_dynamic_p1',100)
assert cases[-1]['dynamic_early_tau']==0 and cases[-1]['dynamic_late_tau']<cases[-1]['static_tau']
out=dict(heuristic_same_L100=heur,latest_table_reuses_same_control=True,representative_LR_policies=cases)
(D/'audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
text=['# 跨轮实验可比性审计','',
'仅核对现有原始记录，没有新增实验。结论：启发式未显示持续增强；动态策略几轮采用的推理L、训练模型、阈值方案和质量约束不同，不能把结果变化直接归因于数据集规模。','',
'## 相同SIFT1M、相同推理L100的启发式','']
text+=table(['轮次','Recall %','读取/查询','展开/查询','QPS中位数','QPS观测范围','原生QPS'],[[r['run'],f'{r["recall"]:.2f}',f'{r["reads"]:.3f}',f'{r["expansions"]:.3f}',f'{r["qps"]:.1f}',f'{r["qps_min"]:.1f}–{r["qps_max"]:.1f}',f'{r["baseline_qps"]:.1f}'] for r in heur])
text+=['','三轮规则都是rank>72且completed>100、保护缓存命中。所有旧启发式重复的返回ID与最近共享对照完全相同，读取数和展开数也相同。QPS并非单调增长：627.6→619.6→634.3。观测范围重叠；原始记录不能区分调度、I/O服务时间、运行顺序与二进制实现差异各自的贡献。不能从中声称启发式规则变强。',
'最近追加动态L100实验没有重测启发式，634.3是同一份历史共享记录，已核对source hash及整条汇总记录相等。L300、不同Recall目标、SIFT100M的启发式数值不进入此趋势表。','',
'## 三个“动态策略提升”的代表性LR结果','']
text+=table(['实验','推理L','静态τ','动态早期τ','动态晚期τ','rank门槛','展开门槛','静态→动态Recall','P95降低 %','读取增加 %'],[[r['case'],r['L'],f'{r["static_tau"]:.6f}',f'{r["dynamic_early_tau"]:.6f}',f'{r["dynamic_late_tau"]:.6f}',r['rank'],r['exp'],f'{r["static_recall"]:.2f}→{r["dynamic_recall"]:.2f}',f'{r["p95_reduction_pct"]:.2f}',f'{r["read_increase_pct"]:.2f}'] for r in cases])
text+=['','score小于阈值才跳过，阈值越高越激进：',
'- 100M例子保持早期阈值不变、晚期显著提高，因此更积极剪枝；Recall从99.63%降至99.57%，满足共同99.5%下限，但不是严格同Recall的16.7%收益。',
'- 1M推理L300例子早期降低阈值、晚期提高；总体Recall同为99.94%，平均读取增加，但慢查询读取减少，因而P95下降约8.4%。',
'- 最新1M推理L100例子早期不剪枝，晚期阈值0.01291仍低于静态阈值0.02496。在相同候选状态上，它始终比静态策略保守。更高Recall和更多读取正是预期交换，不能据此判定1M不适合动态阈值。MLP也相同：静态0.04252，动态早期0、晚期0.02804。',
'这三种是不同工作点，不能视为同一动态控制器只改变数据量的消融。','',
'## 其他未控制因素','',
'同为L100时，原生DiskANN在SIFT1M的测试Recall为99.67%，在SIFT100M为96.95%，说明工作点检索难度不同。此外1M原索引使用float向量，100M使用uint8；两者PQ均34字节。缓存都设10000节点，名义覆盖比是1%与0.01%（不等于实际命中率）。训练L范围、模型权重、训练轮次及标定参考L也变过，因此这些结果不是纯数据量消融。',
'早期LR阈值标定分母包含全部retain候选并强制缓存命中保留；新纯L对比按cache-miss retain标定。即使都写1% retain-FNR，也不意味着相同跳过数量或相同最终Recall。',
'最新L100启发式Recall99.52%，动态L100两模型为99.65%。在1000查询×Top10上相差13个命中；直接比较吞吐不能证明启发式在同质量下更好。',
'近期新增动态组使用此前共享静态/对照时间，没有同期交错重测；它节省了重复对照，但不能支持很小的跨批次计时因果结论。','',
'## 能得出的结论','',
'新权重、工作点和质量要求的变化，加上计时批次差异，足以使表面差距变化。目前没有证据证明启发式算法随着实验次数持续变强，也没有足够证据把动态收益差异归因于1M/100M规模本身。若要隔离规模影响，需另外预先冻结同一策略、训练流程和质量口径，并在同期交错对照中测量；本审计不自动启动该实验。','']
(D/'REPORT.md').write_text('\n'.join(text));print('AUDIT COMPLETE',D/'REPORT.md')
for r in cases:print(r)
