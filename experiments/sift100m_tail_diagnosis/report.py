"""Render the diagnostic tables with explicit timing and causal limitations."""
from pathlib import Path
import csv,json
import numpy as np

D=Path(__file__).resolve().parent
def rows(name):return list(csv.DictReader((D/name).open()))
summary=rows('summary.csv');cohorts=rows('cohorts.csv');paired=rows('paired_cohorts.csv');routes=rows('route_comparison.csv')
def v(r,k):return float(r[k])
def fmt(r,k,d=3):return f'{v(r,k):.{d}f}'
def table(headers,rs):return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs]
full={r['key']:r for r in summary if r['mode']=='full'}
replay={r['key']:r for r in summary if r['mode']=='replay'}
manifest=json.loads((D/'manifest.json').read_text())
text=['# SIFT100M：尾延迟诊断','',
'主要发现：LR的收益主要体现在平均读取量与吞吐；在其最慢5%查询上，读取量和I/O轮数反而高于启发式。同决策重放绕过Gate计算后，尾部差距仍在。平均I/O批量效率损失与大量路径绕行均未得到本次数据支持。','',
'使用验证集提前锁定的最低读取量配置，质量目标99.5%与99.7%；另加同L500的未剪枝控制。每配置1000条原测试查询、4线程、3次重复。诊断结果独立于原正式报告，不覆盖原结果。',
'所有模式逐查询返回ID和离散计数必须与原实验完全一致，reader字节必须与Linux read_bytes相同，且不允许交换内存。具体核验见 verification.json。',
'','## 完整搜索：读取量与串行轮数','']
text+=table(['配置','Recall %','P95 ms','QPS','读取/查询','I/O轮/查询','读取/I/O轮','空轮/查询'],[[k,fmt(r,'recall',2),fmt(r,'p95_ms'),fmt(r,'qps',1),fmt(r,'reads',2),fmt(r,'io_rounds',2),fmt(r,'reads_per_io_round',2),fmt(r,'empty_rounds',2)] for k,r in full.items()])
text+=['','P95按每次运行计算后取三次中位数；读取、轮数均为逐查询量。I/O轮指一次同步reader调用，Beam=4。空轮指该次搜索迭代没有未缓存节点读取。','','## 读取量在查询之间如何分布','']
text+=table(['配置','最小读取','P25','中位读取','P75','P95读取','最大读取','延迟-读取相关系数'],[[r['key'],fmt(r,'reads_min',1),fmt(r,'reads_p25',1),fmt(r,'reads_p50',1),fmt(r,'reads_p75',1),fmt(r,'reads_p95',1),fmt(r,'reads_max',1),fmt(r,'latency_reads_correlation')] for r in rows('query_distribution.csv')])
text+=['','LR的平均读取量下降不意味着尾部读取量也下降。启发式按展开进度与排名执行规则，本实验观察到其读取量分布集中；LR在不同查询之间剪枝强度差异很大。相关系数仅用于描述。','','## 相同决策重放：绕过Gate计算','',
'profile模式先记录每个候选的跳过决策；replay直接消费该决策序列，绕过排名扫描、状态构造和模型评分。逐查询ID、读取、展开、跳过数与full一致。full和replay顺序在重复间交换。']
text+=['']+table(['配置','Full P95 ms','Replay P95 ms','P95降低 %','Full范围 ms','Replay范围 ms'],[[k,fmt(full[k],'p95_ms'),fmt(r,'p95_ms'),f'{100*(1-v(r,"p95_ms")/v(full[k],"p95_ms")):.2f}',f'{fmt(full[k],"p95_min_ms")}–{fmt(full[k],"p95_max_ms")}',f'{fmt(r,"p95_min_ms")}–{fmt(r,"p95_max_ms")}'] for k,r in replay.items()])
text+=['','该干预去掉整个Gate决策路径，而非只去掉LR点积。replay仍有决策带读取及分支开销；计时差包含运行波动，不能当作精确的模型耗时或显著性证明。','','## 在同一批最慢查询上配对比较','',
'每个质量目标分别以LR的逐查询三次中位延迟选出最慢50条查询，再让所有方法在相同ID上比较。不是把不同方法各自不同的最慢查询混在一起。','']
for target in [99.5,99.7]:
    ps=[r for r in paired if float(r['target'])==target and r['cohort']=='LR_tail']
    text += [f'### 质量目标 {target}%：LR最慢50条','']
    text+=table(['方法','平均延迟 ms','Reader ms','非Reader ms','读取/查询','I/O轮/查询','读取/轮'],[[r['family'],fmt(r,'mean_ms'),fmt(r,'io_ms'),fmt(r,'non_io_ms'),fmt(r,'reads',2),fmt(r,'io_rounds',2),fmt(r,'reads_per_round',2)] for r in ps])
    lr=next(r for r in ps if r['family']=='LR');heur=next(r for r in ps if r['family']=='Heuristic')
    delta=v(lr,'mean_ms')-v(heur,'mean_ms');io_delta=v(lr,'io_ms')-v(heur,'io_ms')
    text+=['',f'同查询LR减Heuristic：总延迟 {delta:+.3f} ms，Reader时间 {io_delta:+.3f} ms，非Reader残差 {delta-io_delta:+.3f} ms。','']
text+=['## LR最慢50条的分项计时','']
lrco=[r for r in cohorts if r['key'].startswith('LR') and r['cohort']=='own_slowest_5pct']
text+=table(['配置','排名扫描 μs','特征构造 μs','评分 μs','三项合计 μs','Full平均 ms','Replay平均 ms','非Reader减少 ms'],[[r['key'],fmt(r,'rank_us',1),fmt(r,'feature_us',1),fmt(r,'score_us',1),f'{sum(v(r,k) for k in ["rank_us","feature_us","score_us"]):.1f}',fmt(r,'full_mean_ms'),fmt(r,'replay_mean_ms'),fmt(r,'full_minus_replay_non_reader_ms')] for r in lrco])
text+=['','这些分项来自独立profile模式，包含每候选时钟读取的扰动；评分项还包含调用/缓存分支，不能与离线纯评分微基准直接等同。profile也记录节点轨迹，其总延迟不用于正式性能结论。Reader时间包含同步提交与完成等待，不能单独解释为SSD服务时间。非Reader残差还包含调度与其他计算；原cpu_us只覆盖部分计算，不参与加法分解。','','## 搜索路径与质量损失','',
'以下与同L500未剪枝DiskANN比较，排除不同L造成的直接混淆。节点轨迹来自profile第0次；返回结果已与原正式实验核对。','']
rr=[]
for key in replay:
    rs=[r for r in routes if r['key']==key]
    rr.append([key,f'{np.mean([float(r["expanded_outside_same_L_baseline"]) for r in rs]):.2f}',sum(float(r['expanded_outside_same_L_baseline'])>0 for r in rs),f'{np.mean([float(r["baseline_expansions_skipped"]) for r in rs]):.2f}',sum(float(r['hits_delta'])<0 for r in rs),sum(float(r['hits_delta'])>0 for r in rs),f'{sum(float(r["hits_delta"]) for r in rs):.0f}'])
text+=table(['配置','基线未展开的新节点/查询','含新节点的查询数','基线展开而本组跳过/查询','命中减少查询数','命中增加查询数','总命中差'],rr)
text+=['','出现基线未展开的新节点能够证明轨迹变化；净命中损失证明剪枝具有质量代价。两者不能定位某个必需桥接节点，也不能证明全部额外轮数由误判引起。若要识别具体桥接误剪，仍需逐节点恢复干预。','','## 机制解释及证据边界','',
'代码在选择候选后先增加num_seen，再决定是否跳过；skip的continue不退还Beam名额。但此次测量没有支持“LR破坏平均I/O批量效率”的猜测：LR与启发式每个有I/O的轮次都接近4次读取，LR平均I/O轮更少。更多无I/O空轮不等于更多磁盘等待，不能混淆。相关代码见本目录native/pq_flash_index.cpp的num_seen++与history_skip分支。',
'当前证据指向查询之间的收益分配：LR在大量查询上省读取、提高平均吞吐，但在最慢查询上保留更多节点。启发式在这些查询上继续按固定进度规则剪枝，因而I/O轮数与尾部读取量更低；这伴随逐查询命中差异。相同决策replay检验表明Gate计算本身能解释多少差距，剩余差距不能归咎于LR点积。',
'这些数据定位到“尾部查询没有得到足够剪枝收益”，但没有证明对这些查询强制加大剪枝仍能满足整体质量要求。模型保守性、标签目标与尾延迟目标不一致，以及训练/部署轨迹偏移仍是待区分的解释，不作为已证明原因。',
'严格目标质量在独立测试中存在轻微差异：99.5%目标Heuristic/LR测试Recall分别99.64%/99.63%，99.7%目标分别99.71%/99.72%；不是精确相等Recall。验证锁定的DiskANN 99.7%目标在测试仅99.68%，不得当作满足99.7%质量要求的赢家。',
'样本限于原冻结1000条测试查询及本机WSL2环境。三次重复用于检查稳定性，不构成跨设备推广或统计显著性保证。','','## 文件','',
'- summary.csv：每配置/模式三次汇总与P95范围。',
'- paired_cohorts.csv、paired_queries_*.csv：固定查询集合的配对数据。',
'- cohorts.csv：全体与各自最慢5%的计时、轮数分解。',
'- route_comparison.csv：同L节点集合与逐查询命中差。',
'- manifest.json、verification.json：配置来源、输入hash和逐模式一致性核验。',
'- runs/：逐查询CSV、返回ID、节点轨迹与资源记录。','']
(D/'REPORT.md').write_text('\n'.join(text))
verification=json.loads((D/'verification.json').read_text())
assert verification['passed'] and all(verification['original_input_hashes_unchanged'].values())
(D/'COMPLETE.json').write_text(json.dumps(dict(complete=True,configurations=len(full),repeats=3,configuration_mode_repeats=len(summary)*3,queries_per_run=1000,verification_passed=True,report='REPORT.md'),indent=2)+'\n')
print('REPORT COMPLETE',D/'REPORT.md')
