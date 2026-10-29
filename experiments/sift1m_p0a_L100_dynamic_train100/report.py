"""Append saved train-L100 dynamic policies to the previous P0-A table."""
from pathlib import Path
import csv,json
D=Path(__file__).resolve().parent;A=D.parent/'sift1m_p0a_L100_train_compare'
def load(n):return json.loads((D/n).read_text())
def save(name,rows):
    with (D/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]
assert load('verification.json')['passed']
P=load('PROTOCOL.json');new=load('summary.json');combined=load('combined_summary.json')
ordered=[r for r in combined if r['kind']==0]+[r for r in combined if r['kind']==1]
for L in [100,300]:
    for family in ['MLP','LR']:ordered+=sorted([r for r in combined if r['training_L']==L and r['family']==family],key=lambda r:r['kind'])
text=['# SIFT1M P0-A补充：纯L100训练模型的动态策略','',
'本轮没有重新训练或修改模型，直接复用上一轮保存的models_train100/state.bin与lr.bin。搜索完整100万条128维SIFT索引，推理L=100、Beam4、K10、4查询线程、10,000节点缓存。',
'为MLP、LR各增加8个验证配置。阈值沿用对应模型在L100验证轨迹上的native分数标定；要求验证最终Recall不低于对应静态默认，再选最低P95，测试前锁定。两新增组在同一1000条测试查询上各16次重复，轮换先后位置各8次。',
'原有8组及DiskANN/启发式共享对照直接引用上一轮结果，没有重跑或改写。原有组与新增组并非同期交错测量，因此很小的时延/QPS差异可能包含时间漂移；读取、GT命中和逐查询离散计数另行严格核验。','',
'## 完整十组结果：推理全部L100','',
'每组均为16次中位数；仅“训练L100、动态”两行是本轮新增计时，其他行是引用的既有结果。','']
text+=table(['方法','训练L','策略','Recall %','P95 ms','P95范围 ms','QPS','KiB/查询','动态额外跳过/查询'],[[r['family'],r['training_L'] or '—','动态' if r['kind'] in [4,5] else '静态' if r['kind'] in [2,3] else '对照',f'{r["recall"]:.2f}',f'{r["p95_ms"]:.3f}',f'{r["p95_min_ms"]:.3f}–{r["p95_max_ms"]:.3f}',f'{r["qps"]:.1f}',f'{r["physical_bytes"] /1024:.2f}',f'{r["extra_skips"]:.2f}'] for r in ordered])
effects=[]
for r in new:
    static=next(x for x in combined if x['training_L']==100 and x['family']==r['family'] and x['kind'] in [2,3])
    effects.append(dict(family=r['family'],recall_delta_pp=r['recall']-static['recall'],p95_reduction_pct=100*(1-r['p95_ms']/static['p95_ms']),qps_increase_pct=100*(r['qps']/static['qps']-1),IO_reduction_pct=100*(1-r['physical_bytes']/static['physical_bytes'])))
save('new_vs_static.csv',effects)
text+=['','## 新增动态组相对同权重静态组','',
'该表使用此前静态组计时，非同期对比。实际Recall不同不能当作严格同质量加速。','']
text+=table(['模型','Recall变化 pp','P95降低 %','QPS提升 %','读取降低 %'],[[r['family'],f'{r["recall_delta_pp"]:+.2f}',f'{r["p95_reduction_pct"]:.2f}',f'{r["qps_increase_pct"]:.2f}',f'{r["IO_reduction_pct"]:.2f}'] for r in effects])
bykey={r['key']:r for r in new};detail=load('DYNAMIC_SELECTION.json')
text+=['','## 选择的动态参数','']
text+=table(['模型','基础τ','晚期τ','rank门槛','已展开门槛','静态验证Recall','动态验证Recall','验证质量达标'],[[r['family'],f'{bykey[r["key"]]["tau"]:.6f}',f'{bykey[r["key"]]["late_tau"]:.6f}',bykey[r['key']]['rank'],bykey[r['key']]['exp'],f'{r["static_validation_recall"]:.2f}',f'{r["dynamic_validation_recall"]:.2f}',r['quality_matched']] for r in detail])
text+=['','缓存命中固定保留。所谓晚期是rank和已展开数均严格大于门槛；阈值提高时更积极跳过。若基础阈值与晚期阈值都为0，该配置实际关闭剪枝，不得解释为动态策略收益。','',
'## 核验与限制','',
'两新增组的全部重复返回ID、读取、展开、跳过及动态计数一致；Recall从返回ID与GT复算，reader字节与Linux read_bytes一致，无内存交换。保存模型、runner、原有汇总和查询文件hash在本轮前后完全一致。关闭动作的验证配置复现原生ID和计数。',
'沿用已分析过的留出查询，仅一个训练种子；代理retain-FNR不是最终Recall风险保证。有限8档动态策略不代表全局最优。I/O为WSL2客户机块层流量，不是SSD内部NAND计数。',
'新结果见summary.csv、all_runs.json；完整10组见combined_summary.csv；模型来源、哈希和选择协议见PROTOCOL.json；测试前选择锁见SELECTION_LOCK.json；核验见verification.json。旧实验目录未修改。','']
if (D/'IO_REJECTIONS.json').exists():text += [f'本轮另有{len(load("IO_REJECTIONS.json"))}条I/O不一致记录被作废、归档并按原配置重测，未计入有效结果。','']
(D/'REPORT.md').write_text('\n'.join(text))
done=load('MEASUREMENTS_COMPLETE.json')|dict(analysis_complete=True,report='REPORT.md');(D/'COMPLETE.json').write_text(json.dumps(done,indent=2)+'\n');(D/'STATUS.json').write_text(json.dumps(dict(complete=True,phase='complete',report='REPORT.md'))+'\n')
print('REPORT COMPLETE')
for r in new:print(r['family'],r['recall'],r['p95_ms'],r['qps'],r['physical_bytes']/1024,r['extra_skips'])
