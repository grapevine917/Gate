"""Print pure training-L100 versus training-L300 at fixed inference-L100."""
from pathlib import Path
import csv,json
import numpy as np
D=Path(__file__).resolve().parent
def load(n):return json.loads((D/n).read_text())
def save(name,rows):
    with (D/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]
def label(r):return r['family'] if r['kind'] in [0,1] else f'{r["family"]}（训练L={r["training_L"]}，'+('动态）' if r['kind'] in [4,5] else '静态）')
assert load('verification.json')['passed']
rows=load('summary.json');train=load('TRAIN_COMPLETE.json');detail=load('DYNAMIC_SELECTION.json')
text=['# SIFT1M P0-A：推理L100，比较纯L100与纯L300训练','',
'所有实验推理固定L=100、Beam=4、K=10、4查询线程、10,000节点BFS缓存；完整100万条128维SIFT索引。同一1000条测试查询，8组按8种循环顺序各重复2次，共16次/组。DiskANN与Heuristic各只有一份共享对照，以下为16次中位数。','',
'## 训练来源','',
'此前“宽队列新模型”是在L=30、50、75、100、160、300、500、1000上混合训练，不能标成纯L300。本次为满足单独比较，两组均按同一协议从头重训：纯L100和纯L300；并非复用历史L100或混合L权重。',
'两组使用相同1200条训练查询、相同的每查询候选样本数，各76800样本；每组的MLP/LR共享特征、样本和retain代理标签。训练中200条查询仅用于轮次选择，再按选定轮次拟合全部1200条。MLP为17-128-64-1，LR为17维线性模型；最多80轮、同一初始化种子和优化流程，各自独立选择最佳轮次。',
'单L训练中的常量列（包括候选队列容量和L）没有可学习的变化，因此其首层/线性权重固定为0。这样既不改变训练域的预测，也避免把L300→L100时常量特征的任意偏移误当作模型能力。两组使用相同处理，其他有变化的状态特征保持原样。','']
text+=table(['训练L','候选样本','训练查询','MLP选中轮次','常量列','LR原生误差','MLP原生误差'],[[r['training_L'],r['samples'],r['queries'],r['MLP_best_epoch'],str(r['constant_columns']),f'{r["native_errors"]["LR"]:.2g}',f'{r["native_errors"]["MLP"]:.2g}'] for r in train['training']])
text+=['','## 测试结果：全部推理L100','',
'静态阈值全部按各模型在同一验证L100轨迹上的retain-FNR1%标定（cache-miss分母，缓存命中固定保留）。动态阈值只为训练L300组测量；按验证最终Recall不低于对应静态模型，再选最低P95。参数在测试前锁定。代理FNR和验证Recall都不保证测试最终Recall相等。','']
text+=table(['方法','训练L','策略','测试Recall %','P95 ms','P95范围 ms','QPS','KiB/查询','跳过/查询','动态额外跳过/查询'],[[r['family'],r['training_L'] or '—','动态' if r['kind'] in [4,5] else '静态' if r['kind'] in [2,3] else '对照',f'{r["recall"]:.2f}',f'{r["p95_ms"]:.3f}',f'{r["p95_min_ms"]:.3f}–{r["p95_max_ms"]:.3f}',f'{r["qps"]:.1f}',f'{r["physical_bytes"]/1024:.2f}',f'{r["skips"]:.2f}',f'{r["extra_skips"]:.2f}'] for r in rows])
effects=[]
for family in ['MLP','LR']:
    a=next(r for r in rows if r['family']==family and r['training_L']==100 and r['kind'] in [2,3]);b=next(r for r in rows if r['family']==family and r['training_L']==300 and r['kind'] in [2,3]);c=next(r for r in rows if r['family']==family and r['kind'] in [4,5])
    for baseline,r,comparison in [(a,b,'训练L300静态 vs 训练L100静态'),(b,c,'训练L300动态 vs 训练L300静态')]:effects.append(dict(family=family,comparison=comparison,recall_delta_pp=r['recall']-baseline['recall'],p95_reduction_pct=100*(1-r['p95_ms']/baseline['p95_ms']),qps_increase_pct=100*(r['qps']/baseline['qps']-1),IO_reduction_pct=100*(1-r['physical_bytes']/baseline['physical_bytes'])))
save('effects.csv',effects)
text+=['','## 对比变化','',
'实际测试Recall不同的行不能解释为严格同Recall收益；P95范围是重复观测范围，不是置信区间。','']
text+=table(['模型','比较','Recall变化 pp','P95降低 %','QPS提升 %','读取降低 %'],[[r['family'],r['comparison'],f'{r["recall_delta_pp"]:+.2f}',f'{r["p95_reduction_pct"]:.2f}',f'{r["qps_increase_pct"]:.2f}',f'{r["IO_reduction_pct"]:.2f}'] for r in effects])
bykey={r['key']:r for r in rows}
text+=['','## 动态配置与验证质量','']
text+=table(['模型','基础阈值','晚期阈值','rank门槛','已展开门槛','静态验证Recall','动态验证Recall','验证质量达标'],[[r['family'],f'{bykey[r["key"]]["tau"]:.6f}',f'{bykey[r["key"]]["late_tau"]:.6f}',bykey[r['key']]['rank'],bykey[r['key']]['exp'],f'{r["static_validation_recall"]:.2f}',f'{r["dynamic_validation_recall"]:.2f}',r['quality_matched']] for r in detail])
# Tail comparisons use the same query IDs for the corresponding train300 static model.
tails=[]
for family in ['MLP','LR']:
    static=next(r for r in rows if r['family']==family and r['training_L']==300 and r['kind'] in [2,3]);dynamic=next(r for r in rows if r['family']==family and r['kind'] in [4,5]);data={}
    for r in [static,dynamic,rows[0],rows[1]]:
        aa=[np.genfromtxt(D/'runs/test'/f'{r["key"]}_r{rep}.csv',delimiter=',',names=True) for rep in range(16)]
        data[r['key']]={k:np.median(np.stack([a[k] for a in aa]),axis=0) for k in ['latency_us','io_us','reads','hits']}
    ix=np.argsort(data[static['key']]['latency_us'])[-50:]
    for r in [static,dynamic,rows[0],rows[1]]:
        a=data[r['key']];tails.append(dict(tail_defined_by=family,key=r['key'],label=label(r),queries=50,mean_ms=float(a['latency_us'][ix].mean()/1000),reader_ms=float(a['io_us'][ix].mean()/1000),reads=float(a['reads'][ix].mean()),hits=int(a['hits'][ix].sum())))
save('paired_tail.csv',tails)
text+=['','## 静态训练L300模型最慢50条查询上的配对诊断','']
text+=table(['尾部依据','方法','平均延迟 ms','Reader ms','读取/查询','总命中'],[[r['tail_defined_by'],r['label'],f'{r["mean_ms"]:.3f}',f'{r["reader_ms"]:.3f}',f'{r["reads"]:.2f}',r['hits']] for r in tails])
text+=['','## 核验与边界','',
'所有重复返回ID、读取、展开、跳过和动态计数一致；Recall由返回ID与GT复算。reader字节与Linux read_bytes一致；无内存交换。原生/启发式返回ID与上一轮相同L100配置一致。输入和权重哈希均未变化。I/O为WSL2客户机块层流量。',
'本轮用已分析过的留出查询，属于探索性复测；只有一个训练种子，16次计时不包含训练随机性。两组最大训练轮次和选轮流程相同，实际最佳轮次不同，结论是同一训练流程下的训练L比较，不是固定迭代次数的纯梯度消融。训练L100的动态结果按请求省略。',
'原始结果见summary.csv、all_runs.json；训练信息见TRAIN_COMPLETE.json；参数锁见SELECTION_LOCK.json；核验见verification.json。复现依次运行prepare_train.py、evaluate.py、report.py。旧模型和旧实验结果保持不变。','']
if (D/'IO_REJECTIONS.json').exists():
    rejected=load('IO_REJECTIONS.json');text+=['## 作废记录审计','',f'共{len(rejected)}条记录因内核/reader读取量不一致而作废并归档，未纳入任何表格；按相同配置重测，取首个通过核验的记录，不按耗时快慢选重试结果。原始记录及原因见IO_REJECTIONS.json和rejected_io/。所有128条有效测试均满足严格I/O一致性。','']
(D/'REPORT.md').write_text('\n'.join(text))
complete=load('MEASUREMENTS_COMPLETE.json')|dict(analysis_complete=True,report='REPORT.md');(D/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n');(D/'STATUS.json').write_text(json.dumps(dict(complete=True,phase='complete',report='REPORT.md'))+'\n')
print('REPORT COMPLETE')
for r in rows:print(label(r),r['recall'],r['p95_ms'],r['qps'],r['physical_bytes']/1024)
