"""Publish the completed P0-A while the larger P0-B grid is still running."""
from pathlib import Path
import csv,json
D=Path(__file__).resolve().parent
rows={r['key']:r for r in csv.DictReader((D/'p0_a_summary.csv').open())};selection=json.loads((D/'P0A_SELECTION_LOCK.json').read_text());train=json.loads((D/'TRAIN_COMPLETE.json').read_text())
labels={'DiskANN':'DiskANN','Heuristic':'Heuristic','MLP':'静态 MLP','LR':'静态 LR','DynamicMLP':'动态 MLP','DynamicLR':'动态 LR'}
text=['# SIFT1M 宽候选队列 P0-A：已完成','',
'完整100万条128维SIFT索引；L300（原P0-A为L100）、Beam4、K10、4查询线程、10,000节点BFS缓存。同一1000条测试查询，六方法循环顺序各2次，共12次；各表时间为12次中位数。',
f'MLP与LR从头训练，共享{train["prune_samples"]}个mixed-L样本，覆盖30至1000的8档L。动态版本与相应静态版本使用相同权重。对照只测一套；原始数据同时供P0-B复用。',
'静态阈值按验证L300 retain-FNR1%标定；动态参数要求验证Recall不低于对应静态默认，再选最低P95；所有选择在测试前锁定。测试实际Recall可能不同。','',
'|方法|Recall %|P95 ms|P95范围 ms|QPS|KiB/查询|跳过/查询|动态额外跳过/查询|',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for c in selection:
    r=rows[c['key']];assert int(r['repeats'])==12
    text.append('|'+ '|'.join([labels[r['family']],f'{float(r["recall"]):.2f}',f'{float(r["p95_ms"]):.3f}',f'{float(r["p95_min_ms"]):.3f}–{float(r["p95_max_ms"]):.3f}',f'{float(r["qps"]):.1f}',f'{float(r["physical_bytes"])/1024:.2f}',f'{float(r["skips"]):.2f}',f'{float(r["progress_extra_skips"]):.2f}'])+'|')
text+=['','全部重复返回ID、读取、展开、跳过计数一致；Recall由ID和GT复算，reader字节与Linux read_bytes一致，无内存交换。I/O为WSL2客户机块层计数。','',
'固定L的比较不等于同Recall最优配置优势；还需结合P0-B的L—阈值扫描。完整P0-B仍在运行，进度见STATUS.json。原始结果见p0_a_summary.csv、p0_a_runs.json和runs/test/。','']
(D/'P0_A_REPORT.md').write_text('\n'.join(text));print('P0-A REPORT WRITTEN')
