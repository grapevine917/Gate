from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;LG=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((R/'PROTOCOL.json').read_text());assert all(sha(Path(p))==v for p,v in protocol['original_hashes'].items())
audit=json.loads((R/'constraint_audit.json').read_text());assert len(audit)==4 and all(x['constraints_verified'] for x in audit.values())
report=R/'REPORT.md';s=report.read_text();s+='''
## 结果解读与约束审计

本次网格下，四个相同Recall插值点上，三种约束的SSD请求、展开次数和延迟均高于原MLP。以Recall=99.5%的插值估计为例，原MLP为76.414请求/q，max_skip_1为92.408（+20.9%），max_skip_2为82.847（+8.4%），keep_best_1为79.820（+4.5%）。这些是有限网格的探索性估计，不能等同可部署单阈值配置。

也存在严格相同实测Recall的对照：验证目标99.3%选择的MLP与max_skip_2，在测试均为99.400%，分别为69.197与79.543请求/q，78.587与88.933展开/q；三轮延迟中位数分别为4945.27与6537.92µs/q。这个配对同样未显示联合保护的成本收益。

在极端阈值1.01下，对全部300条测试查询逐beam重建有效决策，确认Skip上限/至少Keep一个约束成立，且有效Skip数与原生计数相同；每个cache-miss仅调用一次模型；审计模式与计时模式最终ID相同。详见constraint_audit.json。

这说明简单的beam数量约束在本次设置下主要增加了保留成本，并未显示有利的Recall–成本权衡；不能据此排除利用节点关系的更复杂联合策略。最高分保护的队列复制开销计入延迟，其延迟差也包含这一实现成本。

![Recall与成本曲线](curves.svg)

复现：`bash experiments/beam_constraints/run.sh`。已有原始输出会复用；需要全新计时应在新实验目录运行或先归档本实验runs目录。
'''
report.write_text(s)
files=[R/'build.py',R/'run.py',R/'audit.py',R/'plot.py',R/'finalize.py',R/'collect.cpp',R/'native/collect',R/'native/pq_flash_index.cpp',R/'native/online_gate.h',R/'native/beam_policy.h']
(R/'PROVENANCE.json').write_text(json.dumps({str(p.relative_to(LG)):sha(p) for p in files},indent=2)+'\n')
p=R/'COMPLETE.json';c=json.loads(p.read_text());c['constraint_audit_passed']=True;c['report']='REPORT.md';p.write_text(json.dumps(c,indent=2)+'\n')
