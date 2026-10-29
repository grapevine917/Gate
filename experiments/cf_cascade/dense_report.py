from pathlib import Path
import json,hashlib,html
import numpy as np
BASE=Path(__file__).resolve().parent;R=BASE/'dense';LG=BASE.parents[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((R/'PROTOCOL.json').read_text());results=json.loads((R/'exact_matched_results.json').read_text());cmp=json.loads((R/'paired_comparisons.json').read_text());check=json.loads((R/'verification.json').read_text());val=json.loads((R/'validation.json').read_text());test=json.loads((R/'test.json').read_text())
lines=['# 原始MLP / CF风险二次剪枝 / 原分数二次剪枝：严格同Recall对照','','## 设置','','冻结已有原始17维MLP和反事实双输出模型，不重新训练。SIFT1M，沿用上一实验互斥的500/250/250查询划分，仅在原验证及测试各250条上评估。L100、beam4、K10、4线程、10000节点静态BFS缓存。原DiskANN源码和原模型不修改，独立库只加入Keep后的二阶段分支。', '',f"二阶段第一阶段阈值固定为生成E4-1标签时的原始阈值 {protocol['first_stage_threshold']:.17g}，缓存命中始终保护。",'','三组：','','1. **MLP**：原分数 `p<tau`，63个单阶段阈值。','2. **CF cascade**：先原MLP；只在cache-miss且原MLP Keep时调用CF，若 `risk<rho` 再Skip；57个risk阈值，不使用收益门槛。CF输入仍只有相同17维状态。','3. **Score cascade**：先原MLP；只在Keep时用同一次推理的原分数判断 `p<tau2`；63个二次阈值，不重复推理。','','Score cascade 数学上等价于单阶段阈值 `max(tau1,tau2)`，它控制了“二次剪枝只是更激进”这一解释。实际动态搜索中已逐阈值核验最终ID、Recall、请求和展开计数完全一致。','','## 相同Recall的定义与评价范围','','预先声明全网格，再分别运行完整验证/测试曲线。主比较只使用三组在同一250条测试查询上**最终GT命中总数严格相等**的工作点；Recall=命中数/2500，不用小数四舍五入相等、不用相同阈值，也不用相同验证Recall下限替代。每个精确命中数处，分别取本组网格中SSD请求最少的配置。','','这是在固定评估集上建立实测同Recall曲线的描述性对照：利用测试曲线命中数进行工作点匹配并在点内选低成本配置，因此不是完全独立的部署参数选择。两个模型权重保持不变，本轮网格在本轮测试扫描前冻结；未按测试训练或修改模型。验证集精确匹配后冻结的配置及其测试结果另存 `validation_locked_test_results.csv`，没有将其不同测试Recall冒充相同。没有使用插值结果、决策随机混合或逐query配额。','','## 严格相同实测Recall：SSD请求比较','','SSD请求和展开均为每query均值；延迟为三轮均值延迟的中位数。CF请求变化相对同Recall的原MLP；负数表示节省。','','|实测Recall%|GT命中总数/2500|MLP请求/q|CF请求/q|原分数二次请求/q|CF请求变化|CF二次Skip/q|','|---:|---:|---:|---:|---:|---:|---:|']
for c in cmp:lines.append(f"|{c['recall']:.3f}|{c['total_hits']}|{c['baseline_reads']:.3f}|{c['cf_reads']:.3f}|{c['score_cascade_reads']:.3f}|{c['cf_ssd_change_pct']:+.2f}%|{c['cf_secondary_skips']:.3f}|")
lines+=['','## 配置、展开与延迟','','|实测Recall%|策略|第一阶段tau|二阶段阈值|请求/q|展开/q|延迟µs/q|二阶段调用/q|','|---:|---|---:|---:|---:|---:|---:|---:|']
for r in results:
 secondary='—' if r['family']=='MLP' else f"{r['secondary']:.7g}"
 lines.append(f"|{r['recall']:.3f}|{r['family']}|{r['tau']:.7g}|{secondary}|{r['reads']:.3f}|{r['expansions']:.3f}|{r['latency_us']:.2f}|{r['secondary_calls']:.3f}|")
lines+=['','## 配对查询SSD差的95%区间','','2000次配对query bootstrap，CF减去原MLP。区间条件于当前选定配置，未在每个bootstrap样本重新匹配Recall，也不包含模型训练或曲线选点不确定性。','','|相同实测Recall%|请求差/q|95%CI|','|---:|---:|---|']
for c in cmp:
 lo,hi=c['paired_read_delta_ci95'];lines.append(f"|{c['recall']:.3f}|{c['paired_read_delta']:+.3f}|[{lo:+.3f}, {hi:+.3f}]|")
nontrivial=[c for c in cmp if c['cf_secondary_skips']>0]
wins=[c for c in nontrivial if c['cf_reads']<c['baseline_reads']]
lines+=['','## 结果解读','',f"共找到{len(cmp)}个三组精确相同Recall的实测工作点，其中{len(nontrivial)}个CF实际发生二次Skip；CF在{len(wins)}个点的SSD请求少于原MLP。完整工作点全部列出，未只挑有利点。"]
if nontrivial:
 best=min(nontrivial,key=lambda c:c['cf_ssd_change_pct']);worst=max(nontrivial,key=lambda c:c['cf_ssd_change_pct'])
 lines.append(f"CF相对原MLP的请求变化范围为{best['cf_ssd_change_pct']:+.2f}%至{worst['cf_ssd_change_pct']:+.2f}%。这只说明当前冻结模型和有限网格下的实测曲线，不能据单个点宣称普遍优胜或在未观察查询上保持同样Recall。")
lines+=['','原分数二次剪枝没有新的节点风险信息；其策略等价性已由全网格动态路径检验。CF二次策略若节省请求，则需以本表的同Recall成本差判断，不能由风险AUC提高、相同阈值或相同验证Recall下限推导。','','Keep-only门控改善了反事实模型在原Skip节点上无标签却被调用的问题，但二次Skip仍会改变后续轨迹，不能把单节点收益标签相加为联合收益。原E4查询池此前已被多次观察；这些结果属于探索性对照。SSD是提交的请求数。LinuxAlignedFileReader使用O_DIRECT，绕过OS页缓存；每配置先预热同一查询集，但这不等同OS页缓存命中，也不代表冷设备性能。','','## 核验与复现','',f"验证和测试各{check['test_score_equivalences']}个原分数二阶段配置，均与对应单阶段有效阈值的最终ID和请求/展开计数一致。原E4固定阈值与既有E1程序最终ID一致。每个query原生断言二次调用数恰好等于第一阶段Keep的非缓存节点数、第一和第二阶段Skip和等于实际Skip总数。所有Recall独立从保存ID与GT重算；选定同Recall配置三轮ID及计数一致，原始源码及模型哈希未变。",'','复现：`bash experiments/cf_cascade/run.sh`。已有完整曲线结果会在输入锁一致时复用。', '', '![实测Recall与SSD请求曲线](curves.svg)']
counts={f:sum(c['family']==f for c in json.loads((R/'configs.json').read_text())) for f in ['MLP','CFCascade','ScoreCascade']}
text='\n'.join(lines)+'\n'
text=text.replace('63个单阶段阈值',str(counts['MLP'])+'个单阶段阈值').replace('57个risk阈值',str(counts['CFCascade'])+'个risk阈值').replace('63个二次阈值',str(counts['ScoreCascade'])+'个二次阈值')
text=text.replace('预先声明全网格，再分别运行完整验证/测试曲线。','本轮在已观察粗网格后做统一加密：原分数0.02至0.2步长0.002并合并原63个点；CF风险对数网格从51加密至101个点并保留端点。加密网格在本轮运行前冻结，未修改模型。这是探索性网格精化，不是新的确认性测试。随后分别运行完整验证/测试曲线。')
(R/'REPORT.md').write_text(text)
# Standalone SVG showing full actual curves and exact matched points, no interpolation.
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="470"><rect width="1120" height="470" fill="white"/><g font-family="sans-serif" fill="#222"><text x="30" y="28" font-size="17">Keep-only CF cascade: actual equal-Recall comparisons</text>'];colors={'MLP':'#2563eb','CFCascade':'#dc2626','ScoreCascade':'#16a34a'}
for i,(title,rows) in enumerate([('Test grid: actual configurations',test),('Exact-matched configurations',results)]):
 x0=65+i*555;y0=70;w=455;h=310
 def xx(v):return x0+(v-98.95)/.85*w
 def yy(v):return y0+h-(v-45)/65*h
 svg.append(f'<defs><clipPath id="c{i}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath></defs><text x="{x0}" y="52" font-size="14">{title}</text>')
 for t in [99,99.2,99.4,99.6,99.8]:svg.append(f'<path d="M{xx(t)} {y0}V{y0+h}" stroke="#ddd"/><text x="{xx(t)}" y="{y0+h+20}" font-size="11" text-anchor="middle">{t:.1f}</text>')
 for t in [50,60,80,100]:svg.append(f'<path d="M{x0} {yy(t)}H{x0+w}" stroke="#ddd"/><text x="{x0-7}" y="{yy(t)+4}" font-size="11" text-anchor="end">{t}</text>')
 svg.append(f'<g clip-path="url(#c{i})">')
 for family in ['ScoreCascade','MLP','CFCascade']:
  for a in rows:
   if a['family']!=family:continue
   rad=4.8 if family=='ScoreCascade' else 2.8
   fill='none' if family=='ScoreCascade' else colors[family]
   svg.append(f'<circle cx="{xx(a["recall"])}" cy="{yy(a["reads"])}" r="{rad}" fill="{fill}" stroke="{colors[family]}" stroke-width="1.5"><title>{html.escape(a["key"])}: {a["recall"]:.3f}%, {a["reads"]:.3f} SSD/query</title></circle>')
 svg.append(f'</g><text x="{x0+w/2}" y="425" font-size="12" text-anchor="middle">Recall@10 (%)</text><text x="{x0}" y="448" font-size="12">SSD / query; blue: MLP, red: CF, green ring: score cascade</text>')
svg.append('</g></svg>');(R/'curves.svg').write_text(''.join(svg))
locks=json.loads((R/'INPUT_LOCK.json').read_text());assert all(sha(Path(p))==v for p,v in locks.items())
files=[p for p in R.iterdir() if p.suffix in ['.py','.h','.cpp','.sh']]+[BASE/'native/search',BASE/'native/pq_flash_index.cpp',BASE/'native/online_gate.h',BASE/'native/policy.h']
dump(R/'PROVENANCE.json',{str(p.relative_to(LG)):sha(p) for p in files});dump(R/'COMPLETE.json',dict(complete=True,exact_common_recall_points=len(cmp),nontrivial_points=len(nontrivial),cf_lower_ssd_points=len(wins),inputs_unchanged=True,report='REPORT.md'));print('REPORT COMPLETE')
