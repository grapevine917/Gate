"""Report locked mild-policy tests and separate empirical Recall matching."""
from common import *
assert (D/'MEASUREMENTS_COMPLETE.json').exists()
main=json.loads((D/'main_summary.json').read_text()); aligned=json.loads((D/'aligned_summary.json').read_text())
M={r['key']:r for r in main};B={r['key']:r for r in aligned}
lock=json.loads((D/'MAIN_SELECTION_LOCK.json').read_text())
vm=json.loads((D/'ALIGNMENT_VALIDATION_LOCK.json').read_text());tm=json.loads((D/'ALIGNMENT_TEST_MATCH.json').read_text())
names={0:'DiskANN',1:'Heuristic',2:'Static MLP',3:'Static LR',4:'Mild MLP',5:'Mild LR'}
def table(rows):
    fields=[('Group',lambda r:names[r['kind']]),('L',lambda r:str(r['inference_L'])),('Recall %',lambda r:f"{r['recall']:.2f}"),('P50 ms',lambda r:f"{r['p50_ms']:.4f}"),('P95 ms',lambda r:f"{r['p95_ms']:.4f}"),('P99 ms',lambda r:f"{r['p99_ms']:.4f}"),('Mean ms',lambda r:f"{r['mean_ms']:.4f}"),('QPS',lambda r:f"{r['qps']:.1f}"),('KiB/query',lambda r:f"{r['physical_bytes']/1024:.2f}"),('Reads/query',lambda r:f"{r['reads']:.3f}"),('Exp/query',lambda r:f"{r['expansions']:.3f}"),('Skips/query',lambda r:f"{r['skips']:.3f}"),('Model calls/query',lambda r:f"{r['model_calls']:.3f}"),('IO ms',lambda r:f"{r['io_ms']:.4f}"),('IO rounds/query',lambda r:f"{r['io_rounds']:.3f}"),('Eligible/query',lambda r:f"{r['eligible']:.3f}"),('Extra skips/query',lambda r:f"{r['extra_skips']:.3f}")]
    return '| '+' | '.join(x[0] for x in fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+'\n'.join('| '+' | '.join(f(r) for _,f in fields)+' |' for r in rows)+'\n'
main=[M[c['key']] for c in lock['configs']];save(D/'main_results.csv',main)
save(D/'validation_grid.csv',json.loads((D/'validation.json').read_text()))
textout=['# SIFT1M：L=100 训练模型的温和动态策略验证','',
'模型复用已保存的纯 L=100 MLP/LR，没有重新训练。SIFT1M，K=10，验证/测试各 1000 查询；主实验推理 L=100、4 搜索线程、beam=4、缓存 10000 节点。168 个唯一参数配置在测试前冻结。',
'', '## 策略与主实验','',
'早期阈值严格等于对应静态模型阈值；仅当候选 rank > r 且已完成扩展数 > e 时，阈值提高至 multiplier × 静态阈值。模型分数低于当前阈值时剪枝；缓存命中节点受保护。触发条件及 multiplier 是人工设计并通过验证集选取，模型权重不变。',
'', '主实验先要求验证 Recall 不低于静态组，再在实际有额外剪枝的候选中选择验证 P95 最低者；若无合格候选，报告质量条件未满足。以下选择在测试前锁定：','']
for d in lock['details']:
    r=M[d['key']]
    textout.append(f"- {d['family']}：early τ={r['tau']:.10g}，late τ={r['late_tau']:.10g}（×{r['multiplier']:g}），rank>{r['rank']} 且 completed>{r['exp']}。验证 Recall：静态 {d['static_validation_recall']:.2f}%，温和 {d['mild_validation_recall']:.2f}%；满足质量条件：{d['quality_matched']}。")
textout+=['','六组在本轮交错运行，12 次循环轮换顺序，共享同时期对照；延迟和 QPS 为各轮指标的中位数，IO/扩展等计数及输出 ID 在重复间完全一致。', '',table(main)]
effects=[]
activation=[]
for family in ['MLP','LR']:
    s=M[f'{family}_static_b1'];d=next(r for r in main if r['family']==family and r['kind'] in [4,5])
    effect=dict(family=family,recall_delta_pp=d['recall']-s['recall'],p95_change_percent=(d['p95_ms']/s['p95_ms']-1)*100,qps_change_percent=(d['qps']/s['qps']-1)*100,bytes_change_percent=(d['physical_bytes']/s['physical_bytes']-1)*100,expansion_change_percent=(d['expansions']/s['expansions']-1)*100,extra_skips_per_query=d['extra_skips'])
    effects.append(effect)
    textout.append(f"{family} 温和相对静态：Recall {effect['recall_delta_pp']:+.2f} 个百分点；P95 {effect['p95_change_percent']:+.2f}%；QPS {effect['qps_change_percent']:+.2f}%；物理读取字节 {effect['bytes_change_percent']:+.2f}%；额外剪枝 {d['extra_skips']:.3f} 次/查询。")
    a=np.genfromtxt(prefix('test',d,0).with_suffix('.csv'),delimiter=',',names=True)
    z=dict(family=family,eligible_queries=int(np.count_nonzero(a['progress_eligible'])),extra_skip_queries=int(np.count_nonzero(a['progress_extra_skips'])),queries=1000,p95_min_ms=d['p95_min_ms'],p95_max_ms=d['p95_max_ms'])
    activation.append(z)
    textout.append(f"晚期规则覆盖 {z['eligible_queries']}/1000 个查询，实际增加剪枝的查询 {z['extra_skip_queries']}/1000；动态组 12 轮 P95 范围 {d['p95_min_ms']:.4f}–{d['p95_max_ms']:.4f} ms。")
save(D/'main_effects.csv',effects)
save(D/'main_activation.csv',activation)
# Compare the same difficult queries, rather than different methods' own tails.
tail=[]
for family in ['MLP','LR']:
    s=M[f'{family}_static_b1'];d=next(r for r in main if r['family']==family and r['kind'] in [4,5])
    arrays={}
    for r in [s,d]:
        arrays[r['key']]=[np.genfromtxt(prefix('test',r,rep).with_suffix('.csv'),delimiter=',',names=True) for rep in range(12)]
    med={k:np.median([a['latency_us'] for a in aa],axis=0) for k,aa in arrays.items()};ii=np.argsort(med[s['key']])[-50:]
    sa=arrays[s['key']][0];da=arrays[d['key']][0]
    for i in ii:
        tail.append(dict(family=family,query=int(i),static_median_ms=float(med[s['key']][i]/1000),mild_median_ms=float(med[d['key']][i]/1000),static_reads=int(sa['reads'][i]),mild_reads=int(da['reads'][i]),static_hits=int(sa['hits'][i]),mild_hits=int(da['hits'][i]),eligible=int(da['progress_eligible'][i]),extra_skips=int(da['progress_extra_skips'][i])))
    tt=[r for r in tail if r['family']==family];a=np.mean([r['static_median_ms'] for r in tt]);b=np.mean([r['mild_median_ms'] for r in tt])
    textout.append(f"按静态 {family} 的逐查询 12 轮中位延迟选出最慢 50 个查询，固定同一查询集合：平均逐查询中位延迟 {a:.4f} → {b:.4f} ms（{(b/a-1)*100:+.2f}%），命中数总计 {sum(r['static_hits'] for r in tt)} → {sum(r['mild_hits'] for r in tt)}。这是配对尾部诊断，区别于整组 P95。")
save(D/'main_paired_tail.csv',tail)
def alignment_section(match,title,csvname,description):
    textout.extend(['',title,'',description,''])
    if not match.get('available'):
        textout.append('没有可用的六组匹配：'+match['reason']);return
    rows=[B[k] for k in match['keys']];save(D/csvname,rows)
    lo=min(r['total_hits'] for r in rows);hi=max(r['total_hits'] for r in rows)
    textout.append(f"锁定/匹配依据的命中数区间 {match['hits_min']}–{match['hits_max']}，模式 {match['mode']}；最终测试实际命中数 {lo}–{hi}/10000，Recall 差距 {(hi-lo)/100:.2f} 个百分点。" + (' 测试 Recall 严格相等。' if lo==hi else ' 此表测试 Recall 未严格对齐，必须连同残差阅读。'))
    textout.extend(['',table(rows),'配置（静态/动态同模型的基础阈值严格相同）：',''])
    for r in rows:textout.append(f"- {names[r['kind']]}：{r['key']}；L={r['inference_L']}，τ={r['tau']:.10g}，late τ={r['late_tau']:.10g}，rank>{r['rank']}，completed>{r['exp']}。")
    for family in ['MLP','LR']:
        s=next(r for r in rows if r['family']==family and r['kind'] in [2,3]);d=next(r for r in rows if r['family']==family and r['kind'] in [4,5])
        textout.append(f"{family} 温和对静态：Recall {d['recall']-s['recall']:+.2f} pp，P95 {(d['p95_ms']/s['p95_ms']-1)*100:+.2f}%，QPS {(d['qps']/s['qps']-1)*100:+.2f}%，物理字节 {(d['physical_bytes']/s['physical_bytes']-1)*100:+.2f}%。")
alignment_section(vm,'## Recall 对照一：验证集锁定','alignment_validation_locked.csv','在验证集选择最接近 99.60% 的共同整数命中数，随后锁定配置再测试；最终测试的质量残差如实列出。使用与主实验相同六类方法，原生 DiskANN 允许调 L，其余推理 L=100。表中计时是锁定/匹配结束后独立的 6 次交错复测中位数。')
alignment_section(tm,'## Recall 对照二：实际测试质量匹配（描述性）','alignment_test_matched.csv','在测试前冻结的有限网格中匹配实际测试整数命中数，不插值；每个同模型静态/动态组合的 early τ 相同，动态必须产生额外剪枝。仅使用验证 P95 决定同质量候选的取舍，不使用测试速度挑选参数；之后独立进行 6 次交错计时。由于匹配使用了测试命中数，这属于描述性同质量比较，不能充当未见测试集的部署结果。')
textout+=['','## 校验与限制','',
'原始 ID 重算 Recall；每次 reader_requests×4096、reader_bytes 和 Linux read_bytes 严格相等；无交换；输出 ID、IO 和扩展计数在重复间一致；阈值不变的动态 identity 配置与静态输出完全相同。源模型、输入数据和原生程序的 SHA256 保持不变。物理 IO 不一致的尝试归档于 rejected_io，并重试，不混入统计。',
f"本轮 IO 计数不一致归档次数：{len(json.loads((D/'IO_REJECTIONS.json').read_text())) if (D/'IO_REJECTIONS.json').exists() else 0}。",
'', '同 Recall 仅指总命中率相同，不保证成功的是同一批查询。只有一个已训练模型种子，有限且不均匀的参数网格；验证集各配置只扫描一轮，选参可能受计时波动影响；最终重复计时不能替代重训练不确定性，也不构成收益显著性的检验。该查询拆分此前已被检查，本实验属于探索性验证。全量指标保存在 CSV，原始逐查询数据与协议保留在本目录。',
'', '## 复现','', '项目根目录依次运行：', '', '```bash', '.venv/bin/python experiments/sift1m_mild_dynamic_recall_aligned/prepare_validate.py', '.venv/bin/python experiments/sift1m_mild_dynamic_recall_aligned/evaluate_test.py', '.venv/bin/python experiments/sift1m_mild_dynamic_recall_aligned/report.py', '```', '', '脚本保留已通过严格检查的原始测量，重复执行会复用本目录结果；需要全新计时应使用另一个实验目录，并重新冻结协议。', '']
(D/'REPORT.md').write_text('\n'.join(textout))
dump(D/'COMPLETE.json',dict(complete=True,report=str(D/'REPORT.md'),verification=json.loads((D/'verification.json').read_text()),main_effects=effects,validation_alignment=vm,test_alignment=tm))
dump(D/'STATUS.json',dict(phase='complete',complete=True,report=str(D/'REPORT.md')))
print('\n'.join(textout))
