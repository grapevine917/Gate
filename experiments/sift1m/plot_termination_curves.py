#!/usr/bin/env python3
"""Report and plot budget ET, learned ET, and learned expansion pruning."""
import csv,json,math
import numpy as np
from plot_online_curves import (image_surface,svg_surface,pdf_surface,create,rgb,paint,
    text,segment,arc,fill,width,move,line,stroke,save,restore,rectangle,clip,translate,
    rotate,png,finish,destroy,surface_destroy)
from run_online_m3 import E
ROOT=E/'learned_termination';OUT=ROOT/'online'
COLORS={'DiskANN':(.18,.25,.32),'Budget Early Termination':(.86,.40,.10),
    'Learned ET Top3':(.12,.43,.75),'Learned ET All17':(.49,.24,.67),
    'Learned Top3 Expansion Pruning':(.08,.56,.36)}

def dot(c,x,y,color,r=4.5):
    rgb(c,*color);arc(c,x,y,r,0,2*math.pi);fill(c)

def render(path,rows,title='Learned early termination vs budget ET and expansion pruning',notes=None):
    W,H=1840,1030
    surf=image_surface(0,W,H) if path.suffix=='.png' else svg_surface(str(path).encode(),W,H) if path.suffix=='.svg' else pdf_surface(str(path).encode(),W,H)
    c=create(surf);rgb(c,1,1,1);paint(c)
    text(c,40,37,title,26,True)
    text(c,40,67,'SIFT1M | Same 1,000 test queries | L=100 | K=10 | beam=4 | threads=4 | cache=10,000 | 3 joint trials, median',15)
    positions=[(50,104),(390,104),(850,104),(50,136),(500,136)]
    for (method,col),(x,y) in zip(COLORS.items(),positions):
        segment(c,x,y,x+30,y,col,2.5);dot(c,x+15,y,col)
        text(c,x+42,y+5,'DiskANN, no model' if method=='DiskANN' else method,15)
    keys=[('ssd_reads_per_query','SSD read requests / query'),('expansions_per_query','Expansions / query'),
          ('latency_us','Mean query latency (ms)')]
    minrec=min(r['recall_pct'] for r in rows)
    lo_all=max(0,math.floor((minrec-1)/5)*5)
    hi_detail=math.ceil(max(r['recall_pct'] for r in rows)*10)/10+.05
    for row in range(2):
        lo=lo_all if row==0 else 99.0;hi=100.0 if row==0 else hi_detail
        for panel,(key,title) in enumerate(keys):
            left=85+panel*611;top=215+row*365;pw=475;ph=255;bottom=top+ph
            candidates=rows if row==0 else [r for r in rows if r['recall_pct']>=lo]
            vals=[r[key]/1000 if key=='latency_us' else r[key] for r in candidates]
            if key=='latency_us':vals+=[r[k]/1000 for r in candidates for k in ('latency_min_us','latency_max_us')]
            if row==0:
                xmin=0;xmax=math.ceil(max(vals)/(1 if key=='latency_us' else 20))*(1 if key=='latency_us' else 20)
            else:
                pad=max((max(vals)-min(vals))*.1,.15 if key=='latency_us' else 2)
                xmin=min(vals)-pad;xmax=max(vals)+pad
            xp=lambda v:left+(v-xmin)/(xmax-xmin)*pw
            yp=lambda v:bottom-(v-lo)/(hi-lo)*ph
            text(c,left-25,top-30,('All points: ' if row==0 else 'Recall >=99%: ')+title,17,True)
            for v in np.linspace(lo,hi,5):
                segment(c,left,yp(v),left+pw,yp(v),(.88,.90,.92));text(c,left-54,yp(v)+5,f'{v:.2f}' if row else f'{v:g}',13)
            for v in np.linspace(xmin,xmax,5):
                segment(c,xp(v),top,xp(v),bottom,(.93,.94,.95));text(c,xp(v)-15,bottom+25,f'{v:.2f}' if key=='latency_us' else f'{v:.1f}',12)
            segment(c,left,top,left,bottom,(.25,.3,.35));segment(c,left,bottom,left+pw,bottom,(.25,.3,.35))
            text(c,left+95,bottom+49,title,14)
            save(c);translate(c,left-67,top+198);rotate(c,-math.pi/2);text(c,0,0,'Recall@10 (%)',14);restore(c)
            save(c);rectangle(c,left,top,pw,ph);clip(c)
            for method,col in COLORS.items():
                rr=sorted([r for r in rows if r['method']==method],key=lambda r:r['parameter'],
                    reverse=method=='Budget Early Termination')
                rgb(c,*col);width(c,2.4)
                for i,r in enumerate(rr):
                    x=xp(r[key]/1000 if key=='latency_us' else r[key]);y=yp(r['recall_pct'])
                    (move if i==0 else line)(c,x,y)
                stroke(c)
                for r in rr:
                    x=xp(r[key]/1000 if key=='latency_us' else r[key]);y=yp(r['recall_pct'])
                    if key=='latency_us':segment(c,xp(r['latency_min_us']/1000),y,xp(r['latency_max_us']/1000),y,col)
                    dot(c,x,y,col,6 if method=='DiskANN' else 4.5)
            restore(c)
    if notes is None:notes=[
        'Learned ET: P(safe-to-stop), break before next beam; thresholds calibrated by validation unsafe-stop query rates (0% to 10%).',
        'Expansion pruning: validation retain-candidate FNR targets 0.1% to 2%. These risk metrics differ; compare actual recall / cost.',
        'Warm-host timings; feature construction and native inference included. SSD reads are index requests. Latency bars: trial min/max.']
    for y,note in zip((947,975,1003),notes):text(c,40,y,note,14)
    if path.suffix=='.png':assert png(surf,str(path).encode())==0
    finish(surf);destroy(c);surface_destroy(surf)

def main():
    assert json.loads((OUT/'COMPLETE.json').read_text())['complete']
    rows=json.loads((OUT/'summary.json').read_text());base=rows[0]
    training=json.loads((ROOT/'training_summary.json').read_text())
    for ext in ('png','svg','pdf'):render(OUT/f'recall_cost_curves.{ext}',rows)
    # Observed non-dominated measurements for readers comparing at matched actual Recall.
    frontier=[]
    for method in COLORS:
        rr=[r for r in rows if r['method']==method]
        for cost in ('ssd_reads_per_query','expansions_per_query','latency_us'):
            for r in rr:
                dominated=any(v['recall_pct']>=r['recall_pct'] and v[cost]<=r[cost] and
                    (v['recall_pct']>r['recall_pct'] or v[cost]<r[cost]) for v in rr)
                if not dominated:frontier.append(dict(method=method,cost=cost,key=r['key'],recall_pct=r['recall_pct'],value=r[cost]))
    (OUT/'observed_frontiers.json').write_text(json.dumps(frontier,indent=2)+'\n')
    reductions=[]
    for r in rows:
        reductions.append(dict(key=r['key'],method=r['method'],recall_loss_pp=base['recall_pct']-r['recall_pct'],
            reads_reduction_pct=100*(1-r['ssd_reads_per_query']/base['ssd_reads_per_query']),
            expansions_reduction_pct=100*(1-r['expansions_per_query']/base['expansions_per_query']),
            latency_reduction_pct=100*(1-r['latency_us']/base['latency_us'])))
    (OUT/'reductions_vs_baseline.json').write_text(json.dumps(reductions,indent=2)+'\n')
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    lines=['# Learned early termination baseline','',
        'SIFT1M，固定L=100、K=10、beam=4、threads=4、cache=10000；与前序相同1000条held-out test query。训练/验证query仍为8000/1000，互不重叠。',
        'Top3: candidate_queue_rank、completed_expansions、visited_count。All17: 原始17维pre_io_state。各模型d→128→64→1，ReLU/ReLU/sigmoid、BCE Adam、batch=2048、lr=.001、alpha=.0001、seed=42；train-only标准化，validation logloss早停（最多60轮、patience=8、min_delta=.0001）。',
        '新数据在完整baseline真实轨迹的beam边界采集：至少K个节点已精确评分且队列仍有未展开候选。safe=1 iff 全部baseline最终top10已精确评分；此时返回当前top10可逐项保留完整baseline结果。用既有展开metadata独立核验标签，全部state与原pre_io数据一致。',
        'y是safe-to-stop，不复用候选级pruning标签；最终节点ID/GT/未来状态不进入模型。ET在新beam选择和SSD请求提交前判定，p_safe>=tau就break整个搜索；未命中阈值则完整处理下一beam。至少K exact-score guard避免不完整topK。',
        '', '| Model | Validation safe-stop AUC | Test safe-stop AUC | best epoch |',
        '|---|---:|---:|---:|']
    for m in training:lines.append(f'| ET {m["model"]} | {m["validation_auc"]:.6f} | {m["test_auc"]:.6f} | {m["best_epoch"]} |')
    lines+=['', 'Data boundaries: '+json.dumps(manifest['splits']), '',
        '## Calibration','',
        'ET阈值使用C++原生validation概率。每条query的unsafe前缀最大p_safe决定是否会误停；safe标签沿query单调。选择最小float32阈值使validation误停query数不超过目标预算；严格排除首个超预算分数及ties。风险目标为0%、0.1%、0.25%、0.5%、1%、2%、5%、10%，1000 query下0.25%预算向下取整为2条。',
        'Top3 expansion pruning沿用validation候选级retain FNR目标0.1%、0.25%、0.5%、1%、2%。两类风险分母和标签不同，不能将相同百分比解释为相同风险；使用真实Recall–成本曲线比较。验证误停为零不保证测试零误停。',
        'BudgetEarlyTermination使用现有io_limit，预算20、40、60、70、80、85、90、95、100、105、110、120、140个索引读取，在beam边界检查，最多超出3个读取。无模型DiskANN不带任何gate。',
        '', '## Online measurements','',
        f'全部{len(rows)}配置交错随机重复3次，共{len(rows)*3}次测量；所有模型重新计时，未拼接旧latency。报告median mean-query latency/QPS，特征构造、实际native模型推理或预算判断计入；模型加载不计入，主机OS缓存为预热状态。',
        'ET Top3/All17统一构造边界17维state再选择输入；不加载q/v feature表。Expansion pruning统一重建candidate state，每候选实时MLP推理。',
        '', '| Method | Parameter | tau | Recall@10 | Reads/query | Expansions/query | Latency ms | QPS | Test stopped queries | Test unsafe queries |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        parameter='—' if r['method']=='DiskANN' else str(r['parameter'])+' reads' if r['budget'] is not None else f'{r["parameter"]:g}% '+('unsafe query target' if r['method'].startswith('Learned ET') else 'retain FNR target')
        tau='—' if r['threshold'] is None else f'{r["threshold"]:.9g}'
        stopped='—' if r['test_stopped_queries'] is None else str(r['test_stopped_queries'])
        unsafe='—' if r['test_unsafe_stopped_queries'] is None else str(r['test_unsafe_stopped_queries'])
        lines.append(f'| {r["method"]} | {parameter} | {tau} | {r["recall_pct"]:.3f}% | {r["ssd_reads_per_query"]:.3f} | {r["expansions_per_query"]:.3f} | {r["latency_us"]/1000:.3f} | {r["qps"]:.1f} | {stopped} | {unsafe} |')
    lines+=['','## Curves','','![Recall / costs](online/recall_cost_curves.png)','',
        '曲线仅连接实测点，不插值调参或平滑拟合；下排放大Recall>=99%。Latency横线显示重复测量min/max。observed_frontiers.json记录各方法每个cost维度的实测非支配点，仅用于描述，不用于test部署参数选择。',
        '', '## Interpretation', '',
        '本次safe-stop任务All17的Test AUC为0.956429，高于Top3的0.950089；前序expansion pruning的Top3接近All17结论不能直接移植到终止任务。新增state在本次ET低风险区域仍有收益。',
        '代表实测点：All17 ET（validation unsafe-query目标0.25%）Recall=99.62%、Reads=83.463、Expansions=92.869；Budget100为99.57%、99.896、109.331；Top3 pruning（validation retain FNR目标1%）为99.58%、83.825、93.229。All17 ET这一点在召回和三个cost维度均优于这两个实测配置。',
        'Top3 ET并非在每个区域都优于Top3 pruning：两者Recall=99.45%的实测点中，ET（unsafe-query目标0.5%）Reads=78.552，pruning（retain FNR目标2%）Reads=75.691。结论应按真实Recall–cost而不是统一风险百分比比较。',
        'All17 ET 0.25%目标在test实际误停5/1000=0.5%；验证零误停配置的两个ET在test也各有1条unsafe query。safe-stop是分类预测，不是新的安全保证；验证风险目标和测试实际误停均已报告。',
        '', '## Verification','',
        '关闭ET（tau=1.1）时，全部validation/test的有序ID和距离逐项复现baseline；state、label和Python/C++概率核验通过。所有在线ET的第一次停止位置、展开数、推理次数和停止计数逐query与baseline原生概率预测一致。每个safe停止均逐项保留baseline有序ID和距离；unsafe停止恰好对应baseline结果变化。',
        '全部重复测量结果ID一致，Recall独立重算通过。Budget I/O overshoot<=3核验；baseline和Top3 pruning保持前序搜索行为。实际Recall与SSD/Expansion由真实搜索输出，未复用离线mask。',
        'SSD Reads是索引read请求，不是硬件物理读取；当前时延是预热主机性能。查询此前用于其他探索性分析，因此最终外推应使用新查询/数据集。',
        '', '## Reproduce','', '在learned-gate目录运行：','',
        '```bash','cmake --build DiskANN/build --target search_disk_index -j 4',
        '.venv/bin/python experiments/sift1m/collect_termination_data.py',
        '.venv/bin/python experiments/sift1m/train_termination.py',
        '.venv/bin/python experiments/sift1m/calibrate_termination.py',
        '.venv/bin/python experiments/sift1m/run_online_termination.py',
        '.venv/bin/python experiments/sift1m/plot_termination_curves.py','```']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(complete=True,learned_et_models=2,
        standalone_whole_search_stop=True,configs=len(rows),measured_runs=len(rows)*3,plots=True),indent=2)+'\n')
    print('Learned ET report and PNG / SVG / PDF curves exported.',flush=True)

if __name__=='__main__':main()
