#!/usr/bin/env python3
"""Render measured heuristic/learned recall-read and recall-latency curves."""
import json,math
import numpy as np
from plot_online_curves import (image_surface,svg_surface,pdf_surface,create,rgb,paint,
    text,segment,arc,fill,width,move,line,stroke,save,restore,rectangle,clip,translate,
    rotate,png,finish,destroy,surface_destroy)
from run_online_m3 import E
OUT=E/'top3_heuristic';ONLINE=OUT/'online'
COLORS={'DiskANN':(.18,.25,.32),'Rank':(.86,.40,.10),'Rank+Progress':(.08,.56,.36),
        'Learned Top3':(.12,.43,.75),'Learned All17':(.49,.24,.67)}

def dot(c,x,y,color,r=4.5):
    rgb(c,*color);arc(c,x,y,r,0,2*math.pi);fill(c)

def render(path,rows):
    W,H=1440,1040
    surf=image_surface(0,W,H) if path.suffix=='.png' else svg_surface(str(path).encode(),W,H) if path.suffix=='.svg' else pdf_surface(str(path).encode(),W,H)
    c=create(surf);rgb(c,1,1,1);paint(c)
    text(c,40,37,'Heuristic vs learned expansion pruning: real live searches',25,True)
    text(c,40,66,'1,000 test queries | L=100 | K=10 | beam=4 | threads=4 | cache=10,000 | 3 interleaved trials, median',14)
    positions=[(50,102),(350,102),(720,102),(50,133),(430,133)]
    for (method,color),(x,y) in zip(COLORS.items(),positions):
        segment(c,x,y,x+30,y,color,2.5);dot(c,x+15,y,color)
        name='DiskANN, no gate' if method=='DiskANN' else 'rank > r' if method=='Rank' else 'rank > r AND completed > e' if method=='Rank+Progress' else method
        text(c,x+40,y+5,name,15)
    lo_all=max(0,math.floor(min(r['recall_pct'] for r in rows)*10)/10-.05)
    hi=math.ceil(max(r['recall_pct'] for r in rows)*10)/10+.05
    for row in range(2):
        lo=lo_all if row==0 else 99.5
        for panel,key in enumerate(('ssd_reads_per_query','latency_us')):
            left=90+panel*705;top=205+row*375;pw=560;ph=255;bottom=top+ph
            vals=[r[key]/1000 if key=='latency_us' else r[key] for r in rows]
            if key=='latency_us':vals +=[r[k]/1000 for r in rows for k in ('latency_min_us','latency_max_us')]
            pad=max((max(vals)-min(vals))*.1,.2 if key=='latency_us' else 2)
            xmin=min(vals)-pad;xmax=max(vals)+pad
            xp=lambda v:left+(v-xmin)/(xmax-xmin)*pw
            yp=lambda v:bottom-(v-lo)/(hi-lo)*ph
            title='Recall@10 vs SSD Reads/query' if panel==0 else 'Recall@10 vs Latency'
            text(c,left-30,top-27,('All points: ' if row==0 else 'High recall: ')+title,18,True)
            for v in np.linspace(lo,hi,5):
                segment(c,left,yp(v),left+pw,yp(v),(.88,.90,.92));text(c,left-57,yp(v)+5,f'{v:.2f}',13)
            for v in np.linspace(xmin,xmax,5):
                segment(c,xp(v),top,xp(v),bottom,(.93,.94,.95))
                text(c,xp(v)-16,bottom+25,f'{v:.2f}' if panel else f'{v:.1f}',13)
            segment(c,left,top,left,bottom,(.25,.3,.35));segment(c,left,bottom,left+pw,bottom,(.25,.3,.35))
            text(c,left+160,bottom+50,'Mean query latency (ms)' if panel else 'SSD read requests / query',14)
            save(c);translate(c,left-70,top+195);rotate(c,-math.pi/2);text(c,0,0,'Recall@10 (%)',14);restore(c)
            save(c);rectangle(c,left,top,pw,ph);clip(c)
            for method,color in COLORS.items():
                rr=sorted([r for r in rows if r['method']==method],key=lambda r:r['target_fnr_pct'])
                rgb(c,*color);width(c,2.3)
                for i,r in enumerate(rr):
                    x=xp(r[key]/1000 if panel else r[key]);y=yp(r['recall_pct'])
                    (move if i==0 else line)(c,x,y)
                stroke(c)
                for r in rr:
                    x=xp(r[key]/1000 if panel else r[key]);y=yp(r['recall_pct'])
                    if panel:segment(c,xp(r['latency_min_us']/1000),y,xp(r['latency_max_us']/1000),y,color)
                    dot(c,x,y,color,6 if method=='DiskANN' else 4.5)
            restore(c)
    text(c,40,953,'All gates: validation FNR targets 0.1%, 0.25%, 0.5%, 1%, 2%. Integer r/e grid maximizes validation SkipRate within budget.',13)
    text(c,40,980,'No test tuning or offline mask replay. Same beam-slot skip semantics. Model inference / integer decisions included in latency.',13)
    text(c,40,1007,'Warm-host timings; latency bars show trial min/max. Rank+Progress uses two of the Top3 features; visited_count is unused.',13)
    if path.suffix=='.png':assert png(surf,str(path).encode())==0
    finish(surf);destroy(c);surface_destroy(surf)

def main():
    assert json.loads((ONLINE/'COMPLETE.json').read_text())['complete']
    rows=json.loads((ONLINE/'summary.json').read_text())
    for ext in ('png','svg','pdf'):render(ONLINE/f'recall_cost_curves.{ext}',rows)
    offline=json.loads((OUT/'offline_summary.json').read_text())
    selected=json.loads((OUT/'selected_parameters.json').read_text())
    lines=['# Top3 heuristic baselines','',
        '两条请求的规则：rank>r；rank>r AND completed_expansions>e。只使用Top3中的rank和completed，不使用visited_count。',
        'rank为zero-based队列位置，在标记当前候选expanded之前取得。completed为beam开始前已完成精确评分的展开数；beam内未完成的读取不计入。',
        'r=-1..99、e=-1..validation最大completed，穷举整数网格。在每档validation retain FNR预算下最大化样本加权SkipRate；并列依次选择较少误跳过、较大r、较大e。e=-1等价于只用rank。',
        '所有参数先保存锁定，然后才评估test。未根据test选择参数；learned Top3/All17使用原有validation校准tau。目标FNR不是在线Recall损失保证。',
        '', '| Rule | val FNR target | r | e | Validation SkipRate | Validation actual FNR | Test SkipRate | Test actual FNR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in selected:
        m=next(r for r in offline if r['key']==c['key'] and r['split']=='test')
        e='—' if c['method']=='Rank' else str(c['e'])
        lines.append(f'| {c["method"]} | {c["target_fnr_pct"]:g}% | {c["r"]} | {e} | {c["validation_skip_rate"]*100:.3f}% | {c["validation_retain_fnr"]*100:.4f}% | {m["skip_rate"]*100:.3f}% | {m["retain_fnr"]*100:.4f}% |')
    lines += ['','## Live curves','','![Recall / cost](online/recall_cost_curves.png)','',
        '两条heuristic和Learned Top3、Learned All17及无模型DiskANN重新交错测量。固定L=100、K=10、beam=4；每个配置1000 test query、3 repeats，中位数，含实际决策开销。',
        '[Full online measurements](online/REPORT.md) · [Online CSV](online/summary.csv) · [Offline CSV](offline_summary.csv)',
        '', '## Interpretation', '',
        '验证FNR目标1%时，Rank规则r=89，Rank+Progress规则r=72/e=100；两者test Recall均为99.52%，Reads为92.497/91.744。Learned Top3为Recall 99.58%、Reads 83.825，Learned All17为99.59%、83.209；这两条简单规则在这一档位均被learned的召回/读取点支配。',
        '相同Recall=99.64%的实测点中，Rank最少Reads=100.411，Rank+Progress最少Reads=101.389，Learned Top3最少Reads=93.286，Learned All17最少Reads=89.327。该比较使用各曲线已有实测点，不插值或重新根据test选择部署参数。',
        'Rank+Progress的各r/e组合分别在validation优化，规则集合不一定嵌套；在线Recall不必随FNR目标单调。例如0.25%档在test上被0.5%档支配，图中保留原实测点。参数优化目标是离线validation标签SkipRate，而不是在线validation Recall/I/O；结论仅针对这两条规则及这一调参流程。',
        '', '## Reproduce','', '在learned-gate目录依次运行：','',
        '```bash','cmake --build DiskANN/build --target search_disk_index -j 4',
        '.venv/bin/python experiments/sift1m/calibrate_heuristics.py',
        '.venv/bin/python experiments/sift1m/run_online_heuristics.py',
        '.venv/bin/python experiments/sift1m/plot_heuristic_curves.py','```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(complete=True,validation_only_calibration=True,
        live_heuristics=True,measured_runs=63,plots=True),indent=2)+'\n')
    print('Heuristic curves exported: PNG / SVG / PDF',flush=True)

if __name__=='__main__':main()
