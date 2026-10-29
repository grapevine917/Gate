#!/usr/bin/env python3
"""Export measured recall/cost curves through native Cairo (PNG/SVG/PDF)."""
import ctypes as ct
import csv,json,math,argparse
from pathlib import Path
import numpy as np
from run_online_m3 import mat
E=Path(__file__).resolve().parent;OUT=E/'online_curves'
C=ct.CDLL('libcairo.so.2');P=ct.c_void_p;D=ct.c_double;I=ct.c_int
def bind(name,ret,args):
    fn=getattr(C,'cairo_'+name);fn.restype=ret;fn.argtypes=args;return fn
image_surface=bind('image_surface_create',P,[I,I,I])
svg_surface=bind('svg_surface_create',P,[ct.c_char_p,D,D])
pdf_surface=bind('pdf_surface_create',P,[ct.c_char_p,D,D])
create=bind('create',P,[P]);destroy=bind('destroy',None,[P]);surface_destroy=bind('surface_destroy',None,[P])
finish=bind('surface_finish',None,[P]);png=bind('surface_write_to_png',I,[P,ct.c_char_p])
rgb=bind('set_source_rgb',None,[P,D,D,D]);rgba=bind('set_source_rgba',None,[P,D,D,D,D])
paint=bind('paint',None,[P]);move=bind('move_to',None,[P,D,D]);line=bind('line_to',None,[P,D,D])
stroke=bind('stroke',None,[P]);fill=bind('fill',None,[P]);width=bind('set_line_width',None,[P,D])
rectangle=bind('rectangle',None,[P,D,D,D,D]);arc=bind('arc',None,[P,D,D,D,D,D])
font=bind('select_font_face',None,[P,ct.c_char_p,I,I]);size=bind('set_font_size',None,[P,D])
show=bind('show_text',None,[P,ct.c_char_p]);save=bind('save',None,[P]);restore=bind('restore',None,[P])
clip=bind('clip',None,[P]);translate=bind('translate',None,[P,D,D]);rotate=bind('rotate',None,[P,D])
dash=bind('set_dash',None,[P,ct.POINTER(D),I,D])
COLORS={'DiskANN varying L':(0.12,0.36,0.71),'Early Termination':(0.88,0.40,0.09),'Learned Expansion Pruning':(0.08,0.55,0.36)}

def text(c,x,y,value,fs=14,bold=False,color=(0.14,0.18,0.23)):
    rgb(c,*color);font(c,b'DejaVu Sans',0,int(bold));size(c,fs);move(c,x,y);show(c,str(value).encode())
def segment(c,x1,y1,x2,y2,color,lw=1):
    rgb(c,*color);width(c,lw);move(c,x1,y1);line(c,x2,y2);stroke(c)
def marker(c,x,y,method,r=4.8):
    color=COLORS[method];rgb(c,*color)
    if method=='DiskANN varying L':arc(c,x,y,r,0,2*math.pi)
    elif method=='Early Termination':move(c,x,y-r-1);line(c,x+r+1,y+r);line(c,x-r-1,y+r);line(c,x,y-r-1)
    elif method=='State-only Expansion Pruning':move(c,x,y-r-1);line(c,x+r+1,y);line(c,x,y+r+1);line(c,x-r-1,y);line(c,x,y-r-1)
    else:rectangle(c,x-r,y-r,2*r,2*r)
    fill(c)

def render(path,data):
    W,H=1440,1010
    suffix=path.suffix
    surf=image_surface(0,W,H) if suffix=='.png' else svg_surface(str(path).encode(),W,H) if suffix=='.svg' else pdf_surface(str(path).encode(),W,H)
    c=create(surf);rgb(c,1,1,1);paint(c)
    text(c,50,42,'Recall / cost tradeoffs on held-out SIFT queries',25,True)
    text(c,50,70,'1,000 test queries | K=10 | beam=4 | threads=4 | cache=10,000 | 3 interleaved trials, median',15)
    legend=[('DiskANN varying L','DiskANN varying L'),('Early Termination','Early Termination (I/O budget, L=100)'),
            ('Learned Expansion Pruning','Learned Expansion Pruning (M3, L=100)')]
    if 'State-only Expansion Pruning' in COLORS:
        legend.append(('State-only Expansion Pruning','State-only Expansion Pruning (L=100)'))
        positions=[(60,101),(680,101),(60,131),(680,131)]
    else:positions=[(60,103),(340,103),(825,103)]
    for (method,label),(x,y) in zip(legend,positions):
        segment(c,x,y,x+30,y,COLORS[method],2.5);marker(c,x+15,y,method)
        text(c,x+40,y+5,label,14)
    minrec=min(r['recall_pct'] for r in data)
    ymin=max(0,math.floor((minrec-1)/5)*5)
    maxreads=math.ceil(max(r['ssd_reads_per_query'] for r in data)/20)*20
    maxlat=math.ceil(max(r['latency_max_us'] for r in data)/1000)
    panels=[(70,165,'ssd_reads_per_query',maxreads,ymin,100,'Recall@10 vs SSD Reads (all points)'),
            (785,165,'latency_us',maxlat,ymin,100,'Recall@10 vs Latency (all points)'),
            (70,570,'ssd_reads_per_query',maxreads,99,99.9,'High-recall detail: SSD Reads'),
            (785,570,'latency_us',maxlat,99,99.9,'High-recall detail: Latency')]
    for x0,y0,key,xmax,lo,hi,title in panels:
        pw,ph=570,295;left=x0+30;top=y0+25;bottom=top+ph
        text(c,x0,y0-4,title,18,True)
        xmin=(60 if key!='latency_us' else 4) if lo==99 else 0
        xp=lambda v:left+(v-xmin)/(xmax-xmin)*pw
        yp=lambda v:bottom-(v-lo)/(hi-lo)*ph
        yticks=np.linspace(lo,hi,6 if lo<99 else 4)
        for val in yticks:
            y=yp(val);segment(c,left,y,left+pw,y,(0.88,0.9,0.92))
            text(c,left-58,y+5,f'{val:.1f}',13)
        xticks=np.linspace(xmin,xmax,6) if key!='latency_us' or lo<99 else np.arange(xmin,xmax+1)
        for val in xticks:
            x=xp(val);segment(c,x,top,x,bottom,(0.93,0.94,0.95))
            text(c,x-13,bottom+24,f'{val:g}',13)
        segment(c,left,top,left,bottom,(0.25,0.3,0.35),1.2)
        segment(c,left,bottom,left+pw,bottom,(0.25,0.3,0.35),1.2)
        text(c,left+170,bottom+48,'SSD read requests / query' if key!='latency_us' else 'Mean query latency (ms)',14)
        save(c);translate(c,left-65,top+205);rotate(c,-math.pi/2);text(c,0,0,'Recall@10 (%)',14);restore(c)
        save(c);rectangle(c,left,top,pw,ph);clip(c)
        for method,color in COLORS.items():
            rows=[r for r in data if r['method']==method]
            rows.sort(key=lambda r:r['parameter'],reverse='Pruning' in method)
            points=[(xp(r[key]/1000 if key=='latency_us' else r[key]),yp(r['recall_pct'])) for r in rows]
            rgb(c,*color);width(c,2.4)
            pattern=(D*2)(7,4) if method=='Early Termination' else (D*2)(0,0)
            dash(c,pattern,2 if method=='Early Termination' else 0,0)
            for i,(x,y) in enumerate(points):
                (move if i==0 else line)(c,x,y)
            stroke(c);dash(c,pattern,0,0)
            if key=='latency_us':
                for r,(_,y) in zip(rows,points):
                    segment(c,xp(r['latency_min_us']/1000),y,xp(r['latency_max_us']/1000),y,color,1)
            for x,y in points:marker(c,x,y,method)
        restore(c)
        # Sparse labels retain readability at dense high-recall points.
        if lo==99:
            labels=[r for r in data if (r['method']=='DiskANN varying L' and r['parameter'] in (70,100,150))
                    or ('Pruning' in r['method'] and r['parameter'] in (0.1,1,2))
                    or (r['method']=='Early Termination' and r['parameter'] in (95,105))]
            for r in labels:
                if not lo<=r['recall_pct']<=hi:continue
                x=xp(r[key]/1000 if key=='latency_us' else r[key]);y=yp(r['recall_pct'])
                method=r['method'];label=f'L={r["parameter"]}' if method=='DiskANN varying L' else f'B={r["parameter"]}' if method=='Early Termination' else f'FNR={r["parameter"]:g}%'
                dx,dy=7,-12
                if 'Pruning' in method:
                    dx,dy=(-85,-14) if r['parameter']==1 else (8,25 if r['parameter']==2 else 21)
                    if method=='State-only Expansion Pruning':
                        label=f'State {r["parameter"]:g}%'
                        dx,dy=(8,-18) if r['parameter']==2 else (-80,28) if r['parameter']==1 else (8,-17)
                    elif 'State-only Expansion Pruning' in COLORS:label=f'M3 {r["parameter"]:g}%'
                elif method=='Early Termination':
                    dx,dy=((8,30) if key!='latency_us' else (-60,-20)) if r['parameter']==95 else (8,-23)
                elif r['parameter']==100:dy=-5
                text(c,x+dx,y+dy,label,11,color=COLORS[method])
    text(c,55,958,'Early Termination: existing DiskANN beam-boundary I/O cutoff; up to 3 reads above nominal budget.',13)
    text(c,55,980,'M3: 512 MB vectors in RAM. State-only: no full-v table. Both include gate inference. Latency bars: trial min/max.' if 'State-only Expansion Pruning' in COLORS else 'M3: validation-calibrated FNR targets; 512 MB candidate vectors in RAM; inference included. Latency bars: trial min/max.',13)
    if suffix=='.png':assert png(surf,str(path).encode())==0
    finish(surf);destroy(c);surface_destroy(surf)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-state',action='store_true')
    args=parser.parse_args()
    if args.with_state:
        OUT=E/'online_curves_state'
        COLORS['State-only Expansion Pruning']=(0.58,0.25,0.70)
    data=json.loads((OUT/'joint_summary.json').read_text())
    assert set(r['method'] for r in data)==set(COLORS)
    runs=json.loads((OUT/'joint_runs.json').read_text())
    gt=mat(E/'online_m3/test_gt.bin','<u4')
    for r in data:
        trials=[t for t in runs if t['key']==r['key']]
        assert len(trials)==3
        expected=None
        for t in trials:
            ids=mat(OUT/(t['label']+'_100_idx_uint32.bin' if t['method']!='DiskANN varying L'
                         else t['label']+f'_{t["L"]}_idx_uint32.bin'),'<u4')
            if expected is None:expected=ids
            else:np.testing.assert_array_equal(ids,expected)
            stats=np.genfromtxt(OUT/(t['label']+'.csv'),delimiter=',',names=True)
            assert len(stats)==1000
            if t['method']=='Early Termination':
                assert stats['ssd_reads'].max()<=t['parameter']+3
                assert not stats['gate_evaluated'].any()
            elif 'Pruning' in t['method']:
                np.testing.assert_array_equal(stats['expansions'],stats['gate_evaluated']-stats['gate_skipped'])
        hits=int(np.any(expected[:,:,None]==gt[:,None,:10],axis=2).sum())
        assert abs(hits/100-r['recall_pct'])<1e-6
    for ext in ['png','svg','pdf']:render(OUT/f'recall_cost_curves.{ext}',data)
    # Verify baseline at L=100 and old gate results agree on search behavior.
    old=json.loads((E/'online_m3/summary.json').read_text())
    baseline=next(r for r in data if r['key']=='L100')
    prior=next(r for r in old if r['suite']=='test1000' and r['config']=='baseline')
    for k in ['recall_pct','ssd_reads_per_query','expansions_per_query']:assert baseline[k]==prior[k]
    for r in data:
        if r['method']!='Learned Expansion Pruning':continue
        prior=next(v for v in old if v['suite']=='test1000' and v['config']==r['key'])
        for k in ['recall_pct','ssd_reads_per_query','expansions_per_query']:assert r[k]==prior[k]
    # Reads budget at 140 should reproduce unrestricted L=100 on this dataset.
    high=next(r for r in data if r['key']=='io140')
    for k in ['recall_pct','ssd_reads_per_query','expansions_per_query']:assert high[k]==baseline[k]
    lines=['# Recall / cost comparison','','![Curves](recall_cost_curves.png)','',
           '所有方法都使用相同 1000 个 held-out query；三轮所有配置交错随机运行，取中位数。上排展示全部点，下排放大 Recall>=99%。',
           'DiskANN varying L：L=20,40,50,60,70,80,90,100,120,150。',
           'Early Termination：固定 L=100；按现有 io_limit 接口，在 beam 边界检查 I/O 预算 B=20,40,60,70,80,85,90,95,100,105,110,120,140；可能最多超出3个读取，横轴使用实际请求数。',
           'Learned Expansion Pruning：M3，L=100；validation FNR targets=0.1%,0.25%,0.5%,1%,2%；使用上一实验的固定 tau。512MB v 表常驻RAM，推理成本计入延迟。',
           'Latency 是 mean query latency，横线显示三轮 min/max；主机为预热状态。没有从其他 query 集合拼接曲线。',
           f'曲线仅连接实测点，没有平滑拟合。所有{len(runs)}次运行的结果ID重复一致，Recall独立重算通过，I/O预算与gate计数核对通过。',
           'Early Termination 在这里具体指 I/O-budget cutoff，不代表所有提前终止算法。',
           '绘图用本地 Cairo，导出 PNG/SVG/PDF；CSV 和逐次测量保留。重现：先 run_online_curves.py，再 plot_online_curves.py。',
           '已核验 L100 与既有基线、M3 各阈值的 Recall/Reads/Expansion 一致；io140 与 unrestricted L100 一致。']
    if args.with_state:
        lines += ['', 'State-only MLP：17→128→64→1，仅使用 pre_io_state，不加载完整 v 表。五档 FNR 目标与 M3 相同，各自只用 validation 校准阈值。',
                  '本次所有四种方法重新交错测量，未将不同实验时段的 latency 拼到同一图中。', '',
                  '| 模型 | FNR target | tau | Recall@10 % | Expansion/query | SSD Reads/query | Mean latency ms | QPS |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in data:
            if 'Pruning' not in r['method']:continue
            name='State-only' if r['method'].startswith('State-only') else 'M3'
            lines.append(f'| {name} | {r["parameter"]:g}% | {r["threshold"]:.9g} | {r["recall_pct"]:.3f} | {r["expansions_per_query"]:.3f} | {r["ssd_reads_per_query"]:.3f} | {r["latency_us"]/1000:.3f} | {r["qps"]:.2f} |')
        lines += ['', '重现：prepare_online_state_mlp.py；run_online_curves.py --with-state；plot_online_curves.py --with-state。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'verification.json').write_text(json.dumps(dict(measured_runs=len(runs),repeat_result_ids_identical=True,
        recalls_independently_recomputed=True,early_io_budget_max_overshoot=3,gate_counters_verified=True,
        old_baseline_and_gate_behavior_parity=True,io140_matches_unrestricted_L100=True),indent=2))
    print('Plots exported: PNG, SVG, PDF; search behavior parity checks passed.')

if __name__=='__main__':main()
