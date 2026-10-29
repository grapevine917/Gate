#!/usr/bin/env python3
"""Plot offline feature ablation and three live recall/cost curves via Cairo."""
import json,math
import numpy as np
from plot_online_curves import (image_surface,svg_surface,pdf_surface,create,rgb,paint,
    text,segment,arc,fill,width,move,line,stroke,save,restore,rectangle,clip,translate,
    rotate,png,finish,destroy,surface_destroy)
from run_online_m3 import E
OUT=E/'feature_ablation';COLORS={'DiskANN':(.18,.25,.32),'Top1':(.86,.40,.10),
    'Top3':(.12,.43,.75),'All17':(.49,.24,.67)}

def surface(path,w,h):
    return image_surface(0,w,h) if path.suffix=='.png' else svg_surface(str(path).encode(),w,h) if path.suffix=='.svg' else pdf_surface(str(path).encode(),w,h)
def done(path,s,c):
    if path.suffix=='.png':assert png(s,str(path).encode())==0
    finish(s);destroy(c);surface_destroy(s)
def dot(c,x,y,color,r=4.5):
    rgb(c,*color);arc(c,x,y,r,0,2*math.pi);fill(c)

def offline(path,rows):
    s=surface(path,1320,490);c=create(s);rgb(c,1,1,1);paint(c)
    text(c,35,36,'State feature ablation: identical MLP training protocol',24,True)
    text(c,35,65,'Validation selects thresholds; test reports AUC and actual skip rates. Feature selection is exploratory.',14)
    panels=[('test_auc','Test AUC',.90,.94),('test_skip_rate_0.1pct','SkipRate @ validation FNR = 0.1%',0,.07),
            ('test_skip_rate_1pct','SkipRate @ validation FNR = 1%',0,.20)]
    for panel,(key,title,lo,hi) in enumerate(panels):
        left=65+panel*435;top=135;pw=365;ph=245;bottom=top+ph
        text(c,left-20,110,title,17,True)
        yp=lambda v:bottom-(v-lo)/(hi-lo)*ph
        for v in np.linspace(lo,hi,5):
            segment(c,left,yp(v),left+pw,yp(v),(.88,.90,.92))
            label=f'{v:.2f}' if key=='test_auc' else f'{v*100:g}%'
            text(c,left-48,yp(v)+5,label,12)
        for i,r in enumerate(rows):
            x=left+15+i*72;v=r[key];color=COLORS.get(r['model'],(.12,.56,.37))
            rgb(c,*color);rectangle(c,x,yp(v),42,bottom-yp(v));fill(c)
            text(c,x-1,bottom+24,r['model'],13)
            text(c,x-7,yp(v)-9,f'{v:.4f}' if key=='test_auc' else f'{v*100:.2f}%',12)
        segment(c,left,bottom,left+pw,bottom,(.25,.3,.35))
    text(c,35,449,'Top5 adds candidate_pq_over_best + unexpanded_count; actual test FNR is reported in CSV / report.',14)
    done(path,s,c)

def online(path,rows):
    s=surface(path,1820,970);c=create(s);rgb(c,1,1,1);paint(c)
    text(c,40,36,'Live DiskANN state feature ablation: recall / cost tradeoffs',25,True)
    text(c,40,66,'1,000 held-out queries | L=100 | K=10 | beam=4 | threads=4 | cache=10,000 | 3 interleaved trials, median',15)
    for i,(label,col) in enumerate(COLORS.items()):
        x=50+i*340;segment(c,x,100,x+30,100,col,2.5);dot(c,x+15,100,col)
        text(c,x+42,105,'DiskANN, no model' if label=='DiskANN' else label+' expansion pruning',15)
    keys=[('ssd_reads_per_query','SSD read requests / query'),
          ('expansions_per_query','Expansions / query'),('latency_us','Mean query latency (ms)')]
    minrec=min(r['recall_pct'] for r in rows)
    for row in range(2):
        lo=max(0,math.floor(minrec*10)/10-.05) if row==0 else 99.5
        hi=math.ceil(max(r['recall_pct'] for r in rows)*10)/10+.05
        for panel,(key,title) in enumerate(keys):
            left=82+panel*602;top=175+row*375;pw=475;ph=255;bottom=top+ph
            values=[r[key]/1000 if key=='latency_us' else r[key] for r in rows]
            if key=='latency_us':
                values+=[r['latency_min_us']/1000 for r in rows]+[r['latency_max_us']/1000 for r in rows]
            span=max(values)-min(values);pad=max(span*.12,.25 if key=='latency_us' else 2)
            xmin=min(values)-pad;xmax=max(values)+pad
            xp=lambda v:left+(v-xmin)/(xmax-xmin)*pw
            yp=lambda v:bottom-(v-lo)/(hi-lo)*ph
            text(c,left-30,top-24,('All points: ' if row==0 else 'High recall: ')+title,17,True)
            for v in np.linspace(lo,hi,5):
                segment(c,left,yp(v),left+pw,yp(v),(.88,.90,.92))
                text(c,left-53,yp(v)+5,f'{v:.2f}',12)
            for v in np.linspace(xmin,xmax,5):
                segment(c,xp(v),top,xp(v),bottom,(.93,.94,.95))
                text(c,xp(v)-16,bottom+24,f'{v:.2f}' if key=='latency_us' else f'{v:.1f}',12)
            segment(c,left,top,left,bottom,(.25,.3,.35));segment(c,left,bottom,left+pw,bottom,(.25,.3,.35))
            text(c,left+95,bottom+49,title,14)
            save(c);translate(c,left-65,top+185);rotate(c,-math.pi/2);text(c,0,0,'Recall@10 (%)',14);restore(c)
            save(c);rectangle(c,left,top,pw,ph);clip(c)
            for model,col in COLORS.items():
                rr=sorted([r for r in rows if r['model']==model],key=lambda r:r['target_fnr_pct'])
                rgb(c,*col);width(c,2.3)
                for i,r in enumerate(rr):
                    x=xp(r[key]/1000 if key=='latency_us' else r[key]);y=yp(r['recall_pct'])
                    (move if i==0 else line)(c,x,y)
                stroke(c)
                for r in rr:
                    x=xp(r[key]/1000 if key=='latency_us' else r[key]);y=yp(r['recall_pct'])
                    if key=='latency_us':segment(c,xp(r['latency_min_us']/1000),y,xp(r['latency_max_us']/1000),y,col)
                    dot(c,x,y,col,6 if model=='DiskANN' else 4.5)
            restore(c)
    text(c,40,912,'Pruning curves: validation FNR targets 0.1%, 0.25%, 0.5%, 1%, 2%. Cost points are real live searches; no mask replay.',14)
    text(c,40,939,'Inference and common live state construction included. Warm-host timings; latency bars show trial min/max.',14)
    done(path,s,c)

def main():
    offline_rows=json.loads((OUT/'offline_summary.json').read_text())
    online_rows=json.loads((OUT/'online/summary.json').read_text())
    for ext in ('png','svg','pdf'):
        offline(OUT/f'offline_ablation.{ext}',offline_rows)
        online(OUT/f'online/recall_cost_curves.{ext}',online_rows)
    report=OUT/'REPORT.md';body=report.read_text().split('\n## Figures')[0]
    report.write_text(body+'\n## Figures\n\n![Offline](offline_ablation.png)\n\n![Online](online/recall_cost_curves.png)\n\n[Online detailed report](online/REPORT.md)\n')
    print('Ablation plots exported: PNG / SVG / PDF',flush=True)

if __name__=='__main__':main()
