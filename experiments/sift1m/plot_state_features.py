#!/usr/bin/env python3
"""Render feature-importance bars and exact TreeSHAP summary using native Cairo."""
import json,math
from pathlib import Path
import numpy as np
import plot_online_curves as g
E=Path(__file__).resolve().parent;OUT=E/'state_feature_importance'

def surface(path,w,h):
    return g.image_surface(0,w,h) if path.suffix=='.png' else g.svg_surface(str(path).encode(),w,h) if path.suffix=='.svg' else g.pdf_surface(str(path).encode(),w,h)
def done(path,c,s):
    if path.suffix=='.png':assert g.png(s,str(path).encode())==0
    g.finish(s);g.destroy(c);g.surface_destroy(s)

def bars(path,rows):
    w,h=1850,685;s=surface(path,w,h);c=g.create(s);g.rgb(c,1,1,1);g.paint(c)
    g.text(c,35,38,'Which search-state features does XGBoost use?',25,True)
    g.text(c,35,66,'Effective early-stopped model: 250 trees | all 111,841 held-out expansions / 1,000 queries | test AUC=0.930310',15)
    panels=[('XGBoost gain','gain_fraction','Normalized average split gain (%)',(0.12,0.36,0.71),100),
            ('Permutation importance','permutation_auc_drop_mean','Decrease in held-out ROC-AUC',(0.88,0.40,0.09),1),
            ('Exact TreeSHAP','shap_mean_abs_logodds','Mean |SHAP| (retain log-odds)',(0.08,0.55,0.36),1)]
    for panel,(title,key,xlabel,color,factor) in enumerate(panels):
        origin=25+panel*610;left=origin+235;pw=325;top=135;row_h=25
        ranked=sorted(rows,key=lambda r:r[key],reverse=True)
        values=np.array([r[key]*factor for r in ranked]);xmax=100 if factor==100 else 0.25 if key=='permutation_auc_drop_mean' else 2.5
        xmin=min(0,values.min()*1.1);xp=lambda x:left+(x-xmin)/(xmax-xmin)*pw
        g.text(c,origin,108,title,20,True)
        for val in np.linspace(xmin,xmax,5 if factor==100 else 6):
            x=xp(val);g.segment(c,x,top-15,x,top+17*row_h,(0.90,0.92,0.94))
            g.text(c,x-12,top+17*row_h+20,f'{val:.3f}' if factor==1 and xmax<1 else f'{val:.2f}',12)
        for i,r in enumerate(ranked):
            y=top+i*row_h;g.text(c,origin,y+4,r['feature'],12)
            g.rgb(c,*color);g.rectangle(c,min(xp(0),xp(r[key]*factor)),y-8,max(abs(xp(r[key]*factor)-xp(0)),0.6),16);g.fill(c)
            if key=='permutation_auc_drop_mean':
                sd=r['permutation_auc_drop_std'];g.segment(c,xp(r[key]-sd),y,xp(r[key]+sd),y,(0.1,0.12,0.14),1)
            label=f'{r[key]*factor:.2f}' if factor==100 else f'{r[key]:.4f}'
            g.text(c,xp(r[key]*factor)+5,y+4,label,11)
        g.text(c,origin+170,top+17*row_h+48,xlabel,13)
    g.text(c,35,635,'Permutation: 10 row shuffles per feature; error bars = shuffle SD. Correlated features can create out-of-distribution states.',14)
    g.text(c,35,660,'SHAP: exact native tree-path-dependent contributions. Positive SHAP favors retain; negative favors skip. Constants are not tested interventions.',14)
    done(path,c,s)

def beeswarm(path,rows,sample_indices):
    w,h=1220,700;s=surface(path,w,h);c=g.create(s);g.rgb(c,1,1,1);g.paint(c)
    ordered=sorted(rows,key=lambda r:r['shap_mean_abs_logodds'],reverse=True)
    names=json.loads((E/'training_data_l100/manifest.json').read_text())['pre_io_state_columns']
    x=np.load(E/'training_data_l100/test/pre_io_state.npy',mmap_mode='r')[sample_indices]
    shap=np.load(OUT/'test_shap_values_with_bias.npy',mmap_mode='r')[sample_indices,:17]
    g.text(c,40,38,'Exact TreeSHAP: effect and feature-value direction',23,True)
    g.text(c,40,66,'Ranks use all test rows; display sample = 2 expansions per held-out query (2,000 rows), seed=42',14)
    left,top,pw,rh=260,124,780,28
    low=float(np.min(shap));high=float(np.max(shap));pad=0.03*(high-low);low-=pad;high+=pad
    xp=lambda v:left+(v-low)/(high-low)*pw
    for v in np.linspace(low,high,7):
        xx=xp(v);g.segment(c,xx,top-14,xx,top+17*rh,(0.90,0.92,0.94))
        g.text(c,xx-14,top+17*rh+24,f'{v:.1f}',13)
    g.segment(c,xp(0),top-14,xp(0),top+17*rh,(0.4,0.45,0.5),1.3)
    rng=np.random.default_rng(42)
    for i,r in enumerate(ordered):
        j=names.index(r['feature']);yy=top+i*rh;g.text(c,35,yy+4,r['feature'],13)
        vals=shap[:,j];f=x[:,j];fmin,fmax=np.percentile(f,[2,98])
        color_value=np.clip((f-fmin)/max(float(fmax-fmin),1e-12),0,1)
        # Density-aware vertical spreading in SHAP bins; no invented feature values.
        bins=np.clip(((vals-low)/(high-low)*80).astype(int),0,79);counts=np.bincount(bins,minlength=80)
        jitter=rng.uniform(-1,1,len(vals))*11*np.sqrt(counts[bins]/max(1,counts.max()))
        order=rng.permutation(len(vals))
        for k in order:
            v=float(color_value[k]);g.rgba(c,0.15+0.77*v,0.44-0.18*v,0.80-0.40*v,0.48)
            g.arc(c,xp(float(vals[k])),yy+float(jitter[k]),1.65,0,2*math.pi);g.fill(c)
    g.text(c,left+200,top+17*rh+53,'SHAP value (contribution to retain log-odds)',14)
    for i in range(220):
        v=1-i/219;g.rgb(c,0.15+0.77*v,0.44-0.18*v,0.80-0.40*v);g.rectangle(c,1090,top+i*1.7,17,1.9);g.fill(c)
    g.text(c,1080,top-15,'High',13);g.text(c,1080,top+398,'Low',13)
    g.save(c);g.translate(c,1140,top+250);g.rotate(c,-math.pi/2);g.text(c,0,0,'Feature value',14);g.restore(c)
    g.text(c,40,668,'Blue = low feature value; red = high. SHAP describes this fitted model, not causal effects.',14)
    done(path,c,s)

def main():
    assert json.loads((OUT/'COMPLETE.json').read_text())['complete']
    rows=json.loads((OUT/'importance.json').read_text())
    m=np.load(E/'training_data_l100/test/metadata.npy',mmap_mode='r');q=m['query_id'];rng=np.random.default_rng(42)
    sample=np.concatenate([rng.choice(np.flatnonzero(q==qid),2,replace=False) for qid in np.unique(q)])
    np.save(OUT/'shap_display_sample_indices.npy',sample)
    for ext in ['png','svg','pdf']:
        bars(OUT/f'importance_comparison.{ext}',rows)
        beeswarm(OUT/f'shap_summary.{ext}',rows,sample)
    report=OUT/'REPORT.md';prior=report.read_text().split('\n## Figures')[0]
    report.write_text(prior+'\n## Figures\n\n![Importance comparison](importance_comparison.png)\n\n![SHAP summary](shap_summary.png)\n\nSHAP图仅抽样显示，每个query两条；排名与mean abs数值来自全部111841条。蓝色代表低特征值，红色代表高特征值。\n')
    print('Feature importance and SHAP plots exported: PNG/SVG/PDF.',flush=True)

if __name__=='__main__':main()
