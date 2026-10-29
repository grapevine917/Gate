from common import *
import math,importlib.util
spec=importlib.util.spec_from_file_location('e1_chart_helpers',E1/'plot_figures.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F);F.OUT=R/'figures';F.OUT.mkdir(exist_ok=True)
from plot_online_curves import rgba
def panel(c,box,series,key,title,scale=100,signed=False):
 x,y,w,h=box;left=x+68;top=y+35;pw=w-105;ph=h-105;bottom=top+ph
 vals=[r[key]*scale for _,rows,_ in series for r in rows if r[key] is not None]
 intervals=[r[key+s]*scale for _,rows,_ in series for r in rows for s in ['_lo','_hi'] if r.get(key+s) is not None]
 lo=min([0]+vals+intervals) if signed else 0;hi=max([0]+vals+intervals)
 span=max(hi-lo,.01 if signed else 1);lo-=span*.12 if signed else 0;hi+=span*.12
 if scale==100 and not signed:hi=min(100,hi)
 if hi==lo:hi=lo+1
 yp=lambda v:bottom-(v-lo)/(hi-lo)*ph
 F.text(c,x+12,y+10,title,14 if len(title)>43 else 17,True)
 for v in np.linspace(lo,hi,5):F.segment(c,left,yp(v),left+pw,yp(v),(.87,.9,.92));F.text(c,x+4,yp(v)+4,f'{v:.3f}' if signed else f'{v:.2f}',11)
 F.segment(c,left,top,left,bottom,(.3,.3,.3));F.segment(c,left,yp(0),left+pw,yp(0),(.3,.3,.3))
 nseries=len(series);bw=min(45,pw/(4*(nseries+1)))
 for k,(_,rows,col) in enumerate(series):
  for j,r in enumerate(rows):
   xx=left+(j+.5)*pw/4+(k-(nseries-1)/2)*(bw+3)
   if r[key] is None:F.text(c,xx-12,bottom-8,'N/A',10);continue
   value=r[key]*scale;F.rgb(c,*col);F.rectangle(c,xx-bw/2,min(yp(value),yp(0)),bw,max(abs(yp(value)-yp(0)),.8));F.fill(c)
   if r.get(key+'_lo') is not None:
    yl,yh=yp(r[key+'_lo']*scale),yp(r[key+'_hi']*scale);F.segment(c,xx,yl,xx,yh,(.12,.16,.2),1.2);F.segment(c,xx-3,yl,xx+3,yl,(.12,.16,.2));F.segment(c,xx-3,yh,xx+3,yh,(.12,.16,.2))
 for j in range(4):F.text(c,left+(j+.5)*pw/4-10,bottom+25,f'G{j+1}',13)
 F.text(c,left-5,bottom+48,'G1 Q+/S+     G2 Q+/S-     G3 Q-/S+     G4 Q-/S-',11)
def render_all(a,risk):
 same=a['MLP']['same_node']['groups'];cross=a['MLP']['cross_node']['groups'];blue=(.15,.43,.68);orange=(.86,.4,.15)
 ss=[('same node',same,blue),('cross-node matched',cross,orange)]
 def legend(c):
  for i,(name,_,col) in enumerate(ss):F.segment(c,45+280*i,91,75+280*i,91,col,5);F.text(c,85+280*i,96,name,14)
 def draw(c):
  legend(c)
  for i,(key,title,scale) in enumerate([('agreement','Decision agreement (%)',100),('delta_p','Mean absolute score difference',1),('false_skip_rate','Model-reference False Skip Rate (%)',100),('skip_precision','Skip reuse precision (%)',100)]):panel(c,(20+(i%2)*610,135+(i//2)*340,590,315),ss,key,title,scale)
 F.canvas('joint_groups',1260,835,'E2: query similarity × 17D state similarity','Train-only scaling and group cutoffs | independent MLP traces | bars: target-query cluster bootstrap 95% CI',draw)
 def draw(c):
  p=np.load(R/'MLP_same_node_pairs.npz');rng=np.random.default_rng(SEED);take=rng.choice(len(p['delta_p']),min(5000,len(p['delta_p'])),replace=False)
  for j,key in enumerate(['query_distance','state_distance']):
   left,top,pw,ph=85+j*610,155,490,330;values=np.log1p(p[key][take]) if key=='state_distance' else p[key][take];y=p['delta_p'][take];xmin=float(values.min());xmax=float(values.max())
   xp=lambda v:left+(v-xmin)/(xmax-xmin)*pw;yp=lambda v:top+ph-v*ph
   title='Query L2 vs score difference' if j==0 else 'log(1 + normalized state L2) vs score difference';F.text(c,left-25,120,title,17,True)
   for v in np.linspace(0,1,6):F.segment(c,left,yp(v),left+pw,yp(v),(.88,.9,.92));F.text(c,left-40,yp(v)+4,f'{v:.1f}',12)
   for v in np.linspace(xmin,xmax,5):F.text(c,xp(v)-14,top+ph+25,f'{v:.1f}',12)
   for x,y,k in zip(values,y,take):
    col=blue if p['agreement'][k] else (.85,.17,.15);rgba(c,*col,.24);F.arc(c,xp(x),yp(y),1.8,0,2*math.pi);F.fill(c)
   F.segment(c,left,top,left,top+ph,(.3,.3,.3));F.segment(c,left,top+ph,left+pw,top+ph,(.3,.3,.3));F.text(c,left,top+ph+55,'Blue: same decision. Red: disagreement. Score difference is absolute retain-probability gap.',11)
 F.canvas('distance_score_difference',1260,575,'E2: distances and model score differences','5,000 seeded same-node event pairs shown; all event pairs retained in NPZ artifacts.',draw)
 def draw(c):
  series=[('same node',risk['same_node']['groups'],blue),('cross-node matched',risk['cross_node']['groups'],orange)];legend(c)
  for i,(key,title,scale,signed) in enumerate([('recall_delta_pp','Mean final Recall change (percentage points)',1,True),('harm_rate','Final Recall decreased: all replacements (%)',100,False),('harm_given_false_skip','Recall loss given donor Skip / current Keep (%)',100,False),('read_delta','Submitted read change / replacement',1,True)]):panel(c,(20+(i%2)*610,135+(i//2)*340,590,315),series,key,title,scale,signed)
 F.canvas('intervention_risk',1260,835,'E2: one historical decision substituted into fresh search','20,161 unique replacement searches | exact prefix-state checks | event-weighted results; repeated donors clustered by target query',draw)
 def draw(c):
  left,top,pw,ph=90,160,1000,270;methods=['query_only','state_only','joint'];rows=a['MLP']['same_node']['predictive_metrics'];colors=[blue,orange,(.2,.59,.38)]
  for v in np.linspace(.5,1,6):yy=top+ph-(v-.5)/.5*ph;F.segment(c,left,yy,left+pw,yy,(.88,.9,.92));F.text(c,40,yy+4,f'{v:.2f}',13)
  for i,(r,col) in enumerate(zip(rows,colors)):
   xx=left+100+i*330;hh=(r['auc']-.5)/.5*ph;F.rgb(c,*col);F.rectangle(c,xx,top+ph-hh,130,hh);F.fill(c);F.text(c,xx,top+ph-hh-12,f"{r['auc']:.4f}",16,True);F.text(c,xx-10,top+ph+30,methods[i],15)
  F.text(c,90,top+ph+66,'AUROC for decision agreement. All probes fitted on train/train event pairs; scored on history/test pairs.',14)
 F.canvas('agreement_explanation',1180,540,'E2: held-out agreement explanation','Frozen one-dimensional log-distance logistic probes and their joint model; this is not gate retraining.',draw)
