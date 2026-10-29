from common import *
import importlib.util,math
spec=importlib.util.spec_from_file_location('e1_chart_helpers',E1/'plot_figures.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F);F.OUT=R/'figures';F.OUT.mkdir(exist_ok=True)
COLORS=[(.17,.43,.68),(.84,.35,.18),(.19,.59,.38),(.56,.36,.69)]
def bars(c,box,rows,series,title,signed=False,percent=True):
 x,y,w,h=box;left=x+70;top=y+40;pw=w-95;ph=h-110;bottom=top+ph
 scale=100 if percent else 1;vals=[r[k]*scale for k,_,_ in series for r in rows if r[k] is not None];lo=min([0]+vals) if signed else 0;hi=max([0]+vals);span=max(hi-lo,.01);lo-=span*.1 if signed else 0;hi+=span*.15
 yp=lambda v:bottom-(v-lo)/(hi-lo)*ph;F.text(c,x+10,y+10,title,16,True)
 for v in np.linspace(lo,hi,5):F.segment(c,left,yp(v),left+pw,yp(v),(.88,.9,.92));F.text(c,x+3,yp(v)+4,f'{v:.3f}' if hi<1 else f'{v:.2f}',11)
 n=len(rows);ns=len(series);bw=min(28,pw/n/(ns+1))
 for j,r in enumerate(rows):
  for k,(key,label,col) in enumerate(series):
   if r[key] is None:continue
   xx=left+(j+.5)*pw/n+(k-(ns-1)/2)*(bw+1);v=r[key]*scale;F.rgb(c,*col);F.rectangle(c,xx-bw/2,min(yp(v),yp(0)),bw,max(.7,abs(yp(v)-yp(0))));F.fill(c)
  label=r['strategy'].replace('OracleA','OA').replace('OracleB','OB');F.text(c,left+(j+.5)*pw/n-12,bottom+23,label,11)
 for i,(_,label,col) in enumerate(series):F.segment(c,left+i*180,bottom+47,left+20+i*180,bottom+47,col,4);F.text(c,left+27+i*180,bottom+51,label,11)
def render_all(rows,result,static,upper):
 def draw(c):
  bars(c,(20,120,610,310),rows,[('model_saving','ModelSaving',COLORS[0]),('reuse_rate','ReuseRate',COLORS[1])],'Prediction avoidance (%)')
  bars(c,(650,120,610,310),rows,[('delta_recall_pp','Recall change pp',COLORS[0])],'Additional Recall change (percentage points)',True,False)
  bars(c,(20,465,610,310),rows,[('false_skip_rate','FalseSkipRate',COLORS[1])],'Model-reference False Skip Rate (%)')
  bars(c,(650,465,610,310),rows,[('latency_us','mean us',COLORS[0]),('p95_us','p95 us',COLORS[2])],'Tape graph replay latency (microseconds)',False,False)
 F.canvas('reuse_quality',1300,825,'E3: offline prediction reuse, quality and graph replay','SIFT1M | frozen MLP | 1,000 history / 1,000 test queries | matching overhead excluded | Oracle A/B are not deployable',draw)
 def draw(c):
  baseline=next(r for r in rows if r['strategy']=='R0');rr=[{**r,'read_saving':1-r['reads']/baseline['reads'],'expansion_saving':1-r['expansions']/baseline['expansions']} for r in rows]
  bars(c,(20,130,610,340),rr,[('model_saving','ModelSaving',COLORS[0]),('read_saving','Read reduction',COLORS[1])],'Model inference saving and submitted read reduction (%)',True)
  bars(c,(650,130,610,340),rr,[('reuse_rate','ReuseRate',COLORS[2]),('expansion_saving','Expansion reduction',COLORS[3])],'Reused decisions and expansion change (%)',True)
 F.canvas('calls_reads',1300,510,'E3: prediction avoidance is not additional I/O avoidance','All reductions relative to the same R0 graph replay; negative reduction means an increase.',draw)
def render_both(rows):
 rows=[{**r,'strategy':r['strategy'].replace('OracleA_both','OA2').replace('OracleB_both','OB2')} for r in rows]
 base=next(r for r in rows if r['strategy']=='D0');rows=[{**r,'read_saving':1-r['reads']/base['reads']} for r in rows]
 def draw(c):
  bars(c,(20,120,610,330),rows,[('model_saving','ModelSaving',COLORS[0]),('reuse_rate','ReuseRate',COLORS[2])],'Both-decision inference avoidance (%)')
  bars(c,(650,120,610,330),rows,[('delta_recall_pp','Recall change pp',COLORS[1])],'Recall change (percentage points)',True,False)
  bars(c,(20,490,610,330),rows,[('model_saving','ModelSaving',COLORS[0]),('read_saving','Read reduction',COLORS[1])],'Inference avoidance and read reduction (%)',True)
  bars(c,(650,490,610,330),rows,[('latency_us','mean us',COLORS[0]),('p95_us','p95 us',COLORS[3])],'Offline-tape replay latency (microseconds)',False,False)
 F.canvas('both_decisions',1300,870,'E3 extension: reusing historical Keep as well as Skip','D4 = Hybrid with top3 queries; OA2/OB2 = privileged model-decision / quality-menu oracles. Matching overhead excluded.',draw)
