"""Render local Chapter 4 figures/tables and copyable LaTeX blocks.

From the repository root:
 .\.venv\Scripts\python.exe script/thesis/results_plan_data.py --run-dir <run> --stage physical
 .\.venv\Scripts\python.exe script/thesis/results_plan_data.py --run-dir <run> --stage coefficients
 .\.venv\Scripts\python.exe script/thesis/results_plan_data.py --run-dir <run> --stage ivp
 .\.venv\Scripts\python.exe script/thesis/generate_results_plan.py --run-dir <run>

No fitting or manuscript editing is performed. Output: img/04_results.
Use --examples for representative assets and --output-root to relocate output.
"""
from pathlib import Path
import os,json,re,hashlib,argparse
import tempfile
os.environ.setdefault('MPLBACKEND','Agg')
REPO=Path(__file__).resolve().parents[2]
if __package__:
 from . import manuscript_assets as assets
else:
 import manuscript_assets as assets
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'sindy-viv/results-plan-mpl-cache'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker,font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
FIGDIR=assets.OUTPUT_ROOT/'04_results';TABS=FIGDIR/'data';REFERENCE=TABS/'reference'
PT=72/72.27;W=140/25.4;SMALL=.4*W
FIGURES=[];SOURCES={};PEAKS=[];LPDS=[1.5,2.,3.,3.5,4.]
font_manager.findfont('cmr10',fallback_to_default=False)
plt.rcParams.update({'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm','text.usetex':False,'font.size':10*PT,'axes.labelsize':10*PT,'axes.titlesize':11*PT,'xtick.labelsize':10*PT,'ytick.labelsize':10*PT,'legend.fontsize':10*PT,'axes.unicode_minus':False,'axes.formatter.use_mathtext':True,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'lines.linewidth':1.,'grid.linewidth':.4,'grid.alpha':.5,'pdf.fonttype':42,'svg.fonttype':'path','legend.frameon':False,'savefig.dpi':450})
COLORS=['#111111','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#7B3294','#8C510A','#999999'];MARKERS=['o','x','+','s','^','D','v','P','*','<']

def dump(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def read(p):
 p=Path(p);SOURCES[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text(encoding='utf-8-sig'))
def csv(name,**kwargs):
 p=TABS/name;SOURCES[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p,**kwargs)
def key(kind,lpd,vr):return f'{kind}_'+('' if lpd is None else f'LD{lpd:g}_')+f'VR{vr}'
def final_bbox(fig):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bad=[]
 for ax in fig.axes:
  b=ax.get_tightbbox(renderer)
  if b is not None and (b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5):bad.append([round(x,1) for x in b.extents])
 return bad

def save(fig,stem,caption,main=True,**extra):
 # 3D artists use projected bounds; their labels are checked from the rendered PDF.
 bad=final_bbox(fig) if not extra.get('is_3d') else []
 if bad:raise ValueError(f'Clipped figure: {stem}: {bad}')
 fig.savefig(FIGDIR/f'{stem}.svg');fig.savefig(FIGDIR/f'{stem}.png',dpi=450);fig.savefig(FIGDIR/f'{stem}.pdf');BOOK.savefig(fig)
 FIGURES.append(dict(stem=stem,page=len(FIGURES)+1,caption=caption,width_mm=float(fig.get_figwidth()*25.4),height_mm=float(fig.get_figheight()*25.4),main=main,**extra));plt.close(fig);print('FIGURE',stem,flush=True)

def axis(ax):
 ax.grid(True,lw=.4,alpha=.5);ax.tick_params(width=.8,length=2.5,pad=2)
 for side in ('left','bottom'):ax.spines[side].set_linewidth(.8)
 ax.yaxis.set_major_locator(ticker.MaxNLocator(3));ax.xaxis.set_major_locator(ticker.MaxNLocator(5,integer=True))

def clear_legend(ax):
 # Keep the requested upper-right legend clear of the plotted data.
 legend=ax.get_legend()
 if legend is None:return
 for _ in range(35):
  ax.figure.canvas.draw();box=legend.get_window_extent().padded(3)
  collision=False
  for line in ax.lines:
   path=line.get_path().transformed(line.get_transform())
   if path.intersects_bbox(box,filled=False) or any(box.contains(*v) for v in path.vertices):
    collision=True;break
  if not collision:return
  lo,hi=ax.get_ylim();ax.set_ylim(lo,lo+1.15*(hi-lo))
 raise ValueError('Could not clear upper-right legend')

def marked_line(ax,x,y,label):
 ax.plot(x,y,color='#444444',lw=1.,label=label)
 for j,(xx,yy) in enumerate(zip(x,y)):ax.plot([xx],[yy],linestyle='none',marker=MARKERS[j%len(MARKERS)],ms=3.3,mew=.7,color=COLORS[j%len(COLORS)])

def simple_single(frame,frequency=False):
 fig,ax=plt.subplots(figsize=(SMALL,SMALL*2/3));fig.subplots_adjust(left=.25,right=.97,bottom=.30,top=.96)
 column='f_CL' if frequency else 'A_h';label=r'$f_{C_L}^*$' if frequency else r'$A_h$'
 marked_line(ax,frame.VR,frame[column],label)
 if frequency:ax.plot(frame.VR,1/frame.VR,'--',color='#D55E00',lw=1.,label=r'$f_n^*$')
 ax.set_xlabel('$V_R$',labelpad=1);ax.set_ylabel(label,labelpad=2);axis(ax);ax.set_xticks(range(3,9))
 ax.legend(loc='upper right',handlelength=1.1,handletextpad=.3,borderaxespad=.2,labelspacing=.15,fontsize=10*PT);clear_legend(ax)
 token='frequency' if frequency else 'amplitude'
 save(fig,f'single_{token}_vs_VR',f'Single-cylinder {"dominant lift frequency" if frequency else "displacement amplitude"} in the established-oscillation window. '+('The dashed line is the structural natural frequency, $f_n^*=1/V_R$.' if frequency else 'Amplitude is half the peak-to-peak displacement over the selected window.'))

def tandem_trend(frame,lpd=None,vr=None,frequency=False,main=False):
 selected=frame[frame.LD==lpd] if lpd is not None else frame[frame.VR==vr]
 xcol='VR' if lpd is not None else 'LD';parameter=r'$V_R$' if lpd is not None else r'$L/D$'
 fig,axes=plt.subplots(2,1,figsize=(SMALL,SMALL*4/3),sharex=True)
 fig.subplots_adjust(left=.31 if frequency else .27,right=.97,bottom=.15,top=.90,hspace=.25)
 for i,ax in enumerate(axes):
  rows=selected[selected.cylinder==i].sort_values(xcol);column='f_CL' if frequency else 'A_h';label=rf'$f_{{C_{{L,{i}}}}}^*$' if frequency else rf'$A_{{h,{i}}}$'
  marked_line(ax,rows[xcol],rows[column],label)
  if frequency:ax.plot(rows[xcol],1/rows.VR,'--',color='#D55E00',label=r'$f_n^*$',lw=1.)
  ax.set_ylabel(label,labelpad=2);axis(ax)
  if lpd is not None:ax.set_xticks([5,8,11,14])
  else:ax.set_xticks(LPDS,labels=['1.5','2','3','3.5','4'])
  if i==0:ax.legend(loc='upper right',fontsize=10*PT,handlelength=1.1,handletextpad=.25,borderaxespad=.15,labelspacing=.15);clear_legend(ax)
 axes[1].set_xlabel(parameter,labelpad=1)
 axes[0].set_title(rf'$L/D={lpd:g}$' if lpd is not None else rf'$V_R={vr:g}$',loc='right',fontsize=11*PT,pad=4)
 token=('LD'+str(lpd).replace('.','p')) if lpd is not None else f'VR{vr}'
 quantity='frequency' if frequency else 'amplitude'
 save(fig,f'tandem_{quantity}_vs_{xcol}_{token}',f'Tandem-cylinder {"dominant lift frequencies" if frequency else "displacement amplitudes"} '+(f'at $L/D={lpd:g}$ as $V_R$ varies.' if lpd is not None else f'at $V_R={vr:g}$ as $L/D$ varies.')+' The upper and lower panels show cylinders 0 and 1, respectively. '+('Dashed lines mark $f_n^*=1/V_R$.' if frequency else 'Amplitudes are half peak-to-peak values; no frequency reference is plotted on an amplitude axis.'),main=main)

def waterfall(run,kind,lpd=None,cylinder=0,main=True):
 width=W;height=.66*W
 fig=plt.figure(figsize=(width,height));ax=fig.add_axes([.01,.06,.90,.90],projection='3d')
 vrs=range(3,9) if kind=='single' else range(5,15)
 for j,vr in enumerate(vrs):
  path=run/'cache'/f'{key(kind,lpd,vr)}_spectrum.npz'
  with np.load(path) as z:
   names=z['names'].tolist();freq=z['frequency'];amp=z['amplitude'][:,names.index(f'C_{{L{cylinder}}}')]
  ax.plot(freq,np.full_like(freq,vr),amp,lw=1.,color=COLORS[j%10])
  peaks=find_peaks(amp,height=.05*float(amp.max()))[0]
  if len(peaks):ax.scatter(freq[peaks],np.full(len(peaks),vr),amp[peaks],s=8,marker='o',color=COLORS[j%10],edgecolors='#222222',linewidths=.2,depthshade=False)
  for b in peaks:PEAKS.append(dict(kind=kind,LD=lpd,VR=vr,cylinder=cylinder,frequency=float(freq[b]),amplitude=float(amp[b]),relative_to_largest_peak=float(amp[b]/amp.max())))
 ax.set(xlim=(0,2.5),xlabel='$f^*$',ylabel='$V_R$',zlabel=rf'$|\widehat C_{{L,{cylinder}}}|$' if kind=='tandem' else r'$|\widehat C_L|$')
 ax.set_xticks([0,1,2,2.5]);ax.set_yticks(list(vrs)[::2]);ax.zaxis.set_major_locator(ticker.MaxNLocator(3))
 ax.tick_params(labelsize=10*PT,pad=1);ax.xaxis.labelpad=4;ax.yaxis.labelpad=4;ax.zaxis.labelpad=4
 for dim in (ax.xaxis,ax.yaxis,ax.zaxis):dim.line.set_linewidth(.8);dim._axinfo['grid'].update(linewidth=.4,color=(.4,.4,.4,.5))
 ax.view_init(elev=24,azim=-62)
 suffix='' if lpd is None else f'_LD{str(lpd).replace(".","p")}_CL{cylinder}'
 save(fig,f'{kind}_CL_waterfall{suffix}',f'One-sided lift FFT amplitudes for the {"single cylinder" if kind=="single" else f"tandem cylinder {cylinder} at $L/D={lpd:g}$"} in the established-oscillation window. The signals are demeaned, use a rectangular FFT window and are shown over $0\leq f^*\leq2.5$. Markers identify all resolved local peaks at least $5\%$ of the largest lift peak, including the fundamental; amplitudes are not normalized.',main=main,is_3d=True)

def filtering(run):
 with np.load(run/'cache/single_VR7_filtering.npz') as z:t=z['time_star'];raw=z['raw'];filtered=z['filtered']
 fig,axes=plt.subplots(2,1,figsize=(.9*W,.60*W),sharex=True);fig.subplots_adjust(left=.16,right=.97,bottom=.20,top=.97,hspace=.20)
 for j,ax in enumerate(axes):
  ax.plot(t,raw[:,j],color='black',lw=1.,label='Original');ax.plot(t,filtered[:,j],color='red',ls='--',lw=1.,label='Filtered');ax.set_ylabel('$h$' if j==0 else '$C_L$',labelpad=3);axis(ax)
 axes[0].legend(loc='upper right',ncol=2,columnspacing=.7,fontsize=10*PT,handlelength=1.4);clear_legend(axes[0]);axes[1].set_xlabel('$t^*$',labelpad=1);axes[1].set_xlim(1000,1050)
 save(fig,'single_VR7_filtering','Original and fourth-order Butterworth-filtered single-cylinder signals at $V_R=7$ on the evaluation window, $t^*=1000$--$1050$. The complete history is filtered before the window is selected; the cutoff is $f_c^*=2$.')


def lift_derivative_example(run):
 data=csv('single_VR5_lift_derivatives.csv')
 fig,axes=plt.subplots(3,1,figsize=(W,.66*W),sharex=True)
 fig.subplots_adjust(left=.15,right=.96,bottom=.13,top=.91,hspace=.17)
 for ax,column,label in zip(axes,['CL','CLdot_tau','CLddot_tau'],[r'$C_L$',r'$\dot C_L$',r'$\ddot C_L$']):
  ax.plot(data.tstar,data[column],color='#0072B2',lw=1.)
  ax.set_ylabel(label,labelpad=2,fontsize=10.5*PT);axis(ax);ax.tick_params(labelsize=10.5*PT);ax.set_xlim(500,525)
 axes[0].set_title(r'$V_R=5$',loc='right',fontsize=12*PT,pad=5)
 axes[-1].set_xlabel('$t^*$',fontsize=10.5*PT,labelpad=2)
 save(fig,'single_VR5_lift_derivatives','Filtered lift and its first two derivatives for the single cylinder at $V_R=5$, on $500\\leq t^*\\leq525$. Dots denote differentiation with respect to $\\tau=t^*/V_R$. The full record is processed before the displayed interval is selected.')


def heatmap(frame,stem,title,caption,horizontal_tandem=False,small=False,scale=None):
 a=frame.to_numpy(float);nr,nc=a.shape
 if horizontal_tandem:left=.57;right=.78;cell=(W-left-right)/nc;tick=10;annotation=9
 else:left=1.02;right=1.10;cell=.50;tick=10 if not small else 9;annotation=9 if not small else 8.5
 bottom=.60;top=.38;fw=left+nc*cell+right;fh=bottom+nr*cell+top
 fig=plt.figure(figsize=(fw,fh));ax=fig.add_axes([left/fw,bottom/fh,nc*cell/fw,nr*cell/fh])
 limit=scale or max(float(np.abs(a).max()),1e-9);norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit);cmap=plt.get_cmap('RdBu_r',2048)
 for i,j in np.ndindex(a.shape):
  colour=cmap(norm(a[i,j]));ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=colour,edgecolor='#666666',linewidth=.3,antialiased=False,rasterized=False,gid=f'heatmap_cell_{i}_{j}'))
  ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',fontsize=annotation*PT,color='white' if abs(a[i,j])>.65*limit else '#111111')
 def feature(s):
  s=str(s);s=s[2:] if s.startswith('1 ') else s
  return '$'+s.replace(' ',r'\,')+'$'
 ax.set(xlim=(-.5,nc-.5),ylim=(nr-.5,-.5));ax.set_aspect('equal');ax.set_xticks(range(nc),labels=[f'{float(v):g}' for v in frame.columns],fontsize=tick*PT);ax.set_yticks(range(nr),labels=[feature(v) for v in frame.index],fontsize=tick*PT)
 ax.tick_params(length=0,pad=3);ax.set_xlabel('$L/D$' if small else '$V_R$',fontsize=tick*PT,labelpad=5);ax.set_ylabel('Library term',fontsize=tick*PT,labelpad=3);ax.set_title(title,fontsize=11*PT,pad=6)
 for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.8)
 cax=fig.add_axes([(left+nc*cell+.09)/fw,bottom/fh,.11/fw,nr*cell/fh]);m=ScalarMappable(norm=norm,cmap=cmap);m.set_array(a);cb=fig.colorbar(m,cax=cax,ticks=[-limit,0,limit]);cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'));cb.ax.tick_params(labelsize=tick*PT,width=.8,pad=2);cb.set_label('Coefficient Value',fontsize=tick*PT,labelpad=3)
 save(fig,stem,caption,matrix_renderer='vector_rectangles',colour_limits=[-limit,limit])

def get_trace(run,kind,lpd,vr,method):
 name=key(kind,lpd,vr)
 if method=='theory':
  tag=f'{kind}_theory_'+('' if lpd is None else f'LD{lpd:g}_')+f'VR{vr}'
  path=REFERENCE/'data'/f'{tag}.npz';metadata=read(path.with_suffix('.json'));names_key='variable_names'
 else:
  path=run/'cache'/f'{name}_{method}_overall_reduced_ivp.npz';metadata=read(path.with_suffix('.json'));names_key='names'
 if not metadata.get('complete'):raise ValueError(f'Cannot plot incomplete structural trace as a full IVP: {path}')
 SOURCES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
 with np.load(path) as z:return dict(time=z['time_star'].copy(),reference=z['reference'].copy(),prediction=z['prediction'].copy(),names=z[names_key].tolist())

def structural_ivps(run):
 styles={'individual':('Individual','#0072B2','--'),'parametric_vr':('Parameterized','$D55E00','-.'),'theory':('Theoretical','#009E73',':')};styles['parametric_vr']=('Parameterized','#D55E00','-.')
 fig,axes=plt.subplots(2,2,figsize=(W,.75*W),sharex='col',sharey='row',gridspec_kw={'height_ratios':[2,1]});fig.subplots_adjust(left=.15,right=.97,bottom=.22,top=.91,wspace=.18,hspace=.13)
 for col,vr in enumerate((4,5)):
  ref=get_trace(run,'single',None,vr,'theory');j=ref['names'].index('h_0');axes[0,col].plot(ref['time'],ref['reference'][:,j],color='black',lw=1.,label='Reference')
  for method,(label,colour,ls) in styles.items():
   r=get_trace(run,'single',None,vr,method);k=r['names'].index('h_0');axes[0,col].plot(r['time'],r['prediction'][:,k],color=colour,ls=ls,lw=1.,label=label);axes[1,col].plot(r['time'],r['prediction'][:,k]-r['reference'][:,k],color=colour,ls=ls,lw=1.)
  for row in (0,1):axis(axes[row,col]);axes[row,col].set_xlim(600,650);axes[row,col].set_xticks([600,625,650])
  axes[0,col].set_title(rf'$V_R={vr}$',loc='right',fontsize=11*PT);axes[0,col].tick_params(bottom=False)
 axes[0,0].set_ylabel('$h$');axes[1,0].set_ylabel(r'$\widehat h-h$');fig.supxlabel('$t^*$',y=.10,fontsize=10*PT)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.015),handlelength=1.4,columnspacing=.7,fontsize=10*PT)
 save(fig,'single_structural_IVP_overall_reduced','Displacement IVPs at $V_R=4$ and 5 for the individual, $V_R$-parameterized and theoretical equations. Identified models use transient-inclusive training. Measured lift is prescribed. Upper panels show displacement and lower panels prediction minus reference.')
 tandem_styles={'individual':('Individual','#0072B2','--'),'parametric_vr':('$V_R$ fit','#D55E00','-.'),'parametric_vr_lpd':('$V_R,L/D$ fit','#7B3294','--'),'theory':('Theoretical','#009E73',':')}
 for lpd,vr in [(2.,12),(4.,6)]:
  fig,axes=plt.subplots(2,1,figsize=(W,.64*W),sharex=True);fig.subplots_adjust(left=.13,right=.97,bottom=.24,top=.89,hspace=.16)
  ref=get_trace(run,'tandem',lpd,vr,'theory')
  for i,ax in enumerate(axes):
   j=ref['names'].index(f'h_{i}');ax.plot(ref['time'],ref['reference'][:,j],color='black',lw=1.,label='Reference')
   for method,(label,colour,ls) in tandem_styles.items():
    r=get_trace(run,'tandem',lpd,vr,method);k=r['names'].index(f'h_{i}');ax.plot(r['time'],r['prediction'][:,k],color=colour,ls=ls,lw=1.,label=label)
   axis(ax);ax.set_ylabel(rf'$h_{i}$');ax.set_xlim(1350,1400)
  axes[0].tick_params(bottom=False);axes[1].set_xticks([1350,1375,1400]);fig.suptitle(rf'$L/D={lpd:g},\quad V_R={vr}$',fontsize=11*PT,y=.98);fig.supxlabel('$t^*$',fontsize=10*PT,y=.12)
  handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,.015),handlelength=1.4,columnspacing=.7,fontsize=10*PT)
  save(fig,f'tandem_structural_IVP_LD{lpd:g}_VR{vr}_overall_reduced',f'Tandem structural IVPs at $L/D={lpd:g}$, $V_R={vr}$ using transient-inclusive fits. Cylinder 0 is above and cylinder 1 below. Measured lift histories are prescribed. Individual, fixed-spacing $V_R$ and joint $V_R$--$L/D$ fits use the same training and evaluation windows.')

def aggregate_tables(run):
 frames=[pd.read_csv(run/name) for name in ['existing_structural_ivp_metrics.csv','new_structural_ivp_metrics.csv','joint_structural_ivp_metrics.csv']];metrics=pd.concat(frames,ignore_index=True)
 # The existing theoretical reference uses the same corrected input and later evaluation windows.
 for kind in ('single','tandem'):
  if kind=='single':
   frame=pd.read_csv(REFERENCE/'tables/single_structural_h_ivp_nrmse.csv')
   for _,r in frame.iterrows():metrics=pd.concat([metrics,pd.DataFrame([dict(kind=kind,method='theory',LD=None,VR=r.VR,cylinder=0,complete=True,NRMSE_pct=r.Theoretical_pct)])],ignore_index=True)
  else:
   for i in (0,1):
    frame=pd.read_csv(REFERENCE/f'tables/tandem_structural_h{i}_ivp_nrmse.csv')
    for _,r in frame.iterrows():metrics=pd.concat([metrics,pd.DataFrame([dict(kind=kind,method='theory',LD=r.LD,VR=r.VR,cylinder=i,complete=True,NRMSE_pct=r.Theoretical_pct)])],ignore_index=True)
 metrics.to_csv(TABS/'structural_ivp_metrics_long.csv',index=False)
 for kind in ('single','tandem'):
  group=metrics[metrics.kind==kind]
  if kind=='single':pivot=group.pivot(index='VR',columns='method',values='NRMSE_pct').reset_index()
  else:pivot=group.pivot(index=['LD','VR'],columns=['method','cylinder'],values='NRMSE_pct').reset_index();pivot.columns=['_'.join(str(x) for x in c if str(x)!='') if isinstance(c,tuple) else c for c in pivot.columns]
  pivot.to_csv(TABS/f'{kind}_structural_ivp_nrmse.csv',index=False)
 maxima=[]
 for (kind,method,cylinder),group in metrics.groupby(['kind','method','cylinder']):
  valid=group[group.complete&group.NRMSE_pct.notna()];w=valid.loc[valid.NRMSE_pct.idxmax()]
  maxima.append(dict(kind=kind,method=method,cylinder=int(cylinder),maximum_NRMSE_pct=float(w.NRMSE_pct),VR=int(w.VR),LD=None if pd.isna(w.LD) else float(w.LD),complete=int(group.complete.sum()),total=len(group)))
 pd.DataFrame(maxima).to_csv(TABS/'structural_ivp_maxima.csv',index=False)
 pd.DataFrame(columns=['result','training_window','status','reason']).to_csv(TABS/'pending_results.csv',index=False)
 return metrics,pd.DataFrame(maxima)

def tex_number(value):return '--' if pd.isna(value) else f'{float(value):.4g}'

def table_latex():
 tables=[]
 single=csv('single_structural_ivp_nrmse.csv').sort_values('VR')
 tables.append(dict(stem='single_structural_ivp_nrmse',label='tab:04_single_ivp',
  caption=r'Single-cylinder displacement IVP NRMSE (\%) on the later evaluation windows. Measured lift is prescribed.',
  headers=[r'$V_R$','Individual',r'$V_R$ fit','Theoretical'],
  rows=[[str(int(row.VR))]+[tex_number(row[name]) for name in ['individual','parametric_vr','theory']] for _,row in single.iterrows()]))
 tandem=csv('tandem_structural_ivp_nrmse.csv').sort_values(['LD','VR'])
 for cylinder in (0,1):
  for lpd,group in tandem.groupby('LD',sort=True):
   token=str(float(lpd)).replace('.','p')
   tables.append(dict(stem=f'tandem_h{cylinder}_ivp_LD{token}',label=f'tab:04_tandem_h{cylinder}_ivp_LD{token}',
    caption=rf'Tandem displacement $h_{cylinder}$ IVP NRMSE (\%) at $L/D={lpd:g}$ on $t^*=1350$--$1400$. Measured lift is prescribed.',
    headers=[r'$V_R$','Individual',r'$V_R$ fit',r'$V_R,L/D$ fit','Theoretical'],
    rows=[[str(int(row.VR))]+[tex_number(row[f'{method}_{cylinder}']) for method in ['individual','parametric_vr','parametric_vr_lpd','theory']] for _,row in group.iterrows()]))
 assets.write_tables(FIGDIR,tables,dict(headers=[r'$V_R$','Individual',r'$V_R$ fit','Theoretical'],
  caption=r'Displacement IVP NRMSE (\%): replace this caption.',label='tab:04_empty_template'))
 return tables


def render_all(run,examples=False):
 single=csv('single_physical_summary.csv').sort_values('VR');tandem=csv('tandem_physical_summary.csv')
 simple_single(single);simple_single(single,True);waterfall(run,'single')
 # All spacings and velocities are exported; only the selected representatives enter the main text.
 for lpd in ([1.5] if examples else LPDS):
  for freq in (False,True):tandem_trend(tandem,lpd=lpd,frequency=freq,main=lpd==1.5)
 for vr in ([6] if examples else range(5,15)):
  for freq in (False,True):tandem_trend(tandem,vr=vr,frequency=freq,main=vr==6)
 for lpd in ([1.5] if examples else LPDS):
  for i in ([0] if examples else (0,1)):waterfall(run,'tandem',lpd,i,main=lpd==1.5)
 filtering(run)
 lift_derivative_example(run)
 individual=csv('single_structural_individual_coefficients.csv',index_col=0);parameterized=csv('single_structural_parameterized_effective_coefficients.csv',index_col=0);scale=max(float(np.abs(f.to_numpy()).max()) for f in [individual,parameterized])
 heatmap(individual,'single_individual_coefficients_overall_reduced',r'$\ddot h$: Single Cylinder','Single-cylinder structural acceleration coefficients fitted individually using transient-inclusive training. The complete linear library is shown.',scale=scale)
 heatmap(parameterized,'single_parameterized_effective_coefficients_overall_reduced',r'$\ddot h$: Single Cylinder','Effective single-cylinder coefficients evaluated at each $V_R$ from the one shared parameterized model fitted to transient-inclusive data. The complete linear response library and colour scale match the individual-fit map.',scale=scale)
 for lpd in ([1.5] if examples else (1.5,4.)):
  for i in (0,1):
   frame=csv(f'tandem_LD{lpd:g}_ddoth{i}_individual_coefficients.csv',index_col=0)
   heatmap(frame,f'tandem_individual_LD{str(lpd).replace(".","p")}_ddoth{i}_overall_reduced',rf'$\ddot h_{i}$: $L/D={lpd:g}$',f'Individual tandem acceleration coefficients for cylinder {i} at $L/D={lpd:g}$, fitted on transient-inclusive training data. All seven linear-library terms, including zeros, are shown.',horizontal_tandem=True)
 frames=[csv(f'tandem_ddoth{i}_parameterized_coefficients.csv',index_col=0) for i in (0,1)];assert frames[0].index.tolist()==frames[1].index.tolist()
 scale=max(float(np.abs(f.to_numpy()).max()) for f in frames)
 for i,frame in enumerate(frames):heatmap(frame,f'tandem_parameterized_ddoth{i}_all_LD_overall_reduced',rf'$\ddot h_{i}$: parameterized by $V_R$',f'Tandem $V_R$-parameterized acceleration coefficients for cylinder {i} across all spacings, using transient-inclusive training. Each spacing has its own fit. Both cylinder maps use the same ordered union of active terms, including zero rows, and the same colour scale.',small=True,scale=scale)
 joint=[csv(f'tandem_ddoth{i}_joint_coefficients.csv',index_col=0) for i in (0,1)]
 for i,frame in enumerate(joint):heatmap(frame,f'tandem_joint_ddoth{i}_all_LD_overall_reduced',rf'$\ddot h_{i}$: parameterized by $V_R,L/D$',f'Joint $V_R$--$L/D$ structural coefficients for cylinder {i}, fitted to all 50 transient-inclusive trajectories. The effective $V_R$-polynomial coefficients are shown at each spacing, on the same visible library and colour scale as the fixed-spacing fits. No active coefficient depends on $L/D$.',small=True,scale=scale)
 pd.DataFrame(PEAKS).to_csv(TABS/'waterfall_highlighted_peaks.csv',index=False)
 structural_ivps(run)
 return single,tandem

def main(argv=None):
 global BOOK,FIGDIR,TABS,REFERENCE
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run-dir',type=Path,required=True,help='Existing preparation cache/status directory.')
 parser.add_argument('--output-root',type=Path,default=assets.OUTPUT_ROOT,help='Parent of 04_results (default: repository img).')
 parser.add_argument('--examples',action='store_true',help='Render one or two examples per figure group.')
 parser.add_argument('--reuse-figures',action='store_true',help='Reuse this local figure manifest; refresh snippets/tables without rendering.')
 args=parser.parse_args(argv);run=args.run_dir.resolve();os.chdir(REPO)
 FIGDIR,TABS=assets.chapter_paths(args.output_root,'04_results');REFERENCE=TABS/'reference'
 FIGURES.clear();SOURCES.clear();PEAKS.clear()
 for stage in ['physical','coefficients','ivp','joint']:
  state=json.loads((run/f'{stage}_status.json').read_text())
  if state['phase']!='Finished':raise RuntimeError(f'{stage} data are not ready: {state}')
 aggregate_tables(run)
 pdf_name='results_plan_figures.pdf'
 if args.reuse_figures:
  if (TABS/'source_sha256.json').exists():SOURCES.update(json.loads((TABS/'source_sha256.json').read_text()))
  FIGURES.extend(read(FIGDIR/'figure_manifest.json'))
 else:
  with PdfPages(FIGDIR/pdf_name,metadata={'Title':'Chapter 4: physical response and structural identification'}) as BOOK:
   render_all(run,examples=args.examples)
 table_latex()
 mode=read(FIGDIR/'generation_manifest.json')['render_mode'] if args.reuse_figures else ('examples' if args.examples else 'all')
 assets.export_chapter(FIGDIR,FIGURES,pdf_name,'04_results','Chapter 4 figures and tables',mode)
 extra=[Path(__file__),REPO/'script/thesis/results_plan_data.py',REPO/'script/thesis/manuscript_assets.py']
 extra += [run/name for name in ['existing_structural_ivp_metrics.csv','new_structural_ivp_metrics.csv','joint_structural_ivp_metrics.csv']]
 extra += [REFERENCE/'tables'/name for name in ['single_structural_h_ivp_nrmse.csv','tandem_structural_h0_ivp_nrmse.csv','tandem_structural_h1_ivp_nrmse.csv']]
 for path in extra:SOURCES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
 dump(TABS/'source_sha256.json',SOURCES)
 info=dict(repository=str(REPO),output_directory=str(FIGDIR),render_mode=mode,
  training_window='overall_reduced',physical_analysis_window='sinusoidal',evaluation_window='sinusoidal_evaluation',
  figures=len(FIGURES),main_figures=sum(f['main'] for f in FIGURES),appendix_figures=sum(not f['main'] for f in FIGURES),
  manuscript_edited=False,structural_motion='Case 3 with prescribed measured lift')
 dump(FIGDIR/'generation_manifest.json',info)
 dump(run/'results_figures_status.json',dict(phase='Finished',**info))
 print('FINISHED RESULTS',json.dumps(info),flush=True)
 return info


if __name__=='__main__':main()
