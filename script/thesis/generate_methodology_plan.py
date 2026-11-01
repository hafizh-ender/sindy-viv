"""Generate local Chapter 3 figures/tables and copyable LaTeX blocks.
Run from the ver2 root with --run-dir state/results_plan_runs/<run>.
Raw data use the same ver2 loaders as fitting, including tandem repeat removal.
"""
from pathlib import Path
import os,json,argparse,hashlib,re
os.environ.setdefault('MPLBACKEND','Agg')
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
if __package__:
 from . import generate_results_plan as style
else:
 import generate_results_plan as style
from sindy_viv.case_bryan.analysis.data_windows import data_selection,load_trajectory
REPO=Path(__file__).resolve().parents[2]
FIGDIR=style.assets.OUTPUT_ROOT/'03_methodology';TABS=FIGDIR/'data'
SOURCES={};RECORDS=[];PT=style.PT;W=style.W
plt.rcParams.update({'axes.labelsize':10.5*PT,'xtick.labelsize':10.5*PT,'ytick.labelsize':10.5*PT})

def read(path):
 path=Path(path);SOURCES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text())
def key(kind,lpd,vr):return kind+('' if lpd is None else f'_LD{lpd:g}')+f'_VR{vr}'
def thin(time,values,bins=3000):
 # Preserve within-bin extrema; do not resample any data used for calculations.
 if len(time)<=2*bins:return time,values
 edges=np.linspace(0,len(time),bins+1,dtype=int);indices=[0,len(time)-1]
 for a,b in zip(edges[:-1],edges[1:]):
  if b>a:indices.extend([a+int(np.argmin(values[a:b])),a+int(np.argmax(values[a:b]))])
 indices=np.unique(indices);return time[indices],values[indices]
def save(fig,stem,caption,main=False):style.save(fig,stem,caption,main=main)
def history(run,kind,lpd,vr):
 stem=key(kind,lpd,vr);cached=run/'methodology_cache'/f'{stem}.npz'
 if cached.exists():
  SOURCES[str(cached.resolve())]=hashlib.sha256(cached.read_bytes()).hexdigest()
  with np.load(cached) as z:return z['time'].copy(),z['values'].copy(),z['names'].tolist()
 names=['h_0','C_{L0}'] if kind=='single' else ['h_0','C_{L0}','h_1','C_{L1}']
 selection=data_selection('single' if kind=='single' else 'two_tandem',vr_list=[vr],lpd_list=None if kind=='single' else [lpd])
 time,values=load_trajectory(selection,vr,lpd,names)
 np.savez_compressed(cached,time=time,values=values,names=names)
 SOURCES[str(cached.resolve())]=hashlib.sha256(cached.read_bytes()).hexdigest()
 return time,values,names

def plot_history(run,kind,lpd,vr):
 time,values,names=history(run,kind,lpd,vr);tstar=time*vr
 if kind=='single':
  fig,axes=plt.subplots(2,1,figsize=(W,.47*W),sharex=True);fig.subplots_adjust(left=.15,right=.98,bottom=.16,top=.90,hspace=.18)
  for ax,j,label in [(axes[0],0,'$h$'),(axes[1],1,'$C_L$')]:
   xx,yy=thin(tstar,values[:,j]);ax.plot(xx,yy,lw=.7,color='#111111');ax.set_ylabel(label,labelpad=2);style.axis(ax);ax.set_xlim(tstar[0],tstar[-1]);ax.xaxis.set_major_locator(plt.MaxNLocator(4))
  axes[0].set_title(rf'$V_R={vr}$',loc='right',fontsize=12*PT,pad=5);fig.supxlabel('$t^*$',fontsize=10.5*PT,y=.02)
 else:
  fig,axes=plt.subplots(2,1,figsize=(W,.60*W),sharex=True)
  fig.subplots_adjust(left=.13,right=.98,bottom=.23,top=.90,hspace=.26)
  for row,(ax,label) in enumerate(zip(axes,['$h$','$C_L$'])):
   for c,colour in [(0,'#0072B2'),(1,'#D55E00')]:
    j=2*c+row;xx,yy=thin(tstar,values[:,j],bins=2500)
    ax.plot(xx,yy,color=colour,lw=.6,alpha=.62,label=f'Cylinder {c}')
   ax.set_ylabel(label,labelpad=2);style.axis(ax);ax.set_xlim(tstar[0],tstar[-1]);ax.xaxis.set_major_locator(plt.MaxNLocator(4))
   inset=ax.inset_axes([.59,.36,.38,.60],zorder=5)
   mask=(tstar>=600)&(tstar<=650)
   for c,colour in [(0,'#0072B2'),(1,'#D55E00')]:
    j=2*c+row;xx,yy=thin(tstar[mask],values[mask,j],bins=1500)
    inset.plot(xx,yy,color=colour,lw=.9,label=f'Cylinder {c}')
   inset.set_xlim(600,650);inset.set_xticks([610,640]);inset.yaxis.set_major_locator(plt.MaxNLocator(2))
   inset.tick_params(labelsize=10.5*PT,length=2,pad=1);inset.grid(True,lw=.3,alpha=.4)
   for spine in inset.spines.values():spine.set_visible(True);spine.set_linewidth(.6)
   inset.set_title(r'$600\leq t^*\leq650$',fontsize=10.5*PT,pad=2,bbox=dict(facecolor='white',edgecolor='none',pad=.4))
   for label in inset.get_xticklabels()+inset.get_yticklabels():label.set_bbox(dict(facecolor='white',edgecolor='none',pad=.2))
  fig.suptitle(rf'$L/D={lpd:g},\quad V_R={vr}$',fontsize=12*PT,y=.98);fig.supxlabel('$t^*$',fontsize=10.5*PT,y=.13)
  handles,labels=inset.get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.015),fontsize=10.5*PT)

 stem=key(kind,lpd,vr)+'_history';main=(kind=='single' and vr==5) or (kind=='tandem' and lpd==3 and vr==5)
 caption=(f'Raw single-cylinder displacement and lift histories at $V_R={vr}$. The full record is shown on the reconstructed fluid-time grid.' if kind=='single' else f'Raw tandem histories at $L/D={lpd:g}$, $V_R={vr}$ on the reconstructed fluid-time grid. Insets show $600\leq t^*\leq650$. Cylinder 0 is upstream and cylinder 1 downstream.')
 save(fig,stem,caption,main)
 RECORDS.append(dict(kind=kind,LD=lpd,VR=vr,samples=len(time),tstar_start=float(tstar[0]),tstar_stop=float(tstar[-1]),delta_tstar=float((time[1]-time[0])*vr),main=main))
 print('HISTORY',stem,flush=True)

def plot_windows(run,kind,lpd,vr):
 time,values,names=history(run,kind,lpd,vr);tstar=time*vr
 cfgdir=REPO/'configs/fit_window'/('single' if kind=='single' else 'two_tandem')
 configs={name:read(cfgdir/f'{name}.json') for name in ['overall_reduced','sinusoidal','sinusoidal_evaluation']}
 def bounds(name):
  d=configs[name]
  if kind=='tandem':d=d[f'{lpd:.1f}']
  return d[str(vr)]['tstar0'],d[str(vr)]['tstarf']
 fig,axes=plt.subplots(2,1,figsize=(W,.57*W),sharex=True,sharey=True);fig.subplots_adjust(left=.14,right=.98,bottom=.24,top=.88,hspace=.38)
 for ax,mode,title in zip(axes,['overall_reduced','sinusoidal'],['Transient-inclusive training','Established-oscillation training']):
  a,b=bounds(mode);c,d=bounds('sinusoidal_evaluation');ax.axvspan(a,b,color='#56B4E9',alpha=.28,lw=0);ax.axvspan(c,d,color='#009E73',alpha=.35,lw=0)
  xx,yy=thin(tstar,values[:,0]);ax.plot(xx,yy,color='#111111',lw=.6);ax.set_ylabel('$h$' if kind=='single' else '$h_0$',labelpad=2);style.axis(ax);ax.set_xlim(tstar[0],tstar[-1]);ax.set_title(title,loc='left',fontsize=10.5*PT,pad=3)
  ax.xaxis.set_major_locator(plt.MaxNLocator(4))
 title=rf'Single cylinder: $V_R={vr}$' if kind=='single' else rf'Tandem: $L/D={lpd:g},\ V_R={vr}$'
 fig.suptitle(title,fontsize=12*PT,y=.98);fig.supxlabel('$t^*$',fontsize=10.5*PT,y=.13)
 fig.legend(handles=[Patch(facecolor='#56B4E9',alpha=.5,label='Training'),Patch(facecolor='#009E73',alpha=.6,label='Later evaluation')],ncol=2,loc='lower center',bbox_to_anchor=(.5,.01),fontsize=10.5*PT)
 stem=key(kind,lpd,vr)+'_windows';save(fig,stem,'Alternative fitting windows and the common later evaluation interval for '+(f'the single cylinder at $V_R={vr}$.' if kind=='single' else f'the tandem cylinders at $L/D={lpd:g}$, $V_R={vr}$.')+' The two training windows define separate fits; they are not consecutive train/validation partitions.',main=kind=='single' and vr==5)

def methodology_tables():
 tables=[]
 path=TABS/'trajectory_summary.csv'
 if path.exists():
  frame=pd.read_csv(path)
  for (kind,lpd),group in frame.fillna({'LD':-1}).groupby(['kind','LD'],sort=True):
   token=kind if lpd==-1 else kind+'_LD'+str(float(lpd)).replace('.','p')
   caption='Measured trajectory samples and fluid-time range: '+(r'single cylinder.' if lpd==-1 else rf'tandem, $L/D={lpd:g}$.')
   tables.append(dict(stem='trajectory_'+token,label='tab:03_trajectory_'+token,caption=caption,
    headers=[r'$V_R$','Samples',r'$t^*_{start}$',r'$t^*_{end}$',r'$\Delta t^*$'],
    rows=[[str(int(row.VR)),str(int(row.samples)),f'{row.tstar_start:.6g}',f'{row.tstar_stop:.6g}',f'{row.delta_tstar:.6g}'] for _,row in group.sort_values('VR').iterrows()]))
 path=TABS/'hyperparameter_sweep_values.csv'
 if path.exists():
  frame=pd.read_csv(path)
  tables.append(dict(stem='available_hyperparameter_values',label='tab:03_available_hyperparameters',
   caption='Archived parameter values; this table does not launch or define a new study.',
   headers=['Parameter','Value','Role'],rows=[[style.assets.tex_escape(row.parameter),f'{row.value:g}',style.assets.tex_escape(row.role)] for _,row in frame.iterrows()]))
 style.assets.write_tables(FIGDIR,tables,dict(headers=[r'$V_R$','Samples',r'$t^*_{start}$',r'$t^*_{end}$',r'$\Delta t^*$'],
  caption='Measured trajectory summary: replace this caption.',label='tab:03_empty_template'))
 return tables


def main(argv=None):
 global FIGDIR,TABS
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run-dir',type=Path,required=True,help='History cache directory; figures are written to the local chapter folder.')
 parser.add_argument('--output-root',type=Path,default=style.assets.OUTPUT_ROOT)
 parser.add_argument('--examples',action='store_true',help='Render representative single/tandem histories and two window examples.')
 parser.add_argument('--reuse-figures',action='store_true',help='Reuse local figures and refresh LaTeX blocks; no manuscript editing.')
 args=parser.parse_args(argv);run=args.run_dir.resolve();os.chdir(REPO)
 FIGDIR,TABS=style.assets.chapter_paths(args.output_root,'03_methodology')
 (run/'methodology_cache').mkdir(parents=True,exist_ok=True)
 style.FIGDIR=FIGDIR;style.FIGURES.clear();SOURCES.clear();RECORDS.clear()
 pdf_name='methodology_plan_figures.pdf'
 if args.reuse_figures:
  if (TABS/'source_sha256.json').exists():SOURCES.update(json.loads((TABS/'source_sha256.json').read_text()))
  style.FIGURES.extend(read(FIGDIR/'figure_manifest.json'))
 else:
  with PdfPages(FIGDIR/pdf_name) as style.BOOK:
   for vr in ([5] if args.examples else range(3,9)):plot_history(run,'single',None,vr)
   for lpd in ([3.] if args.examples else [1.5,2.,3.,3.5,4.]):
    for vr in ([5] if args.examples else range(5,15)):plot_history(run,'tandem',lpd,vr)
   for kind,lpd,vr in ([('single',None,5),('single',None,7)] if args.examples else [('single',None,5),('single',None,7),('tandem',3.,5)]):
    plot_windows(run,kind,lpd,vr)
  summary_name='trajectory_examples.csv' if args.examples else 'trajectory_summary.csv'
  pd.DataFrame(RECORDS).to_csv(TABS/summary_name,index=False)
 methodology_tables()
 mode=read(FIGDIR/'generation_manifest.json')['render_mode'] if args.reuse_figures else ('examples' if args.examples else 'all')
 style.assets.export_chapter(FIGDIR,style.FIGURES,pdf_name,'03_methodology','Chapter 3 figures and tables',mode)
 for path in [Path(__file__),REPO/'script/thesis/generate_results_plan.py',REPO/'script/thesis/manuscript_assets.py']:
  SOURCES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
 (TABS/'source_sha256.json').write_text(json.dumps(SOURCES,indent=2),encoding='utf-8')
 info=dict(phase='Finished',render_mode=mode,figures=len(style.FIGURES),output_directory=str(FIGDIR),manuscript_edited=False)
 (FIGDIR/'generation_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
 (run/'methodology_figures_status.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
 print('FINISHED METHODOLOGY',json.dumps(info),flush=True)
 return info


if __name__=='__main__':main()
