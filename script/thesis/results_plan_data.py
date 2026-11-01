"""Prepare local Chapter 4 numerical inputs, using saved ver2 fits.
No models are fitted. Raw spectra use the repository's corrected loaders and FFT
API. Missing IVP summaries are evaluated from existing overall_reduced models.
Run generate_results_plan.py for figure/document exports after these stages.
Outputs stay under img/04_results/data; no manuscript is edited.
"""
from pathlib import Path
import os,sys,json,re,hashlib,argparse
from datetime import datetime,timezone
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sindy_viv.case_bryan.analysis.data_windows import data_selection,iter_raw_windows,load_trajectory
from sindy_viv.case_bryan.analysis.frequency import compute_frequency_analysis
from sindy_viv.case_bryan.analysis.amplitude import compute_signal_amplitudes
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.evaluate_sindy.functions import _iter_sindy_predictions
from sindy_viv.utils import initiate_object
REPO=Path(__file__).resolve().parents[2]
if __package__:
 from . import manuscript_assets as assets
else:
 import manuscript_assets as assets
TABS=assets.OUTPUT_ROOT/'04_results/data';REFERENCE=TABS/'reference'
LPDS=[1.5,2.,3.,3.5,4.];SOURCES={}

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def track(path):
 p=Path(path).resolve();SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(path):return json.loads(track(path).read_text(encoding='utf-8-sig'))
def stage(run,name,message,**extra):
 value=dict(phase='Running',stage=message,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
 dump(run/f'{name}_status.json',value);print(message,flush=True)
def tag(kind,lpd,vr):return f'{kind}_'+('' if lpd is None else f'LD{lpd:g}_')+f'VR{vr}'
def base(kind,mode):
 ds='oscillating-cylinder-viv' if kind=='single' else 'oscillating-cylinder-tandem-viv'
 filt='butterworth_single_flim2' if kind=='single' else 'butterworth_two_tandem_flim2'
 root=REPO/'results'/ds/'dynamics_case_3'/f'filtering={filt}'/'finite_difference_basic'/mode
 return root/('polynomial_1' if mode=='individual' else 'state_polynomial_1/input_polynomial_2')/'stlsq_vanilla_basic'
def model(kind,mode,vr=None,lpd=None,phase='overall_reduced'):
 paths=[p for p in base(kind,mode).rglob('model.csv') if f'fit_window={phase}' in p.parts]
 if vr is not None:paths=[p for p in paths if f'VR = {vr}' in p.parts]
 if lpd is not None:paths=[p for p in paths if f'LpD = {lpd:.1f}' in p.parts or f'LpD={lpd:g}' in p.parts]
 if len(paths)!=1:raise ValueError(f'Model selection {kind,mode,vr,lpd,phase}: {paths}')
 p=track(paths[0]);m=read(p.parent/'training.json');cfg=m['configurations']['optimizer']['kwargs']
 assert cfg['threshold']==.1 and cfg['alpha']==0 and not cfg.get('normalize_columns',False)
 assert m['fit_window']['name']==phase
 return p.parent,pd.read_csv(p,index_col=0),m

def existing_summary(kind,mode,which,window):
 paths=[p for p in base(kind,mode).rglob(f'{which}_suite_summary.json') if 'fit_window=overall_reduced' in p.parts and f'evaluation_window={window}' in p.parts]
 if len(paths)!=1:raise ValueError(f'Expected one {kind,mode,which,window} summary; found {len(paths)}')
 return read(paths[0])

def prepare_physical(run):
 rows=[];harmonics=[]
 for kind in ('single','tandem'):
  config=track(REPO/f'configs/fit_window/{"single" if kind=="single" else "two_tandem"}/sinusoidal.json')
  selection=data_selection('single' if kind=='single' else 'two_tandem',fit_window_case=config)
  names=['h_0','C_{L0}'] if kind=='single' else ['h_0','C_{L0}','h_1','C_{L1}']
  for vr,lpd,tstar,values,meta in iter_raw_windows(selection,names):
   key=tag(kind,lpd,vr);cache=run/'cache'/f'{key}_spectrum.npz';jsonfile=run/'cache'/f'{key}_physical.json'
   if cache.exists() and jsonfile.exists():
    saved=json.loads(jsonfile.read_text());rows.extend(saved['rows']);harmonics.extend(saved['harmonics']);continue
   stage(run,'physical',f'FFT and amplitudes: {key}')
   fft=compute_frequency_analysis(tstar,values,names,fft_window='rectangular')
   amplitudes=compute_signal_amplitudes(tstar,values,names,method='peak_to_peak')
   cutoff=fft.frequencies<=2.5
   np.savez_compressed(cache,frequency=fft.frequencies[cutoff],amplitude=fft.amplitudes[cutoff],names=names)
   local=[];localh=[]
   for i in range(1 if kind=='single' else 2):
    hj=names.index(f'h_{i}');cj=names.index(f'C_{{L{i}}}');hp=int(fft.peak_indices[hj]);cp=int(fft.peak_indices[cj]);fh=float(fft.frequencies[hp]);fc=float(fft.frequencies[cp]);clamp=fft.amplitudes[:,cj]
    power=clamp**2;total=float(power[1:].sum());above=float(power[fft.frequencies>2].sum()/total)
    entry=dict(kind=kind,LD=lpd,VR=vr,cylinder=i,A_h=amplitudes[f'h_{i}'],A_CL=amplitudes[f'C_{{L{i}}}'],f_h=fh,f_CL=fc,f_n=1/vr,f_CL_over_fn=fc*vr,fft_CL_peak_amplitude=float(clamp[cp]),frequency_resolution=float(fft.frequencies[1]),tstar0=float(tstar[0]),tstarf=float(tstar[-1]),samples=len(tstar),spectral_power_above_2_pct=100*above)
    local.append(entry);peakset=set(find_peaks(clamp)[0])
    for n in range(1,10):
     expected=n*fh
     if expected>2.5:continue
     nearest=int(np.argmin(np.abs(fft.frequencies-expected)));lo=max(1,nearest-2);hi=min(len(clamp),nearest+3);candidates=[b for b in range(lo,hi) if b in peakset]
     b=max(candidates,key=lambda b:clamp[b]) if candidates else int(np.argmax(clamp[lo:hi]))+lo
     localh.append(dict(kind=kind,LD=lpd,VR=vr,cylinder=i,harmonic=n,expected_frequency=expected,observed_frequency=float(fft.frequencies[b]),amplitude=float(clamp[b]),relative_to_largest_CL_peak=float(clamp[b]/clamp[cp]),resolved_local_peak=bool(candidates),prominent=bool(candidates and clamp[b]>=.05*clamp[cp])))
   dump(jsonfile,dict(rows=local,harmonics=localh,window=meta));rows.extend(local);harmonics.extend(localh)
 df=pd.DataFrame(rows);hs=pd.DataFrame(harmonics)
 for kind in ('single','tandem'):
  df[df.kind==kind].to_csv(TABS/f'{kind}_physical_summary.csv',index=False)
  hs[hs.kind==kind].to_csv(TABS/f'{kind}_lift_harmonics.csv',index=False)
 # Filter the complete VR7 history before selecting its later evaluation interval.
 stage(run,'physical','Filtering comparison: single VR7')
 cfg=read(REPO/'configs/filtering/butterworth_single_flim2.json');filtering=initiate_object(**cfg)
 selection=data_selection('single',vr_list=[7]);time,data=load_trajectory(selection,7,None,['h_0','C_{L0}'])
 filtered=filtering(data,time);region=select_time_region(time,7,1000,1050);tstar=time[region]*7
 np.savez_compressed(run/'cache/single_VR7_filtering.npz',time_star=tstar,raw=data[region],filtered=filtered[region],names=['h_0','C_{L0}'])
 frows=[]
 for j,name in enumerate(['h_0','C_{L0}']):
  raw=compute_frequency_analysis(tstar,data[region,j],[name]);flt=compute_frequency_analysis(tstar,filtered[region,j],[name]);p=int(raw.peak_indices[0])
  frows.append(dict(variable=name,raw_peak_frequency=float(raw.frequencies[p]),filtered_peak_frequency=float(flt.frequencies[flt.peak_indices[0]]),raw_amplitude=float(raw.amplitudes[p,0]),filtered_amplitude_at_raw_peak=float(flt.amplitudes[p,0]),amplitude_change_pct=100*(flt.amplitudes[p,0]/raw.amplitudes[p,0]-1),signal_change_nrmse_pct=100*np.sqrt(np.mean((data[region,j]-filtered[region,j])**2))/np.std(data[region,j])))
 pd.DataFrame(frows).to_csv(TABS/'single_VR7_filtering_comparison.csv',index=False)
 prepare_lift_derivatives(run)
 dump(run/'physical_sources.json',SOURCES);dump(run/'physical_status.json',dict(phase='Finished',trajectories=56,rows=len(rows)))

def prepare_lift_derivatives(run):
 """Rebuild the missing derivative CSV through the canonical preparation API."""
 from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
 from sindy_viv.case_bryan.prepare_sindy.trajectory import prepare_trajectory
 layout=get_case_layout(7,'single');selection=data_selection('single',vr_list=[5])
 time,values=load_trajectory(selection,5,None,list(layout.measured_names))
 differentiator=initiate_object(**read(REPO/'configs/differentiator/finite_difference_basic.json'))
 filtering=initiate_object(**read(REPO/'configs/filtering/butterworth_single_flim2.json'))
 trajectory=prepare_trajectory(time,values,list(layout.measured_names),differentiator,7,'single',filtering=filtering)
 region=select_time_region(trajectory.time,5,500,525)
 lift=trajectory.inputs[:,trajectory.input_names.index('C_{L0}')]
 frame=pd.DataFrame(dict(tstar=trajectory.time[region]*5,CL=lift[region],
  CLdot_tau=trajectory.states[region,0],CLddot_tau=trajectory.derivatives[region,0]))
 frame.to_csv(TABS/'single_VR5_lift_derivatives.csv',index=False)
 dump(run/'lift_derivative_sources.json',dict(configurations=dict(
  differentiator=read(REPO/'configs/differentiator/finite_difference_basic.json'),
  filtering=read(REPO/'configs/filtering/butterworth_single_flim2.json')),
  derivative_coordinate='tau=t*/VR',full_record_processing_before_crop=True,samples=len(frame)))
 return frame


def prepare_coefficients(run):
 a={phase:[model('single','individual',vr=vr,phase=phase) for vr in range(3,9)] for phase in ['overall_reduced']}
 models=a['overall_reduced'];terms=models[0][1].index.tolist()
 individual=pd.DataFrame({vr:m[r'\ddot{h}_0'] for vr,(_,m,_) in zip(range(3,9),models)})
 individual.to_csv(TABS/'single_structural_individual_coefficients.csv')
 _,param,_=model('single','parametric_vr')
 effective=pd.DataFrame(0.,index=terms,columns=range(3,9))
 for name,value in param[r'\ddot{h}_0'].items():
  parameter,response=name.split(' ',1)
  for vr in effective.columns:
   factor={'1':1.,'V_R':float(vr),'V_R^2':float(vr)**2}[parameter]
   effective.loc[response,vr]+=factor*value
 effective.to_csv(TABS/'single_structural_parameterized_effective_coefficients.csv')
 param.to_csv(TABS/'single_structural_parameterized_tensor_coefficients.csv')
 for lpd in (1.5,4.):
  group=[model('tandem','individual',vr=vr,lpd=lpd) for vr in range(5,15)]
  for i in (0,1):pd.DataFrame({vr:m[rf'\ddot{{h}}_{i}'] for vr,(_,m,_) in zip(range(5,15),group)}).to_csv(TABS/f'tandem_LD{lpd:g}_ddoth{i}_individual_coefficients.csv')
 group=[model('tandem','parametric_vr',lpd=lpd) for lpd in LPDS]
 frames={i:pd.DataFrame({lpd:m[rf'\ddot{{h}}_{i}'] for lpd,(_,m,_) in zip(LPDS,group)}) for i in (0,1)}
 shared=[term for term in frames[0].index if any(np.any(f.loc[term].to_numpy()!=0) for f in frames.values())]
 for i,f in frames.items():f.loc[shared].to_csv(TABS/f'tandem_ddoth{i}_parameterized_coefficients.csv')
 # Report TRAINING derivative errors, not later-window reconstruction errors.
 d=existing_summary('single','parametric_vr','derivative','overall_reduced');rows=[]
 for r in d['records']:
  for name,value in r['values'].items():rows.append(dict(VR=r['VR'],target=name,training_derivative_NRMSE_pct=None if value is None else 100*value))
 pd.DataFrame(rows).to_csv(TABS/'single_parameterized_training_derivative_nrmse.csv',index=False)
 # Existing completed individual and single-parameterized IVP summaries.
 rows=[]
 for kind,mode in [('single','individual'),('single','parametric_vr'),('tandem','individual')]:
  d=existing_summary(kind,mode,'integral','sinusoidal_evaluation')
  for r in d['records']:
   for i in range(1 if kind=='single' else 2):
    v=r['values'][f'h_{i}']
    rows.append(dict(kind=kind,method=mode,LD=r.get('LpD'),VR=r['VR'],cylinder=i,complete=r['complete'],NRMSE_pct=None if not r['complete'] or v is None else 100*v))
 pd.DataFrame(rows).to_csv(run/'existing_structural_ivp_metrics.csv',index=False)
 dump(run/'coefficient_sources.json',SOURCES)
 dump(run/'coefficient_details.json',dict(single_parameterized_equation={k:float(v) for k,v in param[r'\ddot{h}_0'].items() if v!=0},tandem_shared_terms=shared))
 dump(run/'coefficients_status.json',dict(phase='Finished'))

def integrate_one(run,kind,mode,vr,lpd=None,theory=False):
 key=tag(kind,lpd,vr)+'_'+('theory' if theory else mode)+'_overall_reduced';cache=run/'cache'/f'{key}_ivp.npz';meta_path=run/'cache'/f'{key}_ivp.json'
 if cache.exists() and meta_path.exists():return json.loads(meta_path.read_text())
 if theory:
  # Theoretical coefficients are taken from the existing reference model CSV.
  filename=f'{"single" if kind=="single" else "tandem"}_theory_'+('' if lpd is None else f'LD{lpd:g}_')+f'VR{vr}.json'
  original=read(REFERENCE/'data'/filename)
  directory=Path(original['model_csv']).parent;track(directory/'model.csv');fit=None
 else:directory,_,_=model(kind,mode,vr if mode=='individual' else None,None if mode=='parametric_vr_lpd' else lpd);fit=REPO/f'configs/fit_window/{"single" if kind=="single" else "two_tandem"}/overall_reduced.json'
 start,stop=((1000,1050) if vr==7 else (600,650)) if kind=='single' else (1350,1400)
 r=next(_iter_sindy_predictions(3,REPO/'configs/optimizer/stlsq_vanilla_basic.json',evaluation_type='integral',cylinder_type='single' if kind=='single' else 'two_tandem',parameterization={'individual':'individual','parametric_vr':'vr','parametric_vr_lpd':'vr_lpd'}[mode],differentiator_case=REPO/'configs/differentiator/finite_difference_basic.json',state_feature_library_case=REPO/'configs/feature_library/polynomial_1.json',input_feature_library_case=None if mode=='individual' else REPO/'configs/feature_library/polynomial_2.json',vr_list=[vr],lpd_list=None if kind=='single' else [lpd],tstar0=start,tstarf=stop,model_dir=directory,fit_window_case=fit,filtering_case=REPO/f'configs/filtering/butterworth_{"single" if kind=="single" else "two_tandem"}_flim2.json',method='RK45',rtol=1e-6,atol=1e-9,max_state_factor=10.,allow_partial_integration=True,save_plot=False))
 names=r['variable_names'];complete=r['solver']['complete'];metrics={}
 for i in range(1 if kind=='single' else 2):
  j=names.index(f'h_{i}');metrics[f'h_{i}']=None if not complete else float(100*np.sqrt(np.mean((r['prediction'][:,j]-r['reference'][:,j])**2))/np.std(r['reference'][:,j]))
 payload=dict(kind=kind,method='theory' if theory else mode,LD=lpd,VR=vr,complete=complete,NRMSE_pct=metrics,model_csv=str(directory/'model.csv'),model_sha256=hashlib.sha256((directory/'model.csv').read_bytes()).hexdigest(),solver=r['solver'],requested_tstar=[start,stop])
 np.savez_compressed(cache,time_star=r['time']*vr,reference=r['reference'],prediction=r['prediction'],names=names)
 dump(meta_path,payload);return payload

def prepare_ivps(run):
 rows=[]
 for lpd in LPDS:
  for vr in range(5,15):
   stage(run,'ivp',f'Overall-reduced tandem VR-parameterized IVP: LD={lpd:g}, VR={vr}')
   result=integrate_one(run,'tandem','parametric_vr',vr,lpd)
   for i in (0,1):rows.append(dict(kind='tandem',method='parametric_vr',LD=lpd,VR=vr,cylinder=i,complete=result['complete'],NRMSE_pct=result['NRMSE_pct'][f'h_{i}']))
 for kind,lpd,vr in [('single',None,4),('single',None,5),('tandem',2.,12),('tandem',4.,6)]:
  for mode in ('individual','parametric_vr'):
   stage(run,'ivp',f'Plot trace: {tag(kind,lpd,vr)}, {mode}')
   integrate_one(run,kind,mode,vr,lpd)
  # Existing theory traces have exactly the desired forcing/window; preserve them as a separate baseline.
  filename=f'{kind}_theory_'+('' if lpd is None else f'LD{lpd:g}_')+f'VR{vr}'
  old=read(REFERENCE/'data'/f'{filename}.json');track(REFERENCE/'data'/f'{filename}.npz')
  # Keep the theory trace file identity in the run metadata for the renderer.
  dump(run/'cache'/f'{tag(kind,lpd,vr)}_theory_trace.json',dict(npz=str(REFERENCE/'data'/f'{filename}.npz'),metadata=old))
 pd.DataFrame(rows).to_csv(run/'new_structural_ivp_metrics.csv',index=False)
 dump(run/'ivp_sources.json',SOURCES);dump(run/'ivp_status.json',dict(phase='Finished',tandem_parameterized_cases=50))

def prepare_joint(run):
 directory,coeff,meta=model('tandem','parametric_vr_lpd')
 coeff.to_csv(TABS/'tandem_joint_tensor_coefficients.csv')
 shared=pd.read_csv(TABS/'tandem_ddoth0_parameterized_coefficients.csv',index_col=0).index.tolist()
 equations={}
 for i in (0,1):
  col=coeff[rf'\ddot{{h}}_{i}'];active=col[col!=0]
  equations[str(i)]={str(k):float(v) for k,v in active.items()}
  effective=pd.DataFrame(0.,index=shared,columns=LPDS)
  for term,value in active.items():
   if term not in effective.index:raise ValueError('New joint active term needs a full parameter evaluation: '+term)
   effective.loc[term,:]=value
  effective.to_csv(TABS/f'tandem_ddoth{i}_joint_coefficients.csv')
 rows=[]
 for lpd in LPDS:
  for vr in range(5,15):
   stage(run,'joint',f'Joint overall-reduced structural IVP: LD={lpd:g}, VR={vr}')
   result=integrate_one(run,'tandem','parametric_vr_lpd',vr,lpd)
   for i in (0,1):rows.append(dict(kind='tandem',method='parametric_vr_lpd',LD=lpd,VR=vr,cylinder=i,complete=result['complete'],NRMSE_pct=result['NRMSE_pct'][f'h_{i}']))
 pd.DataFrame(rows).to_csv(run/'joint_structural_ivp_metrics.csv',index=False)
 dump(run/'joint_coefficient_details.json',dict(model_csv=str(directory/'model.csv'),equations=equations))
 dump(run/'joint_sources.json',SOURCES);dump(run/'joint_status.json',dict(phase='Finished',trajectories=50))

def main(argv=None):
 global TABS,REFERENCE
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run-dir',type=Path,help='Preparation/cache directory; default is the local Chapter 4 data/run folder.')
 parser.add_argument('--output-root',type=Path,default=assets.OUTPUT_ROOT)
 parser.add_argument('--stage',choices=['import','physical','derivatives','coefficients','ivp','joint'],required=True)
 parser.add_argument('--input-archive',type=Path,help='Numerical-input ZIP from the manuscript cleanup; used only by stage import.')
 args=parser.parse_args(argv);os.chdir(REPO)
 _,TABS=assets.chapter_paths(args.output_root,'04_results');REFERENCE=TABS/'reference'
 run=(args.run_dir or TABS/'run').resolve();run.mkdir(parents=True,exist_ok=True);(run/'cache').mkdir(exist_ok=True)
 if args.stage=='import':
  if not args.input_archive:parser.error('--stage import requires --input-archive')
  records=assets.import_regeneration_inputs(args.input_archive,args.output_root)
  print(f'Imported {len(records)} verified numerical files into local chapter data folders.')
  return records
 for file in ['src/sindy_viv/case_bryan/input.py','src/sindy_viv/case_bryan/preprocessing.py','src/sindy_viv/case_bryan/analysis/frequency.py']:track(REPO/file)
 try:
  return {'physical':prepare_physical,'derivatives':prepare_lift_derivatives,'coefficients':prepare_coefficients,'ivp':prepare_ivps,'joint':prepare_joint}[args.stage](run)
 except Exception as error:
  dump(run/f'{args.stage}_status.json',dict(phase='Failed',error=str(error)));raise


if __name__=='__main__':main()
