"""Full CFD-data recomputation: single locally, three tandem groups on Slurm.

Dry-run by default. No archive import, figure generation or numerical gate.
See script/FULL_RECOMPUTE.md. Each successful new fit gets two derivative
evaluations and one independently supervised complete-motion/lift IVP.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import uuid
import warnings

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
CONFIGS=REPO/'configs/experiments/full_recompute'
GROUPS=('single','tandem_lpd_1p5','tandem_lpd_3p0','tandem_lpd_4p0')
ALPHAS=(1e-6,1e-4,1e-3,1e-2,.1,.5,5.)
THRESHOLDS=(.001,.005,.01,.05,.1,.5,1.)
POLICY='new_ridge_two_parents_alpha_ties_v1'
CAVEAT=('Unnormalized polynomial matrices can be rank deficient. GELSD uses '
 'cond=max(N,P)*float64_eps. Expanded libraries can increase finite-precision '
 'OLS SSE through singular-value truncation. Flag this; do not stop or rescale. '
 'Existing full-record, noncausal filtering precedes differentiation/cropping '
 'and can share information across training/validation boundaries.')
COLUMNS='''run_id group model_family vr lpd poly_degree ridge_alpha threshold
parent_run_id parent_selection_role coef_l2_norm_cl0 coef_l2_norm_cl1
active_terms_cl0 active_terms_cl1 train_cldot0_nrmse val_cldot0_nrmse
train_cldot1_nrmse val_cldot1_nrmse ivp_completed ivp_status ivp_failure_reason
requested_t_end actual_t_end ivp_h0_nrmse ivp_h1_nrmse ivp_cl0_nrmse ivp_cl1_nrmse
fit_seconds ivp_seconds job_state artifact_path experiment_id fit_status
requested_t_start train_cldot0_sse train_cldot1_sse fit_samples val_samples
feature_count matrix_rank rank_cutoff singular_min singular_max
feature_norm_min feature_norm_median feature_norm_max condition_estimate
coef_l2_norm active_terms_total normalization_method coef_norm_coordinates
val_cldot_mean_nrmse ivp_mean_nrmse ivp_hdot0_nrmse ivp_hdot1_nrmse
partial_h0_nrmse partial_h1_nrmse partial_cl0_nrmse partial_cl1_nrmse
parent_ivp_completed parent_ivp_status parent_ivp_mean_nrmse
parent_val_cldot_mean_nrmse selection_policy_version selection_manifest
transition derivative_eval_seconds diagnostic_seconds data_id code_hash
python_version pysindy_version scipy_version numpy_version sklearn_version
git_revision optimizer_class optimizer_kwargs error warnings_json
ols_nonmonotonic_sse numerical_caveat attempt_finished'''.split()


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def mean(values,all_required=False):
    values=list(values)
    if all_required and not all(finite(x) for x in values):return None
    values=[x for x in values if finite(x)]
    return sum(values)/len(values) if values else None
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,Path):return str(value)
    if hasattr(value,'tolist'):return clean(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
def identity(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    content=json.dumps(clean(value),indent=2,allow_nan=False)+'\n'
    for retry in range(21):
        try:
            temporary.write_text(content,encoding='utf-8');temporary.replace(path);return
        except PermissionError:
            if retry==20:raise
            time.sleep(.2)
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            acquire=lambda:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            release=lambda:msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            acquire=lambda:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            release=lambda:fcntl.flock(stream,fcntl.LOCK_UN)
        try:acquire()
        except OSError as error:raise RuntimeError('Another process owns '+str(path)) from error
        try:yield
        finally:stream.seek(0);release()


def code_hash():
    files=[Path(__file__).resolve(),*sorted((REPO/'src').rglob('*.py'))]
    # Git's Windows/Linux newline conversion must not change the scientific ID.
    return identity({p.relative_to(REPO).as_posix():hashlib.sha256(p.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest() for p in files})
def environment():
    result={key:importlib.metadata.version(package) for key,package in
        [('pysindy_version','pysindy'),('scipy_version','scipy'),('numpy_version','numpy'),('sklearn_version','scikit-learn')]}
    try:commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True,check=False).stdout.strip() or None
    except OSError:commit=None
    return dict(result,python_version=platform.python_version(),platform=platform.platform(),git_revision=commit,
        node=platform.node(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        blas_threads={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
def specification(group):
    spec=read(CONFIGS/(group+'.json'))
    expected={'single':(None,[3,4,5,6,7,8]),'tandem_lpd_1p5':(1.5,[5,10,14]),
        'tandem_lpd_3p0':(3.,[5,10,14]),'tandem_lpd_4p0':(4.,[5,10,14])}[group]
    if (spec['lpd'],spec['vr'])!=expected:raise ValueError('Fixed group definition changed.')
    if (spec['alphas']!=list(ALPHAS) or spec['thresholds']!=list(THRESHOLDS) or
        spec['polynomial_degrees']!=[1,2,3,4,5] or spec['regularized_degrees']!=[3,5] or
        spec['normalize_columns'] or spec['unbias'] or not spec['include_bias'] or
        spec['motion_source']!='theoretical' or spec['dynamics_case']!=6):
        raise ValueError('The fixed full-recomputation convention was changed; use an explicit new configuration version.')
    return spec
def budget(n):return dict(cases=n,ols=5*n,ridge_only=14*n,stlsq_only=14*n,fixed=33*n,conditional_max=28*n,total_max=61*n)
def make_plan():
    groups={group:dict(budget=budget(len(specification(group)['vr'])),config=specification(group)) for group in GROUPS}
    return dict(experiment_id=specification('single')['experiment_id'],created_utc=now(),
        source='full new CFD-data fits; no archived fitted models, metrics or parent lists',
        groups=groups,budget=budget(15),stage_order=['A:ols','B:ridge_only','D:stlsq_only','C:ridge_stlsq'],
        conditional_selection=dict(policy=POLICY,max_distinct_parents_per_case_degree=2,failed_ivps_eligible=True,
          threshold_fits_per_parent=7,parents_from='new Stage-B checkpoints only'),
        numerical_caveat=CAVEAT,execution_split='six single cases local; nine tandem cases in three Slurm allocations')


def configure_paths(cylinder,data_root):
    from sindy_viv.case_bryan import constants, input as loaders
    name='SINGLE_CYLINDER_DATA_DIR' if cylinder=='single' else 'TWO_TANDEM_CYLINDERS_DATA_DIR'
    setattr(constants,name,str(data_root));setattr(loaders,name,str(data_root))
def resolve_case(spec,vr,args,env,version):
    from sindy_viv.case_bryan import constants as c
    from sindy_viv.case_bryan.analysis.data_windows import data_selection
    single=spec['cylinder_type']=='single';lpd=spec['lpd']
    default=c.SINGLE_CYLINDER_DATA_DIR if single else c.TWO_TANDEM_CYLINDERS_DATA_DIR
    data_root=(args.single_data_dir if single else args.tandem_data_dir) or REPO/default
    data_root=Path(data_root).resolve()
    folder=data_root/(f'Re = {c.SINGLE_CYLINDER_RE}/Mstar = {c.SINGLE_CYLINDER_MSTAR}/VR = {vr}' if single else f'LpD = {lpd}/VR = {vr}')
    filenames=c.SINGLE_CYLINDER_STATE_VARIABLE_FILENAMES if single else c.TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES
    files={filename:dict(path=str(folder/filename),bytes=(folder/filename).stat().st_size,sha256=sha(folder/filename)) for filename in filenames.values()}
    snapshots={key:read(REPO/spec[key]) for key in ['training_window','validation_window','filtering','differentiator']}
    bounds={key:list(data_selection(spec['cylinder_type'],[vr],None if single else [lpd],fit_window_case=REPO/spec[key]).bounds[(lpd,vr)]) for key in ['training_window','validation_window']}
    return dict(group=spec['group'],vr=vr,lpd=lpd,cylinder_type=spec['cylinder_type'],
        case_id=('single' if single else 'lpd_'+str(lpd).replace('.','p'))+f'_vr_{vr}',
        data_root=str(data_root),data_files=files,data_id=identity({name:info['sha256'] for name,info in files.items()}),
        snapshots=snapshots,bounds=bounds,code_hash=version,environment=env,
        structural_mstar=float(c.SINGLE_CYLINDER_MSTAR_TRUE if single else c.TWO_TANDEM_CYLINDERS_MSTAR_TRUE))
def descriptor(family,alpha,threshold,spec):
    name={'ols':'SVDLeastSquares','ridge_only':'RidgeOnly','ridge_stlsq':'SequentialSVD','stlsq_only':'SequentialSVD'}[family]
    kwargs=dict(normalize_columns=False,unbias=False)
    if family!='ols':kwargs['alpha']=float(alpha)
    if family in ('ridge_stlsq','stlsq_only'):kwargs.update(threshold=float(threshold),max_iter=spec['sequential_max_iter'])
    return dict(module='sindy_viv.optimizers',class_name=name,kwargs=kwargs)
def job(case,spec,family,degree,alpha=0.,threshold=0.,lineage=None):
    optimizer=descriptor(family,alpha,threshold,spec)
    key=dict(experiment_id=spec['experiment_id'],spec=spec,data_id=case['data_id'],vr=case['vr'],lpd=case['lpd'],
        snapshots=case['snapshots'],bounds=case['bounds'],code_hash=case['code_hash'],structural_mstar=case['structural_mstar'],
        versions={k:v for k,v in case['environment'].items() if k.endswith('_version')},
        model_family=family,poly_degree=degree,ridge_alpha=float(alpha),threshold=float(threshold),
        parent_run_id=(lineage or {}).get('parent_run_id'),optimizer=optimizer)
    return dict(case,experiment_id=spec['experiment_id'],spec=spec,model_family=family,poly_degree=degree,
        ridge_alpha=float(alpha),threshold=float(threshold),optimizer=optimizer,lineage=lineage or {},
        scientific_key=identity(key),run_id=f'{family}_p{degree}_'+identity(key)[:24])
def fixed_jobs(case,spec,stage):
    if stage=='ols':return [job(case,spec,'ols',p) for p in spec['polynomial_degrees']]
    if stage=='ridge':return [job(case,spec,'ridge_only',p,a) for p in spec['regularized_degrees'] for a in spec['alphas']]
    if stage=='stlsq_only':return [job(case,spec,'stlsq_only',p,0,t) for p in spec['regularized_degrees'] for t in spec['thresholds']]
    raise ValueError(stage)
def directory(output,config):return Path(output)/config['group']/config['case_id']/config['run_id']


def settings(folder,config):
    filenames=dict(differentiator=Path(config['spec']['differentiator']).name,filtering=Path(config['spec']['filtering']).name,
        training_window='sinusoidal.json',validation_window='sinusoidal_evaluation.json',optimizer='optimizer.json',library=f'polynomial_{config["poly_degree"]}.json')
    values=dict(config['snapshots'],optimizer=config['optimizer'],library=dict(module='pysindy.feature_library',class_name='PolynomialLibrary',
        kwargs=dict(degree=config['poly_degree'],include_bias=True)))
    paths={name:folder/'settings'/filename for name,filename in filenames.items()}
    for name,path in paths.items():
        if path.exists() and read(path)!=values[name]:raise ValueError('Frozen settings changed: '+str(path))
        if not path.exists():write(path,values[name])
    return paths
def options(config,paths):
    return dict(dynamics_case=6,cylinder_type=config['cylinder_type'],parameterization='individual',vr_list=[config['vr']],
        lpd_list=None if config['lpd'] is None else [config['lpd']],differentiator_case=paths['differentiator'],
        filtering_case=paths['filtering'],state_feature_library_case=paths['library'],optimizer_case=paths['optimizer'])
def prepared(config):
    from sindy_viv.case_bryan.analysis.data_windows import data_selection,load_trajectory
    from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
    from sindy_viv.case_bryan.prepare_sindy.trajectory import prepare_trajectory
    from sindy_viv.case_bryan.model_storage.fit_windows import crop_training_trajectory
    from sindy_viv.utils import initiate_object
    selection=data_selection(config['cylinder_type'],[config['vr']],None if config['lpd'] is None else [config['lpd']])
    layout=get_case_layout(6,config['cylinder_type'])
    t,values=load_trajectory(selection,config['vr'],config['lpd'],list(layout.measured_names))
    result=prepare_trajectory(t,values,list(layout.measured_names),initiate_object(**config['snapshots']['differentiator']),
        6,config['cylinder_type'],filtering=initiate_object(**config['snapshots']['filtering']))
    return crop_training_trajectory(result.time,result.states,result.derivatives,result.inputs,config['lpd'],config['vr'],tuple(config['bounds']['training_window']))
def diagnostics(config,optimizer=None):
    import numpy as np
    from scipy import linalg
    from sindy_viv.feature_library import prepare_library
    token=identity([config['data_id'],config['snapshots'],config['code_hash'],config['poly_degree']])[:20]
    path=Path(config['output_root'])/config['group']/'diagnostics'/config['case_id']/(token+'.json')
    if path.exists():return read(path)
    start=time.perf_counter();t,x,y,u,meta=prepared(config);response=np.column_stack((x,u))
    library=prepare_library(dict(module='pysindy.feature_library',class_name='PolynomialLibrary',kwargs=dict(degree=config['poly_degree'],include_bias=True))).fit(response[:1])
    norms=np.zeros(len(library.get_feature_names()))
    for first in range(0,len(x),10000):
        block=np.asarray(library.transform(response[first:first+10000]));norms+=np.sum(block*block,axis=0)
    norms=np.sqrt(norms);cutoff=max(len(x),len(norms))*np.finfo(float).eps
    if config['model_family']=='ols' and optimizer is not None:
        singular=optimizer.singular_;rank=optimizer.rank_
    else:
        matrix=np.asarray(library.transform(response));singular=linalg.svdvals(matrix);del matrix
        rank=int(np.sum(singular>cutoff*singular[0]))
    result=dict(feature_count=len(norms),fit_samples=len(x),matrix_rank=rank,rank_cutoff=cutoff,
        singular_values=singular,singular_min=float(singular[-1]),singular_max=float(singular[0]),
        condition_estimate=float(singular[0]/singular[-1]) if rank==len(norms) and singular[-1]>0 else None,
        feature_norm_min=float(norms.min()),feature_norm_median=float(np.median(norms)),feature_norm_max=float(norms.max()),
        target_energy=np.sum(y*y,axis=0),training_sample_metadata=meta,diagnostic_seconds=time.perf_counter()-start)
    write(path,result);return clean(result)


def fit_worker(folder,config):
    import numpy as np
    import pandas as pd
    from sindy_viv.case_bryan.compute_sindy.functions import compute_sindy
    from sindy_viv.case_bryan.evaluate_sindy.functions import evaluate_sindy
    paths=settings(folder,config);opts=options(config,paths);model=None
    result=dict(fit_status='failed',derivative_status='not_run');receipt=folder/'fit_receipt.json'
    try:
        if receipt.exists():
            saved=read(receipt)
            if sha(folder/'model.csv')!=saved['model_sha256'] or sha(folder/'training.json')!=saved['training_sha256']:
                raise ValueError('Saved fit artifacts changed.')
            result.update(saved)
        else:
            start=time.perf_counter();staging=folder/'.fit'
            fitted=compute_sindy(**opts,fit_window_case=paths['training_window'],results_root=staging,save_pkl=False)
            model=next(iter(fitted.values()));sources=list(staging.rglob('model.csv'))
            if len(sources)!=1:raise ValueError('Expected exactly one new individual model.')
            for name in ('model.csv','training.json'):sources[0].with_name(name).replace(folder/name)
            saved=dict(fit_seconds=time.perf_counter()-start,model_sha256=sha(folder/'model.csv'),training_sha256=sha(folder/'training.json'))
            write(receipt,saved);result.update(saved)
            for parent,_,files in os.walk(staging,topdown=False):
                if not files:Path(parent).rmdir()
        weights=pd.read_csv(folder/'model.csv',index_col=0).to_numpy(float)
        if not np.all(np.isfinite(weights)):raise ValueError('Nonfinite coefficients.')
        result.update(fit_status='completed',coef_l2_norm=float(np.linalg.norm(weights)),active_terms_total=int(np.sum(np.abs(weights)>1e-12)))
        for i in (0,1):
            values=weights[:,i] if i<weights.shape[1] else None
            result[f'coef_l2_norm_cl{i}']=float(np.linalg.norm(values)) if values is not None else None
            result[f'active_terms_cl{i}']=int(np.sum(np.abs(values)>1e-12)) if values is not None else None
        start=time.perf_counter()
        for window,prefix in [('training_window','train'),('validation_window','val')]:
            a,b=config['bounds'][window]
            record=evaluate_sindy(**opts,model_dir=folder,evaluation_type='derivative',tstar0=a,tstarf=b,
                save_plot=False,return_data=True,report_directory=folder/'evaluation/derivative'/prefix)[config['vr'] if config['lpd'] is None else (config['lpd'],config['vr'])]
            for i in (0,1):
                result[f'{prefix}_cldot{i}_nrmse']=record['metrics'].get(rf'\dot{{C}}_{{L{i}}}',{}).get('NRMSE')
                if prefix=='train':result[f'train_cldot{i}_sse']=float(np.sum((record['prediction'][:,i]-record['reference'][:,i])**2)) if i<weights.shape[1] else None
            result['fit_samples' if prefix=='train' else 'val_samples']=record['sample_count']
            write(folder/'evaluation'/f'derivative_{prefix}.json',{k:record[k] for k in record if k not in ('time','reference','prediction')})
            del record
        result.update(derivative_status='completed',derivative_eval_seconds=time.perf_counter()-start,
            val_cldot_mean_nrmse=mean([result.get('val_cldot0_nrmse'),result.get('val_cldot1_nrmse')]))
        result.update(diagnostics(config,None if model is None else model.optimizer))
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}';traceback.print_exc()
    return clean(result)


def ivp_fields(record,cylinder):
    solver=record.get('solver') or {};requested=record.get('requested_tstar_region');actual=record.get('tstar_region');count=record.get('sample_count',0)
    endpoint=actual[-1] if actual and count>1 and finite(actual[-1]) else None
    completed=bool(record.get('integration_complete') is True and solver.get('complete') is True and solver.get('status')==0
        and solver.get('completed_samples')==solver.get('requested_samples') and count>1)
    if completed and actual and requested:
        spacing=(actual[-1]-actual[0])/(count-1)
        completed=abs(actual[-1]-requested[-1])<=spacing+1e-9 and abs(actual[0]-requested[0])<=spacing+1e-9
    reason=solver.get('termination_reason','solver_failure');message=solver.get('message',str(reason))
    status='completed' if completed else 'state_limit' if reason=='state_limit' else 'nonfinite' if 'nonfinite' in message.lower() else 'solver_failure'
    result=dict(ivp_completed=completed,ivp_status=status,ivp_failure_reason=None if completed else message,actual_t_end=endpoint,solver=solver)
    for i in (0,1):
        for token,name in [('h',f'h_{i}'),('cl',f'C_{{L{i}}}'),('hdot',rf'\dot{{h}}_{i}')]:
            value=record.get('metrics',{}).get(name,{}).get('NRMSE') if i==0 or cylinder=='two_tandem' else None
            result[f'ivp_{token}{i}_nrmse']=value if completed else None
            if token!='hdot':result[f'partial_{token}{i}_nrmse']=value if not completed else None
    result['ivp_mean_nrmse']=mean([result[f'ivp_{token}{i}_nrmse'] for i in range(1 if cylinder=='single' else 2) for token in ('h','cl')],all_required=True) if completed else None
    return clean(result)
def ivp_worker(folder,config):
    from sindy_viv.case_bryan.evaluate_sindy.functions import evaluate_sindy
    paths=settings(folder,config);opts=options(config,paths);solver=config['spec']['solver'];a,b=config['bounds']['validation_window']
    start=time.perf_counter()
    record=evaluate_sindy(**opts,model_dir=folder,evaluation_type='integral',coupled_motion=True,case_3_model=None,
        structural_mstar=config['structural_mstar'],tstar0=a,tstarf=b,method=solver['method'],rtol=solver['rtol'],atol=solver['atol'],
        max_state_factor=solver['max_state_factor'],save_plot=False,return_data=False,report_directory=folder/'evaluation/ivp')[config['vr'] if config['lpd'] is None else (config['lpd'],config['vr'])]
    return dict(ivp_fields(record,config['cylinder_type']),ivp_seconds=time.perf_counter()-start,ivp_metrics_path=record['metrics_path'])
def worker(request_path):
    request=read(request_path);config=request['config'];folder=Path(request['directory'])
    if code_hash()!=config['code_hash']:raise RuntimeError('Source changed after the run was frozen.')
    configure_paths(config['cylinder_type'],config['data_root'])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:result={'fit':fit_worker,'ivp':ivp_worker}[request['phase']](folder,config)
        except Exception as error:
            traceback.print_exc();result=dict(error=f'{type(error).__name__}: {error}')
        result['warnings_json']=json.dumps([str(item.message) for item in captured])
    write(request['response'],result)
    return 0
def run_phase(folder,config,phase,timeout=None):
    response=folder/(phase+'_phase.json');request=folder/(phase+'_request.json')
    if response.exists():return read(response)
    write(request,dict(config=config,directory=str(folder),phase=phase,response=str(response)))
    threads=os.environ.get('SLURM_CPUS_PER_TASK','1')
    env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,MKL_NUM_THREADS=threads)
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',str(request)]
    start=time.perf_counter()
    with (folder/'stdout.log').open('a',encoding='utf-8') as stdout,(folder/'stderr.log').open('a',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=REPO,env=env,stdout=stdout,stderr=stderr,shell=False)
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()
            result=dict(ivp_completed=False,ivp_status='timeout',actual_t_end=None,ivp_seconds=time.perf_counter()-start,
                ivp_failure_reason='Supervisor wall-time limit including startup; no verified accepted endpoint available.')
            write(response,result);return result
        except BaseException:process.terminate();process.wait();raise
    if not response.exists():
        result=dict(error=f'{phase} worker exited {process.returncode} without a response.')
        if phase=='ivp':result.update(ivp_completed=False,ivp_status='missing_output',actual_t_end=None)
        write(response,result)
    return read(response)


def attempt(output,config,resume):
    folder=directory(output,config);path=folder/'record.json'
    if path.exists():
        if not resume:raise ValueError('Existing new-run checkpoint; use --resume: '+str(folder))
        old=read(folder/'config.json')
        if old['scientific_key']!=config['scientific_key']:raise ValueError('Immutable configuration identity changed.')
        row=read(path)
        for filename,key in [('model.csv','model_sha256'),('training.json','training_sha256')]:
            if row.get(key) and sha(folder/filename)!=row[key]:raise ValueError('Saved artifact changed: '+str(folder/filename))
        if row.get('attempt_finished'):return row,False
    else:
        folder.mkdir(parents=True,exist_ok=True);write(folder/'config.json',config)
        row={key:None for key in COLUMNS}
        row.update({k:config[k] for k in ['run_id','experiment_id','group','vr','lpd','poly_degree','ridge_alpha','threshold','model_family','data_id','code_hash']})
        row.update(config['lineage'])
        row.update({k:v for k,v in config['environment'].items() if k in COLUMNS},
            fit_status='pending',ivp_completed=False,ivp_status='not_run',job_state='fitting',artifact_path=str(folder),
            requested_t_start=config['bounds']['validation_window'][0],requested_t_end=config['bounds']['validation_window'][1],
            normalization_method='none',coef_norm_coordinates='original polynomial basis',numerical_caveat=CAVEAT,
            optimizer_class=config['optimizer']['class_name'],optimizer_kwargs=json.dumps(config['optimizer']['kwargs'],sort_keys=True),attempt_finished=False)
        write(path,row)
    write(folder/'status.json',dict(phase='Fitting',updated_utc=now()))
    try:
        row.update(run_phase(folder,config,'fit'));write(path,row)
        if row.get('fit_status')=='completed':
            write(folder/'status.json',dict(phase='Integrating',updated_utc=now()))
            result=run_phase(folder,config,'ivp',config['spec']['solver']['wall_seconds'])
            if result.get('error') and 'ivp_status' not in result:
                result.update(ivp_completed=False,ivp_status='solver_failure',actual_t_end=None,ivp_failure_reason=result['error'])
            result['warnings_json']=json.dumps(json.loads(row.get('warnings_json') or '[]')+json.loads(result.get('warnings_json') or '[]'))
            row.update(result)
            row['job_state']='finished' if row.get('derivative_status')=='completed' else 'derivative_evaluation_failed'
        else:row.update(fit_status='failed',job_state='fit_failed')
        if not row.get('ivp_completed'):
            for key in COLUMNS:
                if key.startswith('ivp_') and key.endswith('_nrmse'):row[key]=None
        if config['lineage']:
            row['transition']=('completed' if row.get('parent_ivp_completed') else 'failed')+' -> '+('completed' if row.get('ivp_completed') else 'failed')
        row.update(attempt_finished=True,finished_utc=now())
        write(folder/'evaluation/ivp.json',{key:row.get(key) for key in row if key.startswith(('ivp_','partial_','requested_','actual_'))})
        write(path,row);write(folder/'status.json',dict(phase=row['job_state'],ivp_status=row['ivp_status'],updated_utc=now(),error=row.get('error')))
        return row,True
    except BaseException as error:
        write(folder/'status.json',dict(phase='Interrupted',error=str(error),updated_utc=now()));raise


def select_parents(rows):
    valid=[r for r in rows if r.get('fit_status')=='completed']
    derivable=sorted([r for r in valid if finite(r.get('val_cldot_mean_nrmse'))],key=lambda r:(r['val_cldot_mean_nrmse'],r['ridge_alpha'],r['run_id']))
    completed=[r for r in valid if r.get('ivp_completed') and finite(r.get('ivp_mean_nrmse'))]
    chosen=[];has_completion=any(r.get('ivp_completed') for r in valid)
    if has_completion:
        first=min(completed,key=lambda r:(r['ivp_mean_nrmse'],r['ridge_alpha'],r['run_id'])) if completed else None
        if first:chosen.append(('A',first))
        second=next((r for r in derivable if first is None or r['ridge_alpha']!=first['ridge_alpha']),None)
    else:
        first=derivable[0] if derivable else None
        if first:chosen.append(('A',first))
        others=[r for r in valid if first is not None and r['ridge_alpha']!=first['ridge_alpha'] and not r.get('ivp_completed')]
        verified=[r for r in others if finite(r.get('actual_t_end'))]
        second=min(verified,key=lambda r:(-r['actual_t_end'],r['ridge_alpha'],r['run_id'])) if verified else next((r for r in derivable if first and r['ridge_alpha']!=first['ridge_alpha']),None)
    if second:chosen.append(('B',second))
    selected={r['run_id']:role for role,r in chosen}
    candidates=[]
    for row in sorted(rows,key=lambda r:(r['ridge_alpha'],r['run_id'])):
        eligible=(finite(row.get('val_cldot_mean_nrmse')) or row.get('ivp_completed') and finite(row.get('ivp_mean_nrmse')) or
            first is not None and not has_completion and finite(row.get('actual_t_end')))
        reason='selected_'+selected[row['run_id']] if row['run_id'] in selected else 'fit_failed' if row.get('fit_status')!='completed' else 'not_selected' if eligible else 'unrankable'
        candidates.append({key:row.get(key) for key in ['run_id','ridge_alpha','fit_status','ivp_completed','ivp_status','val_cldot_mean_nrmse','ivp_mean_nrmse','actual_t_end']}|dict(reason=reason))
    return dict(policy=POLICY,candidates=candidates,selected=[dict(role=role,run_id=row['run_id'],ridge_alpha=row['ridge_alpha']) for role,row in chosen],reason=None if chosen else 'no_eligible_parent')
def selection(output,case,spec,degree):
    expected=[c for c in fixed_jobs(case,spec,'ridge') if c['poly_degree']==degree];rows=[]
    for config in expected:
        path=directory(output,config)/'record.json'
        if not path.exists() or not read(path).get('attempt_finished'):return None,[],dict(reason='stage_b_incomplete')
        row=read(path)
        if row.get('fit_status')=='completed' and sha(path.parent/'model.csv')!=row.get('model_sha256'):raise ValueError('Stage-B coefficients changed.')
        if row['experiment_id']!=spec['experiment_id']:raise ValueError('Cannot select archived parents.')
        rows.append(row)
    signature=identity(rows);context=identity([case['data_id'],case['code_hash'],spec])[:20]
    path=Path(output)/spec['group']/'selection'/case['case_id']/context/f'p{degree}.json'
    if path.exists():
        manifest=read(path)
        if manifest['input_signature']!=signature:raise ValueError('Frozen parent inputs changed. Start an explicit new experiment; no silent reselection.')
    else:
        manifest=dict(select_parents(rows),input_signature=signature,case_id=case['case_id'],degree=degree,
            experiment_id=spec['experiment_id'],created_utc=now(),source_records=[str(directory(output,c)/'record.json') for c in expected])
        write(path,manifest)
    return path,rows,manifest
def children(case,spec,path,rows,manifest):
    lookup={row['run_id']:row for row in rows};jobs=[]
    for parent in manifest['selected']:
        row=lookup[parent['run_id']]
        lineage=dict(parent_run_id=row['run_id'],parent_selection_role=parent['role'],parent_ivp_completed=row['ivp_completed'],
            parent_ivp_status=row['ivp_status'],parent_ivp_mean_nrmse=row.get('ivp_mean_nrmse'),
            parent_val_cldot_mean_nrmse=row.get('val_cldot_mean_nrmse'),selection_policy_version=POLICY,selection_manifest=str(path))
        jobs.extend(job(case,spec,'ridge_stlsq',manifest['degree'],row['ridge_alpha'],threshold,lineage) for threshold in spec['thresholds'])
    return jobs


def records(output,group,experiment_id):
    rows=[read(path) for path in (Path(output)/group).glob('*/*/record.json')]
    return sorted([row for row in rows if row.get('experiment_id')==experiment_id],key=lambda r:r['run_id'])
def refresh(output,spec,cases):
    folder=Path(output)/spec['group'];rows=records(output,spec['group'],spec['experiment_id'])
    expected=[config for case in cases for stage in ('ols','ridge','stlsq_only') for config in fixed_jobs(case,spec,stage)]
    slots=[];conditional=[];pending_slots=0
    for case in cases:
        for degree in spec['regularized_degrees']:
            context=identity([case['data_id'],case['code_hash'],spec])[:20]
            path=folder/'selection'/case['case_id']/context/f'p{degree}.json'
            if path.exists():
                manifest=read(path);parents=[read(Path(p)) for p in manifest['source_records']]
                conditional.extend(children(case,spec,path,parents,manifest))
                slots.append(dict(case_id=case['case_id'],degree=degree,path=str(path),selected=manifest['selected'],reason=manifest['reason']))
            else:pending_slots+=1
    write(folder/'parent_selection_manifest.json',dict(experiment_id=spec['experiment_id'],policy=POLICY,slots=slots,unselected_slots=pending_slots))
    done={row['run_id'] for row in rows if row.get('attempt_finished')}
    expected_ids={c['run_id'] for c in expected};child_ids={c['run_id'] for c in conditional}
    for case in cases:
        previous=None;caveats=[]
        for config in fixed_jobs(case,spec,'ols'):
            row=next((r for r in rows if r['run_id']==config['run_id']),None)
            if row is None:continue
            current=[row.get('train_cldot0_sse')]+([row.get('train_cldot1_sse')] if case['lpd'] is not None else [])
            tolerance=[1e-10*max(energy,1.)+1e-7*max(sse,1e-30) if finite(sse) else None for energy,sse in zip(row.get('target_energy',[1.]*len(current)),current)]
            nonmonotonic=bool(previous and any(finite(a) and finite(b) and b>a+t for a,b,t in zip(previous,current,tolerance) if finite(t)))
            row['ols_nonmonotonic_sse']=nonmonotonic
            if nonmonotonic:caveats.append(dict(run_id=row['run_id'],degree=config['poly_degree'],previous_sse=previous,sse=current,tolerance=tolerance))
            if all(finite(v) for v in current):previous=current
        if caveats:write(folder/case['case_id']/'ols_numerical_caveats.json',dict(caveat=CAVEAT,records=caveats))
    write_csv(folder/'summary.csv',rows)
    progress=dict(experiment_id=spec['experiment_id'],planned=budget(len(cases)),attempted=len(rows),terminal=len(done),
        fixed_terminal=len(done&expected_ids),conditional_terminal=len(done&child_ids),conditional_planned=len(child_ids),
        remaining_fixed=len(expected_ids-done),remaining_selected_threshold=len(child_ids-done),pending_selection_slots=pending_slots,
        remaining_upper_bound=len(expected_ids-done)+len(child_ids-done)+14*pending_slots,
        ivp_completed=sum(bool(r.get('ivp_completed')) for r in rows),fit_failures=sum(r.get('fit_status')=='failed' for r in rows),
        all_stages_attempted=not(expected_ids-done or child_ids-done or pending_slots),updated_utc=now())
    write(folder/'progress.json',progress);return progress,rows
def report(output):
    output=Path(output)
    try:
        with lock(output/'.report.lock'):
            lines=['# Full recomputation progress','',f'Updated: {now()}','',
                'All lift fits are new CFD-data fits. Hard-coded theoretical motion, normalization off, explicit GELSD cutoff; no archive recovery.','',
                '| Group | Terminal attempts | Fixed completed | IVPs completed | Remaining upper bound | All stages attempted |',
                '| --- | ---: | ---: | ---: | ---: | --- |']
            merged=[];visible=[]
            for group in GROUPS:
                path=output/group/'progress.json'
                if not path.exists():
                    lines.append(f'| {group} | 0 | 0 | 0 | {budget(6 if group=="single" else 3)["total_max"]} | Not computed/submitted |');continue
                progress=read(path);rows=records(output,group,progress['experiment_id']);merged.extend(rows);visible.append(group)
                lines.append(f'| {group} | {progress["terminal"]} | {progress["fixed_terminal"]} | {progress["ivp_completed"]} | {progress["remaining_upper_bound"]} | {progress["all_stages_attempted"]} |')
            lines+=['','## Derivative reconstruction (best validation score per target)','',
                '| Group / case | Target | Training NRMSE | Validation NRMSE | Run | Active target terms |',
                '| --- | --- | ---: | ---: | --- | ---: |']
            for group in visible:
                group_rows=[r for r in merged if r['group']==group]
                for vr in sorted({r['vr'] for r in group_rows}):
                    for target in range(1 if group=='single' else 2):
                        field=f'val_cldot{target}_nrmse'
                        candidates=[r for r in group_rows if r['vr']==vr and finite(r.get(field))]
                        if not candidates:continue
                        best=min(candidates,key=lambda r:(r[field],r['run_id']))
                        training=best.get(f'train_cldot{target}_nrmse')
                        train_text=f'{training:.6g}' if finite(training) else 'null'
                        lines.append(f'| {group} / VR={vr} | CLdot{target} | {train_text} | {best[field]:.6g} | {best["run_id"]} | {best.get(f"active_terms_cl{target}")} |')
            lines+=['','Completed IVPs can still be inaccurate. Derivative and componentwise integrated-state errors remain separate. Failed IVPs have null full-window scores.','',CAVEAT,'',
                '## Best completed IVPs by component','',
                '| Group / case | Component | NRMSE | Run | Degree | Alpha | Threshold | Active lift terms |',
                '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |']
            for group in visible:
                rows=[r for r in merged if r['group']==group]
                for vr in sorted({r['vr'] for r in rows}):
                    for token in (['h0','cl0'] if group=='single' else ['h0','h1','cl0','cl1']):
                        candidates=[r for r in rows if r['vr']==vr and r.get('ivp_completed') and finite(r.get('ivp_'+token+'_nrmse'))]
                        if not candidates:continue
                        best=min(candidates,key=lambda r:(r['ivp_'+token+'_nrmse'],r['run_id']))
                        lines.append(f'| {group} / VR={vr} | {token} | {best["ivp_"+token+"_nrmse"]:.6g} | {best["run_id"]} | {best["poly_degree"]} | {best["ridge_alpha"]:g} | {best["threshold"]:g} | {best.get("active_terms_total")} |')
            if len(visible)==4 and len({r['experiment_id'] for r in merged})==1:
                write_csv(output/'summary_all.csv',merged)
            else:lines+=['','Merged CSV deferred until all four group result directories are present for the same experiment.']
            temporary=output/('FULL_RECOMPUTE_REPORT.'+uuid.uuid4().hex+'.tmp')
            temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8');temporary.replace(output/'FULL_RECOMPUTE_REPORT.md')
    except RuntimeError:pass  # Another group is publishing; its report reads all groups.


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=GROUPS,default='single')
    parser.add_argument('--all-groups',action='store_true')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stage',choices=['all','ols','ridge','stlsq_only','ridge_threshold'],default='all')
    parser.add_argument('--vr',type=int,nargs='+')
    parser.add_argument('--max-jobs','--stop-after',type=int)
    parser.add_argument('--output-root',type=Path,default=REPO/'result/experiment')
    parser.add_argument('--write-plan',type=Path)
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--single-data-dir',type=Path)
    parser.add_argument('--tandem-data-dir',type=Path)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs<1:parser.error('--max-jobs must be positive')
    args.output_root=args.output_root.resolve()
    return args
def main(argv=None):
    args=parse_args(argv)
    if args.worker:return worker(args.worker)
    if args.report:report(args.output_root);return 0
    plan=make_plan()
    if args.write_plan:write(args.write_plan,plan)
    if not args.execute:
        selected=GROUPS if args.all_groups else [args.group]
        plan['selection']=dict(groups=selected,stage=args.stage,vr=args.vr,execute=False,
            selected_budget=budget(sum(len(specification(g)['vr']) if args.vr is None else len(set(args.vr)&set(specification(g)['vr'])) for g in selected)))
        print(json.dumps(plan,indent=2));return 0
    if args.all_groups:raise ValueError('Execute one environment-specific group at a time.')
    if args.group!='single' and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Tandem execution is HPC-only; local dry-runs are allowed.')
    if args.group=='single' and os.environ.get('SLURM_JOB_ID'):raise ValueError('Single group is reserved for the local workstation.')
    threads=os.environ.get('SLURM_CPUS_PER_TASK','1')
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]=threads
    spec=specification(args.group);env=environment();version=code_hash()
    vrs=[vr for vr in spec['vr'] if args.vr is None or vr in args.vr]
    if not vrs:raise ValueError('No selected VR belongs to this group.')
    # Always resolve the complete group so a smoke subset and full resume share identity.
    cases=[resolve_case(spec,vr,args,env,version) for vr in spec['vr']]
    context=identity([spec,version,[(c['vr'],c['data_id'],c['snapshots']) for c in cases],
        {k:v for k,v in env.items() if k.endswith('_version')}])
    folder=args.output_root/args.group;new=0;reused=0;interrupted=False;stopped=False
    def terminate(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    with lock(folder/'.controller.lock'):
        meta=folder/'group_manifest.json'
        if meta.exists() and read(meta)['context_id']!=context:raise ValueError('Existing output belongs to a different recomputation context; use a fresh output root/config version.')
        if not meta.exists():write(meta,dict(context_id=context,experiment_id=spec['experiment_id'],spec=spec,cases=cases,code_hash=version,environment=env,created_utc=now()))
        launch=folder/'launches'/(now().replace(':','').replace('.','_')+'.json')
        write(launch,dict(started_utc=now(),argv=sys.argv,environment=env,stage=args.stage,selected_vr=vrs))
        if not (args.output_root/'full_recompute_plan.json').exists():write(args.output_root/'full_recompute_plan.json',plan)
        stages=['ols','ridge','stlsq_only','ridge_threshold'] if args.stage=='all' else [args.stage]
        write(folder/'status.json',dict(phase='Running',pid=os.getpid(),started_utc=now(),environment=env))
        try:
            for case in cases:
                if case['vr'] not in vrs:continue
                for stage in stages:
                    configs=[]
                    if stage!='ridge_threshold':configs=fixed_jobs(case,spec,stage)
                    else:
                        for degree in spec['regularized_degrees']:
                            path,rows,manifest=selection(args.output_root,case,spec,degree)
                            if path is None:
                                print(f'SKIP {case["case_id"]} p{degree}: Stage B incomplete',flush=True);continue
                            configs.extend(children(case,spec,path,rows,manifest))
                    for config in configs:
                        if args.max_jobs is not None and new>=args.max_jobs:stopped=True;break
                        if code_hash()!=version:raise RuntimeError('Source changed while executing; stop rather than mixing conventions.')
                        config['output_root']=str(args.output_root)
                        row,attempted=attempt(args.output_root,config,args.resume);new+=int(attempted);reused+=int(not attempted)
                        progress,_=refresh(args.output_root,spec,cases);report(args.output_root)
                        print(f'{args.group} VR={case["vr"]} {row["model_family"]} p={row["poly_degree"]} alpha={row["ridge_alpha"]:g} '
                            f'threshold={row["threshold"]:g}: fit={row["fit_status"]} IVP={row["ivp_status"]}; '
                            f'terminal={progress["terminal"]} fixed={progress["fixed_terminal"]}/{progress["planned"]["fixed"]} '
                            f'conditional={progress["conditional_terminal"]}/{progress["conditional_planned"]} remaining<={progress["remaining_upper_bound"]}',flush=True)
                    if stopped:break
                if stopped:break
        except KeyboardInterrupt:
            interrupted=True;print('Interrupted; saved phases will resume.',flush=True)
        finally:
            progress,_=refresh(args.output_root,spec,cases)
            phase='Finished' if progress['all_stages_attempted'] else 'Interrupted' if interrupted else 'StoppedAfterBudget' if stopped else 'PartiallyComputed'
            write(folder/'status.json',dict(phase=phase,new_attempts=new,reused=reused,ended_utc=now(),environment=env,progress=progress))
            launch_metadata=read(launch);launch_metadata.update(ended_utc=now(),phase=phase);write(launch,launch_metadata)
            report(args.output_root)
    return 130 if interrupted else 0


if __name__=='__main__':raise SystemExit(main())
