"""Evaluate saved Case 6 OLS coefficients and compare per-state IVPs with Case 1."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('SINDY_LIFT_REPO_ROOT',str(Path(__file__).resolve().parents[1]))).resolve()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    for attempt in range(21):
        try:
            tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
            return
        except PermissionError:
            if attempt==20:raise
            time.sleep(.25)


def complete(record):
    solver=record.get('solver') or record.get('coupled_solver') or {}
    return (solver.get('complete') is True and solver.get('status')==0
            and solver.get('completed_samples',0)==solver.get('requested_samples',-1)
            and solver.get('requested_samples',0)>1)


def scores(record):
    if record.get('coupled_nrmse'):return record['coupled_nrmse']
    return {name:value.get('NRMSE') for name,value in record.get('metrics',{}).items()}


def csv_report(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def best_by_variable(rows):
    groups={}
    for row in rows:
        if not row.get('stable'):continue
        for name,value in row.get('ivp_nrmse',{}).items():
            if not (name.startswith('h_') or name.startswith('C_{L')):continue
            if value is None or not math.isfinite(value):continue
            key=(row['family'],row['cylinder'],row.get('LpD'),row['VR'],name)
            candidate=dict(row,variable=name,NRMSE=value)
            if key not in groups or (value,row['id'])<(groups[key]['NRMSE'],groups[key]['id']):groups[key]=candidate
    return [groups[key] for key in sorted(groups,key=repr)]


def refresh(output,plan):
    output=Path(output);rows=read(output/'original_case1.json')
    rows+= [read(p) for p in (output/'checks').glob('*/result.json')]
    for directory in plan['screening_runs']:
        for path in (Path(directory)/'trials').glob('*/result.json'):
            row=read(path)
            if row['case']!=6 or row['stage'] not in ('ridge','threshold') or row.get('status')!='completed':continue
            rows.append(dict(id=row['trial_id'],family='Case6_screened',cylinder=row['cylinder'],
                VR=row['VR'],LpD=row['LpD'],degree=row['degree'],alpha=row['alpha'],threshold=row['threshold'],
                stable=bool(row.get('stable')) and complete(row),ivp_nrmse=scores(row),source_model=row['model_path'],
                metrics_path=row.get('coupled_metrics'),normalize_columns=row['normalize_columns']))
    write(output/'comparison.json',rows)
    flat=[dict(row,variable=name,NRMSE=value) for row in rows for name,value in row.get('ivp_nrmse',{}).items()
          if name.startswith('h_') or name.startswith('C_{L')]
    fields=['family','cylinder','LpD','VR','degree','alpha','threshold','normalize_columns','stable','variable','NRMSE','source_model','metrics_path']
    csv_report(output/'comparison.csv',flat,fields)
    best=best_by_variable(rows);write(output/'best_by_variable.json',best)
    csv_report(output/'best_by_variable.csv',best,fields)
    matched=[]
    for row in rows:
        if row['family']!='Case6_OLS':continue
        others=[other for other in rows if other['family']=='Case1_OLS' and other['cylinder']==row['cylinder']
                and other['VR']==row['VR'] and other.get('LpD')==row.get('LpD') and other['degree']==row['degree']]
        for other in others:
            for name,value in row.get('ivp_nrmse',{}).items():
                if not (name.startswith('h_') or name.startswith('C_{L')):continue
                matched.append(dict(cylinder=row['cylinder'],VR=row['VR'],LpD=row['LpD'],degree=row['degree'],variable=name,
                    case6_complete=row['stable'],case1_complete=other['stable'],case6_NRMSE=value,
                    case1_NRMSE=other['ivp_nrmse'].get(name),case6_model=row['source_model'],case1_model=other['source_model']))
    csv_report(output/'degree_matched_ols.csv',matched,['cylinder','LpD','VR','degree','variable','case6_complete',
        'case1_complete','case6_NRMSE','case1_NRMSE','case6_model','case1_model'])


def prepare(output,baseline,screening_runs):
    output=Path(output).resolve();baseline=Path(baseline).resolve()
    if (output/'plan.json').exists():raise ValueError('Comparison already prepared; use --resume.')
    output.mkdir(parents=True,exist_ok=True)
    base=read(baseline/'plan.json');configs=baseline/'input_snapshot/configs'
    shutil.copytree(baseline/'input_snapshot/src',output/'snapshot/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(configs,output/'snapshot/configs',ignore=shutil.ignore_patterns('__pycache__'))
    (output/'snapshot/script').mkdir();shutil.copy2(__file__,output/'snapshot/script/check_ols_ivp_case1.py')
    donors={}
    for cylinder,metadata in base['donor_snapshots'].items():
        target=output/'snapshot/donors'/cylinder;target.mkdir(parents=True)
        metadata=Path(metadata);shutil.copy2(metadata.parent/'model.csv',target/'model.csv')
        info=read(metadata);info.pop('model_directory',None);write(target/'training.json',info)
        donors[cylinder]=str(target/'training.json')
    jobs=[]
    for path in (baseline/'trials').glob('polynomial_*/result.json'):
        row=read(path)
        if row['case']!=6 or row['optimizer']!='ols' or row['status']!='completed':continue
        model=Path(row['model_path']);assert sha(model)==row['model_sha256']
        info=read(model.parent/'training.json');optimizer=info['configurations']['optimizer']['kwargs']
        assert optimizer['alpha']==0 and optimizer['threshold']==0 and not optimizer.get('normalize_columns',False)
        identity=row['trial_id'];target=output/'snapshot/models'/identity;target.mkdir(parents=True)
        shutil.copy2(model,target/'model.csv');info.pop('model_directory',None);write(target/'training.json',info)
        for name,cfg in info['configurations'].items():write(target/'settings'/(name+'.json'),cfg)
        windows=read(configs/'fit_window'/row['cylinder']/'sinusoidal_evaluation.json')
        bound=windows[str(row['VR'])] if row['cylinder']=='single' else windows[str(row['LpD'])][str(row['VR'])]
        report=model.parent/'evaluation'
        if row['LpD'] is not None:report/=f"LpD = {row['LpD']}"
        report/=f"VR = {row['VR']}/ols_ivp_case1_comparison"
        jobs.append(dict(id=identity,cylinder=row['cylinder'],VR=row['VR'],LpD=row['LpD'],degree=row['degree'],
            model_directory=str(target),model_sha256=sha(model),source_model=str(model),report_directory=str(report),
            donor=donors[row['cylinder']],tstar0=bound['tstar0'],tstarf=bound['tstarf']))
    # Start with the user's VR=3 Poly3 question, then the remaining saved OLS models.
    jobs.sort(key=lambda j:(j['cylinder']!='single',j['VR']!=3,j['degree']!=3,j['VR'],j['LpD'] or 0,j['degree']))
    originals=[];skipped=0
    for dataset,cylinder in [('oscillating-cylinder-viv','single'),('oscillating-cylinder-tandem-viv','two_tandem')]:
        for metadata in (ROOT/'results'/dataset/'dynamics_case_1').rglob('training.json'):
            info=read(metadata)
            if info['parameterization']!='individual' or info.get('fit_window',{}).get('name')!='sinusoidal':continue
            if info.get('filtering',{}).get('name')!=f'butterworth_{cylinder}_flim2':continue
            trajectory=info['trajectories'][0];vr=int(trajectory['VR']);lpd=trajectory.get('LpD')
            windows=read(configs/'fit_window'/cylinder/'sinusoidal_evaluation.json')
            if str(vr) not in windows and cylinder=='single':continue
            bound=windows[str(vr)] if cylinder=='single' else windows.get(str(lpd),{}).get(str(vr))
            if not bound:continue
            for metrics_path in (metadata.parent/'evaluation').rglob('*integral*metrics*.json'):
                record=read(metrics_path)
                if record.get('evaluation')!='integral' or record.get('requested_tstar_region')!=[bound['tstar0'],bound['tstarf']]:skipped+=1;continue
                opt=info['configurations']['optimizer']['kwargs'];poly=info['configurations']['state_feature_library']['kwargs'].get('degree')
                row=dict(id=str(metrics_path),family='Case1_original',cylinder=cylinder,VR=vr,LpD=lpd,
                    degree=poly,alpha=opt.get('alpha'),threshold=opt.get('threshold'),normalize_columns=opt.get('normalize_columns',False),
                    stable=complete(record),ivp_nrmse=scores(record),source_model=str(metadata.parent/'model.csv'),
                    metrics_path=str(metrics_path),metrics_sha256=sha(metrics_path),training_tstar_region=trajectory.get('tstar_region'),
                    requested_tstar_region=record['requested_tstar_region'])
                originals.append(row)
                if opt.get('alpha')==0 and opt.get('threshold')==0 and not opt.get('normalize_columns',False):
                    originals.append(dict(row,family='Case1_OLS'))
    write(output/'original_case1.json',originals)
    plan=dict(prepared_utc=now(),baseline=str(baseline),screening_runs=[str(Path(p).resolve()) for p in screening_runs],
        jobs=jobs,case1_metric_records=len([r for r in originals if r['family']=='Case1_original']),case1_skipped_windows=skipped,
        method=base.get('method','RK45'),rtol=base.get('rtol',1e-6),atol=base.get('atol',1e-9),max_state_factor=base.get('max_state_factor',10),
        comparison='Separate h and CL IVP NRMSE for each cylinder; no averaging. Full/partial completion explicitly distinguished.',
        case1_tandem_available=any(r['cylinder']=='two_tandem' for r in originals),
        note='Case 1 uses its own fitted full dynamics; Case 6 uses the frozen parameterized Case 3 motion donor. Case 1 metrics are archived original results, not refits.')
    write(output/'plan.json',plan)
    write(output/'frozen_manifest.json',{str(p.relative_to(output)):sha(p) for p in (output/'snapshot').rglob('*') if p.is_file()})
    refresh(output,plan);write(output/'status.json',dict(phase='Prepared',jobs=len(jobs),updated_utc=now()))
    print(f'Prepared {len(jobs)} saved OLS IVPs; imported {plan["case1_metric_records"]} original Case 1 assessments.',flush=True)


def worker(output,identity):
    from sindy_viv.case_bryan.evaluate_sindy.functions import evaluate_sindy
    plan=read(output/'plan.json');job=next(j for j in plan['jobs'] if j['id']==identity)
    model=Path(job['model_directory']);assert sha(model/'model.csv')==job['model_sha256']
    settings=model/'settings';cylinder=job['cylinder'];pair=dict(vr_list=[job['VR']],lpd_list=None if cylinder=='single' else [job['LpD']])
    row=dict(job,family='Case6_OLS',alpha=0,threshold=0,normalize_columns=False,stable=False,ivp_nrmse={},started_utc=now())
    try:
        key=job['VR'] if cylinder=='single' else (job['LpD'],job['VR'])
        record=evaluate_sindy(6,settings/'optimizer.json',cylinder_type=cylinder,parameterization='individual',**pair,
            differentiator_case=settings/'differentiator.json',filtering_case=settings/'filtering.json',
            state_feature_library_case=settings/'state_feature_library.json',model_dir=model,
            evaluation_type='integral',coupled_motion=True,case_3_model=job['donor'],tstar0=job['tstar0'],tstarf=job['tstarf'],
            method=plan['method'],rtol=plan['rtol'],atol=plan['atol'],max_state_factor=plan['max_state_factor'],
            save_plot=False,return_data=False,report_directory=job['report_directory'])[key]
        row.update(status='completed',stable=complete(record),ivp_nrmse=scores(record),solver=record['solver'],metrics_path=str(record['metrics_path']))
    except Exception as error:row.update(status='error',error=str(error),error_type=type(error).__name__)
    assert sha(model/'model.csv')==job['model_sha256']
    row['finished_utc']=now();write(output/'checks'/identity/'result.json',row)
    print(json.dumps({key:row.get(key) for key in ('id','status','stable','ivp_nrmse','error')}),flush=True)
    return int(row['status']=='error')


@contextmanager
def comparison_lock(output):
    """Keep comparison writers exclusive without depending on a study runner."""
    with (Path(output)/'.run.lock').open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            acquire=lambda:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            release=lambda:msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            acquire=lambda:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            release=lambda:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError('This comparison is already active.') from error
        try:
            yield
        finally:
            stream.seek(0);release()


def run(output):
    with comparison_lock(output):
        plan=read(output/'plan.json')
        for relative,value in read(output/'frozen_manifest.json').items():
            if sha(output/relative)!=value:raise ValueError(f'Frozen comparison input changed: {relative}')
        for index,job in enumerate(plan['jobs']):
            checkpoint=output/'checks'/job['id']/'result.json'
            if checkpoint.is_file() and read(checkpoint).get('status')=='completed':continue
            directory=checkpoint.parent;directory.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,PYTHONPATH=str(output/'snapshot/src'),SINDY_LIFT_REPO_ROOT=str(ROOT),MPLBACKEND='Agg',
                     PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (directory/'stdout.log').open('w',encoding='utf-8') as stdout,(directory/'stderr.log').open('w',encoding='utf-8') as stderr:
                worker_script=os.environ.get('SINDY_OLS_IVP_WORKER_SCRIPT',str(output/'snapshot/script/check_ols_ivp_case1.py'))
                process=subprocess.Popen([sys.executable,'-u',worker_script,
                    '--output',str(output),'--worker',job['id']],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
                try:
                    while process.poll() is None:
                        write(output/'status.json',dict(phase='EvaluatingOLS',active_job=job['id'],worker_pid=process.pid,
                            job_index=index+1,jobs=len(plan['jobs']),updated_utc=now()))
                        time.sleep(2)
                except BaseException:
                    if process.poll() is None:process.terminate()
                    process.wait();raise
            if not checkpoint.is_file():write(checkpoint,dict(job,family='Case6_OLS',status='error',stable=False,ivp_nrmse={},
                error=f'Worker exited {process.returncode}; inspect stderr.log',alpha=0,threshold=0))
            refresh(output,plan)
        errors=[p.parent.name for p in (output/'checks').glob('*/result.json') if read(p).get('status')=='error']
        write(output/'status.json',dict(phase='FinishedWithErrors' if errors else 'Finished',errors=errors,updated_utc=now()))
        return int(bool(errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--screening-runs',type=Path,nargs='*',default=[])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args();output=args.output.resolve()
    if args.worker:raise SystemExit(worker(output,args.worker))
    if not args.resume:
        if args.baseline is None:parser.error('--baseline is required when preparing a comparison')
        prepare(output,args.baseline,args.screening_runs)
    if not args.prepare_only:raise SystemExit(run(output))
