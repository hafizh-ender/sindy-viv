"""Run full Poly3 threshold assessments for explicitly selected VR/case/Ridge groups.

A JSON plan contains a groups array with vr, case and ridge preset names.
Threshold-zero scores are reused from the completed overnight study; each new
threshold receives its own fit, full suite, logs and resumable checkpoint.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import pandas as pd

from run_overnight_study import Experiment, REPO, Study, THRESHOLDS, commands, timestamp, write_json
from sindy_viv.case_bryan.evaluate_sindy.suite_cli import suite_parser


PREVIOUS = REPO / 'state/overnight_study_runs/20261027-180933Z'


def selected_commands(experiment, configs, vr):
    """Limit fitting and every diagnostic in the shared full suite to one VR."""
    result = commands(experiment, configs)
    for argv in result:
        start = argv.index('--vr_list') + 1
        end = next(i for i in range(start, len(argv)) if argv[i].startswith('--'))
        argv[start:end] = [str(vr)]
    return result


def comparison_row(group, threshold, score, reused):
    """Extract full-window errors and count nonzero coefficients in the saved model."""
    case, vr = group['case'], group['vr']
    derivative, = [r for r in score['derivative_records'] if r['VR'] == vr]
    ivp, = [r for r in score['ivp_records'] if r['VR'] == vr]
    channel = r'\dot{C}_{L0}' if case == 1 else r'\ddot{C}_{L0}'
    model = Path(derivative['metrics_path']).parents[2] / 'model.csv'
    coefficients = pd.read_csv(model, index_col=0)
    return dict(case=case, VR=vr, degree=3, ridge=group['ridge'], threshold=threshold,
                normalize_columns=False, ivp_kind=score['primary_ivp'],
                complete=ivp['complete'], reused=reused,
                derivative_nrmse=derivative['values'][channel],
                h0_nrmse=ivp['values'].get('h_0') if ivp['complete'] else None,
                CL0_nrmse=ivp['values'].get('C_{L0}') if ivp['complete'] else None,
                nonzero_coefficients=int((coefficients.to_numpy() != 0).sum()),
                nonzero_lift_coefficients={c: int((coefficients[c] != 0).sum())
                                           for c in coefficients if 'C' in c},
                model_path=str(model), derivative_metrics=derivative['metrics_path'],
                ivp_metrics=ivp['metrics_path'])


def baseline(group, previous, configs):
    """Find the exact unnormalized, threshold-zero Ridge score for one group."""
    settings = json.loads((configs / 'optimizer' / (group['ridge'] + '.json')).read_text())['kwargs']
    if settings['normalize_columns'] or settings['threshold'] != 0 or settings.get('unbias', True):
        raise ValueError('Selected presets must retain unnormalized Ridge without thresholding.')
    candidate, = [c for c in previous if c['degree'] == 3 and c['alpha'] == settings['alpha']
                   and not c['normalize_columns'] and c['threshold'] == 0]
    score = next(c for c in candidate['cases'] if c['case'] == group['case'])
    return comparison_row(group, 0., score, True)


def save_comparison(study, rows, groups):
    """Checkpoint all results and select the lowest mean h/CL error per complete group."""
    ordered = sorted(rows, key=lambda r: (r['VR'], r['case'], r['ridge'], r['threshold']))
    write_json(study.output / 'comparison.json', ordered)
    pd.DataFrame(ordered).drop(columns=['nonzero_lift_coefficients']).to_csv(
        study.output / 'comparison.csv', index=False)
    selections = []
    for group in groups:
        matches = [r for r in ordered if r['VR'] == group['vr'] and r['case'] == group['case']
                   and r['ridge'] == group['ridge']]
        complete = [r for r in matches if r['complete']]
        best = min(complete, key=lambda r: ((r['h0_nrmse'] + r['CL0_nrmse']) / 2,
                                            -r['threshold'])) if complete else None
        selections.append(dict(group=group, assessed=len(matches), expected=8,
                               all_assessed=len(matches) == 8, best_so_far=best))
    write_json(study.output / 'selection.json', dict(
        criterion='Minimum mean full-window h0/CL0 IVP NRMSE; exact ties prefer larger threshold.',
        errors_are_ratios=True, selections=selections))


def run_trial(study, group, label, threshold):
    """Fit and fully assess one single-VR trial, reusing a successful checkpoint on resume."""
    optimizer = study.threshold_config(group['ridge'], label, threshold)
    experiment = Experiment(f"vr{group['vr']}_c{group['case']}_{Path(optimizer).name}",
                            'single', group['case'], 3, optimizer)
    trial = study.output / 'experiments' / experiment.tag
    trial.mkdir(parents=True, exist_ok=True)
    checkpoint = trial / 'result.json'
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    if saved.get('fit_exit_code') != 0 or saved.get('suite_exit_code') != 0:
        fit, suite = selected_commands(experiment, study.configs, group['vr'])
        write_json(trial / 'commands.json', dict(experiment=asdict(experiment), fit=fit, suite=suite))
        study.status('Running', active_experiment=experiment.tag)
        started = time.time()
        fit_code = study.run_command(fit, trial, 'fit')
        suite_code = study.run_command(suite, trial, 'suite') if fit_code == 0 else None
        saved = dict(fit_exit_code=fit_code, suite_exit_code=suite_code)
        if fit_code == 0 and suite_code == 0:
            saved['score'] = study.read_reports(experiment, suite, started)
        write_json(checkpoint, saved)
    if saved.get('fit_exit_code') == 0 and saved.get('suite_exit_code') == 0:
        return comparison_row(group, threshold, saved['score'], False)
    return None


def main():
    """Validate the explicit plan, snapshot numerical inputs and run sequential assessments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--plan', type=Path, required=True,
        help='JSON containing selected vr, case and ridge groups.',
    )
    parser.add_argument(
        '--output_dir', type=Path, required=True,
        help='Frozen inputs, subprocess logs, comparison tables and resume checkpoints.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate selectors and baseline availability without fitting or writing outputs.',
    )
    args = parser.parse_args()
    groups = json.loads(args.plan.read_text())['groups']
    keys = [(g['vr'], g['case'], g['ridge']) for g in groups]
    if len(keys) != len(set(keys)) or not groups:
        raise ValueError('Plan groups must be nonempty and unique.')
    if any(g['vr'] not in (5, 6, 8) or g['case'] not in (1, 2, 4) for g in groups):
        raise ValueError('This comparison supports VR 5/6/8 and Cases 1/2/4.')
    previous = json.loads((PREVIOUS / 'candidate_results.json').read_text())
    for group in groups:
        baseline(group, previous, REPO / 'configs')
        _, suite = selected_commands(Experiment('check', 'single', group['case'], 3,
                                                group['ridge']), REPO / 'configs', group['vr'])
        if suite_parser('single').parse_args(suite[2:]).vr_list != [group['vr']]:
            raise ValueError('Commands must select exactly one VR.')
    if args.dry_run:
        print(f'{len(groups)} groups: {7*len(groups)} new fits/full suites; '
              f'{len(groups)} reused threshold-zero baselines.')
        return 0
    study = Study(args.output_dir, resume=(args.output_dir / 'study_status.json').exists())
    try:
        for relative in ('filtering/butterworth_single_flim2.json',
                         'differentiator/finite_difference_basic.json',
                         'feature_library/polynomial_3.json',
                         'fit_window/single/sinusoidal.json',
                         'fit_window/single/sinusoidal_evaluation.json'):
            if json.loads((study.configs / relative).read_text()) != json.loads(
                    (PREVIOUS / 'input_snapshot/configs' / relative).read_text()):
                raise ValueError(f'Cannot reuse baselines with changed config: {relative}')
        write_json(study.output / 'selected_plan.json', dict(groups=groups))
        write_json(study.output / 'policy.json', dict(
            degree=3, normalization=False, unbias=False,
            thresholds=[v for _, v in THRESHOLDS], previous_study=str(PREVIOUS),
            primary_case4_ivp='coupled_integral', errors_are_ratios=True,
            new_configurations=7*len(groups), reused_baselines=len(groups), runtime_limit=None))
        rows = [baseline(group, previous, study.configs) for group in groups]
        save_comparison(study, rows, groups)
        for group in groups:
            for label, threshold in THRESHOLDS:
                row = run_trial(study, group, label, threshold)
                if row is not None:
                    rows.append(row)
                save_comparison(study, rows, groups)
        study.status('FinishedWithErrors' if study.errors else 'Finished',
                     active_experiment=None, active_stage=None, active_command_pid=None,
                     assessment_count=len(rows), finished_utc=timestamp())
        return int(bool(study.errors))
    except Exception as error:
        study.status('Failed', error=str(error), active_command_pid=None, finished_utc=timestamp())
        raise


if __name__ == '__main__':
    raise SystemExit(main())
