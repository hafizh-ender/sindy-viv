"""Assess case-specific Poly3 Ridge thresholds at VR=3 using the full suite.

Reuse the completed threshold-zero baselines and Case 4 threshold assessments
from the overnight study. Fit the fourteen new Case 1/2 configurations only.
Save per-case comparison rows alongside full checkpoints and subprocess logs.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import pandas as pd

from run_overnight_study import Experiment, REPO, Study, THRESHOLDS, commands, timestamp, write_json


RIDGES = {1: 'ridge_alpha1e-4', 2: 'ridge_alpha5e-1', 4: 'ridge_alpha1e0'}
PREVIOUS = REPO / 'state/overnight_study_runs/20261027-180933Z'


def vr3_commands(experiment, configs):
    """Restrict the shared fitting and full-assessment commands to VR=3."""
    result = commands(experiment, configs)
    for argv in result:
        start = argv.index('--vr_list') + 1
        end = next(i for i in range(start, len(argv)) if argv[i].startswith('--'))
        argv[start:end] = ['3']
    return result


def comparison_row(case, ridge, threshold, score, reused):
    """Extract complete-window errors without reporting truncated IVP errors."""
    derivative = next(r for r in score['derivative_records'] if r['VR'] == 3)
    ivp = next(r for r in score['ivp_records'] if r['VR'] == 3)
    channel = r'\dot{C}_{L0}' if case == 1 else r'\ddot{C}_{L0}'
    return dict(case=case, VR=3, degree=3, ridge=ridge, threshold=threshold,
                normalize_columns=False, ivp_kind=score['primary_ivp'],
                complete=ivp['complete'], reused=reused,
                derivative_nrmse=derivative['values'][channel],
                h0_nrmse=ivp['values'].get('h_0') if ivp['complete'] else None,
                CL0_nrmse=ivp['values'].get('C_{L0}') if ivp['complete'] else None,
                derivative_metrics=derivative['metrics_path'],
                ivp_metrics=ivp['metrics_path'])


def main():
    """Run fourteen new fits, reuse ten assessments, and persist comparison tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output_dir', type=Path, required=True,
        help='Directory for frozen inputs, checkpoints and comparison tables.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate reuse selections and print the run size without fitting.',
    )
    args = parser.parse_args()
    previous = json.loads((PREVIOUS / 'candidate_results.json').read_text())
    reused = []
    for case, ridge in RIDGES.items():
        for threshold in ([0.] + [value for _, value in THRESHOLDS] if case == 4 else [0.]):
            candidate, = [c for c in previous if c['degree'] == 3
                          and c['alpha'] == {1: .0001, 2: .5, 4: 1.}[case]
                          and not c['normalize_columns'] and c['threshold'] == threshold]
            score = next(c for c in candidate['cases'] if c['case'] == case)
            reused.append(comparison_row(case, ridge, threshold, score, True))
    if args.dry_run:
        print('VR=3, Poly3: 14 new configurations / 28 commands; 10 reused assessments.')
        return 0
    study = Study(args.output_dir, resume=(args.output_dir / 'study_status.json').exists())
    # Reuse only assessments with identical numerical input configurations.
    for relative in ('filtering/butterworth_single_flim2.json',
                     'differentiator/finite_difference_basic.json',
                     'feature_library/polynomial_3.json',
                     'fit_window/single/sinusoidal.json',
                     'fit_window/single/sinusoidal_evaluation.json'):
        assert json.loads((study.configs / relative).read_text()) == json.loads(
            (PREVIOUS / 'input_snapshot/configs' / relative).read_text()), relative
    write_json(study.output / 'policy.json', dict(
        VR=3, degree=3, ridges=RIDGES, normalization=False, unbias=False,
        thresholds=[v for _, v in THRESHOLDS], previous_study=str(PREVIOUS),
        primary_case4_ivp='coupled_integral', errors_are_ratios=True,
        new_configurations=14, reused_assessments=10, runtime_limit=None))
    rows = reused[:]
    for case in (1, 2):
        ridge = RIDGES[case]
        for label, threshold in THRESHOLDS:
            optimizer = study.threshold_config(ridge, label, threshold)
            experiment = Experiment(f'vr3_c{case}_{Path(optimizer).name}', 'single', case, 3, optimizer)
            trial = study.output / 'experiments' / experiment.tag
            trial.mkdir(parents=True, exist_ok=True)
            checkpoint = trial / 'result.json'
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
            else:
                saved = {}
            if saved.get('fit_exit_code') != 0 or saved.get('suite_exit_code') != 0:
                fit, suite = vr3_commands(experiment, study.configs)
                write_json(trial / 'commands.json', dict(experiment=asdict(experiment), fit=fit, suite=suite))
                study.status('Running', active_experiment=experiment.tag)
                started = time.time()
                fit_code = study.run_command(fit, trial, 'fit')
                suite_code = study.run_command(suite, trial, 'suite') if fit_code == 0 else None
                score = study.read_reports(experiment, suite, started)
                saved = dict(fit_exit_code=fit_code, suite_exit_code=suite_code, score=score)
                write_json(checkpoint, saved)
            if saved.get('fit_exit_code') == 0 and saved.get('suite_exit_code') == 0:
                rows.append(comparison_row(case, ridge, threshold, saved['score'], False))
            write_json(study.output / 'comparison.json', rows)
            pd.DataFrame(rows).sort_values(['case', 'threshold']).to_csv(
                study.output / 'comparison.csv', index=False)
    study.status('FinishedWithErrors' if study.errors else 'Finished',
                 active_experiment=None, active_stage=None, active_command_pid=None,
                 assessment_count=len(rows), finished_utc=timestamp())
    return int(bool(study.errors))


if __name__ == '__main__':
    raise SystemExit(main())
