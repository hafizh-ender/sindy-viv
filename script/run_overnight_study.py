"""Run the approved tandem, ridge, conditional Poly5 and threshold studies.

Existing fitting/evaluation CLIs perform every numerical calculation. This
orchestrator snapshots inputs, logs subprocesses, scores saved reports, and
checkpoints decisions so an interrupted study can resume without repeating
completed experiments. It never sets an IVP or experiment runtime limit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

import numpy as np
import pandas as pd

from sindy_viv.case_bryan.analysis.data_windows import data_selection
from sindy_viv.case_bryan.evaluate_sindy.single_suite import _model_directories, _summary_folder
from sindy_viv.case_bryan.evaluate_sindy.suite_cli import suite_parser


REPO = Path(__file__).resolve().parents[1]
CASES = (1, 2, 4)
THRESHOLDS = (('1e0', 1.), ('5e-1', .5), ('1e-1', .1), ('5e-2', .05),
              ('1e-2', .01), ('5e-3', .005), ('1e-3', .001))
RIDGES = ('ridge_alpha1e-4', 'ridge_alpha1e-3', 'ridge_alpha1e-2', 'ridge_basic',
          'ridge_alpha1e-1', 'ridge_alpha5e-1', 'ridge_alpha1e0', 'ridge_alpha5e0')


@dataclass(frozen=True)
class Experiment:
    """Identify one multi-trajectory fit and its matching full evaluation suite."""

    tag: str
    cylinder: str
    case: int
    degree: int
    optimizer: str
    mode: str = 'individual'
    window: str = 'sinusoidal'


def write_json(path: Path, payload: object) -> None:
    """Atomically replace a UTF-8 checkpoint, rejecting nonfinite JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def timestamp() -> str:
    """Return an unambiguous UTC timestamp for logs and checkpoint metadata."""
    return datetime.now(timezone.utc).isoformat()


def tandem_experiments() -> list[Experiment]:
    """Build the approved 17 tandem configurations, excluding held Part F."""
    experiments = []
    for part, mode in (('A', 'individual'), ('B', 'vr')):
        for number, (optimizer, window) in enumerate(
            ((optimizer, window) for optimizer in ('ols', 'stlsq_vanilla_basic')
             for window in ('overall_reduced', 'sinusoidal')), 1
        ):
            experiments.append(Experiment(f'tandem_{part}{number:02d}', 'two_tandem',
                                          3, 1, optimizer, mode, window))
    for part, case in zip('CDE', CASES):
        for degree in (2, 3, 4):
            experiments.append(Experiment(f'tandem_{part}{degree:02d}', 'two_tandem',
                                          case, degree, 'ols'))
    return experiments


def ridge_names() -> list[str]:
    """Return all eight ridge strengths with normalization off and on."""
    return [name + suffix for name in RIDGES for suffix in ('', '_normalized')]


def commands(experiment: Experiment, config_root: Path) -> tuple[list[str], list[str]]:
    """Build matching fit/suite argv lists with all approved diagnostic windows."""
    cylinder = experiment.cylinder
    window_root = config_root / 'fit_window' / cylinder
    common = ['--dynamics_case', str(experiment.case), '--vr_list',
              *map(str, range(3, 9) if cylinder == 'single' else range(5, 15))]
    if cylinder == 'two_tandem':
        common += ['--lpd_list', '1.5', '2', '3', '3.5', '4']
    common += ['--differentiator_case', str(config_root / 'differentiator/finite_difference_basic.json'),
               '--filtering_case', str(config_root / 'filtering' / f'butterworth_{cylinder}_flim2.json'),
               '--state_feature_library_case', str(config_root / 'feature_library' / f'polynomial_{experiment.degree}.json'),
               '--optimizer_case', str(config_root / 'optimizer' / (experiment.optimizer + '.json')),
               '--fit_window_case', str(window_root / (experiment.window + '.json'))]
    if experiment.mode == 'vr':
        common += ['--input_feature_library_case', str(config_root / 'feature_library/polynomial_2.json')]
    mode = 'individual' if experiment.mode == 'individual' else 'parametric_vr'
    fit = [sys.executable, str(REPO / 'script' / f'perform_sindy_{cylinder}_{mode}.py'), *common]
    suite = [sys.executable, str(REPO / 'script' / f'evaluate_sindy_{cylinder}_suite.py'), *common,
             '--parameterization', experiment.mode, '--derivative_window_cases',
             *[str(window_root / (name + '.json')) for name in ('overall_reduced', 'sinusoidal', 'sinusoidal_evaluation')],
             '--integral_window_case', str(window_root / 'sinusoidal_evaluation.json'),
             '--spectral_window_cases', str(window_root / 'sinusoidal.json'),
             str(window_root / 'sinusoidal_evaluation.json'), '--method', 'RK45', '--max_state_factor', '10']
    if experiment.mode == 'vr':
        suite += ['--training_vr_list', *map(str, range(5, 15))]
    if experiment.case in (1, 2):
        suite += ['--phase_comparison']
    elif experiment.case == 3:
        suite += ['--weight_heatmap']
    return fit, suite


def lift_score(records: list[dict], case: int, cylinder: str, expected: set[tuple]) -> float | None:
    """Average lift-derivative NRMSE over every required trajectory and cylinder.

    Missing, duplicate, incomplete or nonfinite derivative records invalidate
    the average rather than silently dropping difficult cases. IVP completion
    does not filter derivative scores. Case 1 uses CLdot; Cases 2/4 use CLddot.
    """
    values = []
    seen = set()
    prefix = r'\dot' if case == 1 else r'\ddot'
    for row in records:
        key = (row.get('LpD'), row['VR'])
        if key in seen or key not in expected or not row.get('complete'):
            return None
        seen.add(key)
        channels = [row.get('values', {}).get(rf'{prefix}{{C}}_{{L{i}}}')
                    for i in range(1 if cylinder == 'single' else 2)]
        if any(value is None or not math.isfinite(value) for value in channels):
            return None
        values.append(float(np.mean(channels)))
    return float(np.mean(values)) if seen == expected else None


def completion_count(records: list[dict], expected: set[tuple]) -> int:
    """Count unique full-window IVPs; partial, missing and invalid rows count zero."""
    by_key = {}
    for row in records:
        key = (row.get('LpD'), row['VR'])
        if key in by_key:
            raise ValueError('Duplicate IVP records cannot be counted as independent completions.')
        by_key[key] = row
    return sum(bool(by_key.get(key, {}).get('complete')) for key in expected)


def candidate_rank(candidate: dict) -> tuple:
    """Prefer more completed IVPs, then lower mean lift error, then a stable name."""
    error = candidate['derivative_nrmse']
    return (-candidate['completed'], math.inf if error is None else error, candidate['optimizer'])


def select_threshold_candidates(candidates: list[dict], minimum: int = 3) -> list[dict]:
    """Keep every fully stable candidate and fill to three using completion/error rank."""
    ranked = sorted(candidates, key=candidate_rank)
    full = [candidate for candidate in ranked if candidate['fully_stable']]
    return full if len(full) >= minimum else ranked[:minimum]


def select_poly5_candidates(candidates: list[dict], target: float = .01) -> tuple[list[dict], str]:
    """Apply the strict Poly5 gate across both lower orders, deduplicating ridge pairs.

    No full-stability candidate, or any fully stable candidate meeting the
    derivative target, skips Poly5. Partial fallback candidates never qualify.
    Missing derivative scores cannot justify an adaptive accuracy decision.
    """
    full = [c for c in candidates if c['fully_stable'] and c['derivative_nrmse'] is not None]
    if not full:
        return [], 'No fully stable Poly3/4 combination with complete derivative assessment.'
    if min(c['derivative_nrmse'] for c in full) <= target:
        return [], 'A fully stable Poly3/4 combination meets the 1% lift-derivative target.'
    unique = {}
    for candidate in sorted(full, key=candidate_rank):
        unique.setdefault(candidate['optimizer'], candidate)
    return list(unique.values()), 'Fully stable Poly3/4 combinations miss the derivative target; extend only these ridge pairs.'


def aggregate_candidate(degree: int, optimizer: str, cases: list[dict], settings: dict) -> dict:
    """Give Cases 1/2/4 equal accuracy weight and sum their primary IVP completions."""
    scores = [case['derivative_nrmse'] for case in cases]
    completed = sum(case['completed'] for case in cases)
    expected = sum(case['expected'] for case in cases)
    valid = len(cases) == 3 and all(score is not None for score in scores)
    return dict(degree=degree, optimizer=optimizer, alpha=settings['alpha'],
                normalize_columns=settings['normalize_columns'], threshold=settings['threshold'],
                derivative_nrmse=float(np.mean(scores)) if valid else None,
                completed=completed, expected=expected,
                fully_stable=completed == expected and expected > 0 and valid,
                cases=cases)


class Study:
    """Own frozen inputs, sequential subprocesses, checkpoints and adaptive decisions."""

    def __init__(self, output: Path, resume: bool = False):
        """Initialize an isolated study directory; refuse accidental reuse without --resume."""
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output / 'study_status.json'
        if self.status_path.exists() and not resume:
            raise ValueError('Study already exists; use --resume to reuse its checkpoints.')
        self.configs = self.output / 'input_snapshot/configs'
        source = self.output / 'input_snapshot/src'
        if not self.configs.exists():
            shutil.copytree(REPO / 'configs', self.configs, ignore=shutil.ignore_patterns('__pycache__'))
        if not source.exists():
            shutil.copytree(REPO / 'src', source, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
        write_json(self.output / 'source_manifest.json', {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob('*.py'))})
        self.environment = dict(os.environ, PYTHONPATH=str(source), MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
        self.groups = []
        self.errors = []
        self.state = dict(phase='Preparing', active_experiment=None, active_command_pid=None,
                          started_utc=timestamp(), updated_utc=None, errors=self.errors)
        write_json(self.output / 'policy.json', dict(
            derivative_target=.01, primary_case4_ivp='coupled_integral',
            ranking='completed case/VR IVPs descending, equally averaged lift-derivative NRMSE ascending',
            minimum_threshold_candidates=3, thresholds=[value for _, value in THRESHOLDS],
            poly5_requires_full_stability=True, runtime_limit=None,
            tandem_poly5='quartic lift-derivative mean above 1% across Cases 1/2/4; derivative-only decision',
            source_snapshot=str(source), configuration_snapshot=str(self.configs)))

    def status(self, phase: str, **changes) -> None:
        """Persist active work and emit a short progress line suitable for background logs."""
        self.state.update(changes, phase=phase, updated_utc=timestamp())
        write_json(self.status_path, self.state)
        print(f'{timestamp()} {phase}: {changes}', flush=True)

    def run_command(self, argv: list[str], trial: Path, stage: str) -> int:
        """Run one CLI without a timeout, capturing separate stdout/stderr and its PID."""
        with (trial / f'{stage}.stdout.log').open('w', encoding='utf-8') as stdout, \
             (trial / f'{stage}.stderr.log').open('w', encoding='utf-8') as stderr:
            process = subprocess.Popen(argv, cwd=REPO, env=self.environment, stdout=stdout, stderr=stderr)
            self.status('Running', active_stage=stage, active_command_pid=process.pid)
            code = process.wait()
        self.status('Running', active_command_pid=None)
        if code:
            self.errors.append(dict(experiment=self.state['active_experiment'], stage=stage, exit_code=code))
        return code

    def read_reports(self, experiment: Experiment, argv: list[str], started: float) -> dict:
        """Read only fresh reports for matching selection/window and count fitted terms."""
        options = vars(suite_parser(experiment.cylinder).parse_args(argv[2:]))
        options.setdefault('training_lpd_list', None)
        selection = data_selection(experiment.cylinder, options['vr_list'], options.get('lpd_list'))
        expected = set(selection.bounds)
        result = dict(case=experiment.case, derivative_nrmse=None, completed=0,
                      expected=len(expected), primary_ivp='coupled_integral' if experiment.case == 4 else 'integral',
                      derivative_records=[], ivp_records=[], models=[], source_reports={})
        try:
            directories = _model_directories(options, selection)
        except (ValueError, FileNotFoundError) as error:
            result['error'] = str(error)
            return result
        window = data_selection(experiment.cylinder, selection.vrs,
                                None if experiment.cylinder == 'single' else selection.lpds,
                                fit_window_case=options['integral_window_case'])
        folder = _summary_folder(directories, window, options)
        for kind in ('derivative', result['primary_ivp']):
            path = folder / f'{kind}_suite_summary.json'
            if not path.is_file() or path.stat().st_mtime < started:
                continue
            payload = json.loads(path.read_text(encoding='utf-8'))
            rows = payload['records']
            if any(tuple(row['requested_tstar_region']) != window.bounds[(row.get('LpD'), row['VR'])]
                   for row in rows):
                raise ValueError('Report bounds differ from the study evaluation window.')
            result['source_reports'][kind] = str(path)
            if kind == 'derivative':
                result['derivative_records'] = rows
                result['derivative_nrmse'] = lift_score(rows, experiment.case, experiment.cylinder, expected)
            else:
                result['ivp_records'] = rows
                result['completed'] = completion_count(rows, expected)
        for directory in sorted(set(directories.values())):
            path = directory / 'model.csv'
            if not path.is_file() or path.stat().st_mtime < started:
                continue
            matrix = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
            result['models'].append(dict(path=str(path), feature_count=len(matrix),
                                         nonzero_coefficients=int(np.count_nonzero(matrix)),
                                         retained_features=int(np.count_nonzero(np.any(matrix != 0, axis=1))),
                                         per_equation_nonzero=np.count_nonzero(matrix, axis=0).tolist()))
        return result

    def experiment(self, experiment: Experiment) -> dict:
        """Fit and fully assess a configuration, or reuse its completed study checkpoint."""
        trial = self.output / 'experiments' / experiment.tag
        trial.mkdir(parents=True, exist_ok=True)
        checkpoint = trial / 'result.json'
        if checkpoint.exists():
            previous = json.loads(checkpoint.read_text(encoding='utf-8'))
            if previous.get('fit_exit_code') == 0 and previous.get('suite_exit_code') == 0:
                return previous['score']
        fit, suite = commands(experiment, self.configs)
        write_json(trial / 'commands.json', dict(experiment=asdict(experiment), fit=fit, suite=suite))
        self.status('Running', active_experiment=experiment.tag)
        started = time.time()
        fit_code = self.run_command(fit, trial, 'fit')
        suite_code = self.run_command(suite, trial, 'suite') if fit_code == 0 else None
        try:
            score = self.read_reports(experiment, suite, started)
        except Exception as error:
            score = dict(case=experiment.case, derivative_nrmse=None, completed=0,
                         expected=6 if experiment.cylinder == 'single' else 50, error=str(error))
            self.errors.append(dict(experiment=experiment.tag, stage='report', error=str(error)))
        write_json(checkpoint, dict(experiment=asdict(experiment), fit_exit_code=fit_code,
                                   suite_exit_code=suite_code, score=score, completed_utc=timestamp()))
        return score

    def group(self, degree: int, optimizer: str, stage: str) -> dict:
        """Run all three single-cylinder cases for one optimizer/order and summarize them."""
        cases = [self.experiment(Experiment(f'{stage}_p{degree}_{Path(optimizer).name}_c{case}',
                                           'single', case, degree, optimizer)) for case in CASES]
        settings = json.loads((self.configs / 'optimizer' / (optimizer + '.json')).read_text())['kwargs']
        candidate = aggregate_candidate(degree, optimizer, cases, settings)
        candidate['stage'] = stage
        self.groups.append(candidate)
        self.save_groups()
        return candidate

    def save_groups(self) -> None:
        """Update detailed JSON and a compact CSV without rounding numerical scores."""
        write_json(self.output / 'candidate_results.json', self.groups)
        rows = []
        for group in self.groups:
            row = {key: value for key, value in group.items() if key != 'cases'}
            models = [model for case in group['cases'] for model in case.get('models', [])]
            row['mean_nonzero_coefficients'] = float(np.mean([m['nonzero_coefficients'] for m in models])) if models else None
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.output / 'candidate_results.csv', index=False)

    def threshold_config(self, ridge: str, label: str, threshold: float) -> str:
        """Create a distinct threshold preset, retaining the selected ridge and normalization."""
        relative = Path('threshold_sweep') / f'{Path(ridge).name}_threshold{label}'
        payload = json.loads((self.configs / 'optimizer' / (ridge + '.json')).read_text())
        payload['kwargs'].update(threshold=threshold, unbias=False)
        for root in (self.configs, REPO / 'configs'):
            path = root / 'optimizer' / relative.with_suffix('.json')
            if path.exists() and json.loads(path.read_text()) != payload:
                raise ValueError(f'Refusing to replace a conflicting optimizer configuration: {path}')
            write_json(path, payload)
        return relative.as_posix()

    def run(self, *, skip_tandem: bool = False) -> None:
        """Execute approved stages, optionally resuming directly at ridge by user request."""
        if skip_tandem:
            write_json(self.output / 'tandem_stopped_by_user.json', dict(
                updated_utc=timestamp(), skip_remaining_tandem=True,
                reason='User requested stopping remaining tandem work and continuing at single-cylinder ridge.'))
            self.status('StartingRidge', active_experiment=None, active_stage=None)
        else:
            tandem = tandem_experiments()
            tandem_scores = {experiment.tag: self.experiment(experiment) for experiment in tandem}
            quartic = [tandem_scores[f'tandem_{part}04']['derivative_nrmse'] for part in 'CDE']
            tandem_error = float(np.mean(quartic)) if all(value is not None for value in quartic) else None
            tandem_poly5 = tandem_error is not None and tandem_error > .01
            write_json(self.output / 'tandem_poly5_decision.json', dict(
                quartic_mean_lift_derivative_nrmse=tandem_error, target=.01, run_poly5=tandem_poly5,
                criterion='Derivative accuracy only; no IVP stability gate.', missing_accuracy=tandem_error is None))
            if tandem_poly5:
                for part, case in zip('CDE', CASES):
                    self.experiment(Experiment(f'tandem_{part}_poly5', 'two_tandem', case, 5, 'ols'))
        baselines = [self.group(degree, ridge, 'ridge') for degree in (3, 4) for ridge in ridge_names()]
        selected, reason = select_poly5_candidates(baselines)
        write_json(self.output / 'single_poly5_decision.json', dict(
            reason=reason, selected_ridge_pairs=[c['optimizer'] for c in selected],
            qualifying_lower_order_candidates=selected))
        poly5 = [self.group(5, candidate['optimizer'], 'ridge_poly5') for candidate in selected]
        eligible = baselines + poly5
        decisions = {}
        for degree in sorted({candidate['degree'] for candidate in eligible}):
            candidates = [candidate for candidate in eligible if candidate['degree'] == degree]
            chosen = select_threshold_candidates(candidates)
            decisions[str(degree)] = dict(selected=[candidate['optimizer'] for candidate in chosen],
                                          fully_stable_count=sum(c['fully_stable'] for c in candidates),
                                          available_candidates=len(candidates),
                                          minimum_shortfall=max(0, 3-len(chosen)))
            write_json(self.output / 'threshold_selection.json', decisions)
            for candidate in chosen:
                for label, threshold in THRESHOLDS:
                    optimizer = self.threshold_config(candidate['optimizer'], label, threshold)
                    self.group(degree, optimizer, 'threshold')
        winners = {}
        for degree in sorted({candidate['degree'] for candidate in self.groups}):
            candidates = [candidate for candidate in self.groups if candidate['degree'] == degree]
            stable = [candidate for candidate in candidates if candidate['fully_stable']]
            winners[str(degree)] = dict(
                best_available=min(candidates, key=candidate_rank),
                best_fully_stable=min(stable, key=candidate_rank) if stable else None)
        write_json(self.output / 'final_selection.json', winners)
        self.status('FinishedWithErrors' if self.errors else 'Finished',
                    active_experiment=None, active_stage=None, active_command_pid=None,
                    candidate_count=len(self.groups), finished_utc=timestamp())


def preflight(config_root: Path) -> dict:
    """Validate all fixed inputs and full trajectory windows without loading CFD data."""
    for cylinder in ('single', 'two_tandem'):
        for window in ('overall_reduced', 'sinusoidal', 'sinusoidal_evaluation'):
            data_selection(cylinder, fit_window_case=config_root / 'fit_window' / cylinder / (window + '.json'))
        for mode in ('individual', 'parametric_vr'):
            if not (REPO / 'script' / f'perform_sindy_{cylinder}_{mode}.py').is_file():
                raise FileNotFoundError(f'Missing {cylinder} {mode} fitting CLI.')
    paths = [config_root / 'optimizer' / (name + '.json') for name in ridge_names()]
    for path in paths:
        kwargs = json.loads(path.read_text())['kwargs']
        if kwargs['threshold'] != 0 or kwargs.get('unbias', True):
            raise ValueError(f'Ridge must retain regularization without thresholding: {path}')
    for degree in range(1, 6):
        path = config_root / 'feature_library' / f'polynomial_{degree}.json'
        if json.loads(path.read_text())['kwargs']['degree'] != degree:
            raise ValueError(f'Incorrect polynomial degree: {path}')
    return dict(tandem_fixed_configurations=17, tandem_fixed_commands=34,
                single_fixed_ridge_configurations=96, single_fixed_ridge_commands=192,
                adaptive_poly5=True, adaptive_thresholds=True)


def parse_args() -> argparse.Namespace:
    """Parse a dry-run, fresh study or checkpoint-resume request."""
    parser = argparse.ArgumentParser(
        description='Run the approved adaptive tandem and single-cylinder optimizer study.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Study logs/checkpoints; defaults to state/overnight_study_runs/<UTC timestamp>.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=os.environ.get('SINDY_STUDY_RESUME') == '1',
        help='Reuse completed experiment checkpoints in --output_dir.',
    )
    parser.add_argument(
        '--skip-tandem',
        action='store_true',
        default=os.environ.get('SINDY_STUDY_SKIP_TANDEM') == '1',
        help='Skip all tandem stages and proceed directly to the single-cylinder ridge sweep.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs and print the fixed grid without fitting or writing study outputs.',
    )
    args = parser.parse_args()
    if args.resume and args.output_dir is None:
        parser.error('--resume requires --output_dir')
    return args


def main() -> int:
    """Preflight and run sequentially, preserving fatal errors for unattended inspection."""
    args = parse_args()
    os.chdir(REPO)
    frozen = args.output_dir / 'input_snapshot/configs' if args.output_dir else None
    counts = preflight(frozen if frozen is not None and frozen.exists() else REPO / 'configs')
    counts['tandem_skipped_by_user'] = args.skip_tandem
    if args.dry_run:
        print(json.dumps(dict(counts=counts, tandem=[asdict(e) for e in tandem_experiments()],
                              ridge_pairs=ridge_names(), thresholds=THRESHOLDS), indent=2))
        return 0
    output = args.output_dir or REPO / 'state/overnight_study_runs' / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%SZ')
    study = Study(output, args.resume)
    write_json(study.output / 'preflight.json', counts)
    try:
        study.run(skip_tandem=args.skip_tandem)
    except BaseException as error:
        study.status('Failed', fatal_error=f'{type(error).__name__}: {error}')
        raise
    return 1 if study.errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
