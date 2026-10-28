"""Assess manually entered tandem Case 3 VR models without fitting or changing CSVs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from sindy_viv.case_bryan.analysis.data_windows import data_selection
from sindy_viv.case_bryan.evaluate_sindy.single_suite import (
    _coefficient_table, evaluate_two_tandem_suite,
)
from sindy_viv.case_bryan.model_storage.paths import number_text
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / (
    'results/oscillating-cylinder-tandem-viv/dynamics_case_3/'
    'filtering=butterworth_two_tandem_flim2/finite_difference_basic/'
    'parametric_vr/state_polynomial_1/input_polynomial_2/theoretical'
)


def model_selection(root: Path, configs: Path) -> tuple[dict, dict]:
    """Validate five explicit manual model folders and all 50 evaluation selections.

    Return a (L/D, VR)-to-directory mapping and a hash/library manifest. CSV
    features and targets must match the configured tensor library exactly;
    metadata fitting prepares names only and never fits coefficients.
    """
    selection = data_selection('two_tandem')
    libraries = dict(
        state_feature_library=json.loads((configs / 'feature_library/polynomial_1.json').read_text()),
        parameter_feature_library=json.loads((configs / 'feature_library/polynomial_2.json').read_text()))
    layout = get_case_layout(3, 'two_tandem')
    directories, models = {}, []
    for lpd in selection.lpds:
        directory = root / ('LpD=' + number_text(lpd)) / 'fit_window=overall_reduced'
        table = _coefficient_table(directory, layout, 'vr', selection.vrs[0], lpd,
                                   substitute_parameters=False, configurations=libraries)
        path = directory / 'model.csv'
        models.append(dict(LpD=lpd, path=str(path.resolve()),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           features=len(table), targets=list(table.columns)))
        directories.update({(lpd, vr):directory for vr in selection.vrs})
    for window in ('overall_reduced', 'sinusoidal', 'sinusoidal_evaluation'):
        data_selection('two_tandem', fit_window_case=configs / 'fit_window/two_tandem' / (window + '.json'))
    return directories, dict(cylinder_type='two_tandem', dynamics_case=3, parameterization='vr',
                             model_origin='manually entered theoretical coefficients; no training performed',
                             vr_list=list(selection.vrs), lpd_list=list(selection.lpds),
                             models=models, library_configurations=libraries,
                             report_window_label='overall_reduced (existing model folder label)',
                             config_root=str(configs.resolve()))


def verify_unchanged(manifest: dict) -> None:
    """Reject a changed source CSV, preserving the original file without rewriting it."""
    for model in manifest['models']:
        if hashlib.sha256(Path(model['path']).read_bytes()).hexdigest() != model['sha256']:
            raise ValueError(f"Manual model changed during evaluation: {model['path']}")


def write_status(path: Path, phase: str, **details) -> None:
    """Write this evaluator's status separately from the supervising queue status."""
    payload = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(payload, allow_nan=False), flush=True)


def parse_args() -> argparse.Namespace:
    """Parse manual model selection, output logging and optional predecessor checks."""
    parser = argparse.ArgumentParser(
        description='Run full assessment of five manually specified tandem Case 3 VR models.',
    )
    parser.add_argument(
        '--model_root',
        type=Path,
        default=DEFAULT_ROOT,
        help='The theoretical root containing LpD=<value>/fit_window=overall_reduced/model.csv.',
    )
    parser.add_argument(
        '--config_root',
        type=Path,
        default=Path(os.environ.get('SINDY_THEORETICAL_CONFIG_ROOT', str(REPO / 'configs'))),
        help='Library, filtering, differentiation and evaluation-window configurations.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path(os.environ['SINDY_THEORETICAL_OUTPUT']) if 'SINDY_THEORETICAL_OUTPUT' in os.environ else None,
        help='Status/hash/result manifests; numerical diagnostics stay under the theoretical model tree.',
    )
    parser.add_argument(
        '--after_study',
        type=Path,
        default=Path(os.environ['SINDY_THEORETICAL_AFTER_STUDY']) if 'SINDY_THEORETICAL_AFTER_STUDY' in os.environ else None,
        help='Require the predecessor study to have finished all stages, including with recorded errors.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate models/windows and print a manifest without evaluating or writing outputs.',
    )
    return parser.parse_args()


def main() -> int:
    """Gate on predecessor completion, run existing full suites, and verify source hashes."""
    args = parse_args()
    os.chdir(REPO)
    directories, manifest = model_selection(args.model_root.resolve(), args.config_root.resolve())
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if args.after_study is not None:
        predecessor = json.loads(args.after_study.read_text(encoding='utf-8'))
        if predecessor['phase'] not in ('Finished', 'FinishedWithErrors'):
            raise RuntimeError('The overnight study has not completed all stages; theoretical evaluation will not start.')
    output = args.output_dir or REPO / 'state/theoretical_evaluation_runs' / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%SZ')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    status = output / 'evaluation_status.json'
    write_status(status, 'Running', model_count=5, trajectory_count=50)
    configs = args.config_root.resolve()
    windows = configs / 'fit_window/two_tandem'
    try:
        # Backfill plots from any in-flight overnight suite that imported the old layout.
        from refresh_tandem_coefficient_heatmaps import refresh_combined_heatmaps
        refresh_combined_heatmaps(REPO / 'results/oscillating-cylinder-tandem-viv',
                                  output / 'preceding_coefficient_plot_refresh.json')
        result = evaluate_two_tandem_suite(
            dynamics_case=3, parameterization='vr',
            differentiator_case=configs / 'differentiator/finite_difference_basic.json',
            filtering_case=configs / 'filtering/butterworth_two_tandem_flim2.json',
            state_feature_library_case=configs / 'feature_library/polynomial_1.json',
            input_feature_library_case=configs / 'feature_library/polynomial_2.json',
            optimizer_case='theoretical', fit_window_case=windows / 'overall_reduced.json',
            derivative_window_cases=[windows / (name + '.json') for name in
                                     ('overall_reduced', 'sinusoidal', 'sinusoidal_evaluation')],
            integral_window_case=windows / 'sinusoidal_evaluation.json',
            spectral_window_cases=[windows / 'sinusoidal.json', windows / 'sinusoidal_evaluation.json'],
            vr_list=manifest['vr_list'], lpd_list=manifest['lpd_list'],
            model_directories=directories, weight_heatmap=True, method='RK45', max_state_factor=10.,
        )
        verify_unchanged(manifest)
        (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        write_status(status, 'FinishedWithErrors' if result['failures'] else 'Finished',
                     model_coefficients_unchanged=True, failure_count=len(result['failures']))
        return 1 if result['failures'] else 0
    except BaseException as error:
        write_status(status, 'Failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
