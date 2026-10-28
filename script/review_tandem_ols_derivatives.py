"""Review existing tandem OLS lift-derivative errors without fitting any models.

Read complete sinusoidal-evaluation summaries for Cases 1/2/4 and Poly2/3/4.
Preserve cylinder-specific errors, write full-precision JSON and Markdown grids,
and draw square-cell comparison maps with consistent scales within each case.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from sindy_viv.case_bryan.analysis.data_windows import save_figure
from sindy_viv.case_bryan.analysis.error_heatmaps import style_heatmap
from sindy_viv.case_bryan.plot import create_subplots


REPO = Path(__file__).resolve().parents[1]
CASES = (1, 2, 4)
DEGREES = (2, 3, 4)
LPDS = (1.5, 2., 3., 3.5, 4.)
VRS = tuple(range(5, 15))


def load_records(results_root):
    """Require all 900 finite lift errors with unique, complete parameter coverage."""
    records, sources = [], []
    expected = {(lpd, vr) for lpd in LPDS for vr in VRS}
    for case in CASES:
        for degree in DEGREES:
            base = results_root / 'oscillating-cylinder-tandem-viv' / f'dynamics_case_{case}'
            base /= ('filtering=butterworth_two_tandem_flim2/finite_difference_basic/'
                     f'individual/polynomial_{degree}/ols')
            matches = list(base.glob('evaluation/summary/reference=finite_difference_basic/'
                                    'evaluation_window=sinusoidal_evaluation/**/derivative_suite_summary.json'))
            if len(matches) != 1:
                raise ValueError(f'Expected one summary for Case {case}, Poly{degree}: {matches}')
            source = matches[0]
            payload = json.loads(source.read_text())
            rows = payload['records']
            keys = [(r['LpD'], r['VR']) for r in rows]
            if payload['failures'] or len(keys) != len(expected) or set(keys) != expected:
                raise ValueError(f'Incomplete or duplicate parameter coverage: {source}')
            for row in rows:
                if not row['complete'] or row['requested_tstar_region'] != [1350., 1400.]:
                    raise ValueError(f'Unexpected scoring window or incomplete record: {source}')
                for cylinder in (0, 1):
                    prefix = r'\dot' if case == 1 else r'\ddot'
                    target = prefix + f'{{C}}_{{L{cylinder}}}'
                    value = row['values'][target]
                    if not np.isfinite(value) or value < 0:
                        raise ValueError(f'Invalid NRMSE: {source}, {target}')
                    records.append(dict(case=case, degree=degree, cylinder=cylinder,
                                        LpD=row['LpD'], VR=row['VR'], target=target,
                                        derivative_nrmse=value, metrics_path=row['metrics_path'],
                                        source_summary=str(source)))
            sources.append(str(source))
    return records, sources


def error_matrix(records, case, degree, cylinder):
    """Arrange one cylinder's exact errors as L/D rows and VR columns."""
    lookup = {(r['LpD'], r['VR']): r['derivative_nrmse'] for r in records
              if (r['case'], r['degree'], r['cylinder']) == (case, degree, cylinder)}
    return np.array([[lookup[lpd, vr] for vr in VRS] for lpd in LPDS])


def summarize(records):
    """Report equal-weight cylinder statistics and matched Poly3-to-Poly4 changes."""
    statistics, changes = [], []
    for case in CASES:
        for cylinder in (0, 1):
            for degree in DEGREES:
                values = error_matrix(records, case, degree, cylinder)
                statistics.append(dict(case=case, degree=degree, cylinder=cylinder,
                                       count=values.size, mean=float(values.mean()),
                                       median=float(np.median(values)), maximum=float(values.max()),
                                       below_one_percent=int((values < .01).sum())))
            third = error_matrix(records, case, 3, cylinder)
            fourth = error_matrix(records, case, 4, cylinder)
            changes.append(dict(case=case, cylinder=cylinder, count=third.size,
                                quartic_worse=int((fourth > third).sum()),
                                quartic_better=int((fourth < third).sum()),
                                quartic_equal=int((fourth == third).sum())))
    return statistics, changes


def plot_case(records, case, output):
    """Compare three degrees and both cylinders with one logarithmic NRMSE scale."""
    values = 100 * np.array([r['derivative_nrmse'] for r in records if r['case'] == case])
    positive = values[values > 0]
    norm = LogNorm(vmin=10.**np.floor(np.log10(positive.min())),
                   vmax=10.**np.ceil(np.log10(positive.max())))
    figure, axes = create_subplots(2, 3, figsize=(21, 8.5), squeeze=False, sharex=False)
    try:
        for cylinder in (0, 1):
            for column, degree in enumerate(DEGREES):
                ax = axes[cylinder, column]
                matrix = 100 * error_matrix(records, case, degree, cylinder)
                image = ax.imshow(matrix, norm=norm, cmap='viridis', aspect='equal')
                ax.set_xticks(range(len(VRS)), [str(vr) for vr in VRS])
                ax.set_yticks(range(len(LPDS)), [f'{lpd:g}' for lpd in LPDS])
                ax.set_xlabel(r'$V_R$')
                ax.set_ylabel(r'$L/D$')
                ax.set_title(f'Poly{degree}, cylinder {cylinder}', fontsize=12)
                style_heatmap(ax, image, matrix)
        colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.85, pad=.015)
        colorbar.set_label('Lift-derivative NRMSE (%; logarithmic scale)')
        colorbar.outline.set_linewidth(.8)
        stem = output / f'case_{case}_poly2_to_poly4_derivative'
        figure.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
        save_figure(figure, stem.with_suffix('.png'))
    finally:
        plt.close(figure)


def write_report(records, statistics, changes, output):
    """Write summary statistics and every L/D-VR error as percentage tables."""
    lines = ['# Tandem OLS lift-derivative review', '',
             'Existing individual OLS fits, Butterworth filtering, finite-difference derivatives.',
             'Sinusoidal training: t*=600-1350. Evaluation: t*=1350-1400.',
             'Table errors and plot annotations are percentages; JSON errors are ratios.',
             'Case 1 assesses CLdot; Cases 2/4 assess CLddot. Different target derivatives '
             'prevent interpreting their relative NRMSE alone as a common physical accuracy ranking.',
             'No new fits or IVPs were run. Complete Poly1/Poly5 summaries are unavailable.', '',
             '| Case | Cylinder | Degree | Mean NRMSE (%) | Median NRMSE (%) | Below 1% |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in statistics:
        lines.append(f"| {row['case']} | {row['cylinder']} | {row['degree']} | "
                     f"{100*row['mean']:.4f} | {100*row['median']:.4f} | {row['below_one_percent']}/50 |")
    lines += ['', '## Matched Poly4 versus Poly3 evaluation errors', '',
              '| Case | Cylinder | Poly4 worse | Poly4 better | Equal |',
              '|---:|---:|---:|---:|---:|']
    for row in changes:
        lines.append(f"| {row['case']} | {row['cylinder']} | {row['quartic_worse']}/50 | "
                     f"{row['quartic_better']}/50 | {row['quartic_equal']}/50 |")
    for case in CASES:
        lines += ['', f'## Case {case}', '',
                  f'![Case {case} comparison](case_{case}_poly2_to_poly4_derivative.png)', '']
        for cylinder in (0, 1):
            for degree in DEGREES:
                lines += [f'### Cylinder {cylinder}, Poly{degree}', '',
                          '| L/D | ' + ' | '.join(str(vr) for vr in VRS) + ' |',
                          '|---:|' + '---:|' * len(VRS)]
                matrix = error_matrix(records, case, degree, cylinder)
                for lpd, values in zip(LPDS, matrix):
                    lines.append(f'| {lpd:g} | ' + ' | '.join(f'{100*v:.4g}' for v in values) + ' |')
                lines.append('')
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    """Validate saved summaries, then export a reproducible numerical and visual review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results_root', type=Path, default=REPO / 'results',
        help='Root containing the saved tandem model evaluation summaries.',
    )
    parser.add_argument(
        '--output_dir', type=Path,
        help='Review directory; defaults to state/tandem_ols_derivative_review/<UTC timestamp>.',
    )
    args = parser.parse_args()
    records, sources = load_records(args.results_root)
    statistics, changes = summarize(records)
    output = args.output_dir or REPO / 'state/tandem_ols_derivative_review' / datetime.now(
        timezone.utc).strftime('%Y%m%d-%H%M%SZ')
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(errors_are_ratios=True, derivative_target=.01,
                   evaluation_tstar=[1350., 1400.], degrees=list(DEGREES),
                   records=records, statistics=statistics, matched_changes=changes,
                   source_summaries=sources)
    (output / 'comparison.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    for case in CASES:
        plot_case(records, case, output)
    write_report(records, statistics, changes, output)
    print(f'Reviewed {len(records)} lift-derivative errors. Report: {output / "report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
