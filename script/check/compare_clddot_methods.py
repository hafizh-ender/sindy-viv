"""Compare direct lift acceleration with differentiating lift rate on measured data.

Defaults: single VR=7 and tandem L/D=1.5, VR=9, both cylinders, Butterworth
filtering, finite differences and SG base presets, sinusoidal_evaluation windows.
This manual diagnostic saves PNGs, CSVs and JSON; it never fits SINDy models.
Run from the repository root with the project Python environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sindy_viv.case_bryan.analysis.data_windows import data_selection, load_trajectory
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.plot import create_subplots
from sindy_viv.differentiation import perform_second_derivative
from sindy_viv.utils import initiate_object, load_config


def parse_args() -> argparse.Namespace:
    """Select the two datasets, plot zoom and output directory for this manual check."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        '--cylinder_type',
        choices=('both', 'single', 'two_tandem'),
        default='both',
        help='Datasets to compare. Default: both.',
    )
    parser.add_argument(
        '--single_vr',
        type=int,
        default=7,
        help='Single-cylinder reduced velocity. Default: 7.',
    )
    parser.add_argument(
        '--tandem_vr',
        type=int,
        default=9,
        help='Tandem reduced velocity. Default: 9.',
    )
    parser.add_argument(
        '--lpd',
        type=float,
        default=1.5,
        help='Tandem L/D. Default: 1.5.',
    )
    parser.add_argument(
        '--zoom_cycles',
        type=float,
        default=2.,
        help='Zoom duration in natural periods (tau units), centered in the window. Default: 2.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('state/clddot_comparison'),
        help='Destination for comparison PNGs, CSVs and summary JSON. Default: state/clddot_comparison.',
    )
    args = parser.parse_args()
    if not np.isfinite(args.zoom_cycles) or args.zoom_cycles <= 0:
        parser.error('--zoom_cycles must be positive and finite.')
    return args


def comparison_statistics(direct: np.ndarray, repeated: np.ndarray) -> list[dict]:
    """Describe method disagreement per cylinder; neither method is ground truth."""
    reports = []
    for cylinder in range(direct.shape[1]):
        first, second = direct[:, cylinder], repeated[:, cylinder]
        difference = first - second
        rms = float(np.sqrt(np.mean(first**2)))
        difference_rms = float(np.sqrt(np.mean(difference**2)))
        correlation = (float(np.corrcoef(first, second)[0, 1])
                       if np.std(first) > 0 and np.std(second) > 0 else None)
        reports.append(dict(
            cylinder=cylinder,
            direct_rms=rms,
            repeated_rms=float(np.sqrt(np.mean(second**2))),
            difference_rms=difference_rms,
            relative_rms_difference_percent=100*difference_rms/rms if rms else None,
            maximum_absolute_difference=float(np.max(np.abs(difference))),
            correlation=correlation,
        ))
    return reports


def plot_comparison(
    tstar: np.ndarray, direct: np.ndarray, repeated: np.ndarray,
    title: str, vr: int, zoom_cycles: float, destination: Path,
) -> dict:
    """Plot overlays and direct-minus-repeated differences on full and zoom intervals."""
    cylinders = direct.shape[1]
    midpoint = .5 * (tstar[0] + tstar[-1])
    halfspan = min(.5*zoom_cycles*vr, .5*(tstar[-1]-tstar[0]))
    zoom = (tstar >= midpoint-halfspan) & (tstar <= midpoint+halfspan)
    if np.count_nonzero(zoom) < 2:
        raise ValueError('The requested zoom contains fewer than two samples.')
    figure, axes = create_subplots(
        2*cylinders, 2, sharex=False, squeeze=False,
        figsize=(10.5, 3.2*cylinders+1.0),
    )
    figure.suptitle(title, fontsize=12)
    for cylinder in range(cylinders):
        for column, region in enumerate((slice(None), zoom)):
            overlay, difference = axes[2*cylinder:2*cylinder+2, column]
            x = tstar[region]
            overlay.plot(x, direct[region, cylinder], color='black', lw=.9,
                         label='Direct second derivative')
            overlay.plot(x, repeated[region, cylinder], color='red', ls='--', lw=.9,
                         label='Derivative of lift rate')
            difference.plot(x, (direct-repeated)[region, cylinder], color='#2465a4', lw=.8)
            difference.axhline(0., color='.55', lw=.5)
            if column == 0:
                overlay.set_ylabel(rf'$d^2 C_{{L{cylinder}}}/d\tau^2$')
                difference.set_ylabel(rf'$\Delta\ddot{{C}}_{{L{cylinder}}}$')
            if cylinder == 0:
                overlay.set_title('Evaluation window' if column == 0 else 'Zoom', fontsize=10)
            for axis in (overlay, difference):
                axis.set_xlim(x[0], x[-1])
                axis.grid(alpha=.18, lw=.5)
                axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useMathText=True)
            overlay.tick_params(labelbottom=False)
            difference.set_xlabel(r'$t^*$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.005),
                  ncol=2, fontsize=9, frameon=False)
    figure.savefig(destination, dpi=200, bbox_inches='tight')
    plt.close(figure)
    return dict(zoom_tstar_region=[float(tstar[zoom][0]), float(tstar[zoom][-1])])


def compare_dataset(
    cylinder_type: str, vr: int, lpd: float | None, output: Path, zoom_cycles: float,
) -> dict:
    """Load and filter a full record, compare both methods, then crop/save the evaluation window."""
    window_path = Path('configs/fit_window') / cylinder_type / 'sinusoidal_evaluation.json'
    selection = data_selection(
        cylinder_type, [vr], None if lpd is None else [lpd], fit_window_case=window_path,
    )
    names = ['C_{L0}'] if cylinder_type == 'single' else ['C_{L0}', 'C_{L1}']
    time, raw = load_trajectory(selection, vr, lpd, names)
    filter_path = Path('configs/filtering') / f'butterworth_{cylinder_type}_flim2.json'
    filter_config = load_config(filter_path)
    values = initiate_object(**filter_config)(raw, time)
    del raw
    bounds = selection.bounds[(lpd, vr)]
    region = select_time_region(time, vr, *bounds)
    tstar = time[region]*vr
    descriptor = f'single_VR={vr}' if lpd is None else f'two_tandem_LpD={lpd:g}_VR={vr}'
    parameter_title = rf'$V_R={vr}$' if lpd is None else rf'$L/D={lpd:g},\quad V_R={vr}$'
    report = dict(
        cylinder_type=cylinder_type, VR=vr, LpD=lpd,
        fit_window_case=str(window_path),
        requested_tstar_region=list(bounds),
        actual_tstar_region=[float(tstar[0]), float(tstar[-1])],
        full_record_samples=len(time), evaluation_samples=len(tstar),
        time_coordinate='tau=t*/VR; first and second derivatives are with respect to tau',
        filtering_config=filter_config,
        processing='Filter once, differentiate full record, then crop to sinusoidal_evaluation.',
        difference_definition='direct second derivative minus derivative of first derivative',
        normalization='RMS difference divided by RMS direct, expressed as percent; not error against ground truth',
        methods={},
    )
    for method_name, config_name in (
        ('finite_difference', 'finite_difference_basic'),
        ('savitzky_golay', f'savitzky_golay_{cylinder_type}_base'),
    ):
        config_path = Path('configs/differentiator') / f'{config_name}.json'
        config = load_config(config_path)
        method = initiate_object(**config)
        print(f'{descriptor}: computing {method_name} on {len(time):,} full-record samples.', flush=True)
        first = method(values, time)
        direct = perform_second_derivative(time, values, method)[region].copy()
        repeated = method(first, time)[region].copy()
        if not np.all(np.isfinite(direct)) or not np.all(np.isfinite(repeated)):
            raise ValueError('Nonfinite derivative values in the requested evaluation window.')
        statistics = comparison_statistics(direct, repeated)
        stem = output / f'{descriptor}_{method_name}'
        method_title = ('Finite differences: direct d=2 vs repeated d=1'
                        if method_name == 'finite_difference'
                        else 'Savitzky-Golay: direct deriv=2 vs repeated deriv=1')
        zoom_metadata = plot_comparison(
            tstar, direct, repeated,
            f'{method_title}\n{parameter_title}; '
            rf'$t^*={tstar[0]:g}\ldots{tstar[-1]:g}$ (sinusoidal evaluation)',
            vr, zoom_cycles, Path(str(stem)+'.png'),
        )
        columns = [tstar, time[region]]
        header = ['tstar', 'tau']
        for cylinder in range(values.shape[1]):
            columns.extend((values[region, cylinder], first[region, cylinder],
                            direct[:, cylinder], repeated[:, cylinder],
                            direct[:, cylinder]-repeated[:, cylinder]))
            header.extend(f'{quantity}_{cylinder}' for quantity in
                          ('CL_filtered', 'CLdot', 'CLddot_direct', 'CLddot_repeated', 'difference'))
        np.savetxt(str(stem)+'.csv', np.column_stack(columns), delimiter=',',
                   header=','.join(header), comments='', fmt='%.17g')
        report['methods'][method_name] = dict(
            differentiator_config=config, statistics=statistics,
            figure=str(stem)+'.png', csv=str(stem)+'.csv', **zoom_metadata,
        )
        for item in statistics:
            print(f"  Cylinder {item['cylinder']}: RMS difference/direct RMS = "
                  f"{item['relative_rms_difference_percent']:.6g}%", flush=True)
        del first, direct, repeated
    return report


def main() -> None:
    """Run only this requested comparison and save reproducible settings/statistics."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for cylinder, vr, lpd in (
        ('single', args.single_vr, None), ('two_tandem', args.tandem_vr, args.lpd),
    ):
        if args.cylinder_type in ('both', cylinder):
            reports.append(compare_dataset(cylinder, vr, lpd, args.output_dir, args.zoom_cycles))
    summary_path = args.output_dir / 'summary.json'
    summary_path.write_text(json.dumps(reports, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Saved comparison figures, CSVs and {summary_path}.', flush=True)


if __name__ == '__main__':
    main()
