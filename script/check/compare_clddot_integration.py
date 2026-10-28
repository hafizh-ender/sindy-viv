"""Reconstruct lift rate by integrating both saved lift-acceleration estimates.

Read compare_clddot_methods.py outputs, use identical initial lift rates and
cumulative trapezoids in tau, and save reference/reconstruction/error plots.
This measures reintegration consistency against numerical CLdot, not physical
accuracy against an independent measured lift-rate channel. No model is fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sindy_viv.case_bryan.evaluate_sindy.derivative_assessment import reconstruct_from_derivatives
from sindy_viv.case_bryan.plot import create_subplots
from sindy_viv.metrics import normalized_root_mean_squared_error


def parse_args() -> argparse.Namespace:
    """Choose saved derivative comparisons and the reconstruction report destination."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        '--source_dir',
        type=Path,
        default=Path('state/clddot_comparison'),
        help='Directory containing comparison CSVs and summary.json. Default: state/clddot_comparison.',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('state/clddot_integration'),
        help='Destination for reconstruction plots, CSVs and JSON. Default: state/clddot_integration.',
    )
    parser.add_argument(
        '--zoom_cycles',
        type=float,
        default=2.,
        help='Centered zoom duration in natural periods (tau units). Default: 2.',
    )
    args = parser.parse_args()
    if not np.isfinite(args.zoom_cycles) or args.zoom_cycles <= 0:
        parser.error('--zoom_cycles must be positive and finite.')
    return args


def reconstruction_statistics(reference: np.ndarray, reconstruction: np.ndarray) -> list[dict]:
    """Report each channel's RMSE/std(reference), absolute errors and final drift."""
    errors = reconstruction-reference
    nrmse = normalized_root_mean_squared_error(reconstruction, reference, reduction='none')
    return [dict(
        cylinder=cylinder,
        nrmse=float(nrmse[cylinder]),
        rmse=float(np.sqrt(np.mean(errors[:, cylinder]**2))),
        maximum_absolute_error=float(np.max(np.abs(errors[:, cylinder]))),
        mean_error=float(np.mean(errors[:, cylinder])),
        initial_error=float(errors[0, cylinder]),
        final_error=float(errors[-1, cylinder]),
    ) for cylinder in range(reference.shape[1])]


def plot_reconstruction(
    tstar: np.ndarray, reference: np.ndarray, direct: np.ndarray, repeated: np.ndarray,
    title: str, vr: int, zoom_cycles: float, destination: Path,
) -> None:
    """Show reconstructed lift rates and signed reference errors over full and zoom windows."""
    midpoint = .5*(tstar[0]+tstar[-1])
    halfspan = min(.5*zoom_cycles*vr, .5*(tstar[-1]-tstar[0]))
    zoom = (tstar >= midpoint-halfspan) & (tstar <= midpoint+halfspan)
    if np.count_nonzero(zoom) < 2:
        raise ValueError('The zoom must contain at least two samples.')
    cylinders = reference.shape[1]
    figure, axes = create_subplots(
        2*cylinders, 2, sharex=False, squeeze=False,
        figsize=(10.5, 3.2*cylinders+1.),
    )
    figure.suptitle(title, fontsize=12)
    for cylinder in range(cylinders):
        for column, region in enumerate((slice(None), zoom)):
            overlay, error = axes[2*cylinder:2*cylinder+2, column]
            x = tstar[region]
            overlay.plot(x, reference[region, cylinder], color='.65', lw=1.8,
                         label=r'Reference $\dot C_L$')
            overlay.plot(x, direct[region, cylinder], color='black', lw=.9,
                         label=r'Integral of direct $\ddot C_L$')
            overlay.plot(x, repeated[region, cylinder], color='red', ls='--', lw=.9,
                         label=r'Integral of $D(\dot C_L)$')
            error.plot(x, (direct-reference)[region, cylinder], color='black', lw=.8)
            error.plot(x, (repeated-reference)[region, cylinder], color='red', ls='--', lw=.8)
            error.axhline(0., color='.6', lw=.5)
            if column == 0:
                overlay.set_ylabel(rf'$\dot C_{{L{cylinder}}}$')
                error.set_ylabel(rf'$\dot C_{{L{cylinder}}}$ error')
            if cylinder == 0:
                overlay.set_title('Evaluation window' if column == 0 else 'Zoom', fontsize=10)
            for axis in (overlay, error):
                axis.set_xlim(x[0], x[-1])
                axis.grid(alpha=.18, lw=.5)
                axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useMathText=True)
            overlay.tick_params(labelbottom=False)
            error.set_xlabel(r'$t^*$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.005),
                  ncol=3, fontsize=9, frameon=False)
    figure.savefig(destination, dpi=200, bbox_inches='tight')
    plt.close(figure)


def compare_method(dataset: dict, name: str, method: dict, args: argparse.Namespace) -> dict:
    """Integrate both derivative histories from one initial value and save their comparison."""
    source = args.source_dir / Path(method['csv']).name
    table = np.genfromtxt(source, delimiter=',', names=True, dtype=np.float64)
    time, tstar = table['tau'], table['tstar']
    cylinders = len(method['statistics'])
    reference = np.column_stack([table[f'CLdot_{i}'] for i in range(cylinders)])
    accelerations = {
        route: np.column_stack([table[f'CLddot_{route}_{i}'] for i in range(cylinders)])
        for route in ('direct', 'repeated')
    }
    if not np.all(np.isfinite(reference)):
        raise ValueError('The lift-rate reference must be finite.')
    reconstructions, solvers, statistics = {}, {}, {}
    for route, acceleration in accelerations.items():
        reconstructed, solver = reconstruct_from_derivatives(
            time, acceleration, reference[0].copy(), method='trapezoid',
        )
        np.testing.assert_array_equal(reconstructed[0], reference[0])
        reconstructions[route], solvers[route] = reconstructed, solver
        statistics[route] = reconstruction_statistics(reference, reconstructed)
    vr, lpd = dataset['VR'], dataset['LpD']
    parameters = rf'$V_R={vr}$' if lpd is None else rf'$L/D={lpd:g},\quad V_R={vr}$'
    method_name = 'Finite differences' if name == 'finite_difference' else 'Savitzky-Golay'
    stem = args.output_dir / (source.stem+'_CLdot_reconstruction')
    plot_reconstruction(
        tstar, reference, reconstructions['direct'], reconstructions['repeated'],
        f'Lift-rate reconstruction: {method_name}\n{parameters}; '
        rf'$t^*={tstar[0]:g}\ldots{tstar[-1]:g}$ (sinusoidal evaluation)',
        vr, args.zoom_cycles, Path(str(stem)+'.png'),
    )
    columns, header = [tstar, time], ['tstar', 'tau']
    for cylinder in range(cylinders):
        columns.append(reference[:, cylinder])
        header.append(f'CLdot_reference_{cylinder}')
        for route in ('direct', 'repeated'):
            columns.extend((reconstructions[route][:, cylinder],
                            reconstructions[route][:, cylinder]-reference[:, cylinder]))
            header.extend((f'CLdot_reconstructed_{route}_{cylinder}', f'error_{route}_{cylinder}'))
    np.savetxt(str(stem)+'.csv', np.column_stack(columns), delimiter=',',
               header=','.join(header), comments='', fmt='%.17g')
    for cylinder in range(cylinders):
        print(f"{dataset['cylinder_type']} {name}, cylinder {cylinder}: "
              f"direct NRMSE={statistics['direct'][cylinder]['nrmse']:.7g}; "
              f"repeated NRMSE={statistics['repeated'][cylinder]['nrmse']:.7g}", flush=True)
    return dict(source_csv=str(source), differentiator_config=method['differentiator_config'],
                statistics=statistics, solvers=solvers, initial_CLdot=reference[0].tolist(),
                figure=str(stem)+'.png', csv=str(stem)+'.csv')


def main() -> None:
    """Read the saved comparison selections and write standalone reintegration reports."""
    args = parse_args()
    source_summary = args.source_dir / 'summary.json'
    datasets = json.loads(source_summary.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for dataset in datasets:
        report = {key: value for key, value in dataset.items() if key != 'methods'}
        report.pop('difference_definition', None)
        report.pop('normalization', None)
        report.update(
            source_summary=str(source_summary),
            integration_method='SciPy cumulative_trapezoid in tau; no resets or drift/offset corrections',
            reference='Numerical first derivative of the same Butterworth-filtered lift for this method',
            initial_condition='Both routes use the reference CLdot at the first evaluation sample',
            nrmse_definition='RMSE(reconstruction-reference) / population standard deviation(reference)',
            limitation='Reintegration consistency only; the numerical CLdot reference is not independent ground truth',
            methods={},
        )
        for name, method in dataset['methods'].items():
            report['methods'][name] = compare_method(dataset, name, method, args)
        reports.append(report)
    destination = args.output_dir / 'summary.json'
    destination.write_text(json.dumps(reports, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Saved four reconstruction figures and {destination}.', flush=True)


if __name__ == '__main__':
    main()
