"""Generate the three figures used in the marked single-cylinder results section.

From sindy-viv-ver2:
    .\.venv\Scripts\python.exe script/thesis/generate_single_structural_results_figures.py

Reads saved Case 3 fits and cached, complete IVP traces. No fitting/integration.
Writes individual SVG/450-dpi PNG files, one three-page vector PDF for direct
LaTeX inclusion, coefficient CSVs, and a source/checksum/NRMSE manifest.
Figure sizes are physical millimetres. Font sizes are TeX points (1/72.27 in),
converted to Matplotlib points (1/72 in) so native inclusion preserves their size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path, help='Default: img/04_results/single_structural')
    parser.add_argument('--trace-dir', type=Path, help='Default: img/04_results/data/reference/data')
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = (args.output_dir or repo/'img/04_results/single_structural').resolve()
    traces = (args.trace_dir or repo/'img/04_results/data/reference/data').resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'sindy-viv/matplotlib-thesis-results-cache'))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    pt = lambda value: value*72/72.27
    font = font_manager.findfont('cmr10', fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
        'text.usetex': False, 'font.size': pt(11), 'axes.labelsize': pt(12),
        'axes.titlesize': pt(12), 'xtick.labelsize': pt(10.5), 'ytick.labelsize': pt(10.5),
        'legend.fontsize': pt(10.5), 'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'xtick.major.width': .6,
        'ytick.major.width': .6, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42, 'svg.fonttype': 'path', 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.dpi': 450,
    })
    sources: dict[str, str] = {}
    figures: list[dict] = []

    def source(path: Path) -> Path:
        path = path.resolve()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def load_model(mode: str, phase: str, vr: int | None = None):
        root = repo/'results/oscillating-cylinder-viv/dynamics_case_3/filtering=butterworth_single_flim2/finite_difference_basic'/mode
        root /= 'polynomial_1' if mode == 'individual' else 'state_polynomial_1/input_polynomial_2'
        root /= 'stlsq_vanilla_basic'
        paths = [p for p in root.rglob('model.csv') if f'fit_window={phase}' in p.parts]
        if vr is not None:
            paths = [p for p in paths if f'VR = {vr}' in p.parts]
        if len(paths) != 1:
            raise ValueError(f'Expected one model for {(mode, phase, vr)}; found {len(paths)}')
        p = source(paths[0])
        meta = json.loads(source(p.parent/'training.json').read_text())
        settings = meta['configurations']['optimizer']['kwargs']
        if meta['fit_window']['name'] != phase or settings['alpha'] != 0 or settings['threshold'] != .1:
            raise ValueError(f'Unexpected fit metadata: {p}')
        return pd.read_csv(p, index_col=0), p, meta

    def save(fig, stem: str, **metadata):
        fig.canvas.draw()
        fig.savefig(out/f'{stem}.svg')
        fig.savefig(out/f'{stem}.png', dpi=450)
        book.savefig(fig)
        figures.append(dict(stem=stem, pdf_page=len(figures)+1,
                            width_mm=float(fig.get_figwidth()*25.4),
                            height_mm=float(fig.get_figheight()*25.4), **metadata))
        plt.close(fig)

    phases = ['overall_reduced', 'sinusoidal']
    coefficients = {}
    training = {}
    for phase in phases:
        models = [load_model('individual', phase, vr) for vr in range(3, 9)]
        coefficients[phase] = pd.DataFrame({vr: model[r'\ddot{h}_0'] for vr, (model, _, _) in zip(range(3, 9), models)})
        training[phase] = [trajectory for _, _, meta in models for trajectory in meta['trajectories']]
        if coefficients[phase].index.tolist() != ['1', 'h_0', r'\dot{h}_0', 'C_{L0}']:
            raise ValueError('Expected the complete four-term single-cylinder linear library')
    limit = max(float(np.abs(frame.to_numpy()).max()) for frame in coefficients.values())
    parameterized, parameterized_path, _ = load_model('parametric_vr', 'sinusoidal')
    a = float(parameterized.loc['1 h_0', r'\ddot{h}_0'])
    b = float(parameterized.loc['V_R^2 C_{L0}', r'\ddot{h}_0'])
    summary_paths = [p for p in parameterized_path.parent.rglob('integral_suite_summary.json')
                     if 'evaluation_window=sinusoidal_evaluation' in p.parts]
    if len(summary_paths) != 1:
        raise ValueError('Expected one later-window IVP summary for the sinusoidal parameterized fit')
    summary = json.loads(source(summary_paths[0]).read_text())
    if summary['failures'] or len(summary['records']) != 6 or not all(r['complete'] for r in summary['records']):
        raise ValueError('Cannot report a full-window maximum from incomplete/missing IVPs')
    worst = max(summary['records'], key=lambda r: r['values']['h_0'])

    def trace(tag: str, expected_model: Path | None = None):
        meta = json.loads(source(traces/f'{tag}.json').read_text())
        if not meta['complete']:
            raise ValueError(f'Incomplete cached IVP: {tag}')
        if expected_model is not None and Path(meta['model_csv']).resolve() != expected_model.resolve():
            raise ValueError(f'Wrong training model in cached IVP: {tag}')
        source(Path(meta['model_csv']))
        with np.load(source(traces/f'{tag}.npz')) as data:
            result = {key: data[key].copy() for key in data.files}
        result['metadata'] = meta
        return result

    with PdfPages(out/'single_structural_figures.pdf', metadata={'Title': 'Single-cylinder structural results'}) as book:
        for phase in phases:
            frame = coefficients[phase]
            frame.to_csv(out/f'structural_coefficients_individual_ddoth_fit_{phase}.csv')
            # Native width 138.43 mm; square 14.224 mm cells; colourbar exactly as tall as the matrix.
            left, right, bottom, top, cell = .90, 1.19, .67, .40, .56
            width, height = left+6*cell+right, bottom+4*cell+top
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_axes([left/width, bottom/height, 6*cell/width, 4*cell/height])
            # Closed vector rectangles keep every cell uniform in SVG and PDF viewers.
            from matplotlib.patches import Rectangle
            from matplotlib.cm import ScalarMappable
            cmap = plt.get_cmap('RdBu_r', 2048)
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            for row, col in np.ndindex(frame.shape):
                colour = cmap(norm(frame.iat[row, col]))
                patch = Rectangle((col-.5, row-.5), 1, 1, facecolor=colour,
                                  edgecolor=colour, linewidth=.4, antialiased=False,
                                  rasterized=False, zorder=0,
                                  gid=f'heatmap_cell_{row}_{col}')
                ax.add_patch(patch)
            image = ScalarMappable(norm=norm, cmap=cmap)
            image.set_array(frame.to_numpy())
            ax.set_xlim(-.5, 5.5)
            ax.set_ylim(3.5, -.5)
            ax.set_aspect('equal')
            assert len(ax.images) == 0 and all(not p.get_rasterized() for p in ax.patches)
            ax.set_xticks(range(6), labels=range(3, 9), fontsize=pt(11))
            ax.set_yticks(range(4), labels=['$1$', '$h$', r'$\dot h$', '$C_L$'], fontsize=pt(11))
            ax.set_xlabel('$V_R$', fontsize=pt(11), labelpad=6)
            ax.set_ylabel('Library term', fontsize=pt(11), labelpad=6)
            ax.set_title(r'$\ddot h$: single cylinder', fontsize=pt(12), pad=8)
            ax.tick_params(which='major', length=0, pad=4)
            ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
            ax.grid(which='minor', color='#444444', linewidth=.3, alpha=1.)
            ax.tick_params(which='minor', bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(.65)
            values = frame.to_numpy()
            for row, col in np.ndindex(values.shape):
                value = values[row, col]
                ax.text(col, row, f'{value:.3f}', ha='center', va='center', fontsize=pt(10),
                        color='white' if abs(value)>.65*limit else '#111111')
            cax = fig.add_axes([(left+6*cell+.13)/width, bottom/height, .14/width, 4*cell/height])
            cb = fig.colorbar(image, cax=cax, ticks=[-limit, 0, limit])
            cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            cb.ax.tick_params(labelsize=pt(11), width=.5, pad=3)
            cb.set_label('Coefficient', fontsize=pt(11), labelpad=6)
            cb.outline.set_linewidth(.6)
            fig.canvas.draw()
            assert abs(ax.get_window_extent().height-cax.get_window_extent().height)<1e-6
            save(fig, f'structural_coefficients_individual_ddoth_fit_{phase}', training_phase=phase,
                 colour_limits=[-limit, limit], matrix_interpolation='none', matrix_renderer='Rectangle',
                 matrix_representation='vector_rectangles', matrix_rasterized=False, training=training[phase])

        width = 140/25.4
        fig, axes = plt.subplots(2, 2, figsize=(width, .75*width), sharex='col', sharey='row',
                                 gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(left=.155, right=.965, bottom=.22, top=.91, wspace=.18, hspace=.13)
        styles = {'Individual': ('#0072B2', '--'), 'Parameterized': ('#D55E00', '-.'), 'Theoretical': ('#009E73', ':')}
        for col, vr in enumerate((4, 5)):
            _, individual_path, _ = load_model('individual', 'sinusoidal', vr)
            entries = {'Individual': trace(f'single_individual_VR{vr}', individual_path),
                       'Parameterized': trace(f'single_parameterized_VR{vr}', parameterized_path),
                       'Theoretical': trace(f'single_theory_VR{vr}')}
            reference = entries['Theoretical']
            j = reference['variable_names'].tolist().index('h_0')
            axes[0, col].plot(reference['time_star'], reference['reference'][:, j], color='black', lw=1.1, label='Reference')
            for label, data in entries.items():
                k = data['variable_names'].tolist().index('h_0')
                np.testing.assert_array_equal(data['time_star'], reference['time_star'])
                np.testing.assert_array_equal(data['reference'][:, k], reference['reference'][:, j])
                colour, linestyle = styles[label]
                axes[0, col].plot(data['time_star'], data['prediction'][:, k], color=colour, ls=linestyle, lw=1.05, label=label)
                axes[1, col].plot(data['time_star'], data['prediction'][:, k]-data['reference'][:, k], color=colour, ls=linestyle, lw=1.)
            for row in (0, 1):
                axes[row, col].set_xlim(600, 650)
                axes[row, col].set_xticks([600, 625, 650])
                axes[row, col].yaxis.set_major_locator(ticker.MaxNLocator(3))
                axes[row, col].grid(alpha=.18, lw=.5)
            axes[1, col].ticklabel_format(axis='y', style='plain', useOffset=False)
            axes[0, col].tick_params(axis='x', bottom=False)
            axes[0, col].set_title(rf'$V_R={vr}$', loc='right', fontsize=pt(12), pad=6)
        axes[0, 0].set_ylabel('$h$', labelpad=4)
        axes[1, 0].set_ylabel(r'$\widehat h-h$', labelpad=3)
        fig.supxlabel('$t^*$', y=.105, fontsize=pt(12))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .015), ncol=4,
                   fontsize=pt(10.5), columnspacing=.8, handlelength=1.8, handletextpad=.4, borderaxespad=0.)
        save(fig, 'structural_IVP_VR4_VR5', fitted_training_phase='sinusoidal', validation_tstar=[600, 650], panel_height_ratio=[2, 1])

    for path, checksum in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'Source changed during generation: {path}')
    result = dict(figures=figures, pdf='single_structural_figures.pdf', font=font,
                  font_sizes_unit='TeX pt; converted to Matplotlib bp by multiplying by 72/72.27',
                  parameterized_equation=dict(training_phase='sinusoidal', restoring=a, lift_prefactor=b),
                  maximum_parameterized_displacement_nrmse_pct=100*worst['values']['h_0'],
                  maximum_parameterized_displacement_nrmse_VR=worst['VR'],
                  displacement_ivp_nrmse_pct={str(r['VR']): 100*r['values']['h_0'] for r in summary['records']},
                  source_sha256=sources)
    (out/'single_structural_generation_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Generated {len(figures)} figures in {out}')
    print(f'Maximum parameterized displacement NRMSE: {100*worst["values"]["h_0"]:.6f}% at VR={worst["VR"]}')


if __name__ == '__main__':
    main()
