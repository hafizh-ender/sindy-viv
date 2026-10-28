"""Plot computed FFT amplitudes and phases; frequency numerics live in analysis.frequency."""

from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from sindy_viv.case_bryan.plot import create_subplots, fft_figure_size, LINE_WIDTH, select_plot_variables
from sindy_viv.case_bryan.analysis.frequency import FrequencyAnalysis, compute_frequency_analysis


def _components(component):
    """Validate the requested FFT component and return its ordered panel names."""
    if component not in ('amplitude', 'phase', 'both'):
        raise ValueError('component must be amplitude, phase, or both.')
    return ('amplitude', 'phase') if component == 'both' else (component,)


def _limits(frequencies, xlim):
    """Return validated display limits and a native-bin slice including their edge bins.

    Reject nonfinite/reversed limits and a display starting beyond Nyquist;
    the slice affects rendering only, not dominant-frequency selection.
    """
    limits = (0., min(1., frequencies[-1])) if xlim is None else tuple(xlim)
    if len(limits) != 2 or not np.all(np.isfinite(limits)) or not 0 <= limits[0] < limits[1]:
        raise ValueError('Frequency limits must be finite, increasing and nonnegative.')
    if limits[0] >= frequencies[-1]:
        raise ValueError(f'Frequency limits start at or above the Nyquist frequency {frequencies[-1]:g}.')
    visible = slice(max(0, np.searchsorted(frequencies, limits[0])-1),
                    min(len(frequencies), np.searchsorted(frequencies, limits[1], side='right')+1))
    return limits, visible


def plot_frequency_analysis(
    tstar: np.ndarray, data: np.ndarray, variable_names: Sequence[str], *,
    xlim: Sequence[float] | None = None, title: str = '',
    plot_variables: Sequence[str] | None = None, plot_cylinders: Sequence[int] | None = None,
    fft_window: str = 'rectangular', component: str = 'both',
    analysis: FrequencyAnalysis | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot FFT amplitude or phase in square panels without changing the FFT numerics.

    CLI callers save amplitude and phase separately. component='both' retains the
    two-column Python API for existing callers. Each channel's full-spectrum
    dominant bin controls its annotations, even when the display is zoomed.
    The caller owns the returned figure and must close it.
    """
    tstar, data = np.asarray(tstar, dtype=np.float64), np.asarray(data)
    if not variable_names or data.shape != (len(tstar), len(variable_names)):
        raise ValueError('FFT data must have one aligned column per variable name.')
    selected = select_plot_variables(variable_names, plot_variables, plot_cylinders)
    data = data[:, [list(variable_names).index(name) for name in selected]]
    if analysis is None:
        analysis = compute_frequency_analysis(tstar, data, selected, fft_window=fft_window)
    elif (tuple(analysis.variable_names) != tuple(selected) or analysis.sample_count != len(tstar)
          or analysis.fft_window != fft_window or tuple(analysis.tstar_region) != (float(tstar[0]), float(tstar[-1]))):
        raise ValueError('Precomputed FFT must match the selected channels, time interval and FFT window.')
    components = _components(component)
    limits, visible = _limits(analysis.frequencies, xlim)
    figure, axes = create_subplots(len(selected), len(components), squeeze=False,
        figsize=fft_figure_size(len(selected), len(components)))
    try:
        heading = 'Frequency Analysis' if component == 'both' else 'Frequency '+component.capitalize()
        figure.suptitle(heading+('\n'+title if title else ''), fontsize=14)
        for row, name in enumerate(selected):
            peak = int(analysis.peak_indices[row])
            for column, kind in enumerate(components):
                ax = axes[row, column]
                values = analysis.amplitudes if kind == 'amplitude' else analysis.phases
                ax.plot(analysis.frequencies[visible], values[visible, row], linewidth=LINE_WIDTH)
                ax.set(xlim=limits, ylabel=rf'$|{name}|$' if kind=='amplitude' else rf'$\phi_{{{name}}}$')
                ax.set_box_aspect(1)
                if kind == 'amplitude':
                    ax.margins(y=.2)
                else:
                    ax.set_ylim(-np.pi, np.pi)
                    ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],
                                  [r'$-\pi$',r'$-\pi/2$','$0$',r'$\pi/2$',r'$\pi$'])
                if row == 0:
                    ax.set_title(kind.capitalize())
                if row == len(selected)-1:
                    ax.set_xlabel(r'$f^*=fD/U_\infty$')
                if peak < 0:
                    label = 'No resolved oscillation' if kind=='amplitude' else 'Phase undefined'
                elif kind == 'amplitude':
                    label = rf'$f^*_\mathrm{{max}}={analysis.frequencies[peak]:.4f}$'
                else:
                    label = rf'$\phi(f^*_\mathrm{{max}})={analysis.phases[peak,row]/np.pi:.4f}\pi$'
                ax.text(.95,.95,label,transform=ax.transAxes,fontsize=10,va='top',ha='right',
                        bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
                if peak >= 0 and limits[0] <= analysis.frequencies[peak] <= limits[1]:
                    ax.plot(analysis.frequencies[peak],values[peak,row],'o',color='red',markersize=3)
        return figure, axes
    except BaseException:
        plt.close(figure)
        raise


def plot_frequency_comparison(
    prediction: FrequencyAnalysis, reference: FrequencyAnalysis | None = None, *,
    title: str = '', xlim: Sequence[float] | None = None, component: str = 'both',
) -> tuple[Figure, np.ndarray]:
    """Plot precomputed model/reference amplitude or phase in square panels."""
    components = _components(component)
    limits, visible = _limits(prediction.frequencies, xlim)
    names = prediction.variable_names
    figure, axes = create_subplots(len(names),len(components),squeeze=False,
        figsize=fft_figure_size(len(names), len(components)))
    try:
        for row,name in enumerate(names):
            for column,kind in enumerate(components):
                ax = axes[row,column]
                for spectrum,label,color,style in ((reference,'Reference','black','-'),
                                                    (prediction,'SINDy','red','--')):
                    if spectrum is None:
                        continue
                    values = spectrum.amplitudes if kind=='amplitude' else spectrum.phases
                    ax.plot(prediction.frequencies[visible],values[visible,row],label=label,
                            color=color,linestyle=style,linewidth=LINE_WIDTH)
                ax.set(xlim=limits,xlabel=r'$f^*=fD/U_\infty$',
                       ylabel=rf'$|{name}|$' if kind=='amplitude' else rf'$\phi_{{{name}}}$')
                ax.set_box_aspect(1)
                if kind == 'phase':
                    ax.set_ylim(-np.pi,np.pi)
                ax.legend(frameon=False,fontsize=10)
        assessment = title.replace('_', ' ').capitalize()+' Assessment' if title else 'Assessment'
        figure.suptitle('Frequency\n'+assessment,fontsize=14)
        return figure,axes
    except BaseException:
        plt.close(figure)
        raise
