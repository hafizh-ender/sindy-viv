
from __future__ import annotations

from collections.abc import Mapping, Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from typing import Literal
import numpy as np
import re

import matplotlib.pyplot as plt

PLOT_COLORS = ('black', 'red', 'blue')
LINE_WIDTH = 0.6


def fft_figure_size(rows: int, columns: int = 1) -> tuple[float, float]:
    """Compact individual FFT figures to 75% of their former width and height."""
    height = 5*rows+.5 if columns == 1 else 3*rows+1
    return 4.5, .75*height


def format_parameter_title(parameter_values: Mapping[str, float]) -> str:
    """Format parameters in math text; omit fixed Re/mass labels for single-cylinder plots."""
    symbols = {'VR': 'V_R', 'Vr': 'V_R', 'vr': 'V_R', 'LpD': 'L/D', 'lpd': 'L/D'}
    tandem = any(name in parameter_values for name in ('L/D', 'LpD', 'lpd'))
    physical = {'Re', 're', 'm^*', 'm*', 'mstar', 'Mstar', 'm_star'}
    terms = r',\quad '.join(f'{symbols.get(name, name)}={value}' for name, value in parameter_values.items()
                           if tandem or name not in physical)
    return f'${terms}$' if terms else ''


def format_nrmse(value: float) -> str:
    """Use three significant digits, switching to math-text scientific form below 0.01."""
    if 0 < abs(value) < .01:
        mantissa, exponent = f'{value:.2e}'.split('e')
        if int(exponent) <= -3:
            return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    return f'{value:.3g}'


PLOT_VARIABLE_TEMPLATES = {
    'h': 'h_{}', 'hdot': r'\dot{{h}}_{}', 'hddot': r'\ddot{{h}}_{}',
    'CL': 'C_{{L{}}}', 'CLdot': r'\dot{{C}}_{{L{}}}', 'CLddot': r'\ddot{{C}}_{{L{}}}',
    'CD': 'C_{{D{}}}',
}


def select_plot_variables(
    variable_names: Sequence[str],
    plot_variables: Sequence[str] | None = None,
    plot_cylinders: Sequence[int] | None = None,
) -> list[str]:
    """Select available channels by variable family/exact name and cylinder ID.

    Families apply to every selected cylinder; exact LaTeX names select individual
    channels. Both filters intersect, and source column order is retained.
    Unknown variables/cylinders and empty selections raise ValueError.

    Parameters:
        variable_names (Sequence[str]): Available channel names in data-column order.
        plot_variables (Sequence[str] or None): Families h, hdot, hddot, CL, CLdot, CLddot, CD, or exact channel names; None selects all. Default: None
        plot_cylinders (Sequence[int] or None): Cylinder IDs 0 and/or 1; None selects all available cylinders. Default: None
    """
    names = list(variable_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Plot variable names must be nonempty and unique.")
    channels = {
        template.format(cylinder): (family, cylinder)
        for family, template in PLOT_VARIABLE_TEMPLATES.items() for cylinder in (0, 1)
    }
    selected = names
    if plot_variables is not None:
        requested = [plot_variables] if isinstance(plot_variables, str) else list(plot_variables)
        available_families = {channels[name][0] for name in names if name in channels}
        unknown = [name for name in requested if name not in names and name not in available_families]
        if not requested or unknown:
            raise ValueError(f"Unknown or empty plot_variables {requested!r}; available channels: {names!r}.")
        selected = [name for name in selected if name in requested or channels.get(name, (None, None))[0] in requested]
    if plot_cylinders is not None:
        cylinders = list(plot_cylinders)
        available_cylinders = {channels[name][1] for name in names if name in channels}
        if (not cylinders or any(isinstance(cylinder, (bool, np.bool_))
                or not isinstance(cylinder, (int, np.integer)) or cylinder not in available_cylinders
                for cylinder in cylinders)):
            raise ValueError(f"plot_cylinders must select available cylinder IDs {sorted(available_cylinders)!r}.")
        selected = [name for name in selected if channels.get(name, (None, None))[1] in cylinders]
    if not selected:
        raise ValueError("The variable and cylinder filters select no common channels.")
    return selected


def plot_selection_suffix(
    variable_names: Sequence[str],
    plot_variables: Sequence[str] | None = None,
    plot_cylinders: Sequence[int] | None = None,
) -> str:
    """Return a filename suffix identifying explicitly selected plot channels.

    Parameters:
        variable_names (Sequence[str]): Available channel names in data-column order.
        plot_variables (Sequence[str] or None): Variable families or exact channel names. Default: None
        plot_cylinders (Sequence[int] or None): Cylinder IDs to display. Default: None
    """
    if plot_variables is None and plot_cylinders is None:
        return ''
    selected = select_plot_variables(variable_names, plot_variables, plot_cylinders)
    return '_variables=' + '-'.join(re.sub(r'[^A-Za-z0-9_-]', '', name) for name in selected)


def set_axis_fonts(ax: Axes) -> None:
    """Use the time-series tick/label sizes on either 2D or 3D axes."""
    ax.tick_params(labelsize=10)
    for name in ('xaxis', 'yaxis', 'zaxis'):
        axis = getattr(ax, name, None)
        if axis is not None:
            axis.label.set_size(12)


def create_subplots(
    rows: int,
    columns: int = 1,
    *,
    sharex: bool = True,
    squeeze: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Return a figure and paper-style axes, with six inches per column by default.

    rows/columns set the layout; figsize overrides its size. squeeze follows
    Matplotlib, so axes may be one Axes or an array. Top/right spines are hidden.
    The caller owns the returned figure and must close it after use.

    Parameters:
        rows (int): Number of subplot rows.
        columns (int): Number of subplot columns. Default: 1
        sharex (bool): Whether subplot axes share the horizontal scale. Default: True
        squeeze (bool): Whether singleton subplot dimensions are removed from the returned axes. Default: True
        figsize (tuple[float, float] or None): Figure width and height in inches; None uses the standard per-panel size. Default: None
    """
    figure, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, rows) if figsize is None else figsize, sharex=sharex,
        squeeze=squeeze, layout='constrained',
    )
    for ax in np.asarray(axes, dtype=object).flat:
        ax.set_prop_cycle(color=PLOT_COLORS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        ax.tick_params(width=0.6)
        set_axis_fonts(ax)
        ax.grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.35)
    return figure, axes


def _validate_plot_data(time: ArrayLike, data: ArrayLike, names: Sequence[str]) -> np.ndarray:
    """Validate real finite N-by-channel data before any figure is allocated.

    Parameters:
        time (array-like): Recorded fluid time t* aligned with the plotted sample rows.
        data (array-like): Measured samples with one column per supplied variable name.
        names (Sequence[str]): Ordered variable names identifying array columns.
    """
    values = np.asarray(data)
    if values.ndim != 2 or values.shape != (len(time), len(names)) or not names:
        raise ValueError("Plot data must have one aligned column per nonempty variable name.")
    if len(set(names)) != len(names) or values.dtype.kind not in "biuf":
        raise ValueError("Plot names must be unique and values must be real numeric data.")

    # Bound finite-check scratch space even for very long plotted trajectories.
    for start in range(0, len(values), 10_000):
        if not np.all(np.isfinite(values[start:start + 10_000])):
            raise ValueError("Plot data must contain finite values.")
    return values


def _time_windows(
    time: ArrayLike,
    xlim: Sequence[float] | None,
    start: float,
    end: float | None,
    crop_to_xlim: bool,
) -> tuple[np.ndarray, Sequence[float], slice | np.ndarray, slice]:
    """Return validated time/limits and statistics/visible row selectors.

    Time is recorded t*. Sorted axes use slices; unsorted axes retain input line
    order. Cropping is opt-in and retains neighboring boundary segments.
    Empty statistics intervals or invalid bounds raise ValueError.

    Parameters:
        time (array-like): Recorded fluid time t* aligned with the plotted sample rows.

        xlim (Sequence[float] or None): Displayed time limits in recorded t*, or None for the available range.
        start (float): First recorded t* included in the statistics window.
        end (float or None): Last recorded t* included in the statistics window; None uses the available end.
        crop_to_xlim (bool): Whether to limit plotted samples to the visible time interval.
    """
    time = np.asarray(time)

    if time.ndim != 1 or not len(time) or not np.all(np.isfinite(time)):
        raise ValueError("Time must be a finite nonempty 1-D array.")

    if xlim is None:
        xlim = (time[0], time[-1])

    if len(xlim) != 2 or not np.all(np.isfinite(xlim)):
        raise ValueError("xlim must contain two finite bounds.")
    if end is None:
        end = time[-1]
    if not np.isfinite(start) or not np.isfinite(end) or start > end:
        raise ValueError("The statistics interval must have finite ordered bounds.")

    # Sorted timestamps permit slice views instead of a new mask for every channel.
    if np.all(time[1:] >= time[:-1]):
        stats = slice(np.searchsorted(time, start, side='left'), np.searchsorted(time, end, side='right'))
        visible = slice(None)

        # Cropping is opt-in so default plots retain all samples for later panning.
        if crop_to_xlim:
            low, high = sorted(xlim)
            visible = slice(
                max(0, np.searchsorted(time, low, side='left') - 1),
                min(len(time), np.searchsorted(time, high, side='right') + 1),
            )
    else:

        # Preserve line ordering for callers supplying unsorted timestamps.
        stats = (time >= start) & (time <= end)
        visible = slice(None)

    # Reject empty statistics windows before allocating a figure.
    if not time[stats].size:
        raise ValueError("The y-axis statistics interval contains no samples.")

    return time, xlim, stats, visible


def _matching_columns(data: ArrayLike, source_names: Sequence[str], chosen_names: Sequence[str]) -> np.ndarray:

    # Reuse the original array when channel names already match in order.
    """Align selected channel names, returning a view for contiguous columns.

    Noncontiguous/reordered selections require a copy. Callers validate unique
    names and ensure every chosen name exists before invoking this helper.

    Parameters:
        data (array-like): Measured samples with one column per supplied variable name.
        source_names (Sequence[str]): Names identifying the existing data columns.
        chosen_names (Sequence[str]): Names selecting and ordering the returned columns.
    """
    if list(source_names) == list(chosen_names):
        return data
    indices = [source_names.index(name) for name in chosen_names]

    # A contiguous selection can also be represented by a view.
    if indices == list(range(indices[0], indices[0] + len(indices))):
        return data[:, indices[0]:indices[0] + len(indices)]

    return data[:, indices]

def plot_data(
    # Data related arguments
    time: ArrayLike, # t^* = U_inf t / D, NOT tau = f_N t. Shape: (N,)
    data: ArrayLike, # Shape: (N, num_state_variables)
    state_variable_names: Sequence[str],
    parameter_values_dict: Mapping[str, float],

    # Visualization related arguments
    xlim: Sequence[float] | None = None,

    ylim_mean_time_start: float = 0,
    ylim_mean_time_end: float | None = None,

    crop_to_xlim: bool = False,
    plot_variables: Sequence[str] | None = None,
    plot_cylinders: Sequence[int] | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Return a figure and axes for finite N-by-channel signals on recorded t*.

    Names describe columns in order; parameter_values_dict supplies title labels.
    xlim controls display, while ylim_mean_time_start/end select independent
    y-limit statistics. crop_to_xlim limits stored line samples when True;
    default False permits later panning. Inputs are validated before allocation,
    and rendering failures close partial figures. The caller closes successful figures.

    Parameters:
        time (array-like): Recorded fluid time t* aligned with the plotted sample rows.
        data (array-like): Measured samples with one column per supplied variable name.
        state_variable_names (Sequence[str]): Ordered names identifying the supplied data columns before plot selection.
        parameter_values_dict (Mapping[str, float]): Parameter labels and values displayed with the figure.
        xlim (Sequence[float] or None): Displayed time limits in recorded t*, or None for the available range. Default: None
        ylim_mean_time_start (float): First recorded t* included in y-limit statistics. Default: 0
        ylim_mean_time_end (float or None): Last recorded t* included in y-limit statistics; None uses the available end. Default: None
        crop_to_xlim (bool): Whether to limit plotted samples to the visible time interval. Default: False
        plot_variables (Sequence[str] or None): Variable families or exact channel names to display. Default: None
        plot_cylinders (Sequence[int] or None): Cylinder IDs to display. Default: None
    """
    time, xlim, stats, visible = _time_windows(time, xlim, ylim_mean_time_start, ylim_mean_time_end, crop_to_xlim)
    data = _validate_plot_data(time, data, state_variable_names)
    selected_names = select_plot_variables(state_variable_names, plot_variables, plot_cylinders)
    data = _matching_columns(data, state_variable_names, selected_names)
    state_variable_names = selected_names
    fig, axs = create_subplots(len(state_variable_names))
    try:

        suptitle_str = format_parameter_title(parameter_values_dict)

        # Avoid an empty math expression when no parameter values were supplied.
        if suptitle_str:
            fig.suptitle(suptitle_str, fontsize=14)

        for i, state_variable_name in enumerate(state_variable_names):
            ax = axs[i] if len(state_variable_names) > 1 else axs

            # Plot the data for the current state variable
            ax.plot(time[visible], data[visible, i], label=f"${state_variable_name}$", color=PLOT_COLORS[0], linewidth=LINE_WIDTH)

            # Turn on Y-label for all subplots
            ax.set_ylabel(f"${state_variable_name}$", fontsize=12)

            # Turn on X-label for the last subplot only
            if i == len(state_variable_names) - 1:
                ax.set_xlabel("$t^*$", fontsize=12)

            # Find mean value of the data in the specified time range
            values = data[stats, i]
            mean_value = values.mean()
            relative_max_value = max(abs(values.min() - mean_value), abs(values.max() - mean_value))

            # Set the y-axis limits based on the mean value and relative maximum value
            ax.set_ylim(mean_value - 1.1 * relative_max_value, mean_value + 1.1 * relative_max_value)

            # Set the x-axis limits
            ax.set_xlim(xlim)


    except BaseException:

        # Close partially drawn figures when Matplotlib raises before returning to the caller.
        plt.close(fig)
        raise

    return fig, axs

def plot_compare_prior_check(
    # Data related arguments
    time: ArrayLike, # t^* = U_inf t / D, NOT tau = f_N t
    data_true: np.ndarray,
    data_pred: np.ndarray,
    true_state_variable_names: Sequence[str],
    pred_state_variable_names: Sequence[str],
    parameter_values_dict: Mapping[str, float],

    error_metrics_dict: Mapping[str, Mapping[str, float]] | None = None, # Optional dictionary containing error metrics for each state variable

    # Visualization related arguments
    xlim: Sequence[float] | None = None,

    ylim_mean_time_start: float = 0,
    ylim_mean_time_end: float | None = None,

    ylim_source: Literal["reference", "prediction", "both", "true", "pred"] = 'reference', # 'reference', 'prediction', or 'both'; legacy 'true'/'pred' also work.

    crop_to_xlim: bool = False,
    plot_variables: Sequence[str] | None = None,
    plot_cylinders: Sequence[int] | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Align shared reference/prediction channels by name and return comparison axes.

    Both arrays share the same recorded t* axis; channel orders may differ.
    Unknown/nonshared channels are omitted, and no shared channels raises
    ValueError. See plot_compare for limits, statistics, metric labels and
    figure ownership. Noncontiguous channel alignment may copy input columns.

    Parameters:
        time (array-like): Recorded fluid time t* aligned with the plotted sample rows.
        data_true (numpy.ndarray): Reference samples, with columns ordered by their reference names.
        data_pred (numpy.ndarray): Predicted samples, with columns ordered by their prediction names.
        true_state_variable_names (Sequence[str]): Names identifying reference columns.
        pred_state_variable_names (Sequence[str]): Names identifying prediction columns.
        parameter_values_dict (Mapping[str, float]): Parameter labels and values displayed with the figure.
        error_metrics_dict (Mapping[str, Mapping[str, float]] or None): Per-variable metrics included in comparison annotations. Default: None
        xlim (Sequence[float] or None): Displayed time limits in recorded t*, or None for the available range. Default: None
        ylim_mean_time_start (float): First recorded t* included in y-limit statistics. Default: 0
        ylim_mean_time_end (float or None): Last recorded t* included in y-limit statistics; None uses the available end. Default: None
        ylim_source (Literal['reference', 'prediction', 'both', 'true', 'pred']): Data used to calculate y limits: reference, prediction, or both; true and pred are aliases. Default: 'reference'
        crop_to_xlim (bool): Whether to limit plotted samples to the visible time interval. Default: False
        plot_variables (Sequence[str] or None): Variable families or exact shared channel names to display. Default: None
        plot_cylinders (Sequence[int] or None): Cylinder IDs to display. Default: None
    """
    time = np.asarray(time)
    if time.ndim != 1:
        raise ValueError("Time must be a 1-D array.")
    data_true = _validate_plot_data(time, data_true, true_state_variable_names)
    data_pred = _validate_plot_data(time, data_pred, pred_state_variable_names)
    state_variable_names = [var for var in true_state_variable_names if var in pred_state_variable_names]

    if len(state_variable_names) == 0:
        raise ValueError("No common state variable names found between true and predicted data.")

    # Resolve filters on shared names before aligning differently ordered arrays.
    state_variable_names = select_plot_variables(state_variable_names, plot_variables, plot_cylinders)

    # Align both arrays to the true-variable order before comparing channel values.
    data_true_to_compare = _matching_columns(data_true, true_state_variable_names, state_variable_names)
    data_pred_to_compare = _matching_columns(data_pred, pred_state_variable_names, state_variable_names)

    return plot_compare(
        time=time,
        data_true=data_true_to_compare,
        data_pred=data_pred_to_compare,
        state_variable_names=state_variable_names,
        error_metrics_dict=error_metrics_dict,
        parameter_values_dict=parameter_values_dict,
        xlim=xlim,
        ylim_mean_time_start=ylim_mean_time_start,
        ylim_mean_time_end=ylim_mean_time_end,
        ylim_source=ylim_source,
        crop_to_xlim=crop_to_xlim,
    )

def plot_compare(
    # Data related arguments
    time: ArrayLike, # t^* = U_inf t / D, NOT tau = f_N t
    data_true: np.ndarray,
    data_pred: np.ndarray,
    state_variable_names: Sequence[str],
    parameter_values_dict: Mapping[str, float],

    error_metrics_dict: Mapping[str, Mapping[str, float]] | None = None, # Optional dictionary containing error metrics for each state variable

    # Visualization related arguments
    xlim: Sequence[float] | None = None,

    ylim_mean_time_start: float = 0,
    ylim_mean_time_end: float | None = None,

    ylim_source: Literal["reference", "prediction", "both", "true", "pred"] = 'reference', # 'reference', 'prediction', or 'both'; legacy 'true'/'pred' also work.
    crop_to_xlim: bool = False,
    plot_variables: Sequence[str] | None = None,
    plot_cylinders: Sequence[int] | None = None,
    comparison_labels: tuple[str, str] = ('Reference', 'Predicted'),
) -> tuple[Figure, Axes | np.ndarray]:
    """Return comparison axes for finite aligned reference/prediction signals.

    Time and display/statistics limits use t*. ylim_source chooses reference,
    prediction or both (legacy true/pred aliases remain accepted). Independent
    statistics bounds set symmetric limits about the selected mean, with 10%
    padding. comparison_labels names the reference and prediction lines.
    error_metrics_dict supplies per-channel labels. Cropping is opt-in;
    successful figures belong to the caller and failed rendering closes them.

    Parameters:
        time (array-like): Recorded fluid time t* aligned with the plotted sample rows.
        data_true (numpy.ndarray): Reference samples, with columns ordered by their reference names.
        data_pred (numpy.ndarray): Predicted samples, with columns ordered by their prediction names.
        state_variable_names (Sequence[str]): Ordered names identifying both supplied arrays before plot selection.
        parameter_values_dict (Mapping[str, float]): Parameter labels and values displayed with the figure.
        error_metrics_dict (Mapping[str, Mapping[str, float]] or None): Per-variable metrics included in comparison annotations. Default: None
        xlim (Sequence[float] or None): Displayed time limits in recorded t*, or None for the available range. Default: None
        ylim_mean_time_start (float): First recorded t* included in y-limit statistics. Default: 0
        ylim_mean_time_end (float or None): Last recorded t* included in y-limit statistics; None uses the available end. Default: None
        ylim_source (Literal['reference', 'prediction', 'both', 'true', 'pred']): Data used to calculate y limits: reference, prediction, or both; true and pred are aliases. Default: 'reference'
        crop_to_xlim (bool): Whether to limit plotted samples to the visible time interval. Default: False
        plot_variables (Sequence[str] or None): Variable families or exact channel names to display. Default: None
        plot_cylinders (Sequence[int] or None): Cylinder IDs to display. Default: None
    """
    sources = {'true': 'reference', 'pred': 'prediction', 'reference': 'reference', 'prediction': 'prediction', 'both': 'both'}
    if ylim_source not in sources:
        raise ValueError("ylim_source must be 'reference', 'prediction', or 'both'.")
    ylim_source = sources[ylim_source]
    time, xlim, stats, visible = _time_windows(time, xlim, ylim_mean_time_start, ylim_mean_time_end, crop_to_xlim)
    data_true = _validate_plot_data(time, data_true, state_variable_names)
    data_pred = _validate_plot_data(time, data_pred, state_variable_names)
    selected_names = select_plot_variables(state_variable_names, plot_variables, plot_cylinders)
    data_true = _matching_columns(data_true, state_variable_names, selected_names)
    data_pred = _matching_columns(data_pred, state_variable_names, selected_names)
    state_variable_names = selected_names
    ylim_data = data_true if ylim_source == 'reference' else data_pred
    fig, axs = create_subplots(len(state_variable_names))
    try:

        suptitle_str = format_parameter_title(parameter_values_dict)

        # Avoid an empty math expression when no parameter values were supplied.
        if suptitle_str:
            fig.suptitle(suptitle_str, fontsize=14)

        for i, state_variable_name in enumerate(state_variable_names):
            ax = axs[i] if len(state_variable_names) > 1 else axs

            # Plot the true data for the current state variable
            ax.plot(time[visible], data_true[visible, i], label=f"{comparison_labels[0]} ${state_variable_name}$", color=PLOT_COLORS[0], linewidth=LINE_WIDTH)

            # Plot the predicted data for the current state variable
            ax.plot(time[visible], data_pred[visible, i], label=f"{comparison_labels[1]} ${state_variable_name}$", color=PLOT_COLORS[1], linestyle='--', linewidth=LINE_WIDTH)

            # Turn on X-label for the last subplot only
            if i == len(state_variable_names) - 1:
                ax.set_xlabel("$t^*$", fontsize=12)

            # Turn on Y-label for all subplots
            ax.set_ylabel(f"${state_variable_name}$", fontsize=12)

            # Add error metrics to top right corner of the subplot if provided
            if error_metrics_dict is not None and state_variable_name in error_metrics_dict:
                error_metrics = error_metrics_dict[state_variable_name]
                error_metrics_str = ', '.join(
                    f"${key}={format_nrmse(value).strip('$') if key == 'NRMSE' else f'{value:.4f}'}$"
                    for key, value in error_metrics.items())

                ax.text(0.98, 0.95, error_metrics_str, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
                        )

            # Calculate limits from the selected source without concatenating the full trajectories.
            if ylim_source == 'both':
                reference_values, predicted_values = data_true[stats, i], data_pred[stats, i]
                mean_value = (reference_values.mean() + predicted_values.mean()) / 2
                relative_max_value = max(abs(reference_values.min() - mean_value), abs(reference_values.max() - mean_value), abs(predicted_values.min() - mean_value), abs(predicted_values.max() - mean_value))
            else:
                values = ylim_data[stats, i]
                mean_value = values.mean()
                relative_max_value = max(abs(values.min() - mean_value), abs(values.max() - mean_value))

            # Set the y-axis limits based on the mean value and relative maximum value
            if relative_max_value == 0:
                relative_max_value = max(abs(mean_value)*.05, 1e-12)
            ax.set_ylim(mean_value - 1.1 * relative_max_value, mean_value + 1.1 * relative_max_value)

            # Set the x-axis limits
            ax.set_xlim(xlim)


    except BaseException:

        # Close partially drawn figures when Matplotlib raises before returning to the caller.
        plt.close(fig)
        raise

    return fig, axs
