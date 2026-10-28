"""Bryan diagnostic presentation for feature correlations and condition summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from sindy_viv.case_bryan.analysis.data_windows import save_figure
from sindy_viv.case_bryan.analysis.error_heatmaps import (
    fit_heatmap_annotations, heatmap_colorbar, style_heatmap_grid,
)
from sindy_viv.case_bryan.plot import create_subplots


def feature_labels(names):
    """Preserve duplicate library names as distinct CSV columns, in original order."""
    if len(set(names)) == len(names):
        return list(names)
    return [f'F{index+1:04d}: {name}' for index, name in enumerate(names)]


def plot_correlation(result, path: Path, title: str, show=False):
    """Keep every matrix cell; use feature IDs/sparse ticks for large libraries."""
    count = result.feature_count
    size = min(12., max(5., .35*count+2))
    figure, ax = create_subplots(1, figsize=(size, size), squeeze=True, sharex=False)
    try:
        cmap = plt.get_cmap('coolwarm').copy()
        cmap.set_bad('lightgray')
        image = ax.imshow(np.ma.masked_invalid(result.correlation), cmap=cmap,
                          vmin=-1, vmax=1, aspect='equal')
        labels = feature_labels(result.feature_names)
        indices = np.unique(np.linspace(0, count-1, min(30, count), dtype=int))
        ticks = ([rf'${labels[i]}$' for i in indices] if count <= 20 else
                 [f'F{i+1:04d}' for i in indices])
        ax.set_xticks(indices, ticks, rotation=90)
        ax.set_yticks(indices, ticks)
        ax.tick_params(labelsize=9 if count <= 20 else 8)
        ax.set_title(title, fontsize=12)
        style_heatmap_grid(ax, (count, count), grid_linewidth=min(.6, 12/count))
        if count <= 12:
            for (row, column), value in np.ndenumerate(result.correlation):
                ax.text(column, row, 'N/A' if not np.isfinite(value) else f'{value:.2f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if np.isfinite(value) and abs(value) > .65 else 'black')
        heatmap_colorbar(ax, image, 'Pearson correlation')
        if result.constant_columns:
            ax.set_xlabel('Gray: N/A (constant feature)', fontsize=9)
        return save_figure(figure, path, show)
    finally:
        plt.close(figure)


def _condition_label(value):
    """Format a heatmap cell as a compact number, scientific math text, or infinity."""
    if not np.isfinite(value):
        return r'$\infty$'
    if value >= 1000:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    return f'{value:.3g}'


def plot_condition_heatmap(records, vrs, lpds, path, *, single, show=False, normalized=False):
    """Plot one row or L/D by VR; normalized labels the supplied values as scaled Theta.

    Callers provide raw or normalized condition values in records. Infinite
    values are gray cells; finite values share a logarithmic color scale.
    """
    lookup = {(row['LpD'], row['VR']):row['condition_number'] for row in records}
    vrs, lpds = sorted(vrs), sorted(lpds, key=lambda x: -1 if x is None else x)
    grid = np.array([[lookup[(lpd, vr)] for vr in vrs] for lpd in lpds])
    figure, ax = create_subplots(1, figsize=(max(6, .8*len(vrs)), 2.2 if single else max(4, .8*len(lpds))),
                                 squeeze=True, sharex=False)
    try:
        finite = grid[np.isfinite(grid)]
        minimum, maximum = (float(finite.min()), float(finite.max())) if len(finite) else (1., 10.)
        maximum = maximum if maximum > minimum else minimum*10
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('lightgray')
        image = ax.imshow(np.ma.masked_invalid(grid), origin='lower', aspect='equal', cmap=cmap,
                          norm=LogNorm(vmin=minimum, vmax=maximum))
        ax.set_xticks(range(len(vrs)), [f'{vr:g}' for vr in vrs])
        ax.set_yticks(range(len(lpds)), ['single'] if single else [f'{lpd:g}' for lpd in lpds])
        ax.set_xlabel(r'$V_R$')
        if not single:
            ax.set_ylabel(r'$L/D$')
        style_heatmap_grid(ax, grid.shape)
        for (row, column), value in np.ndenumerate(grid):
            rgb = image.cmap(image.norm(value))[:3] if np.isfinite(value) else (1, 1, 1)
            ax.text(column, row, _condition_label(value), ha='center', va='center', fontsize=9,
                    color='black' if np.dot(rgb, [.299, .587, .114]) > .5 else 'white')
        if len(finite):
            heatmap_colorbar(ax, image, r'$\kappa_2(\widehat{\Theta})$' if normalized else r'$\kappa_2(\Theta)$')
        if np.any(~np.isfinite(grid)):
            ax.set_xlabel(r'$V_R$'+'\n'+r'Gray $\infty$: deficient column rank')
        fit_heatmap_annotations(figure, ax)
        return save_figure(figure, path, show)
    finally:
        plt.close(figure)
