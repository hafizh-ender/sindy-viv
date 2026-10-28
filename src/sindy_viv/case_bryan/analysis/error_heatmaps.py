"""Title-free NRMSE heatmaps shared by numerical and SINDy IVP summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .data_windows import channel_token, save_figure
from sindy_viv.case_bryan.plot import set_axis_fonts, format_nrmse


# Give the ten-column tandem IVP grids more room for their numeric annotations.
TANDEM_IVP_HEATMAP_SCALE = 1.28
HEATMAP_GRID_LINEWIDTH = .6
HEATMAP_SPINE_LINEWIDTH = .8


def heatmap_colorbar(ax, image, label):
    """Keep the colorbar aligned with the height of an equal-scale heatmap."""
    colorbar_axes = make_axes_locatable(ax).append_axes('right', size='5%', pad=.12)
    colorbar = ax.figure.colorbar(image, cax=colorbar_axes)
    set_axis_fonts(colorbar.ax)
    colorbar.set_label(label, fontsize=12)
    colorbar.outline.set_edgecolor('black')
    colorbar.outline.set_linewidth(HEATMAP_SPINE_LINEWIDTH)
    return colorbar


def style_heatmap_grid(ax, shape, *, grid_linewidth=HEATMAP_GRID_LINEWIDTH):
    """Use the shared white grid and black frame; inner lines never exceed the frame."""
    grid_linewidth = min(grid_linewidth, HEATMAP_SPINE_LINEWIDTH)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(HEATMAP_SPINE_LINEWIDTH)
    ax.set_xticks(np.arange(1, shape[1])-.5, minor=True)
    ax.set_yticks(np.arange(1, shape[0])-.5, minor=True)
    ax.grid(False, which='major')
    ax.grid(which='minor', color='white', linewidth=grid_linewidth, linestyle='-', alpha=1.)
    ax.tick_params(which='minor', bottom=False, top=False, left=False, right=False)


def style_heatmap(ax, image, grid, *, weights=False, grid_linewidth=HEATMAP_GRID_LINEWIDTH):
    """Use square cells, the shared grid/frame, and contrast-aware annotations.

    Coefficients use three decimal places and 8-point text. Nonzero magnitudes
    below 0.001 retain their sign with a less-than label. NRMSE keeps its
    scientific-format policy and 9-point text.
    """
    grid = np.asarray(grid, dtype=float)
    ax.set_aspect('equal', adjustable='box')
    style_heatmap_grid(ax, grid.shape, grid_linewidth=grid_linewidth)
    for (row, column), value in np.ndenumerate(grid):
        if not np.isfinite(value):
            label = 'N/A'
        elif weights:
            label = ('0.000' if value == 0 else ('−<0.001' if value < 0 else '<0.001')) if abs(value) < .001 else f'{value:.3f}'
        else:
            label = format_nrmse(value)
        color = 'black'
        if np.isfinite(value):
            red, green, blue, _ = image.cmap(image.norm(value))
            color = 'black' if .299*red+.587*green+.114*blue > .5 else 'white'
        ax.text(column, row, label, ha='center', va='center', color=color, fontsize=8 if weights else 9)


def fit_heatmap_annotations(figure, ax):
    """Grow square cells to contain their annotations, retaining the shared font size."""
    for _ in range(4):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        padding = 4*figure.dpi/72
        scale = 1.
        for text in ax.texts:
            x,y = text.get_position()
            left,bottom = ax.transData.transform((x-.5,y-.5))
            right,top = ax.transData.transform((x+.5,y+.5))
            bounds = text.get_window_extent(renderer)
            scale = max(scale,(bounds.width+padding)/abs(right-left),(bounds.height+padding)/abs(top-bottom))
        if scale <= 1.001:
            return
        figure.set_size_inches(figure.get_size_inches()*scale*1.02)
    figure.canvas.draw()


def save_nrmse_heatmaps(frame, names, folder, stem, *, cylinder_type, vrs, lpds, show=False,
                        figure_scale=1., grid_linewidth=HEATMAP_GRID_LINEWIDTH):
    """Save per-state heatmaps, with optional figure scaling and inner-grid width."""
    paths = []
    columns = sorted(vrs)
    for name in names:
        if cylinder_type == 'single':
            grid = frame.set_index('VR')[name].reindex(columns).to_numpy(dtype=float)[None, :]
            labels = ['single']
        else:
            labels = [f'{lpd:g}' for lpd in sorted(lpds)]
            grid = frame.pivot(index='LpD', columns='VR', values=name).reindex(
                index=sorted(lpds), columns=columns).to_numpy(dtype=float)
        width = max(6, len(columns)*.6)
        height = 2.2 if cylinder_type == 'single' else max(4, len(labels)*.6)
        fig, ax = plt.subplots(figsize=(width*figure_scale, height*figure_scale))
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('lightgray')
        image = ax.imshow(np.ma.masked_invalid(grid), origin='lower', aspect='equal', cmap=cmap)
        ax.set_xticks(range(len(columns)), [f'{vr:g}' for vr in columns])
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel(r'$V_R$')
        if cylinder_type == 'two_tandem':
            ax.set_ylabel(r'$L/D$')
        style_heatmap(ax, image, grid, grid_linewidth=grid_linewidth)
        set_axis_fonts(ax)
        heatmap_colorbar(ax, image, 'NRMSE')
        fig.tight_layout()
        fit_heatmap_annotations(fig, ax)
        paths.append(save_figure(fig, Path(folder)/f'{stem}_{channel_token(name)}_heatmap.png', show))
    return paths
