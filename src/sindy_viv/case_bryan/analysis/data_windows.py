"""Shared raw-data selection for amplitude and diagnostic plots; bounds use t*."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator, Sequence

import numpy as np

from sindy_viv.case_bryan import constants as c
from sindy_viv.case_bryan.input import load_data_single_cylinder, load_data_two_tandem_cylinders
from sindy_viv.case_bryan.model_storage.fit_windows import load_fit_window_config, prepare_fit_window_bounds
from .paths import analysis_directory
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.validation import validate_selection


@dataclass
class DataSelection:
    """Validated dataset names, physical selections and per-trajectory t* bounds."""

    cylinder_type: str
    case_name: str
    re: float
    mstar: float
    names: tuple[str, ...]
    vrs: tuple[int, ...]
    lpds: tuple[float | None, ...]
    bounds: dict[tuple[float | None, int], tuple[float | None, float | None]]
    window_name: str


def data_selection(
    cylinder_type: str = 'single', vr_list: Sequence[float] | None = None,
    lpd_list: Sequence[float] | None = None, fit_window_case: str | Path | None = None,
    tstar0: float | None = None, tstarf: float | None = None,
) -> DataSelection:
    """Validate raw selections before loading; null bounds use available endpoints.

    Single JSONs map VR to bounds; tandem JSONs map L/D then VR. Explicit t*
    bounds and a window config are mutually exclusive. Bryan VR folders use
    integer labels, including when callers supply integer-valued floats.
    """
    if cylinder_type not in ('single', 'two_tandem'):
        raise ValueError('cylinder_type must be single or two_tandem.')
    single = cylinder_type == 'single'
    if single and lpd_list is not None:
        raise ValueError('lpd_list requires two_tandem data.')
    vrs = validate_selection(vr_list if vr_list is not None else (range(3, 9) if single else range(5, 15)), 'vr_list')
    if any(int(vr) != vr for vr in vrs):
        raise ValueError('Bryan data VR selections must be integers.')
    vrs = tuple(int(vr) for vr in vrs)
    lpds = (None,) if single else tuple(float(x) for x in validate_selection(
        lpd_list if lpd_list is not None else (1.5, 2., 3., 3.5, 4.), 'lpd_list'))
    if fit_window_case is not None and (tstar0 is not None or tstarf is not None):
        raise ValueError('Use either fit_window_case or tstar0/tstarf, not both.')
    for bound in (tstar0, tstarf):
        if bound is not None and (isinstance(bound, bool) or not np.isfinite(bound)):
            raise ValueError('Time bounds must be finite numbers or None.')
    if tstar0 is not None and tstarf is not None and tstar0 >= tstarf:
        raise ValueError('tstar0 must be smaller than tstarf.')
    pairs = [(lpd, vr) for lpd in lpds for vr in vrs]
    bounds = dict.fromkeys(pairs, (tstar0, tstarf))
    if fit_window_case is not None:
        bounds = prepare_fit_window_bounds(load_fit_window_config(fit_window_case, cylinder_type), pairs, cylinder_type)
    return DataSelection(
        cylinder_type, c.SINGLE_CYLINDER_CASE_NAME if single else c.TWO_TANDEM_CYLINDERS_CASE_NAME,
        c.SINGLE_CYLINDER_RE if single else c.TWO_TANDEM_CYLINDERS_RE,
        c.SINGLE_CYLINDER_MSTAR if single else c.TWO_TANDEM_CYLINDERS_MSTAR,
        c.SINGLE_CYLINDER_STATE_VARIABLES if single else c.TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
        vrs, lpds, bounds, Path(fit_window_case).stem if fit_window_case is not None else 'explicit',
    )


def load_trajectory(selection: DataSelection, vr: int, lpd: float | None, names: Sequence[str]):
    """Load full measured channels in float64 tau; differentiation must precede cropping."""
    if selection.cylinder_type == 'single':
        return load_data_single_cylinder(selection.re, selection.mstar, vr, variables=list(names))
    return load_data_two_tandem_cylinders(selection.re, selection.mstar, vr, lpd, variables=list(names))


def iter_raw_windows(selection: DataSelection, names: Sequence[str]) -> Iterator[tuple]:
    """Yield owning cropped arrays and metadata, releasing each full CSV trajectory."""
    for lpd in selection.lpds:
        for vr in selection.vrs:
            time, data = load_trajectory(selection, vr, lpd, names)
            start, stop = selection.bounds[(lpd, vr)]
            region = select_time_region(time, vr, start, stop)
            tstar, values = time[region].copy()*vr, data[region].copy()
            del time, data
            metadata = dict(VR=vr, LpD=lpd, tstar_region=[float(tstar[0]), float(tstar[-1])],
                            requested_tstar0=start, requested_tstarf=stop,
                            sample_count=len(tstar), fit_window=selection.window_name)
            yield vr, lpd, tstar, values, metadata
            del tstar, values


def result_id(value: Any) -> str:
    """Stable short identity for selected parameters, bounds and processing options."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()[:12]


def channel_token(name: str) -> str:
    """Turn a LaTeX channel label into a portable filename component."""
    return re.sub(r'[^A-Za-z0-9_-]', '', name)


def save_figure(figure, path: Path, show: bool = False, *, tight: bool = True) -> Path:
    """Save at 300 DPI and release a figure; tight=False retains the full 3D canvas."""
    import matplotlib.pyplot as plt
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches='tight' if tight else None)
        if show:
            plt.show()
    finally:
        plt.close(figure)
    return path
