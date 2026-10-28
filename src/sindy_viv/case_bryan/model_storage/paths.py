"""Dataset-specific model paths and saved-model discovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from sindy_viv._typing import CylinderType, DynamicsCase, Parameterization
from typing import Any

import hashlib
import json
from pathlib import Path


from .fit_windows import load_fit_window_config, prepare_fit_window_bounds


def number_text(value: Any) -> str:
    """Use round-trip numeric labels without an unnecessary trailing .0.

    Parameters:
        value (object): Numeric value to format as a round-trip path label.
    """
    text = repr(float(value))
    return text[:-2] if text.endswith('.0') else text


def model_base_directory(
    results_root: str | PathLike[str],
    case_name: str,
    dynamics_case: DynamicsCase,
    differentiator_name: str,
    fit_mode: str,
    cylinder_type: CylinderType,
    parameterization: Parameterization,
    state_feature_library_case: str | PathLike[str] | None,
    input_feature_library_case: str | PathLike[str] | None,
    optimizer_case: str | PathLike[str],
    re: float,
    mstar: float,
    vr: float,
    lpd: float,
    *,
    weak_feature_library_case: str | PathLike[str] | None = None,
    filtering_case: str | PathLike[str] | None = None,
) -> Path:
    """Build the shared model path before dataset/window folders.

    Library and differentiator configuration stems identify settings. Single
    datasets include Re/Mstar; tandem data retain their fixed dataset layout.
    VR/LpD appear here only for individual fits; parametric selections and
    fitting windows are appended by training_directory.
    Weak fits insert the weak-library config stem immediately after fit_mode;
    None selects the default weak-library label, "default".
    Filtering inserts filtering=<config-stem> before the differentiator folder;
    omitting it preserves the existing unfiltered paths.

    Parameters:
        results_root (str or PathLike[str]): Root directory for fitted models and evaluation artifacts.
        case_name (str): Dataset label used in saved-model paths.
        dynamics_case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5.
        differentiator_name (str): Differentiator label used in model paths.
        fit_mode (str): Fit-family label used in saved-model paths.
        cylinder_type (str): Cylinder configuration: single or two_tandem.
        parameterization (str): Explicit fitting family: individual, vr, lpd, or vr_lpd.

        state_feature_library_case (str, PathLike[str], or None): Response-library configuration path for states and measured controls.
        input_feature_library_case (str, PathLike[str], or None): Parameter-library configuration path, required for explicit parametric fitting.
        optimizer_case (str or PathLike[str]): Optimizer configuration path.

        re (float): Reynolds number used in dataset labels; None selects the cylinder-specific value.
        mstar (float): Mass ratio; workflow entry points use the cylinder-specific value when None.
        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.
        weak_feature_library_case (str, PathLike[str], or None): Weak-library config path identifying weak fits; only its stem is used. Default: None
        filtering_case (str, PathLike[str], or None): Optional filtering config path whose stem identifies the preprocessing folder. Default: None
    """
    directory = Path(results_root) / case_name / f"dynamics_case_{dynamics_case}"
    if filtering_case is not None:
        directory /= 'filtering=' + Path(filtering_case).stem
    directory = directory / differentiator_name / fit_mode
    if fit_mode.startswith('weak_'):
        directory /= Path(weak_feature_library_case or 'default.json').stem
    if parameterization == "individual":
        directory /= Path(state_feature_library_case).stem
    else:
        directory = directory / f"state_{Path(state_feature_library_case).stem}" / f"input_{Path(input_feature_library_case).stem}"
    directory /= Path(optimizer_case).stem
    if cylinder_type == "single":
        directory = directory / f"Re = {re}" / f"Mstar = {mstar}"
        if parameterization == "individual":
            directory /= f"VR = {vr}"
    elif parameterization == "individual":
        directory = directory / f"LpD = {lpd}" / f"VR = {vr}"
    return directory


def training_directory(
    base_directory: str | PathLike[str],
    parameterization: Parameterization,
    trajectories: Sequence[Mapping[str, Any]],
    fit_window_case: str | PathLike[str] | None = None,
) -> Path:
    """Append dataset selection and fitting-window identity to a model base path.

    Parametric paths include sorted LpD/VR sets. Named configs add their stem;
    shared bounds use readable labels and trajectory-dependent bounds use a
    stable mapping hash. This function constructs a Path and performs no I/O.

    Parameters:
        base_directory (str or PathLike[str]): Model path before dataset and fitting-window folders.
        parameterization (str): Explicit fitting family: individual, vr, lpd, or vr_lpd.
        trajectories (Sequence[Mapping[str, Any]]): Trajectory records or iterable entries in the format described by this function.
        fit_window_case (str, PathLike[str], or None): Config path whose stem labels the folder; None preserves the legacy unnamed layout. Default: None
    """
    directory = Path(base_directory)
    if parameterization != 'individual':
        lpds = sorted({entry['LpD'] for entry in trajectories if entry['LpD'] is not None})
        vrs = sorted({entry['VR'] for entry in trajectories})
        if lpds:
            directory /= 'LpD=' + ','.join(number_text(value) for value in lpds)
        directory /= 'VR=' + ','.join(number_text(value) for value in vrs)

    if fit_window_case is not None:

        # Keep named categories readable while retaining distinct destinations when their limits change.
        directory /= 'fit_window=' + Path(fit_window_case).stem

    bounds = {(entry['requested_tstar0'], entry['requested_tstarf']) for entry in trajectories}
    if len(bounds) == 1:
        start, stop = next(iter(bounds))
        if start is None and stop is None:
            label = 'all'
        else:
            label = ('start' if start is None else number_text(start)) + '..' + ('end' if stop is None else number_text(stop))
    else:

        # Hash the parameter-to-window mapping so reordering equivalent selections keeps the same ID.
        windows = sorted((entry['LpD'], entry['VR'], entry['requested_tstar0'], entry['requested_tstarf'])
                         for entry in trajectories)
        payload = json.dumps(windows, separators=(',', ':'), allow_nan=False)
        label = 'per-trajectory-' + hashlib.sha256(payload.encode()).hexdigest()[:12]

    return directory / ('time=' + label)


def resolve_model_directory(
    base_directory: str | PathLike[str],
    cylinder_type: CylinderType,
    parameterization: Parameterization,
    vr: float,
    lpd: float,
    *,
    model_dir: str | PathLike[str] | None = None,
    training_vr_list: Sequence[float] | None = None,
    training_lpd_list: Sequence[float] | None = None,
    fit_window_case: str | PathLike[str] | None = None,
) -> Path:
    """Resolve one saved model independently of the evaluated parameter values.

    Search named/direct-window layouts and supported legacy coefficient paths.
    Optional training selectors match datasets; fit_window_case also matches
    saved requested bounds. model_dir explicitly selects a directory but still
    undergoes metadata/selection checks. Returns its Path; missing or ambiguous
    matches raise FileNotFoundError/ValueError without modifying saved results.

    Parameters:
        base_directory (str or PathLike[str]): Model path before dataset and fitting-window folders.
        cylinder_type (str): Cylinder configuration: single or two_tandem.
        parameterization (str): Explicit fitting family: individual, vr, lpd, or vr_lpd.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        model_dir (str, PathLike[str], or None): Explicit saved-model directory; None searches by model settings and training selectors. Default: None

        training_vr_list (Sequence[float] or None): Reduced velocities used to identify the saved training dataset independently of evaluation data. Default: None
        training_lpd_list (Sequence[float] or None): Cylinder spacings used to identify the saved training dataset independently of evaluation data. Default: None
        fit_window_case (str, PathLike[str], or None): Training-window config used to match saved metadata; None leaves window selection open. Default: None
    """
    base_directory = Path(base_directory)
    window_config = load_fit_window_config(fit_window_case, cylinder_type) if fit_window_case is not None else None
    if model_dir is not None:
        candidates = [Path(model_dir)]
        if not (candidates[0] / 'model.csv').is_file():
            raise FileNotFoundError(f"No model.csv in model_dir={model_dir}.")
    else:
        if parameterization == 'individual':
            pattern = 'time=*/model.csv'
        elif cylinder_type == 'single':
            pattern = 'VR=*/time=*/model.csv'
        else:
            pattern = 'LpD=*/VR=*/time=*/model.csv'

        # Discover both named config categories and the earlier direct-window directory layout.
        named_pattern = pattern.replace('time=*/', 'fit_window=*/time=*/')
        candidates = [path.parent for search in (pattern, named_pattern) for path in base_directory.glob(search)]

        # Retain access to existing coefficient files without moving or rewriting old experiments.
        legacy = base_directory / f'LpD = {lpd}' if cylinder_type == 'two_tandem' and parameterization == 'vr' else base_directory
        if parameterization != 'lpd' and (legacy / 'model.csv').is_file():
            candidates.append(legacy)

    selected = []
    window_errors = []
    for directory in candidates:
        metadata = None
        metadata_path = directory / 'training.json'
        if metadata_path.is_file():

            # Read the training metadata to determine the cylinder type, parameterization, and trajectory list.
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

            # Skip directories that don't match the requested cylinder type or parameterization, even if they contain a model.csv.
            if metadata['cylinder_type'] != cylinder_type or metadata['parameterization'] != parameterization:
                continue

            # Get the list of trajectories from the metadata.
            trajectories = metadata['trajectories']

            # Extract the VR and L/D values from the trajectories.
            vrs = {entry['VR'] for entry in trajectories}
            lpds = {entry['LpD'] for entry in trajectories if entry['LpD'] is not None}
        elif parameterization != 'individual' and directory.parent.name.startswith('VR='):

            # Older weak-evaluation fixtures may contain coefficients without copied training metadata.
            vrs = {float(value) for value in directory.parent.name[3:].split(',')}
            lpds = ({float(value) for value in directory.parent.parent.name[4:].split(',')}
                    if cylinder_type == 'two_tandem' else set())
        else:
            vrs, lpds = None, None

        if parameterization == 'vr' and cylinder_type == 'two_tandem' and lpds is not None and lpds != {lpd}:
            continue

        if parameterization == 'lpd' and vrs != {vr}:
            continue

        if parameterization == 'individual' and vrs is not None and (vrs != {vr} or (lpd is not None and lpds != {lpd})):
            continue

        if training_vr_list is not None and vrs != set(training_vr_list):
            continue

        if training_lpd_list is not None and lpds != set(training_lpd_list):
            continue

        if window_config is not None:
            if metadata is None or metadata.get('fit_window', {}).get('name') != Path(fit_window_case).stem:
                continue
            pairs = [(entry['LpD'], entry['VR']) for entry in trajectories]
            try:
                expected_bounds = prepare_fit_window_bounds(window_config, pairs, cylinder_type)
            except ValueError as error:
                window_errors.append(str(error))
                continue

            # Compare the training trajectories, never the held-out evaluation VR or L/D.
            if any(expected_bounds[(entry['LpD'], entry['VR'])] !=
                   (entry['requested_tstar0'], entry['requested_tstarf']) for entry in trajectories):
                continue

        selected.append(directory)

    if not selected:
        if window_errors:
            raise ValueError(window_errors[0])
        raise FileNotFoundError(f"No matching saved model under {base_directory}; check training selections, fit_window_case, or model_dir.")

    if len(selected) > 1:
        choices = '\n'.join(str(path) for path in sorted(selected))
        raise ValueError(f"Multiple fitted models match. Select training_vr_list/training_lpd_list and fit_window_case, or pass model_dir:\n{choices}")

    return selected[0]
