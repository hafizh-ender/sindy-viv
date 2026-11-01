"""Compose saved Case 1/2 lift equations with independently saved Case 3 motion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pysindy as ps

from sindy_viv.case_bryan.model_storage.model_info import ModelInfo
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.feature_library import prepare_library
from sindy_viv.utils import add_dot_text
from .model_evaluation import load_coefficients
from .single_suite import evaluate_suite


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    return path




class _HybridLibrary:
    """Evaluate each saved basis using its own channel order and parameters.

    Case 3 lift controls are taken from the current full-state samples, including
    every IVP solver stage. They are never prescribed measured lift histories.
    The two bases need not contain the same features or polynomial degrees.
    """

    def __init__(self, base, motion, response_columns, parameters):
        self.base, self.motion = base, motion
        self.response_columns, self.parameters = response_columns, parameters

    def transform(self, samples):
        samples = np.asarray(samples)
        response = samples[:, self.response_columns]
        if self.parameters:
            response = np.column_stack((response, np.broadcast_to(
                self.parameters, (len(samples), len(self.parameters)))))
        return np.column_stack((np.asarray(self.base.transform(samples)),
                                np.asarray(self.motion.transform(response))))


class Case3Hybrid:
    """Replace all motion targets, retaining every fitted Case 1/2 lift target."""

    def __init__(self, base: ModelInfo, motion: ModelInfo):
        if base.metadata['dynamics_case'] not in (1, 2) or motion.metadata['dynamics_case'] != 3:
            raise ValueError('Select a Case 1/2 base model and a Case 3 motion model.')
        if base.metadata['cylinder_type'] != motion.metadata['cylinder_type']:
            raise ValueError('Both models must use the same cylinder type.')
        for field in ('Re', 'mstar'):
            if field in base.metadata and field in motion.metadata and base.metadata[field] != motion.metadata[field]:
                raise ValueError(f'Models have different {field} values.')
        self.base, self.motion, self.cache = base, motion, {}

    def __call__(self, library, coefficients, *, states, state_names, parameter_names, vr, lpd):
        self.base.validate_selection(vr, lpd)
        self.motion.validate_selection(vr, lpd)
        layout = get_case_layout(3, self.motion.metadata['cylinder_type'])
        columns = [list(state_names).index(name) for name in layout.response_names]
        mode = self.motion.metadata['parameterization']
        names = {'individual': [], 'vr': ['V_R'], 'lpd': ['L/D'], 'vr_lpd': ['L/D', 'V_R']}[mode]
        values = {'V_R': vr, 'L/D': lpd}
        parameters = tuple(values[name] for name in names)
        key = (vr, lpd)
        if key not in self.cache:
            configs = self.motion.metadata['configurations']
            response = prepare_library(configs['state_feature_library'])
            motion_library = response if mode == 'individual' else ps.ParameterizedLibrary(
                feature_library=response, parameter_library=prepare_library(configs['parameter_feature_library']),
                num_features=len(layout.response_names), num_parameters=len(names))
            controls = states[:, [list(state_names).index(name) for name in layout.input_names]]
            if parameters:
                controls = np.column_stack((controls, np.broadcast_to(parameters, (len(states), len(parameters)))))
            motion_coefficients = load_coefficients(self.motion.directory, motion_library,
                states[:, [list(state_names).index(name) for name in layout.state_names]],
                layout.state_names, input_data=controls, input_names=list(layout.input_names)+names)
            self.cache[key] = motion_library, motion_coefficients
        motion_library, motion_coefficients = self.cache[key]
        targets = [list(state_names).index(name) for name in layout.state_names]
        retained = coefficients.copy()
        retained[:, targets] = 0.
        replacement = np.zeros((len(motion_coefficients), len(state_names)))
        replacement[:, targets] = motion_coefficients
        return (_HybridLibrary(library, motion_library, columns, parameters),
                np.vstack((retained, replacement)))


def case3_folder(directory):
    """Mirror all Case 3 settings below dynamics_case_3, including training bounds."""
    directory = Path(directory).resolve()
    for ancestor in directory.parents:
        if ancestor.name == 'dynamics_case_3':
            return directory.relative_to(ancestor)
    identity = hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    return Path('external_'+identity)/directory.name


def evaluate_case3_hybrid(case_12_model, case_3_model, *, derivative_window_cases,
                          integral_window_case, spectral_window_cases=None,
                          vr_list=None, lpd_list=None, **solver_options):
    """Run the ordinary full-state suite with saved motion-equation replacement.

    Inputs accept model directories, training.json paths, or model-info JSONs.
    A model-info JSON uses the training schema and model_directory (relative to
    that JSON is allowed); it supports manually entered theoretical Case 3 CSVs.
    No source CSV/configuration/metadata is modified and no coefficients are fit.
    """
    base, motion = ModelInfo.read(case_12_model), ModelInfo.read(case_3_model)
    hybrid = Case3Hybrid(base, motion)
    metadata = base.metadata
    trajectories = metadata['trajectories']
    vrs = sorted({row['VR'] for row in trajectories}) if vr_list is None else vr_list
    lpds = (sorted({row['LpD'] for row in trajectories}) if metadata['cylinder_type'] == 'two_tandem'
            and lpd_list is None else lpd_list)
    if metadata['cylinder_type'] == 'single' and lpds is not None:
        raise ValueError('lpd_list requires tandem models.')
    directories = {}
    for lpd in ([None] if lpds is None else lpds):
        for vr in vrs:
            base.validate_selection(vr, lpd)
            motion.validate_selection(vr, lpd)
            directories[vr if lpd is None else (lpd, vr)] = base.directory
    provenance = dict(kind='case_3_hybrid', base=base.manifest(), case_3=motion.manifest(),
        replaced_targets=add_dot_text(get_case_layout(3, metadata['cylinder_type']).state_names),
        lift_source='all fitted Case 1/2 lift equations retained',
        motion_controls='evolving full-state lift; measured states only for derivative reconstruction and IVP initialization')
    # Reconstruct settings from snapshots rather than trusting possibly changed
    # or unavailable paths recorded when the source models were trained.
    with TemporaryDirectory(prefix='case3-hybrid-') as temporary:
        root = Path(temporary)
        configs = metadata['configurations']
        if 'differentiator' not in configs:
            raise ValueError('Case 1/2 metadata must include its reference differentiator.')
        differentiator_name = 'differentiator'
        for ancestor in base.directory.parents:
            if ancestor.name in ('individual', 'parametric_vr', 'parametric_lpd', 'parametric_vr_lpd',
                                 'weak_individual', 'weak_parametric_vr', 'weak_parametric_lpd', 'weak_parametric_vr_lpd'):
                if not ancestor.parent.name.startswith('filtering='):
                    differentiator_name = ancestor.parent.name
                break
        (root/'differentiation').mkdir()
        differentiator = _json(root/'differentiation'/(differentiator_name+'.json'), configs['differentiator'])
        state = _json(root/'state_library.json', configs['state_feature_library'])
        parameter = (_json(root/'parameter_library.json', configs['parameter_feature_library'])
                     if metadata['parameterization'] != 'individual' else None)
        filtering = (_json(root/'filtering.json', configs['filtering']) if 'filtering' in configs else None)
        fit = metadata.get('fit_window', {})
        fit_path = _json(root/(Path(fit['name']).name+'.json'), fit['config']) if fit.get('config') is not None else None
        return evaluate_suite(dynamics_case=metadata['dynamics_case'], cylinder_type=metadata['cylinder_type'],
            parameterization=metadata['parameterization'], differentiator_case=differentiator,
            state_feature_library_case=state, input_feature_library_case=parameter,
            optimizer_case='saved_optimizer.json', filtering_case=filtering, fit_window_case=fit_path,
            derivative_window_cases=derivative_window_cases, integral_window_case=integral_window_case,
            spectral_window_cases=derivative_window_cases if spectral_window_cases is None else spectral_window_cases,
            vr_list=vrs, lpd_list=lpds, model_directories=directories, model_adapter=hybrid,
            training_vr_list=sorted({row['VR'] for row in trajectories})
                if metadata['parameterization'] in ('vr', 'vr_lpd') else None,
            training_lpd_list=sorted({row['LpD'] for row in trajectories})
                if metadata['parameterization'] in ('lpd', 'vr_lpd') else None,
            report_subdirectory=Path('case_3_hybrid')/case3_folder(motion.directory),
            report_metadata=provenance, re=metadata.get('Re'), mstar=metadata.get('mstar'), **solver_options)
