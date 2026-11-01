"""Compile Case 6/7 lift and Case 3/theoretical motion into one autonomous IVP."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.feature_library.base import BaseFeatureLibrary

from sindy_viv.case_bryan.model_storage.model_info import ModelInfo
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.feature_library import prepare_library
from sindy_viv.utils import add_dot_text
from .model_evaluation import load_coefficients


def complete_state_names(case: int, cylinder_type: str) -> list[str]:
    """Full states are interleaved by cylinder; Case 6 has no lift-rate state."""
    if case not in (4, 6, 7):
        raise ValueError('Complete lift models support Cases 4, 6, and 7.')
    return [name for i in get_case_layout(case, cylinder_type).cylinder_ids
            for name in ((f'h_{i}', rf'\dot{{h}}_{i}', f'C_{{L{i}}}') if case == 6 else
                         (f'h_{i}', rf'\dot{{h}}_{i}', f'C_{{L{i}}}', rf'\dot{{C}}_{{L{i}}}'))]


def substitute_parameters(library: BaseFeatureLibrary, coefficients: np.ndarray,
                          response_names: Sequence[str], parameters: Sequence[float]) -> tuple[BaseFeatureLibrary, np.ndarray, list[str]]:
    """Contract a fitted tensor into response coefficients at this VR/L/D first.

    ParameterLibrary tensor order is parameter feature outermost, response
    feature innermost. The resulting model contains no parameter channels.
    """
    if not isinstance(library, ps.ParameterizedLibrary):
        if len(parameters):
            raise ValueError('Parameter values require a ParameterizedLibrary.')
        return library, coefficients, library.get_feature_names(list(response_names))
    parameter_library, response_library = library.libraries[:2]
    if len(parameters) != parameter_library.n_features_in_:
        raise ValueError('Parameter values do not match the fitted tensor library.')
    weights = np.asarray(parameter_library.transform(np.asarray(parameters, dtype=float)[None, :]))[0]
    features = response_library.get_feature_names(list(response_names))
    tensor = coefficients.reshape(len(weights), len(features), coefficients.shape[1])
    effective = np.tensordot(weights, tensor, axes=(0, 0))
    if not np.all(np.isfinite(effective)):
        raise ValueError('Parameter substitution produced nonfinite coefficients.')
    return response_library, effective, features


class MotionModels:
    """Select exactly one saved Case 3 donor per evaluated trajectory, or theory."""

    def __init__(self, sources=None) -> None:
        if isinstance(sources, MotionModels):
            self.models = sources.models
            return
        paths = [] if sources is None else ([sources] if isinstance(sources, (str, Path)) else list(sources))
        self.models = [ModelInfo.read(path) for path in paths]
        if any(model.metadata['dynamics_case'] != 3 for model in self.models):
            raise ValueError('Motion donors must be Case 3 models.')
        directories = [model.directory for model in self.models]
        if len(set(directories)) != len(directories):
            raise ValueError('Case 3 donors must be distinct saved models.')

    def select(self, cylinder_type: str, vr: float, lpd: float | None) -> ModelInfo | None:
        if not self.models:
            return None
        matches = []
        for model in self.models:
            if model.metadata['cylinder_type'] != cylinder_type:
                continue
            try:
                model.validate_selection(vr, lpd)
            except (ValueError, FileNotFoundError):
                continue
            matches.append(model)
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one compatible Case 3 model for {cylinder_type}, VR={vr}, L/D={lpd}; found {len(matches)}.')
        return matches[0]

    def report_folder(self, model: ModelInfo | None = None) -> Path:
        if not self.models:
            return Path('theoretical_motion')
        if model is None and len(self.models) == 1:
            model = self.models[0]
        if model is not None:
            for parent in model.directory.parents:
                if parent.name == 'dynamics_case_3':
                    return Path('case_3_hybrid')/model.directory.relative_to(parent)
        identity = hashlib.sha256('|'.join(sorted(str(x.directory) for x in
            ([model] if model is not None else self.models))).encode()).hexdigest()[:12]
        return Path('case_3_hybrid')/('models='+identity)


class CompleteLibrary:
    """A union of saved response bases, evaluated in the full IVP channel order."""

    def __init__(self, sources, feature_names: Sequence[str]) -> None:
        self.sources = sources
        self.feature_names = list(feature_names)
        lookup = {name:i for i, name in enumerate(feature_names)}
        self.mapping = [[(i, lookup[name]) for i, name in enumerate(names) if name in lookup]
                        for _, _, names in sources]

    def transform(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples)
        result = np.empty((len(samples), len(self.feature_names)))
        for (library, columns, _), mapping in zip(self.sources, self.mapping):
            if not mapping:
                continue
            values = np.asarray(library.transform(samples[:, columns]))
            source, destination = zip(*mapping)
            result[:, destination] = values[:, source]
        return result

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        return self.feature_names.copy()


def compile_complete_model(library: BaseFeatureLibrary, coefficients: np.ndarray,
                           reference: np.ndarray, *, case: int, cylinder_type: str,
                           vr: float, lpd: float | None, mstar: float,
                           parameters: Sequence[float] = (), motion_models=None,
                           base_model_dir: str | Path | None = None,
                           output_directory: str | Path | None = None) -> tuple[CompleteLibrary, np.ndarray, dict]:
    """Return one effective full-state library/matrix with exact identities.

    Both source tensors are substituted independently at the evaluated VR/L/D,
    then their named features are merged. Case 3's learned h' column is ignored.
    No measured motion/lift history is supplied to the autonomous prediction.
    """
    if case not in (6, 7):
        raise ValueError('Compiled lift models require Case 6 or 7.')
    names = complete_state_names(case, cylinder_type)
    if reference.ndim != 2 or reference.shape[1] != len(names):
        raise ValueError('Complete reference has the wrong full-state channel order/shape.')
    lift_layout, motion_layout = get_case_layout(case, cylinder_type), get_case_layout(3, cylinder_type)
    response, lift_weights, lift_features = substitute_parameters(
        library, coefficients, lift_layout.response_names, parameters)
    sources = [(response, [names.index(name) for name in lift_layout.response_names], lift_features)]
    blocks = []
    lift_block = np.zeros((len(lift_features), len(names)))
    lift_block[:, [names.index(name) for name in lift_layout.state_names]] = lift_weights
    blocks.append(lift_block)
    donors = MotionModels(motion_models)
    donor = donors.select(cylinder_type, vr, lpd)
    motion_columns = [names.index(name) for name in motion_layout.response_names]
    if donor is None:
        if not np.isfinite(vr) or vr <= 0 or not np.isfinite(mstar) or mstar <= 0:
            raise ValueError('vr and structural mstar must be finite and positive.')
        motion_library = ps.PolynomialLibrary(degree=1, include_bias=False)
        motion_library.fit(reference[:, motion_columns])
        motion_features = motion_library.get_feature_names(list(motion_layout.response_names))
        motion_weights = np.zeros((len(motion_features), len(motion_layout.state_names)))
        for i in motion_layout.cylinder_ids:
            target = list(motion_layout.state_names).index(rf'\dot{{h}}_{i}')
            motion_weights[motion_features.index(f'h_{i}'), target] = -4*np.pi**2
            motion_weights[motion_features.index(f'C_{{L{i}}}'), target] = 2*vr**2/(np.pi*mstar)
        provenance = dict(source='theoretical', structural_mstar=float(mstar))
    else:
        if base_model_dir is not None and (Path(base_model_dir)/'training.json').is_file():
            base_metadata = json.loads((Path(base_model_dir)/'training.json').read_text(encoding='utf-8'))
            for field in ('Re', 'mstar'):
                if field in base_metadata and field in donor.metadata and base_metadata[field] != donor.metadata[field]:
                    raise ValueError(f'Case 3 and lift models have different {field} values.')
        config = donor.metadata['configurations']
        motion_library = prepare_library(config['state_feature_library'])
        parameter_names = {'individual': [], 'vr': ['V_R'], 'lpd': ['L/D'], 'vr_lpd': ['L/D', 'V_R']}[donor.metadata['parameterization']]
        values = {'V_R': vr, 'L/D': lpd}
        motion_parameters = tuple(values[name] for name in parameter_names)
        if parameter_names:
            motion_library = ps.ParameterizedLibrary(feature_library=motion_library,
                parameter_library=prepare_library(config['parameter_feature_library']),
                num_features=len(motion_columns), num_parameters=len(parameter_names))
        controls = reference[:, [names.index(name) for name in motion_layout.input_names]]
        if motion_parameters:
            controls = np.column_stack((controls, np.broadcast_to(motion_parameters, (len(reference), len(motion_parameters)))))
        motion_weights = load_coefficients(donor.directory, motion_library,
            reference[:, [names.index(name) for name in motion_layout.state_names]], motion_layout.state_names,
            input_data=controls, input_names=list(motion_layout.input_names)+parameter_names)
        motion_library, motion_weights, motion_features = substitute_parameters(
            motion_library, motion_weights, motion_layout.response_names, motion_parameters)
        provenance = dict(source='case_3', **donor.manifest())
    motion_block = np.zeros((len(motion_features), len(names)))
    for i in motion_layout.cylinder_ids:
        target = names.index(rf'\dot{{h}}_{i}')
        motion_block[:, target] = motion_weights[:, list(motion_layout.state_names).index(rf'\dot{{h}}_{i}')]
    sources.append((motion_library, motion_columns, motion_features))
    blocks.append(motion_block)
    # Explicit full-state identity rows eliminate any learned kinematic error.
    identity = ps.IdentityLibrary().fit(reference)
    identity_block = np.zeros((len(names), len(names)))
    for i in lift_layout.cylinder_ids:
        identity_block[names.index(rf'\dot{{h}}_{i}'), names.index(f'h_{i}')] = 1.
        if case == 7:
            identity_block[names.index(rf'\dot{{C}}_{{L{i}}}'), names.index(f'C_{{L{i}}}')] = 1.
    sources.append((identity, list(range(len(names))), names))
    blocks.append(identity_block)
    features = list(dict.fromkeys(name for _, _, source_names in sources for name in source_names))
    combined = pd.DataFrame(0., index=features, columns=add_dot_text(names))
    for (_, _, source_names), weights in zip(sources, blocks):
        combined.loc[source_names, :] += weights
    combined = combined.loc[(combined != 0).any(axis=1)]
    if not np.all(np.isfinite(combined.to_numpy())):
        raise ValueError('Complete model coefficients must be finite.')
    base_provenance = None
    if base_model_dir is not None:
        directory = Path(base_model_dir)
        base_provenance = dict(model_directory=str(directory),
            model_sha256=hashlib.sha256((directory/'model.csv').read_bytes()).hexdigest())
        if (directory/'training.json').is_file():
            base_provenance['training'] = json.loads((directory/'training.json').read_text(encoding='utf-8'))
    manifest = dict(dynamics_case=case, state_names=names, time_coordinate='tau=t*/VR',
        parameterization='individual', parameters_substituted=dict(VR=vr, LpD=lpd),
        motion=provenance, lift_model=base_provenance,
        kinematic_identities=['d(h_i)/d(tau) = hdot_i']+(['d(CL_i)/d(tau) = CLdot_i'] if case == 7 else []),
        prescribed_controls=False, features=list(combined.index), targets=list(combined.columns))
    if output_directory is not None:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        combined.to_csv(directory/'complete_model.csv')
        (directory/'complete_model.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return CompleteLibrary(sources, combined.index), combined.to_numpy(), manifest
