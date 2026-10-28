"""Prepare Bryan inputs and write case-specific reports for generic library assessment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps

from sindy_viv.feature_library.assessment import assess_feature_library
from sindy_viv.feature_library.functions import prepare_library
from sindy_viv.utils import initiate_object, load_config
from sindy_viv.case_bryan.analysis.data_windows import data_selection, load_trajectory
from sindy_viv.case_bryan.analysis.paths import analysis_directory
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.model_storage.paths import number_text, training_directory
from sindy_viv.case_bryan.plot import format_parameter_title
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.case_bryan.prepare_sindy.trajectory import prepare_trajectory

from .assessment_plot import feature_labels, plot_condition_heatmap, plot_correlation


def _json(path, value):
    """Create parent folders and write report metadata as finite, standards-compliant JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _number(value):
    """Return a finite condition number or the JSON-safe string Infinity."""
    return float(value) if np.isfinite(value) else 'Infinity'


def trajectory_groups(selection, parameterization):
    """Use precisely the model grouping and parameter order used by compute_sindy."""
    if parameterization not in ('individual', 'vr', 'lpd', 'vr_lpd'):
        raise ValueError('parameterization must be individual, vr, lpd, or vr_lpd.')
    if parameterization in ('lpd', 'vr_lpd') and selection.cylinder_type != 'two_tandem':
        raise ValueError('L/D parameterization requires two_tandem data.')
    pairs = [(lpd, vr) for lpd in selection.lpds for vr in selection.vrs]
    if parameterization == 'individual':
        return [[pair] for pair in pairs], []
    if parameterization == 'vr':
        return [[(lpd, vr) for vr in selection.vrs] for lpd in selection.lpds], ['V_R']
    if parameterization == 'lpd':
        return [[(lpd, vr) for lpd in selection.lpds] for vr in selection.vrs], ['L/D']
    return [pairs], ['L/D', 'V_R']


def prepare_assessment_samples(selection, dynamics_case, vr, lpd, differentiator, filtering, parameter_names):
    """Filter/prepare full tau records, then crop and append selected parameters."""
    layout = get_case_layout(dynamics_case, selection.cylinder_type)
    time, data = load_trajectory(selection, vr, lpd, layout.measured_names)
    total = len(time)
    trajectory = prepare_trajectory(time, data, list(layout.measured_names), differentiator,
        dynamics_case, selection.cylinder_type,
        with_derivatives=False, filtering=filtering)
    del time, data
    bounds = selection.bounds[(lpd, vr)]
    region = select_time_region(trajectory.time, vr, *bounds)
    components = [trajectory.states[region]]
    if trajectory.inputs is not None:
        components.append(trajectory.inputs[region])
    if parameter_names:
        values = {'V_R':vr, 'L/D':lpd}
        components.append(np.broadcast_to([values[name] for name in parameter_names],
                                          (len(components[0]), len(parameter_names))))
    samples = np.concatenate(components, axis=1)
    tstar = trajectory.time[region]*vr
    metadata = dict(VR=vr, LpD=lpd, requested_tstar0=bounds[0], requested_tstarf=bounds[1],
                    tstar_region=[float(tstar[0]), float(tstar[-1])], sample_count=len(samples),
                    total_samples=total)
    return samples, metadata


def _save_report(result, folder, metadata, title, show, normalized_only=False):
    """Write one matrix assessment and return its PNG/CSV/JSON path mapping.

    Retain every feature column, map duplicate names to distinct CSV labels,
    and snapshot preparation settings, singular values and numerical-rank rules.
    The correlation figure is closed after saving or display.
    normalized_only writes just the normalized condition and its metadata JSON,
    leaving every raw/correlation artifact untouched.
    """
    folder.mkdir(parents=True, exist_ok=True)
    normalized = dict(
        condition_number=_number(result.normalized_condition_number),
        numerical_rank=result.normalized_rank, rank_deficient=result.normalized_rank_deficient,
        absolute_rank_tolerance=result.normalized_rank_tolerance,
        singular_values=result.normalized_singular_values.tolist(),
        column_l2_norms=result.column_l2_norms.tolist(),
        zero_columns=np.flatnonzero(result.column_l2_norms == 0).tolist(),
        definition='2-norm condition of Theta with nonzero columns divided by their L2 norms; no centering; bias/zero columns retained',
        algorithm='SVD of column-normalized reduced QR factor; norms span all samples in this trajectory or pool',
    )
    normalized_json = folder/'condition_number_normalized.json'
    normalized_metadata = folder/'condition_number_normalized_metadata.json'
    _json(normalized_json, {'condition_number':normalized['condition_number']})
    _json(normalized_metadata, dict(metadata, column_normalization='unit_l2_without_centering', matrix=dict(sample_count=result.sample_count,
        feature_count=result.feature_count, feature_names=list(result.feature_names), normalized=normalized)))
    extra = dict(normalized_condition_json=normalized_json, normalized_metadata_json=normalized_metadata)
    if normalized_only:
        return extra
    labels = feature_labels(result.feature_names)
    correlation_csv = folder/'correlation_matrix.csv'
    pd.DataFrame(result.correlation, index=labels, columns=labels).rename_axis('feature').to_csv(
        correlation_csv, na_rep='N/A')
    _json(folder/'condition_number.json', {'condition_number':_number(result.condition_number)})
    definition = dict(
        sample_count=result.sample_count, feature_count=result.feature_count,
        feature_names=list(result.feature_names), csv_labels=labels,
        plot_ids={f'F{i+1:04d}':name for i,name in enumerate(result.feature_names)},
        constant_columns=[dict(index=i, name=result.feature_names[i]) for i in result.constant_columns],
        numerical_rank=result.rank, rank_deficient=result.rank_deficient,
        absolute_rank_tolerance=result.rank_tolerance,
        singular_values=result.singular_values.tolist(),
        condition_number=_number(result.condition_number),
        correlation_definition='Pearson correlation of centered feature columns; constant pairs are N/A',
        condition_definition='2-norm of raw, uncentered, unnormalized Theta; infinity for deficient column rank',
        condition_algorithm='SVD of merged reduced QR factors; no Gram-matrix condition approximation',
        undefined_encoding=dict(correlation_csv='N/A', condition_json='Infinity'),
        correlation_plot=dict(all_cells=True, maximum_tick_labels=30, feature_id_ticks=result.feature_count>20,
                              annotate_cells=result.feature_count<=12))
    _json(folder/'correlation_metadata.json', dict(metadata, matrix=definition))
    figure = plot_correlation(result, folder/'correlation_matrix.png', title, show)
    return dict(correlation_csv=correlation_csv, metadata_json=folder/'correlation_metadata.json',
                condition_json=folder/'condition_number.json', correlation_figure=figure, **extra)


def assess_bryan_feature_library(
    state_feature_library_case, *, dynamics_case, cylinder_type='single',
    vr_list=None, lpd_list=None, fit_window_case=None, tstar0=None, tstarf=None,
    parameterization='individual', input_feature_library_case=None,
    differentiator_case='configs/differentiator/finite_difference_basic.json', filtering_case=None,
    batch_size=8192, rank_rtol=None, results_root='results', output_dir=None, show=False,
    normalized_only=False,
):
    """Assess every selected trajectory and, for tensors, each pooled training group.

    No saved model or optimizer is required. Controls and derived lift rates use
    the same order and tau coordinate as training. Derivative targets are not
    computed. Parameterized per-trajectory matrices often have proportional
    columns; pooled reports quantify the actual multi-trajectory fitting matrix.
    Default outputs follow the shared analysis layout, including Re/Mstar
    folders for single-cylinder data. Explicit output_dir overrides retain
    their custom case-name root.
    Both raw and unit-L2-column condition reports are written by default.
    normalized_only preserves existing raw/correlation artifacts while adding
    normalized JSON, metadata, summary tables and heatmaps.
    """
    layout = get_case_layout(dynamics_case, cylinder_type)
    if fit_window_case is None and tstar0 is None and tstarf is None:
        fit_window_case = Path(__file__).resolve().parents[4]/'configs/fit_window'/cylinder_type/'overall.json'
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    groups, parameter_names = trajectory_groups(selection, parameterization)
    if (input_feature_library_case is None) != (parameterization == 'individual'):
        raise ValueError('Parameterized assessments require input_feature_library_case; individual assessments omit it.')
    state_config = load_config(state_feature_library_case)
    parameter_config = load_config(input_feature_library_case) if parameter_names else None
    derivative_config = load_config(differentiator_case) if layout.rate_names else None
    filtering_config = load_config(filtering_case) if filtering_case is not None else None
    filtering = initiate_object(**filtering_config) if filtering_config is not None else None
    library = prepare_library(state_config)
    if parameter_names:
        library = ps.ParameterizedLibrary(feature_library=library,
            parameter_library=prepare_library(parameter_config), num_features=len(layout.response_names),
            num_parameters=len(parameter_names))
    settings = Path(f'dynamics_case_{dynamics_case}')
    if filtering_case is not None:
        settings /= 'filtering='+Path(filtering_case).stem
    settings /= Path(differentiator_case).stem if derivative_config else 'no_state_differentiation'
    settings /= 'individual' if not parameter_names else 'parametric_'+parameterization
    settings /= 'state_'+Path(state_feature_library_case).stem
    if parameter_names:
        settings /= 'input_'+Path(input_feature_library_case).stem
    settings /= 'rank_rtol=auto' if rank_rtol is None else f'rank_rtol={rank_rtol:g}'
    base = (analysis_directory(results_root, selection.case_name, cylinder_type,
                               selection.re, selection.mstar)/'feature_library_assessment'
            if output_dir is None else Path(output_dir)/selection.case_name)
    requested = [dict(LpD=lpd, VR=vr, requested_tstar0=selection.bounds[(lpd,vr)][0],
                      requested_tstarf=selection.bounds[(lpd,vr)][1])
                 for lpd in selection.lpds for vr in selection.vrs]
    folder = training_directory(base/settings, 'vr' if cylinder_type=='single' else 'vr_lpd',
                                requested, fit_window_case)
    snapshot = dict(schema_version=1, dynamics_case=dynamics_case, cylinder_type=cylinder_type,
        parameterization=parameterization, parameter_names=parameter_names,
        response_names=list(layout.response_names), input_names=list(layout.response_names)+parameter_names,
        Re=selection.re, mstar=selection.mstar, batch_size=batch_size, rank_rtol=rank_rtol,
        processing_order='load tau; filter full measurements; prepare states/controls; crop in t*; append parameters; transform',
        derivative_coordinate='tau=t*/VR', data_dtype='float64', column_normalization='none',
        fit_window=None if fit_window_case is None else dict(name=Path(fit_window_case).stem,
            source=str(fit_window_case), config=load_config(fit_window_case)),
        configurations=dict(state_feature_library=state_config, parameter_feature_library=parameter_config,
                            differentiator=derivative_config, filtering=filtering_config))
    records, pooled_records, individual_reports, pooled_reports = [], [], [], []
    normalized_records, normalized_pooled_records = [], []
    for pairs in groups:
        samples, trajectories = [], []
        for lpd, vr in pairs:
            print(f'Feature-library assessment: Case {dynamics_case}, VR={vr}, L/D={lpd}', flush=True)
            differentiator = initiate_object(**derivative_config) if derivative_config is not None else None
            values, metadata = prepare_assessment_samples(selection, dynamics_case, vr, lpd,
                                                         differentiator, filtering, parameter_names)
            samples.append(values)
            trajectories.append(metadata)
        assessment = assess_feature_library(library, samples, snapshot['input_names'],
                                             batch_size=batch_size, rank_rtol=rank_rtol)
        del samples, values
        for result, metadata in zip(assessment.individual, trajectories):
            destination = folder/'individual'
            if metadata['LpD'] is not None:
                destination /= 'LpD='+number_text(metadata['LpD'])
            destination /= 'VR='+number_text(metadata['VR'])
            parameters = {'V_R':metadata['VR']}
            if metadata['LpD'] is not None:
                parameters['L/D'] = metadata['LpD']
            report = _save_report(result, destination, dict(snapshot, scope='trajectory', trajectory=metadata),
                                  'Feature Correlation\n'+format_parameter_title(parameters), show, normalized_only)
            individual_reports.append(report)
            records.append(dict(VR=metadata['VR'], LpD=metadata['LpD'], sample_count=result.sample_count,
                feature_count=result.feature_count, numerical_rank=result.rank,
                condition_number=result.condition_number, rank_deficient=result.rank_deficient,
                condition_json=str(destination/'condition_number.json')))
            normalized_records.append(dict(VR=metadata['VR'], LpD=metadata['LpD'], sample_count=result.sample_count,
                feature_count=result.feature_count, numerical_rank=result.normalized_rank,
                condition_number=result.normalized_condition_number, rank_deficient=result.normalized_rank_deficient,
                condition_json=str(report['normalized_condition_json'])))
        if parameter_names:
            destination = training_directory(folder/'pooled', parameterization, trajectories, None)
            report = _save_report(assessment.pooled, destination,
                dict(snapshot, scope='pooled_training_group', trajectories=trajectories),
                'Feature Correlation\nPooled Training Group', show, normalized_only)
            pooled_reports.append(report)
            pooled_records.append(dict(trajectories=trajectories, condition_number=_number(assessment.pooled.condition_number),
                numerical_rank=assessment.pooled.rank, sample_count=assessment.pooled.sample_count,
                condition_json=str(destination/'condition_number.json')))
            normalized_pooled_records.append(dict(trajectories=trajectories,
                condition_number=_number(assessment.pooled.normalized_condition_number),
                numerical_rank=assessment.pooled.normalized_rank, sample_count=assessment.pooled.sample_count,
                condition_json=str(report['normalized_condition_json'])))
    folder.mkdir(parents=True, exist_ok=True)
    heatmap = folder/'condition_number_heatmap.png'
    if not normalized_only:
        pd.DataFrame(records).to_csv(folder/'condition_number_summary.csv', index=False)
        _json(folder/'condition_number_summary.json', dict(snapshot,
            records=[dict(row, condition_number=_number(row['condition_number'])) for row in records],
            pooled_training_groups=pooled_records,
            heatmap_scope='individual trajectory matrices, including tensor columns when parameterized',
            heatmap_color_scale='logarithmic for finite values; gray cells for infinite condition numbers'))
        heatmap = plot_condition_heatmap(records, selection.vrs, selection.lpds,
            heatmap, single=cylinder_type=='single', show=show)
    pd.DataFrame(normalized_records).to_csv(folder/'condition_number_normalized_summary.csv', index=False)
    _json(folder/'condition_number_normalized_summary.json', dict(snapshot,
        column_normalization='unit_l2_without_centering',
        records=[dict(row, condition_number=_number(row['condition_number'])) for row in normalized_records],
        pooled_training_groups=normalized_pooled_records,
        heatmap_scope='individual trajectory matrices, including tensor columns when parameterized',
        heatmap_color_scale='logarithmic for finite values; gray cells for infinite condition numbers'))
    normalized_heatmap = plot_condition_heatmap(normalized_records, selection.vrs, selection.lpds,
        folder/'condition_number_normalized_heatmap.png', single=cylinder_type=='single', show=show, normalized=True)
    return dict(individual_reports=individual_reports, pooled_reports=pooled_reports,
                summary_csv=folder/'condition_number_summary.csv', summary_json=folder/'condition_number_summary.json',
                condition_heatmap=heatmap, records=records, pooled_records=pooled_records,
                normalized_records=normalized_records, normalized_pooled_records=normalized_pooled_records,
                normalized_summary_csv=folder/'condition_number_normalized_summary.csv',
                normalized_summary_json=folder/'condition_number_normalized_summary.json',
                normalized_condition_heatmap=normalized_heatmap)
