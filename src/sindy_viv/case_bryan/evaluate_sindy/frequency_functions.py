"""Frequency reports for reconstructed derivatives and simulated SINDy states."""

from __future__ import annotations
from os import PathLike
from typing import Any
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sindy_viv.case_bryan.analysis.frequency import FrequencyAnalysis, compute_frequency_analysis
from sindy_viv.case_bryan.model_storage.paths import number_text
from .functions import _iter_sindy_predictions


def compare_frequency_analysis(
    reference: FrequencyAnalysis, prediction: FrequencyAnalysis,
) -> dict[str, dict[str, Any]]:
    """Compare peaks and wrapped phase at the reference peak on identical grids.

    Dominant phases at different peak frequencies are recorded independently,
    not subtracted. Phase differences use the same reference-frequency bin and
    are undefined when either signal has unresolved amplitude there. Constant
    channels have no frequency or phase comparison. Frequency errors are
    prediction minus reference; phase is wrapped into [-pi, pi].
    """
    if (reference.variable_names != prediction.variable_names
            or reference.sample_count != prediction.sample_count
            or reference.tstar_region != prediction.tstar_region
            or reference.fft_window != prediction.fft_window
            or not np.array_equal(reference.frequencies, prediction.frequencies)):
        raise ValueError('Frequency comparisons require identical names, time grids, intervals and FFT windows.')
    ref = reference.summary()['channels']
    pred = prediction.summary()['channels']
    result = {}
    for j, name in enumerate(reference.variable_names):
        r, p = ref[name], pred[name]
        index = int(reference.peak_indices[j])
        frequency_error = None if r['frequency'] is None or p['frequency'] is None else p['frequency']-r['frequency']
        phase = None
        if index >= 0:
            rphase, pphase = reference.phases[index, j], prediction.phases[index, j]
            if np.isfinite(rphase) and np.isfinite(pphase):
                phase = float(np.angle(np.exp(1j*(pphase-rphase))))
        result[name] = dict(
            reference_frequency=r['frequency'], prediction_frequency=p['frequency'],
            frequency_difference=frequency_error,
            relative_frequency_error=None if frequency_error is None else frequency_error/r['frequency'],
            reference_dominant_amplitude=r['amplitude'], prediction_dominant_amplitude=p['amplitude'],
            reference_dominant_phase_rad=r['phase_rad'], prediction_dominant_phase_rad=p['phase_rad'],
            dominant_bins_match=index >= 0 and index == int(prediction.peak_indices[j]),
            phase_comparison_frequency=r['frequency'],
            phase_difference_rad_at_reference_frequency=phase,
            prediction_amplitude_at_reference_frequency=None if index < 0 else float(prediction.amplitudes[index, j]),
        )
    return result


def _evaluate_frequency(
    kind: str, dynamics_case: int, optimizer_case: str | PathLike[str], *,
    compare_reference: bool, fft_window: str, amplitude_floor: float,
    save_spectra: bool, save_plot: bool, return_data: bool, options: dict[str, Any],
) -> dict[float | tuple[float, float], dict[str, Any]]:
    """Process and release one model trajectory at a time, writing frequency artifacts."""
    from sindy_viv.case_bryan.analysis.frequency import FFT_WINDOWS
    if fft_window not in FFT_WINDOWS:
        raise ValueError(f'fft_window must be one of {FFT_WINDOWS!r}.')
    # Plot/scoring controls belong to the time-domain evaluator; this module owns frequency reports.
    options = dict(options)
    options.pop('evaluation_type', None)
    result = {}
    for item in _iter_sindy_predictions(
        dynamics_case, optimizer_case, evaluation_type=kind, with_reference=compare_reference,
        save_plot=False, return_data=True, **options,
    ):
        tstar = item['time']*item['vr']
        predicted = compute_frequency_analysis(tstar, item['prediction'], item['variable_names'],
                                               fft_window=fft_window, amplitude_floor=amplitude_floor)
        reference = None
        comparison = None
        if compare_reference:
            reference = compute_frequency_analysis(tstar, item['reference'], item['variable_names'],
                                                   fft_window=fft_window, amplitude_floor=amplitude_floor)
            comparison = compare_frequency_analysis(reference, predicted)
        report_dir = item['report_dir'] / 'frequency'
        report_dir.mkdir(parents=True, exist_ok=True)
        start, stop = (number_text(value) for value in tstar[[0, -1]])
        stem = (f"{item['kind']}_frequency_{start}_{stop}_"
                f"{item['reference_differentiator_name']}_fft={fft_window}")
        prefix = report_dir / stem
        model_path = prefix.with_name(stem+'_model.csv')
        pd.DataFrame.from_dict(predicted.summary()['channels'], orient='index').rename_axis('variable').to_csv(model_path)
        parameters = {name: value.item() if isinstance(value, np.generic) else value
                      for name, value in item['parameter_values'].items()}
        payload = dict(kind=item['kind'], model_path=str(item['model_dir'] / 'model.csv'),
                       parameters=parameters, model=predicted.summary(),
                       reference=None if reference is None else reference.summary(), comparison=comparison,
                       amplitude_floor=float(amplitude_floor), solver=item['solver'])
        # Do not let decimal endpoints be interpreted as a file extension.
        metadata_path = report_dir / (stem+'.json')
        metadata_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        record = dict(model_path=item['model_dir'] / 'model.csv', frequency_path=metadata_path,
                      dominant_frequency_path=model_path, model=predicted.summary(), comparison=comparison)
        if reference is not None:
            reference_path = report_dir / (stem+'_reference.csv')
            comparison_path = report_dir / (stem+'_comparison.csv')
            pd.DataFrame.from_dict(reference.summary()['channels'], orient='index').rename_axis('variable').to_csv(reference_path)
            pd.DataFrame.from_dict(comparison, orient='index').rename_axis('variable').to_csv(comparison_path)
            record.update(reference=reference.summary(), reference_frequency_path=reference_path,
                          comparison_path=comparison_path)
        if save_spectra:
            arrays = dict(frequencies=predicted.frequencies, model_amplitudes=predicted.amplitudes,
                          model_phases=predicted.phases, variable_names=np.array(predicted.variable_names))
            if reference is not None:
                arrays.update(reference_amplitudes=reference.amplitudes, reference_phases=reference.phases)
            spectra_path = report_dir / (stem+'_spectra.npz')
            np.savez_compressed(spectra_path, **arrays)
            record['spectra_path'] = spectra_path
        if save_plot:
            import matplotlib.pyplot as plt
            from sindy_viv.case_bryan.frequency_plot import plot_frequency_comparison
            figure, _ = plot_frequency_comparison(predicted, reference, title=item['kind'])
            try:
                figure_path = report_dir / (stem+'.png')
                figure.savefig(figure_path, dpi=200)
                record['figure_path'] = figure_path
            finally:
                plt.close(figure)
        if return_data:
            record.update(prediction_spectrum=predicted, reference_spectrum=reference)
        result[item['key']] = record
        del item, predicted, reference, tstar
    return result


def evaluate_frequency_derivative(
    dynamics_case: int, optimizer_case: str | PathLike[str], *, compare_reference: bool = True,
    fft_window: str = 'rectangular', amplitude_floor: float = 0., save_spectra: bool = False,
    save_plot: bool = False, return_data: bool = False, **evaluation_options: Any,
) -> dict[float | tuple[float, float], dict[str, Any]]:
    """Reconstruct XdotSINDy, save its dominant frequencies, optionally compare Xdot.

    evaluation_options are the model/data selectors accepted by evaluate_sindy,
    including normal/weak, training windows and reference differentiator config.
    compare_reference=False computes model spectra alone and skips numerical
    derivative targets (lift-rate states still require differentiation). Analysis
    uses the selected evaluation interval in t*, never tau frequencies.
    CSV and JSON reports live under the model's evaluation/.../frequency folder.
    """
    return _evaluate_frequency('derivative', dynamics_case, optimizer_case,
                               compare_reference=compare_reference, fft_window=fft_window,
                               amplitude_floor=amplitude_floor, save_spectra=save_spectra,
                               save_plot=save_plot, return_data=return_data, options=evaluation_options)


def evaluate_frequency_integral(
    dynamics_case: int, optimizer_case: str | PathLike[str], *, compare_reference: bool = True,
    fft_window: str = 'rectangular', amplitude_floor: float = 0., save_spectra: bool = False,
    save_plot: bool = False, return_data: bool = False, **evaluation_options: Any,
) -> dict[float | tuple[float, float], dict[str, Any]]:
    """Simulate XSINDy, save its dominant frequencies, optionally compare measured X.

    Uses the same selectors and solver settings as integral evaluate_sindy.
    coupled_motion=True closes Case 4/5 motion using the structural equations;
    structural_mstar selects the physical mass ratio. Both model-only and
    comparison modes write dominant frequency/amplitude/phase reports. Phase
    comparisons always use a common frequency bin and the same time origin.
    """
    return _evaluate_frequency('integral', dynamics_case, optimizer_case,
                               compare_reference=compare_reference, fft_window=fft_window,
                               amplitude_floor=amplitude_floor, save_spectra=save_spectra,
                               save_plot=save_plot, return_data=return_data, options=evaluation_options)
