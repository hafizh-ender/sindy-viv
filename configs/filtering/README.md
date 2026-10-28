# Filtering and smoothing

Filtering is optional. Add `--filtering_case configs/filtering/savgol_single_base.json`
to a standard or weak fitting command, derivative/integral/frequency evaluation,
evaluation suite, derivative assessment, or derivative plotting command.
Python APIs accept the same `filtering_case` keyword. Omit it for the existing
unfiltered workflow.

The processing order is:

1. Load measurements and convert timestamps to float64 tau = t*/VR.
2. Filter every loaded measured channel independently along sample axis 0.
3. Construct states and controls, then calculate required derivatives in tau.
4. Crop to the fitting or evaluation window and fit/evaluate as usual.

Measured displacement, velocity, lift, and any loaded acceleration controls all
receive the same filter. Derived lift rates are calculated from filtered lift;
constant VR/L/D parameter columns are appended afterward. Filtering does not
resample, trim, shift timestamps, change channel order, or modify input arrays.
State/control storage retains the requested float32/float64 precision.

| Config | SciPy implementation | Settings |
| --- | --- | --- |
| `savgol_single_base.json`, `savgol_two_tandem_base.json` | [`signal.savgol_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html), `deriv=0` | Odd `window_length` in samples, `polyorder` smaller than the window, `mode`, `cval` |
| `butterworth_*_base.json`, `butterworth_*_flim2.json` | `signal.butter(output="sos")` followed by [`signal.sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html) | `cutoff`, `order`, `btype`, `normalized`, `padtype`, `padlen`; forward/backward filtering avoids phase lag |
| `moving_average_single_base.json` | [`ndimage.uniform_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html), `origin=0` | Odd `window_length` in samples, `mode`, `cval`; centered average |

These are starting presets; tune the bandwidth for the sampling interval and
signal frequencies of the selected dataset. Butterworth `normalized=true`
expresses cutoff as a fraction of Nyquist, strictly between 0 and 1. With
`normalized=false`, cutoff is in cycles per unit tau, and sampling frequency is
inferred from tau without another VR scaling. Bandpass/bandstop use two cutoff
values. SciPy validates cutoffs and padding lengths. Sample-window filters reject
windows larger than the record; all filters require finite measurements and
uniform, strictly increasing timestamps. Invalid settings raise errors rather
than silently shortening windows or skipping trajectories.

Configs use the same `module` / `class_name` / `kwargs` descriptor format as
differentiators. Implementations live in
[`prepare_sindy/filtering.py`](../../src/sindy_viv/case_bryan/prepare_sindy/filtering.py).
Direct preparation also accepts a `(data, time)` callable via
`prepare_trajectory(..., filtering=...)`.

Filtered model folders insert the config stem immediately before the
differentiator folder:

```text
results/<dataset>/dynamics_case_<N>/filtering=<config-stem>/<differentiator>/<fit-mode>/...
```

The remaining library, optimizer, dataset and fitting-window folders follow the
existing layout. Unfiltered paths remain unchanged. Use the same
`--filtering_case` when evaluating the model; it selects this folder and applies
the same preprocessing to evaluation measurements, including when `--model_dir`
is supplied. `training.json` snapshots the filtering config, name, source and
processing order. Assessment/derivative-plot outputs also include the filtering
config folder and snapshot.

For example, append the option to a single-cylinder fit:

```powershell
python script/perform_sindy_single_individual.py --dynamics_case 2 --vr_list 3 --differentiator_case configs/differentiator/savitzky_golay_single_base.json --state_feature_library_case configs/feature_library/polynomial_3.json --optimizer_case configs/optimizer/stlsq_basic.json --filtering_case configs/filtering/savgol_single_base.json
```

## Frequency diagnostics

Run time-series and frequency diagnostics independently of SINDy fitting or differentiation:

```powershell
uv run python script/assess_filtering_single.py --filtering_case configs/filtering/savgol_single_base.json --vr_list 3 4 5 6 7 8 --plot_variables h CL --fit_window_case configs/fit_window/single/sinusoidal.json
uv run python script/assess_filtering_two_tandem.py --filtering_case configs/filtering/butterworth_two_tandem_flim2.json --vr_list 5 6 7 --lpd_list 1.5 2 3 --plot_variables h CL --tstar0 300 --tstarf 600
```

`script/assess_filtering.py` provides the same workflow with
`--cylinder_type single` (default) or `--cylinder_type two_tandem`. The Python API
is `sindy_viv.case_bryan.analysis.filtering_diagnostics.assess_filtering`.

- `--plots both` (default) generates raw/filtered time-series comparisons and
  FFT overlays for every VR/L/D pair, plus 2D aggregate frequency plots.
  Select `individual` or `aggregate` to generate
  either set of figures alone.
- Time-series plots overlay raw and filtered measurements on the same recorded
  t* axis, with one panel per selected channel. They use the standard time-series
  comparison renderer: 6 inches wide, 1 inch per channel (minimum height 1.5 inches), the shared fonts and
  line styles, mean-centered y-axis padding covering both signals, and 300 DPI.
  Use `--fit_window_case` for
  trajectory-specific windows or `--tstar0`/`--tstarf` for explicit bounds;
  do not combine the two forms. Omitted endpoints use the available record.
  Filtering always precedes cropping. Each `filtering_time_series.png` has a
  JSON companion recording requested/actual bounds, channels and filter settings.
  Add `--no_time_series` to keep only the existing frequency diagnostics.
- Individual spectra have one panel per selected measured channel. Use
  `--plot_variables` and `--plot_cylinders` to choose channels, and `--xlim 0 1`
  to restrict the displayed f* range. Dominant-bin metrics always use all FFT
  bins, independently of display limits.
- Aggregates show dominant frequency and dominant-bin amplitude versus VR,
  with separate L/D curves for tandem data. Tandem diagnostics also plot
  versus L/D, grouped by VR. Select `--aggregate_axis vr`, `lpd`, or `both`
  (default). Single-cylinder diagnostics use VR.
- `--fft_window hann` (or another existing FFT window) applies identically to
  raw and filtered spectra; the default is `rectangular`. `--amplitude_floor`
  sets the resolution threshold. Constant/unresolved frequencies remain null
  in JSON and gaps in line plots.
- `--save_spectra` also exports full raw/filtered FFT amplitudes and their
  shared frequency bins in NPZ files.

The filter receives the full record in tau before raw and filtered signals are
cropped to identical reporting windows. FFTs use recorded t*, so frequencies
are f*=cycles/t*, without another VR conversion. Each trajectory is loaded and
released individually. Per-trajectory JSONs snapshot both peak summaries and
the filtering config; aggregate CSV/JSON reports are always saved, even when
only individual figures are requested. CSVs also report frequency shifts and
filtered/raw amplitude ratios at the original raw peak. The raw and filtered
dominant peaks are selected independently and may represent different bins.

Outputs follow the existing raw-analysis placement:

```text
results/<dataset>/analysis/[Re/Mstar or LpD]/VR = <VR>/
  filtering_diagnostics/filtering=<config>/fft_window=.../amplitude_floor=.../channels=.../
    fit_window=.../time=.../filtering_frequency.json
    fit_window=.../time=.../filtering_time_series.png
    fit_window=.../time=.../filtering_time_series.json
    fit_window=.../time=.../display_frequency=.../filtering_spectra.png

results/<dataset>/analysis/[Re/Mstar]/filtering_diagnostics/
  filtering=<config>/fft_window=.../amplitude_floor=.../channels=.../
    [LpD=...]/VR=.../fit_window=.../time=.../
      filtering_frequency_summary.csv
      filtering_frequency_summary.json
      filtering_<channel>_vs_VR.png
      filtering_<channel>_vs_LpD.png
```

Explicit time bounds omit `fit_window=...`; `--output_dir` overrides the root
and separates `single`/`two_tandem`, `individual`, and `aggregate` results.
Matching settings overwrite the same artifacts. Figures are saved at 300 DPI;
add `--show` to display them too.
