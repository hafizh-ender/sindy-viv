# Raw-data and saved-evaluation plots

## Functions and dependencies

| Module | Functions/classes | Purpose |
| --- | --- | --- |
| `data_windows.py` | `DataSelection`, `data_selection`, `load_trajectory`, `iter_raw_windows` | Validate physical/time selections and yield aligned raw trajectories or cropped windows. |
| `data_windows.py`, `paths.py` | `result_id`, `channel_token`, `save_figure`, `analysis_directory` | Identify settings/channels, save/close figures, and construct physical analysis directories. |
| `frequency.py` | `FrequencyAnalysis`, `compute_frequency_analysis`, `complete_frequency_analysis`, `dominant_frequency_analysis`, `compute_phase_difference` | Compute FFT arrays and dominant-bin/shared-reference phase summaries. |
| `amplitude.py` | `compute_signal_amplitudes`, `compute_amplitude`, `plot_amplitude_single`, `plot_amplitude_two_tandem` | Compute the selected amplitude definition, save per-case records, and plot parameter summaries. |
| `filtering_diagnostics.py` | `assess_filtering` | Compare full-record filtered measurements against raw signals on the same reporting window. |
| `diagnostic_plots.py` | `plot_derivatives`, `plot_frequency_waterfalls`, `plot_frequency_surfaces` | Present prepared derivatives and independent discrete spectra across parameters. |
| `frequency_summary.py`, `phase_summary.py` | `plot_frequency_vr_summary`, `plot_phase_vr_summary` | Read saved numerical frequency reports and plot VR/L/D trends. |
| `derivative_ivp_summary.py` | `plot_derivative_ivp_errors` | Aggregate the saved numerical-derivative reconstruction errors. |
| `error_heatmaps.py` | `style_heatmap_grid`, `style_heatmap`, `heatmap_colorbar`, `fit_heatmap_annotations`, `save_nrmse_heatmaps` | Keep grids, annotation sizing and NRMSE rendering consistent across producers. |
| `integration.py` | Commented historical prototype, no callable API | Active sampled-derivative reconstruction is `evaluate_sindy.derivative_assessment.reconstruct_from_derivatives`. |
| `plot_cli.py` | `raw_plot_parser` and the `*_main` dispatchers | Share validation/CLI selection logic across analysis scripts. |

Raw diagnostics begin with `data_selection` and processed loaders; they need no
saved SINDy model. Summary readers begin with saved JSON/CSV reports and avoid
repeating the underlying computation. Numerical derivative reconstruction lives
in `evaluate_sindy.derivative_assessment`; fitted-model prediction lives in
`evaluate_sindy`. These functions reuse the arithmetic and presentation helpers
above. FFT figure construction is in the parent `frequency_plot.py`, separate
from FFT arithmetic, so plotting does not change the numerical results.

For numerical FFT/phase analysis without creating files:

```python
import numpy as np
from sindy_viv.case_bryan.analysis.frequency import (
    compute_frequency_analysis, compute_phase_difference,
)

tstar = np.arange(1000) / 100
signals = np.column_stack((
    np.cos(2*np.pi*.5*tstar),
    .8*np.cos(2*np.pi*.5*tstar+.3),
))
spectrum = compute_frequency_analysis(tstar, signals, ["h_0", "C_{L0}"])
phase = compute_phase_difference(spectrum, "C_{L0}", "h_0")
summary = spectrum.summary()
```

`spectrum` retains native FFT bins for plotting or further analysis. `summary`
contains serializable dominant-bin metadata; `phase` uses the displacement
reference bin for both channels rather than subtracting unrelated peak phases.

## Running diagnostics

Run these commands from the repository root with the project Python environment.
All plot commands save figures at 300 DPI and close them; add `--show` to display
them where supported. Rerun the plot commands to regenerate existing figures at
300 DPI.
`--plot_variables` accepts existing families (`h`, `hdot`, `hddot`, `CL`,
`CLdot`, `CLddot`, `CD`) or exact channel names. `--plot_cylinders` selects
cylinder 0 and/or 1. Each command supports `--help`.
Figure titles, legends and parameter axes render reduced velocity as `$V_R$`
and cylinder spacing as `$L/D$` using Matplotlib math text.
Single-cylinder titles omit fixed Re and mass-ratio values. Their physical
settings remain in configuration metadata and existing result-folder identities.
Raw time-series, computed-derivative and filtering comparison figures use the
same standard width and height per variable. Filtering comparisons retain raw
and filtered line labels, with the standard centered title and no separate
upper-right figure legend.
Filtering FFT and aggregate panels are square. Raw curves are black and
filtered curves red; only the first filtering FFT panel has a legend.
Frequency-analysis CLIs save separate `_amplitude.png` and `_phase.png` figures
with square panels and one shared numerical JSON report per trajectory.
Raw FFT amplitude/phase titles omit Re and mass ratio for both single and tandem
data, retaining VR, tandem L/D and cylinder number, and the selected time interval.
Individual raw/filtering/model FFT figures use a compact 4.5-inch canvas width,
25% smaller in width and height than the previous layout, retaining text sizes
and square panels. Aggregate parameter-summary figures keep their existing sizes.
All heatmaps use 0.6-point white inner grid lines and 0.8-point black outer
spines. Custom inner grid widths are capped at the outer spine width.
NRMSE values below 0.01 use three-significant-digit scientific annotations
(exponent -3 and below). Heatmap cells grow as needed to contain annotations
without reducing their shared 9-point font. Coefficient heatmaps have no title
and use three decimal places with 8-point annotations. Nonzero coefficient
magnitudes below 0.001 use signed less-than labels; saved CSV/JSON values keep
full precision. Their colorbar reads `Coefficient`.

## Single-cylinder diagnostic runner

Run `run_new_data_diagnostics_single.bat` from PowerShell or Command Prompt:

```powershell
.\run_new_data_diagnostics_single.bat --dry-run
.\run_new_data_diagnostics_single.bat
```

The dry run prints the 72 commands. The normal run executes them sequentially
for VR 3-8, using this checkout's `.venv` and source code. Every selected
trajectory uses its own bounds from `configs/fit_window/single/`, including
VR 7's distinct intervals. The former data-diagnostic batches are archived
under `old_run/`.

| Stage | Windows | Outputs |
| --- | --- | --- |
| Raw time histories | `overall_reduced`, `sinusoidal`, `sinusoidal_evaluation` | All measured channels, one figure per VR/window |
| Frequency | `overall_reduced`, `sinusoidal` | Individual h/CL FFT spectra, dominant-frequency 2D summaries with the natural-frequency curve, and waterfalls |
| Amplitude | `sinusoidal` | Half peak-to-peak and rectangular-window FFT amplitudes, with 2D summaries |
| Phase | `sinusoidal` | CL-minus-h phase divided by pi, versus reduced velocity |
| Filtering | All three windows | Raw/filtered h/CL time-series and FFT comparisons, with 2D frequency/amplitude summaries |
| Derivatives with/without filtering | All three windows | Case 2 derivative histories, trapezoid IVP/recorded-channel assessments, and matching 2D NRMSE heatmaps |

Filtering uses `butterworth_single_flim2.json` and `savgol_single_base.json`.
Each filter/window is paired with `finite_difference_basic.json` (order 2)
and `savitzky_golay_single_base.json`. A separate no-preprocessing-filter
baseline omits `--filtering_case` (Python `None`) for both differentiators
and stores assessment reports and heatmap summaries under
`derivative_assessment/dynamics_case_<N>/filtering=None/<differentiator>/`.
It runs on all three windows, adding 18 commands: histories, trapezoidal
reconstruction assessments and matching NRMSE heatmaps. SG differentiation
still uses its configured local polynomial window. Individual FFT displays cover f* = 0-2;
their peak summaries still use the full positive-frequency spectrum. Raw and
derivative time histories include all channels; the other overviews use h/CL.

The runner checks its scripts/configs before starting, computes reports before
their summaries, saves figures at 300 DPI, and overwrites matching destinations.
It counts failures and continues with later commands; its final exit code is
nonzero if any command failed. Edit `DIAG_VRS` and `DIAG_VARIABLES` near the top
to change the selected reduced velocities or overview channels.

## Two-tandem-cylinder diagnostic runner

```powershell
.\run_new_data_diagnostics_two_tandem.bat --dry-run
.\run_new_data_diagnostics_two_tandem.bat
```

This runner uses the same 72-command workflow for both cylinders across VR
5-14 and L/D = 1.5, 2, 3, 3.5, 4. Every pair takes its own recorded-time bounds
from `configs/fit_window/two_tandem/`: `overall_reduced`, `sinusoidal`, and
`sinusoidal_evaluation`. Raw and derivative histories include all channels;
frequency, amplitude and filtering overviews use h/CL from both cylinders.

The filters are `butterworth_two_tandem_flim2.json` and
`savgol_two_tandem_base.json`, each paired with `finite_difference_basic.json`
and `savitzky_golay_two_tandem_base.json` for the Case 2 derivative diagnostics.
Both differentiators also run with `filtering=None` on all three windows,
including histories, reconstruction assessments and matching heatmaps.
Filtering includes individual time-series/FFT comparisons and 2D summaries
versus both VR and L/D. Dominant-frequency summaries also include a VR/L/D
surface; amplitude summaries cover both axes and a surface. Phase plots show
CL-minus-h for each cylinder and downstream-minus-upstream displacement phase
versus VR, with separate L/D curves. Derivative assessments run before their
matching VR/L/D NRMSE heatmaps.

The tandem runner uses this checkout's `.venv` and source code, checks required
inputs, and counts failures while continuing with later commands. Change
`DIAG_VRS`, `DIAG_LPDS`, or `DIAG_VARIABLES` near the top to select cases/channels.

## Raw-analysis output layout

Raw plots and analyses share `results/<dataset>/analysis/`. Single-cylinder
results group by Re and Mstar, and tandem results by L/D. Individual analyses
live inside their VR folder; aggregate analyses live beside the VR folders.
Overall state-variable plots retain their existing filenames directly in VR.

```text
results/oscillating-cylinder-viv/analysis/Re = 150/Mstar = 5.1/
  VR = 3/
    plot_full_range.png
    frequency_analysis/...
    derivative_plot/dynamics_case_2/filtering=None/<differentiator>/fit_window=.../time=.../...
    derivative_assessment/dynamics_case_2/<differentiator>/time=.../...
  frequency_waterfalls/fft_window=.../VR=.../fit_window=.../time=.../...
  frequency_vr_summary/fft_window=.../VR=.../fit_window=.../time=.../...
  phase_vr_summary/fft_window=.../VR=.../fit_window=.../time=.../...
  amplitude/
    data/method=.../VR = 3/fit_window=.../time=.../amplitude.json
    method=.../VR=.../fit_window=.../time=.../amplitude_C_L0_vs_VR.png

results/oscillating-cylinder-tandem-viv/analysis/
  LpD = 1.5/
    VR = 5/
      plot_full_range.png
      frequency_analysis/...
      derivative_plot/dynamics_case_2/filtering=None/<differentiator>/fit_window=.../time=.../...
      derivative_assessment/dynamics_case_2/<differentiator>/time=.../...
    frequency_waterfalls/fft_window=.../VR=.../fit_window=.../time=.../...
  frequency_vr_summary/fft_window=.../LpD=.../VR=.../fit_window=.../time=.../...
  phase_vr_summary/fft_window=.../LpD=.../VR=.../fit_window=.../time=.../...
  amplitude/
    data/method=.../LpD = 1.5/VR = 5/fit_window=.../time=.../amplitude.json
    method=.../LpD=.../VR=.../fit_window=.../time=.../...
```

Tandem comparisons/surfaces spanning several L/D values use the common
`analysis` aggregate directory. Each L/D's waterfall uses that L/D directory.
Amplitude plots are aggregate; their per-trajectory JSON/CSV inputs are kept
under `amplitude/data`. SINDy fitting and model-evaluation directories, including
their error summaries, retain their existing layout.

## Available plots

All entries below are implemented. Cylinder-specific filenames have separate
`single` and `two_tandem` versions; model evaluation also has individual and
parametric versions.

| Plot | Entry point in `script/` |
|---|---|
| Raw displacement, velocity, acceleration, lift and drag histories | `plot_all_variables_single.py`, `plot_all_variables_two_tandem.py` |
| Computed derivative histories | `plot_derivative_single.py`, `plot_derivative_two_tandem.py` |
| Raw-data FFT amplitude and phase spectra | `frequency_analysis_single.py`, `frequency_analysis_two_tandem.py` |
| Feature-library Pearson matrices, raw condition numbers, and VR/L/D condition heatmaps | `assess_feature_library.py` ([guide](../feature_library/README.md#feature-library-assessment)) |
| Raw/filtered time-series and FFT overlays per VR/L/D, plus 2D dominant-frequency and amplitude comparisons versus VR and L/D | `assess_filtering.py`, `assess_filtering_single.py`, `assess_filtering_two_tandem.py` |
| FFT waterfalls with discrete VR curves | `plot_frequency_surface_all_single.py`, `plot_frequency_surface_all_two_tandem.py` (legacy names) |
| Saved dominant frequency versus VR, with tandem L/D curves and an optional surface | `plot_frequency_vr_summary_single.py`, `plot_frequency_vr_summary_two_tandem.py` |
| Raw CL-minus-h phase versus VR, plus tandem h1-minus-h0 phase | `plot_phase_vr_summary_single.py`, `plot_phase_vr_summary_two_tandem.py` |
| Time-series amplitude versus VR; tandem also versus L/D and a VR/L/D surface | `amplitude_analysis.py` |
| SINDy derivative reconstruction versus reference derivatives | `evaluate_derivative_sindy_*.py` |
| SINDy state simulation versus reference states, including coupled motion/lift | `evaluate_integral_sindy_*.py`; partial runs save requested-window and modeled-window views |
| Model/reference FFT amplitude and phase comparisons, for derivative or integral predictions | `evaluate_frequency_sindy.py --save_plot` |
| Numerical-derivative IVP reconstruction and computed/recorded hdot/hddot comparisons | `assess_derivative.py` |
| Saved numerical-derivative IVP NRMSE heatmaps | `plot_error_heatmap_derivative_assessment.py` |
| Saved SINDy IVP NRMSE heatmaps with per-trajectory evaluation windows | `plot_error_heatmap_sindy_ivp.py` |
| Saved SINDy derivative-reconstruction NRMSE heatmaps with per-trajectory evaluation windows | `plot_error_heatmap_sindy_derivative.py` |
| Saved derivative/integral NRMSE versus VR, plus tandem VR/L/D heatmaps | `plot_error_heatmap_derivative_sindy_*.py`, `plot_error_heatmap_integral_sindy_*.py` |
| Interactive recorded/numerical acceleration integration and raw/smoothed-lift structural comparisons | `check/check_derivative.py`, `check/check_smoothing.py` (display only) |

Amplitude line plots use `case_bryan.plot.create_subplots`: hidden top/right
spines, 0.8-point bottom/left spines, 0.6-point lines and ticks, 10-point tick
labels/legends, and 12-point axis labels. Small distinct markers identify
parameter groups. Titles use 14 points and short capitalized labels such as
`Time-Series Amplitude`, `Hilbert Envelope Amplitude`, or `FFT Amplitude`.
The amplitude 3D surface uses the same font sizes. Waterfall plots share these
fonts and the title `Frequency Waterfall` (with L/D for tandem cases), retaining
their original 3D layout, curve widths, colors and frequency samples.

Rerun the plotting commands to refresh existing PNGs. For saved amplitudes,
`amplitude_analysis.py --action plot` reloads the matching amplitude JSONs
without recomputing the raw-data amplitudes.

## Computed derivative histories

Filtering has its own time-series and frequency diagnostics, independent of differentiation or
model fitting. Use `--filtering_case configs/filtering/savgol_single_base.json` with the
`assess_filtering_*.py` commands. `--plots individual`, `aggregate`, or `both`
selects the figures; individual diagnostics include time-series comparisons
using `--fit_window_case` or `--tstar0`/`--tstarf`. Add `--no_time_series` to
omit these comparisons. Numerical comparison reports are always saved. See
[filtering diagnostics and examples](../../../../configs/filtering/README.md#frequency-diagnostics).

```powershell
python script/plot_derivative_single.py --dynamics_case 1 --vr_list 3 4 --differentiator_case configs/differentiator/savitzky_golay_single_base.json --fit_window_case configs/fit_window/single/sinusoidal.json --plot_variables hdot hddot CLdot
python script/plot_derivative_two_tandem.py --dynamics_case 4 --vr_list 5 --lpd_list 1.5 --differentiator_case configs/differentiator/savitzky_golay_two_tandem_base.json --fit_window_case configs/fit_window/two_tandem/sinusoidal.json --plot_variables CLddot
```

These commands construct the same state data as SINDy, differentiate the full
record, and then crop. They plot computed Xdot without comparing recorded
derivatives or running an IVP. The horizontal axis uses recorded t*, while
derivative values remain rates in tau=t*/VR. A JSON companion saves the
differentiator configuration, actual interval, sample count and channel names.

PNG and JSON filenames contain readable channel names instead of hash suffixes,
for example `derivatives_doth_0_ddoth_0_dotC_L0.png`. Settings use the same
dataset and time folders as individual fitted models:

```text
results/oscillating-cylinder-viv/analysis/Re = 150/Mstar = 5.1/VR = 3/
  derivative_plot/dynamics_case_1/filtering=None/savitzky_golay_single_base/
    fit_window=sinusoidal/time=300..600/
      derivatives_doth_0_ddoth_0_dotC_L0.png
      derivatives_doth_0_ddoth_0_dotC_L0.json
```

Tandem plots live under `analysis/LpD = <value>/VR = <value>/derivative_plot/`.
For every dynamics case, unfiltered derivative plots and their JSON companions
use `dynamics_case_<N>/filtering=None/<differentiator>/`. Named filters use
their existing `filtering=<config-stem>` branch at the same position.
Case and differentiator follow this folder, without repeating physical labels.
Explicit bounds
omit `fit_window=` and use `time=300..600`; open bounds use `time=all`,
`time=start..600`, or `time=300..end`. Each trajectory uses its own bounds,
so different VR windows remain readable. `--output_dir` replaces the base
directory while retaining LpD/VR, case, differentiator and time folders to
separate individual trajectories.
Matching settings and channels overwrite the same PNG/JSON, including when
the data or differentiator config contents change. Different channel selections
have different readable filenames. Existing files with hash suffixes remain
in their previous folders; rerunning writes to the new layout.

## Stacked FFT spectra over discrete VR

```powershell
python script/plot_frequency_surface_all_single.py --vr_list 3 4 5 6 7 8 --fit_window_case configs/fit_window/single/sinusoidal.json --plot_variables h CL --fft_window hann
python script/plot_frequency_surface_all_two_tandem.py --vr_list 5 6 7 --lpd_list 1.5 2.0 --fit_window_case configs/fit_window/two_tandem/sinusoidal.json --plot_variables CL --plot_cylinders 0 1
```

One figure is saved per channel and, for tandem data, per L/D. Each VR has its
own frequency-versus-amplitude curve, placed at a fixed VR coordinate in 3D.
Only selected VR values appear as axis ticks. Curves retain their native FFT
bins; there is no surface or interpolation between VR cases, including when
the analysis windows have different durations. The original script filenames
are retained for command compatibility.

The JSON, NPZ and PNG share a readable channel filename, such as
`frequency_waterfall_C_L0.png`, without a hash suffix. Settings are stored in
folders using the fitted-model conventions. For example:

```text
results/oscillating-cylinder-viv/analysis/Re = 150/Mstar = 5.1/frequency_waterfalls/
  fft_window=hann/VR=3,4,5,6,8/
    fit_window=overall_reduced/time=100..650/
      frequency=0..1/max_points=2000/frequency_waterfall_C_L0.png
```

For trajectory-dependent bounds, the shared SINDy path builder uses
`time=per-trajectory-<mapping ID>`, with the exact bounds retained in JSON.
Tandem waterfalls live in `analysis/LpD = <value>/frequency_waterfalls/`;
single-cylinder waterfalls live beside the VR folders under Re/Mstar.
Explicit time bounds omit the
`fit_window=` folder; open bounds use `time=all`, `time=start..600`, or
`time=300..end`. Frequency limits and point caps have their own folders.
Matching settings overwrite the same three files, even if the measured data
change. Adding another plotted variable does not change existing channel paths.
`--output_dir` replaces the base directory while retaining these setting folders.

`--xlim 0 0.5` selects a frequency band. `--max_frequency_points 2000` caps bins
per curve: if needed, each frequency block retains its largest native peak.
Original dominant frequencies/amplitudes/phases remain in JSON. NPZ files
store `vr`, `point_counts`, and concatenated `frequencies`/`amplitudes` vectors;
`trace_offsets[i]:trace_offsets[i+1]` selects the curve for `vr[i]`.

For both raw plot families, `--fit_window_case` supplies analysis intervals
by VR, or by L/D then VR. Alternatively use `--tstar0`/`--tstarf`, but do not
combine them with a fit-window config. Null endpoints use available endpoints;
selected `TO_BE_FILLED` values must be completed. Figures normally go under
the physical-case paths shown above, under `analysis/`.

## Dominant frequency versus VR from saved analyses

Run `frequency_analysis_single.py` or `frequency_analysis_two_tandem.py` first.
The summary scripts read their existing `frequency_analysis_*.json` files;
they do not load raw measurements or compute an FFT. For example:

```powershell
python script/frequency_analysis_single.py --vr_list 3 4 5 6 7 8 --fit_window_case configs/fit_window/single/sinusoidal.json --plot_variables h CL
python script/plot_frequency_vr_summary_single.py --vr_list 3 4 5 6 7 8 --fit_window_case configs/fit_window/single/sinusoidal.json --plot_variables h CL --include_natural_frequency

python script/frequency_analysis_two_tandem.py --vr_list 5 6 7 8 9 10 11 12 13 14 --lpd_list 1.5 2 3 3.5 4 --fit_window_case configs/fit_window/two_tandem/sinusoidal.json --plot_variables h CL
python script/plot_frequency_vr_summary_two_tandem.py --vr_list 5 6 7 8 9 10 11 12 13 14 --lpd_list 1.5 2 3 3.5 4 --fit_window_case configs/fit_window/two_tandem/sinusoidal.json --plot_variables h CL --include_natural_frequency
```

Single-cylinder output has one dominant-frequency-versus-VR figure per selected
channel. Tandem output has one figure per channel with a separate labeled curve
for every selected L/D; use `--plot_cylinders 0` or `1` to select a cylinder.
Add tandem `--surface` for an additional 3D plot with L/D horizontally, VR along
the other horizontal axis, and dominant f* vertically. A selection with only one
VR or L/D is shown as points. Unresolved frequencies (for example constant drag)
remain gaps in curves and blank/null values in CSV/JSON.

`--include_natural_frequency` overlays the structural reference `f_N*=1/VR`.
Both measured peaks and the reference use `f*=fD/U_inf`. These are the saved
dominant positive-frequency FFT bins; spectrum display limits do not redefine
the dominant peak.

Window and taper selectors must match the original analysis. Use its
`--fit_window_case`, or the same `--tstar0`/`--tstarf`, plus the same `--fft_window`
(default `rectangular`). Omitting all time selectors chooses saved full-record
analyses. The summary does not crop existing spectra. Missing trajectories or
channels cause an error asking you to run frequency analysis first, before any
summary output is written. Equivalent reports from different spectrum display
limits/channel subsets can be merged; conflicting saved results cause an error.

Figures and per-channel CSV/JSON tables use readable names under
`analysis/Re = .../Mstar = .../frequency_vr_summary/` for single data and
`analysis/frequency_vr_summary/` for tandem data, for example:

```text
fft_window=rectangular/LpD=1.5,2,3,3.5,4/VR=5,6,7,8,9,10,11,12,13,14/
  fit_window=sinusoidal/time=600..1400/natural_frequency=on/
    frequency_C_L0_vs_VR.png
    frequency_C_L0_vs_VR.csv
    frequency_C_L0_vs_VR.json
    frequency_C_L0_surface.png  # when --surface is supplied
```

Single data use `Re = .../Mstar = ...` before the summary category. Window folders
follow the same shared SINDy conventions as amplitude and waterfall summaries.
Each exported row retains the original frequency, FFT amplitude, cosine phase,
frequency resolution, requested/actual time window and source JSON paths.
`--results_root` selects the root containing saved analyses; `--output_dir`
overrides the summary base while retaining the setting folders. Matching
settings overwrite matching files. Line plots use the time-series plot style.

## Raw-data phase differences versus VR

```powershell
python script/plot_phase_vr_summary_single.py --fit_window_case configs/fit_window/single/sinusoidal.json
python script/plot_phase_vr_summary_two_tandem.py --fit_window_case configs/fit_window/two_tandem/sinusoidal.json
```

Single data produce `(phi_CL0 - phi_h0)/pi`. Tandem data produce that plot plus
`(phi_CL1 - phi_h1)/pi` and `(phi_h1 - phi_h0)/pi`, with one curve per L/D.
The defaults cover all VR and L/D cases. Use `--vr_list` and tandem `--lpd_list`
to select a subset. Instead of `--fit_window_case`, explicit `--tstar0` and
`--tstarf` select one interval for every trajectory; omitting all bounds uses
the full record. `--fft_window` selects the taper (default `rectangular`).

These commands load the raw h and CL channels and compute their FFTs together;
they do not require earlier frequency analysis. Each pair uses both phases at
the reference displacement's dominant positive-frequency bin: h0 for CL0-h0
and h1-h0, h1 for CL1-h1. This avoids subtracting phases from different
frequencies when the two signals have different dominant peaks. The cosine
phase difference is wrapped into [-pi, pi], then divided by pi; positive means
the first signal leads the reference. Unresolved phases remain gaps/N/A.

Figures use 300 DPI, the data time-series line/font style, and y limits
[-1.05, 1.05]. PNG/CSV/JSON files are saved under `phase_vr_summary` at the
physical aggregate level, with the usual FFT-window, L/D/VR, fit-window and
time subfolders. Example filenames are `phase_CL0_minus_h0_vs_VR.png` and
`phase_h1_minus_h0_vs_VR.png`. Matching settings overwrite existing outputs;
CSV/JSON files retain the common frequency, radians, normalized phase and
selected time intervals. `--results_root`, `--output_dir` and `--show` are
supported. `run_data_diagnostics.bat` includes commands for both configurations.

## Amplitude computation and saved-result plots

```powershell
python script/amplitude_analysis.py --cylinder_type single --vr_list 3 4 5 6 7 8 --fit_window_case configs/fit_window/single/sinusoidal.json --plot_variables h CL --method peak_to_peak
python script/amplitude_analysis.py --cylinder_type two_tandem --vr_list 5 6 --lpd_list 1.5 2.0 --fit_window_case configs/fit_window/two_tandem/sinusoidal.json --plot_variables h CL --method peak_to_peak
```

The default `--action both` computes and saves per-trajectory amplitudes, then
loads those files to plot them. Use `--action compute` to save JSON/CSV only;
use `--action plot` to reload matching saved results without reading raw CSVs.
Results live under the physical aggregate's `analysis/.../amplitude/` folder.
Single-cylinder plots show
amplitude versus VR. Tandem plots also show amplitude versus L/D and a joint
VR/L/D surface, separately for each channel.

The default is the time-series amplitude `peak_to_peak`: `(max(x)-min(x))/2`
within the selected time window. The method name refers to the range, but the
reported amplitude is **half** that range. It uses window-wide extrema rather
than averaging amplitudes over cycles. Thus transients and isolated
extrema affect it. Select a steady interval when comparing steady oscillations.
Existing commands with an explicit `--method fft` still request spectral
amplitude; remove that flag or replace it with `--method peak_to_peak`.

Files no longer have hash suffixes. Per-trajectory results follow individual
SINDy dataset/window folders, for example:

```text
results/oscillating-cylinder-viv/analysis/Re = 150/Mstar = 5.1/amplitude/
  data/method=peak_to_peak/VR = 3/
    fit_window=sinusoidal/time=300..600/amplitude.json
    fit_window=sinusoidal/time=300..600/amplitude.csv
```

Figures and combined summary JSON/CSV live directly under `amplitude/`, grouped
by method, LpD/VR selections and fit-window/time folders. Re/Mstar (single)
already belongs to the physical parent directory and is not repeated.
For example, a tandem summary uses:

```text
method=peak_to_peak/LpD=1.5,2/VR=5,6/fit_window=sinusoidal/time=600..1400/
  amplitude_summary.json
  amplitude_summary.csv
  amplitude_C_L0_vs_VR.png
  amplitude_C_L0_vs_LpD.png
  amplitude_C_L0_surface.png
```

As in fitted models, per-trajectory results use `LpD = <value>/VR = <value>`
while combined plots use sorted selection folders. Different per-case bounds
use SINDy's `time=per-trajectory-<mapping ID>` folder. Explicit/open bounds use
the same conventions as waterfall plots. `--output_dir` replaces the figure
base directory while retaining method, dataset and window folders. Matching
settings overwrite matching files; changing the selected channels does not
change a channel's figure filename. Saved JSON records the amplitude definition.

All methods remove the channel mean (which does not affect the range):

| Method | Reported amplitude |
|---|---|
| `peak_to_peak` (default) | Half the range of the selected time series |
| `hilbert` | Mean magnitude of the analytic-signal envelope; includes endpoint effects |
| `rms` | sqrt(2) times the RMS of the centered signal |
| `fft` | Amplitude at the dominant positive-frequency bin, with FFT-window gain correction |

The methods agree for an adequately sampled sinusoid observed over whole
periods with a matching Fourier bin, but measure different properties of
irregular/multifrequency signals. FFT amplitude isolates the strongest frequency
component, which is useful for separating a fundamental from its harmonics;
it does not measure the full time-series excursion. `--fft_window` applies
only to `fft`, and adds a separate `fft_window=<name>` folder only for that method.
Constant channels have zero oscillation amplitude.

Python entry points in `analysis.amplitude` are `compute_signal_amplitudes`
(array-only), `compute_amplitude` (load/crop/save), `plot_amplitude_single`,
and `plot_amplitude_two_tandem` (read saved results). The last two accept the
same selectors and method, plus `output_dir`/`show`; use `variables` for channel
families in these Python APIs. The numerical frequency module remains free of
data loading, plotting and file writes.

## Title-free diagnostic and SINDy error heatmaps

These commands read previously saved errors and never repeat differentiation,
fitting or integration. Both save one 300-DPI heatmap per selected state, with
VR horizontally and L/D vertically for tandem data. Single-cylinder data use
one row across VR. Figures have no title; case, differentiator, model settings
and windows are identified by their folders and channel filenames. CSV and
JSON companions retain NRMSE values, source files and actual selected intervals.

For numerical-derivative IVP diagnostics, first run `assess_derivative.py` over
the desired bounds. Then aggregate those reports, for example:

```powershell
python script/plot_error_heatmap_derivative_assessment.py --cylinder_type two_tandem --dynamics_case 2 --differentiator_case configs/differentiator/savitzky_golay_two_tandem_base.json --tstar0 850 --tstarf 900
python script/plot_error_heatmap_derivative_assessment.py --cylinder_type single --dynamics_case 2 --differentiator_case configs/differentiator/savitzky_golay_single_base.json --fit_window_case configs/fit_window/single/sinusoidal_evaluation.json
```

Here `--fit_window_case` supplies **evaluation bounds**, independently for each
trajectory. Use it instead of `--tstar0`/`--tstarf`. The single example reads
450..500 reports for VR 3,4,5,6,8 and 850..900 for VR 7. With no window selectors,
the numerical-diagnostic summary reads modern `time=all` reports. Migrated
reports without time subfolders are also supported. `--integration_method`
optionally selects a saved reconstruction method, such as `trapezoid`;
reports from different integration methods are not silently combined.

Numerical-diagnostic summaries live at the physical aggregate level under
`analysis/.../derivative_assessment/dynamics_case_<n>/<differentiator>/summary/`,
followed by `integration_method=...`, LpD/VR selection, optional fit-window
name and `time=...`. Filenames include `derivative_ivp_nrmse_h_0_heatmap.png`
and `derivative_ivp_nrmse_summary.csv`/`.json`.

Diagnostic and suite error heatmaps share square cells (equal axis scaling),
visible borders, a white cell grid, and contrast-aware value annotations. Missing
or undefined errors display N/A. Rerun the plot commands to refresh existing PNGs;
the saved metrics and output paths retain their existing conventions.
Numerical-derivative diagnostic summaries use figures 25% larger than the base
heatmap dimensions and the shared 0.6-point inner grid.

For the SINDy IVP version, first run the corresponding
`evaluate_integral_sindy_*.py` command over each requested evaluation window.
Then use:

```powershell
python script/plot_error_heatmap_sindy_ivp.py --cylinder_type two_tandem --dynamics_case 3 --differentiator_case configs/differentiator/savitzky_golay_two_tandem_base.json --state_feature_library_case configs/feature_library/polynomial_1.json --optimizer_case configs/optimizer/stlsq_basic.json --training_fit_window_case configs/fit_window/two_tandem/sinusoidal.json --fit_window_case configs/fit_window/two_tandem/sinusoidal_evaluation.json
```

In this new SINDy command, `--training_fit_window_case` identifies the fitted
model, while `--fit_window_case` identifies evaluation bounds. Alternatively
omit the latter and give `--tstar0 850 --tstarf 900`. These heatmaps summarize
integrated `XSINDy` versus recorded `X`. Use `--coupled_motion` for saved coupled
Case 4/5 simulations. Parametric and weak model selectors are the same as in
the existing SINDy error scripts. Incomplete IVPs are excluded from complete-
window summaries. Their missing cells can be shown with `--allow_missing`.

SINDy outputs retain the fitting/evaluation chain, with `reference=...`,
`evaluation_window=<name>` when supplied, dataset selection, training
`fit_window=<name>` and evaluation `time=...` folders. Their filenames use
`integral_nrmse_*_heatmap.png` or `coupled_integral_nrmse_*_heatmap.png`.
For direct SINDy derivative reconstruction (`XdotSINDy` versus `Xdot`), use
`plot_error_heatmap_sindy_derivative.py` with the same model and window options
as the IVP script:

```powershell
python script/plot_error_heatmap_sindy_derivative.py --cylinder_type two_tandem --dynamics_case 3 --differentiator_case configs/differentiator/savitzky_golay_two_tandem_base.json --state_feature_library_case configs/feature_library/polynomial_1.json --optimizer_case configs/optimizer/stlsq_basic.json --training_fit_window_case configs/fit_window/two_tandem/sinusoidal.json --fit_window_case configs/fit_window/two_tandem/sinusoidal_evaluation.json
```

Run `evaluate_derivative_sindy_*.py` first for those scoring windows. This script
selects derivative reports and derivative channel names, and saves
`derivative_nrmse_*_heatmap.png` and `derivative_nrmse_summary.csv`/`.json`.
The SINDy derivative and IVP scripts have identical window semantics:
`--training_fit_window_case` selects the fit, while `--fit_window_case` supplies
per-trajectory scoring bounds instead of `--tstar0`/`--tstarf`. Both produce
title-free 300-DPI heatmaps with separate derivative/integral filenames.

All three commands support `--vr_list`, `--lpd_list`, `--plot_variables`,
`--plot_cylinders`, `--allow_missing`, `--results_root` and `--output_dir`.
Missing reports cause an actionable error; ambiguous reports are rejected.
Matching output paths overwrite previous summaries. NRMSE remains a ratio,
not a percentage.

## NRMSE summaries from existing SINDy reports

First run the corresponding derivative/integral evaluation. These commands
read the saved metrics; they do not refit models or repeat simulation:

```powershell
python script/plot_error_heatmap_derivative_sindy_single.py --dynamics_case 1 --vr_list 3 4 --differentiator_case configs/differentiator/savitzky_golay_single.json --state_feature_library_case configs/feature_library/polynomial_3.json --optimizer_case configs/optimizer/stlsq_basic.json --fit_window_case configs/fit_window/single/sinusoidal.json --tstar0 300 --tstarf 600
python script/plot_error_heatmap_integral_sindy_single.py --dynamics_case 1 --vr_list 3 4 --differentiator_case configs/differentiator/savitzky_golay_single.json --state_feature_library_case configs/feature_library/polynomial_3.json --optimizer_case configs/optimizer/stlsq_basic.json --fit_window_case configs/fit_window/single/sinusoidal.json --tstar0 300 --tstarf 600
python script/plot_error_heatmap_derivative_sindy_two_tandem.py --dynamics_case 1 --vr_list 5 6 --lpd_list 1.5 2.0 --differentiator_case configs/differentiator/savitzky_golay_two_tandem.json --state_feature_library_case configs/feature_library/polynomial_3.json --optimizer_case configs/optimizer/stlsq_basic.json --tstar0 600 --tstarf 1400
python script/plot_error_heatmap_integral_sindy_two_tandem.py --dynamics_case 1 --vr_list 5 6 --lpd_list 1.5 2.0 --differentiator_case configs/differentiator/savitzky_golay_two_tandem.json --state_feature_library_case configs/feature_library/polynomial_3.json --optimizer_case configs/optimizer/stlsq_basic.json --tstar0 600 --tstarf 1400
```

Single-cylinder scripts plot VR versus NRMSE. Tandem scripts save both curves
for each L/D and L/D-by-VR heatmaps. NRMSE values retain the ratio stored in the
evaluation JSON; they are not converted to percent. CSV and JSON companions
record plotted values and source reports.

Model selectors match evaluation: `--parameterization individual` (default),
`vr`, `lpd`, or `vr_lpd`, `--input_feature_library_case`, `--weak`,
`--weak_feature_library_case`, `--training_vr_list`, `--training_lpd_list`,
and `--model_dir`. `--individual` and `--parameterized` are aliases for
`individual` and `vr`. Integral Case 4/5 scripts accept `--coupled_motion`.

Here `--fit_window_case` identifies the training configuration; it does not
change the scoring interval of an existing report. `--tstar0`/`--tstarf` and
`--evaluation_differentiator_case` select the evaluation report. Multiple
matching reports cause an error rather than selecting the newest one.
`--allow_missing` draws missing trajectories as gaps/gray cells and records
their reasons; at least one valid report must exist.

Default summaries go in `evaluation/summary/` under the common directory of
the selected models. Reference differentiator, sorted LpD/VR selections,
training-window name (when supplied), and evaluation time have readable
subfolders. PNG, CSV and JSON filenames have no hash suffixes. For example:

```text
evaluation/summary/reference=savitzky_golay_two_tandem_base/
  LpD=1.5,2,3,3.5,4/VR=5,6,7,8,9,10,11,12,13,14/
    fit_window=sinusoidal/time=850..900/
      derivative_nrmse_ddotC_L0_heatmap.png
      derivative_nrmse_ddotC_L0_vs_VR.png
      derivative_nrmse_summary.csv
      derivative_nrmse_summary.json
```

Integral and coupled-integral summaries use `integral_nrmse` and
`coupled_integral_nrmse` prefixes respectively. The `time=` folder represents
the evaluation selectors: `time=all` when neither bound is supplied,
`time=start..900` or `time=850..end` for open bounds. Actual report windows
remain in the summary JSON. All plot families accept `--output_dir`, which
replaces the base while retaining the setting/time subfolders; raw analysis
and model discovery use `--results_root`. Matching output paths overwrite
existing files. Old hashed files remain in their previous locations; rerun
the summary command to write the new layout.
