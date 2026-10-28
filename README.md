# SINDy for VIV

Research code for a Master's thesis applying SINDy to single and tandem
oscillating-cylinder VIV data.

The importable package is under src/sindy_viv. Command-line entry points in
script delegate to its computation/evaluation APIs. Configuration files under
configs select differentiators, state/parameter libraries, optimizers and
trajectory-owned fitting windows.

For function/class inventories, API examples and call relationships, start with
the [source package guide](src/sindy_viv/README.md) and
[CLI guide](script/README.md). Complex workflow folders have their own guides;
small, self-describing utility folders use source docstrings.

`run_all.bat` runs four workflows sequentially: single-cylinder data diagnostics,
tandem data diagnostics, filtered single-cylinder experiments, then filtered
tandem experiments. Use `run_all.bat --dry-run` to inspect all 218 Python commands.
Failures are reported while later workflows continue.

`run_diagnostics_and_single.bat` runs only both data-diagnostic batches and the
20 active single-cylinder experiments (184 Python commands total). It ends
after single-cylinder evaluation. Both fitting runners preserve Part F and
E05-E08 as `REM ON HOLD` calls, excluded from execution until explicitly
reactivated. Active single-cylinder IDs are A01-A04, B01-B04, C01-C04, D01-D04 and E01-E04.

- [Computation workflows](src/sindy_viv/case_bryan/compute_sindy/README.md)
- [Derivative, integral, frequency, and coupled-motion evaluation](src/sindy_viv/case_bryan/evaluate_sindy/README.md)
- [Derivative plots, stacked FFT spectra, amplitudes, and saved NRMSE summaries](src/sindy_viv/case_bryan/analysis/README.md)
- [Wake feature libraries](src/sindy_viv/case_bryan/feature_library/README.md)
- [Reusable feature-matrix correlation and conditioning](src/sindy_viv/feature_library/README.md)
- [Fitting-window configurations](configs/fit_window/README.md)
- [Optional filtering and smoothing](configs/filtering/README.md)

Run the data-diagnostic checklist with `run_new_data_diagnostics_single.bat`
or `run_new_data_diagnostics_two_tandem.bat`; add `--dry-run` to inspect each
runner's 72 commands. They cover raw signals, frequency/amplitude/phase,
filtering comparisons and unfiltered/filtered derivative assessments with aggregates
over the configured windows. The tandem runner includes both cylinders,
VR 5-14 and all five L/D values.

Run `run_new_single.bat` to fit and evaluate the 20 active filtered single-cylinder
experiments in Parts A-E. Use `run_new_single.bat --dry-run` to inspect all
40 fitting/evaluation commands. Parts A/B use both `ols.json` and
`stlsq_vanilla_basic.json` (alpha=0, threshold=0.1); Parts C-E use OLS only.
Column normalization is off. A/B use state degree 1, with VR degree 2 in B;
individual C-E fits use polynomial degrees 1-4. Optimizers have separate result folders.
The runner includes the requested derivative,
IVP, frequency/amplitude, phase and coefficient diagnostics; see the
[experiment breakdown](src/sindy_viv/case_bryan/evaluate_sindy/README.md#filtered-single-cylinder-experiment-runner).

Run `run_new_two_tandem.bat` for the 17 active tandem experiments
(34 fitting/evaluation commands), covering both cylinders, VR 5-14 and
L/D 1.5, 2, 3, 3.5 and 4. Use `run_new_two_tandem.bat --dry-run` to inspect
the commands. A/B use OLS and vanilla STLSQ (alpha=0, threshold=0.1);
C-E use OLS at degrees 2-4. Part B parameterizes VR separately at each L/D. Part F and E05-E08 are currently on hold. See the
[tandem experiment breakdown](src/sindy_viv/case_bryan/evaluate_sindy/README.md#filtered-two-tandem-cylinder-experiment-runner).

Use `script/assess_feature_library.py` for per-trajectory feature-correlation
matrices, condition-number JSONs, and single/tandem condition heatmaps under
the dataset's analysis tree (`Re = 150/Mstar = 5.1/feature_library_assessment`
for single-cylinder data). It supports individual and parameterized
tensor libraries, including pooled training-group reports; see the
[assessment usage guide](src/sindy_viv/case_bryan/feature_library/README.md#feature-library-assessment).

CSV timestamps are fluid time t*. Processed loading converts them once to
float64 dynamic time tau=t*/VR. Preparation and differentiation do not divide
time by VR again. Fitting/evaluation bounds use recorded t*; state/control
storage defaults to float64 and can explicitly use float32.

DataFrame-based raw loaders and conversion helpers now live in
sindy_viv.case_bryan.compatibility.dataframe_loading and dataframe_conversion;
imports through input/preprocessing remain compatible. Bryan's specialized
uniform timestamp reconstruction is retained. WeakSINDy supports Cases 1-5,
controls, and individual or parametric fits. Weak integration settings live in
[configs/weak_feature_library](configs/weak_feature_library/README.md), separately
from the pointwise response/parameter bases in `configs/feature_library`.

Public and nontrivial internal functions document their array/layout contracts
and carry type annotations. Annotations support readers and static tooling;
runtime validation still checks shapes, finite values and physical selections.

Run the regression suite with the project's Python environment:

    python -m unittest discover -s tests

Tests use temporary directories outside the repository for generated artifacts.

Bryan-specific support modules are grouped by purpose:

- `prepare_sindy/case_layout.py`: state and measured-control column definitions.
- `prepare_sindy/trajectory.py`: prepared arrays, derivatives, and parameter columns.
- `prepare_sindy/filtering.py`: optional SciPy Savitzky-Golay, Butterworth, or moving-average smoothing after loading and before differentiation.
- `model_storage/fit_windows.py`: fitting-window validation and cropping.
- `model_storage/paths.py`: artifact paths and saved-model discovery.
- `evaluate_sindy/model_evaluation.py`: prediction, integration, and reports.
- `compatibility/`: historical DataFrame loading/conversion and wake pickle callables.

The small `feature_library/legacy.py` import alias is retained only so existing
model pickles can resolve their previous callable path. New fitting uses the
descriptive compatibility module directly.

The full-data polynomial experiment is reproducible and resumable with
`python script/polynomial_sindy_study.py --output results/polynomial_study_20261018 --workers 2 --memory-fraction 0.7 --max-fit-memory-gib 3.13`.
Models exceeding its RAM estimate are recorded as skipped.

Add `--report-only` to regenerate study tables and figures without fitting models.

Time-history, FFT, and SINDy evaluation plots accept `--plot_variables` and
`--plot_cylinders`. Omit either option to include all available values.
Cylinder 0 is the single/upstream cylinder; cylinder 1 is the downstream cylinder.
Variable families are `h`, `hdot`, `hddot`, `CL`, `CLdot`, `CLddot`, and `CD`;
exact LaTeX channel names are also accepted. Filters select plotted channels,
so derivative comparisons use derivative names such as `hddot` and `CLddot`.
Only variables available in that dataset or dynamics case can be selected.

For example, plot displacement and lift for the downstream tandem cylinder:

    python script/frequency_analysis_two_tandem.py --lpd_list 1.5 --vr_list 5 --plot_variables h CL --plot_cylinders 1

The Python plotting functions and `evaluate_sindy` expose the same
`plot_variables` and `plot_cylinders` arguments. Evaluation metrics and model
states remain complete. Explicit plot selections get distinct figure filenames.

Both FFT scripts accept `--fft_window rectangular` (default), `hann`, `hamming`,
`blackman`, `blackmanharris`, `bartlett`, or `flattop`. The Python
`plot_frequency_analysis` function accepts the same `fft_window` argument.
For example, append `--fft_window hann` to either frequency-analysis command.
This taper is separate from the selected t* interval (`tstar0`/`tstarf`).
To use trajectory-specific intervals from an existing fitting-window JSON instead,
pass `--fit_window_case` to either raw frequency script:

    python script/frequency_analysis_single.py --vr_list 3 4 5 6 7 8 --fit_window_case configs/fit_window/single/sinusoidal.json
    python script/frequency_analysis_two_tandem.py --lpd_list 1.5 --vr_list 5 6 --fit_window_case configs/fit_window/two_tandem/overall.json

Each selected VR (and L/D for tandem data) uses its own `tstar0`/`tstarf` entry;
null endpoints select the available start/end. Do not combine this option with
explicit `--tstar0`/`--tstarf`. The configuration name is included in output
filenames, and JSON summaries record the requested and actual intervals.
Fill any selected `TO_BE_FILLED` entries before using a partially completed
configuration such as the tandem `sinusoidal.json` preset.

Windows are periodic, applied after mean removal, and normalized by their sum
for single-sided amplitudes. Nonrectangular windows are identified in figure
filenames so they can be compared with the default rectangular results.

`run_new_single_alpha5e-1.bat` is a historical, standalone incremental runner
for C21-C24 and D21-D24 (8 experiments, 16 commands). These alpha=0.5 checks
are excluded from the current checklist and `run_all.bat`.

`run_new_single_ols.bat` runs the 16 OLS-only single-cylinder configurations
and matching evaluations with `ols.json` (STLSQ alpha=0, threshold=0,
normalization off). Results use the separate `ols` optimizer folder; Part F
and E05-E08 remain on hold. This standalone runner does not repeat raw-data
diagnostics and is not added to `run_all.bat`.

`run_new_single_poly5_ols.bat` is a separate incremental run for Parts C-E:
individual Dynamics Cases 1, 2 and 4 with `polynomial_5.json`, OLS, Butterworth
filtering and sinusoidal training at VR 3-8. Its six commands fit the three
configurations and run their usual assessment suites, including phase comparisons
for Cases 1/2 and the coupled IVP for Case 4. Use `--dry-run` to inspect them.
It is excluded from `run_all.bat` and does not start the deferred tandem work.

`run_feature_library_assessment.bat` runs the individual feature-library
assessment for Dynamics Cases 1-4 crossed with polynomial degrees 1-4, first
for single VR 3-8, then tandem VR 5-14 and L/D 1.5, 2, 3, 3.5, 4. Both use
Butterworth filtering and sinusoidal fit windows; FD constructs lift-rate
states where needed. Its single/tandem component batches each contain 16
commands. The 32-command workflow writes correlation/condition reports under
the dataset's `analysis` tree under `feature_library_assessment`, with Re/Mstar
folders for single-cylinder data; it fits no SINDy model.
Use `--dry-run` to inspect the selection.

Feature assessments save both raw and unit-L2-column condition numbers, with
separate `condition_number_normalized.json`, normalized metadata, summary
tables and `condition_number_normalized_heatmap.png`. Use
`assess_feature_library.py --normalized_only` with the same selection options
to add normalized reports while preserving existing raw/correlation outputs.

## Adaptive overnight study

`run_overnight.bat` executes the approved tandem grid, its derivative-only
conditional Poly5 check, the single-cylinder ridge sweep, the strict single
Poly5 gate, and the selected threshold sweeps. It delegates all fitting and
full assessments to the existing CLIs. It does not rerun raw-data diagnostics.
Use `--dry-run` for input validation without fitting or output creation.

See the [orchestrator guide](script/README.md#adaptive-overnight-study) for
selection rules, logs, retained-term reports, and checkpoint resume. The
standalone `run_all.bat` retains its four fixed workflows; it does not start the
adaptive ridge/threshold stages.
