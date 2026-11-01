# SINDy for VIV

Research code for a Master's thesis applying SINDy to single and tandem
oscillating-cylinder VIV data.

The importable package is under src/sindy_viv. Command-line entry points in
script delegate to its computation/evaluation APIs. Configuration files under
configs select differentiators, state/parameter libraries, optimizers and
trajectory-owned fitting windows.

`src/sindy_viv` is independent of repository scripts and scratch work. Its
numerical defaults are bundled with the installed package. Scripts aggregate
the package APIs; durable outputs belong in `results/`, while `state/` is for
temporary orchestration and validation. Data CSVs and explicitly selected
configuration/model files are external inputs, not external code dependencies.

For function/class inventories, API examples and call relationships, start with
the [source package guide](src/sindy_viv/README.md) and
[CLI guide](script/README.md). Complex workflow folders have their own guides;
small, self-describing utility folders use source docstrings.

- [Computation workflows](src/sindy_viv/case_bryan/compute_sindy/README.md)
- [Derivative, integral, frequency, and coupled-motion evaluation](src/sindy_viv/case_bryan/evaluate_sindy/README.md)
- [Derivative plots, stacked FFT spectra, amplitudes, and saved NRMSE summaries](src/sindy_viv/case_bryan/analysis/README.md)
- [Wake feature libraries](src/sindy_viv/case_bryan/feature_library/README.md)
- [Reusable feature-matrix correlation and conditioning](src/sindy_viv/feature_library/README.md)
- [Fitting-window configurations](configs/fit_window/README.md)
- [Optional filtering and smoothing](configs/filtering/README.md)

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
uniform timestamp reconstruction is retained. Tandem processed loading first
removes consecutive reporting repeats only when the recorded time and all ten
motion/force channels exactly match the preceding record. Detection reads all ten
channels in float64, including for subset selections, before any float32 cast.
The shared cleanup precedes uniform reconstruction, acceleration alignment and
conversion to tau. Genuine samples with identical rounded timestamps are retained;
raw CSVs and the compatibility raw loaders are unchanged. The reconstructed clock
still uses the nominal initial step (0.00288675 in the audited L/D=1.5, VR=5 data),
so its precision remains limited by the printed timestamps. Validation across all
50 tandem trajectories found 502000 raw rows each, 500001 after removing 1999
reporting repeats, and 500000 after the existing acceleration-alignment trim. Previously saved tandem models/diagnostics must be
regenerated to reflect this corrected sampling; they are not migrated automatically.

Supported dynamics cases are **1, 2, 3, 4, 6, and 7**; Case 5 was removed.
Case 6 learns CLdot from CL with h/hdot controls. Case 7 learns CLddot
from CLdot with h/hdot/CL controls. Their suites add complete motion/lift IVPs
using theoretical or saved Case 3 acceleration coefficients with exact
kinematic identities. See the [Case 6/7 CLI guide](script/README.md#lift-only-identification-cases-6-and-7).

WeakSINDy supports Cases 1, 2, 3, 4, 6, and 7,
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
Selected `TO_BE_FILLED` entries must be filled before use; the current single
`first_half.json` still contains placeholders. The sinusoidal presets are populated.

Windows are periodic, applied after mean removal, and normalized by their sum
for single-sided amplitudes. Nonrectangular windows are identified in figure
filenames so they can be compared with the default rectangular results.

Feature assessments save both raw and unit-L2-column condition numbers, with
separate `condition_number_normalized.json`, normalized metadata, summary
tables and `condition_number_normalized_heatmap.png`. Use
`assess_feature_library.py --normalized_only` with the same selection options
to add normalized reports while preserving existing raw/correlation outputs.

## Study orchestration and retained inputs

The old run/review scripts, legacy `script/old/` implementations, and previous
Case 6/7 studies and models have been removed. The subsequent first-order lift
experiment setup, diagnostics and generated outputs have also been removed.
The new [full recomputation controller](script/FULL_RECOMPUTE.md) refits Case 6
from CFD data: six single-cylinder cases locally and nine tandem cases through
three full SLURM jobs. Outputs use `result/experiment/` (singular `result`).
The supported Case 6/7 APIs remain available for individual operations.

Reusable Case 3 coefficients remain in
[`inputs/case3_motion/`](inputs/case3_motion/README.md). These fitted numerical
inputs are independent of the erased studies. Raw CFD data remain external.

`script/evaluate_theoretical_two_tandem.py` and
`script/refresh_tandem_coefficient_heatmaps.py` are retained for theoretical
Case 3 assessment and migration of older combined coefficient reports.
See the [CLI guide](script/README.md#theoretical-tandem-assessment-and-legacy-coefficient-plots).

`state/` contains thesis work and temporary audit records. Numerical code in
`src/sindy_viv` does not depend on scratch code or deleted study controllers.

Thesis-specific figure/table generators are grouped in
[`script/thesis/`](script/thesis/README.md). They export to local
`img/03_methodology` and `img/04_results` folders with copyable LaTeX blocks;
manuscript text and placement are deferred. See the illustrated
[plotting settings](MANUSCRIPT_PLOTTING_SETTINGS.md). General scientific fitting,
evaluation and diagnostic commands remain directly under `script/`.
