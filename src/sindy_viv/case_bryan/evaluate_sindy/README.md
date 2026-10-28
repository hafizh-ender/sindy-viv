# Bryan-case SINDy evaluation

## Functions and dependencies

| Module | Main functions | Input/output relationship |
| --- | --- | --- |
| `functions.py` | `evaluate_sindy` | Resolve a saved model, prepare one trajectory, predict derivatives or integrate an IVP, and save metrics/plots. |
| `model_evaluation.py` | `load_coefficients`, `predict_derivative`, `pointwise_predictor` | Rebuild named coefficient evaluation without refitting. |
| `model_evaluation.py` | `simulate`, `solve_state_ivp`, `reference_limit_options`, `report_evaluation` | Integrate with finite-state/recovery handling and report complete or accepted-prefix trajectories. |
| `derivative_assessment.py` | `reconstruct_from_derivatives`, `assess_derivatives` | Reintegrate numerical derivatives independently of a fitted SINDy model. |
| `coupled_integration.py` | `coupled_reference`, `simulate_coupled_lift` | Close Case 4/5 motion using structural equations with the fitted lift dynamics. |
| `frequency_functions.py` | `compare_frequency_analysis`, `evaluate_frequency_derivative`, `evaluate_frequency_integral` | Transform model/reference predictions into frequency, amplitude and phase assessments. |
| `error_plots.py` | `plot_saved_errors` | Discover saved error reports and render aggregate curves/heatmaps without another fit or IVP solve. |
| `single_suite.py` | `evaluate_suite`, `evaluate_single_suite`, `evaluate_two_tandem_suite` | Orchestrate the requested windows/diagnostics and reuse each computed prediction. |
| `suite_cli.py` | `suite_parser` | Share single/tandem suite options across thin script entry points. |

Window/model identity is supplied by [model_storage](../model_storage/README.md).
State/control order comes from [prepare_sindy](../prepare_sindy/README.md), metrics
from the generic numerical module, and spectral arithmetic/shared heatmaps from
[analysis](../analysis/README.md). Function docstrings document exact arguments,
return mappings, units and partial-IVP behavior. The command examples below
describe which selectors choose training artifacts versus evaluation data.

## Evaluate saved models

`functions.py` exposes `evaluate_sindy`. Explicit `if`/`elif` branches select
`evaluation_type="derivative"` or `"integral"`, `cylinder_type="single"` or
`"two_tandem"`, and `parameterization="individual"`, `"vr"`, or `"vr_lpd"`.

| `cylinder_type` | `parameterization` | Saved model | Evaluation return keys |
| --- | --- | --- | --- |
| `single` | `individual` | One model per VR | `vr` |
| `single` | `vr` | One model across VR | `vr` |
| `two_tandem` | `individual` | One model per L/D and VR | `(lpd, vr)` |
| `two_tandem` | `vr` | One model across VR at each L/D | `(lpd, vr)` |
| `two_tandem` | `lpd` | One model across L/D at each VR | `(lpd, vr)` |
| `two_tandem` | `vr_lpd` | One model across L/D and VR | `(lpd, vr)` |

The shared function supports physical cases 1 through 5 and standard or weakly fitted
models. Validation,
coefficient alignment, numerical prediction, integration, and reporting helpers
are in `model_evaluation.py`.

For filtered models, pass the same `filtering_case` / `--filtering_case` used
for fitting. This selects `filtering=<config-stem>` before the differentiator
folder and smooths full loaded evaluation measurements before differentiation
and cropping. Derivative, integral, frequency, and suite evaluations support it.
See [filter settings and processing order](../../../../configs/filtering/README.md).

## Derivative reconstruction

```python
from sindy_viv.case_bryan.evaluate_sindy import evaluate_sindy

evaluations = evaluate_sindy(
    evaluation_type="derivative",
    cylinder_type="single",
    parameterization="vr",
    dynamics_case=2,
    differentiator_case="configs/differentiator/savitzky_golay_single.json",
    state_feature_library_case="configs/feature_library/polynomial_3.json",
    input_feature_library_case="configs/feature_library/polynomial_2.json",
    optimizer_case="configs/optimizer/constrained_sr3_basic.json",
    vr_list=[5],
    tstar0=100,
    tstarf=150,
    save_plot=True,
)
result = evaluations[5]
print(result["metrics"])
```

Derivative evaluation compares the fitted vector field `f(X, parameters)` with
the reference `X_dot` obtained from the chosen differentiator. All reference
derivatives are constructed on the complete selected trajectory before cropping,
preserving the differentiator's context at evaluation-window boundaries. Only
samples in the requested region are predicted and scored. A derivative reference
containing nonfinite samples is rejected; select an interior region when the
differentiator leaves undefined endpoint values.

## Initial-value simulation

```python
from sindy_viv.case_bryan.evaluate_sindy import evaluate_sindy

evaluations = evaluate_sindy(
    evaluation_type="integral",
    cylinder_type="two_tandem",
    parameterization="vr_lpd",
    dynamics_case=2,
    differentiator_case="configs/differentiator/savitzky_golay_two_tandem.json",
    state_feature_library_case="configs/feature_library/polynomial_3.json",
    input_feature_library_case="configs/feature_library/polynomial_2.json",
    optimizer_case="configs/optimizer/constrained_sr3_basic.json",
    vr_list=[8],
    lpd_list=[3.0],
    tstar0=100,
    tstarf=110,
    method="DOP853",
    rtol=1e-8,
    atol=1e-10,
    solver_options={"max_step": 0.01},
)
result = evaluations[(3.0, 8)]
print(result["solver"])
```

Integration uses `scipy.integrate.solve_ivp` with the first selected reference
state as its initial condition. The right-hand side evaluates the fitted library
on the evolving model state; it does not use a time-interpolated measured
derivative or reference trajectory. The prediction is compared with `X` at the
same selected sample times, including the initial condition.

Case 1 constructs the state directly from displacement, velocity, and lift and
needs no reference differentiation. Case 2 differentiates only the measured lift
signals to construct the reference CLdot states. It does not compute unused
X_dot or CLddot arrays. Both cylinders are integrated together in a tandem model.

The integration method, tolerances, and additional SciPy options are
configurable. `solver_options` may include `max_step`, `first_step`, `jac`, or
`events`, but cannot replace the equation, initial condition, or selected times.
Ordinary PySINDy `PolynomialLibrary` models cache their monomial indices once
per IVP and evaluate the same polynomial equations directly at solver stages.
This avoids rebuilding the feature-library transform for every single sample;
feature order, coefficients, and measured-control interpolation are preserved.
Custom/composite libraries retain their configured transforms.
Failed or early-terminated time-domain integrations retain the finite samples
completed on the requested sample grid. Both ordinary and coupled Case 4/5
evaluations save the partial trajectory in NPZ, diagnostics/metrics in JSON,
and (unless `--no_plot` is set) two comparison figures:

- `*_partial_requested.png`: completed reference/prediction samples, with xlim
  equal to the original requested interval, such as 300..400.
- `*_partial_modeled.png`: the same samples, with xlim limited to their completed
  interval, such as 300..320.

The figures include every modeled channel unless explicit plot filters select
a subset. Their y limits include both reference and prediction so an exploding
prediction remains visible. If failure occurs before any accepted progress,
the known initial condition is saved and shown as a point; the zoom view has
small padding around that point. No rejected trial steps are saved as predictions.

The NPZ contains `time` (tau), `time_star`, `reference`, `prediction`, and
`variable_names`. JSON sets `integration_complete=false`, distinguishes the
requested and completed intervals/sample counts, and labels the metrics
`metrics_scope="completed_interval"`. These scores describe only that prefix;
saved-error summaries exclude incomplete integrations. The CLI issues a warning
and continues with the remaining selected trajectories. Invalid configurations
and unrelated programming errors still raise normally. Successful IVPs retain
their usual one-figure output. Existing CLI commands need no new flag.

Integral evaluation uses SciPy's native terminal events to stop excessive
state magnitudes. Set `--max_state_factor N` (default `10`; finite `N > 1`).
For each modeled state `j`, its limit is:

```text
limit[j] = N * max(abs(reference[:, j]))
```

The reference is cropped to the selected evaluation window (`tstar0`..`tstarf`)
before limits are computed. Limits are constant over that IVP and independent
for every state and cylinder; measured controls and constant parameters are not
limited. Coupled Case 4/5 integration uses the full reference state
`[h, hdot, CL, CLdot]` per cylinder, so all evolved components have limits.
For an identically zero reference channel, its limit is `max(atol[j], machine
epsilon)` (or scalar `atol`) to avoid a terminal event at the initial zero state.

The event returns the smallest normalized margin to the limits, with
`terminal=True` and `direction=-1`. SciPy detects the first outward crossing and
locates its time using native root finding. Existing Python `solver_options`
events are preserved. The saved trajectory contains only requested-grid samples
at or before the event; its exact root time and state are recorded separately in
solver diagnostics. The partial trajectory, JSON, and two plots are saved as
above, and the evaluator continues with the next trajectory.

```bat
python script/evaluate_integral_sindy_single_individual.py --dynamics_case 4 --vr_list 8 --differentiator_case configs/differentiator/savitzky_golay_single.json --state_feature_library_case configs/feature_library/polynomial_2.json --optimizer_case configs/optimizer/stlsq_basic.json --fit_window_case configs/fit_window/single/overall_reduced.json --tstar0 450 --tstarf 500 --max_state_factor 5
```

All six integral evaluation CLIs accept this option, as does the frequency CLI
in integral mode. It applies to ordinary and weak fits and every parameterization.
The Python API accepts `max_state_factor=None` to disable this event. Derivative
reconstruction has no IVP and is unaffected.

The previous custom runtime/RHS-count limits and timed progress checks have been
removed. `solver_options` now accepts native SciPy options; the former
`max_wall_time`, `max_rhs_evaluations`, and `progress_interval` keys raise an
explicit error. Per-trajectory loading/integration/completion messages remain.
Native events require accepted-step progress and therefore do not impose a
wall-clock runtime guarantee if a solver struggles before a crossing.

Solver JSON includes `max_state_factor`, `reference_max_abs`, `state_limits`,
`state_limit_variable_names`, and `zero_reference_indices`. A limit-triggered
stop has native `status=1`, `termination_reason="state_limit"`, and the exact
`state_limit_event_time` (tau), `state_limit_event_tstar`,
`state_limit_event_state`, and `state_limit_reached_indices`.
The accepted-sample recorder is retained solely to recover partial results if
an unrelated numerical RHS exception occurs; the magnitude stop itself is
handled by SciPy's event machinery.

Ordinary and coupled simulations supply their RHS to the same `solve_state_ivp`
driver. It validates solver options once and uses one result-validation and
diagnostics path for both strict and partial evaluation. The recovery wrapper
only records accepted samples and returns a solver result on numerical errors;
it does not choose stopping criteria. Coupled motion/lift equations are evaluated
directly from `[h, hdot, CL, CLdot]` arrays in the canonical library order.

Diagnostics include method, tolerances, status, message, and evaluation counts.
Low-level helpers keep strict full-coverage checking by default; partial
recovery is enabled by the time-domain integration evaluator. Frequency
evaluation still requires a complete simulated interval.

An integration failure records the solver method, parameter values,
requested tau interval, last returned sample time, sample count, and state
magnitude. SciPy's "Required step size is less than spacing between numbers"
can occur when the learned dynamics diverge or become singular, as well as
with difficult numerical stiffness. A good derivative reconstruction score
does not establish that the fitted model has stable trajectories. Switching
solvers cannot repair a model whose solution diverges within the requested
interval. The reported last time refers to a returned sample, which may
precede the solver's internal stopping time.

## Selecting data and time

- `vr_list` selects the trajectories to evaluate; it is independent of the VR
  values used during parametric training. A single-item list evaluates one VR.
- Tandem evaluators select the Cartesian product of `lpd_list` and `vr_list`.
  VR-only parametric models require a saved model at each selected L/D; joint
  VR/L/D models can also evaluate new spacings when reference data is available.
- `tstar0` and `tstarf` select an inclusive interval in recorded time
  `t* = V_R * tau`. The loaders return model time `tau`. Integration and
  reference differentiation use `tau`; plots and selection use `t*`.
- Omitted bounds use the corresponding trajectory endpoints. Requested bounds
  must fall within that trajectory's available range and contain at least two
  samples. Non-grid-aligned bounds select the first and last samples inside the
  interval. No interpolation or extrapolation is needed.
- `xlim` affects the plot only. It does not select the metric or simulation
  region. `ylim_mean_time` controls plot statistics in `t*`.
- Time remains float64. `dtype` defaults to `np.float64` and controls reference
  data precision; an explicit `np.float32` remains supported. `block_size`
  bounds temporary derivative feature matrices and metric buffers.
  Full-trajectory preparation buffers are released before prediction/simulation.

## Models, weak fits, and reference differentiation

Arguments identify the saved `model.csv` using the same directory layout as
`compute_sindy`. All models take `state_feature_library_case`; parametric
models additionally require `input_feature_library_case`. Parameterization is
explicit: omitting the input library from a parametric evaluation raises an
error. Select `parameterization="individual"` and remove the input-library
argument to evaluate individual models.
Feature and derivative-column names are validated and reordered against the
saved table. Coefficients and optimizers are never refitted. The configured
pointwise libraries reconstruct the functions used in the saved CSV.

State and parameter libraries use the general `prepare_library(library_config)`
function from `sindy_viv.feature_library`, matching computation. It instantiates
libraries from parsed configuration dictionaries and preserves tensor boundaries
and input-index groups during outer multiplication without changing feature
values, names, or CSV alignment.

Set `weak=True` to read `weak_individual`, `weak_parametric_vr`,
`weak_parametric_lpd`, or `weak_parametric_vr_lpd` results. Evaluation uses the returned pointwise
equations, rather than weak integration windows. For weak cases 1/3, pass
`differentiator_case=None`. Derivative evaluation then requires
`evaluation_differentiator_case` to construct its reference. Integral evaluation
of weak cases 1/3 needs neither differentiator.

Pass `weak_feature_library_case` / `--weak_feature_library_case` with the same
weak-library config used for training. Its filename stem selects the subfolder
between `weak_individual` (or `weak_parametric_*`) and the pointwise feature
library. Omitting it selects `default`. Evaluation uses only this name and does
not load the weak integration settings. Older results without that subfolder
remain accessible through `model_dir` / `--model_dir`.

`differentiator_case` identifies the training configuration and model path.
`evaluation_differentiator_case` optionally selects a different reference
differentiator without changing the model path; otherwise the training
differentiator is reused. Cases 2 and 4 need a reference differentiator for both
evaluation methods.

All four dynamics cases are supported in all six workflows. Cases 3 and 4
evaluate forced systems: lift (case 3) or motion (cases 4/5) comes from measured
control channels, while the model states evolve independently from their
initial values. Integral evaluation interpolates those controls linearly in
tau at solver stages. Derivative evaluation uses controls at the selected
sample times. Both outputs compare only the modeled states or their derivatives.

## Outputs and CLI

Each call returns a dictionary of evaluation records with:

- `time` (model tau), `time_star`, `reference`, `prediction`, and `variable_names`.
- Per-variable `NRMSE`, computed exclusively on the selected samples.
- `model_path`, `metrics_path`, `figure_path`, and integral `solver` diagnostics.

Reports are saved below the corresponding model directory:
`evaluation/[LpD = .../]VR = .../`. Filenames include evaluation type, actual
selected t* endpoints, and reference-differentiator name. JSON includes the
model path, parameters, sample count, time regions, metrics, and solver details.
Undefined normalized metrics for constant or zero reference channels are JSON
`null` and remain NumPy NaN in the returned metrics. Plots are saved by default;
`save_plot=False` disables them. Saved figures are closed to release memory.
Metrics JSON is written before rendering the figure, so it remains available
if plotting or image saving raises an error.

All six integral CLI scripts save both outputs by default for each selected
trajectory: `integral_error_metrics_*.json` and `integral_comparison_*.png`.
The figure compares every state channel, including both cylinders and the
lift-rate states, and displays each channel's metrics. Passing
`--no_plot` writes metrics only. Reports require a successful integration over
the requested region; failed or prematurely stopped solves raise an error.

The twelve `script/evaluate_{derivative,integral}_sindy_<configuration>.py` files
parse arguments and call `evaluate_sindy` with their fixed selectors. Existing
commands retain their configuration, `--xlim`, and
`--ylim_mean_time` options. Both evaluation methods accept `--tstar0`,
`--tstarf`, `--weak`, `--evaluation_differentiator_case`, `--block_size`,
`--results_root`, and `--no_plot`. Integral CLIs also accept `--method`,
`--rtol`, `--atol`, `--max_state_factor`, and `--solver_options` as a JSON mapping (for native SciPy options).

Python callers should replace the old case-specific function imports with
`evaluate_sindy` and the three selectors. The old case modules have been removed;
return keys and scoring arguments retain their existing form; saved-model selection is described below.

All evaluation scripts use `--state_feature_library_case`. Their filenames
select the parameterization; parametric scripts still require
`--input_feature_library_case`. To use individual models, select the individual
script and remove the input-library argument.

[Reference: SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)


## L/D-only models and selecting the training fit

Use `parameterization="lpd"` for tandem models fitted across spacings at one
fixed VR. `vr_list` selects the fixed-VR models and `lpd_list` selects the data
to evaluate, including L/D values not used in fitting. The parameter library
receives only L/D, matching computation. New CLI entry points are
`evaluate_derivative_sindy_two_tandem_parametric_lpd.py` and
`evaluate_integral_sindy_two_tandem_parametric_lpd.py`.

Evaluation dataset choices and scoring bounds do not identify the training
selection. The following optional arguments select the saved fit:

- `training_vr_list` / `--training_vr_list`: VR values used to fit the model.
- `training_lpd_list` / `--training_lpd_list`: L/D values used to fit the model.
- `fit_window_case` / `--fit_window_case`: trajectory-window JSON config under
  `configs/fit_window/single/` or `configs/fit_window/two_tandem/`, shared by
  every dynamics case. This replaces the earlier `fit_window` folder selector.
- `model_dir` / `--model_dir`: exact directory containing `model.csv`.

A unique matching fit is discovered automatically. Multiple matches raise an
error listing the available paths, rather than silently selecting a newer or
older model. Training selection filters are order independent. A fixed-L/D
VR model cannot be reused at another L/D, and a fixed-VR L/D model cannot be
reused at another VR; use the joint mode for both varying parameters.

After filling the sinusoidal config and fitting with it, evaluate an L/D-only model trained on L/D={1.5,2,3} at VR=8 using
the existing L/D=3.5 dataset, while scoring t*=200..300:

```powershell
python script/evaluate_integral_sindy_two_tandem_parametric_lpd.py --dynamics_case 3 --differentiator_case configs/differentiator/savitzky_golay_two_tandem.json --state_feature_library_case configs/feature_library/polynomial_3.json --input_feature_library_case configs/feature_library/polynomial_2.json --optimizer_case configs/optimizer/stlsq_basic.json --vr_list 8 --lpd_list 3.5 --training_vr_list 8 --training_lpd_list 1.5 2 3 --fit_window_case configs/fit_window/two_tandem/sinusoidal.json --tstar0 200 --tstarf 300
```

Evaluation `tstar0` and `tstarf` remain scalar scoring bounds and may differ from
the training bounds. The evaluation directory is beneath the selected fitting
window beside its `model.csv`, `training.json`, and pickle.
Legacy models without training-selection/time folders remain readable; use
`model_dir` to select an old model explicitly when necessary. Legacy models
without training metadata cannot be filtered by their unknown training sets.


Window configs describe trajectories rather than SINDy dynamics cases. The
single-cylinder schema is `VR -> {tstar0, tstarf}`; tandem uses
`LpD -> VR -> {tstar0, tstarf}`. Config bounds are matched against the saved
training trajectories, not the held-out data being evaluated. Editing a config
will select a model fitted with the new limits; pass `model_dir` without a
window config to read an older saved version explicitly. Evaluation scoring
bounds remain independent of these training-window settings.
## Plot limits and style

All derivative and integral evaluation CLIs accept
`--ylim_source reference`, `--ylim_source prediction`, or `--ylim_source both`.
The default is `reference`. This selects the data used for y-axis limits within
`ylim_mean_time`; it does not change evaluation samples or metrics. The Python
API uses the same `ylim_source` argument. Legacy `true` and `pred` values remain
accepted by the plotting/API functions.

Active plots share the `case_bryan.plot` template: thin black, red, and blue
lines, hidden top/right spines, and six inches per subplot column. Frequency
figures reserve extra height for their multi-line titles.

All evaluation CLIs also accept `--plot_variables` and `--plot_cylinders`.
For example, `--plot_variables hddot CLdot --plot_cylinders 1` displays only
those available derivative channels for the downstream tandem cylinder.
Integral plots use state names, such as `h` and `CL`, instead of their derivatives.
Families `h`, `hdot`, `hddot`, `CL`, `CLdot`, `CLddot`, and `CD`, or exact
LaTeX names, are supported. The two filters intersect; unavailable variables
or cylinders raise ValueError. Omit either option to include all available values.

These options filter figures only: prediction, integration, and all reported
metrics retain every modeled channel. Explicit selections are recorded in the
figure filename, preserving the default full-channel figure. The Python API
uses `plot_variables` and `plot_cylinders` with the same defaults of None.



## Batch return memory and library setup

The Python API retains return_data=True for compatibility. Pass
return_data=False to retain only metrics, artifact paths, solver diagnostics,
variable names, sample_count, time_region (tau) and tstar_region. Full time,
time_star, reference and prediction arrays are then omitted from returned
records. Metrics and requested figures are generated in both modes.

All derivative/integral CLIs default to summary-only returns. --return_data
opts into full trajectories. Fitted metadata and aligned CSV coefficients are
reused per resolved model for known shape-only libraries. Unknown/data-adaptive
and reciprocal libraries retain their original per-trajectory full fitting
and validation. Shared case layouts keep fitting and integral reference
preparation consistent without changing time scaling or physical state choices.


## Numerical derivative assessment before fitting

`assess_derivatives` in `derivative_assessment.py` checks the numerical derivative
without fitting a SINDy model. It differentiates the full record first, crops
states and targets together, then performs two independent assessments:

1. Solve `X' = interpolated_Xdot(tau)` from the measured state at the first
   selected sample and compare reconstructed X with recorded X using NRMSE.
   The default `trapezoid` method exactly integrates the piecewise-linear
   derivative history. Adaptive `RK45`/`DOP853` are also available.
2. For Cases 1/2/3, compare `D(h)` with recorded hdot and `D(recorded hdot)`
   with recorded hddot, separately for each cylinder. Comparisons use matching
   timestamps, with no automatic shift to compensate for source-data staggering.

The IVP uses prescribed derivative histories, independently of any learned
SINDy equations. Cases 4/5 assess their lift states and have no motion-state
recorded-derivative comparison. No additional smoothing stage is applied.

```powershell
python script/assess_derivative.py --dynamics_case 1 --cylinder_type single --vr_list 3 --differentiator_case configs/differentiator/savitzky_golay_single.json
```

Reports and optional plots are saved under
`results/<dataset>/analysis/<physical case>/VR = <vr>/derivative_assessment/dynamics_case_<n>/<differentiator>/time=.../`.
The physical case is `Re = .../Mstar = ...` for a single cylinder or
`LpD = ...` for tandem cylinders; these labels are not repeated below
`derivative_assessment`.
Each channel has NRMSE; constant-reference normalized metrics are null. A
configuration snapshot records the full-record differentiation and same-time
comparison conventions. Use `--no_plot` for reports only.

## Frequency assessment

`case_bryan/analysis/frequency.py` contains numerical computation only:

- `compute_frequency_analysis(tstar, data, variable_names, ...)` returns full
  single-sided amplitude/phase spectra, means and dominant positive-frequency bins.
- `dominant_frequency_analysis(...)` returns a serializable dominant-bin summary.
- `complete_frequency_analysis(...)` returns the full numerical result.

Frequency is `f*=cycles/t*`; callers convert loader time using `tstar=VR*tau`.
Means are removed before applying a periodic FFT window, and amplitudes are
corrected for coherent gain. Phase is a cosine phase in radians relative to
**the first selected sample**. The mean is reported separately. Constant or
unresolved channels have null dominant frequency and phase. FFT bins are not
interpolated; reports include their frequency resolution. Frequencies require
uniform sampling. Plotting lives separately in `case_bryan/frequency_plot.py`.
The raw `script/frequency_analysis_*.py` scripts now save dominant-bin JSON
summaries beside their figures.

Model frequency functions live in `evaluate_sindy/frequency_functions.py`:

```python
from sindy_viv.case_bryan.evaluate_sindy import (
    evaluate_frequency_derivative, evaluate_frequency_integral,
)

reports = evaluate_frequency_derivative(
    dynamics_case=1,
    optimizer_case="configs/optimizer/stlsq_basic.json",
    state_feature_library_case="configs/feature_library/polynomial_3.json",
    weak=True,
    weak_feature_library_case="configs/weak_feature_library/default.json",
    vr_list=[3],
    evaluation_differentiator_case="configs/differentiator/savitzky_golay_single.json",
)
```

`evaluate_frequency_derivative` reconstructs XdotSINDy from the saved equations.
With `compare_reference=True` (default), it also constructs reference Xdot and
compares their frequency and phase. `evaluate_frequency_integral` instead
simulates XSINDy and compares it with recorded X. Both support normal and weak
models, all case/cylinder/parameter selections, model-directory selection, and
separate fitting/evaluation windows through the existing evaluation keywords.

Set `compare_reference=False` (`--model_only`) to save only model frequencies.
Derivative targets are then skipped; a differentiator is still required when
lift-rate states must be constructed. This is distinct from choosing a training
differentiator config solely to identify a standard-model directory.

Every mode saves model dominant frequency/amplitude/phase CSV and a metadata JSON
under `evaluation/.../frequency`. Comparison mode also saves reference and
comparison CSV files. JSON records the selected interval, phase origin, FFT
window, frequency resolution, model identity, and solver diagnostics.

Frequency error is prediction minus reference. The CSV records each signal's
own dominant-bin phase, but the reported phase difference uses **the reference
peak frequency for both signals**, wrapped into [-pi, pi]. It is null if either
signal has unresolved amplitude at that frequency. Phases at different peak
frequencies are never subtracted. `amplitude_floor` can mark weak amplitudes
unresolved; a numerical roundoff floor is always applied.

```powershell
python script/evaluate_frequency_sindy.py --evaluation_type derivative --dynamics_case 1 --vr_list 3 --state_feature_library_case configs/feature_library/polynomial_3.json --optimizer_case configs/optimizer/stlsq_basic.json --weak --evaluation_differentiator_case configs/differentiator/savitzky_golay_single.json
```

Change `--evaluation_type` to `integral` for state-trajectory frequency assessment.
Use `--save_plot` for spectrum comparison figures, `--save_spectra` for full NPZ
spectra, and `--fft_window` for a taper. `--tstar0`/`--tstarf` select the common
assessment interval. Additional model/data/solver selectors are listed by `--help`.
Amplitude and phase comparisons are saved as separate `_amplitude.png` and
`_phase.png` files with square panels. Returned records contain `figure_paths`,
`amplitude_figure_path`, and `phase_figure_path`; `figure_path` identifies the
amplitude figure for compatibility. Suite FFT comparisons remain amplitude-only
over f*=0..2, with square panels.
Individual FFT figures use the shared compact layout: a 4.5-inch canvas width
and 25% smaller width/height, with square panels and the existing text sizes.
Aggregate frequency/amplitude/phase trends keep their existing figure sizes.

## Closed motion/lift integration for Cases 4 and 5

Pass `coupled_motion=True` to integral `evaluate_sindy` or
`evaluate_frequency_integral`, or add `--coupled_motion` to their integral CLIs.
This evolves `[h, hdot, CL, CLdot]` per cylinder; tandem states list cylinder 0
then cylinder 1. The known structural equations supply

```text
h'    = hdot
hdot' = -4*pi^2*h + 2*VR^2/(pi*structural_mstar)*CL
```

At every solver stage the **evolving** motion variables are passed into the
fitted lift library. For Case 4, hddot is recomputed from the equation above;
for Case 5, only evolving h/hdot are supplied. Measured motion after the initial
state does not drive this simulation. Both fitted lift equations are retained;
if exact CL'=CLdot was enforced during training, that relation remains enforced.
Constant VR/L/D parameter columns are supplied in their original library order.

`structural_mstar` / `--structural_mstar` selects the physical mass ratio,
independently of the rounded dataset label. Its default is the cylinder-specific
true mass ratio from constants.py. Set it explicitly for synthetic data or
another structural model. All rates and integration times use tau=t*/VR.

Closed-loop reports use the `coupled_integral` filename prefix, so prescribed
motion and coupled results are retained separately. They include every evolved
state, per-state NRMSE, and the physical mass ratio in solver metadata. The same
mode can be assessed in the frequency domain with `--coupled_motion`.

```powershell
python script/evaluate_integral_sindy_single_individual.py --dynamics_case 4 --vr_list 3 --differentiator_case configs/differentiator/savitzky_golay_single.json --state_feature_library_case configs/feature_library/polynomial_3.json --optimizer_case configs/optimizer/stlsq_basic.json --coupled_motion
```

The feature/optimizer/differentiator configs in an evaluation command must match
an existing saved fit; these commands evaluate models rather than training them.
# Two-tandem experiment batch

Run `run_two_tandem.bat` from the repository root to execute its Parts A-D
checklist. All 70 experiments are enabled: 10 individual Case 3 configurations
in A, 10 VR-parametric Case 3 configurations in B, and 25 individual
configurations each for Cases 1 and 2 in C and D. Every command selects all
VRs 5-14 and L/D values 1.5, 2, 3, 3.5 and 4. Part B fits separately at each
L/D using a degree-2 VR library.

Each fit is followed by `evaluate_sindy_two_tandem_suite.py`. Derivatives are
assessed over `overall_reduced`, `sinusoidal` and `sinusoidal_evaluation`;
IVPs use `sinusoidal_evaluation` (currently t* 1350-1400 throughout the grid).
Derivative frequency/amplitude comparisons use the latter two windows, and
IVP spectral comparisons use the focused evaluation window. Part A adds both
cylinders' hddot coefficient heatmaps at each L/D; Parts C/D add each cylinder's
CL-minus-h IVP phase comparisons. These evaluations retain both cylinders and
aggregate across VR/L/D using the existing suite output folders.

The batch uses the project virtual environment. A failed fit skips its suite;
subsequent experiments continue after fit or evaluation failures. The final
exit code is nonzero if any command failed. Matching fits, evaluation tables
and 300-DPI figures overwrite their existing files. Running the batch performs
training as well as evaluation; it is not a summary-only plotting command.

# Single-cylinder experiment batch

From the repository root, run `run_single.bat` to fit and evaluate the 122
configurations in its commented Parts A–F checklist. The batch uses the project
virtual environment and continues after errors. A failed fit skips that fit's
evaluation suite; the batch returns a nonzero exit code at the end if any fitting
or evaluation command failed.

To regenerate figures using existing fitted models, run `run_single_suite.bat`.
It enables all 122 suite evaluations from Parts A–F, including the entries that
are currently commented out in `run_single.bat`. It recomputes derivative
predictions and IVPs using saved coefficients and overwrites matching evaluation
reports and figures. Requirements are the project virtual environment, raw CSVs,
and matching `model.csv`/`training.json` files; training settings and window JSONs
must identify those saved fits. Previously generated raw-data diagnostics are
not required. Failed suites are counted and subsequent evaluations continue;
the batch returns a nonzero exit code at the end if any suite failed.

`script/evaluate_sindy_single_suite.py` evaluates one already fitted experiment.
Its `--fit_window_case` identifies training, while `--derivative_window_cases`
and `--integral_window_case` supply per-VR evaluation bounds. Parametric fits use
`--parameterization vr`, `--input_feature_library_case`, and an explicit
`--training_vr_list` independent of the evaluation VR selection. For example:

```powershell
python script/evaluate_sindy_single_suite.py --dynamics_case 3 --vr_list 3 4 5 6 7 8 --differentiator_case configs/differentiator/finite_difference_basic.json --state_feature_library_case configs/feature_library/polynomial_1.json --optimizer_case configs/optimizer/stlsq_basic.json --fit_window_case configs/fit_window/single/sinusoidal.json --derivative_window_cases configs/fit_window/single/overall_reduced.json configs/fit_window/single/sinusoidal.json configs/fit_window/single/sinusoidal_evaluation.json --integral_window_case configs/fit_window/single/sinusoidal_evaluation.json --spectral_window_cases configs/fit_window/single/sinusoidal.json configs/fit_window/single/sinusoidal_evaluation.json --weight_heatmap
```

The suite saves derivative time comparisons for all three windows and standard
IVP comparisons for the evaluation window. Cases 4/5 additionally run the coupled
structural/lift IVP. Native SciPy events stop at `--max_state_factor` times each
reference state's maximum absolute value (default 10); existing partial-trajectory
reports and both time-series views are preserved.

Summary heatmaps have VR columns and state/derivative rows, with N/A for incomplete
IVPs and unavailable normalized errors. Frequency and amplitude curves compare
model and reference across VR, using only complete IVPs. Frequencies and FFT
amplitudes use a rectangular window; time-series amplitude is half the selected
signal range. Per-VR FFT comparison plots display f*=0..2 without phase panels.
`--phase_comparison` additionally plots CL-minus-h phase for Case 1/2 IVPs, using
the same reference-displacement peak bin for both channels and both signals.

Summaries live inside the existing fitting chain under
`evaluation/summary/reference=.../evaluation_window=.../VR=.../fit_window=.../time=.../`.
Per-VR spectral/amplitude tables live under that model's
`evaluation/VR = .../analysis/time=.../evaluation_window=.../`. JSON companions
include completion status, solver diagnostics, actual intervals, and source
reports. Matching runs overwrite matching summaries. Every figure is 300 dpi;
error and coefficient heatmaps omit titles. Frequency/amplitude trend plots have
descriptive titles and variable-only y-axis labels. `--weight_heatmap` supports
individual and parameterized Case 3 fits and plots every response feature in
the hddot equation, preserving signed small coefficients. For parameterized
models, coefficients are evaluated at each selected VR/L/D using the saved
parameter feature library; no model is refitted. Single-cylinder runs save one
map across the selected VR cases. Parameterized tandem runs save separate
hddot0 and hddot1 heatmaps at each L/D, matching individual-model reports.

All coefficient heatmaps label the colorbar **Coefficient**. In addition to
the effective VR maps, VR-parameterized models save
`coefficients/hddot_tensor_coefficients_heatmap.png` with CSV and
`hddot_tensor_coefficients_summary.json` companions. These retain the fitted
tensor features (including VR powers) without substituting parameter values.
Single-cylinder models have one `SINDy` column. Tandem VR models save separate
`hddot0_tensor_coefficients_heatmap.png` and
`hddot1_tensor_coefficients_heatmap.png` files, each with one column per L/D
and full tensor terms in its rows; CSV/JSON companions are separate as well.
The JSON records the original feature names and source model CSVs. The existing
`--weight_heatmap` flag remains supported; generated filenames use
`coefficients` instead of `weights`. Cell annotations use three decimals and
8-point text, with signed less-than labels below magnitude 0.001. CSV/JSON
values retain full precision.

The strict van der Pol acceleration library already has maximum degree three.
The separate `case4_single_selected_polynomial_4_vdp_acceleration.json` label
retains the same allowed terms for the checklist's degree-four restriction.

## Tandem evaluation suite

Manually specified coefficient CSVs can be assessed without training metadata
through `evaluate_two_tandem_suite(..., model_directories=...)`. The mapping must
cover every selected `(L/D, VR)` pair. Each directory must contain `model.csv`;
explicit library descriptors still validate every feature and target name.
For folders without `training.json`, fit/training selectors are omitted from
model lookup. The supplied fit-window name labels the report and does not imply
that the manual model was fitted. Original coefficient CSVs are never rewritten.

`run_theoretical_two_tandem.bat` evaluates the five manually entered theoretical
VR models under `.../input_polynomial_2/theoretical/LpD=<value>/fit_window=overall_reduced/`.
It covers VR 5-14 and L/D 1.5, 2, 3, 3.5 and 4 with the usual Case 3 suite:
three derivative windows, sinusoidal-evaluation IVPs, FFT/amplitude assessments,
effective coefficient maps per L/D and the aggregate map retaining VR terms.
Use `--dry-run` to validate model names/windows without evaluating trajectories.
Reports are saved under the theoretical model tree's `evaluation` directories;
status/hash manifests go to `state/theoretical_evaluation_runs` by default.
The `--after_study` option requires all predecessor stages to have finished.

`script/evaluate_sindy_two_tandem_suite.py` provides the same diagnostics for both
tandem cylinders. It defaults to VR=5..14 and L/D=1.5,2,3,3.5,4. Use `--vr_list`
and `--lpd_list` for smaller selections. Fit the models first, then run, for example:

```powershell
python script/evaluate_sindy_two_tandem_suite.py --dynamics_case 3 --vr_list 5 6 7 8 9 10 11 12 13 14 --lpd_list 1.5 2 3 3.5 4 --differentiator_case configs/differentiator/finite_difference_basic.json --state_feature_library_case configs/feature_library/polynomial_1.json --optimizer_case configs/optimizer/stlsq_basic.json --fit_window_case configs/fit_window/two_tandem/sinusoidal.json --derivative_window_cases configs/fit_window/two_tandem/overall_reduced.json configs/fit_window/two_tandem/sinusoidal.json configs/fit_window/two_tandem/sinusoidal_evaluation.json --integral_window_case configs/fit_window/two_tandem/sinusoidal_evaluation.json --spectral_window_cases configs/fit_window/two_tandem/sinusoidal.json configs/fit_window/two_tandem/sinusoidal_evaluation.json --weight_heatmap
```

The suite supports `--parameterization individual`, `vr`, `lpd`, and `vr_lpd`.
Parametric models require `--input_feature_library_case`. The independent training
selectors are `--training_vr_list` and `--training_lpd_list`: VR models default to
the selected VR list, L/D models default to the selected L/D list, and joint models
default to both. Give the original training lists when evaluating a held-out VR or
L/D. Standard and coupled IVPs use the cylinder-specific physical mass ratio.

The common summary folder includes both L/D and VR selections. Inside it, each
`LpD = ...` folder contains VR-by-variable NRMSE heatmaps and frequency/amplitude
curves. The common folder also contains VR-by-L/D error heatmaps for each variable.
These per-variable IVP heatmaps (standard and coupled integral evaluation) use
28% larger figure width and height to give the cell values more room. Rerun the
suite or `plot_error_heatmap_sindy_ivp.py` to regenerate them from evaluation results.
`--phase_comparison` on Cases 1/2 produces separate h–CL phase comparisons for
cylinders 0 and 1, each using its own reference displacement peak frequency.
For individual Case 3 models, `--weight_heatmap` saves `hddot0_coefficients_heatmap.png`
and `hddot1_coefficients_heatmap.png` under `coefficients/LpD = .../`, with matching CSV
and source-model JSON files. Cross-cylinder feature coefficients remain in their
appropriate target equation. Parameterized Case 3 models likewise save
`hddot0_coefficients_heatmap.png` and `hddot1_coefficients_heatmap.png` per L/D,
with response-term rows and selected VR columns. VR-parameterized models also
save `coefficients/hddot0_tensor_coefficients_heatmap.png` and
`hddot1_tensor_coefficients_heatmap.png` across all selected L/D values,
retaining VR terms separately for each acceleration equation, plus CSV/JSON
companions. Partial IVPs are N/A and omitted from spectral trends;
per-case folders include L/D to prevent overwrites. All figures are 300 dpi and
aggregate titles follow the same rules as the single-cylinder suite.

Suite heatmaps use square cells, visible borders, grid lines, and numeric cell
annotations, matching the diagnostic heatmaps. Inner grid lines are 0.6 pt,
with 0.8-pt black outer spines across all heatmaps. Custom inner widths are
capped at the outer spine width. All suite comparison curves use
the diagnostic line width (0.6 pt): reference is solid and SINDy is dashed at the
same width. Frequency and amplitude trend y labels show only the variable, including
its cylinder index. Their titles are `Dominant Frequency (variable)`,
`Amplitude, Peak-to-Peak`, and `Amplitude, Dominant FFT`. The peak-to-peak amplitude
continues to mean half the signal range. Phase curves retain normalization by pi and use
y limits [-1.05, 1.05]. Rerun the evaluation or plotting commands to refresh PNGs.
NRMSE cell annotations use scientific notation for nonzero values below 0.01.
Cells enlarge as needed to fit the annotations at the standard font size.

## Filtered single-cylinder experiment runner

Run from the repository root:

```powershell
.\run_new_single.bat --dry-run
.\run_new_single.bat
```

The Parts A-F catalog contains 41 experiments. Currently 20 experiments
(40 commands) are active: A01-A04, B01-B04, C01-C04, D01-D04 and E01-E04.
All of F and E05-E08 are on hold, preserved as `REM ON HOLD` calls. Each
active fit is immediately followed by its matching evaluation suite.
The original notes are preserved in `old_run/run_new_single_checklist.txt`.
All experiments use VR 3-8 and `finite_difference_basic.json` (order 2).
Filtering uses `butterworth_single_flim2.json` for every experiment.
The batch uses this checkout's `.venv` and source code, and keeps the existing
filter/config/training-window result paths.

| Part | Catalog experiments | Models and variations |
| --- | ---: | --- |
| A | 4 | Individual Case 3, polynomial degree 1; Butterworth, OLS and vanilla STLSQ with both training windows; hddot coefficient heatmaps |
| B | 4 | Case 3 parameterized by VR with degree 2; Butterworth, OLS and vanilla STLSQ with both training windows; effective hddot coefficient heatmaps |
| C | 4 | Individual Case 1, state degrees 1-4; Butterworth, OLS, sinusoidal training; IVP h/CL phase comparisons |
| F01-F06 | 6 | VR-parameterized Case 1, immediately after C; state degrees 2/3 crossed with VR degrees 2/3/4 |
| D | 4 | The Part C settings for Case 2 |
| F07-F12 | 6 | VR-parameterized Case 2, immediately after D; state degrees 2/3 crossed with VR degrees 2/3/4 |
| E | 7 | Individual Case 4, OLS degrees 1-4 and three held restricted libraries; standard and coupled IVPs |
| F13-F18 | 6 | VR-parameterized Case 4, immediately after E; state degrees 2/3 crossed with VR degrees 2/3/4 |

Parts A/B use both OLS (`ols`: alpha=0, threshold=0) and vanilla STLSQ
(`stlsq_vanilla_basic`: alpha=0, threshold=0.1). Parts C-E use OLS only.
Column normalization is off. All active fits use Butterworth filtering.
Held calls retain their previous optimizer settings; the tandem runner
uses OLS/vanilla STLSQ for A/B and OLS degrees 2-4 for C-E.
A01-A02 and B01-B02 use OLS; A03-A04 and B03-B04 use vanilla STLSQ.
Each pair uses `overall_reduced` then `sinusoidal` training. A/B use state
polynomial degree 1, and B uses VR polynomial degree 2. C/D retain C01-C04 and D01-D04 only. E retains E01-E06 and E08.
All C/D/E/F experiments use Butterworth filtering and sinusoidal training.
Existing IDs remain, with A03-A04 and B03-B04 added for vanilla STLSQ. Execution order is A, B, C, F01-F06,
D, F07-F12, E, F13-F18. The 18 parameterized experiments retain their
original settings and follow their corresponding individual dynamics cases.

E05/E06 use the existing `case4_single_selected_polynomial_3_symmetric_only`
and `case4_single_selected_polynomial_3_no_hCL` libraries. E08 uses
`wake_case4_vdp_acceleration_single` (linear hddot coupling),
with linear restoring and van der Pol damping.

Every suite evaluates derivatives on `overall_reduced`, `sinusoidal`, and
`sinusoidal_evaluation`. Standard IVPs use `sinusoidal_evaluation`; Case 4
also runs the structural/lift coupled IVP. Frequency and amplitude comparisons
use rectangular FFT windows, with per-VR FFT displays limited to f*=0..2.
Derivative spectral comparisons use only `sinusoidal` and
`sinusoidal_evaluation`. Amplitude trends include half peak-to-peak and
dominant FFT amplitudes. Partial IVPs are N/A in the title-free NRMSE heatmaps
and omitted from spectral trends. Parts A/B additionally save the signed hddot
coefficient heatmaps; Parts C/D additionally save IVP CL-minus-h phases.

Only the active calls execute. Preflight checks verify required scripts/configs.
Failed fits skip their matching evaluations; failed evaluations do not stop
later fits. The final summary lists failed experiment stages and skipped
evaluations, with a nonzero exit code if any command failed. Matching outputs
are overwritten. Edit `SINDY_VRS`, `SINDY_IVP_METHOD`, or
`SINDY_MAX_STATE_FACTOR` near the top to change those settings.

## Filtered two-tandem-cylinder experiment runner

```powershell
.\run_new_two_tandem.bat --dry-run
.\run_new_two_tandem.bat
```

The tandem runner contains 17 active configurations and 34 fit/evaluation
commands: OLS and vanilla STLSQ for A/B, and OLS degrees 2-4 for C-E. Part F and E05-E08 remain on hold. It includes both
cylinders for every combination of VR 5-14 and L/D 1.5, 2, 3, 3.5 and 4.
Parts A/B have four experiments each, C/D have three active experiments each,
and E has three active experiments plus three held restricted libraries. F has
eighteen held experiments, split into groups following C, D and E. Vanilla
STLSQ uses alpha=0 and threshold=0.1.

All experiments use `finite_difference_basic.json`, with
`butterworth_two_tandem_flim2.json` for filtering.
Fitting and evaluation windows come from `configs/fit_window/two_tandem/`.
Individual fits produce one model per L/D/VR pair; Parts B/F fit a
VR-parameterized model separately at each L/D, retaining the selected
training VR list when resolving saved models.

Every fit is followed by `evaluate_sindy_two_tandem_suite.py`. Derivative
evaluation covers `overall_reduced`, `sinusoidal` and
`sinusoidal_evaluation`; IVPs use `sinusoidal_evaluation`. Spectral
comparisons cover the two sinusoidal windows. Parts A/B retain the Case 3
coefficient heatmaps: separate acceleration targets for individual models,
and separate acceleration-target effective-coefficient maps per L/D for parameterized models.
Parts C/D include CL-minus-h phase comparisons for both cylinders. Case 4
suites also evaluate the coupled structural/lift IVP.

Part E uses tandem response libraries with the canonical ten-channel Case 4
ordering. E05 uses
`case4_two_tandem_selected_polynomial_3_symmetric_only.json`, retaining
linear and cubic terms plus the bias. E06 uses
`case4_two_tandem_selected_polynomial_3_no_hCL.json`, retaining separate
wake-only and motion-only polynomials through degree three and excluding
mixed wake/motion products. These allow products within each group across
cylinders. E08 uses `wake_case4_vdp_acceleration_two_tandem.json`, retaining
linear restoring and van der Pol damping for both cylinders, with no
additional inter-cylinder product features.

The runner uses this checkout's `.venv`, validates scripts/configs before
fitting, and continues after failures. A failed fit skips its suite; the
final summary lists failed stages and skipped evaluations. Matching outputs
use the existing tandem filtering/model/window folder conventions.
