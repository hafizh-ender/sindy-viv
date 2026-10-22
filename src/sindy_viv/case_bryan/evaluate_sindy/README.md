# Bryan-case SINDy evaluation

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
Failed or early-terminated integrations raise an error instead of scoring a
shortened region. Diagnostics include method, tolerances, status, message, and
evaluation counts.

An integration failure also reports the solver method, parameter values,
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
`--rtol`, `--atol`, and `--solver_options` as a JSON mapping.

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
`results/<dataset>/derivative_assessment/dynamics_case_<n>/<differentiator>/...`.
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
