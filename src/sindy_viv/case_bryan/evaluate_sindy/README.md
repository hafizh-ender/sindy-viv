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

The shared function supports physical cases 1 through 4 and standard or weakly fitted
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

Set `weak=True` to read `weak_individual`, `weak_parametric_vr`, or
`weak_parametric_vr_lpd` results. Evaluation uses the returned pointwise
equations, rather than weak integration windows. For weak case 1, pass
`differentiator_case=None`. Derivative evaluation then requires
`evaluation_differentiator_case` to construct its reference. Integral evaluation
of weak case 1 needs neither differentiator.

`differentiator_case` identifies the training configuration and model path.
`evaluation_differentiator_case` optionally selects a different reference
differentiator without changing the model path; otherwise the training
differentiator is reused. Cases 2 and 4 need a reference differentiator for both
evaluation methods.

All four dynamics cases are supported in all six workflows. Cases 3 and 4
evaluate forced systems: lift (case 3) or motion (case 4) comes from measured
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
