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
| `two_tandem` | `vr_lpd` | One model across L/D and VR | `(lpd, vr)` |

The shared function supports physical cases 1 and 2 and standard or weakly fitted
models. Validation,
coefficient alignment, numerical prediction, integration, and reporting helpers
are in `_evaluation.py`.

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

Set `weak=True` to read `weak_individual`, `weak_parametric_vr`, or
`weak_parametric_vr_lpd` results. Evaluation uses the returned pointwise
equations, rather than weak integration windows. For weak case 1, pass
`differentiator_case=None`. Derivative evaluation then requires
`evaluation_differentiator_case` to construct its reference. Integral evaluation
of weak case 1 needs neither differentiator.

`differentiator_case` identifies the training configuration and model path.
`evaluation_differentiator_case` optionally selects a different reference
differentiator without changing the model path; otherwise the training
differentiator is reused. Case 2 needs a reference differentiator for both
evaluation methods.

## Outputs and CLI

Each call returns a dictionary of evaluation records with:

- `time` (model tau), `time_star`, `reference`, `prediction`, and `variable_names`.
- Per-variable `NRMSE` and `RL2E`, computed exclusively on the selected samples.
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

All five integral CLI scripts save both outputs by default for each selected
trajectory: `integral_error_metrics_*.json` and `integral_comparison_*.png`.
The figure compares every state channel, including both cylinders and the
case-2 lift-rate states, and displays each channel's metrics. Passing
`--no_plot` writes metrics only. Reports require a successful integration over
the requested region; failed or prematurely stopped solves raise an error.

The ten `script/evaluate_{derivative,integral}_sindy_<configuration>.py` files
parse arguments and call `evaluate_sindy` with their fixed selectors. Existing
commands retain their configuration, `--xlim`, and
`--ylim_mean_time` options. Both evaluation methods accept `--tstar0`,
`--tstarf`, `--weak`, `--evaluation_differentiator_case`, `--block_size`,
`--results_root`, and `--no_plot`. Integral CLIs also accept `--method`,
`--rtol`, `--atol`, and `--solver_options` as a JSON mapping.

Python callers should replace the old case-specific function imports with
`evaluate_sindy` and the three selectors. The old case modules have been removed;
model lookup, report paths, return keys, and other CLI arguments retain their existing form.

All evaluation scripts use `--state_feature_library_case`. Their filenames
select the parameterization; parametric scripts still require
`--input_feature_library_case`. To use individual models, select the individual
script and remove the input-library argument.

[Reference: SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
