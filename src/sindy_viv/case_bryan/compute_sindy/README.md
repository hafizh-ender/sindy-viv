# Bryan-case SINDy computations

`functions.py` implements `compute_sindy`; `weak_functions.py` implements
`compute_weak_sindy`. Both are exported by this package.
The standard function selects loading, preparation, fitting, and saving with
explicit `if`/`elif` branches for `cylinder_type` and `parameterization`.

## Functions and dependencies

| Module | Function | Role |
| --- | --- | --- |
| `functions.py` | `compute_sindy` | Prepare full trajectories, crop fitting windows, construct response/parameter libraries and optimizers, call generic pointwise fitting, and save model metadata. |
| `weak_functions.py` | `compute_weak_sindy` | Use the same physical preparation and grouping, build temporal weak equations, and save a pointwise model with its weak-setting identity. |

Both functions load named channels through `case_bryan.input`, use
`prepare_sindy.trajectory.prepare_trajectory` and `prepare_sindy.optimizer.prepare_optimizer`, and construct paths
through [model_storage](../model_storage/README.md). Numerical fitting delegates
to `sindy.functions.perform_sindy` or `perform_weak_sindy`; it is reusable with
non-Bryan arrays. Saved `model.csv` and `training.json` are later consumed by
[evaluation](../evaluate_sindy/README.md). The selection table and examples below
describe the API return keys and configuration combinations.

| `cylinder_type` | `parameterization` | Fit | Return value |
| --- | --- | --- | --- |
| `single` | `individual` | One model per VR | `{vr: model}` |
| `single` | `vr` | One model across selected VR values | `model` |
| `two_tandem` | `individual` | One model per spacing and VR | `{(lpd, vr): model}` |
| `two_tandem` | `vr` | One model across VR values at each spacing | `{lpd: model}` |
| `two_tandem` | `lpd` | One model across spacings at each fixed VR | `{vr: model}` |
| `two_tandem` | `vr_lpd` | One model across selected spacings and VR values | `model` |

The computation package is `sindy_viv.case_bryan.compute_sindy`. The scripts are
in the repository's existing `script/` directory.

## Standard SINDy

Standard and weak fits accept `filtering_case` (CLI: `--filtering_case`), with
SciPy Savitzky-Golay, Butterworth, and moving-average presets in
[`configs/filtering`](../../../../configs/filtering/README.md).
Filtering runs on full loaded measurements before differentiation and cropping.
The config stem adds `filtering=<stem>` before the differentiator folder in
saved-model paths; `training.json` snapshots its settings. Use the same option
in evaluation to select the model and filter its reference data. Omit the option
to retain the existing unfiltered behavior and paths.

All six workflows support `dynamics_case=1` through `5`:

| Case | States per cylinder | Measured controls per cylinder |
| --- | --- | --- |
| 1 | h, hdot, CL | None |
| 2 | h, hdot, CL, CLdot | None |
| 3 | h, hdot | CL |
| 4 | CL, CLdot | h, hdot, hddot |
| 5 | CL, CLdot | h, hdot |

For cases 3, 4 and 5, `state_feature_library_case` acts on states and measured
controls. In parametric runs, `input_feature_library_case` still acts only on
VR, L/D, or `[L/D, V_R]`; controls do not become parameters. The fitted model receives
controls before constant parameters in its `u` arrays.

All differentiation uses the loader's tau without dividing
by VR again. Cases 2, 4 and 5 construct lift-rate states with the supplied
first differentiator. Their lift-acceleration targets use native second
derivatives of filtered lift when supported (FD `d=2`, SG `deriv=2`), with
repeated first differentiation as the fallback for other callables or
unsupported settings. Case 4 loads the recorded acceleration channel as a control.
Case 5 omits that channel entirely and needs only measured h, hdot and CL.

For Case 5, `case5_single_lift3_motion1.json` combines cubic lift terms with
linear motion terms. `case5_single_lift3_motion1_tensored.json` also includes
lift-motion products. Both live under `configs/feature_library/` and use
single-cylinder column groups `[0, 1]` (CL, CLdot) and `[2, 3]` (h, hdot).
Use `--dynamics_case 5`; Case 4 configs with acceleration-specific indices
must be replaced or adjusted for this shorter response layout.

```python
from sindy_viv.case_bryan.compute_sindy import compute_sindy

model = compute_sindy(
    cylinder_type="single",
    parameterization="vr",
    dynamics_case=2,
    differentiator_case="configs/differentiator/savitzky_golay_single.json",
    state_feature_library_case="configs/feature_library/polynomial_3.json",
    input_feature_library_case="configs/feature_library/inverse_polynomial_2.json",
    optimizer_case="configs/optimizer/stlsq_basic.json",
    vr_list=[3, 4, 5],
    save_pkl=True,
)
```

All fits take `state_feature_library_case`. Select `parameterization="individual"`
for individual models, or `"vr"`/`"lpd"`/`"vr_lpd"` for parametric models; the latter
also require `input_feature_library_case`. Library presence never selects the
fit mode. To switch to individual runs, change `parameterization` and remove
the input-library argument. Tandem fits also accept `lpd_list`. All configurations
accept `results_root`, `re`, `mstar`, and `dtype`. Data and fitting default to
`np.float64` to retain weak feature directions in large least-squares systems;
an explicit `dtype=np.float32` remains supported. Configuration arguments are
JSON file paths. The functions
return fitted PySINDy models and save `model.csv`, plus `model.pkl` when requested.
Individual runs retain returned models; select fewer trajectories when only one
model is needed.

Tensor-product state and parameter libraries are supported, including
polynomial/Fourier combinations. The general `prepare_library(library_config)`
function in `sindy_viv.feature_library` instantiates both state and parameter
libraries from parsed configuration dictionaries. It wraps tensor libraries
before composition, preserving their feature values,
names, and local input-index groups in both computation and evaluation.

The PySINDy 2.1.0 issue depends on multiplication order: a tensor on the left
is flattened, but the outer library supplies input-index groups for the original
factors. `ParameterizedLibrary` puts parameters first, so an unwrapped parameter
tensor fails while a state-only tensor works. A state tensor can fail too when
placed first in a `GeneralizedLibrary` product. The helper handles either role;
ordinary libraries are instantiated without a wrapper. The same preparation
function is also used for individual models; it neither fits libraries nor
depends on Bryan-specific data.

## Reciprocal features

All six branches accept `InversePolynomialLibrary` through the same
JSON configuration loader as PySINDy's built-in libraries. Its constructor
options and defaults match `PolynomialLibrary`, including interaction products
and a bias. Values and outputs retain at least float64 precision.

| Workflow | Where to select reciprocal features | Degree-two parameter features |
| --- | --- | --- |
| `single_individual` | `state_feature_library_case`, acting on states | No parameter inputs |
| `single_parametric_vr` | `input_feature_library_case` | `1`, `1/V_R`, `1/V_R^2` |
| `two_tandem_individual` | `state_feature_library_case`, acting on states | No parameter inputs |
| `two_tandem_parametric_vr` | `input_feature_library_case` | `1`, `1/V_R`, `1/V_R^2`, separately at each L/D |
| `two_tandem_parametric_vr_lpd` | `input_feature_library_case` | `1`, `1/(L/D)`, `1/V_R`, `1/(L/D)^2`, `1/((L/D)*V_R)`, `1/V_R^2` |

For parametric fits, use
`input_feature_library_case="configs/feature_library/inverse_polynomial_2.json"`
with an ordinary polynomial state library. The joint model receives parameters
in `[L/D, V_R]` order, and generates reciprocal products on these parameters
before tensoring them with state features. The CSV feature names, pickle
exports, derivative evaluation, and integral evaluation support these products
in all four dynamics cases, subject to the reciprocal library's nonzero-input requirement.

Individual models have no parameter library: using the inverse configuration
there generates features such as `1/h_0`, not `1/V_R`. Reciprocal state features
are also selectable with `state_feature_library_case` in parametric fits.
However, the supplied single and tandem CSV datasets contain exact zero states,
which the inverse library rejects. Near-zero states also produce large
reciprocals and singularities during simulation. Keep the ordinary polynomial
state library for these oscillating responses.

The physical ConstrainedSR3 preparation requires linear state features and,
for parametric models, the positive-power feature `V_R^2`. A purely reciprocal
state or parameter library cannot supply these terms. To add reciprocal
parameters while retaining physical constraints, configure a
`pysindy.GeneralizedLibrary` with these two children:

```python
PolynomialLibrary(degree=2, include_bias=True)
InversePolynomialLibrary(degree=2, include_bias=False)
```

Keep only the polynomial bias to avoid duplicate feature names. This combined
parameter library supports both `V_R^2` and reciprocal L/DÃ¢â‚¬â€œVR interactions.

## ConstrainedSR3

The shared computation supports
`optimizer_case="configs/optimizer/constrained_sr3_basic.json"`. Constraints are
generated after input preparation, from the pointwise library's actual feature
names. There is no need to write coefficient indices or LHS/RHS arrays in JSON.

Cases 1Ã¢â‚¬â€œ3 constrain the structural equations, with lift treated as a measured
control in case 3. Case 4 constrains only `CL' = CLdot`.
The remaining equations are fitted freely. Parametric structural constraints
require a parameter bias and the positive-power `V_R^2` feature; the state
library must retain the necessary linear state/control terms.

For each cylinder, the preparation enforces the relations used in `script/old/`:

```text
dh_i/dt    = hdot_i
dhdot_i/dt = -4*pi**2*h_i + 2*V_R**2*C_Li/(pi*mstar)
```

Case 2 additionally enforces `dC_Li/dt = CLdot_i`. The lift equation in case 1,
or the lift-rate equation in case 2, remains learned from data. Every other
coefficient in each known equation is fixed to zero, including cross-cylinder
and L/D-dependent terms.

Individual models use the current VR in the lift coefficient. Parametric models
require a parameter-library bias and a `V_R^2` feature, so select
`input_feature_library_case="configs/feature_library/polynomial_2.json"` (or a
higher degree). The state library must contain linear displacement, velocity,
and lift terms; case 2 also needs linear lift-rate terms. Missing required
features raise an explicit error.

```python
model = compute_sindy(
    cylinder_type="single",
    parameterization="vr",
    dynamics_case=2,
    differentiator_case="configs/differentiator/savitzky_golay_single.json",
    state_feature_library_case="configs/feature_library/polynomial_3.json",
    input_feature_library_case="configs/feature_library/polynomial_2.json",
    optimizer_case="configs/optimizer/constrained_sr3_basic.json",
    vr_list=[3, 4, 5],
)
```

`constraint_mstar` defaults to the corresponding `MSTAR_TRUE` constant, matching
the old scripts. Override it to use another physical mass ratio. The existing
`mstar` argument selects the data directory, which uses rounded mass-ratio
labels. These equations use the legacy nondimensionalization and assume zero
structural damping.

LHS and RHS are float64. The default constraint order is `feature`; explicit
`target` order is also supported. The regularizer becomes its weighted variant
with zero penalties on the fully fixed equations. This preserves the physical
coefficients in `coef_` after SR3's sparse proximal step, while keeping the
configured penalties on learned equations. Column normalization and unbiasing
are rejected because they can change or undo the physical equalities. Explicit
constraint arrays and inequality constraints are not combined with these
automatically generated equalities.

Selecting other optimizers retains their existing configuration behavior. PySINDy's
`cvxpy` extra is included in project dependencies: run `uv sync` when updating
an existing installation.

## WeakSINDy

Import `compute_weak_sindy` from `sindy_viv.case_bryan.compute_sindy` or
`sindy_viv.case_bryan.compute_sindy.weak_functions`. Its historical import from
`compute_sindy.functions` remains available.

Weak fits combine two distinct configurations:

- `state_feature_library_case` selects the ordinary pointwise response basis
  from `configs/feature_library`; parametric fits also supply
  `input_feature_library_case` for the parameter basis.
- `weak_feature_library_case` selects PySINDy `WeakPDELibrary` integration
  settings from `configs/weak_feature_library`. Omitting it loads `default.json`.
  The CLI reads `K`, `H_xt` and `p` exclusively from this JSON. Python API
  callers can still override these settings with keyword arguments.

The complete response/parameter products are integrated together. Each trajectory
uses its own temporal grid and `K` windows, so trajectories with different sample
counts and time axes are supported. A temporal adapter uses PySINDy's analytic
piecewise-linear integration weights, feature integration, and integration-by-parts
targets. It samples feasible windows using a local RNG, avoiding PySINDy's global
RNG and unbounded rejection sampling. Each grid-aligned window contains at least
three samples. Both sides are normalized by the test-function integral, giving
each window equal weight without changing coefficient units.

`H_xt` is the window half width in tau (null means selected trajectory span/20),
and must be positive and smaller than half the selected span. `p>=1` defines
`phi=(1-z^2)^p`; `random_state` makes window sampling repeatable. Keep weak
`derivative_order=0`, `implicit_terms=false` and `include_bias=false` because the
pointwise basis determines bias terms and column order.

`tweak0`/`tweakf` select inclusive recorded t* bounds, with open endpoints by
default. Alternatively supply `fit_window_case` for trajectory-specific bounds;
do not combine it with `tweak0`/`tweakf`. Cases 1/3 require no training
differentiator. Cases 2/4/5 require one to construct lift-rate states on full
trajectories before cropping, but weak regression never differentiates its targets.

```python
from sindy_viv.case_bryan.compute_sindy.weak_functions import compute_weak_sindy

models = compute_weak_sindy(
    dynamics_case=1,
    optimizer_case="configs/optimizer/stlsq_basic.json",
    state_feature_library_case="configs/feature_library/polynomial_3.json",
    weak_feature_library_case="configs/weak_feature_library/default.json",
    vr_list=[3],
    random_state=7,
)
```

The five existing `script/perform_weak_sindy_*.py` entry points expose
`--weak_feature_library_case`, `--fit_window_case`, and `--random_state`.
Configure the window count, half width and test degree in the weak-library JSON.
For example:

```powershell
python script/perform_weak_sindy_single_individual.py --dynamics_case 1 --state_feature_library_case configs/feature_library/polynomial_3.json --weak_feature_library_case configs/weak_feature_library/default.json --optimizer_case configs/optimizer/stlsq_basic.json --vr_list 3 --random_state 7
```

Returned models retain the pointwise basis for `predict`, `simulate`, feature
names, and coefficient export. Artifacts include `model.csv`, `training.json`
and optional `model.pkl`. Weak paths include the weak-library config filename
without `.json`, immediately below the fit mode and above the pointwise basis:

```text
.../weak_individual/default/polynomial_1/<optimizer>/...
.../weak_parametric_vr/default/state_polynomial_1/input_polynomial_2/<optimizer>/...
```

Different weak-library config names keep their results in separate folders.
Changing settings within the same config name still overwrites matching results.
Training metadata snapshots the resolved weak descriptor, seed, bounds and
per-trajectory half widths. Evaluation accepts `--weak_feature_library_case`
to select the same folder; omitted means `default.json`.

The six standard CLI entry points also retain their other arguments and defaults.
Each script passes its fixed selectors to the shared function. Python callers
should replace the old case-specific function imports with `compute_sindy` or
`compute_weak_sindy` and the selectors shown above. Dataset selections and fitting-window folders identify each saved model.

Every computation CLI uses `--state_feature_library_case`, including the
individual scripts. Individual scripts select individual fits; parametric
scripts select their named parameterization and require
`--input_feature_library_case`. Switch scripts and remove the input-library
argument when changing a parametric command into an individual command.

Existing weak-model CSV files can still be read by the derivative and integral
evaluation workflows with `weak=True` / `--weak`.

References:

- [PySINDy WeakPDELibrary documentation](https://pysindy.readthedocs.io/en/stable/generated/pysindy.feature_library.WeakPDELibrary.html)
- [PySINDy ConstrainedSR3 documentation](https://pysindy.readthedocs.io/en/stable/generated/pysindy.optimizers.ConstrainedSR3.html)
- [PySINDy SINDyCP overview](https://pysindy.readthedocs.io/en/latest/examples/sindy-addl-examples/36401db/example.html)


## L/D-only parameterization

Use `parameterization="lpd"` with `cylinder_type="two_tandem"`. Each selected
VR gets one model fitted across `lpd_list`; the function returns `{vr: model}`.
The parameter library receives only `L/D`. VR remains fixed for that model and
still converts recorded time to tau in the loader. ConstrainedSR3 uses the fixed
VR in its structural lift coefficient, with parameter bias features `1 h` and
`1 CL`; it does not require a `V_R^2` feature for this mode.

New CLI entry point:

```powershell
python script/perform_sindy_two_tandem_parametric_lpd.py --dynamics_case 3 --differentiator_case configs/differentiator/savitzky_golay_two_tandem.json --state_feature_library_case configs/feature_library/polynomial_3.json --input_feature_library_case configs/feature_library/polynomial_2.json --optimizer_case configs/optimizer/stlsq_basic.json --vr_list 8 --lpd_list 1.5 2 3 --fit_window_case configs/fit_window/two_tandem/sinusoidal.json
```

Cases 1 through 5 support this mode, including measured controls,
reciprocal/tensored parameter libraries, and physical constraints.
WeakSINDy also supports `parameterization="lpd"` through the Python API.

## Fitting time bounds

For standard SINDy, the fitting-window input is `fit_window_case` / `--fit_window_case`, a JSON config
under `configs/fit_window/single/` or `configs/fit_window/two_tandem/`.
Single-cylinder files map `VR -> {tstar0, tstarf}`. Tandem files map
`LpD -> VR -> {tstar0, tstarf}`. The limits are properties of trajectories,
independent of dynamics case, parameterization, and CLI list order.

```text
--fit_window_case configs/fit_window/two_tandem/overall.json
```

`overall.json` uses available endpoints. Fill the first-half or sinusoidal
preset's selected trajectory limits before using it; unused placeholders can
remain. The complete schema and examples are in
[`configs/fit_window/README.md`](../../../../configs/fit_window/README.md).
The full nested config and its source/name are copied into `training.json`,
alongside each trajectory's resolved bounds and actual sample coverage.
Omitting `fit_window_case` loads the cylinder-specific `overall.json` from
this repository. Training has no `tstar0` or `tstarf` arguments or CLI flags;
those options are reserved for evaluation. Set trajectory endpoints in the
JSON file. Selected bounds must be finite or null, increase when both are
specified, lie within available data, and retain at least two samples.

The full signal is differentiated in float64 tau first, then t*, X, X_dot, and U
are cropped together before fitting. The retained time array stays tau; fitting
bounds never introduce another VR division. Retaining differentiation context
avoids new derivative artifacts at fitting-window boundaries. Only complete
signals are used by the differentiator; fitting and metrics can select subsets.

## Training selections and saved artifacts

Every parametric output adds the actual training selection before its fitting
window: `VR=3,4,5` for single cylinders, and `LpD=1.5,2/VR=5,8,11` for tandem
cylinders. VR-only models list one fixed L/D; L/D-only models list one fixed VR;
joint models list the whole selected grid. Selection labels are numerically
sorted, while `training.json` preserves the actual trajectory order.

For the L/D-only command above, the end of the output path is:

```text
parametric_lpd/state_polynomial_3/input_polynomial_2/stlsq_basic/
  LpD=1.5,2,3/VR=8/fit_window=sinusoidal/time=200..600/
    model.csv
    model.pkl          # if --save_pkl
    training.json
    evaluation/
```

Config-driven fits insert `fit_window=<config-stem>/` before the time folder,
for example `LpD=1.5,2,3/VR=8/fit_window=sinusoidal/time=200..600/`.
Changing limits in the same config creates another time folder; the old model
and its saved config snapshot remain available.

All new standard fits, including individual ones, have a time folder. Full
signals use `time=all`. A shared interval uses `time=200..600`; open endpoints
use `start` or `end`. Different per-trajectory intervals use
`time=per-trajectory-<12-character hash>`. This deterministic ID represents the
parameter-to-window mapping, so changing input order without changing that
mapping does not create another window ID. `training.json` records every
requested bound, actual retained t*/tau endpoints, sample count, parameter pair,
and the configuration snapshots. Different datasets or time windows coexist;
repeating the same selection and window writes the same destination.

A directory plus JSON is preferable here to `model_timerange1.csv` and
`timerange1.json`: it preserves the existing coefficient filename and keeps
pickles and evaluation artifacts associated with one fit.
Sequential numbers depend on save order and require a separate lookup table.
The choice is informed by [MLflow's per-run artifacts](https://mlflow.org/docs/latest/tracking/)
and [DVC's structured experiment metadata](https://doc.dvc.org/user-guide/experiment-management);
no new tracking dependency is introduced. This directory convention is a local
design choice, rather than a requirement of either tool.

Existing saved models are left in place and remain readable by evaluation.
See the evaluation README for independent training-model selection.
## Retained model data

Models fitted through `perform_sindy` release the optimizer's training design
matrix `Theta_` before returning or saving a pickle. Fitted coefficients and
library metadata remain available. `model.predict(X_new, u=U_new)` predicts
state derivatives from new states and the measured controls/constant parameters
used by that model; models without controls or parameters need only `X_new`.



## Preparation and dtype

Each raw trajectory is loaded, prepared on its complete float64 tau axis, and
cropped before the next trajectory is loaded. This bounds preparation memory
without changing full-signal derivative stencils. PySINDy fitting still holds
the selected prepared trajectories and its feature matrices.

Preparation functions now accept keyword-only dtype, defaulting to np.float64.
Choose dtype=np.float32 explicitly for float32 states, controls, derivatives
and parameter columns. Time always remains float64. Differentiator computation
precision follows its own configuration; preparation controls returned storage.
Named internal PreparedTrajectory records preserve the existing public tuple
wrappers. The physical Cases 1-4 remain unchanged; Case 5 retains only the lift-rate identity in ConstrainedSR3.

Library metadata setup samples one row only for exact known shape-only
polynomial, Fourier, custom and selected-polynomial implementations (including
composites built entirely from these). Unknown/data-adaptive/reciprocal libraries
retain full-data fitting and validation.
