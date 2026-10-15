# Bryan-case SINDy computations

`functions.py` exposes `compute_sindy` and the `compute_weak_sindy` placeholder.
The standard function selects loading, preparation, fitting, and saving with
explicit `if`/`elif` branches for `cylinder_type` and `parameterization`.

| `cylinder_type` | `parameterization` | Fit | Return value |
| --- | --- | --- | --- |
| `single` | `individual` | One model per VR | `{vr: model}` |
| `single` | `vr` | One model across selected VR values | `model` |
| `two_tandem` | `individual` | One model per spacing and VR | `{(lpd, vr): model}` |
| `two_tandem` | `vr` | One model across VR values at each spacing | `{lpd: model}` |
| `two_tandem` | `vr_lpd` | One model across selected spacings and VR values | `model` |

The computation package is `sindy_viv.case_bryan.compute_sindy`. The scripts are
in the repository's existing `script/` directory.

## Standard SINDy

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
for individual models, or `"vr"`/`"vr_lpd"` for parametric models; the latter
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

All five branches accept `InversePolynomialLibrary` through the same
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
in dynamics cases 1 and 2.

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
parameter library supports both `V_R^2` and reciprocal L/D–VR interactions.

## ConstrainedSR3

The shared computation supports
`optimizer_case="configs/optimizer/constrained_sr3_basic.json"`. Constraints are
generated after input preparation, from the pointwise library's actual feature
names. There is no need to write coefficient indices or LHS/RHS arrays in JSON.

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

## WeakSINDy placeholders

The previous WeakSINDy computation implementation has been removed.
`compute_weak_sindy` accepts the same cylinder and parameterization selectors,
along with the existing weak integration inputs. It and
`sindy_viv.sindy.functions.perform_weak_sindy` raise
`NotImplementedError("WeakSINDy is not implemented.")` immediately. They do not
load data, fit models, or write results. The weak-library helper module has
been removed.

The five `script/perform_weak_sindy_*.py` entry points retain their other CLI
arguments and defaults, including `--tweak0`, `--tweakf`, `--K`, `--H_xt`, `--p`,
and `--random_state`. Their `--help` output remains available; running a
computation reaches its placeholder.

The five standard CLI entry points also retain their other arguments and defaults.
Each script passes its fixed selectors to the shared function. Python callers
should replace the old case-specific function imports with `compute_sindy` or
`compute_weak_sindy` and the selectors shown above. Saved model paths are unchanged.

Every computation CLI uses `--state_feature_library_case`, including the
individual scripts. Individual scripts select individual fits; parametric
scripts select their named parameterization and require
`--input_feature_library_case`. Switch scripts and remove the input-library
argument when changing a parametric command into an individual command.

Existing weak-model CSV files can still be read by the derivative and integral
evaluation workflows with `weak=True` / `--weak`.

References:

- [PySINDy ConstrainedSR3 documentation](https://pysindy.readthedocs.io/en/stable/generated/pysindy.optimizers.ConstrainedSR3.html)
- [PySINDy SINDyCP overview](https://pysindy.readthedocs.io/en/latest/examples/sindy-addl-examples/36401db/example.html)
