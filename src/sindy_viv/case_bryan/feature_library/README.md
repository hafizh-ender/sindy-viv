# Bryan feature libraries and assessment

| Module | Functions/classes | Relationship |
| --- | --- | --- |
| `custom_wake_oscillator.py` | `WakeOscillatorCase2Library`, `WakeOscillatorCase4Library`, `WakeOscillatorCase5Library` | Resolve case-specific names into a generic `SelectedPolynomialLibrary`. |
| `assessment.py` | `trajectory_groups`, `prepare_assessment_samples` | Match training trajectory groups and construct filtered/cropped states, controls and tensor parameters. |
| `assessment.py` | `assess_bryan_feature_library` | Call generic matrix assessment and write individual/pooled reports and summaries. |
| `assessment_plot.py` | `feature_labels`, `plot_correlation`, `plot_condition_heatmap` | Present complete feature matrices, duplicate-name mappings and VR/L/D condition grids. |
| `assessment_cli.py` | `assessment_parser`, `main` | Parse command-line selectors and delegate to the Bryan assessment API. |
| `legacy.py` | Compatibility re-exports | Preserve serialized callable import paths; implementations live in `compatibility`. |

The factories and assessment helpers share `prepare_sindy.case_layout.get_case_layout`.
Generic feature arithmetic and QR/SVD computation stay in the top-level
[feature_library package](../../feature_library/README.md). The sections below
give constructor options, CLI examples, and report definitions; helper
docstrings describe their local inputs, return values and file side effects.

## Restricted wake response libraries

`WakeOscillatorCase2Library`, `WakeOscillatorCase4Library` and
`WakeOscillatorCase5Library` are configuration factories. Each returns a
SelectedPolynomialLibrary, a PySINDy feature library with explicit ordered
exponent vectors. Each selected term is evaluated once using its output column
as working storage. The implementation supports arbitrary channel counts and
does not require lambdas, callable arity introspection, or closure serialization.
The factory names and JSON settings remain unchanged. Old CustomLibrary model
pickles retain their callable lookup through `compatibility/wake_callables.py` and the previous pickle import aliases.

Use these factories as `state_feature_library_case`. In Cases 3/4/5 the response
library acts on states **and measured controls**. `input_feature_library_case`
is the constant-parameter library used only for VR, L/D, or both. Therefore
Case 4 acceleration coupling is supported even though acceleration is a control.

Case 4 single-cylinder response input order is:

```text
CL0, CLdot0, h0, hdot0, hddot0
```

Tandem order is:

```text
CL0, CL1, CLdot0, CLdot1, h0, hdot0, hddot0, h1, hdot1, hddot1
```

These match `prepare_sindy/input.py`. Each factory defines its own Case 2, 4 or 5
layout; all term indices are resolved from those names. Other variable
orders are rejected when feature names are requested. Cases 2 and 5 do not supply
acceleration as a state or control, so requesting acceleration coupling there
raises an error. Cases 1/3 have no second-order wake equation and have no factory.

Case 5 uses the same lift states as Case 4 but omits acceleration. Its
single-cylinder response order is `CL0, CLdot0, h0, hdot0`; tandem order is
`CL0, CL1, CLdot0, CLdot1, h0, hdot0, h1, hdot1`.
`WakeOscillatorCase5Library` defaults to linear restoring, van der Pol
damping, and displacement/velocity coupling. Acceleration coupling is rejected.

Options common to all factories:

| Argument | Values | Features |
|---|---|---|
| `cylinder_type` | `single`, `two_tandem` | Appropriate prepared layout |
| `restoring` | `linear`, `cubic`, or a list | CL and/or CL^3 |
| `damping` | `linear`, `van_der_pol`, `rayleigh`, or a list | CLdot; CLdot and CL^2*CLdot; CLdot and CLdot^3 |
| `coupling` | List of `displacement`, `velocity`, `acceleration`, `nonlinear` | h, hdot, hddot, or full response polynomials |
| `coupling="nonlinear_motion"` | Motion-containing polynomials | All terms through `nonlinear_degree` containing h or hdot, including mixed wake/motion terms; excludes acceleration and extra pure-wake terms |
| `nonlinear_degree` | Positive integer, default 3 | Total degree when coupling includes `nonlinear` |
| `inter_cylinder_degree` | 0, or integer >=2 | Matching-channel products across cylinders through that total degree |
| `include_bias` | Boolean, default false | Optional constant feature |

Strings are accepted in place of single-element lists. Empty lists disable a
term family. The lift-rate feature is always present
for CL'=CLdot. Case 2 also retains h, hdot, and CL to represent
the structural equations. Duplicate terms are removed.

These options select a shared feature matrix, not target-specific coefficient
exclusions. The required lift-rate and structural features remain available to
the wake equation too, even when their optional families are disabled. Use
coefficient constraints when an individual equation must exclude such terms.

The default Case 4 options are linear restoring, van der Pol damping, and
acceleration coupling. They yield exactly four single-cylinder features:

```text
CLdot0, CL0, CL0^2 CLdot0, hddot0
```

These allow the fitted wake equation

```text
CLddot = a CL + b CLdot + c CL^2 CLdot + d hddot.
```

The two damping coefficients are fitted independently, following the expanded
form in the original notes. A strict normalized van der Pol factor
`b*(1-CL^2)*CLdot` additionally requires `c=-b`; this relation is not imposed.
ConstrainedSR3 can still impose the existing kinematic/structural equations.

`configs/feature_library/wake_case4_vdp_nonlinear_motion_single.json` selects
cubic coupling in h and hdot, including mixed CL/CLdot terms. It has 28
features. As with the other libraries, damping coefficients are independent.

Tandem libraries include the same four features for both cylinders. The shared
feature matrix allows a target to use the other cylinder's terms; setting
`inter_cylinder_degree=0` disables products, not cross-cylinder coefficients.
Degree two adds CL0*CL1, CLdot0*CLdot1, h0*h1, hdot0*hdot1, and (Case 4)
hddot0*hddot1. Degree three also adds their asymmetric cubic products.

Ready-to-use Case 4 configs:

- `configs/feature_library/wake_case4_vdp_acceleration_single.json`
- `configs/feature_library/wake_case4_vdp_acceleration_two_tandem.json`
- `configs/feature_library/wake_case4_vdp_motion_two_tandem.json`

The filtered tandem runner also uses two explicit Case 4 polynomial configs.
`case4_two_tandem_selected_polynomial_3_symmetric_only.json` includes the
bias, all ten linear terms and all cubic terms (231 features).
`case4_two_tandem_selected_polynomial_3_no_hCL.json` includes the bias and
separate wake-only/motion-only polynomials through degree three (118
features), excluding mixed wake/motion products while allowing products
between cylinders within each group.

For example:

```powershell
python script/perform_sindy_single_individual.py --dynamics_case 4 --vr_list 4 --differentiator_case configs/differentiator/savitzky_golay_single.json --state_feature_library_case configs/feature_library/wake_case4_vdp_acceleration_single.json --optimizer_case configs/optimizer/stlsq_basic.json --fit_window_case configs/fit_window/single/overall.json
```

Select the parameterized CLI and supply `input_feature_library_case` to tensor
these response terms with the parameter basis. Motion controls remain in the
response library. No differentiation, time conversion, or data-preparation
logic is changed by these factories.

The general implementation lives in
src/sindy_viv/feature_library/selected_polynomial_library.py. Factories use
get_case_layout so preparation, integral evaluation and wake features share
one channel ordering. Existing feature names and coefficient ordering remain
unchanged; new models pickle the exponent vectors directly.

## Feature-library assessment

`script/assess_feature_library.py` checks the correlation and condition number
of the feature matrix without a saved model or optimizer. Numerical computation
lives in the reusable `sindy_viv.feature_library.assessment` module. Bryan
preparation and metadata live in this directory; plots reuse the shared heatmap
grid/frame conventions.

Single-cylinder state/control features:

```powershell
.\.venv\Scripts\python.exe script/assess_feature_library.py `
  --dynamics_case 3 --cylinder_type single --vr_list 3 4 5 6 7 8 `
  --state_feature_library_case configs/feature_library/polynomial_1.json `
  --fit_window_case configs/fit_window/single/sinusoidal.json `
  --filtering_case configs/filtering/butterworth_single_flim2.json
```

Tandem tensors parameterized by VR, fitted separately at each L/D:

```powershell
.\.venv\Scripts\python.exe script/assess_feature_library.py `
  --dynamics_case 3 --n_cylinders 2 --vr_list 5 6 7 8 9 10 11 12 13 14 `
  --lpd_list 1.5 2 3 3.5 4 --parameterization vr `
  --state_feature_library_case configs/feature_library/polynomial_1.json `
  --input_feature_library_case configs/feature_library/polynomial_2.json `
  --fit_window_case configs/fit_window/two_tandem/sinusoidal.json `
  --filtering_case configs/filtering/butterworth_two_tandem_flim2.json
```

Supported modes are `individual` (default), `vr`, `lpd`, and `vr_lpd`; L/D
modes require tandem data. `--input_feature_library_case` is required exactly
when a parameterized mode is selected. Aliases are `--feature_library_case`
for the response library and `--parameter_feature_library_case` for the
parameter library. Select the cylinder type with `--cylinder_type` or use
`--n_cylinders 1/2`. Defaults for VR/L/D match the existing fitting scripts.

Data are loaded in tau. Optional filtering acts on full measured trajectories
before state construction and cropping. Cases 2/4/5 derive lift-rate states on
the full record using `--differentiator_case` (default `finite_difference_basic`);
Cases 1/3 require no differentiation. No derivative targets are computed.
Measured controls remain in the response basis; constant parameters follow
them, ordered `V_R`, `L/D`, or `L/D, V_R`, exactly as in training.

`--fit_window_case` selects trajectory-specific recorded-t* windows. Alternatively,
use `--tstar0`/`--tstarf`; these cannot be combined with a window JSON. With no
window selectors, the cylinder-specific `overall.json` is used. `--batch_size`
controls computation batches, not sampling. `--rank_rtol` optionally overrides
the numerical SVD rank tolerance. `--show` displays the saved figures.

Default report roots are:

```text
results/<case-name>/analysis/[Re = <Re>/Mstar = <mstar>/]feature_library_assessment/
  dynamics_case_<n>/[filtering=<config>]/<state-differentiator-or-none>/
  <individual-or-parametric-mode>/state_<library>/[input_<parameter-library>]/
  rank_rtol=<auto-or-value>/[LpD=<selection>/]VR=<selection>/
  [fit_window=<name>/]time=<bounds-or-per-trajectory-id>/
```

`--results_root` replaces `results`; `--output_dir` replaces the assessment
root while retaining a case-name subfolder. Matching outputs are overwritten.
The Re/Mstar folders are present for single-cylinder default outputs only,
matching the other single-cylinder analyses. Tandem outputs remain directly
under their dataset's analysis folder. An explicit output_dir retains its
custom case-name layout.
Configs, physical data settings, exact column order, source windows, actual
sample intervals, and numerical definitions are retained in JSON metadata.

Each `individual/[LpD=<value>/]VR=<value>/` contains:

- `correlation_matrix.png`: Pearson matrix in [-1, 1], with constant-feature
  pairs shown in gray as N/A.
- `correlation_matrix.csv`: the complete numeric matrix, in feature order.
- `correlation_metadata.json`: full metadata, column labels/IDs, constants,
  singular values, numerical rank, tolerance, and condition-number definition.
- `condition_number.json`: one condition number, with `"Infinity"` used for
  numerically deficient column rank to preserve valid JSON.

The selection-level `condition_number_heatmap.png` has one row across VR for
single-cylinder data, or L/D rows by VR columns for tandem data. Finite values
use a logarithmic color scale; gray cells annotated infinity denote deficient
column rank. `condition_number_summary.csv` and `.json` record all values and
their report paths. Condition numbers use raw Theta, without normalization.

Both raw and column-normalized condition numbers are written. Normalization
divides each nonzero feature column by its full-trajectory (or full pooled-group)
L2 norm without centering. Bias columns are retained; zero columns remain zero
and make the normalized condition infinite. Numerical rank is assessed
independently on each matrix using the same relative SVD tolerance.

Each individual/pooled report also contains `condition_number_normalized.json`
and `condition_number_normalized_metadata.json`. The latter records the norms,
normalized singular values, rank and tolerance. Selection-level outputs add
`condition_number_normalized_heatmap.png` and
`condition_number_normalized_summary.csv` / `.json`. Correlation matrices are
unchanged by this positive column scaling and are not duplicated.
`--normalized_only` adds normalized reports while preserving all existing raw
and correlation artifacts, including their summary files and figures.

For large correlation matrices every cell remains present, but plots limit
ticks to 30 labels and use feature IDs above 20 columns. JSON maps every ID
to its complete feature name. Cell annotations are shown only up to 12 columns.
CSV files always retain every feature; duplicate names are disambiguated by IDs.

Parameterized assessments additionally write `pooled/` reports for each actual
training group: all VRs per L/D in `vr` mode, all L/Ds per VR in `lpd` mode,
or the whole selected grid in `vr_lpd` mode. These include a correlation
PNG/CSV/metadata JSON and condition-number JSON for each pool. The selection
summary lists these pooled condition numbers separately. A fixed-parameter
trajectory can have proportional tensor columns even when the pooled matrix
is full rank; use the pooled report to assess that multi-trajectory fit.
