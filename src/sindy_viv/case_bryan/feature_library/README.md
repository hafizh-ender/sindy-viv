# Restricted wake-oscillator response libraries

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
