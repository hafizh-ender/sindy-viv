# Preparing states, controls, filtering and optimizers

Preparation connects Bryan's measured channels to the generic fitting APIs.
It does not choose output folders or fit a model. Load full trajectories first;
crop them with `model_storage.crop_training_trajectory` after filtering and
required differentiation.

| Module | Functions/classes | Use and relationship |
| --- | --- | --- |
| `case_layout.py` | `CaseLayout`, `get_case_layout` | Describe Cases 1-5 and their ordered measured, state, control and lift-rate names. Every fitting/evaluation path uses these layouts. |
| `trajectory.py` | `PreparedTrajectory`, `prepare_trajectory` | Construct canonical state/control arrays and optional derivative targets on tau. |
| `trajectory.py` | `append_parameter_columns`, `PreparedTrajectory.with_parameters` | Append constants after measured controls while preserving sample order and data precision. |
| `filtering.py` | `SavgolFilter`, `ButterworthFilter`, `MovingAverageFilter` | Callable SciPy smoothing objects accepting `(data, time)` and returning same-shape measurements. |
| `optimizer.py` | `prepare_optimizer`, `prepare_constrained_sr3` | Instantiate an optimizer descriptor or build named kinematic/structural coefficient equalities. |
| `input.py` | `prepare_input_single_individual`, `prepare_input_two_tandem_individual` | Historical tuple-returning adapters over canonical trajectory preparation. |
| `input.py` | `prepare_input_*_parametric_vr`, `prepare_input_two_tandem_parametric_lpd`, `prepare_input_two_tandem_parametric_vr_lpd` | Adapt one or multiple trajectories with the selected constant parameters. |
| `input.py` | `prepare_input_single_case_1` through `_case_5`, tandem equivalents, and case-specific parameterized variants | Explicit-case aliases for existing callers; use `prepare_trajectory` for new code. |
| `input.py` | `iter_prepared_trajectories` | Yield prepared trajectory records for the selected parameter grouping. |

Private helpers validate array/name consistency, attach constants and route the
tuple adapters. Their contracts are documented alongside their implementation.

## Construct a prepared trajectory

```python
from pysindy.differentiation import FiniteDifference
from sindy_viv.case_bryan.constants import SINGLE_CYLINDER_RE, SINGLE_CYLINDER_MSTAR
from sindy_viv.case_bryan.input import load_data_single_cylinder
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.case_bryan.prepare_sindy.filtering import MovingAverageFilter
from sindy_viv.case_bryan.prepare_sindy.trajectory import prepare_trajectory

layout = get_case_layout(2, "single")
time, measurements = load_data_single_cylinder(
    SINGLE_CYLINDER_RE, SINGLE_CYLINDER_MSTAR, 3,
    variables=layout.measured_names,
)
trajectory = prepare_trajectory(
    time, measurements, list(layout.measured_names),
    FiniteDifference(order=2), 2, "single",
    filtering=MovingAverageFilter(window_length=31),
)
parameterized = trajectory.with_parameters([3], ["V_R"])
states = parameterized.states
controls_and_parameters = parameterized.inputs
```

`time` is already float64 tau. The example first smooths every full measured
channel, constructs the Case 2 lift-rate states, and computes derivative
targets. State/control dtype defaults to float64 independently of time dtype.
`as_tuple()` exposes the older four/six-value training tuple. With
`with_derivatives=False`, preparation omits regression targets but still derives
lift rates needed by Cases 2/4/5; feature assessment and IVP references use this
mode. `with_parameters` copies controls into an array followed by constant
parameter columns and returns a new record.

For Cases 2/4/5, `CLdot` is the first derivative of filtered lift. When derivative
targets are requested, `CLddot` is computed directly from that same filtered
lift through `differentiation.perform_second_derivative`: finite differences
use `d=2`, and `SavitzkyGolayDerivative` uses `deriv=2` with the configured
window, polynomial and edge settings. Smoothed finite differences and spectral
derivatives also support native second derivatives. The configured first-
derivative object remains unchanged. No extra filtering stage is inserted.
Custom callables/subclasses and unsupported settings (FD accuracy order 1 or
SG polynomial degree below 2) retain repeated first differentiation. Integral
evaluation and weak training request only lift-rate states, so do not compute
`CLddot` targets. All derivatives use tau before the fitting-window crop.

Cases 1/2 model motion and lift; Case 3 models motion with measured lift as a
control; Cases 4/5 model lift with measured motion controls. Case 4 includes
measured acceleration whereas Case 5 does not. These distinctions determine
the response-library input order, not a caller-defined column sort.

## Configure filters and optimizers

Filter descriptors in `configs/filtering` instantiate these callables through
`utils.initiate_object`. Filtering uses the supplied tau coordinate, including
Butterworth cutoffs specified in cycles per unit time. Normalized Butterworth
cutoffs are fractions of Nyquist. Savitzky-Golay and moving-average window
lengths count samples. See [filter configurations](../../../../configs/filtering/README.md)
for edge modes and examples.

`prepare_optimizer` receives the response library, prepared states, measured
controls and parameter names. Ordinary optimizer descriptors are instantiated
directly. `ConstrainedSR3` additionally resolves feature/target names and builds
the physical equalities before fitting; it is not interchangeable with adding
constraints after coefficients are computed. `compute_sindy` owns this ordering.
