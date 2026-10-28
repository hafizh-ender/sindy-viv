# Source package guide

This package separates reusable SINDy operations from Bryan's cylinder dataset.
Use the generic modules with prepared arrays from another problem; use
`case_bryan` when loading the project's cylinder CSVs and configurations.

| Location | Main functions/classes | Role and next step |
| --- | --- | --- |
| `sindy/functions.py` | `perform_sindy`, `perform_weak_sindy` | Fit prepared arrays and return a pointwise PySINDy model. Bryan computation calls these functions. |
| [feature_library](feature_library/README.md) | `prepare_library`, `SelectedPolynomialLibrary`, `InversePolynomialLibrary`, `assess_feature_library`, `assess_feature_matrix` | Build feature columns, support weak integration, and assess correlation/conditioning. |
| `differentiation` | `perform_derivative`, `perform_second_derivative`, `SavitzkyGolayDerivative` | Differentiate arrays on their supplied time coordinate; select native second derivatives where supported. No dataset loading or time conversion occurs here. |
| `metrics/numerical.py` | `normalized_root_mean_squared_error`, `relative_l2_error` | Score aligned arrays; evaluation passes reference/prediction channels here. |
| `utils.py` | `load_config`, `initiate_object`, `add_dot_text`, formatting helpers | Read object descriptors, instantiate configured classes, and construct derivative labels. |
| `_typing.py` | Shared aliases/protocols | Describe cylinder/case selections and numerical array interfaces. |
| [case_bryan](case_bryan/README.md) | Loading, preparation, fitting, evaluation, analysis | Connect generic operations to the cylinder dataset and its result metadata. |

Individual functions and classes document their arguments, return values,
mutations and numerical conventions in their docstrings. Small folders such as
`metrics` and `differentiation` use those source descriptions instead of extra
folder READMEs.

## Fit an unrelated time-series problem

```python
import numpy as np
import pysindy as ps
from sindy_viv.sindy.functions import perform_sindy

time = np.linspace(0, 10, 501)
states = np.column_stack((np.sin(time), np.cos(time)))
derivatives = np.column_stack((np.cos(time), -np.sin(time)))
model = perform_sindy(
    time=time,
    state_variable_data=states,
    state_variable_differentiated_data=derivatives,
    feature_library=ps.PolynomialLibrary(degree=1),
    optimizer=ps.STLSQ(threshold=1e-3),
    feature_names=["x", "y"],
)
predicted_derivatives = model.predict(states)
```

The caller prepares a consistent time coordinate, states and optional controls.
`perform_sindy` passes them to PySINDy; precomputed derivatives take precedence
over internal differentiation. It releases the optimizer's retained training
design matrix after fitting. New predictions use the fitted coefficients and
library with the supplied new state/control samples.

For multiple trajectories, use lists of aligned time/state/target/control
arrays. `perform_weak_sindy` instead builds temporal integral equations through
`feature_library.weak.build_weak_system` and still returns a pointwise model for
prediction/evaluation. The [feature-library guide](feature_library/README.md)
explains matrix assessment and how it relates to either fitting route.

## Editing conventions

Keep dataset-specific loading, channel ordering and result metadata in
`case_bryan`. Generic numerical modules accept arrays/configured objects and do
not import Bryan data loaders. Describe each function's actual inputs, output
shape, time/normalization conventions and side effects where those matter.
Describe relationships and examples in the relevant folder guide rather than
repeating a complete workflow in every helper docstring.

CLI entry points live in the repository's [script folder](../../script/README.md).
They parse options and delegate to package APIs. Format argparse calls with the
flag and each keyword setting on separate lines; keep option names/defaults
consistent with existing callers.
