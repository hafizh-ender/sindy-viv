# Weak feature-library settings

These configs describe the temporal integration settings of PySINDy's
`WeakPDELibrary`. They are combined with the ordinary response basis selected
from `configs/feature_library`; parametric fits also select a parameter basis
from that directory.

Pass `--weak_feature_library_case configs/weak_feature_library/default.json`.
Omitting it selects this default file. `K` is the number of windows **per
trajectory**, `H_xt` is their half width in dynamic time tau (null means each
cropped trajectory span / 20), and `p` is the positive integer degree in the
test function `(1-z^2)^p`. Set these values in the JSON; the CLI reads them
from the selected weak-library config.
`--random_state` controls repeatable sampling without changing global RNG state.

The response/parameter products are formed before integration. The temporal
grid is supplied separately for each trajectory. Keep `derivative_order=0`,
`implicit_terms=false`, and `include_bias=false`: this workflow fits ODEs,
and the pointwise basis controls bias terms and feature ordering.

Each grid-aligned window must contain at least three samples. `H_xt` must be
positive and smaller than half the selected trajectory span. Both sides of
each weak equation are divided by the test-function integral to give windows
equal weight while preserving coefficient units.

Saved results include this config's filename stem immediately below the weak
fit mode, for example `weak_individual/default/polynomial_1/...`. Select the
same config with `--weak_feature_library_case` when evaluating a saved model.
