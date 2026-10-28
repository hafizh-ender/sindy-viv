# Reusable feature libraries and matrix assessment

Matrix assessments report both raw and unit-L2-column condition numbers.
The normalized variant scales nonzero columns without centering, retains bias
and zero columns, and derives its singular values from the scaled QR factor.
`FeatureMatrixAssessment` exposes `normalized_condition_number`,
`normalized_rank`, `normalized_rank_tolerance`, `normalized_singular_values`,
`column_l2_norms` and `normalized_rank_deficient` alongside the raw results.

This folder defines dataset-independent feature construction, temporal weak
integration and diagnostics. Bryan's physical input ordering belongs in
`case_bryan/feature_library` and `case_bryan/prepare_sindy`.

| Module | Functions/classes | How they connect |
| --- | --- | --- |
| `functions.py` | `prepare_library` | Instantiate a configured PySINDy library; preserve outer input groups for composite tensor libraries. |
| `functions.py` | `uses_shape_only_fit`, `fit_library_metadata` | Fit metadata on one representative row only for known shape-only implementations; otherwise preserve full-data fitting/validation. |
| `selected_polynomial_library.py` | `SelectedPolynomialLibrary` | Evaluate explicitly ordered exponent vectors and validate the expected input-name layout. |
| `inverse_polynomial_library.py` | `InversePolynomialLibrary` | Construct reciprocal polynomial features with finite/nonzero-domain checks. |
| `weak.py` | `validate_weak_settings`, `weak_half_width`, `resolve_weak_library_config`, `build_weak_system` | Resolve integration settings and turn pointwise features/states into temporal weak regression equations. |
| `assessment.py` | `FeatureMatrixAssessment`, `LibraryAssessment` | Store ordered matrix statistics and individual/pooled assessment results. |
| `assessment.py` | `assess_feature_matrix`, `assess_feature_library` | Assess an existing design matrix or build/assess features without an optimizer. |

PySINDy-style objects expose `fit`, `transform` and `get_feature_names`.
`prepare_library` returns an unfitted object, generic SINDy fitting constructs
its regression matrix, and matrix assessment can inspect the same basis
independently. The main constructors, `prepare_library` and assessment APIs are
re-exported by this package. Metadata-fit helpers remain in
`feature_library.functions`; weak-system functions remain in `feature_library.weak`.

## Build a configured or selected library

```python
import numpy as np
from sindy_viv.feature_library import prepare_library, SelectedPolynomialLibrary

samples = np.array([[1., 2.], [2., 3.], [3., 4.]])
configured = prepare_library({
    "module": "pysindy.feature_library",
    "class_name": "PolynomialLibrary",
    "kwargs": {"degree": 2},
})
theta = configured.fit_transform(samples)

selected = SelectedPolynomialLibrary(
    input_names=["x", "y"],
    term_powers=[[1, 0], [0, 1], [2, 1]],
    include_bias=True,
)
selected.fit(samples)
names = selected.get_feature_names(["x", "y"])
selected_theta = selected.transform(samples)
```

The selected columns are the bias, x, y and x-squared times y in that order.
Factories in the Bryan feature-library folder construct these exponent vectors
from physical variable names; the generic implementation does not know which
columns represent motion, lift or parameters. Weak fitting integrates those
pointwise columns without changing their coefficient names.

## Feature-matrix correlation and conditioning

`assessment.py` contains the numerical implementation, independent of Bryan
data, physical parameters, plots, and report paths. It does not fit an optimizer.

```python
import numpy as np
import pysindy as ps
from sindy_viv.feature_library import assess_feature_matrix, assess_feature_library

# Any precomputed dense SINDy feature matrix Theta:
result = assess_feature_matrix(theta, feature_names)
print(result.condition_number, result.rank)
correlation = result.correlation

# Or build a matrix with an existing pointwise PySINDy library:
reports = assess_feature_library(
    ps.PolynomialLibrary(degree=2), samples, input_names=["x", "y"]
)
result = reports.pooled
```

`samples` can be one 2-D NumPy array or a sequence of trajectory arrays with
the same input columns. The library is copied before fitting. For parameterized
fits, pass a `ps.ParameterizedLibrary` and the prepared response/control/parameter
columns in its input order. `reports.individual` contains one assessment per
trajectory; `reports.pooled` assesses the vertical concatenation used for fitting.

Pearson correlation centers each feature column and normalizes by its centered
length. A constant feature, including a bias term, has undefined correlations:
its row and column are `NaN`. These columns remain present in every calculation.

Conditioning uses **raw, uncentered, unnormalized** feature columns. The
2-norm condition number is `sigma_max / sigma_min`. When the matrix lacks
numerically full column rank, the reported value is infinity. This includes
underdetermined matrices with fewer rows than feature columns. The default
numerical-rank threshold is `eps * max(N, P) * sigma_max`, matching the standard
SVD rank convention; `rank_rtol` overrides its relative factor. The result
retains all computed singular values and the absolute rank threshold.

Column units and scale affect this raw condition number. No optimizer
normalization, regularization, centering, constant removal, or sample selection
is applied to the matrix used for conditioning. Correlation and conditioning
therefore measure different properties of the same feature columns.

`batch_size=8192` controls working memory, not subsampling. Correlations use
merged centered moments. Reduced QR factors are merged and then assessed by
SVD, avoiding a full pooled transformed matrix and avoiding the accuracy loss
from conditioning `Theta.T @ Theta`. Known shape-only libraries transform in
batches. Unknown/data-adaptive libraries fit using full data and transform whole
trajectories so batching does not change their semantics. If a library changes
sample rows, build its matrix separately and use `assess_feature_matrix`.
Nonfinite or nonnumeric feature data raise an error; no rows are silently omitted.

Bryan preparation, tensor grouping, CLI selectors, and reports are implemented
in `case_bryan/feature_library/assessment.py`, `assessment_cli.py`, and
`assessment_plot.py`. Their CLI is `script/assess_feature_library.py`;
see the [Bryan usage guide](../case_bryan/feature_library/README.md#feature-library-assessment).
