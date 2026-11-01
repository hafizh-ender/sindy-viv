# Optimizer configurations

These JSON descriptors select the optimizer through the fitting CLI's
`--optimizer_case` argument. `sindy_viv.utils.initiate_object` imports the
configured class and passes its `kwargs` to the constructor. The fitting
workflow combines it with the selected feature library and prepared trajectories.

## Ridge sweep presets

The ridge presets use PySINDy's `STLSQ` with `threshold: 0` and
`unbias: false`: coefficients remain ridge-regularized without sparsity
thresholding or a final unregularized least-squares refit. PySINDy's default
`unbias: true` would undo the ridge penalty on the retained feature support.

| Alpha | Without normalization | With normalization |
| ---: | --- | --- |
| 0.000001 | `ridge_alpha1e-6.json` | Not supplied |
| 0.00001 | `ridge_alpha1e-5.json` | Not supplied |
| 0.00005 | `ridge_alpha5e-5.json` | Not supplied |
| 0.0001 | `ridge_alpha1e-4.json` | `ridge_alpha1e-4_normalized.json` |
| 0.001 | `ridge_alpha1e-3.json` | `ridge_alpha1e-3_normalized.json` |
| 0.01 | `ridge_alpha1e-2.json` | `ridge_alpha1e-2_normalized.json` |
| 0.05 | `ridge_basic.json` | `ridge_basic_normalized.json` |
| 0.1 | `ridge_alpha1e-1.json` | `ridge_alpha1e-1_normalized.json` |
| 0.5 | `ridge_alpha5e-1.json` | `ridge_alpha5e-1_normalized.json` |
| 1 | `ridge_alpha1e0.json` | `ridge_alpha1e0_normalized.json` |
| 5 | `ridge_alpha5e0.json` | `ridge_alpha5e0_normalized.json` |

The previous Case 6/7 lift-only study used all eleven **unnormalized** presets.
The older Cases 1/2/4 study used eight alphas with two normalization settings.
Both study controllers have been removed; presets remain available for future fits.

Normalized presets set `normalize_columns: true`. PySINDy scales each feature
column to unit L2 norm without centering before regression, then returns
coefficients in the original feature units. The penalty therefore acts on
different coefficient scales in the two branches; they are separate sweep axes.

To instantiate a preset without fitting or starting an experiment:

```python
import json
from pathlib import Path
from sindy_viv.utils import initiate_object

config = json.loads(Path("configs/optimizer/ridge_basic_normalized.json").read_text())
optimizer = initiate_object(config["module"], config["class_name"], config["kwargs"])
```

Creating these presets does not schedule any runs. Existing `ols.json`,
`stlsq_vanilla_basic.json`, and `stlsq_basic.json` keep their independent settings.

## OLS and thresholded fits

`ols.json` uses STLSQ with alpha=0, threshold=0, and normalization off.
`ols_sklearn.json` explicitly wraps `sklearn.linear_model.LinearRegression` with
`fit_intercept=false`, `normalize_columns=false`, and `unbias=false`; select it
when an explicit ordinary least-squares backend is needed. These presets have
distinct result identities. A polynomial bias column supplies an intercept.

`stlsq_vanilla_basic.json` uses alpha=0 and threshold=0.1. The previous lift-only
runner generated threshold presets inside each study's frozen configuration
tree, retaining the chosen Ridge alpha, normalization off and unbias false.
The historical threshold values were 1, 0.5, 0.1, 0.05, 0.01, 0.005 and 0.001;
that removed runner required a complete coupled IVP before thresholding.
Archived presets in `threshold_sweep/` do not by themselves define the current
study grid. ConstrainedSR3 examples use `temp/constrained_sr3_basic.json`;
physical constraints are supported for Cases 1-4, not Cases 6/7.

## Full recomputation controller

The [new full recomputation](../../script/FULL_RECOMPUTE.md) generates frozen
per-run descriptors instead of treating these historical STLSQ presets as true
Ridge-only. It uses `SVDLeastSquares`, `RidgeOnly` and `SequentialSVD` from
`sindy_viv.optimizers`, with normalization off and unbias false throughout.
Its group matrix is defined in `configs/experiments/full_recompute`; no models
are fitted merely by creating or reading a preset.
