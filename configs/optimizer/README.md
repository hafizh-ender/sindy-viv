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
| 0.0001 | `ridge_alpha1e-4.json` | `ridge_alpha1e-4_normalized.json` |
| 0.001 | `ridge_alpha1e-3.json` | `ridge_alpha1e-3_normalized.json` |
| 0.01 | `ridge_alpha1e-2.json` | `ridge_alpha1e-2_normalized.json` |
| 0.05 | `ridge_basic.json` | `ridge_basic_normalized.json` |
| 0.1 | `ridge_alpha1e-1.json` | `ridge_alpha1e-1_normalized.json` |
| 0.5 | `ridge_alpha5e-1.json` | `ridge_alpha5e-1_normalized.json` |
| 1 | `ridge_alpha1e0.json` | `ridge_alpha1e0_normalized.json` |
| 5 | `ridge_alpha5e0.json` | `ridge_alpha5e0_normalized.json` |

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
