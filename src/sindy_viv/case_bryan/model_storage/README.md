# Fit windows and saved-model identities

This folder keeps training, diagnostics and model discovery consistent. It
constructs/validates paths and window metadata; computation/evaluation callers
perform the actual model/report writes.

| Module | Function | Result and caller |
| --- | --- | --- |
| `fit_windows.py` | `load_fit_window_config` | Read and validate the single/tandem JSON structure. |
| `fit_windows.py` | `prepare_fit_window_bounds` | Resolve every selected `(L/D, VR)` pair to requested inclusive t* bounds. |
| `fit_windows.py` | `crop_training_trajectory` | Crop aligned tau/state/target/control arrays and return actual-window/sample metadata. |
| `paths.py` | `number_text` | Format numeric selection/path labels without an unnecessary `.0`. |
| `paths.py` | `model_base_directory` | Identify case, filter, differentiator, fitting family, libraries, optimizer and fixed dataset settings. |
| `paths.py` | `training_directory` | Append selected parameter sets, optional window name and a reproducible time label. |
| `paths.py` | `resolve_model_directory` | Locate the saved model matching training selectors; reject missing/ambiguous matches. |

All seven functions are re-exported from `sindy_viv.case_bryan.model_storage`.
Private window-index helpers normalize JSON keys before public selection and
are described in their docstrings.

## Resolve bounds and name an output

```python
from pathlib import Path
from sindy_viv.case_bryan.model_storage import (
    load_fit_window_config, prepare_fit_window_bounds, training_directory,
)

window_file = "configs/fit_window/single/sinusoidal.json"
config = load_fit_window_config(window_file, "single")
bounds = prepare_fit_window_bounds(config, [(None, 3), (None, 4)], "single")
trajectories = [
    dict(LpD=lpd, VR=vr, requested_tstar0=start, requested_tstarf=stop)
    for (lpd, vr), (start, stop) in bounds.items()
]
folder = training_directory(Path("results/example"), "vr", trajectories, window_file)
print(folder)
```

Path construction does not create the directory. Common bounds produce a
readable `time=start..end` label. Per-trajectory bounds produce a stable mapping
ID, so reordering equivalent selections does not change the destination.

`crop_training_trajectory` accepts already prepared full records. Time remains
tau; requested bounds use t*. Explicit windows return owning cropped copies so
the full loaded buffers can be released. Its metadata records requested and
actual t*/tau intervals plus sample counts. Filtering and differentiation belong
before this function, in `prepare_sindy`.

## Find the same model during evaluation

`compute_sindy` combines `model_base_directory` and `training_directory`, then
saves `model.csv` and `training.json` there. Evaluators construct the base again
and call `resolve_model_directory` with the training VR/L/D set and fit-window
config. Evaluation VR/time selectors choose scoring data independently of those
training selectors. An explicit `model_dir` is still checked for compatible
metadata; discovering several candidates produces an actionable error instead
of choosing the latest file.

The reader supports the documented older directory layouts so existing results
remain accessible. [Computation](../compute_sindy/README.md) and
[evaluation](../evaluate_sindy/README.md) describe complete examples and outputs.
