# Trajectory fitting windows

Fitting limits belong to a trajectory and are reused across all dynamics cases
and fitting modes. Configs are separated by cylinder configuration:

```text
configs/fit_window/
  single/
    overall.json
    first_half.json
    overall_reduced.json
    sinusoidal.json
    sinusoidal_evaluation.json
    sinusoidal_evaluation_longer.json
  two_tandem/
    overall.json
    first_half.json
    overall_reduced.json
    sinusoidal.json
    sinusoidal_evaluation.json
    sinusoidal_evaluation_longer.json
```

Single-cylinder configs map VR to the two recorded t* limits:

```json
{
  "3": {"tstar0": null, "tstarf": null},
  "4": {"tstar0": 200, "tstarf": 600}
}
```

Tandem configs map LpD to VR to those same limits:

```json
{
  "1.5": {
    "8": {"tstar0": 200, "tstarf": 600}
  },
  "2.0": {
    "8": {"tstar0": 220, "tstarf": 640}
  }
}
```

JSON object keys are strings. Numeric spellings such as `"2"` and `"2.0"`
refer to the same LpD or VR; duplicate numeric keys are rejected. Dictionary
order does not affect which window is assigned to a trajectory.

`null` uses the available endpoint of that trajectory. Both `overall.json`
files are ready to use. The current `overall_reduced`, `sinusoidal`,
`sinusoidal_evaluation`, and `sinusoidal_evaluation_longer` presets are populated
for all six single-cylinder trajectories and all 50 tandem pairs. Single-cylinder
configs also include populated `last_half.json` and `early_sinusoidal.json`
(the latter selects VR=7 only).

Single-cylinder `first_half.json` still has `TO_BE_FILLED` end bounds; the
tandem first-half preset is populated. Fill placeholders for every selected
trajectory before use; unused entries may retain placeholders. Names do not
automatically detect a sinusoidal interval or compute percentage endpoints.

Pass the same config to standard computation and evaluation:

```text
--fit_window_case configs/fit_window/single/overall.json
--fit_window_case configs/fit_window/two_tandem/sinusoidal.json
```

The config contains no dynamics-case or parameterization setting. For example,
the window at LpD=2, VR=8 is identical when used in individual, VR-only,
L/D-only, or joint fitting, and in dynamics Cases 1, 2, 3, 4, 6, and 7.

`training.json` saves the config name, source path, full nested config snapshot,
and each fitted trajectory's resolved limits, actual endpoints, and sample
count. Fitted artifacts live under `fit_window=<name>/time=<window-ID>/`.
Changing bounds within a category creates a different time folder. Evaluation
matches the current config to the saved training trajectories, including when
evaluating a held-out VR or LpD absent from the config. Its own `tstar0` and
`tstarf` still select the scoring/simulation interval independently.

Training accepts only `fit_window_case`; omitting it loads the matching
`single/overall.json` or `two_tandem/overall.json` bundled in
`sindy_viv/presets/fit_window/`. Repository presets are explicit experiment inputs.
`tstar0` and `tstarf` are evaluation options only; fitting endpoints are read
from JSON. Older saved models can still be selected with `model_dir`.
