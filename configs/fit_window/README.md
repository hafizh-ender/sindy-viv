# Trajectory fitting windows

Fitting limits belong to a trajectory and are reused across all dynamics cases
and fitting modes. Configs are separated by cylinder configuration:

```text
configs/fit_window/
  single/
    overall.json
    first_half.json
    sinusoidal.json
  two_tandem/
    overall.json
    first_half.json
    sinusoidal.json
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

`null` uses the available endpoint of that trajectory. `overall.json` is ready
to use and contains the default datasets. `first_half.json` leaves `tstarf`
as `"TO_BE_FILLED"`; `sinusoidal.json` leaves both endpoints as placeholders.
Fill the selected trajectories with numeric t* values before using them.
Unused trajectories may retain their placeholders. First-half/percentage
endpoints and the interval containing sinusoidal response are supplied by you;
the category name does not calculate or detect them. Copy a preset if another
category, such as `first_10_percent.json`, is useful.

Pass the same config to standard computation and evaluation:

```text
--fit_window_case configs/fit_window/single/overall.json
--fit_window_case configs/fit_window/two_tandem/sinusoidal.json
```

The config contains no dynamics-case or parameterization setting. For example,
the window at LpD=2, VR=8 is identical when used in individual, VR-only,
L/D-only, or joint fitting, and in dynamics cases 1 through 4.

`training.json` saves the config name, source path, full nested config snapshot,
and each fitted trajectory's resolved limits, actual endpoints, and sample
count. Fitted artifacts live under `fit_window=<name>/time=<window-ID>/`.
Changing bounds within a category creates a different time folder. Evaluation
matches the current config to the saved training trajectories, including when
evaluating a held-out VR or LpD absent from the config. Its own `tstar0` and
`tstarf` still select the scoring/simulation interval independently.

Training accepts only `fit_window_case`; omitting it loads the matching
`single/overall.json` or `two_tandem/overall.json` from this repository.
`tstar0` and `tstarf` are evaluation options only; fitting endpoints are read
from JSON. Older saved models can still be selected with `model_dir`.
