# Command-line entry points

Run maintained scripts from the repository root with this checkout's Python:

```powershell
.\.venv\Scripts\python.exe script/assess_feature_library.py --help
```

Scripts parse dataset/config/output selections and delegate to functions in
`src/sindy_viv`. Numerical behavior belongs in those package functions so APIs,
CLI commands and batch runners use the same implementation.

Scientific model/evaluation outputs live under `results/`; focused derivative
diagnostics use `results/diagnostics/`. Thesis-specific generators live in
[`script/thesis/`](thesis/README.md) and export locally under `img/03_methodology`
and `img/04_results`, with copyable figure/table LaTeX. They do not edit the
manuscript. Plotting caches use the system temporary directory.
`state/` is reserved for optional scratch orchestration and validation; reusable
numerical code belongs in `src/sindy_viv`, which never imports scratch code.

The thesis folder contains `generate_methodology_plan.py`,
`generate_results_plan.py`, `generate_single_structural_results_figures.py`,
and `results_plan_data.py`. Invoke them as `python script/thesis/<script>.py`;
see the [thesis script guide](thesis/README.md) for local generation, examples,
data preparation and copyable blocks. Manuscript prose and placement are deferred.

## Case 1/2 with Case 3 motion equations

`evaluate_case3_hybrid.py` takes two saved model directories or their
`training.json` paths. It retains every fitted lift equation from Case 1/2 and
replaces **both** motion equations (the hdot and hddot targets) with Case 3.
The models may use different feature libraries and parameterizations. Case 3
lift controls come from the evolving Case 1/2 lift states during the IVP.
No coefficient fitting or source-file modification occurs.

```powershell
.\.venv\Scripts\python.exe script/evaluate_case3_hybrid.py `
  --case_12_model "path/to/dynamics_case_2/.../training.json" `
  --case_3_model "path/to/dynamics_case_3/.../training.json" `
  --derivative_window_cases configs/fit_window/single/overall_reduced.json configs/fit_window/single/sinusoidal.json configs/fit_window/single/sinusoidal_evaluation.json `
  --integral_window_case configs/fit_window/single/sinusoidal_evaluation.json `
  --spectral_window_cases configs/fit_window/single/sinusoidal.json configs/fit_window/single/sinusoidal_evaluation.json `
  --phase_comparison
```

Use the `two_tandem` evaluation-window configs for tandem models. VR/L/D default
to the Case 1/2 training selection; `--vr_list` and `--lpd_list` select evaluation
trajectories independently. An individual model must match the selected VR/L/D;
a parametric model can vary only its parameterized dimensions. Both sources
must have the same cylinder type, time coordinate, and (when recorded) Re/mstar.
The Case 1/2 metadata snapshots supply the reference differentiator, filtering,
and feature-library settings, even if their original config paths have changed.
Derivative FFT/amplitude analysis defaults to all derivative windows unless
`--spectral_window_cases` supplies a subset. Integral FFT/amplitude analysis,
partial-IVP handling, NRMSE heatmaps, and optional phase analysis use the normal
Case 1/2 suite.

Outputs are beneath the original Case 1/2 model's `evaluation` folder. Both
individual analysis and aggregate summary paths insert
`evaluation_window=<window>/case_3_hybrid/<Case 3 relative model path>/`.
The mirrored Case 3 path contains everything below `dynamics_case_3`, including
filtering, differentiation, parameterization, libraries, optimizer, training
selections, and fitting window/time folders. Time-domain artifacts are in the
hybrid analysis folder's `time_domain` child; spectral artifacts remain beside
it. Each destination includes `hybrid_manifest.json` with both source metadata
snapshots, coefficient hashes, and the replaced target names. Models outside a
`dynamics_case_3` tree receive a path-derived identifier to prevent collisions.

A manually entered theoretical Case 3 CSV without `training.json` can be selected
using a separate **model-info JSON** passed to `--case_3_model`. This uses the
same metadata schema and adds `model_directory` pointing to its existing CSV
folder (absolute or relative to the JSON). For example, a single-cylinder,
individual linear model at VR=3 needs:

```json
{
  "model_directory": "path/to/theoretical/model/folder",
  "dynamics_case": 3,
  "cylinder_type": "single",
  "parameterization": "individual",
  "time_coordinate": "tau=t*/VR",
  "trajectories": [{"VR": 3, "LpD": null}],
  "configurations": {
    "state_feature_library": {
      "module": "pysindy.feature_library",
      "class_name": "PolynomialLibrary",
      "kwargs": {"degree": 1, "include_bias": true}
    }
  }
}
```

For parametric theoretical models also supply `parameter_feature_library` in
`configurations`, set the matching `parameterization`, and list the source
training VR/L/D values. The CSV feature and target names are validated against
these settings; coefficients are never inferred from filenames.

## Lift-only identification: Cases 6 and 7

The previous Case 6/7 study controller and its saved studies were removed at
the user's request. The subsequent first-order lift experiment setup and its
diagnostics/outputs have also been removed. The replacement
[`run_lift_experiments.py`](run_lift_experiments.py) implements the explicitly
authorized [full CFD-data recomputation](FULL_RECOMPUTE.md), with the single
group local and the three tandem groups reserved for Slurm. It defaults to a
dry run and never imports old lift fits or archive-selected parents.
Use the fitting/evaluation CLIs below for individual operations; no unattended
study is launched by these entry points unless explicitly invoked.

The supported dynamics cases are **1, 2, 3, 4, 6, and 7**. Case 5 was removed;
its old preparation aliases, wake factory, and feature presets are no longer
available. Existing saved results are left in place.

| Case | Learned state x | Learned derivative xdot | Measured response controls |
| --- | --- | --- | --- |
| 6 | CL | CLdot | h, hdot |
| 7 | CLdot | CLddot | h, hdot, CL |

For tandem data, states are grouped by channel across the two cylinders;
controls are grouped by cylinder. VR/L/D are appended only for parametric fits,
in the existing parameter-library order. Acceleration is never an input in
either new case. Case 7 derives its state and native second-derivative target
from filtered lift before cropping. Generic polynomial libraries are suitable;
Case 4 feature presets have a different input layout.

Use the existing `perform_sindy_*` / `perform_weak_sindy_*` commands with
`--dynamics_case 6` or `7`. Individual, VR, L/D, and joint VR/L/D fitting retain
their usual meanings. Cases 6/7 do not have a known state equation to constrain
during training, so use an unconstrained optimizer rather than ConstrainedSR3.

The existing `evaluate_sindy_single_suite.py` and
`evaluate_sindy_two_tandem_suite.py` now add a **complete coupled IVP** for these
cases, alongside the standard forced derivative/IVP/FFT/amplitude assessments:

- Case 6 evolves `[h, hdot, CL]`, with h'=hdot exactly.
- Case 7 evolves `[h, hdot, CL, CLdot]`, with h'=hdot and CL'=CLdot exactly.

By default motion uses the theoretical acceleration equation
`hddot = -4*pi^2*h + 2*VR^2*CL/(pi*structural_mstar)`. Override the true mass
ratio using `--structural_mstar` if required. Supply `--case_3_model` followed
by one or more saved Case 3 folders, `training.json` files, or model-info JSONs
to use fitted acceleration instead. The Case 3 hdot equation is always discarded.
Multiple donors can cover individual VR/L/D fits; exactly one must match each
evaluated trajectory. An individual donor cannot be extrapolated to another
VR/L/D. The source models may have different parameterizations and libraries.

For each evaluated trajectory, parameter features from **both** models are
substituted at that VR/L/D before their response coefficients are merged into
one autonomous Xdot model. Only initial measured states initialize this IVP;
later measured motion and lift never drive it. The existing solver limits,
partial-IVP diagnostics, NRMSE heatmaps and frequency/amplitude reports apply.
`--phase_comparison` also assesses CL-minus-h phase for the complete IVP.

Example: individually fitted Case 6 with a VR-parametric Case 3 donor:

```powershell
.\.venv\Scripts\python.exe script/evaluate_sindy_single_suite.py `
  --dynamics_case 6 --parameterization individual --vr_list 3 `
  --differentiator_case configs/differentiator/finite_difference_basic.json `
  --state_feature_library_case configs/feature_library/polynomial_3.json `
  --optimizer_case configs/optimizer/ols.json `
  --fit_window_case configs/fit_window/single/sinusoidal.json `
  --derivative_window_cases configs/fit_window/single/sinusoidal.json configs/fit_window/single/sinusoidal_evaluation.json `
  --integral_window_case configs/fit_window/single/sinusoidal_evaluation.json `
  --spectral_window_cases configs/fit_window/single/sinusoidal_evaluation.json `
  --case_3_model "path/to/VR-parametric/Case3/training.json" `
  --phase_comparison
```

Change the case to 7 for the lift-rate model. Omit `--case_3_model` for theory.
Standalone integral/FFT commands also accept `--coupled_motion` together with
`--case_3_model`; their ordinary forced evaluations use measured controls.

Complete-IVP suite reports stay under the lift model's `evaluation` tree.
Below `evaluation_window=<window>`, theoretical assessments use
`theoretical_motion/`; saved Case 3 assessments use
`case_3_hybrid/<mirrored Case 3 folders>/`. Aggregate summaries with multiple
donors use `case_3_hybrid/models=<selection hash>/`; per-trajectory reports
retain the actual selected donor's folder structure. Each time-domain output
includes `complete_model.csv` and `complete_model.json`: the substituted
coefficient matrix, exact identities, source hashes/metadata, and evaluated
VR/L/D. Standard forced reports retain their existing destinations.

## Theoretical tandem assessment and legacy coefficient plots

`evaluate_theoretical_two_tandem.py` evaluates five manually supplied Case 3
VR-parameterized coefficient models across the full tandem grid. It validates
named coefficients and runs the existing suite without fitting. Use `--help`
for model/configuration/output selectors; `--dry-run` validates without
integrating. Its optional `--after_study` reads a supplied predecessor status;
it does not require an old study runner to remain in the checkout.

The script calls `refresh_tandem_coefficient_heatmaps.py` before assessment.
This is a migration of saved plot files from the older format that stacked
both acceleration targets in one coefficient heatmap. It searches for:

- `hddot_coefficients_summary.json` and its heatmap CSV/PNG: effective
  coefficients across VR, normally grouped by L/D.
- `hddot_tensor_coefficients_summary.json` and its heatmap CSV/PNG: fitted
  tensor coefficients retaining VR powers, with columns across L/D.

Only summaries whose `targets` list contains both hddot0 and hddot1 qualify.
The utility splits their target-prefixed rows into `hddot0_*` and `hddot1_*`
PNG/CSV/JSON reports. It checks the new numeric tables against the originals,
records the original JSON and file hashes in a migration manifest, and then
removes the old combined report triplet. It does not alter `model.csv` or
refit any coefficients. Reports already separated by cylinder are left alone.
The current suite writes separate plots directly. The existing tandem tree
contained no eligible combined summaries in the 2026-11-01 check.

| Command family | Entry points | Package implementation and usage guide |
| --- | --- | --- |
| Pointwise fitting | `perform_sindy_single_*`, `perform_sindy_two_tandem_*` | `compute_sindy.compute_sindy`; [fitting guide](../src/sindy_viv/case_bryan/compute_sindy/README.md). |
| Weak fitting | `perform_weak_sindy_*` | `compute_sindy.compute_weak_sindy` and generic temporal weak integration. |
| Saved-model derivatives/IVPs | `evaluate_derivative_sindy_*`, `evaluate_integral_sindy_*` | `evaluate_sindy.evaluate_sindy`; [evaluation guide](../src/sindy_viv/case_bryan/evaluate_sindy/README.md). |
| Complete experiment reports | `evaluate_sindy_single_suite.py`, `evaluate_sindy_two_tandem_suite.py` | `evaluate_single_suite`/`evaluate_two_tandem_suite`; shared `suite_parser`. |
| Raw time histories and FFTs | `plot_all_variables_*`, `frequency_analysis_*` | Processed loaders and common time-series/frequency figure builders. |
| Filtering, numerical derivatives and amplitudes | `assess_filtering*`, `assess_derivative.py`, `plot_derivative_*`, `amplitude_analysis.py` | `analysis` and `evaluate_sindy.derivative_assessment`; [diagnostic guide](../src/sindy_viv/case_bryan/analysis/README.md). |
| Frequency/phase/surface summaries | `plot_frequency_vr_summary_*`, `plot_phase_vr_summary_*`, `plot_frequency_surface_all_*` | Shared `analysis.plot_cli` dispatchers and numerical frequency/phase analysis. |
| Saved-error heatmaps | `plot_error_heatmap_*` | Read existing derivative/IVP metrics and aggregate without refitting. |
| Model/reference FFT assessment | `evaluate_frequency_sindy.py` | `evaluate_frequency_derivative`/`evaluate_frequency_integral`. |
| Feature correlation/conditioning | `assess_feature_library.py` | Bryan preparation plus generic QR/SVD computation; [assessment guide](../src/sindy_viv/case_bryan/feature_library/README.md#feature-library-assessment). |

The root batch launchers and `old_run/` were removed during cleanup. Use the
Python commands above directly. A `--dry-run` option is available on the study
runners; check `--help` before applying it to other scripts.
`script/check` contains focused manual checks. Historical standalone
implementations in `script/old` have been removed.

`script/check/compare_clddot_methods.py` compares native second derivatives
against differentiating lift rate again, using finite differences and the
configured Savitzky-Golay base preset on the same Butterworth-filtered lift.
Its defaults select single VR=7 and tandem L/D=1.5, VR=9 (both cylinders),
using each dataset's `sinusoidal_evaluation` window. Run it with the project
Python from the repository root; `--help` lists dataset, zoom and output options.
It saves overlay/difference PNGs, comparison CSVs and `summary.json` under
`results/diagnostics/clddot_comparison` by default. Filtering/differentiation use full records
before cropping. RMS differences measure disagreement between methods, not
accuracy against an exact derivative. This manual script performs no fitting.

`script/check/compare_clddot_integration.py` reads those saved comparison CSVs
and reconstructs `CLdot` by integrating each `CLddot` route with the shared
`reconstruct_from_derivatives` trapezoid implementation. Both start from the
same initial numerical `CLdot`. It writes reconstruction/error plots, CSVs and
NRMSE (RMSE divided by reference standard deviation) under
`results/diagnostics/clddot_integration`. The reference is method-specific numerical `CLdot`;
the report checks reintegration consistency, not independent physical accuracy.

## Read selectors consistently

`--cylinder_type`, or a single/tandem-specific filename, identifies the dataset.
`--vr_list` and `--lpd_list` choose physical trajectories. `--dynamics_case`
chooses state/control layout, while `--parameterization` or the fitting filename
chooses whether VR/L/D are tensor parameters. Library/optimizer/filter JSONs
select configured objects. Training and evaluation windows have separate roles;
the package guide for each command defines those selectors explicitly.

`parse_args()` returns a namespace. `main()` runs the corresponding workflow,
returns its result where applicable and reports failures to its batch caller.
Some small scripts import a shared `main` directly instead of duplicating a
wrapper. Shared parsers live in `analysis.plot_cli`, `evaluate_sindy.suite_cli`,
and `feature_library.assessment_cli`; update the shared owner when adding a
selector used by several entry points.

## Add or maintain an argparse option

Use a multiline declaration with the flag and each keyword on separate lines:

```python
parser.add_argument(
    "--batch_size",
    type=int,
    default=8192,
    help=(
        "Rows per computation batch; every selected sample remains in the "
        "diagnostic calculation."
    ),
)
```

Keep each option's type, default, allowed values and help together. Wrap long
help text as adjacent string literals. Empty no-argument calls such as
`add_mutually_exclusive_group()` can remain on one line. Docstrings explain
parser/workflow responsibilities; the argparse help explains user selections.
After changing a shared parser, check both `--help` and the relevant batch dry
run so every generated command still parses.


## Removed study workflows

The old `run*` and `review*` scripts and the legacy `script/old/` directory have
been removed, including the Case 6/7 overnight controller. Old Case 6/7 study
folders and canonical model/evaluation trees were erased. The Case 6/7 model
implementations, ordinary fit/evaluation CLIs, comparison utility, and thesis
figure/planning tools remain. Tests specific to the removed controllers were
removed with them.

[`inputs/case3_motion/`](../inputs/case3_motion/README.md) retains the two
hash-recorded Case 3 donor models for complete-model evaluation. For an
individual complete-model suite, supply the desired `training.json` with
`--case_3_model`. These inputs do not schedule runs or resume deleted studies.
