# Command-line entry points

Run maintained scripts from the repository root with this checkout's Python:

```powershell
.\.venv\Scripts\python.exe script/assess_feature_library.py --help
```

Scripts parse dataset/config/output selections and delegate to functions in
`src/sindy_viv`. Numerical behavior belongs in those package functions so APIs,
CLI commands and batch runners use the same implementation.

## Adaptive overnight study

`evaluate_theoretical_two_tandem.py` is an evaluation-only follow-up for manual
Case 3 VR coefficient CSVs. `model_selection` validates the five L/D models and
50 trajectory selections; `verify_unchanged` checks their hashes after the full
suite. It delegates numerical work to `evaluate_two_tandem_suite` using explicit
model directories, without adding training metadata or fitting coefficients.
Parameterized tandem coefficient maps are separate for hddot0 and hddot1,
including VR-term aggregates across L/D. `refresh_tandem_coefficient_heatmaps.py`
can split existing combined PNG/CSV/JSON reports using their saved full-precision
values. It validates replacements before removing obsolete combined artifacts
and preserves original JSON payloads in its manifest. The queued theoretical
evaluator performs a final refresh after preceding overnight work finishes.
`run_theoretical_two_tandem.bat --dry-run` validates inputs only. The follow-up
can be queued behind the overnight helper; `--after_study <study_status.json>`
prevents it from running if the predecessor was interrupted before completion.

`run_overnight_study.py` orchestrates the approved sequential study, using the
existing fitting and full-suite CLIs. `run_overnight.bat` selects this checkout's
Python. No experiment or IVP timeout is added.

```powershell
.\run_overnight.bat --dry-run
.\run_overnight.bat --output_dir state/overnight_study_runs/my_run
.\run_overnight.bat --output_dir state/overnight_study_runs/my_run --resume
.\run_overnight.bat --output_dir state/overnight_study_runs/my_run --resume --skip-tandem
```

The fixed grid has 17 tandem configurations, followed by 96 single-cylinder
ridge configurations. `--skip-tandem` bypasses all remaining tandem work and
starts directly at the ridge sweep, preserving completed outputs. The full
single-cylinder grid has eight alphas, two normalization settings, two degrees,
and three cases. Every fit has the existing derivative, IVP, FFT/amplitude and
applicable phase/coefficient assessments. Model outputs retain their standard
`results` paths; raw-data diagnostics are not repeated.

The tandem Poly5 decision uses the equally averaged quartic lift-derivative
NRMSE across Cases 1/2/4, VR and both cylinders. If it exceeds 0.01, run Poly5
for all three tandem cases. Missing derivative results are recorded explicitly
and cannot justify an accuracy-based extension. This branch has no IVP gate.

For the single-cylinder ridge sweep, lift reconstruction uses CLdot in Case 1
and CLddot in Cases 2/4. Each case/VR has equal weight; failed IVPs do not
exclude valid derivative errors. Missing derivative records invalidate the
aggregate rather than silently improving it. Stability requires completion of
all 18 case/VR IVPs, using coupled integration for Case 4. Other Case 4 IVP
reports are retained by the full suite.

Single Poly5 is skipped if no fully stable lower-order combination exists,
or if any fully stable Poly3/4 pair meets the 0.01 derivative target. Otherwise,
only fully stable alpha/normalization pairs proceed, deduplicated across orders.

Threshold selection keeps all fully stable pairs per degree. If fewer than
three exist, fill to three by completed-IVP count, then lower lift-derivative
NRMSE. Partial candidates are eligible only here. If strict Poly5 eligibility
produced fewer than three allowed pairs, record the shortfall instead of
introducing ineligible pairs. Thresholds are 1, 0.5, 0.1, 0.05, 0.01, 0.005 and
0.001; each keeps its ridge alpha, normalization and `unbias: false`. Generated
presets live in `configs/optimizer/threshold_sweep/`. Threshold-zero baselines
remain eligible for final selection.

`Experiment` describes one multi-trajectory configuration. `commands` builds
matching fit/suite commands. `lift_score`, `completion_count`, and
`aggregate_candidate` score saved reports; `candidate_rank`,
`select_poly5_candidates`, and `select_threshold_candidates` implement policy.
`Study` snapshots sources/configs, runs sequential subprocesses, counts retained
terms from each `model.csv`, and writes checkpoints and decisions. `preflight`
validates input windows and fixed presets without loading CFD data.

Each output directory contains `study_status.json`, frozen `input_snapshot`,
per-experiment commands/stdout/stderr/checkpoints under `experiments`,
`candidate_results.csv/json`, adaptive-decision JSONs, and `final_selection.json`.
Candidate JSONs retain per-case/VR records and per-model nonzero-term counts.
The background launcher additionally maintains `status.json` and `queue.log`.
Resume reuses successful experiment checkpoints, including completed suites
with partial IVPs; unsuccessful subprocesses are retried. Source/config
snapshots keep a resumed study's inputs consistent.

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
| Process queue controls | `run_data_diagnostics_after_process.ps1`, `set_run_queue_state.ps1`, `run_all_boundary_control.ps1` | Process-dependent launch, pause/resume, and one-time workflow boundary controls; not part of numerical fitting. |

The root `run_all.bat` invokes the four current workflow batches. Each batch
accepts `--dry-run` to print its commands without loading data or fitting models.
`script/check` contains focused manual checks. `script/old` contains standalone
historical implementations; their local metrics, file layouts and time
conventions are documented in source and differ from maintained package APIs.
Those self-describing subfolders do not need additional folder READMEs.

`script/check/compare_clddot_methods.py` compares native second derivatives
against differentiating lift rate again, using finite differences and the
configured Savitzky-Golay base preset on the same Butterworth-filtered lift.
Its defaults select single VR=7 and tandem L/D=1.5, VR=9 (both cylinders),
using each dataset's `sinusoidal_evaluation` window. Run it with the project
Python from the repository root; `--help` lists dataset, zoom and output options.
It saves overlay/difference PNGs, comparison CSVs and `summary.json` under
`state/clddot_comparison` by default. Filtering/differentiation use full records
before cropping. RMS differences measure disagreement between methods, not
accuracy against an exact derivative. This manual script performs no fitting.

`script/check/compare_clddot_integration.py` reads those saved comparison CSVs
and reconstructs `CLdot` by integrating each `CLddot` route with the shared
`reconstruct_from_derivatives` trapezoid implementation. Both start from the
same initial numerical `CLdot`. It writes reconstruction/error plots, CSVs and
NRMSE (RMSE divided by reference standard deviation) under
`state/clddot_integration`. The reference is method-specific numerical `CLdot`;
the report checks reintegration consistency, not independent physical accuracy.

`run_all_boundary_control.ps1` supports a one-time stop before tandem computation.
Its request in `state/run_all_control/stop_after_single.json` records the exact
master PID and UTC start time. The tandem entry point checks that identity and
skips only the requested run. An optional hidden `ApplyAfterExit` helper waits
for the master to exit before installing staged runner updates. Source and
destination hashes are checked before copying, preserving intervening edits
for review. This control does not terminate active Python workers or start a
new experiment run.

The process-wait queue helper also accepts `-BatchPlanPath`: a JSON file with
an ordered `batches` array. Optional `readiness_state` and `required_phase`
fields prevent launching until deferred updates have completed successfully.
The default remains `run_all.bat` when no plan is supplied. Follow-up jobs keep
their own `status.json`, `queue.log`, and per-batch stdout/stderr logs.

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
