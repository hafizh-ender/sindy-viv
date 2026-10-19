# SINDy for VIV

Research code for a Master's thesis applying SINDy to single and tandem
oscillating-cylinder VIV data.

The importable package is under src/sindy_viv. Command-line entry points in
script delegate to its computation/evaluation APIs. Configuration files under
configs select differentiators, state/parameter libraries, optimizers and
trajectory-owned fitting windows.

- [Computation workflows](src/sindy_viv/case_bryan/compute_sindy/README.md)
- [Derivative and integral evaluation](src/sindy_viv/case_bryan/evaluate_sindy/README.md)
- [Wake feature libraries](src/sindy_viv/case_bryan/feature_library/README.md)
- [Fitting-window configurations](configs/fit_window/README.md)

CSV timestamps are fluid time t*. Processed loading converts them once to
float64 dynamic time tau=t*/VR. Preparation and differentiation do not divide
time by VR again. Fitting/evaluation bounds use recorded t*; state/control
storage defaults to float64 and can explicitly use float32.

DataFrame-based raw loaders and conversion helpers now live in
sindy_viv.case_bryan.compatibility.dataframe_loading and dataframe_conversion;
imports through input/preprocessing remain compatible. Bryan's specialized
uniform timestamp reconstruction is retained. WeakSINDy entry points remain
explicit unimplemented placeholders.

Public and nontrivial internal functions document their array/layout contracts
and carry type annotations. Annotations support readers and static tooling;
runtime validation still checks shapes, finite values and physical selections.

Run the regression suite with the project's Python environment:

    python -m unittest discover -s tests

Tests use temporary directories outside the repository for generated artifacts.

Bryan-specific support modules are grouped by purpose:

- `prepare_sindy/case_layout.py`: state and measured-control column definitions.
- `prepare_sindy/trajectory.py`: prepared arrays, derivatives, and parameter columns.
- `model_storage/fit_windows.py`: fitting-window validation and cropping.
- `model_storage/paths.py`: artifact paths and saved-model discovery.
- `evaluate_sindy/model_evaluation.py`: prediction, integration, and reports.
- `compatibility/`: historical DataFrame loading/conversion and wake pickle callables.

The small `feature_library/legacy.py` import alias is retained only so existing
model pickles can resolve their previous callable path. New fitting uses the
descriptive compatibility module directly.

The full-data polynomial experiment is reproducible and resumable with
`python script/polynomial_sindy_study.py --output results/polynomial_study_20261018 --workers 2 --memory-fraction 0.7 --max-fit-memory-gib 3.13`.
Models exceeding its RAM estimate are recorded as skipped.

Add `--report-only` to regenerate study tables and figures without fitting models.
