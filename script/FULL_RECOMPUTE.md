# Full Case 6 recomputation: local single / HPC tandem

This implements [the execution brief](../CODEX_FULL_RECOMPUTE_LOCAL_SINGLE_HPC_TANDEM.md).
All lift coefficients are newly fitted from CFD measurements through the existing
preparation and evaluation APIs. No TAR, historical lift model, metric row,
archived parent list or Case 3 fitted donor is imported. CFD itself is not rerun.
There are no figure or manuscript operations in this runner.

## Fixed matrix

| Group | VR | L/D | Environment | Fixed fits | Maximum total |
| --- | --- | --- | --- | ---: | ---: |
| single | 3,4,5,6,7,8 | null | Local | 198 | 366 |
| tandem_lpd_1p5 | 5,10,14 | 1.5 | HPC | 99 | 183 |
| tandem_lpd_3p0 | 5,10,14 | 3.0 | HPC | 99 | 183 |
| tandem_lpd_4p0 | 5,10,14 | 4.0 | HPC | 99 | 183 |
| Total | 15 conditions | | | **495** | **915** |

Stage A is true OLS at degrees 1–5. Stage B is genuine Ridge at degrees 3/5 and
alpha 1e-6,1e-4,1e-3,1e-2,.1,.5,5. Stage D independently runs zero-penalty
sequential OLS at degrees 3/5 and thresholds .001,.005,.01,.05,.1,.5,1.
Stage C refits every threshold for up to two selected new Stage-B alphas per
condition/degree. The serial order per condition is **A, B, D, C**.

The four [group configs](../configs/experiments/full_recompute) freeze the matrix
and `full_recompute_20261101_v1` experiment identity. Source/content/input hashes
prevent incompatible checkpoints from being reused. A subset or smoke job does
not alter the grid or the experiment's definition of completion.

## Frozen numerical convention

- Case 6 individual lift closures, full polynomial library including bias.
- Train `sinusoidal`; evaluate CLdot at recorded states on both `sinusoidal`
  and `sinusoidal_evaluation`; integrate over the entire latter window.
- Existing full-record Butterworth filtering, differentiation, cropping,
  tau=t*/VR and variable order are preserved. Full-record noncausal filtering
  can share boundary information; this caveat is recorded, not redesigned.
- **No column normalization**, no separate intercept, no final unbias refit.
  Coefficients and thresholds use the original polynomial feature coordinates.
- OLS and zero-penalty active-support refits use SciPy GELSD with the explicit
  relative cutoff `max(N,P)*float64_eps`. Singular values, rank, column norms,
  training SSE and cutoff are retained in diagnostics/checkpoints.
- Ridge-only directly uses sklearn `Ridge(solver='svd', fit_intercept=False)`.
  Its objective is `||Y-XW||² + alpha*||W||²`, including the polynomial bias
  coefficient. It has no sparsity threshold. Ridge+threshold repeatedly refits
  shrinking supports using that same positive penalty. Zero-alpha thresholding
  calls GELSD directly, never a zero-alpha Ridge routine.
- Sequential thresholding has max_iter=100. A cap warning is recorded and any
  remaining below-threshold coefficients are zeroed. Active-term reports count
  abs(coef)>1e-12; that reporting tolerance does not change the model CSV.
- Complete motion uses the repository's **hard-coded theoretical acceleration**:
  `h'=hdot` exactly and `hddot=-4*pi²*h + 2*VR²*CL/(pi*mstar_true)` for each
  cylinder. Single true mass is `2/(pi*.125)`; tandem true mass is 2.54648.
  The lift equation is the newly fitted closure. No measured later states drive
  this autonomous coupled IVP and no old fitted structural coefficients are used.
- RK45, rtol=1e-6, atol=1e-9, max_state_factor=10. A separate IVP worker has a
  120-second supervisor limit including startup/loading. Fits have no artificial
  timeout. Local workers use one BLAS thread; Slurm workers retain the template's
  allocated thread count. Versions/thread settings are recorded.

The known unscaled polynomial cutoff/rank effects are disclosed in each run.
Nonmonotonic OLS training SSE is flagged in group summaries and case diagnostics;
**it does not block execution, trigger scaling changes or start extra studies**.
The previous independent comparison and embedded-cubic observation are background
evidence only and are not reused as new fitted results.

## Parent selection and IVP scores

Only the seven terminal new Ridge checkpoints for the same condition/degree are
eligible. Failed IVPs remain eligible if fitting succeeded. With completed IVPs,
A has the lowest finite full-window mean h/CL NRMSE and B has the best finite
validation CLdot score at a distinct alpha. Without completed IVPs, A has the
best validation derivative score and B has the farthest verified failed endpoint
at a distinct alpha. If no other endpoint is known, the next finite derivative
candidate is the deterministic fallback. Ties use alpha, then run ID. Missing
required scores stay unrankable rather than receiving invented values.

Per-slot manifests freeze all candidates, reasons and selected roles; resumes use
exactly those parents. Changed Stage-B records or artifact hashes are rejected.
Every Stage-C configuration is a new fit, with parent status/score and the
completion transition retained in its checkpoint.

IVP completion and accuracy are separate. Full-window h/CL/hdot scores are null
on state-limit failures, nonfinite results, solver errors and timeouts. Accepted
prefix scores use `partial_*` names. `actual_t_end` is the last finite sampled
endpoint returned by the core evaluator; a killed/exception-only worker has no
verified endpoint and records null. NRMSE uses population standard deviation of
the same-window reference. Ranking means are used only for parent policy; the
progress report selects best h and CL predictions separately.

## Local commands

From repository root with the existing environment:

```powershell
# No fits; full matrix, including all nonempty tandem groups.
.venv/Scripts/python.exe -B script/run_lift_experiments.py --all-groups --dry-run

# Complete six-case local run, or exact checkpoint resume.
.venv/Scripts/python.exe -B -u script/run_lift_experiments.py --group single --stage all --execute --resume --output-root result/experiment

# Refresh the progress report without fitting.
.venv/Scripts/python.exe -B script/run_lift_experiments.py --report --output-root result/experiment
```

The executed smoke test uses one configuration from the unchanged grid, isolated
under `state/full_recompute_smoke`. It is not imported into the full run. The full
local job refits all six cases under `result/experiment/single`.

`--stage`, `--vr`, `--max-jobs`/`--stop-after` allow controlled diagnostics;
they do not redefine a complete group. `--single-data-dir` / `--tandem-data-dir`
override raw roots. Defaults retain `../sindy-viv/data/...` from constants.py.
CLI execution of tandem groups is rejected outside Slurm before loading inputs.

## HPC submission

The three files in [slurm/](../slurm) retain the actual local template's one node,
one task, 12 CPUs, 350G memory, bigmem partition, 12-hour limit, mail settings,
module commands and `source activate tesis_itb_stable`. No account/QOS or cluster
home path is invented. Scripts do not install packages. The verified local
numerical versions are listed in [requirements-full-recompute.txt](../requirements-full-recompute.txt);
the project's broader dependencies/lock remain in pyproject.toml and uv.lock.

After updating the repository on HPC, **submit from repository root** so Slurm
can resolve the already-present tracked log directory:

```bash
sbatch slurm/run_tandem_lpd_1p5.slurm
sbatch slurm/run_tandem_lpd_3p0.slurm
sbatch slurm/run_tandem_lpd_4p0.slurm
```

Each script runs all three VR cases and all four families for its own spacing.
It uses repository-relative code/output paths and the established sibling CFD
data tree. Each group has an independent lock, summary and metadata containing
job ID, node, Git revision, versions and start/end timestamps. No tandem job is
submitted or executed by the local implementation task.

## Checkpoints and reporting

`result/experiment/<group>/<case-id>/<run-id>` contains immutable config/settings,
model.csv, training.json, native derivative/IVP reports, a fit receipt, phase
responses, record.json, status and separate stdout/stderr logs. A fit receipt is
published before derivative/IVP evaluation, so interruption does not require a
completed fit to be repeated. Terminal numerical failures are not retried on
resume. Interrupted unfinished phases can resume after a Slurm wall-time stop.

Each group atomically rebuilds its own summary.csv from checkpoints; failed
fitting attempts remain rows. `progress.json` distinguishes fixed attempts,
selected threshold jobs, unselected slots and a remaining upper bound. A group
is Finished only after fixed stages **and all selected Stage-C jobs** were
attempted. Singles' downstream fields and L/D are blank/null.

The root FULL_RECOMPUTE_REPORT.md reflects visible groups without inventing HPC
results. `summary_all.csv` is deferred until all four groups are visible for the
same experiment. Report publication has its own lock; group workers never append
to a common mutable CSV. Generated models/logs/statuses are ignored by Git;
the portable plan and scheduler-log `.gitkeep` are retained.

Tests: `python -m unittest test_full_recompute test_utils test_prepare_optimizer`
with repository/tests/src on PYTHONPATH. They use small synthetic matrices and
mocked records; no real tandem fitting or integration is part of testing.
