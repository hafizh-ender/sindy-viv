# Explicit SVD optimizer backends

`SVDLeastSquares`, `RidgeOnly` and `SequentialSVD` are ordinary PySINDy
BaseOptimizer subclasses and can be constructed by the existing JSON loader.
They depend only on installed numerical packages, not scripts or scratch code.

OLS uses SciPy `lstsq(lapack_driver='gelsd', cond=max(N,P)*float64_eps)`.
Ridge-only uses actual sklearn `Ridge(solver='svd', fit_intercept=False)` and
requires alpha>0; it has no threshold. SequentialSVD monotonically shrinks each
target's support and refits via direct GELSD for alpha=0 or SVD Ridge otherwise.
No normal equations or explicit matrix inverses are formed.

Normalization and final unbiasing are explicitly rejected for this convention.
The library supplies the bias. Thresholds and saved coefficients use original
feature coordinates. The temporary zero initial guess prevents a redundant
BaseOptimizer initial NumPy regression; each backend computes its own genuine
full-support initial solution. No library is cached between fits.

SequentialSVD records support convergence in `converged_`, iteration count and
coefficient history. Its default maximum is 100; on the cap it emits a warning
and zeros remaining below-threshold coefficients. BaseOptimizer still retains
its standard Theta_ matrix for the life of the fitted object; the experiment
isolates each fit in its own process.

See the [full recomputation guide](../../../script/FULL_RECOMPUTE.md) for fixed
experiment settings, rank caveats, local execution and HPC submission.
