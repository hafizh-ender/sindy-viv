# Frozen full-recomputation group configs

Each file defines one complete Case 6 group for `script/run_lift_experiments.py`.
The single group has six VRs; each of the three tandem spacings has all VRs
5,10,14. Stage C selects parents only from seven newly computed Ridge attempts
for the same condition and degree. There is no archive-recovery input.

The numerical convention is versioned: full bias-including polynomials,
normalization off, no final unbias, direct GELSD OLS with max(N,P)*eps cutoff,
positive-alpha SVD Ridge, and sequential zero/positive-penalty support refits.
Structural acceleration is the existing hard-coded theoretical equation; no
old fitted structural/lift coefficients are reused. Windows and preprocessing
descriptors are snapshotted into every job's settings and immutable identity.

The known rank/cutoff caveat is reported rather than used as an execution gate.
Changing the scientific convention requires an explicit new experiment version
and a fresh compatible output context. Per-job configs also contain source and
CFD-byte fingerprints, optimizer backend, package versions and evaluation bounds.

See [the run guide](../../../script/FULL_RECOMPUTE.md) for the matrix, local
command, three manual Slurm submissions, checkpoints and reporting.
