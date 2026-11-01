"""HPC entry point for the shared recomputation runner, without Git calls.

The local sweep keeps its frozen runner unchanged. Only runtime metadata
collection is adapted here; fitting, evaluation, selection and resume all
remain in run_lift_experiments.py. Workers do not collect Git metadata.
"""
import importlib.metadata
import os
import platform

if __package__:
    from . import run_lift_experiments as runner
else:
    import run_lift_experiments as runner


def batch_environment():
    versions={key:importlib.metadata.version(package) for key,package in
        [('pysindy_version','pysindy'),('scipy_version','scipy'),
         ('numpy_version','numpy'),('sklearn_version','scikit-learn')]}
    return dict(versions,python_version=platform.python_version(),platform=platform.platform(),
        git_revision=None,node=platform.node(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        blas_threads={key:os.environ.get(key) for key in
                     ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})


def main(argv=None):
    original=runner.environment
    runner.environment=batch_environment
    try:
        return runner.main(argv)
    finally:
        runner.environment=original


if __name__=='__main__':
    raise SystemExit(main())
