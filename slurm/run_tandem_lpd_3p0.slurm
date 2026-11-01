#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --partition=bigmem
#SBATCH --mem=350G
#SBATCH --mail-user=hafizh-renanto.akhmad@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=tandem_lpd_3p0
#SBATCH --output=result/experiment/slurm_logs/%x_%j.out
#SBATCH --error=result/experiment/slurm_logs/%x_%j.err

# Resource/environment settings are retained from template_slurm.slurm.
# Submit from repository root: sbatch slurm/run_tandem_lpd_3p0.slurm
set -euo pipefail
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
if [[ "${SLURM_SUBMIT_DIR:-$PWD}" != "$REPO_ROOT" ]]; then
    echo 'Submit from repository root so Slurm can resolve the tracked log directory.' >&2
    exit 2
fi
cd "$REPO_ROOT"
module purge
module list
# Conda activation scripts may reference unset variables.
set +u
source activate tesis_itb_stable
set -u
export UV_CACHE_DIR="$PWD/.uv-cache"
export TMPDIR="$PWD/.tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
printf 'Job %s on %s started %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)"
which python
python --version
printf 'Environment: %s\n' "${VIRTUAL_ENV:-${CONDA_PREFIX:-not-set}}"
# Raw CFD data use ../sindy-viv/data, exactly as case_bryan/constants.py.
# Full new computation: VR=5,10,14; all four families, automatic new-parent selection.
python -B -u script/run_lift_experiments.py --group tandem_lpd_3p0 --stage all --execute --resume --output-root result/experiment
printf 'Job finished %s\n' "$(date -u +%FT%TZ)"
