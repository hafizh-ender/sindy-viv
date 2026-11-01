#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --partition=bigmem
#SBATCH --mem=350G
#SBATCH --mail-user=hafizh-renanto.akhmad@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=tandem_lpd_4p0
#SBATCH --output=result/experiment/slurm_logs/%x_%j.out
#SBATCH --error=result/experiment/slurm_logs/%x_%j.err

# Submit from the repository root, as with template_slurm.slurm.
set -e

module purge
module list

source activate tesis_itb_stable

export UV_CACHE_DIR="$PWD/.uv-cache"
export TMPDIR="$PWD/.tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

echo "Python:"
which python
python --version

echo "Environment:"
echo "$VIRTUAL_ENV"

echo "Running job..."
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -B -u script/run_lift_experiments_hpc.py --group tandem_lpd_4p0 --stage all --execute --resume --output-root result/experiment
