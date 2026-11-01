#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --partition=bigmem
#SBATCH --mem=350G

#SBATCH --mail-user=hafizh-renanto.akhmad@isae-supaero.fr #### votre  mail
#SBATCH --mail-type=FAIL,END                   #### mail quand c'est fini ou erreur
#SBATCH --job-name=tesis_itb                   #### nom du calcul

#SBATCH -o slurm.%j.out                     #### sorti standard STDOUT
#SBATCH -e slurm.%j.err                     #### sorti erreur STDERR

module purge
module list

# Use the project's uv-managed environment
# source .venv/bin/activate
source activate tesis_itb_stable

# Keep uv build/cache/temp files on scratch
export UV_CACHE_DIR="$PWD/.uv-cache"
export TMPDIR="$PWD/.tmp"

echo "Python:"
which python
python --version

echo "Environment:"
echo "$VIRTUAL_ENV"

echo "Running job..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# python ./script/perform_sindy_single_individual.py --dynamics_case 1 --differentiator_case ./configs/differentiator/finite_difference_basic.json --feature_library_case ./configs/feature_library/polynomial_2.json --optimizer_case ./configs/optimizer/stlsq_basic.json
