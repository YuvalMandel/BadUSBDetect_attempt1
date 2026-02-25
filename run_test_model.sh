#!/bin/bash
#SBATCH --job-name=htm_other
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/htm_other_%j.out
#SBATCH --error=logs/htm_other_%j.err

# Initialize Conda in this non-interactive shell
source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh
conda activate htm_keyboard_1

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Avoid oversubscription inside BLAS/NumPy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on $SLURM_CPUS_PER_TASK CPUs"

python htm/htm_test_model.py \
    --model  models/cfg0025_sp20_enc16w3_tm32_act20_al_vf11.0000_tf11.0000.pkl \
    --mode all_other_files \
    --data_root ../UB_keystroke_dataset/ \
    --write_decision_log
