#!/bin/bash
# ============================================================
# SLURM job array: classify many files with one HTM-Velocity model
#
# Reads file paths from FILE_LIST (one path per line).
# Each array task classifies one file and appends a result line
# to logs/hv_classify_<JOBID>.txt.
#
# Build file list example:
#   find ../UB_keystroke_dataset -name "*.txt" > /tmp/files_to_classify.txt
#   find ../BadUSBdataset        -name "*.txt" >> /tmp/files_to_classify.txt
#
# Submit:
#   sbatch slurm/hv_classify_files.sh
#
# Collect results after all tasks finish:
#   cat logs/hv_classify_<JOBID>_*.txt | sort > hv_classify_results.txt
# ============================================================
#SBATCH --job-name=hv_classify
#SBATCH --output=logs/hv_classify_%A_%a.txt
#SBATCH --error=logs/hv_classify_%A_%a.err
#SBATCH --array=0-9999%200
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Edit these -------------------------------------------------
MODEL="hv_models/hv0102_sp30_d24w9_f16w9_q24w7_dk8w3_fk8w3_qk16w7_pm16w7_pd8w5_ps16w5_ws15s1_tm16_act10_wu2_al10_vf10.9952_tf10.9964.pkl"
FILE_LIST="/tmp/files_to_classify.txt"   # one .txt path per line
# TRUE_LABEL="human"                     # uncomment to add CORRECT/WRONG column
# -----------------------------------------------------------------

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

# ---- Pick file for this task ------------------------------------
mapfile -t FILES < "$FILE_LIST"
N_FILES=${#FILES[@]}

if [[ $N_FILES -eq 0 ]]; then
    echo "ERROR: FILE_LIST is empty or missing: $FILE_LIST"
    exit 1
fi

if [[ $SLURM_ARRAY_TASK_ID -ge $N_FILES ]]; then
    exit 0   # array declared larger than file list — nothing to do
fi

FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"

# ---- Run --------------------------------------------------------
CMD="python -u htm_velocity/htm_velocity_classify_file.py --model \"$MODEL\" --file \"$FILE\""
# Uncomment to add ground-truth:
# CMD="$CMD --true_label $TRUE_LABEL"

eval $CMD
