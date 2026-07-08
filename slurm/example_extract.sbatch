#!/bin/bash
# Generic SLURM template for a GPU extraction job.
# Fill in your own account/partition; this repo ships no cluster-specific values.
#SBATCH --job-name=fm_extract
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=fm_extract_%j.out

set -uo pipefail
export FMTRANSFER_ROOT="${FMTRANSFER_ROOT:?set FMTRANSFER_ROOT to your data dir}"
export PYTHONNOUSERSITE=1

# Activate the environment appropriate to the model you are extracting, e.g.:
#   source env_scgpt/bin/activate
#   python scripts/extract_scgpt.py norman
#   python scripts/extract_scgpt.py wessels

echo "done: $(date)"
