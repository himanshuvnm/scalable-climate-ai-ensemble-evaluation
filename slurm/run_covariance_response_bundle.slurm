#!/bin/bash
#SBATCH --job-name=cov_response
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=<cpu_partition>
#SBATCH --output=logs/cov_response_%j.out
#SBATCH --error=logs/cov_response_%j.err

set -euo pipefail

module purge
module load python/3.12

source /path/to/venv/bin/activate

export PROJECT_ROOT=/path/to/scalable-climate-ai-ensemble-evaluation
export DATA_ROOT=/path/to/gefs_or_geos_data
export OUTDIR=$PROJECT_ROOT/outputs/covariance_response

mkdir -p logs "$OUTDIR"

python "$PROJECT_ROOT/scripts/make_covariance_response_bundle.py" \
  --config "$PROJECT_ROOT/configs/gefs_hgt500_example.yaml" \
  --data-root "$DATA_ROOT" \
  --query-index 60 \
  --probe-id 1 \
  --outdir "$OUTDIR"
