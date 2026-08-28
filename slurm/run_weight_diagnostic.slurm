#!/bin/bash
#SBATCH --job-name=weight_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=<cpu_partition>
#SBATCH --output=logs/weight_diag_%j.out
#SBATCH --error=logs/weight_diag_%j.err

set -euo pipefail

module purge
module load python/3.12

source /path/to/venv/bin/activate

export PROJECT_ROOT=/path/to/scalable-climate-ai-ensemble-evaluation
export META_FILE=/path/to/archive_metadata.json
export OUTDIR=$PROJECT_ROOT/outputs/weights

mkdir -p logs "$OUTDIR"

python "$PROJECT_ROOT/scripts/make_query_conditioned_weight_heatmap.py" \
  --meta "$META_FILE" \
  --ell auto \
  --ell-factor 0.25 \
  --exclude-neighbor 16 \
  --outdir "$OUTDIR" \
  --title-tag "Query-conditioned historical weights"
