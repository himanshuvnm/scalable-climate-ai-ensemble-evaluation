#!/bin/bash
#SBATCH --job-name=plot_response
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=<cpu_partition>
#SBATCH --output=logs/plot_response_%j.out
#SBATCH --error=logs/plot_response_%j.err

set -euo pipefail

module purge
module load python/3.12

source /path/to/venv/bin/activate

export PROJECT_ROOT=/path/to/scalable-climate-ai-ensemble-evaluation
export BUNDLE=$PROJECT_ROOT/outputs/covariance_response/example_response_bundle.npz
export OUTDIR=$PROJECT_ROOT/outputs/figures

mkdir -p logs "$OUTDIR"

python "$PROJECT_ROOT/scripts/plot_observation_centered_response.py" \
  --bundle "$BUNDLE" \
  --outdir "$OUTDIR"
