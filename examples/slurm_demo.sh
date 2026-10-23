#!/bin/bash
#SBATCH --job-name=queryplan-demo
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=03:00:00
#
# The demo inside one Slurm job. The driver, its model servers and the planner all live in the allocation, so nothing
# outside the job is needed while it runs. Runs the steps in SEQUENCE one after another on the same GPUs:
#
#   naive       each experiment start to finish on one GPU at a time, the way a plain script runs
#   plan        planned, starting with no cost model (a pilot measures the models first)
#   plan-warm   planned, with the cost model the earlier steps learned
#   first       plan-warm with one experiment (FIRST) in its own top-priority tier
#
#   sbatch examples/slurm_demo.sh                                            # all steps, full size
#   SEQUENCE=plan UNITS=10 EXPS=2 sbatch --gpus=2 --time=00:45:00 examples/slurm_demo.sh   # smoke test
#
# SITE_ENV names a file sourced first, for whatever the cluster needs (a virtualenv with vLLM, a model cache, ...).
set -euo pipefail
if [ -n "${SITE_ENV:-}" ]; then source "$SITE_ENV"; fi
PYTHON=${PYTHON:-python}
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}
OUT=${OUT:-runs/slurm_${SLURM_JOB_ID:-local}}
COMMON=(--units "${UNITS:-500}" --experiments "${EXPS:-6}" --hw "${HW:-GPU}" --mem "${MEM:-80}")
mkdir -p "$OUT" logs
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv > "$OUT/gpus.csv" || true
for step in ${SEQUENCE:-naive plan naive plan-warm first}; do
    n=$(ls -d "$OUT"/[0-9]*_* 2>/dev/null | wc -l)
    dir="$OUT/$((n + 1))_$step"
    echo "=== $(date -Is) $step -> $dir"
    case "$step" in
    naive) "$PYTHON" examples/judge_pipeline.py --policy naive --store "$dir" "${COMMON[@]}" ;;
    plan) "$PYTHON" examples/judge_pipeline.py --policy plan --store "$dir" --cost-model "$dir/cost.json" "${COMMON[@]}"
        cp "$dir/cost.json" "$OUT/cost.json" ;;
    plan-warm) "$PYTHON" examples/judge_pipeline.py --policy plan --store "$dir" --cost-model "$OUT/cost.json" "${COMMON[@]}" ;;
    first) "$PYTHON" examples/judge_pipeline.py --policy plan --store "$dir" --cost-model "$OUT/cost.json" \
        --first "${FIRST:-gemma-urgent}" "${COMMON[@]}" ;;
    *) echo "unknown step $step" >&2; exit 1 ;;
    esac
done
echo "=== $(date -Is) all steps done"
