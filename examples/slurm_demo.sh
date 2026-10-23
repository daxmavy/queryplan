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
#   smoke       a tiny planned run (10 items, 2 experiments) that stops the job early if anything is broken
#   naive       each experiment start to finish on one GPU at a time, the way a plain script runs
#   plan        planned, starting with no cost model (a pilot measures the models first)
#   plan-warm   planned, with the cost model the earlier steps learned
#   first       plan-warm with one experiment (FIRST) in its own top-priority tier
#
#   sbatch examples/slurm_demo.sh                                            # all steps, full size
#   SEQUENCE=smoke sbatch --gpus=2 --time=00:45:00 examples/slurm_demo.sh                  # smoke test only
#
# SITE_ENV names a file sourced first, for whatever the cluster needs (a virtualenv with vLLM, a model cache, ...).
# EXTRA_ARGS are added to every step (e.g. "--local-snapshots /path/to/hub"). Without Slurm the script runs the same
# way on whatever GPUs CUDA_VISIBLE_DEVICES names.
set -euo pipefail
if [ -n "${SITE_ENV:-}" ]; then source "$SITE_ENV"; fi
PYTHON=${PYTHON:-python}
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}
OUT=${OUT:-runs/slurm_${SLURM_JOB_ID:-local}}
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
COMMON=(--units "${UNITS:-500}" --experiments "${EXPS:-6}" --hw "${HW:-GPU}" --mem "${MEM:-80}" ${EXTRA[@]+"${EXTRA[@]}"})
mkdir -p "$OUT" logs
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv > "$OUT/gpus.csv" || true
n=0
for step in ${SEQUENCE:-smoke naive plan naive plan-warm first}; do
    n=$((n + 1))
    dir="$OUT/${n}_$step"
    echo "=== $(date +%FT%T) $step -> $dir"
    case "$step" in
    smoke) "$PYTHON" examples/judge_pipeline.py --policy plan --store "$dir" --units 10 --experiments 2 \
        --hw "${HW:-GPU}" --mem "${MEM:-80}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
    naive) "$PYTHON" examples/judge_pipeline.py --policy naive --store "$dir" "${COMMON[@]}" ;;
    plan) "$PYTHON" examples/judge_pipeline.py --policy plan --store "$dir" --cost-model "$dir/cost.json" "${COMMON[@]}"
        cp "$dir/cost.json" "$OUT/cost.json" ;;
    plan-warm) "$PYTHON" examples/judge_pipeline.py --policy plan --store "$dir" --cost-model "$OUT/cost.json" "${COMMON[@]}" ;;
    first) "$PYTHON" examples/judge_pipeline.py --policy plan --store "$dir" --cost-model "$OUT/cost.json" \
        --first "${FIRST:-gemma-urgent}" "${COMMON[@]}" ;;
    *) echo "unknown step $step" >&2; exit 1 ;;
    esac
done
echo "=== $(date +%FT%T) all steps done"
