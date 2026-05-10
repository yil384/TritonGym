#!/bin/bash
# run_perlmutter.sh
# Perlmutter (shifter) runner: runs deepseek-chat on all three datasets (Standard + OOD + DSL)
# Usage:
#   export DEEPSEEK_API_KEY="sk-..."
#   shifter --image=docker:yil384/tritongym:latest --volume=$PWD:/workspace /bin/bash
#   cd /workspace && bash run_perlmutter.sh

set -euo pipefail

LOGDIR="./logs"
MODEL="deepseek-chat"
AGENTS=("oneshot" "alphaevolve" "geak")

# Three datasets corresponding to Table 1: Standard / OOD / DSL
BENCHMARK_JSONS=(
  "benchmark/data_v2/benchmark.json"
  "benchmark/data_ood_v2/benchmark.json"
  "benchmark/data_dsl_v2/benchmark.json"
)

# Evaluation parameters from the paper: warmup=5, repeat=10 (Section 4.1)
WARMUP=5
REPEAT=10

# Check API key
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: DEEPSEEK_API_KEY not set"
    exit 1
fi

# Collect all operators (deduplicated, recording which benchmark json each op belongs to)
declare -A OP_TO_JSON
total_ops=0
for jf in "${BENCHMARK_JSONS[@]}"; do
  if [ ! -f "$jf" ]; then
    echo "WARNING: $jf not found, skipping"
    continue
  fi
  ops=($(python -c "import json; print(' '.join(json.load(open('$jf')).keys()))"))
  for op in "${ops[@]}"; do
    OP_TO_JSON["$op"]="$jf"
    total_ops=$((total_ops + 1))
  done
done
echo "Found $total_ops operators across ${#BENCHMARK_JSONS[@]} datasets, agents: ${AGENTS[*]}, model: $MODEL"

for agent in "${AGENTS[@]}"; do
  (
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting $MODEL / $agent ..."
    for op in "${!OP_TO_JSON[@]}"; do
      jf="${OP_TO_JSON[$op]}"
      LOGFILE="$LOGDIR/${MODEL}/${agent}/${op}.json"
      if [ -s "$LOGFILE" ]; then
        echo "SKIP $agent/$op (log exists)"
        continue
      fi
      echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Running $op ($agent) [$(basename $(dirname $jf))] ..."
      python -m benchmark.cli --json \
        --model "$MODEL" \
        --agent "$agent" \
        --operator "$op" \
        --logs "$LOGDIR" \
        --benchmark-json "$jf" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT" || echo "FAILED: $agent/$op"
    done
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Done $MODEL / $agent"
  ) &
done

wait
echo "All done. Logs in $LOGDIR/$MODEL/"
