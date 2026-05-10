#!/bin/bash
# run_all.sh
# Automatically reads operators from benchmark/data_v2/benchmark.json
# Each model x agent combination runs in parallel; operators run sequentially within each combination

# JSON file path
JSON_FILE="benchmark/data_v2/benchmark.json"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. Please run: sudo apt-get install jq -y"
    exit 1
fi

# Get all operators
OPERATORS=($(jq -r 'keys[]' $JSON_FILE))

# Define command templates for different models
declare -A MODELS
# Set API keys as environment variables before running, e.g.:
#   export ANTHROPIC_API_KEY="..."
#   export OPENAI_API_KEY="..."
#   export GOOGLE_API_KEY="..."
#   export TOGETHER_API_KEY="..."
#   export DEEPSEEK_API_KEY="..."
MODELS["claude-haiku-4-5-20251001"]="python -m benchmark.cli --json --model claude-haiku-4-5-20251001"
# MODELS["gpt-5"]="python -m benchmark.cli --json --model gpt-5"
# MODELS["gemini-2.5-pro"]="python -m benchmark.cli --json --model gemini-2.5-pro"
# MODELS["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"]="python -m benchmark.cli --json --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# MODELS["deepseek-chat"]="python -m benchmark.cli --json --model deepseek-chat"
# MODELS["Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"]="python -m benchmark.cli --json --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

# Define agents to test
AGENTS=("oneshot" "alphaevolve" "geak")

# Log directory
LOGDIR="./logs"
mkdir -p $LOGDIR

# Outer loop: each model-agent combination runs in a separate subprocess (parallel)
for agent in "${AGENTS[@]}"; do
  for model in "${!MODELS[@]}"; do
    (
      echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting all operators for $model with agent=$agent ..."
      for op in "${OPERATORS[@]}"; do
        LOGFILE="$LOGDIR/${model}/${agent}/${op}.json"

        if [ -s "$LOGFILE" ]; then
          echo "SKIP $model/$agent/$op, log already exists: $LOGFILE"
          continue
        fi

        echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Running $op on $model (agent=$agent) ..."
        eval "${MODELS[$model]} --operator $op --agent $agent --logs $LOGDIR --benchmark-json $JSON_FILE"
      done
      echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished all operators for $model with agent=$agent."
    ) &
  done
done

wait
echo "All model-agent combinations have completed. Logs are in $LOGDIR."
