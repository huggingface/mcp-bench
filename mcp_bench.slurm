#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lewis+hfc@huggingface.co

# Usage: sbatch --qos=high --job-name=mcp-bench-qwen3-30b-a3b-concurrent --nodes=1 mcp_bench.slurm Qwen/Qwen3-30B-A3B-Instruct-2507 main

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.9

# Needed for vLLM / LiteLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export HUGGINGFACE_BASE_URL="http://localhost:8000/v1"

set -x -e

# Record start time for timing the evaluation
SCRIPT_START_TIME=$(date +%s)
echo "Evaluation started at: $(date)"

source ~/.bashrc
source mcpbench/bin/activate

# Server PIDs for cleanup
AGENT_LLM_PID=""
USER_LLM_PID=""

# Function to shut down both servers
function shutdown_servers {
  echo "Shutting down servers..."
  if [[ -n "$AGENT_LLM_PID" && "$AGENT_LLM_PID" != "0" ]]; then
    if kill -0 "$AGENT_LLM_PID" 2>/dev/null; then
      echo "Stopping agent LLM server (PID: $AGENT_LLM_PID)..."
      kill "$AGENT_LLM_PID" || true
      wait "$AGENT_LLM_PID" 2>/dev/null || true
    fi
  fi
  echo "Servers shut down."
}
trap shutdown_servers EXIT ERR SIGINT SIGTERM

# Function to determine tool call parser based on model ID
function get_tool_parser {
    local model_id="$1"
    case "$model_id" in
        *deepseek*|*DeepSeek*) echo "deepseek_v3" ;;
        *kimi*|*Kimi*) echo "kimi_k2" ;;
        *minimax*|*MiniMax*) echo "minimax_m1" ;;
        *hunyuan*|*Hunyuan*) echo "hunyuan_a13b" ;;
        *granite*|*Granite*) echo "granite" ;;
        *xlam*|*xLAM*) echo "xlam" ;;
        *jamba*|*Jamba*) echo "jamba" ;;
        *internlm*|*InternLM*) echo "internlm" ;;
        *mistral*|*Mistral*) echo "mistral" ;;
        *llama*|*Llama*) echo "llama3_json" ;;
        *qwen*|*Qwen*|*hermes*|*Hermes*|*nous-hermes*) echo "hermes" ;;
        *)
            echo "ERROR: No tool call parser found for model: $model_id" >&2
            echo "Supported patterns: deepseek, kimi, minimax, hunyuan, granite, xlam, jamba, internlm, mistral, llama, qwen, hermes" >&2
            exit 1 ;;
    esac
}

MODEL_ID=$1
MODEL_REVISION=$2
# Optional args
[ -z "$3" ] && TENSOR_PARALLEL=False || TENSOR_PARALLEL=$3
[ -z "$4" ] && TRUST_REMOTE_CODE=False || TRUST_REMOTE_CODE=$4
[ -z "${5:-}" ] && LM_EVAL_REPO_ID="HuggingFaceH4/smollm3-post-training-leaderboard" || LM_EVAL_REPO_ID=$5
[ -z "${6:-}" ] && DOMAIN="all" || DOMAIN=$6
# $7 is reserved for system_prompt

TASK_NAME="mcp_bench"

MODEL_NAME=$(echo $MODEL_ID | sed 's/\//__/g') # replaces / with __
# Extract details org from LM_EVAL_REPO_ID
IFS='/' read -r DETAILS_ORG DETAILS_REPO <<< "$LM_EVAL_REPO_ID"
# If no slash was provided, DETAILS_REPO will be empty.
if [ -z "$DETAILS_REPO" ]; then
    DETAILS_REPO="$DETAILS_ORG"
    DETAILS_ORG="HuggingFaceH4"
fi
DETAILS_REPO_ID="$DETAILS_ORG/details_$MODEL_NAME"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"
# We need this flag since we run this script from training jobs that use DeepSpeed and the env vars get progated which causes errors during evaluation
ACCELERATE_USE_DEEPSPEED=false

NUM_GPUS=$(nvidia-smi -L | wc -l)

# Determine appropriate tool call parser for this model
TOOL_PARSER=$(get_tool_parser "$MODEL_ID")
echo "Using tool call parser: $TOOL_PARSER for model: $MODEL_ID"

# Function to check if server is up by checking /health endpoint
function check_server {
    local port=$1
    curl -i http://0.0.0.0:$port/health 2>/dev/null | head -n 1 | grep "200 OK"
}

# Function to wait for server with timeout
function wait_for_server {
    local port=$1
    local name=$2
    echo "Waiting for $name server to start on port $port..."
    local attempt=0
    local max_attempts=180  # 15 minutes total (180 * 5 seconds)

    while ! check_server $port; do
        if [ $attempt -ge $max_attempts ]; then
            echo "ERROR: $name server failed to start after $((max_attempts * 5)) seconds"
            exit 1
        fi
        echo "$name server is not yet available. Checking again in 5 seconds... ($((attempt + 1))/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "$name server is up and running."
}

echo "Starting agent vLLM server on port 8000 with TP=$TP..."
nohup vllm serve $MODEL_ID --revision $MODEL_REVISION \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser $TOOL_PARSER \
    --host 0.0.0.0 --port 8000 \
    >"$OUTPUT_DIR/llm_agent_8000.log" 2>&1 &

AGENT_LLM_PID=$!

wait_for_server 8000 "Agent"

echo "Running mcp-bench evaluation ..."
echo "Trajectories results will be saved to $OUTPUT_DIR"

# Run all tasks
# python run_benchmark.py --models huggingface \
#     --judge-provider o4-mini \
#     --output $OUTPUT_DIR/results.json \
#     --cache-dir $OUTPUT_DIR/cache

# Store benchmark PIDs
BENCHMARK_PIDS=()

# Run single-server tasks
for i in 0 1 2 3; do
    echo "Running task set $i..."
    python run_benchmark.py --models huggingface \
        --judge-provider o4-mini \
        --tasks-file tasks/single_server_${i}.json \
        --output $OUTPUT_DIR/results_single_server_${i}.json \
        --cache-dir $OUTPUT_DIR/cache_single_server_${i} \
        >"$OUTPUT_DIR/single_server_${i}.log" 2>&1 &
    
    BENCHMARK_PIDS+=($!)  # Capture the PID of the background process
done

# Run double-server tasks
for i in 0 1; do
    echo "Running task set double $i..."
    python run_benchmark.py --models huggingface \
        --judge-provider o4-mini \
        --tasks-file tasks/double_server_${i}.json \
        --output $OUTPUT_DIR/results_double_server_${i}.json \
        --cache-dir $OUTPUT_DIR/cache_double_server_${i} \
        >"$OUTPUT_DIR/double_server_${i}.log" 2>&1 &

    BENCHMARK_PIDS+=($!)  # Capture the PID of the background process
done

# Run triple-server tasks
echo "Running task set triple..."
for i in 0 1; do
    python run_benchmark.py --models huggingface \
        --judge-provider o4-mini \
        --tasks-file tasks/triple_server_${i}.json \
        --output $OUTPUT_DIR/results_triple_server_${i}.json \
        --cache-dir $OUTPUT_DIR/cache_triple_server_${i} \
        >"$OUTPUT_DIR/triple_server_${i}.log" 2>&1 &

    BENCHMARK_PIDS+=($!)  # Capture the PID of the background process
done

# Wait only for benchmark processes to complete
for pid in "${BENCHMARK_PIDS[@]}"; do
    wait $pid
done

shutdown_servers

# # echo "Running post-processing ..."
# python dev/scripts/evaluation/tau2_to_lighteval.py $OUTPUT_DIR/simulations/ --revision $MODEL_REVISION -o $OUTPUT_DIR/results/

# OUTPUT_FILEPATHS=$(find $OUTPUT_DIR/results/ -type f -name "*result*.json")
# for filepath in $OUTPUT_FILEPATHS; do    
#     echo "Uploading $filepath to Hugging Face Hub..."
#     filename=$(basename -- "$filepath")
#     for attempt in {1..20}; do
#         if hf upload --repo-type space --private $LM_EVAL_REPO_ID $filepath $OUTPUT_DIR/$filename; then
#             echo "Upload succeeded for $filepath"
#             break
#         else
#             # Exponential backoff with max delay of 60 seconds
#             delay=$((2 ** (attempt - 1)))
#             if [ $delay -gt 60 ]; then
#                 delay=60
#             fi
#             echo "Upload failed for $filepath. Attempt $attempt of 20. Retrying in $delay seconds..."
#             sleep $delay
#         fi
#     done
# done

# # TODO: parse outputs into LightEval format
# echo "Uploading details to Hugging Face Hub..."
# hf upload --revision $MODEL_REVISION.$TASK_NAME.$TIMESTAMP --repo-type dataset --private $DETAILS_REPO_ID $OUTPUT_DIR/simulations/ data/

# echo "Cleaning up ..."
# rm -rf $OUTPUT_DIR

# Calculate and display total runtime
SCRIPT_END_TIME=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))

echo "Evaluation completed at: $(date)"
echo "Total runtime: ${HOURS} hours and ${MINUTES} minutes (${TOTAL_SECONDS} seconds)"

echo "Done!"