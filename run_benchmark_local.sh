#!/bin/bash

# MCP-Bench Local Evaluation Script with Separate Judge Model
#
# This script runs MCP-Bench evaluation with separate agent and judge models served by vLLM.
# The agent model runs on GPUs 0,1,2,3 and the judge model runs on GPUs 4,5,6,7.
#
# Usage Examples:
#   # Run with default values
#   ./run_benchmark_local.sh
#
#   # Run with custom models
#   ./run_benchmark_local.sh --model "meta-llama/Llama-3.1-8B-Instruct" --judge "meta-llama/Llama-3.1-70B-Instruct"
#
#   # Run with custom tasks file
#   ./run_benchmark_local.sh --tasks_file "tasks/single_server_0.json"
#
# Default Values:
#   --model: Qwen/Qwen3-4B-Instruct-2507
#   --revision: main
#   --judge: Qwen/Qwen3-4B-Thinking-2507  
#   --tasks_file: tasks/test.json

# Needed for vLLM / LiteLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1

# Set up environment variables for MCP-Bench
export HUGGINGFACE_BASE_URL="http://localhost:8000/v1"
export HUGGINGFACE_JUDGE_BASE_URL="http://localhost:8001/v1"

# Record start time for timing the evaluation
SCRIPT_START_TIME=$(date +%s)
echo "Evaluation started at: $(date)"

# Server PIDs for cleanup
AGENT_LLM_PID=""
JUDGE_LLM_PID=""

# Function to shut down both servers
function shutdown_servers {
    echo "Shutting down servers..."
    if [[ -n "$AGENT_LLM_PID" && "$AGENT_LLM_PID" != "0" ]]; then
        echo "Stopping agent LLM server (PID: $AGENT_LLM_PID)..."
        kill $AGENT_LLM_PID 2>/dev/null || true
    fi
    if [[ -n "$JUDGE_LLM_PID" && "$JUDGE_LLM_PID" != "0" ]]; then
        echo "Stopping judge LLM server (PID: $JUDGE_LLM_PID)..."
        kill $JUDGE_LLM_PID 2>/dev/null || true
    fi
    if [[ -n "$AGENT_LLM_PID" ]]; then
        wait $AGENT_LLM_PID 2>/dev/null || true
    fi
    if [[ -n "$JUDGE_LLM_PID" ]]; then
        wait $JUDGE_LLM_PID 2>/dev/null || true
    fi
    echo "Servers shut down."
    exit 0
}

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

function get_reasoning_parser {
    local model_id="$1"
    case "$model_id" in
        *deepseek*|*DeepSeek*) echo "deepseek_r1" ;;
        *Qwen3*Thinking*|*qwen3*thinking*) echo "deepseek_r1" ;;
        *gpt-oss*|*GPT-OSS*) echo "GptOss" ;;
        *glm-4.5*|*GLM-4.5*) echo "glm45" ;;
        *hunyuan*|*Hunyuan*) echo "hunyuan_a13b" ;;
        *granite*|*Granite*) echo "granite" ;;
        *mistral*|*Mistral*) echo "mistral" ;;
        *step3*|*Step3*) echo "step3" ;;
        *qwen3*|*Qwen3*) echo "qwen3" ;;
        *smollm3*|*SmolLM3*) echo "qwen3" ;;
        *)
            echo "ERROR: No reasoning parser found for model: $model_id" >&2
            echo "Supported patterns: deepseek, gpt-oss, glm-4.5, hunyuan, granite, mistral, step3, qwen3" >&2
            exit 1 ;;
    esac
}

# Default values
AGENT_MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION="main"
JUDGE_MODEL_ID="Qwen/Qwen3-4B-Thinking-2507"
TASKS_FILE="tasks/test.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            AGENT_MODEL_ID="$2"
            shift 2
            ;;
        --revision)
            MODEL_REVISION="$2"
            shift 2
            ;;
        --judge)
            JUDGE_MODEL_ID="$2"
            shift 2
            ;;
        --tasks_file)
            TASKS_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--model <model_id>] [--revision <revision>] [--judge <judge_model_id>] [--tasks_file <tasks_file>]"
            echo "Defaults: --model Qwen/Qwen3-4B-Instruct-2507 --revision main --judge Qwen/Qwen3-4B-Thinking-2507 --tasks_file tasks/test.json"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
OUTPUT_DIR="eval_results/$AGENT_MODEL_ID/$MODEL_REVISION/mcp_bench/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# Set environment variables with model names
export HUGGINGFACE_MODEL="$AGENT_MODEL_ID"
export HUGGINGFACE_JUDGE_MODEL="$JUDGE_MODEL_ID"

# Determine appropriate tool call parser for agent model
AGENT_TOOL_PARSER=$(get_tool_parser "$AGENT_MODEL_ID")
echo "Using tool call parser: $AGENT_TOOL_PARSER for agent model: $AGENT_MODEL_ID"

# Determine reasoning parser for agent model
AGENT_REASONING_PARSER=$(get_reasoning_parser "$AGENT_MODEL_ID")
echo "Using reasoning parser: $AGENT_REASONING_PARSER for agent model: $AGENT_MODEL_ID"

# Determine appropriate tool call parser for judge model  
JUDGE_TOOL_PARSER=$(get_tool_parser "$JUDGE_MODEL_ID")
echo "Using tool call parser: $JUDGE_TOOL_PARSER for judge model: $JUDGE_MODEL_ID"

# Determine reasoning parser for judge model
JUDGE_REASONING_PARSER=$(get_reasoning_parser "$JUDGE_MODEL_ID")
echo "Using reasoning parser: $JUDGE_REASONING_PARSER for judge model: $JUDGE_MODEL_ID"

# Trap interruption signals and call the shutdown function
trap shutdown_servers SIGINT SIGTERM

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
    local max_attempts=360 # 30 minutes total (360 * 5 seconds)

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

# Start agent model on first 4 GPUs (0,1,2,3) on port 8000
echo "Starting agent vLLM server with 4 GPUs (0,1,2,3) on port 8000..."
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve $AGENT_MODEL_ID --revision $MODEL_REVISION \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser $AGENT_TOOL_PARSER \
    --reasoning-parser $AGENT_REASONING_PARSER \
    --host 0.0.0.0 --port 8000 \
    >"$OUTPUT_DIR/llm_agent_8000.log" 2>&1 &

AGENT_LLM_PID=$!

# Start judge model on last 4 GPUs (4,5,6,7) on port 8001  
echo "Starting judge vLLM server with 4 GPUs (4,5,6,7) on port 8001..."
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve $JUDGE_MODEL_ID --revision $MODEL_REVISION \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser $JUDGE_TOOL_PARSER \
    --reasoning-parser $JUDGE_REASONING_PARSER \
    --no-enable-chunked-prefill \
    --host 0.0.0.0 --port 8001 \
    >"$OUTPUT_DIR/llm_judge_8001.log" 2>&1 &

JUDGE_LLM_PID=$!

wait_for_server 8000 "Agent"
wait_for_server 8001 "Judge"

echo "Running MCP-Bench evaluation..."
echo "Results will be saved to $OUTPUT_DIR"
python run_benchmark.py \
    --models huggingface \
    --judge-provider huggingface-judge \
    --tasks-file "$TASKS_FILE" \
    --distraction-count 0 --disable-judge-stability \
    --output "$OUTPUT_DIR/results.json" \
    --enable-cache \
    --cache-dir "$OUTPUT_DIR/cache"

# Calculate and display total runtime
SCRIPT_END_TIME=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))

echo "Evaluation completed at: $(date)"
echo "Total runtime: ${HOURS} hours and ${MINUTES} minutes \(${TOTAL_SECONDS} seconds\)"

echo "Done!"

shutdown_servers
