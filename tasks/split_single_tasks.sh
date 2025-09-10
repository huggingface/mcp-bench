#!/bin/bash

# Script to split mcpbench_tasks_single_runner_format.json into N groups
# Usage: ./split_single_tasks.sh [--num-groups N]
# Default: N=4 (each file contains 7 server tasks, 14 total tasks)

set -e  # Exit on any error

# Default values
NUM_GROUPS=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-groups)
            NUM_GROUPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--num-groups N]"
            echo "  --num-groups N    Number of groups to split into (default: 4)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate num_groups is a positive integer
if ! [[ "$NUM_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-groups must be a positive integer, got: $NUM_GROUPS"
    exit 1
fi

INPUT_FILE="tasks/mcpbench_tasks_single_runner_format.json"
SCRATCH_DIR="/scratch"

echo "Starting decomposition of $INPUT_FILE into $NUM_GROUPS groups..."

# Create scratch directory
mkdir -p "$SCRATCH_DIR"

# Extract generation_info for reuse
echo "Extracting generation_info..."
jq '.generation_info' "$INPUT_FILE" > "$SCRATCH_DIR/generation_info.json"

# Calculate total number of server tasks and tasks per group
TOTAL_SERVERS=$(jq '.server_tasks | length' "$INPUT_FILE")
SERVERS_PER_GROUP=$((TOTAL_SERVERS / NUM_GROUPS))
REMAINDER=$((TOTAL_SERVERS % NUM_GROUPS))

echo "Total server tasks: $TOTAL_SERVERS"
echo "Splitting into $NUM_GROUPS groups (~$SERVERS_PER_GROUP servers per group)..."

# Split server_tasks into groups
for ((i=0; i<NUM_GROUPS; i++)); do
    start=$((i * SERVERS_PER_GROUP))
    # Distribute remainder among first groups
    if [ $i -lt $REMAINDER ]; then
        start=$((start + i))
        end=$((start + SERVERS_PER_GROUP + 1))
    else
        start=$((start + REMAINDER))
        end=$((start + SERVERS_PER_GROUP))
    fi
    
    echo "Group $i: servers $start to $((end-1))"
    jq ".server_tasks[$start:$end]" "$INPUT_FILE" > "$SCRATCH_DIR/group_$i.json"
done

# Create the output files
echo "Creating output files..."
for ((i=0; i<NUM_GROUPS; i++)); do
    echo "Creating single_server_$i.json..."
    jq -s '{"generation_info": .[0], "server_tasks": .[1]}' \
        "$SCRATCH_DIR/generation_info.json" \
        "$SCRATCH_DIR/group_$i.json" > "tasks/single_server_$i.json"
done

echo "Decomposition complete. Files created:"
ls -la tasks/single_server_*.json

# Verification
echo ""
echo "=== VERIFICATION ==="

# Check file counts
echo "Verifying file structure..."
for ((i=0; i<NUM_GROUPS; i++)); do
    server_count=$(jq '.server_tasks | length' "tasks/single_server_$i.json")
    task_count=$(jq '[.server_tasks[].tasks | length] | add' "tasks/single_server_$i.json")
    echo "File $i: $server_count server tasks, $task_count total tasks"
done

# Verify all task IDs are preserved
echo ""
echo "Verifying task integrity..."
jq -r '.server_tasks[].tasks[].task_id' "$INPUT_FILE" | sort > "$SCRATCH_DIR/original_task_ids.txt"
cat tasks/single_server_*.json | jq -r '.server_tasks[].tasks[].task_id' | sort > "$SCRATCH_DIR/split_task_ids.txt"

original_count=$(wc -l < "$SCRATCH_DIR/original_task_ids.txt")
split_count=$(wc -l < "$SCRATCH_DIR/split_task_ids.txt")

echo "Original file: $original_count task IDs"
echo "Split files: $split_count task IDs"

if cmp -s "$SCRATCH_DIR/original_task_ids.txt" "$SCRATCH_DIR/split_task_ids.txt"; then
    echo "✅ VERIFICATION PASSED: All tasks preserved correctly"
else
    echo "❌ VERIFICATION FAILED: Task mismatch detected"
    exit 1
fi

# Clean up scratch files
rm -f "$SCRATCH_DIR"/{generation_info.json,group_*.json,original_task_ids.txt,split_task_ids.txt}

echo ""
echo "Split complete! Files tasks/single_server_0.json through tasks/single_server_$((NUM_GROUPS-1)).json created successfully."