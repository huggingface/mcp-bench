#!/bin/bash

# Script to split mcpbench_tasks_multi_2server_runner_format.json into 2 files
# Each file contains 15 server tasks (30 total tasks)

set -e  # Exit on any error

INPUT_FILE="tasks/mcpbench_tasks_multi_2server_runner_format.json"
SCRATCH_DIR="/scratch"

echo "Starting decomposition of $INPUT_FILE..."

# Create scratch directory
mkdir -p "$SCRATCH_DIR"

# Extract generation_info for reuse
echo "Extracting generation_info..."
jq '.generation_info' "$INPUT_FILE" > "$SCRATCH_DIR/generation_info.json"

# Split server_tasks into 2 groups of 15 tasks each
echo "Splitting server_tasks into 2 groups..."
jq '.server_tasks[0:15]' "$INPUT_FILE" > "$SCRATCH_DIR/group_0.json"
jq '.server_tasks[15:30]' "$INPUT_FILE" > "$SCRATCH_DIR/group_1.json"

# Create the 2 output files
echo "Creating output files..."
for i in {0..1}; do
    echo "Creating double_server_$i.json..."
    jq -s '{"generation_info": .[0], "server_tasks": .[1]}' \
        "$SCRATCH_DIR/generation_info.json" \
        "$SCRATCH_DIR/group_$i.json" > "tasks/double_server_$i.json"
done

echo "Decomposition complete. Files created:"
ls -la tasks/double_server_*.json

# Verification
echo ""
echo "=== VERIFICATION ==="

# Check file counts
echo "Verifying file structure..."
for i in {0..1}; do
    server_count=$(jq '.server_tasks | length' "tasks/double_server_$i.json")
    task_count=$(jq '[.server_tasks[].tasks | length] | add' "tasks/double_server_$i.json")
    echo "File $i: $server_count server tasks, $task_count total tasks"
done

# Verify all task IDs are preserved
echo ""
echo "Verifying task integrity..."
jq -r '.server_tasks[].tasks[].task_id' "$INPUT_FILE" | sort > "$SCRATCH_DIR/original_task_ids.txt"
cat tasks/double_server_*.json | jq -r '.server_tasks[].tasks[].task_id' | sort > "$SCRATCH_DIR/split_task_ids.txt"

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
echo "Split complete! Files tasks/double_server_0.json and tasks/double_server_1.json created successfully."