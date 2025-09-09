#!/bin/bash

# Script to split mcpbench_tasks_single_runner_format.json into 4 files
# Each file contains 7 server tasks (14 total tasks)

set -e  # Exit on any error

INPUT_FILE="tasks/mcpbench_tasks_single_runner_format.json"
SCRATCH_DIR="/scratch"

echo "Starting decomposition of $INPUT_FILE..."

# Create scratch directory
mkdir -p "$SCRATCH_DIR"

# Extract generation_info for reuse
echo "Extracting generation_info..."
jq '.generation_info' "$INPUT_FILE" > "$SCRATCH_DIR/generation_info.json"

# Split server_tasks into 4 groups of 7 tasks each
echo "Splitting server_tasks into 4 groups..."
jq '.server_tasks[0:7]' "$INPUT_FILE" > "$SCRATCH_DIR/group_0.json"
jq '.server_tasks[7:14]' "$INPUT_FILE" > "$SCRATCH_DIR/group_1.json"
jq '.server_tasks[14:21]' "$INPUT_FILE" > "$SCRATCH_DIR/group_2.json"
jq '.server_tasks[21:28]' "$INPUT_FILE" > "$SCRATCH_DIR/group_3.json"

# Create the 4 output files
echo "Creating output files..."
for i in {0..3}; do
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
for i in {0..3}; do
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
echo "Split complete! Files tasks/single_server_0.json through tasks/single_server_3.json created successfully."