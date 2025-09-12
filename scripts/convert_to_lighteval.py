#!/usr/bin/env python3
"""
Convert results.json to LightEval schema format.

Usage:
    python convert_to_lighteval.py <results_json_path> <model_name>

Example:
    python convert_to_lighteval.py eval_results/Qwen/Qwen3-4B-Instruct-2507/main/mcp_bench/baseline-o4-mini-judge/results.json Qwen3-4B-Instruct-2507
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path


def create_metric_config(metric_name, higher_is_better=True):
    """Create a metric configuration dictionary."""
    return {
        "metric_name": metric_name,
        "higher_is_better": higher_is_better,
        "category": "AGGREGATE",
        "sample_level_fn": "compute",
        "corpus_level_fn": "mean",
        "batched_compute": False
    }


def extract_metrics_from_server_data(server_data):
    """Extract all metric names from a single server's data."""
    metrics = set()
    
    def collect_metrics(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if prefix:
                        metric_name = f"{prefix}_{key}"
                    else:
                        metric_name = key
                    metrics.add(metric_name)
                elif isinstance(value, dict):
                    if prefix:
                        new_prefix = f"{prefix}_{key}"
                    else:
                        new_prefix = key
                    collect_metrics(value, new_prefix)
    
    collect_metrics(server_data)
    return sorted(metrics)


def flatten_server_results(server_data, prefix=""):
    """Flatten nested server results dictionary."""
    flattened = {}
    
    for key, value in server_data.items():
        if isinstance(value, (int, float)):
            if prefix:
                metric_name = f"{prefix}_{key}"
            else:
                metric_name = key
            flattened[metric_name] = value
        elif isinstance(value, dict):
            if prefix:
                new_prefix = f"{prefix}_{key}"
            else:
                new_prefix = key
            flattened.update(flatten_server_results(value, new_prefix))
    
    return flattened


def convert_to_lighteval(results_path, model_name, revision="main"):
    """Convert results.json to LightEval schema format."""
    
    # Read input results
    with open(results_path, 'r') as f:
        input_results = json.load(f)
    
    # Define server type mappings
    server_mappings = {
        "single_server": "single_server",
        "double_server": "double_server", 
        "triple_server": "triple_server",
        "multi_server_aggregate": "multi_server",
        "overall_weighted_average": "overall_server"
    }
    
    # Initialize LightEval schema structure
    lighteval_schema = {
        "config_general": {
            "lighteval_sha": "?",
            "num_fewshot_seeds": 1,
            "max_samples": None,
            "job_id": "0",
            "start_time": None,
            "end_time": None,
            "total_evaluation_time_secondes": "0",
            "model_config": {
                "generation_parameters": {
                    "num_blocks": None,
                    "block_size": None,
                    "early_stopping": None,
                    "repetition_penalty": None,
                    "frequency_penalty": None,
                    "length_penalty": None,
                    "presence_penalty": None,
                    "max_new_tokens": None,
                    "min_new_tokens": None,
                    "seed": None,
                    "stop_tokens": None,
                    "temperature": None,
                    "top_k": None,
                    "min_p": None,
                    "top_p": None,
                    "truncate_prompt": None,
                    "cache_implementation": None,
                    "response_format": None
                },
                "system_prompt": "",
                "model_name": model_name,
                "revision": revision,
                "dtype": None,
                "tensor_parallel_size": None,
                "data_parallel_size": None,
                "pipeline_parallel_size": None,
                "gpu_memory_utilization": None,
                "max_model_length": None,
                "quantization": None,
                "load_format": None,
                "swap_space": None,
                "seed": None,
                "trust_remote_code": False,
                "add_special_tokens": True,
                "multichoice_continuations_start_space": True,
                "pairwise_tokenization": False,
                "max_num_seqs": None,
                "max_num_batched_tokens": None,
                "subfolder": None,
                "is_async": False
            },
            "model_name": model_name
        },
        "results": {},
        "versions": {},
        "config_tasks": {},
        "summary_tasks": {},
        "summary_general": {
            "hashes": {
                "hash_examples": "30ca62dbb5a17a96",
                "hash_full_prompts": "b88291a61f7e5545",
                "hash_input_tokens": "19b88cd943f61351",
                "hash_cont_tokens": "59d94bf1229a004e"
            },
            "truncated": 0,
            "non_truncated": 0,
            "padded": 0,
            "non_padded": 0,
            "num_truncated_few_shots": 0
        }
    }
    
    # Process each server type
    for input_key, server_type in server_mappings.items():
        if input_key not in input_results:
            continue
            
        server_data = input_results[input_key]
        task_key = f"external|mcp_bench:{server_type}|0"
        
        # Flatten server results
        flattened_results = flatten_server_results(server_data)
        
        # Extract metrics for this server type
        metrics = extract_metrics_from_server_data(server_data)
        
        # Create metrics configuration
        metrics_config = []
        for metric_name in metrics:
            # Determine if higher is better based on common patterns
            higher_is_better = not any(pattern in metric_name.lower() for pattern in [
                'cost', 'latency', 'time', 'error', 'loss'
            ])
            metrics_config.append(create_metric_config(metric_name, higher_is_better))
        
        # Add to results
        lighteval_schema["results"][task_key] = flattened_results
        
        # Add task configuration
        lighteval_schema["config_tasks"][task_key] = {
            "name": f"mcp_bench:{server_type}",
            "prompt_function": None,
            "hf_repo": None,
            "hf_subset": None,
            "metrics": metrics_config,
            "hf_revision": None,
            "hf_filter": None,
            "hf_avail_splits": ["overall"],
            "trust_dataset": True,
            "evaluation_splits": ["overall"],
            "few_shots_split": None,
            "few_shots_select": None,
            "generation_size": None,
            "generation_grammar": None,
            "stop_sequence": [],
            "num_samples": None,
            "suite": ["external"],
            "original_num_docs": -1,
            "effective_num_docs": -1,
            "must_remove_duplicate_docs": False,
            "num_fewshots": 0,
            "truncate_fewshots": False,
            "version": 1
        }
        
        # Add summary task
        lighteval_schema["summary_tasks"][task_key] = {
            "hashes": {
                "hash_examples": "10d354e1d33903b8",
                "hash_full_prompts": "4022df94d161ae1d",
                "hash_input_tokens": "2f0974d8416bc585",
                "hash_cont_tokens": "0b5030ccc31ddf31"
            },
            "truncated": 0,
            "non_truncated": 0,
            "padded": 0,
            "non_padded": 0,
            "effective_few_shots": 0,
            "num_truncated_few_shots": 0
        }
    
    # Add "all" aggregate if we have any results
    if lighteval_schema["results"]:
        # Combine all metrics from all server types
        all_metrics = {}
        for task_results in lighteval_schema["results"].values():
            all_metrics.update(task_results)
        lighteval_schema["results"]["all"] = all_metrics
    
    return lighteval_schema


def main():
    parser = argparse.ArgumentParser(description="Convert results.json to LightEval schema format")
    parser.add_argument("results_path", help="Path to the results.json file")
    parser.add_argument("model_name", help="Model name (e.g., Qwen3-4B-Instruct-2507)")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--revision", "-r", default="main", help="Model revision (default: main)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.results_path).exists():
        print(f"Error: Results file not found: {args.results_path}")
        sys.exit(1)
    
    # Convert to LightEval format
    try:
        lighteval_data = convert_to_lighteval(args.results_path, args.model_name, args.revision)
        
        # Generate output filename with timestamp
        if args.output:
            output_path = args.output
        else:
            # Save in same directory as input file
            input_dir = Path(args.results_path).parent
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            output_path = input_dir / f"results_{timestamp}.json"
        
        # Write output
        with open(output_path, 'w') as f:
            json.dump(lighteval_data, f, indent=2)
        
        print(f"Converted results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error converting results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()