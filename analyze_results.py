#!/usr/bin/env python3

import json
import glob
import os
from pathlib import Path


def analyze_results_directory(directory_path, prefix_filter=None):
    """
    Analyze all JSON result files in a directory and compute scores.
    
    Args:
        directory_path (str): Path to directory containing results files
        prefix_filter (str): Optional prefix to filter files (e.g., 'single_server', 'double_server', 'triple_server')
        
    Returns:
        dict: Computed scores for the directory
    """
    # Find all JSON files in the directory
    if prefix_filter:
        json_files = glob.glob(os.path.join(directory_path, f"results_{prefix_filter}_*.json"))
    else:
        json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return None
    
    all_results = []
    
    # Load all JSON files
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    if not all_results:
        print(f"No valid JSON files found in {directory_path}")
        return None
    
    # Compute averages across all files
    metrics = {}
    
    # Fields we need for schema_understanding_score
    schema_fields = ['input_schema_compliance', 'valid_tool_name_rate', 'tool_call_success_rate']
    
    # All score fields that need normalization
    score_fields = ['task_completion_score', 'tool_selection_score', 'planning_effectiveness_and_efficiency_score']
    
    # Collect all metrics
    all_fields = schema_fields + score_fields
    
    for field in all_fields:
        values = []
        for result in all_results:
            if field in result:
                values.append(result[field])
        
        if values:
            metrics[field] = sum(values) / len(values)
        else:
            print(f"Warning: {field} not found in any results")
            metrics[field] = 0.0
    
    # Compute schema_understanding_score
    schema_understanding_score = (
        metrics['input_schema_compliance'] + 
        metrics['valid_tool_name_rate'] + 
        metrics['tool_call_success_rate']
    ) / 3
    
    # Normalize score fields by dividing by 10
    normalized_scores = {}
    for field in score_fields:
        normalized_scores[f"{field}"] = metrics[field] / 10
    
    # Add schema_understanding_score (already normalized as it's based on rates)
    normalized_scores['schema_understanding_score'] = schema_understanding_score
    
    # Compute overall_score by averaging all 4 normalized score fields
    overall_score = (
        normalized_scores['task_completion_score'] + 
        normalized_scores['tool_selection_score'] + 
        normalized_scores['planning_effectiveness_and_efficiency_score'] + 
        normalized_scores['schema_understanding_score']
    ) / 4
    
    normalized_scores['overall_score'] = overall_score
    
    return {
        'directory': directory_path,
        'prefix_filter': prefix_filter,
        'num_files_processed': len(all_results),
        'raw_metrics': metrics,
        'normalized_scores': normalized_scores
    }


def main():
    """
    Main function to process directories.
    Can be called with a specific directory or will process common patterns.
    """
    import sys
    
    if len(sys.argv) > 1:
        # Process specific directory provided as argument
        directory = sys.argv[1]
        if os.path.exists(directory):
            # Analyze each server type separately
            server_types = ['single_server', 'double_server', 'triple_server']
            results_by_type = {}
            
            for server_type in server_types:
                result = analyze_results_directory(directory, server_type)
                if result:
                    results_by_type[server_type] = result
                    print(f"\nResults for {result['directory']} ({server_type}):")
                    print(f"Files processed: {result['num_files_processed']}")
                    print("Normalized Scores:")
                    for key, value in result['normalized_scores'].items():
                        print(f"  {key}: {value:.4f}")
                    print("-" * 60)
            
            # Compute multi-server aggregate scores
            multi_server_scores = None
            if 'double_server' in results_by_type and 'triple_server' in results_by_type:
                double_scores = results_by_type['double_server']['normalized_scores']
                triple_scores = results_by_type['triple_server']['normalized_scores']
                
                multi_server_scores = {}
                for key in double_scores:
                    multi_server_scores[key] = (30/48 * double_scores[key] + 18/48 * triple_scores[key])
                
                print(f"\nMulti-server aggregate scores for {directory}:")
                print("(30/48 * double_server + 18/48 * triple_server)")
                print("Normalized Scores:")
                for key, value in multi_server_scores.items():
                    print(f"  {key}: {value:.4f}")
                print("-" * 60)
            
            # Compute overall weighted average
            if 'single_server' in results_by_type and multi_server_scores is not None:
                single_scores = results_by_type['single_server']['normalized_scores']
                
                overall_weighted_scores = {}
                for key in single_scores:
                    overall_weighted_scores[key] = (56/104 * single_scores[key] + 48/104 * multi_server_scores[key])
                
                print(f"\nOverall weighted average for {directory}:")
                print("(56/104 * single_server + 48/104 * multi_server)")
                print("Normalized Scores:")
                for key, value in overall_weighted_scores.items():
                    print(f"  {key}: {value:.4f}")
                print("-" * 60)
        else:
            print(f"Directory not found: {directory}")
    else:
        # Process all directories matching the pattern
        pattern = "eval_results/*/*/*/mcp_bench/*"
        directories = glob.glob(pattern)
        
        if not directories:
            print("No directories found matching pattern: eval_results/*/*/*/mcp_bench/*")
            return
        
        for directory in sorted(directories):
            if os.path.isdir(directory):
                # Analyze each server type separately
                server_types = ['single_server', 'double_server', 'triple_server']
                results_by_type = {}
                
                for server_type in server_types:
                    result = analyze_results_directory(directory, server_type)
                    if result:
                        results_by_type[server_type] = result
                        print(f"\nResults for {result['directory']} ({server_type}):")
                        print(f"Files processed: {result['num_files_processed']}")
                        print("Normalized Scores:")
                        for key, value in result['normalized_scores'].items():
                            print(f"  {key}: {value:.4f}")
                        print("-" * 40)
                
                # Compute multi-server aggregate scores
                multi_server_scores = None
                if 'double_server' in results_by_type and 'triple_server' in results_by_type:
                    double_scores = results_by_type['double_server']['normalized_scores']
                    triple_scores = results_by_type['triple_server']['normalized_scores']
                    
                    multi_server_scores = {}
                    for key in double_scores:
                        multi_server_scores[key] = (30/48 * double_scores[key] + 18/48 * triple_scores[key])
                    
                    print(f"\nMulti-server aggregate scores for {directory}:")
                    print("(30/48 * double_server + 18/48 * triple_server)")
                    print("Normalized Scores:")
                    for key, value in multi_server_scores.items():
                        print(f"  {key}: {value:.4f}")
                    print("-" * 40)
                
                # Compute overall weighted average
                if 'single_server' in results_by_type and multi_server_scores is not None:
                    single_scores = results_by_type['single_server']['normalized_scores']
                    
                    overall_weighted_scores = {}
                    for key in single_scores:
                        overall_weighted_scores[key] = (56/104 * single_scores[key] + 48/104 * multi_server_scores[key])
                    
                    print(f"\nOverall weighted average for {directory}:")
                    print("(56/104 * single_server + 48/104 * multi_server)")
                    print("Normalized Scores:")
                    for key, value in overall_weighted_scores.items():
                        print(f"  {key}: {value:.4f}")
                    print("-" * 60)


if __name__ == "__main__":
    main()