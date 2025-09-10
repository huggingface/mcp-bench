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
    
    # Define domain structure with their component fields
    domain_structure = {
        'schema_understanding': ['valid_tool_name_rate', 'input_schema_compliance', 'tool_call_success_rate'],
        'task_completion': ['task_fulfillment', 'grounding'],
        'tool_usage': ['tool_appropriateness', 'parameter_accuracy'],
        'planning_effectiveness': ['dependency_awareness', 'parallelism_and_efficiency']
    }
    
    # Collect all metric fields
    all_fields = []
    for domain_fields in domain_structure.values():
        all_fields.extend(domain_fields)
    
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
    
    # Compute domain scores organized by domain structure
    domain_scores = {}
    
    for domain, fields in domain_structure.items():
        domain_data = {}
        
        # Add individual field scores
        for field in fields:
            if field in metrics:
                # Schema understanding fields are already rates (0-1), others need normalization
                if domain == 'schema_understanding':
                    domain_data[field] = metrics[field]
                else:
                    domain_data[field] = metrics[field] / 10  # Normalize by dividing by 10
            else:
                domain_data[field] = 0.0
        
        # Compute domain overall score as average of its fields
        field_values = [domain_data[field] for field in fields if field in domain_data]
        if field_values:
            domain_data['overall_score'] = sum(field_values) / len(field_values)
        else:
            domain_data['overall_score'] = 0.0
            
        domain_scores[domain] = domain_data
    
    # Compute overall score as average of all domain overall scores
    domain_overall_scores = [domain_scores[domain]['overall_score'] for domain in domain_scores]
    overall_score = sum(domain_overall_scores) / len(domain_overall_scores) if domain_overall_scores else 0.0
    
    domain_scores['overall_score'] = overall_score
    
    return {
        'directory': directory_path,
        'prefix_filter': prefix_filter,
        'num_files_processed': len(all_results),
        'raw_metrics': metrics,
        'domain_scores': domain_scores
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
                    print("Domain Scores:")
                    
                    # Print domain hierarchy
                    for domain, domain_data in result['domain_scores'].items():
                        if domain == 'overall_score':
                            continue
                        print(f"  {domain}:")
                        for field, value in domain_data.items():
                            if field == 'overall_score':
                                print(f"    - overall_score: {value:.4f}")
                            else:
                                print(f"    - {field}: {value:.4f}")
                    
                    print(f"  overall_score: {result['domain_scores']['overall_score']:.4f}")
                    print("-" * 60)
            
            # Compute multi-server aggregate scores
            multi_server_scores = None
            if 'double_server' in results_by_type and 'triple_server' in results_by_type:
                double_domain_scores = results_by_type['double_server']['domain_scores']
                triple_domain_scores = results_by_type['triple_server']['domain_scores']
                
                multi_server_scores = {}
                
                # Aggregate domain scores
                for domain in double_domain_scores:
                    if domain == 'overall_score':
                        multi_server_scores[domain] = (30/48 * double_domain_scores[domain] + 18/48 * triple_domain_scores[domain])
                    else:
                        multi_server_scores[domain] = {}
                        for field in double_domain_scores[domain]:
                            multi_server_scores[domain][field] = (30/48 * double_domain_scores[domain][field] + 18/48 * triple_domain_scores[domain][field])
                
                print(f"\nMulti-server aggregate scores for {directory}:")
                print("(30/48 * double_server + 18/48 * triple_server)")
                print("Domain Scores:")
                
                # Print domain hierarchy
                for domain, domain_data in multi_server_scores.items():
                    if domain == 'overall_score':
                        continue
                    print(f"  {domain}:")
                    for field, value in domain_data.items():
                        if field == 'overall_score':
                            print(f"    - overall_score: {value:.4f}")
                        else:
                            print(f"    - {field}: {value:.4f}")
                
                print(f"  overall_score: {multi_server_scores['overall_score']:.4f}")
                print("-" * 60)
            
            # Compute overall weighted average
            overall_weighted_scores = None
            if 'single_server' in results_by_type and multi_server_scores is not None:
                single_domain_scores = results_by_type['single_server']['domain_scores']
                
                overall_weighted_scores = {}
                
                # Aggregate domain scores
                for domain in single_domain_scores:
                    if domain == 'overall_score':
                        overall_weighted_scores[domain] = (56/104 * single_domain_scores[domain] + 48/104 * multi_server_scores[domain])
                    else:
                        overall_weighted_scores[domain] = {}
                        for field in single_domain_scores[domain]:
                            overall_weighted_scores[domain][field] = (56/104 * single_domain_scores[domain][field] + 48/104 * multi_server_scores[domain][field])
                
                print(f"\nOverall weighted average for {directory}:")
                print("(56/104 * single_server + 48/104 * multi_server)")
                print("Domain Scores:")
                
                # Print domain hierarchy
                for domain, domain_data in overall_weighted_scores.items():
                    if domain == 'overall_score':
                        continue
                    print(f"  {domain}:")
                    for field, value in domain_data.items():
                        if field == 'overall_score':
                            print(f"    - overall_score: {value:.4f}")
                        else:
                            print(f"    - {field}: {value:.4f}")
                
                print(f"  overall_score: {overall_weighted_scores['overall_score']:.4f}")
                print("-" * 60)
            
            # Save results as JSON
            output_data = {}
            
            # Add individual server type results at top level
            for server_type, result in results_by_type.items():
                output_data[server_type] = result['domain_scores']
            
            # Add aggregate results
            if multi_server_scores is not None:
                output_data['multi_server_aggregate'] = multi_server_scores
            if overall_weighted_scores is not None:
                output_data['overall_weighted_average'] = overall_weighted_scores
            
            output_file = os.path.join(directory, 'results.json')
            try:
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to: {output_file}")
            except Exception as e:
                print(f"Error saving results to {output_file}: {e}")
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
                        print("Domain Scores:")
                        
                        # Print domain hierarchy
                        for domain, domain_data in result['domain_scores'].items():
                            if domain == 'overall_score':
                                continue
                            print(f"  {domain}:")
                            for field, value in domain_data.items():
                                if field == 'overall_score':
                                    print(f"    - overall_score: {value:.4f}")
                                else:
                                    print(f"    - {field}: {value:.4f}")
                        
                        print(f"  overall_score: {result['domain_scores']['overall_score']:.4f}")
                        print("-" * 40)
                
                # Compute multi-server aggregate scores
                multi_server_scores = None
                if 'double_server' in results_by_type and 'triple_server' in results_by_type:
                    double_domain_scores = results_by_type['double_server']['domain_scores']
                    triple_domain_scores = results_by_type['triple_server']['domain_scores']
                    
                    multi_server_scores = {}
                    
                    # Aggregate domain scores
                    for domain in double_domain_scores:
                        if domain == 'overall_score':
                            multi_server_scores[domain] = (30/48 * double_domain_scores[domain] + 18/48 * triple_domain_scores[domain])
                        else:
                            multi_server_scores[domain] = {}
                            for field in double_domain_scores[domain]:
                                multi_server_scores[domain][field] = (30/48 * double_domain_scores[domain][field] + 18/48 * triple_domain_scores[domain][field])
                    
                    print(f"\nMulti-server aggregate scores for {directory}:")
                    print("(30/48 * double_server + 18/48 * triple_server)")
                    print("Domain Scores:")
                    
                    # Print domain hierarchy
                    for domain, domain_data in multi_server_scores.items():
                        if domain == 'overall_score':
                            continue
                        print(f"  {domain}:")
                        for field, value in domain_data.items():
                            if field == 'overall_score':
                                print(f"    - overall_score: {value:.4f}")
                            else:
                                print(f"    - {field}: {value:.4f}")
                    
                    print(f"  overall_score: {multi_server_scores['overall_score']:.4f}")
                    print("-" * 40)
                
                # Compute overall weighted average
                overall_weighted_scores = None
                if 'single_server' in results_by_type and multi_server_scores is not None:
                    single_domain_scores = results_by_type['single_server']['domain_scores']
                    
                    overall_weighted_scores = {}
                    
                    # Aggregate domain scores
                    for domain in single_domain_scores:
                        if domain == 'overall_score':
                            overall_weighted_scores[domain] = (56/104 * single_domain_scores[domain] + 48/104 * multi_server_scores[domain])
                        else:
                            overall_weighted_scores[domain] = {}
                            for field in single_domain_scores[domain]:
                                overall_weighted_scores[domain][field] = (56/104 * single_domain_scores[domain][field] + 48/104 * multi_server_scores[domain][field])
                    
                    print(f"\nOverall weighted average for {directory}:")
                    print("(56/104 * single_server + 48/104 * multi_server)")
                    print("Domain Scores:")
                    
                    # Print domain hierarchy
                    for domain, domain_data in overall_weighted_scores.items():
                        if domain == 'overall_score':
                            continue
                        print(f"  {domain}:")
                        for field, value in domain_data.items():
                            if field == 'overall_score':
                                print(f"    - overall_score: {value:.4f}")
                            else:
                                print(f"    - {field}: {value:.4f}")
                    
                    print(f"  overall_score: {overall_weighted_scores['overall_score']:.4f}")
                    print("-" * 60)
                
                # Save results as JSON
                output_data = {}
                
                # Add individual server type results at top level
                for server_type, result in results_by_type.items():
                    output_data[server_type] = result['domain_scores']
                
                # Add aggregate results
                if multi_server_scores is not None:
                    output_data['multi_server_aggregate'] = multi_server_scores
                if overall_weighted_scores is not None:
                    output_data['overall_weighted_average'] = overall_weighted_scores
                
                output_file = os.path.join(directory, 'results.json')
                try:
                    with open(output_file, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    print(f"\nResults saved to: {output_file}")
                except Exception as e:
                    print(f"Error saving results to {output_file}: {e}")


if __name__ == "__main__":
    main()