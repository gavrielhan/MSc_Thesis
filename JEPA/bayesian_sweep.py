#!/usr/bin/env python3
"""
Bayesian Parameter Sweep without W&B
Uses Bayesian optimization approach but runs locally without W&B connection issues.
"""

import os
import sys
import json
import time
import subprocess
import random
import math
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_parameter_space(strategy: str) -> Dict[str, Any]:
    """Create parameter space for Bayesian-style optimization."""

    if strategy == "retina_feature_finetune":
        # Fixed LoRA rank and alpha (must match checkpoint)
        return {
            "lr": {
                "type": "float",
                "min": 1e-5,
                "max": 5e-4,
                "log_scale": True
            },
            "weight_decay": {
                "type": "float",
                "min": 1e-3,
                "max": 5e-2,
                "log_scale": True
            },
            "epochs": {
                "type": "categorical",
                "values": [20, 30, 50]
            },
            "lora_dropout": {
                "type": "categorical",
                "values": [0.1, 0.2, 0.3]
            },
            # Fixed parameters
            "lora_r": {"value": 16},
            "lora_alpha": {"value": 16},
            "batch_size": {"value": 1}
        }

    elif strategy == "imagenet_finetune":
        # Can sweep LoRA parameters
        return {
            "lr": {
                "type": "float",
                "min": 1e-5,
                "max": 5e-4,
                "log_scale": True
            },
            "weight_decay": {
                "type": "float",
                "min": 1e-3,
                "max": 5e-2,
                "log_scale": True
            },
            "epochs": {
                "type": "categorical",
                "values": [20, 30, 50]
            },
            "lora_r": {
                "type": "categorical",
                "values": [8, 16, 32, 64]
            },
            "lora_alpha": {
                "type": "categorical",
                "values": [8, 16, 32, 64]
            },
            "lora_dropout": {
                "type": "categorical",
                "values": [0.1, 0.2, 0.3]
            },
            "batch_size": {"value": 1}
        }

    elif strategy == "scratch":
        # Can sweep LoRA parameters
        return {
            "lr": {
                "type": "float",
                "min": 1e-5,
                "max": 5e-4,
                "log_scale": True
            },
            "weight_decay": {
                "type": "float",
                "min": 1e-3,
                "max": 5e-2,
                "log_scale": True
            },
            "epochs": {
                "type": "categorical",
                "values": [20, 30, 50]
            },
            "lora_r": {
                "type": "categorical",
                "values": [8, 16, 32, 64]
            },
            "lora_alpha": {
                "type": "categorical",
                "values": [8, 16, 32, 64]
            },
            "lora_dropout": {
                "type": "categorical",
                "values": [0.1, 0.2, 0.3]
            },
            "batch_size": {"value": 1}
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def sample_parameters(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample parameters from the parameter space (Bayesian-style)."""
    config = {}

    for param_name, param_config in param_space.items():
        if "value" in param_config:
            # Fixed parameter
            config[param_name] = param_config["value"]
        elif param_config["type"] == "float":
            if param_config.get("log_scale", False):
                # Log-uniform sampling
                min_val = param_config["min"]
                max_val = param_config["max"]
                log_min = math.log(min_val)
                log_max = math.log(max_val)
                config[param_name] = math.exp(random.uniform(log_min, log_max))
            else:
                # Uniform sampling
                config[param_name] = random.uniform(param_config["min"], param_config["max"])
        elif param_config["type"] == "categorical":
            # Categorical sampling
            config[param_name] = random.choice(param_config["values"])

    return config


def run_single_experiment(config: Dict[str, Any], strategy: str, experiment_id: int) -> Dict[str, Any]:
    """Run a single experiment with given config."""

    print(f"\n{'=' * 80}")
    print(f"Experiment {experiment_id}: {strategy}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'=' * 80}")
    print(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create temporary config file
    config_file = f"temp_config_{strategy}_{experiment_id}_{int(time.time())}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    try:
        # Run the training script
        cmd = [
            'python',
            os.path.join(os.path.dirname(__file__), 'ijepa_finetune_crossdatasets.py'),
            '--config', config_file,
            '--strategy', strategy,
            '--fold', '0',
            '--sweep_mode'
        ]

        print(f"Running: {' '.join(cmd)}")
        print(f"Process started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   universal_newlines=True)

        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(f"  {line}")
                output_lines.append(line)

        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=process.poll(),
            stdout='\n'.join(output_lines),
            stderr=''
        )

        print(
            f"Process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with return code: {result.returncode}")

        # Parse results
        experiment_result = {
            'experiment_id': experiment_id,
            'strategy': strategy,
            'config': config,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'test_auc': None,
            'train_auc': None,
            'error': None
        }

        if result.returncode == 0:
            print("Experiment completed successfully")
            # Try to parse AUC from stdout
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Test ROC AUC - Class 0:' in line:
                    try:
                        parts = line.split(',')
                        auc_0 = float(parts[0].split(':')[-1].strip())
                        auc_1 = float(parts[1].split(':')[-1].strip())
                        experiment_result['test_auc'] = (auc_0 + auc_1) / 2
                        print(f"Test AUC: {experiment_result['test_auc']:.4f}")
                    except:
                        print(f"Could not parse Test AUC from line: {line}")
                elif 'Train ROC AUC - Class 0:' in line:
                    try:
                        parts = line.split(',')
                        auc_0 = float(parts[0].split(':')[-1].strip())
                        auc_1 = float(parts[1].split(':')[-1].strip())
                        experiment_result['train_auc'] = (auc_0 + auc_1) / 2
                        print(f"Train AUC: {experiment_result['train_auc']:.4f}")
                    except:
                        print(f"Could not parse Train AUC from line: {line}")
        else:
            print(f"Experiment failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            experiment_result['error'] = result.stderr

        return experiment_result

    except subprocess.TimeoutExpired:
        print("Experiment timed out after 1 hour")
        return {
            'experiment_id': experiment_id,
            'strategy': strategy,
            'config': config,
            'return_code': -1,
            'error': 'Timeout after 1 hour',
            'test_auc': None,
            'train_auc': None
        }
    except Exception as e:
        print(f"Experiment failed with exception: {e}")
        return {
            'experiment_id': experiment_id,
            'strategy': strategy,
            'config': config,
            'return_code': -1,
            'error': str(e),
            'test_auc': None,
            'train_auc': None
        }
    finally:
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)


def run_bayesian_sweep(strategy: str, num_runs: int = 30, start_run: int = 1):
    """Run Bayesian-style parameter sweep."""

    print(f"Starting Bayesian-style parameter sweep for {strategy}")
    print(f"Number of runs: {num_runs}")
    print(f"Starting from run: {start_run}")

    # Create parameter space
    param_space = create_parameter_space(strategy)

    results = []
    start_time = time.time()

    for i in range(num_runs):
        run_number = start_run + i
        print(f"\n--- Run {run_number} (batch {i + 1}/{num_runs}) ---")

        # Sample parameters (Bayesian-style)
        config = sample_parameters(param_space)

        # Run the experiment
        result = run_single_experiment(config, strategy, run_number)
        results.append(result)

        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"bayesian_sweep_results_{strategy}_runs{start_run}-{start_run + num_runs - 1}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Small delay between experiments
        time.sleep(5)

    # Find best result
    valid_results = [r for r in results if r.get('test_auc') is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x.get('test_auc', 0))

        print(f"\n{'=' * 80}")
        print(f"Bayesian sweep completed for {strategy}!")
        print(f"{'=' * 80}")
        print(f"Best experiment: {best_result['experiment_id']}")
        print(f"Best test AUC: {best_result['test_auc']:.4f}")
        print(f"Best train AUC: {best_result['train_auc']:.4f}")
        print(f"Best config:")
        for key, value in best_result['config'].items():
            print(f"  {key}: {value}")

        # Save best parameters to simple JSON
        best_params = {
            'strategy': strategy,
            'best_test_auc': best_result['test_auc'],
            'best_train_auc': best_result['train_auc'],
            'parameters': best_result['config']
        }

        with open('best_parameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"\nBest parameters saved to: best_parameters.json")
    else:
        print(f"\nNo successful experiments for {strategy}")


def main():
    """Main function to run Bayesian parameter sweeps."""
    import argparse

    parser = argparse.ArgumentParser(description='Bayesian Parameter Sweep for LoRA Fine-tuning')
    parser.add_argument('--strategy', type=str,
                        choices=['retina_feature_finetune', 'imagenet_finetune', 'scratch', 'all'],
                        required=True,
                        help='Strategy to optimize (or "all" for parallel)')
    parser.add_argument('--runs', type=int, default=30,
                        help='Number of runs for the sweep (default: 30)')
    parser.add_argument('--start-run', type=int, default=1,
                        help='Starting run number (default: 1)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run all strategies in parallel')

    args = parser.parse_args()

    if args.strategy == 'all' or args.parallel:
        # Run all strategies in parallel
        strategies = ['retina_feature_finetune', 'imagenet_finetune', 'scratch']
        print(f"Running all strategies in parallel: {strategies}")

        import subprocess
        import threading

        def run_strategy(strategy):
            cmd = [
                'python',
                os.path.join(os.path.dirname(__file__), 'bayesian_sweep.py'),
                '--strategy', strategy,
                '--runs', str(args.runs)
            ]
            subprocess.run(cmd)

        threads = []
        for strategy in strategies:
            thread = threading.Thread(target=run_strategy, args=(strategy,))
            threads.append(thread)
            thread.start()
            print(f"Started {strategy} sweep")

        # Wait for all to complete
        for thread in threads:
            thread.join()

        print("All parallel sweeps completed!")
    else:
        # Run single strategy
        run_bayesian_sweep(args.strategy, args.runs, args.start_run)


if __name__ == "__main__":
    main() 