import argparse
import json
import os
from run_experiments import main as run_prompt_model
from analyze_results import main as run_analysis
from utils import log_message

def run_experiments_and_analysis(config_path):
    # Run the prompt model experiments
    log_message('Starting the prompt model experiments.', 'INFO')
    try:
        run_prompt_model(config_path)
        log_message('Prompt model experiments completed successfully.', 'INFO')
    except Exception as e:
        log_message(f'Error running prompt model experiments: {e}', 'ERROR')

    # Load the analysis configuration from the same config file
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        results_db_path = config['results_db_path']
        analyzed_results_dir = config.get('analyzed_results_dir', None)
        output_prefix = config.get('output_prefix', 'centrality')

        if analyzed_results_dir is None:
            log_message('Analyzed results directory is not specified in the config file.', 'ERROR')
            return

        # Ensure the analyzed results directory exists
        if not os.path.exists(analyzed_results_dir):
            os.makedirs(analyzed_results_dir)
    except Exception as e:
        log_message(f'Error reading configuration file: {e}', 'ERROR')
        return

    # Run the analysis on the experiment results
    log_message('Starting the analysis of experiment results.', 'INFO')
    try:
        run_analysis(config_path)
        log_message('Analysis of experiment results completed successfully.', 'INFO')
    except Exception as e:
        log_message(f'Error running analysis: {e}', 'ERROR')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run prompt model experiments and analyze results.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    
    # Check if "configs" folder and "results" folder exists, if not, create them
    if not os.path.exists('configs'):
        os.makedirs('configs')
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # Run the experiments and analysis
    run_experiments_and_analysis(args.config)
