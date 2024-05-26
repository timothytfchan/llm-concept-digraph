import json
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import (
    create_tables,
    record_exists,
    insert_record,
    generate_pairs,
    log_message
)

from backends import get_backend

def run_experiment(results_db_path, model_name, persona, temperature, variable_a, variable_b, question):
    # Check if the record already exists
    try:
        record_exists_flag = record_exists(results_db_path, model_name, persona, temperature, variable_a, variable_b)
    except Exception as e:
        log_message(f'Error checking record existence for {variable_a} vs {variable_b} with model {model_name}, persona {persona}, temperature {temperature}: {e}', 'ERROR')
        return

    # If record does not exist, proceed with the experiment
    if not record_exists_flag:
        try:
            backend = get_backend(model_name)
        except Exception as e:
            log_message(f'Error getting backend for {model_name}: {e}', 'ERROR')
            return
        # Retrieve system prompt for the persona
        try:
            with open('personas.json', 'r') as f:
                personas = json.load(f)
            system_prompt = personas[persona]
        except Exception as e:
            log_message(f'Error loading persona for {persona}: {e}', 'ERROR')
            return
        # Format the question with the variables
        try:
            question_formatted = question.format(variable_a=variable_a, variable_b=variable_b)
        except Exception as e:
            log_message(f'Error formatting question for {variable_a} vs {variable_b}: {e}', 'ERROR')
            return
        
        # Make the API request
        try:
            response = backend.complete(system_prompt=system_prompt, user_prompt=question_formatted, temperature=temperature, top_p=1.0).completion
        except Exception as e:
            log_message(f'Error during API request for {variable_a} vs {variable_b}: {e}', 'ERROR')
            return
        
        # Log the raw response for debugging purposes
        try:
            log_message(f'API response for {variable_a} vs {variable_b}: {response}', 'INFO')
        except Exception as e:
            log_message(f'Error logging API response for {variable_a} vs {variable_b}: {e}', 'ERROR')
            return

        # Extract choice using regex pattern
        try:
            choice_matches = re.findall(r'<answer>(.*?)</answer>', response)
            if not choice_matches:
                log_message(f'No valid answer found in the response for {variable_a} vs {variable_b}', 'ERROR')
                return
            choice = choice_matches[-1]
        except Exception as e:
            log_message(f'Error extracting choice from response for {variable_a} vs {variable_b}: {e}', 'ERROR')
            return

        # Insert the record into the database
        try:
            insert_record(results_db_path, model_name, persona, temperature, variable_a, variable_b, response, choice)
            log_message(f'Successfully recorded: {variable_a} vs {variable_b}', 'INFO')
        except Exception as e:
            log_message(f'Error inserting record for {variable_a} vs {variable_b}: {e}', 'ERROR')
            return


def main(config_path):
    log_message('Starting the main process.', 'INFO')
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        log_message('Configuration file loaded successfully.', 'INFO')
    except Exception as e:
        log_message(f'Error reading configuration file: {e}', 'ERROR')
        return

    results_db_path = config.get('results_db_path')
    models = config.get('models', [])
    personas = config.get('personas', [])
    temperatures = config.get('temperatures', [])
    variables = config.get('variables', [])
    question = config.get('question')

    if not all([results_db_path, models, personas, temperatures, variables, question]):
        log_message('Configuration file is missing required fields.', 'ERROR')
        log_message(f'Missing fields: {", ".join([field for field, value in config.items() if not value])}', 'ERROR')
        return
    
    try:
        create_tables(results_db_path)
    except Exception as e:
        log_message(f'Error creating tables in the database: {e}', 'ERROR')
        return

    try:
        variable_pairs = generate_pairs(variables)
        log_message(f'Generated {len(variable_pairs)} variable pairs.', 'INFO')
    except Exception as e:
        log_message(f'Error generating variable pairs: {e}', 'ERROR')
        return

    try:
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for model in models:
                for persona in personas:
                    for temperature in temperatures:
                        for variable_a, variable_b in variable_pairs:
                            futures.append(
                                executor.submit(run_experiment, results_db_path, model, persona, temperature, variable_a, variable_b, question)
                            )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log_message(f'Error in future: {e}', 'ERROR')
        log_message('All pairwise comparisons have been processed.', 'INFO')
    except Exception as e:
        log_message(f'Error during parallel execution: {e}', 'ERROR')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pairwise comparison experiments.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
