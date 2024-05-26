import json
import os
import argparse
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_all_records, log_message

def load_experiment_results(results_db_path):
    return get_all_records(results_db_path)

def build_relation_graph(experiment_results):
    graph = nx.DiGraph()
    for model_name, persona, temperature, variable_a, variable_b, response, choice in experiment_results:
        if choice == 'A':  # A is greater than B
            graph.add_edge(variable_b, variable_a)
        elif choice == 'B':  # B is greater than A
            graph.add_edge(variable_a, variable_b)
        elif choice == 'C':  # A and B are equal
            graph.add_edge(variable_a, variable_b)
            graph.add_edge(variable_b, variable_a)
        else:
            continue
    return graph

def compute_centrality_measures(graph):
    try:
        centrality_measures = {
            'in_degree_centrality': nx.in_degree_centrality(graph),
            'out_degree_centrality': nx.out_degree_centrality(graph),
            'betweenness_centrality': nx.betweenness_centrality(graph),
            'closeness_centrality': nx.closeness_centrality(graph),
            'eigenvector_centrality': nx.eigenvector_centrality(graph),
            'pagerank': nx.pagerank(graph)
        }
    except Exception as e:
        log_message(f'Error computing centrality measures: {e}', 'ERROR')
        centrality_measures = {}
    return centrality_measures

def save_centrality_measures(analyzed_results_dir, centrality_measures, model_name, persona, temperature):
    for measure_name, scores in centrality_measures.items():
        try:
            df = pd.DataFrame(scores.items(), columns=['Variable', measure_name])
            df = df.sort_values(by=measure_name, ascending=False)
            output_path = os.path.join(analyzed_results_dir, f'{model_name}_{persona}_{temperature}_{measure_name}.csv')
            df.to_csv(output_path, index=False)
        except Exception as e:
            log_message(f'Error saving centrality measures {measure_name} for {model_name}, {persona}, {temperature}: {e}', 'ERROR')

def create_graph_visualization(analyzed_results_dir, graph, centrality_measures, model_name, persona, temperature):
    for measure_name, scores in centrality_measures.items():
        try:
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(graph)
            nx.draw_networkx_nodes(graph, pos, node_size=[v * 10000 for v in scores.values()])
            nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=10)
            nx.draw_networkx_labels(graph, pos)
            plt.title(f'Graph Visualization - {measure_name}')
            plt.tight_layout()
            output_path = os.path.join(analyzed_results_dir, f'{model_name}_{persona}_{temperature}_{measure_name}.png')
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            log_message(f'Error creating graph visualization {measure_name} for {model_name}, {persona}, {temperature}: {e}', 'ERROR')

def create_centrality_plots(analyzed_results_dir, centrality_measures, model_name, persona, temperature):
    for measure_name, scores in centrality_measures.items():
        try:
            df = pd.DataFrame(scores.items(), columns=['Variable', measure_name])
            df = df.sort_values(by=measure_name, ascending=True)
            plt.figure(figsize=(12, 12))
            plt.barh(df['Variable'], df[measure_name])
            plt.xticks(rotation=90)
            plt.ylabel('Variable')
            plt.xlabel(measure_name)
            plt.title(f'{measure_name} Scores of Variables')
            plt.tight_layout()
            output_path = os.path.join(analyzed_results_dir, f'{model_name}_{persona}_{temperature}_{measure_name}.png')
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            log_message(f'Error creating centrality plot {measure_name} for {model_name}, {persona}, {temperature}: {e}', 'ERROR')

def compare_centrality_measures(analyzed_results_dir, centrality_measures, comparison_type, fixed_value):
    for measure_name in next(iter(centrality_measures.values())).keys():
        try:
            comparison_df = pd.DataFrame()
            for combination, measures in centrality_measures.items():
                if measure_name in measures:
                    comparison_df[combination] = pd.Series(measures[measure_name])

            comparison_df = comparison_df.sort_index()
            output_path = os.path.join(analyzed_results_dir, f'comparison_{comparison_type}_{fixed_value}_{measure_name}.csv')
            comparison_df.to_csv(output_path)

            # Create comparison plot
            comparison_df.plot(kind='bar', figsize=(15, 7))
            plt.xticks(rotation=45)
            plt.title(f'Comparison of {measure_name} - {comparison_type}: {fixed_value}')
            plt.xlabel('Variables')
            plt.ylabel(measure_name)
            plt.legend(title=f'{comparison_type}')
            plt.tight_layout()
            plt.savefig(os.path.join(analyzed_results_dir, f'comparison_{comparison_type}_{fixed_value}_{measure_name}.png'))
            plt.close()
        except Exception as e:
            log_message(f'Error comparing centrality measures {measure_name} for {comparison_type}, {fixed_value}: {e}', 'ERROR')

def analyze_results_for_combination(filtered_results, analyzed_results_dir, model_name, persona, temperature):
    relation_graph = build_relation_graph(filtered_results)
    centrality_measures = compute_centrality_measures(relation_graph)
    
    save_centrality_measures(analyzed_results_dir, centrality_measures, model_name, persona, temperature)
    create_graph_visualization(analyzed_results_dir, relation_graph, centrality_measures, model_name, persona, temperature)
    create_centrality_plots(analyzed_results_dir, centrality_measures, model_name, persona, temperature)
    log_message(f'Analysis completed for model: {model_name}, persona: {persona}, temperature: {temperature}', 'INFO')

    return centrality_measures

def main(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    results_db_path = config['results_db_path']
    analyzed_results_dir = config['analyzed_results_dir']
    models = config['models']
    personas = config['personas']
    temperatures = config['temperatures']

    if not os.path.exists(analyzed_results_dir):
        os.makedirs(analyzed_results_dir)
    
    # Load the data once
    experiment_results = load_experiment_results(results_db_path)

    model_centralities = {model: {} for model in models}
    persona_centralities = {persona: {} for persona in personas}
    temperature_centralities = {temperature: {} for temperature in temperatures}

    for model_name in models:
        for persona in personas:
            for temperature in temperatures:
                filtered_results = [
                    result for result in experiment_results
                    if result[0] == model_name and result[1] == persona and result[2] == temperature
                ]
                if filtered_results:
                    centrality_measures = analyze_results_for_combination(filtered_results, analyzed_results_dir, model_name, persona, temperature)
                    model_centralities[model_name][(persona, temperature)] = centrality_measures
                    persona_centralities[persona][(model_name, temperature)] = centrality_measures
                    temperature_centralities[temperature][(model_name, persona)] = centrality_measures
                else:
                    log_message(f'No results found for model: {model_name}, persona: {persona}, temperature: {temperature}', 'WARNING')
    
    # Compare models for each (persona, temperature)
    for persona in personas:
        for temperature in temperatures:
            centrality_measures = {
                model_name: model_centralities[model_name][(persona, temperature)]
                for model_name in models if (persona, temperature) in model_centralities[model_name]
            }
            compare_centrality_measures(analyzed_results_dir, centrality_measures, 'model', f'{persona}_{temperature}')
    
    # Compare personas for each (model, temperature)
    for model_name in models:
        for temperature in temperatures:
            centrality_measures = {
                persona: persona_centralities[persona][(model_name, temperature)]
                for persona in personas if (model_name, temperature) in persona_centralities[persona]
            }
            compare_centrality_measures(analyzed_results_dir, centrality_measures, 'persona', f'{model_name}_{temperature}')
    
    # Compare temperatures for each (model, persona)
    for model_name in models:
        for persona in personas:
            centrality_measures = {
                temperature: temperature_centralities[temperature][(model_name, persona)]
                for temperature in temperatures if (model_name, persona) in temperature_centralities[temperature]
            }
            compare_centrality_measures(analyzed_results_dir, centrality_measures, 'temperature', f'{model_name}_{persona}')
    
    log_message('All analyses completed successfully', 'INFO')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze experiment results.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
