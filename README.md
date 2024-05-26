# LLM Concept Digraph
This repository provides a versatile framework for exploring directional relationships between concepts within LLMs. The framework allows you to construct pairs from a set of variables (i.e. concepts) and prompt LLMs with each pair to determine which variable ranks higher in a custom-defined dimension. LLMs can respond by selecting either option or by stating that they are equal. These variables are then interpreted as nodes and the relationships as directed edges in a digraph, where various algorithms are used to analyze and rank them.

The framework is designed to be highly adaptable, allowing you to define your own variables and dimensions of comparison. Use of GPT, Claude, Gemini, Llama, and Mistral models is supported.

## Directory Structure
```plaintext
llm-concept-digraph
├── requirements.txt
├── configs/
│   └── config.json
├── personas.json
├── backends.py
├── utils.py
├── run_experiments.py
├── analyze_results.py
├── main.py
├── results/
│   ├── results.db
│   └── results
├── README.md
└── .env
```

## Installation
```bash
git clone https://github.com/timothytfchan/llm-concept-digraph.git
cd llm-concept-digraph
pip install -r requirements.txt
```

## Configuration
Configure your experiments by editing the JSON files in the configs directory. Each configuration file specifies the models, personas, temperatures, variables, and question for the experiments.

```json
{
    "results_db_path": "results/results_restaurants.db",  // Path to the SQLite database where experiment results are stored
    "analyzed_results_dir": "results/results_restaurants",  // Directory to save the analysis results (CSV files and plots)
    "models": ["gpt-4o-2024-05-13", "gpt-3.5-turbo"],  // List of model names to be used in the experiments
    "personas": ["None", "Default"],  // List of personas to be used in the experiments
    "temperatures": [0.0],  // List of temperatures to be used in the experiments (controls randomness in model responses)
    "variables": [  // List of variables (concepts) to be compared in the experiments
        "Marugame",
        "Coco Ichibanya",
        "Eat Tokyo",
        "Ichikokudo"
    ],
    "question": "Which of the following restaurants is more popular?\\nRestaurant A: {variable_a}\\nRestaurant B: {variable_b}\\n\\nAnswer in <answer></answer> tags.\\nAnswer 'A', i.e. <answer>A</answer> if restaurant A is more popular.\\nAnswer 'B', i.e. <answer>B</answer> if restaurant B is more popular.\\nAnswer 'C', i.e. <answer>C</answer> if they are equally popular.\\nRemember, you must answer within <answer></answer> tags.\\n\\nAnswer:"  // The question template used to prompt the model for each pair of variables
}
```

You must prompt the model to answer within <answer></answer> tags.

Set up your .env file by providing API keys:
```plaintext
GOOGLE_API_KEY = ...
ANTHROPIC_API_KEY = ...
TOGETHER_API_KEY = ...
OPENAI_API_KEY = ...
```

## Running the Experiments and Analysis
Once in the project directory, you can run the experiments and subsequent analysis using the main.py script. Provide the path to your configuration file as an argument:
```bash
python main.py --config path/to/your/config.json
```

You can also run the run_experiments.py and analyze_results.py files individually.

```bash
python run_experiments.py --config path/to/your/config.json
python analyze_results.py --config path/to/your/config.json
```

## Code Overview
### main.py
The main entry point for running both the prompt model experiments and the analysis. It calls functions from run_experiments.py and analyze_results.py sequentially.

### run_experiments.py
Contains the code for running pairwise comparison experiments using various language models. It generates pairs of variables and records the responses in a SQLite database.

### analyze_results.py
Performs analysis on the experiment results by constructing a directed graph of relationships and computing various centrality measures. The results are saved as CSV files and visualized as PNG images.

### utils.py
Utility functions for database operations, logging, and generating pairs of variables.

### backends.py
Contains backend implementations for different language models (e.g., OpenAI GPT, Claude, Gemini, Llama, Mistral).

### personas.json
Defines personas used in the experiments. Keys are persona names and can be referenced in the config file. You can add more personas here.
