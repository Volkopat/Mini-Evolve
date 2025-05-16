# Mini-Evolve
Inspired by Google DeepMind's AlphaEvolve: [AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

Mini-Evolve is a Python-based prototype of an evolutionary coding agent. It leverages Large Language Models (LLMs) to iteratively generate, evaluate, and improve code for well-defined problems. The system is designed to be modular, allowing for easy extension with new problems and configurations.

## Key Features
- **Evolutionary Algorithm**: Implements a core loop of selection, mutation (via LLM), and evaluation.
- **LLM-Powered Code Generation**: Uses an LLM (configurable, e.g., Ollama with various models) to propose code modifications and new solutions.
- **Modular Problem Definitions**: Problems are self-contained units, each with its own configuration, seed program, evaluation logic, and prompt context.
- **Automated Evaluation**: Each problem defines its own `evaluator_logic.py` to score generated programs based on specific criteria.
- **Program Database**: Stores all generated programs, their scores, validity, lineage (parent-child relationships), and other metadata in an SQLite database.
- **Asynchronous Operations**: Utilizes `asyncio` for concurrent LLM API calls, significantly speeding up the generation phase.
- **Configuration Management**: Uses YAML files for main system settings (`config/config.yaml`) and problem-specific parameters (`problems/<problem_name>/problem_config.yaml`).
- **Logging**: Comprehensive logging to console and file (`log/evolution.log`).
- **Reporting**: Generates detailed Markdown reports of evolutionary runs, including statistics, best programs, and evolutionary lineage (`tools/generate_report.py`).

## Project Structure
```
Mini-Evolve/
├── app/                  # Core application logic (evolution loop, LLM generator, evaluator, DB, selection)
├── config/               # Configuration files (main config.yaml)
├── db/                   # SQLite database (program_database.db)
├── docs/                 # Project documentation (implementation details, commands)
├── examples/             # Standalone example scripts
├── log/                  # Log files (evolution.log)
├── problems/             # Modular problem definitions
│   └── <problem_name>/   # Specific problem directory
│       ├── problem_config.yaml
│       ├── prompt_context.yaml
│       ├── seed_program.py
│       └── evaluator_logic.py
├── reports/              # Generated Markdown reports
├── templates/            # Jinja2 prompt templates
├── tools/                # Utility scripts (view_database.py, generate_report.py)
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Setup
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Mini-Evolve
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure the LLM and other settings:**
    - Open `config/config.yaml`.
    - **LLM Configuration**:
        - Set `llm_provider` (e.g., "ollama").
        - Update `api_base_url` if your LLM (e.g., Ollama) is not running locally or on the default port.
        - Specify the `model_name` you wish to use (e.g., "mistral", "llama3").
        - API keys are generally not needed for local Ollama, but adjust if using a cloud provider.
    - **Problem Selection**:
        - Set `current_problem_directory` to point to the desired problem in the `problems/` directory (e.g., `problems/matrix_multiplication_direct`).
    - Review other settings like database path, logging preferences, and evolutionary parameters.

## Running Mini-Evolve

### Main Evolution Loop
To start the evolutionary process for the problem specified in `config/config.yaml`:
```bash
python3 -m app.evolution_loop
```
This will:
1. Initialize or clear the program database (based on `db_reinitialize_on_startup` in `config.yaml`).
2. Load the seed program for the selected problem.
3. Run the evolutionary loop for the configured number of generations, generating, evaluating, and selecting programs.
4. Log progress to the console and `log/evolution.log`.

### Utility Tools

#### View Database
To inspect the contents of the program database:
```bash
python3 -m tools.view_database
```
This script offers various options to query and display programs from the database. Use `python3 -m tools.view_database --help` for more information.

#### Generate Report
After an evolutionary run, you can generate a Markdown report:
```bash
python3 -m tools.generate_report --db_path db/program_database.db
```
(Adjust `--db_path` if your database is located elsewhere).
The report will be saved in the `reports/` directory, named with the problem and timestamp (e.g., `report_matrix_multiplication_direct_YYYYMMDD_HHMMSS.md`).

## Available Problems
The system is designed to be modular. Problems are located in the `problems/` directory. Each problem sub-directory contains:
- `problem_config.yaml`: Defines problem-specific parameters like target metrics, evaluation timeouts, and function signatures.
- `prompt_context.yaml`: Provides detailed context for the LLM, including the problem description, constraints, examples, and desired output format.
- `seed_program.py`: The initial program that kicks off the evolution.
- `evaluator_logic.py`: Contains the `evaluate_program` function, which takes a program string and returns a dictionary of evaluation results (including a `score`).

Currently available example problems include:
- `problems/matrix_multiplication_direct`: Aims to evolve a Python function for matrix multiplication.
- `problems/tensor_decomposition_4x4_complex`: (Placeholder/Example for a more complex task)

To add a new problem, create a new directory under `problems/` and populate it with these files, then update `current_problem_directory` in `config/config.yaml` to point to your new problem.

## Future Work
- More sophisticated selection and diversity maintenance algorithms.
- Integration with formal testing frameworks (e.g., `pytest`).
- UI for experiment tracking and visualization.
- Support for more complex problem types and programming languages.

## Contributing
Contributions are welcome! Please feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change. 