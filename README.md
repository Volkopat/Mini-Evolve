# Mini-Evolve
Inspired by Google DeepMind's AlphaEvolve: [AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

Mini-Evolve is a Python-based prototype of an evolutionary coding agent. It leverages Large Language Models (LLMs) to iteratively generate, evaluate, and improve code for well-defined problems. The system is designed to be modular, allowing for easy extension with new problems and configurations.

## Key Features
- **Evolutionary Algorithm**: Implements a core loop of selection, mutation (via LLM), and evaluation.
- **LLM-Powered Code Generation**: Uses an LLM (configurable, e.g., Ollama with various models, OpenRouter) to propose code modifications and new solutions.
- **Self-Correction**: The LLM can attempt to fix syntax or runtime errors in its own generated code based on evaluation feedback.
- **Hierarchical Task Delegation**: For complex problems, the LLM can act as an orchestrator, breaking down the main task into sub-tasks, delegating their implementation to other LLM calls (potentially to specialized sub-agents or different prompts), and then integrating the results. This is facilitated by `app/llm_tools.py`.
- **Lean Prover Interaction**: Supports problems that involve generating and potentially interacting with Lean code, enabling exploration in formal mathematics.
- **Modular Problem Definitions**: Problems are self-contained units, each with its own configuration, seed program, evaluation logic, and prompt context.
- **Automated Evaluation**: Each problem defines its own `evaluator_logic.py` to score generated programs based on specific criteria.
- **Program Database**: Stores all generated programs, their scores, validity, lineage (parent-child relationships), and other metadata in an SQLite database.
- **Asynchronous Operations**: Utilizes `asyncio` for concurrent LLM API calls, significantly speeding up the generation phase.
- **Configuration Management**: Uses YAML files for main system settings (`config/config.yaml`) and problem-specific parameters (`problems/<problem_name>/problem_config.yaml`).
- **Logging**: Comprehensive logging to console and file (`log/evolution.log`) with multiple levels (DEBUG, VERBOSE, INFO, WARNING, ERROR).
- **Reporting**: Generates detailed Markdown reports of evolutionary runs, including statistics, best programs, and evolutionary lineage (`tools/generate_report.py`).

## Project Structure
```
Mini-Evolve/
├── app/                  # Core application logic (evolution loop, LLM generator, evaluator, DB, selection, llm_tools)
├── config/               # Configuration files (main config.yaml)
├── db/                   # SQLite database (program_database.db)
├── docs/                 # Project documentation (implementation details, commands)
├── examples/             # Standalone example scripts
├── log/                  # Log files (evolution.log)
├── problems/             # Modular problem definitions
│   └── <problem_name>/   # Specific problem directory
│       ├── problem_config.yaml # Contains problem params, function details, and seed_program_code
│       ├── prompt_context.yaml # Provides detailed context for the LLM
│       └── evaluator_logic.py  # Contains the problem-specific evaluation function
├── reports/              # Generated Markdown reports
├── templates/            # Jinja2 prompt templates (e.g., code_generation_prompt.jinja, hierarchical_code_generation_prompt.jinja, delegated_subtask_prompt.jinja, lean_interaction_prompt.jinja)
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
        - Set `llm.provider` (e.g., "ollama_local", "openrouter").
        - For "ollama_local", update `llm.base_url` if Ollama is not running locally or on the default port.
        - For "openrouter", ensure `llm.api_key_env_var` points to the environment variable holding your OpenRouter API key (e.g., "OPENROUTER_API_KEY"). It's recommended to load this via a `.env` file in the project root (create one if it doesn't exist, and add `OPENROUTER_API_KEY="your_key_here"`), or set it in your shell environment.
        - Specify the `llm.model_name` you wish to use (e.g., "mistral", "llama3" for Ollama; "google/gemini-2.5-pro-preview" for OpenRouter).
    - **Problem Selection**:
        - Set `current_problem_directory` to point to the desired problem in the `problems/` directory (e.g., `problems/matrix_multiplication_direct`).
    - Review other settings like database path, logging preferences, evolutionary parameters, and new features:
        - `llm.enable_self_correction` (boolean)
        - `llm.max_correction_attempts` (integer)
        - `llm.enable_hierarchical_generation` (boolean)
        - `llm.max_delegation_depth` (integer)
        - `llm.max_sub_tasks_per_step` (integer)
        - `llm.delegation_iteration_limit` (integer)
        - `llm.use_specialized_prompts_for_hierarchical_tasks` (boolean, controls use of e.g. `delegated_subtask_prompt.jinja`)
        - `llm.lean_interaction.enable` (boolean, overall flag for Lean-related features if needed globally)
        - `llm.lean_interaction.lean_server_url` (string, if direct Lean server interaction is implemented)
        - `logging.log_level` can be set to `VERBOSE` for more detailed output.

## Running Mini-Evolve

### Main Evolution Loop
To start the evolutionary process for the problem specified in `config/config.yaml`:
```bash
python3 -m app.evolution_loop
```
This will:
1. Remove and re-initialize the program database at the start of the run.
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
- `problem_config.yaml`: Defines problem-specific parameters like target metrics, evaluation timeouts, function signatures, and importantly, the `seed_program_code`.
- `prompt_context.yaml`: Provides detailed context for the LLM, including the problem description, constraints, examples, and desired output format. This is crucial for guiding the LLM effectively.
- `evaluator_logic.py`: Contains the `evaluate_program` function, which takes a candidate program module and returns a dictionary of evaluation results (including a `score` and `is_valid` flag).

Currently available example problems include:
- `problems/matrix_multiplication_direct`: Aims to evolve a Python function for matrix multiplication.
- `problems/tensor_decomposition_4x4_complex`: Evolves a function for 4x4 complex matrix tensor decomposition, focusing on minimizing complex multiplications.
- `problems/tsp_heuristic`: Aims to evolve a Python function that provides a heuristic solution to the Traveling Salesperson Problem.
- `problems/set_cover`: Aims to evolve a Python function to find a minimal set cover.
- `problems/chromatic_number_plane`: Explores the Chromatic Number of the Plane problem, potentially involving Python for graph theory and Lean for formal sketches. This problem demonstrates hierarchical task delegation and interaction with mathematical concepts.

To add a new problem, create a new directory under `problems/` and populate it with `problem_config.yaml` (including `seed_program_code`), `prompt_context.yaml`, and `evaluator_logic.py`. Then update `current_problem_directory` in `config/config.yaml` to point to your new problem.
Consider if your problem requires new Jinja templates (e.g., for specific types of Lean interaction or sub-task delegation).

## Future Work
- More sophisticated selection and diversity maintenance algorithms.
- Integration with formal testing frameworks (e.g., `pytest`).
- UI for experiment tracking and visualization.
- Support for more complex problem types and programming languages.
- Enhanced error analysis and feedback mechanisms for the LLM.

## Contributing
Contributions are welcome! Please feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change. 