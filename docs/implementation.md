# Detailed Technical Project Plan: Mini-Evolve (AlphaEvolve-Inspired System)

## Project Goal
To develop a simplified, functional prototype of an evolutionary coding agent that uses a Large Language Model (LLM) to iteratively improve code for a well-defined, simple problem, with a focus on robust technical implementation of core components.

## Core Principles to Emulate (Simplified)
- LLM-driven Code Modification: Use an LLM to suggest changes or new versions of code.
- Automated Evaluation: Automatically score the generated code based on predefined criteria in a secure manner.
- Evolutionary Loop: Iteratively generate, evaluate, and select code to find better solutions.
- Program Database: Maintain a structured collection of promising programs and their metadata.

## Phase 0: Foundational Research, Setup & Scoping

### Original Plan (Summary)
- **Objective**: Understand core concepts, define a narrow scope, choose tools, and set up the environment.
- **Key Tasks (Conceptual)**:
    - Deep dive into source material (AlphaEvolve, FunSearch).
    - Define a precise "Toy Problem".
    - Select LLM and familiarize with its API.
    - Set up Python environment, version control, basic configuration.
    - Create an initial "seed" program.

### Current Status & Key Implemented Structures (Diverging from/Implementing Phase 0)
- **Project Structure**: The project is now organized with a clear directory structure:
    - `app/`: Core evolutionary logic (evaluator, LLM generator, evolution loop, program DB, selection, logger, llm_tools).
    - `config/`: Main configuration (`config.yaml`).
    - `db/`: SQLite database storage (`program_database.db`).
    - `docs/`: Project documentation (`implementation.md`, `commands.md`).
    - `examples/`: Standalone examples (e.g., `llm_api_example.py`).
    - `log/`: Log files (`evolution.log`).
    - `problems/`: Houses modular problem definitions. Each sub-directory (e.g., `matrix_multiplication_direct/`) contains:
        - `problem_config.yaml`: Problem-specific parameters, function details, evaluation criteria.
        - `prompt_context.yaml`: Detailed problem description, constraints, and examples for LLM prompts.
        - `seed_program.py`: The initial code for the evolutionary process for that specific problem.
        - `evaluator_logic.py`: Python module with the `evaluate_program` function for the problem.
    - `reports/`: Stores Markdown reports generated by `tools/generate_report.py`.
    - `templates/`: Jinja2 prompt templates (e.g., `code_generation_prompt.jinja`, `hierarchical_code_generation_prompt.jinja`, `delegated_subtask_prompt.jinja`, `lean_interaction_prompt.jinja`).
    - `tools/`: Utility scripts (`view_database.py`, `generate_report.py`).
- **Modular Problem Definition**: Shifted from a single toy problem to a modular system where each problem is self-contained within its directory in `problems/`.
- **Configuration Management**:
    - **Main Configuration**: `config/config.yaml` for global settings (LLM, evolution loop, DB, logging) and `current_problem_directory` to switch active problems.
    - **Dynamic Loading**: Core components (`app/llm_generator.py`, `app/evolution_loop.py`) dynamically load configurations based on the active problem.
- **Seed Programs**: Now problem-specific, located at `<current_problem_directory>/seed_program.py`.

## Phase 1: Core Components - LLM Interaction & Evaluation

### Original Plan (Summary)
- **Objective**: Implement robust LLM code generation and secure automated evaluation.
- **Key Tasks**:
    - `llm_generator.py`: Prompt engineering (Jinja2), API interaction, response parsing, error handling.
    - `evaluator.py`: Core `evaluate` function, sandboxing (Docker discussed), error handling, metric calculation.
    - `main_test_cycle.py`: Basic integration test script.

### Current Status & Implementation
- **`app/llm_generator.py` (LLM Code Generation Module)**:
    - **Prompt Engineering**: Uses Jinja2. Dynamically assembles prompt context from `prompt_context.yaml`, evolution loop data, and main configuration. Supports multiple templates like `code_generation_prompt.jinja` (default), `hierarchical_code_generation_prompt.jinja` (for orchestrator LLM calls), `delegated_subtask_prompt.jinja` (for worker LLM calls), and `lean_interaction_prompt.jinja` (for tasks involving Lean).
    - **API Interaction**: Employs `httpx` for asynchronous calls to LLM providers. Handles API parameters and code extraction from LLM responses.
    - **Self-Correction**: If `llm.enable_self_correction` is true, `llm_generator.py` can be called by the evolution loop to attempt to fix code that previously failed evaluation (syntax or runtime errors) by providing the error message back to the LLM.
    - **Hierarchical Generation Support**: Interacts with `app/llm_tools.py` to handle hierarchical task delegation. When `llm.enable_hierarchical_generation` is true, the LLM might output delegation requests, which are processed by `llm_tools.py` to make further LLM calls (potentially using specialized prompts) and integrate results.
    - **Error Handling**: Includes mechanisms for API errors and response parsing issues.
- **`app/evaluator.py` (Automated Evaluator Module)**:
    - **Core Function**: `evaluate(program_code_string: str, main_cfg: dict, problem_cfg: dict, current_problem_dir: str) -> dict`.
    - **Dynamic Loading**: Acts as a generic harness. It dynamically imports and executes the `evaluate_program` function from the `evaluator_logic.py` file within the specified `current_problem_dir`.
    - **Sandboxing & Security**: This responsibility is now delegated to each problem's specific `evaluator_logic.py`. The initial plan's Docker suggestion would be implemented here if required by a problem's complexity or security needs.
    - **Metric Calculation**: Handled entirely by the problem-specific `evaluator_logic.py`.
- **Integration Testing**: The `main_test_cycle.py` script has been **removed**. Its role is fulfilled by running the complete `app.evolution_loop` or specific module tests (e.g., `python3 -m app.llm_generator`).

## Phase 2: Simple Evolutionary Loop & Program Database

### Original Plan (Summary)
- **Objective**: Create an iterative loop for generation, evaluation, selection, and storage.
- **Key Tasks**:
    - `program_db.py`: Schema definition (SQLite recommended), API functions (`add_program`, `get_program`, etc.).
    - `selection.py`: Initial selection strategy (e.g., truncation + random).
    - `evolution_loop.py`: Orchestrate the main loop, load config, manage generations.
    - `logger_setup.py`: Logging integration.

### Current Status & Implementation
- **`app/program_db.py` (Program Database Module)**:
    - **Schema**: Implemented largely as planned (program_id, code_string, normalized_code_hash, score, is_valid, generation_discovered, parent_id, llm_prompt, llm_response_raw, evaluation_metrics as JSON, timestamp_added, error_message, self_correction_attempts, delegated_tasks_info as JSON).
    - **Storage**: Uses SQLite. The database path is configured in `config/config.yaml` (typically `db/program_database.db`). The `db/` directory stores the database file.
    - **API Functions**: Key functions like `add_program`, `get_program`, `get_best_n_programs`, `get_unique_programs_from_top_k`, `update_program_after_self_correction`, and `check_if_exists` are implemented.
- **`app/selection.py` (Selection Mechanism)**:
    - **Strategy**: Implements a Truncation + Random Selection approach. It fetches unique programs from the database, prioritizing valid programs but falling back to include invalid ones if no valid candidates are available (to ensure the loop can start).
    - The `select_parents` function is implemented accordingly.
- **`app/evolution_loop.py` (Evolutionary Loop Implementation)**:
    - **Configuration**: Loads parameters from the main `config/config.yaml` and the active problem's `problem_config.yaml`.
    - **Main Loop Logic**:
        1. Initializes the database (by default, removes any existing DB on startup to ensure fresh runs).
        2. Loads, evaluates, and adds the problem-specific seed program from `<current_problem_dir>/seed_program.py`.
        3. Iterates for the configured `num_generations`:
            a. Selects parent programs using `selection.select_parents()`.
            b. For each parent, generates a configured number of child programs using `llm_generator.generate_code_variant()` (asynchronous calls). This step now potentially involves hierarchical generation via `app/llm_tools.py` if enabled, where a single call to `generate_code_variant` might trigger multiple LLM interactions for task breakdown and sub-task implementation.
            c. Evaluates each child program using `evaluator.evaluate()` (which calls the problem-specific logic).
            d. If a program is invalid due to syntax/runtime errors and self-correction is enabled, the loop attempts to correct it using `llm_generator.generate_corrected_code()` for up to `max_correction_attempts`.
            e. Adds new, unique child programs (or their corrected versions) to the database.
            f. Logs generation summaries and checks if a target metric (defined in `problem_config.yaml`) has been achieved for early termination.
- **`app/logger_setup.py` (Logging & Monitoring)**:
    - Utilizes Python's built-in `logging` module.
    - Configured to log to both the console and a file (typically `log/evolution.log`, path defined in `config/config.yaml`).
    - Supports multiple log levels, including a custom `VERBOSE` level (numerical value 15) for highly detailed output, especially useful for tracking LLM interactions and hierarchical task processing.
    - Logs key metrics, generation summaries, and errors as intended.
- **`app/llm_tools.py` (LLM Tools for Hierarchical Generation)**:
    - **Purpose**: Provides helper functions to support hierarchical task delegation by the LLM.
    - **Key Functions**:
        - `extract_delegation_requests()`: Parses the LLM output to find structured requests for sub-task delegation.
        - `generate_delegated_task_code()`: Makes new LLM calls (potentially using `delegated_subtask_prompt.jinja`) to implement the sub-tasks.
        - `integrate_delegated_results()`: Combines the original orchestrator code with the code generated for sub-tasks.
        - `process_hierarchical_requests()`: Manages the overall flow of delegation, iteration, and integration within a single generation step.
    - This module is central to the `llm.enable_hierarchical_generation` feature.
- **Utility Scripts**:
    - `tools/view_database.py`: A command-line tool to inspect the contents of the program database.
    - `tools/generate_report.py`: A command-line tool to generate Markdown summary reports of evolutionary runs, saved in the `reports/` directory.

## Phase 3: Enhancements & Refinements

### Original Plan (Summary)
- **Objective**: Improve system intelligence, efficiency, and robustness.
- **Key Task Areas**: Advanced prompting, diff output, population management, error handling, async operations, parameter tuning, expanding to complex problems, meta-prompt evolution.

### Current Status & Implementation
- **Advanced Prompting Strategies**: The system uses Jinja2 templates extensively. Specific templates support different generation modes:
    - `code_generation_prompt.jinja`: Default prompt for generating code variants.
    - `hierarchical_code_generation_prompt.jinja`: Used when the LLM is expected to act as an orchestrator, potentially breaking down tasks.
    - `delegated_subtask_prompt.jinja`: Used when generating code for a specific sub-task delegated by the orchestrator LLM.
    - `lean_interaction_prompt.jinja`: Tailored for problems involving Lean code generation or interaction.
    - Error feedback for self-correction is also incorporated into prompts.
- **Diff Output/Application**: Not currently implemented.
- **Population Management & Diversity**: Basic program uniqueness is enforced via normalized code hashing in `program_db.py`. Advanced diversity metrics or algorithms like MAP-Elites are future work.
- **Improved Error Handling & Resilience**: A good baseline of error handling exists in API interactions and core loop. Further validation of LLM outputs can be an area for improvement.
- **Asynchronous Operations (for performance)**:
    - **Implemented**: LLM calls made through `httpx.AsyncClient` in `app/llm_generator.py` are asynchronous. `app.evolution_loop.py` uses `asyncio.gather` to manage these concurrent tasks, significantly speeding up child generation.
- **Parameter Tuning & Experiment Management**: Configuration is primarily managed through YAML files. Advanced tools like Hydra or Typer are not yet integrated.
- **Expand to More Complex Problems**: The modular structure (`problems/` directory) is specifically designed to facilitate the addition of new, more complex problems. The `problems/chromatic_number_plane/` problem is an example that leverages hierarchical generation and interacts with mathematical concepts, including generating Lean code snippets.
- **Interaction with Formal Methods**: Rudimentary support for Lean interaction is present through specialized prompts and the ability for `evaluator_logic.py` to handle Lean code in outputs. Direct execution or verification of Lean code is problem-specific and not yet a generalized feature of `app/evaluator.py`.

## Phase 4: Testing, Documentation, and Future Work

### Original Plan (Summary)
- **Objective**: Consolidate, test rigorously, create comprehensive documentation, and outline future work.
- **Key Task Areas**: Unit tests, integration tests, robustness testing, benchmarking, code documentation, README, architectural diagrams, project report.

### Current Status & Implementation (Ongoing)
- **Systematic Testing**: Many modules include `if __name__ == "__main__":` blocks with self-tests. However, a more formal testing suite using a framework like `pytest` for comprehensive unit and integration tests is a desirable future step to achieve high code coverage.
- **Code Documentation & Quality**:
    - **Docstrings**: Present for many key functions and modules.
    - **Type Hinting**: Used in many parts of the codebase.
    - Continuous improvement in these areas is ongoing.
- **Project Documentation**:
    - `README.md`: Provides an overview and setup instructions.
    - `docs/commands.md`: Lists common operational commands.
    - `docs/implementation.md`: This document, detailing the project plan and current status.
- **Project Reporting**: The `tools/generate_report.py` script provides detailed Markdown summaries of evolutionary runs, including statistics, best programs, and evolutionary lineage.
- **Future Work Outline (from original plan, still largely relevant)**:
    - Support for additional programming languages or more diverse problem domains.
    - Implementation of more advanced evolutionary algorithms (e.g., co-evolution, multi-objective optimization).
    - Integration with formal verification tools for code correctness.
    - Self-adaptation mechanisms for evolutionary parameters.
    - Development of a user interface for experiment monitoring and interaction.

## General Technical Considerations (Status against Original Plan)
- **Modularity & Abstraction**: Significantly enhanced with the problem-centric architecture, dynamic configuration loading, and the separation of generic evaluation harnessing from problem-specific logic.
- **Idempotency**: `program_db.add_program` ensures idempotency based on normalized code hashes, preventing duplicate functional programs.
- **Resource Management**: LLM API timeouts are configurable. Further resource monitoring for long runs could be considered.
- **Code Normalization & Hashing**: Implemented in `app/program_db.py` using Python's `ast` module and SHA256 hashing.
- **Security (Sandboxing)**: The responsibility for secure execution of generated code is now delegated to the problem-specific `evaluator_logic.py` found within each problem's directory. This requires careful implementation for each new problem, especially if using `exec`. The original recommendation for Docker remains valid for scenarios requiring robust sandboxing of untrusted code.
- **Ethical AI**: Remains a general consideration for any generative AI project.

This document has been updated to reflect the current state and key architectural changes of the Mini-Evolve project as of the last interaction.
