# Mini-Evolve Project Commands

This file lists common commands for running and interacting with the Mini-Evolve project. All commands should be run from the project's root directory.

## Main Operations

- **Run the main evolutionary loop:**
  ```bash
  python3 -m app.evolution_loop
  ```

- **View the program database contents:**
  ```bash
  python3 -m tools.view_database
  ```

- **Generate a Markdown report of the last run:**
  ```bash
  python3 -m tools.generate_report
  ```

- **Run the LLM API example (standalone LLM test):**
  ```bash
  python3 -m examples.llm_api_example
  ```

## Individual Module Tests/Execution

Some modules contain `if __name__ == "__main__":` blocks that allow them to be run directly for testing or specific tasks. Use the `-m` flag to run them as modules from the project root if they are part of the `app` package or `tools` package (if they are structured to support it).

- **Test the evaluator (runs its internal self-tests):**
  ```bash
  python3 -m app.evaluator
  ```

- **Test the program database (includes DB initialization and sample operations):**
  ```bash
  python3 -m app.program_db
  ```

- **Test the selection mechanism (runs internal self-tests with a dummy DB):**
  ```bash
  python3 -m app.selection
  ```

- **Test the LLM generator (may attempt an LLM call based on current problem config):**
  ```bash
  python3 -m app.llm_generator
  ```

- **Test logger setup (prints log messages to console and file):**
  ```bash
  python3 -m app.logger_setup
  ```

## Notes

- Ensure your Python environment is set up with the dependencies from `requirements.txt`.
- Configuration settings (LLM provider, API keys, paths, active problem, etc.) are managed primarily in `config/config.yaml` and specific problem configurations are in `problems/<problem_name>/`. 