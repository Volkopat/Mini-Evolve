# Detailed Implementation Plan: Adversarial Co-evolution (Triple Parallel Model)

**Goal:** To implement an adversarial co-evolution system with three parallel evolutionary tracks: Prompt, Program, and Evaluator, each driven by a dedicated LLM. The Prompt LLM and Program LLM will collaborate against the Evaluator LLM. All three tracks will incorporate MAP-Elites, self-correction, and hierarchical generation/subtasks.

---

## Phase 1: Prompt Database and Core Prompt LLM Infrastructure

*   **Goal**: Establish the ability to store, retrieve, and evolve prompt strings independently.
*   **Steps**:
    1.  **Create `app/prompt_db.py`**:
        *   Define a new SQLite database schema for prompts. This schema should include:
            *   `prompt_id` (UUID, Primary Key)
            *   `prompt_string` (TEXT, the content of the prompt)
            *   `normalized_prompt_string` (TEXT, for uniqueness checks)
            *   `normalized_prompt_hash` (TEXT, UNIQUE, for uniqueness)
            *   `score` (REAL, prompt's fitness based on programs it generates)
            *   `generation_discovered` (INTEGER)
            *   `parent_prompt_id` (TEXT, for lineage)
            *   `llm_prompt` (TEXT, the prompt used to generate this prompt)
            *   `evaluation_results_json` (TEXT, results of programs generated using this prompt)
            *   `descriptor_1`, `descriptor_2` (REAL, behavioral descriptors for prompts, e.g., length, complexity, specific keywords)
            *   `timestamp_added` (REAL)
        *   Implement `init_db()`, `add_prompt()`, `get_prompt()`, `get_best_prompt()`, `check_if_prompt_exists()` functions.
        *   Implement `add_prompt_to_map_elites()` for prompts.
        *   Add a `get_prompt_db_path()` function.
    2.  **Update `config/config.yaml`**:
        *   Add a new section for `prompt_database` with `type` and `path` (e.g., `db/prompt_database.db`).
    3.  **Create `app/prompt_llm.py`**:
        *   This module will be responsible for prompting the LLM to generate new prompt strings.
        *   Implement a function `generate_prompt_variant(context: dict, client: httpx.AsyncClient) -> Tuple[Optional[str], str]`.
        *   This function will render a new prompt template (e.g., `templates/prompt_generation_prompt.jinja`).
        *   It will receive context such as the performance of previous prompts, and the current problem definition.
        *   It will need to extract the generated prompt string from the LLM's response.
        *   Implement logic for self-correction if the generated prompt is invalid or leads to poor program generation.
        *   Consider if prompts can be generated hierarchically (e.g., a meta-prompt generating sub-prompts).
    4.  **Create `templates/prompt_generation_prompt.jinja`**:
        *   Design a prompt that instructs the LLM to evolve prompts.
        *   Include placeholders for context about prompt performance.
        *   Define output format for new prompt strings.
    5.  **Modify `app/selection.py`**:
        *   Add a new function `select_prompt(num_prompts: int, db_path: str) -> dict` to select parent prompts from `app/prompt_db.py`, incorporating MAP-Elites for prompts.

---

## Phase 2: Evaluator Database and Core Evaluator LLM Infrastructure

*   **Goal**: Establish the ability to store, retrieve, and evolve evaluator logic independently.
*   **Steps**:
    1.  **Create `app/evaluator_db.py`**:
        *   Define a new SQLite database schema for evaluators. This schema should include:
            *   `evaluator_id` (UUID, Primary Key)
            *   `evaluator_code_string` (TEXT, the content of `evaluator_logic.py`)
            *   `normalized_evaluator_code_string` (TEXT, for uniqueness checks)
            *   `normalized_evaluator_code_hash` (TEXT, UNIQUE, for uniqueness)
            *   `challenge_score` (REAL, the evaluator's fitness/complexity score)
            *   `generation_discovered` (INTEGER)
            *   `parent_evaluator_id` (TEXT, for lineage)
            *   `llm_prompt` (TEXT, the prompt used to generate this evaluator)
            *   `evaluation_results_json` (TEXT, results of testing this evaluator against programs)
            *   `descriptor_1`, `descriptor_2` (REAL, behavioral descriptors for evaluators, e.g., number of test cases, complexity of test cases, coverage)
            *   `timestamp_added` (REAL)
        *   Implement `init_db()`, `add_evaluator()`, `get_evaluator()`, `get_best_evaluator()`, `check_if_evaluator_exists()` functions.
        *   Implement `add_evaluator_to_map_elites()` for evaluators.
        *   Add a `get_evaluator_db_path()` function.
    2.  **Update `config/config.yaml`**:
        *   Add a new section for `evaluator_database` with `type` and `path` (e.g., `db/evaluator_database.db`).
    3.  **Create `app/evaluator_llm.py`**:
        *   This module will be responsible for prompting the LLM to generate new `evaluator_logic.py` content.
        *   Implement a function `generate_evaluator_logic(context: dict, client: httpx.AsyncClient) -> Tuple[Optional[str], str]`.
        *   This function will render a new prompt template (e.g., `templates/evaluator_generation_prompt.jinja`).
        *   Implement logic for self-correction if the generated evaluator is invalid or performs poorly.
        *   Consider if evaluators can be generated hierarchically (e.g., a meta-evaluator generating sub-evaluators).
    4.  **Create `templates/evaluator_generation_prompt.jinja`**:
        *   Design a prompt that instructs the LLM to act as an adversarial evaluator.
        *   Include instructions for defining "complexity" and "scoring" for the evaluator itself.
    5.  **Modify `app/evaluator.py`**:
        *   Update the `evaluate` function to accept the `evaluator_code_string` directly, instead of loading from a file path. This allows `evolution_loop.py` to pass the currently selected evaluator's code.
        *   The `evaluator_logic.py` files in `problems/` will now serve as initial seeds for the `evaluator_db`.
    6.  **Modify `app/selection.py`**:
        *   Add a new function `select_evaluator(num_evaluators: int, db_path: str) -> dict` to select parent evaluators from `app/evaluator_db.py`, incorporating MAP-Elites for evaluators.

---

## Phase 3: Program LLM Refinement and Integration

*   **Goal**: Adapt the existing `llm_generator.py` to act as the "Program LLM" and integrate it with the new Prompt and Evaluator databases.
*   **Steps**:
    1.  **Rename/Move `app/llm_generator.py` to `app/program_llm.py`**:
        *   Adapt to act as the "Program LLM", taking a prompt string as input, and continuing to support self-correction and hierarchical generation for programs.
    2.  **Modify `app/program_db.py`**:
        *   Update the `programs` table schema to include `prompt_id` (linking to `app/prompt_db.py`).
        *   Modify `add_program` to store this new `prompt_id`.
    3.  **Modify Program Prompt Templates (e.g., `templates/code_generation_prompt.jinja`, `templates/hierarchical_code_generation_prompt.jinja`, `templates/delegated_subtask_prompt.jinja`, `templates/lean_interaction_prompt.jinja`)**:
        *   Remove prompt evolution instructions.
        *   Ensure they clearly instruct the LLM to generate program code based on the provided prompt string.

---

## Phase 4: Orchestration in `app/evolution_loop.py`

*   **Goal**: Implement the triple parallel evolutionary tracks and their interdependencies.
*   **Steps**:
    1.  **Modify `app/evolution_loop.py`**:
        *   Import `app.prompt_db`, `app.evaluator_db`, `app.prompt_llm`, `app.evaluator_llm`, `app.program_llm`.
        *   Initialize all three databases (`prompt_db`, `program_db`, `evaluator_db`).
        *   Load initial seed prompt, program, and evaluator logic into their respective databases.
        *   Implement the main loop with three distinct, interdependent phases per generation:
            *   **Prompt Evolution Phase**:
                *   Select parent prompts from `prompt_db`.
                *   Call `app.prompt_llm.generate_prompt_variant()` to create new prompts.
                *   Add new prompts to `prompt_db`. (Evaluation of prompts happens indirectly via Program Evolution Phase).
            *   **Program Evolution Phase**:
                *   Select parent programs from `program_db`.
                *   Select a prompt from `prompt_db` (e.g., the current best prompt, or a diverse set).
                *   Fetch the *current best evaluator* from `evaluator_db`.
                *   Call `app.program_llm.generate_code_variant()` (Program LLM) with the selected prompt and evaluator.
                *   Evaluate generated programs using the selected evaluator.
                *   Update `program_db` with new programs.
                *   Update the score of the prompt in `prompt_db` based on the performance of programs generated using it.
            *   **Evaluator Evolution Phase**:
                *   Select parent evaluators from `evaluator_db`.
                *   Fetch the *best performing programs* from `program_db` (these are the "adversaries").
                *   Call `app.evaluator_llm.generate_evaluator_logic()` to create new evaluators.
                *   Test new evaluators against programs from `program_db` to determine `challenge_score`.
                *   Update `evaluator_db` with new evaluators.
        *   Manage the flow of information and dependencies between the three phases.

---

## Phase 5: Refinement and Testing

*   **Goal**: Ensure the new co-evolution system is robust and performs as expected.
*   **Steps**:
    1.  **Update Logging**: Enhance logging to track the co-evolution of all three entities.
    2.  **Update Reporting Tools**: Modify `tools/generate_report.py` and `tools/view_database.py` to display information from `prompt_db` and `evaluator_db`, and to visualize the triple lineage.
    3.  **Comprehensive Testing**: Develop new test cases to validate the co-evolutionary dynamics, including:
        *   Prompt LLM's ability to generate effective prompts.
        *   Program LLM's ability to adapt to prompts and evaluators.
        *   Evaluator LLM's ability to generate challenging evaluators.
        *   The adversarial balance and overall system stability.