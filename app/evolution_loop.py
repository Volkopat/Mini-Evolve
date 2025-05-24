import yaml
import time
import os # For loading seed program
import re # For loading seed program
import asyncio # Added for async
import httpx # Added for async

from . import program_db
from . import prompt_db # New: For prompt evolution
from . import evaluator_db # New: For evaluator evolution
from . import program_llm # Renamed from llm_generator
from . import prompt_llm # New: For prompt generation
from . import evaluator_llm # New: For evaluator generation
from . import evaluator
from . import selection
from . import logger_setup # This will configure logging upon import
from .logger_setup import VERBOSE_LEVEL_NUM # Import the VERBOSE level
from .evolution_phases import run_prompt_evolution_phase, run_program_evolution_phase, run_evaluator_evolution_phase # New

# --- Configuration Loading ---
CONFIG_FILE = "config/config.yaml"

def load_evolution_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            full_config = yaml.safe_load(f)
            # Validate that necessary sections exist
            if 'evolution_loop' not in full_config:
                raise ValueError("'evolution_loop' section missing in config.yaml")
            if 'toy_problem' not in full_config:
                raise ValueError("'toy_problem' section missing in config.yaml")
            return full_config
    except FileNotFoundError:
        print(f"CRITICAL: Configuration file {CONFIG_FILE} not found.")
        raise
    except yaml.YAMLError as e:
        print(f"CRITICAL: Error parsing YAML configuration: {e}")
        raise
    except ValueError as e:
        print(f"CRITICAL: Configuration error: {e}")
        raise

# --- Seed Program Loading ---
def load_seed_program(current_problem_dir: str, seed_function_name: str) -> str:
    logger = logger_setup.get_logger("EvoLoop.SeedLoader")
    seed_filename = "seed_program.py" # Standard name for seed programs
    filepath = os.path.join(current_problem_dir, seed_filename)
    try:
        with open(filepath, "r") as f:
            seed_code_str = f.read()
        if not seed_code_str.strip():
            logger.error("Seed program file '%s' is empty." % filepath)
            raise ValueError("Seed program file is empty.")
        logger.info("Successfully loaded seed program from: %s" % filepath)
        return seed_code_str
    except FileNotFoundError:
        logger.error("Seed program file '%s' not found. Cannot proceed without a seed." % filepath)
        raise 
    except Exception as e:
        logger.error("Error loading seed program from '%s': %s" % (filepath, e))
        raise

# --- Main Evolutionary Loop ---
async def run_evolution():
    main_logger = logger_setup.get_logger("EvolutionLoop")
    main_logger.info("==== Mini-Evolve Session Started (Async) ====")

    # Configs are loaded by program_llm, prompt_llm, and evaluator_llm modules at import time.
    # Access them via their respective module's global variables.
    try:
        # Ensure all LLM modules have loaded their configs
        if not program_llm.main_config or not program_llm.problem_config or not program_llm.program_context_parts:
            main_logger.critical("Program LLM configurations not loaded. Aborting.")
            return
        if not prompt_llm.main_config or not prompt_llm.problem_config or not prompt_llm.prompt_context_parts:
            main_logger.critical("Prompt LLM configurations not loaded. Aborting.")
            return
        if not evaluator_llm.main_config or not evaluator_llm.problem_config or not evaluator_llm.prompt_context_parts:
            main_logger.critical("Evaluator LLM configurations not loaded. Aborting.")
            return
        
        main_cfg = program_llm.main_config # Use program_llm's main_config as the primary
        problem_cfg = program_llm.problem_config # Use program_llm's problem_config as the primary
        
        # Set MAP-Elites config for all relevant DBs
        map_elites_config = main_cfg.get('map_elites', {})
        program_db.set_map_elites_config(map_elites_config)
        prompt_db.set_map_elites_config(map_elites_config) # New
        evaluator_db.set_map_elites_config(map_elites_config) # New

        current_problem_dir = main_cfg.get('current_problem_directory')
        if not current_problem_dir:
             main_logger.critical("current_problem_directory not found in main_config. Aborting.")
             return

    except Exception as e:
        main_logger.critical("Failed to access initial configurations from llm_generator: %s. Aborting evolution loop." % e)
        return

    evo_config = main_cfg.get('evolution_loop', {})
    num_generations = evo_config.get('num_generations', 10)
    num_children_per_parent = evo_config.get('num_children_per_parent', 2)
    top_k_parents_pool_size = evo_config.get('top_k_parents_pool', 10)
    num_parents_to_select = evo_config.get('num_parents_to_select', 2)

    # Problem-specific details for logging or success conditions
    problem_name = os.path.basename(current_problem_dir) # For logging
    seed_function_name = problem_cfg.get('function_details', {}).get('name', 'solve') # Default to solve if not specified
    
    # Evaluation target metrics from problem_config (optional)
    eval_cfg = problem_cfg.get('evaluation', {})
    target_metric_name = eval_cfg.get('target_metric_name', 'score') # Default to checking 'score'
    target_metric_value = eval_cfg.get('target_metric_value') # Can be None
    target_comparison_mode = eval_cfg.get('comparison_mode', 'greater_than_or_equal_to')

    main_logger.info("Evolving problem: %s" % problem_name)
    main_logger.info("Evolutionary Loop Config: Generations=%s, Children/Parent=%s, ParentPoolK=%s, NumParentsSelect=%s" % (
        num_generations, num_children_per_parent, top_k_parents_pool_size, num_parents_to_select))

    # Initialize Databases
    program_db_path = program_db.get_database_path()
    prompt_db_path = prompt_db.get_database_path() # New
    evaluator_db_path = evaluator_db.get_database_path() # New

    # Clean existing databases for a fresh run
    for path in [program_db_path, prompt_db_path, evaluator_db_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
                main_logger.info("Removed existing database: %s" % path)
            except OSError as e:
                main_logger.warning("Could not remove existing database %s: %s. Proceeding with existing DB." % (path, e))
    
    program_db.init_db(db_path=program_db_path)
    prompt_db.init_db(db_path=prompt_db_path) # New
    evaluator_db.init_db(db_path=evaluator_db_path) # New

    main_logger.info("All databases initialized.")

    # Load seed program, prompt, and evaluator strings from problem_config
    seed_program_code_str = problem_cfg.get('seed_program_code')
    if not seed_program_code_str or not seed_program_code_str.strip():
        main_logger.critical("'seed_program_code' not found or is empty in problem_config.yaml. Aborting.")
        return
    
    seed_prompt_string = problem_cfg.get('seed_prompt_string')
    if not seed_prompt_string or not seed_prompt_string.strip():
        main_logger.critical("'seed_prompt_string' not found or is empty in problem_config.yaml. Aborting.")
        return
 
    seed_evaluator_code_str = problem_cfg.get('seed_evaluator_code')
    if not seed_evaluator_code_str or not seed_evaluator_code_str.strip():
        main_logger.critical("'seed_evaluator_code' not found or is empty in problem_config.yaml. Aborting.")
        return

    # 1. Add seed prompt to DB first
    seed_prompt_length = len(seed_prompt_string.splitlines())
    seed_prompt_keywords = len(re.findall(r'\b[A-Za-z_]+\b', seed_prompt_string)) # Simple heuristic
    eval_results_seed_prompt = {"initial_evaluation": "N/A"} # Will be updated later by program performance
 
    seed_prompt_id = prompt_db.add_prompt(
        prompt_string=seed_prompt_string,
        score=0.0, # Initial score, will be updated by program performance
        generation_discovered=0,
        parent_prompt_id=None,
        llm_prompt="N/A (Seed Prompt)",
        evaluation_results=eval_results_seed_prompt,
        descriptor_1=seed_prompt_length,
        descriptor_2=seed_prompt_keywords,
        db_path=prompt_db_path
    )
    if seed_prompt_id:
        main_logger.info("Seed prompt added to DB with ID: %s" % seed_prompt_id)
        if prompt_db.get_map_elites_config().get('enable', False):
            prompt_db.add_prompt_to_map_elites(
                prompt_id=seed_prompt_id,
                score=0.0, # Initial score
                descriptor_values=[seed_prompt_length, seed_prompt_keywords],
                db_path=prompt_db_path
            )
    else:
        main_logger.warning("Seed prompt was NOT added to DB. Evolution may be impacted.")
        # If seed prompt cannot be added, we might need to abort or use a dummy. For now, abort.
        return # Abort evolution if seed prompt fails to add

    # 2. Add seed evaluator to DB
    eval_results_seed_evaluator = {"initial_evaluation": "N/A"} # Will be updated later by program performance
    seed_evaluator_length = len(seed_evaluator_code_str.splitlines())
    seed_evaluator_test_cases = seed_evaluator_code_str.count("test_case") # Heuristic
 
    seed_evaluator_id = evaluator_db.add_evaluator(
        evaluator_code_string=seed_evaluator_code_str,
        challenge_score=0.0, # Initial score, will be updated by program performance
        generation_discovered=0,
        parent_evaluator_id=None,
        llm_prompt="N/A (Seed Evaluator)",
        evaluation_results=eval_results_seed_evaluator,
        descriptor_1=seed_evaluator_length,
        descriptor_2=seed_evaluator_test_cases,
        db_path=evaluator_db_path
    )
    if seed_evaluator_id:
        main_logger.info("Seed evaluator added to DB with ID: %s" % seed_evaluator_id)
        if evaluator_db.get_map_elites_config().get('enable', False):
            evaluator_db.add_evaluator_to_map_elites(
                evaluator_id=seed_evaluator_id,
                challenge_score=0.0, # Initial score
                descriptor_values=[seed_evaluator_length, seed_evaluator_test_cases],
                db_path=evaluator_db_path
            )
    else:
        main_logger.warning("Seed evaluator was NOT added to DB. Evolution may be impacted.")
        # If seed evaluator cannot be added, we might need to abort or use a dummy. For now, abort.
        return # Abort evolution if seed evaluator fails to add

    # Initialize current_evaluator_data with the seed evaluator for the first generation
    current_evaluator_data = {
        "evaluator_code_string": seed_evaluator_code_str,
        "evaluator_id": seed_evaluator_id,
        "challenge_score": 0.0 # Initial score, will be updated by program performance
    }

    # 3. Evaluate seed program using the initial evaluator (now available)
    eval_results_seed_program = evaluator.evaluate(seed_program_code_str, main_cfg, problem_cfg, current_problem_dir, seed_evaluator_code_str)
    
    # 4. Add seed program to DB (now with seed_prompt_id available)
    seed_program_id = program_db.add_program(
        code_string=seed_program_code_str,
        score=eval_results_seed_program.get('score'),
        is_valid=eval_results_seed_program.get('is_valid'),
        generation_discovered=0,
        parent_id=None,
        prompt_id=seed_prompt_id, # Pass the seed prompt ID
        llm_prompt="N/A (Seed Program)",
        evaluation_results=eval_results_seed_program,
        descriptor_1=eval_results_seed_program.get('code_length'),
        descriptor_2=eval_results_seed_program.get('num_function_calls'),
        db_path=program_db_path
    )
    if seed_program_id:
        main_logger.info("Seed program added to DB with ID: %s" % seed_program_id)
        if program_db.get_map_elites_config().get('enable', False):
            program_db.add_program_to_map_elites(
                program_id=seed_program_id,
                score=eval_results_seed_program.get('score'),
                descriptor_values=[eval_results_seed_program.get('code_length'), eval_results_seed_program.get('num_function_calls')],
                db_path=program_db_path
            )
    else:
        main_logger.warning("Seed program was NOT added to DB. Evolution may be impacted.")
        # If seed program cannot be added, we might need to abort or use a dummy. For now, abort.
        return # Abort evolution if seed program fails to add

    async with httpx.AsyncClient() as client:
        for gen in range(1, num_generations + 1):
            main_logger.info("==== Generation %s/%s Starting ====" % (gen, num_generations))

            current_prompt_data = await run_prompt_evolution_phase(
                gen=gen,
                main_cfg=main_cfg,
                problem_cfg=problem_cfg,
                current_problem_dir=current_problem_dir,
                seed_prompt_string=seed_prompt_string,
                num_children_per_parent=num_children_per_parent,
                num_parents_to_select=num_parents_to_select,
                top_k_parents_pool_size=top_k_parents_pool_size,
                prompt_db_path=prompt_db_path,
                client=client
            )

            current_evaluator_data = await run_evaluator_evolution_phase(
                gen=gen,
                main_cfg=main_cfg,
                problem_cfg=problem_cfg,
                current_problem_dir=current_problem_dir,
                seed_evaluator_code_str=seed_evaluator_code_str,
                num_children_per_parent=num_children_per_parent,
                num_parents_to_select=num_parents_to_select,
                top_k_parents_pool_size=top_k_parents_pool_size,
                program_db_path=program_db_path,
                evaluator_db_path=evaluator_db_path,
                client=client
            )

            program_evolution_result = await run_program_evolution_phase(
                gen=gen,
                main_cfg=main_cfg,
                problem_cfg=problem_cfg,
                current_problem_dir=current_problem_dir,
                seed_program_id=seed_program_id,
                seed_evaluator_code_str=seed_evaluator_code_str,
                num_children_per_parent=num_children_per_parent,
                num_parents_to_select=num_parents_to_select,
                top_k_parents_pool_size=top_k_parents_pool_size,
                program_db_path=program_db_path,
                prompt_db_path=prompt_db_path, # New: Pass prompt_db_path
                evaluator_db_path=evaluator_db_path,
                current_prompt_data=current_prompt_data,
                current_evaluator_data=current_evaluator_data, # Added missing argument
                client=client
            )
            # Handle potential early exit from program evolution phase
            if program_evolution_result is False: # Indicates an error or no parents found
                main_logger.warning("Program evolution phase skipped or failed. Continuing to next generation if possible.")
                continue # Skip to next generation


            # --- Check for overall target metric (Program-based) ---
            best_programs_overall = program_db.get_best_n_programs(n=1, db_path=program_db_path)
            if best_programs_overall:
                current_best_program = best_programs_overall[0]
                current_best_score = current_best_program['score']
                current_best_id = current_best_program['program_id']
                main_logger.info("Generation %s Summary: Current best program score in DB = %.4f (ID: %s)" % (gen, current_best_score, current_best_id[:8] if current_best_id else 'N/A'))
                
                if target_metric_value is not None:
                    actual_metric_value = current_best_program['evaluation_results'].get(target_metric_name, 0.0)
                    met_target = False
                    if target_comparison_mode == 'greater_than_or_equal_to':
                        if actual_metric_value >= float(target_metric_value) - 1e-9: met_target = True
                    elif target_comparison_mode == 'less_than_or_equal_to':
                        if actual_metric_value <= float(target_metric_value) + 1e-9: met_target = True
                    elif target_comparison_mode == 'equals':
                        if abs(actual_metric_value - float(target_metric_value)) < 1e-9: met_target = True
                    
                    if met_target:
                        main_logger.info("SUCCESS: Target metric '%s' value %.4f (vs target %.4f) achieved by program %s! Halting evolution." % (
                            target_metric_name, actual_metric_value, float(target_metric_value), current_best_id[:8] if current_best_id else 'N/A'
                        ))
                        break
            else:
                main_logger.warning("Generation %s Summary: No valid programs found in DB to determine best score." % gen)
            
            main_logger.info("---- Generation %s Finished ----\n" % gen)
            if num_generations > 1 and gen < num_generations:
                time.sleep(1)


    main_logger.info("==== Mini-Evolve Session Finished ====")

if __name__ == "__main__":
    asyncio.run(run_evolution()) 