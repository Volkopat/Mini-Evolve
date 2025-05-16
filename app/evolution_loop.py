import yaml
import time
import os # For loading seed program
import re # For loading seed program
import asyncio # Added for async
import httpx # Added for async

from . import program_db
from . import llm_generator # This now loads all configs (main, problem, prompt_parts)
from . import evaluator
from . import selection
from . import logger_setup # This will configure logging upon import

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

    # Configs are loaded by llm_generator module at import time.
    # Access them via llm_generator.main_config, llm_generator.problem_config, llm_generator.prompt_parts
    try:
        if not llm_generator.main_config or not llm_generator.problem_config or not llm_generator.prompt_parts:
            # This might happen if llm_generator.load_all_configs() failed critically
            main_logger.critical("Configurations (main, problem, or prompt_parts) not loaded via llm_generator. Aborting.")
            # Attempt to reload if possible, or re-raise the error from llm_generator
            try:
                llm_generator.load_all_configs() # explicit call for retry
                main_logger.info("llm_generator.load_all_configs() re-attempted.")
                if not llm_generator.main_config or not llm_generator.problem_config or not llm_generator.prompt_parts:
                    main_logger.critical("Still unable to load all configs. Aborting.")
                    return
            except Exception as e:
                main_logger.critical("Failed to reload configs via llm_generator: %s. Aborting." % e)
                return
        
        main_cfg = llm_generator.main_config
        problem_cfg = llm_generator.problem_config
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

    # Initialize Database
    db_path = program_db.get_database_path() # Gets path from program_db which loaded it from main_config
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            main_logger.info("Removed existing database: %s" % db_path)
        except OSError as e:
            main_logger.warning("Could not remove existing database %s: %s. Proceeding with existing DB." % (db_path, e))
    program_db.init_db(db_path=db_path)
    main_logger.info("Program database initialized at: %s" % db_path)

    # Load, evaluate, and add seed program
    try:
        seed_code_str = load_seed_program(current_problem_dir, seed_function_name)
        main_logger.info("Loaded seed program for '%s':\n%s" % (seed_function_name, seed_code_str[:300]))
    except Exception:
        main_logger.critical("Failed to load seed program. Aborting.")
        return

    # Evaluate seed using the refactored evaluator
    eval_results_seed = evaluator.evaluate(seed_code_str, main_cfg, problem_cfg, current_problem_dir)
    main_logger.info("Seed program evaluation: Score=%s, Valid=%s" % (eval_results_seed.get('score'), eval_results_seed.get('is_valid')))
    if eval_results_seed.get('error_message'):
        main_logger.warning("Seed eval error: %s" % eval_results_seed.get('error_message'))

    if not eval_results_seed.get('is_valid'):
        main_logger.warning("Seed program is NOT valid. This may hinder evolution.")
    
    seed_id = program_db.add_program(
        code_string=seed_code_str,
        score=eval_results_seed.get('score'),
        is_valid=eval_results_seed.get('is_valid'),
        generation_discovered=0,
        parent_id=None,
        llm_prompt="N/A (Seed Program)",
        evaluation_results=eval_results_seed,
        db_path=db_path
    )
    if seed_id:
        main_logger.info("Seed program added to DB with ID: %s" % seed_id)
    else:
        main_logger.warning("Seed program was NOT added to DB (might be a duplicate if DB was not cleared or if seed is invalid and not added by policy). Evolution may be impacted.")

    async with httpx.AsyncClient() as client: 
        for gen in range(1, num_generations + 1):
            main_logger.info("---- Generation %s/%s Starting ----" % (gen, num_generations))

            selected_parents = selection.select_parents(
                num_parents=num_parents_to_select, 
                top_k_to_consider=top_k_parents_pool_size,
                db_path=db_path
            )

            if not selected_parents:
                main_logger.warning("Generation %s: No parents selected. Population might be too small or all invalid. Skipping generation." % gen)
                if gen == 1 and not seed_id: 
                     main_logger.error("Generation 1: No seed and no parents. Aborting.")
                     break
                continue

            main_logger.info("Generation %s: Selected %s parents." % (gen, len(selected_parents)))
            
            child_generation_tasks = []
            parent_info_for_tasks = [] 

            for i, parent_program_data in enumerate(selected_parents):
                parent_code = parent_program_data['code_string']
                parent_id = parent_program_data['program_id']
                parent_score = parent_program_data['score']
                main_logger.debug("  Parent %s/%s (ID: %s, Score: %.4f) preparing children tasks..." % (i+1, len(selected_parents), parent_id[:8], parent_score))

                for _ in range(num_children_per_parent): 
                    dynamic_prompt_context = {
                        'parent_code': parent_code,
                    }
                    task = llm_generator.generate_code_variant(context_from_evolution_loop=dynamic_prompt_context, client=client)
                    child_generation_tasks.append(task)
                    parent_info_for_tasks.append({'parent_id': parent_id})
            
            main_logger.info("Generation %s: Launching %s child generation tasks..." % (gen, len(child_generation_tasks)))
            generated_children_results = await asyncio.gather(*child_generation_tasks)
            main_logger.info("Generation %s: All %s child generation tasks completed." % (gen, len(generated_children_results)))

            children_processed_count = 0
            for idx, result_tuple in enumerate(generated_children_results):
                child_code_string, prompt_used_for_child = result_tuple
                current_parent_info = parent_info_for_tasks[idx] 
                parent_id_for_child = current_parent_info['parent_id']
                children_processed_count +=1

                if child_code_string is None:
                    main_logger.warning("    Child generation %s/%s for parent %s failed (LLM returned None or bad format). Prompt used (first 100 chars): %s..." % (children_processed_count, len(generated_children_results), parent_id_for_child[:8] if parent_id_for_child else 'N/A', prompt_used_for_child[:100]))
                    continue
                
                main_logger.debug("    Child %s/%s generated for parent %s. Code (first 200 chars):\n%s..." % (children_processed_count, len(generated_children_results), parent_id_for_child[:8] if parent_id_for_child else 'N/A', child_code_string[:200]))

                eval_results_child = evaluator.evaluate(child_code_string, main_cfg, problem_cfg, current_problem_dir)
                main_logger.info("    Child %s/%s (Parent: %s): Score=%.4f, Valid=%s" % (children_processed_count, len(generated_children_results), parent_id_for_child[:8] if parent_id_for_child else 'N/A', eval_results_child.get('score', 0.0), eval_results_child.get('is_valid')))
                if eval_results_child.get('error_message'):
                     main_logger.warning("    Child eval error: %s" % eval_results_child.get('error_message'))

                if eval_results_child.get('is_valid'):
                    child_id = program_db.add_program(
                        code_string=child_code_string,
                        score=eval_results_child.get('score'),
                        is_valid=eval_results_child.get('is_valid'),
                        generation_discovered=gen,
                        parent_id=parent_id_for_child, 
                        llm_prompt=prompt_used_for_child, 
                        evaluation_results=eval_results_child,
                        db_path=db_path
                    )
                    if child_id:
                        main_logger.info("    Added new valid child to DB: ID=%s, Score=%.4f" % (child_id[:8] if child_id else 'N/A', eval_results_child.get('score',0.0)))
                    else:
                        main_logger.info("    Valid child was a duplicate (by normalized hash), not added. Score=%.4f" % eval_results_child.get('score',0.0))
                else:
                    main_logger.info("    Child code was invalid. Not adding to DB. Error: %s" % eval_results_child.get('error_message', 'N/A'))

            best_programs_overall = program_db.get_best_n_programs(n=1, db_path=db_path)
            if best_programs_overall:
                current_best_program = best_programs_overall[0]
                current_best_score = current_best_program['score']
                current_best_id = current_best_program['program_id']
                main_logger.info("Generation %s Summary: Current best score in DB = %.4f (ID: %s)" % (gen, current_best_score, current_best_id[:8] if current_best_id else 'N/A'))
                
                if target_metric_value is not None:
                    actual_metric_value = current_best_program['evaluation_results'].get(target_metric_name, 0.0) 
                    # Allow for float comparisons
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
            if num_generations > 1 and gen < num_generations: # Avoid sleep on last gen or if only 1 gen
                time.sleep(1) 

    main_logger.info("==== Mini-Evolve Session Finished ====")

if __name__ == "__main__":
    asyncio.run(run_evolution()) 