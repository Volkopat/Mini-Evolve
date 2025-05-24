import asyncio
import httpx
import os
import re
import time
from typing import Dict, Any

from . import program_db
from . import prompt_db
from . import evaluator_db
from . import program_llm
from . import prompt_llm
from . import evaluator_llm
from . import evaluator
from . import selection
from .logger_setup import get_logger, VERBOSE_LEVEL_NUM

async def run_prompt_evolution_phase(
    gen: int,
    main_cfg: Dict[str, Any],
    problem_cfg: Dict[str, Any],
    current_problem_dir: str,
    seed_prompt_string: str,
    num_children_per_parent: int,
    num_parents_to_select: int,
    top_k_parents_pool_size: int,
    prompt_db_path: str,
    client: httpx.AsyncClient
) -> Dict[str, Any]:
    """
    Executes the prompt evolution phase for a given generation.
    Generates new prompts, evaluates them (implicitly via program performance),
    and adds them to the prompt database.
    """
    logger = get_logger("EvolutionLoop.PromptPhase")
    logger.info("\n--- Prompt Evolution Phase ---")

    selected_parent_prompts = selection.select_prompts(
        num_prompts=num_parents_to_select, # Corrected parameter name
        top_k_to_consider=top_k_parents_pool_size,
        db_path=prompt_db_path
    )
    if not selected_parent_prompts:
        logger.warning("Generation %s: No parent prompts selected. Using seed prompt as current." % gen)
        current_prompt_data = {"prompt_string": seed_prompt_string, "prompt_id": "seed_prompt", "score": 0.0}
    else:
        current_prompt_data = selected_parent_prompts[0]
        logger.info("Selected parent prompt (ID: %s, Score: %.4f)" % (current_prompt_data['prompt_id'][:8], current_prompt_data['score']))

    prompt_generation_tasks = []
    for i in range(num_children_per_parent):
        dynamic_prompt_context = {
            'parent_prompt': current_prompt_data['prompt_string'],
            'previous_error_feedback': None
        }
        task = prompt_llm.generate_prompt_variant(
            context_from_evolution_loop=dynamic_prompt_context,
            client=client,
            current_delegation_depth=0
        )
        prompt_generation_tasks.append(task)
    
    generated_prompts_results = await asyncio.gather(*prompt_generation_tasks)
    
    for new_prompt_string, prompt_used_for_prompt in generated_prompts_results:
        if new_prompt_string:
            prompt_length = len(new_prompt_string.splitlines())
            prompt_keywords = len(re.findall(r'\b[A-Za-z_]+\b', new_prompt_string))
            
            new_prompt_id = prompt_db.add_prompt(
                prompt_string=new_prompt_string,
                score=0.0,
                generation_discovered=gen,
                parent_prompt_id=current_prompt_data['prompt_id'],
                llm_prompt=prompt_used_for_prompt,
                evaluation_results={"initial_eval": "pending"},
                descriptor_1=prompt_length,
                descriptor_2=prompt_keywords,
                db_path=prompt_db_path
            )
            if new_prompt_id:
                logger.info("Added new prompt to DB: ID=%s" % new_prompt_id[:8])
                if prompt_db.get_map_elites_config().get('enable', False):
                    prompt_db.add_prompt_to_map_elites(
                        prompt_id=new_prompt_id,
                        score=0.0,
                        descriptor_values=[prompt_length, prompt_keywords],
                        db_path=prompt_db_path
                    )
        else:
            logger.warning("Prompt generation failed.")
    
    best_prompt_overall = prompt_db.get_best_n_prompts(n=1, db_path=prompt_db_path)
    if best_prompt_overall:
        current_prompt_data = best_prompt_overall[0]
        logger.info("Current best prompt for program generation (ID: %s, Score: %.4f)" % (current_prompt_data['prompt_id'][:8], current_prompt_data['score']))
    else:
        logger.warning("No valid prompts in DB. Using seed prompt for program generation.")
        current_prompt_data = {"prompt_string": seed_prompt_string, "prompt_id": "seed_prompt", "score": 0.0}
    
    return current_prompt_data

async def run_program_evolution_phase(
    gen: int,
    main_cfg: Dict[str, Any],
    problem_cfg: Dict[str, Any],
    current_problem_dir: str,
    seed_program_id: str,
    seed_evaluator_code_str: str,
    num_children_per_parent: int,
    num_parents_to_select: int,
    top_k_parents_pool_size: int,
    program_db_path: str,
    prompt_db_path: str, # New: Accept prompt_db_path
    evaluator_db_path: str,
    current_prompt_data: Dict[str, Any],
    current_evaluator_data: Dict[str, Any], # New: Pass current evaluator data
    client: httpx.AsyncClient
) -> None:
    """
    Executes the program evolution phase for a given generation.
    Generates new programs, evaluates them, and adds them to the program database.
    Updates prompt scores based on program performance.
    """
    logger = get_logger("EvolutionLoop.ProgramPhase")
    logger.info("\n--- Program Evolution Phase ---")

    selected_parent_programs = selection.select_parents(
        num_parents=num_parents_to_select,
        top_k_to_consider=top_k_parents_pool_size,
        db_path=program_db_path
    )

    if not selected_parent_programs:
        logger.warning("Generation %s: No parent programs selected. Skipping program evolution." % gen)
        if gen == 1 and not seed_program_id:
             logger.error("Generation 1: No seed program and no parents. Aborting.")
             return # This return will break the outer loop in run_evolution
        return

    logger.info("Generation %s: Selected %s parent programs." % (gen, len(selected_parent_programs)))
    
    best_evaluator_overall = evaluator_db.get_best_n_evaluators(n=1, db_path=evaluator_db_path)
    if best_evaluator_overall:
        current_evaluator_data = best_evaluator_overall[0]
        logger.info("Current best evaluator (ID: %s, Score: %.4f)" % (current_evaluator_data['evaluator_id'][:8], current_evaluator_data['challenge_score']))
    else:
        logger.warning("No valid evaluators in DB. Using seed evaluator.")
        current_evaluator_data = {"evaluator_code_string": seed_evaluator_code_str, "evaluator_id": "seed_evaluator"}

    program_generation_tasks = []
    program_parent_info_for_tasks = []

    for i, parent_program_data in enumerate(selected_parent_programs):
        parent_code = parent_program_data['code_string']
        parent_id = parent_program_data['program_id']
        parent_score = parent_program_data['score']
        parent_eval_results = parent_program_data.get('evaluation_results', {})
        parent_previous_error = parent_eval_results.get('error_message')

        logger.log(VERBOSE_LEVEL_NUM, "Selected Parent Program %s/%s (ID: %s, Score: %.4f)\nParent Code:\n%s",
                        i+1, len(selected_parent_programs), parent_id[:8] if parent_id else 'N/A',
                        parent_score, parent_code)

        logger.debug("  Parent Program %s/%s (ID: %s, Score: %.4f) preparing children tasks..." % (i+1, len(selected_parent_programs), parent_id[:8], parent_score))

        for _ in range(num_children_per_parent):
            dynamic_program_context = {
                'parent_code': parent_code,
                'previous_error_feedback': parent_previous_error
            }
            task = program_llm.generate_code_variant(
                context_from_evolution_loop=dynamic_program_context,
                client=client,
                current_prompt_string=current_prompt_data['prompt_string'],
                current_delegation_depth=0
            )
            program_generation_tasks.append(task)
            program_parent_info_for_tasks.append({'parent_id': parent_id, 'prompt_id': current_prompt_data['prompt_id']})
    
    logger.info("Generation %s: Launching %s program generation tasks..." % (gen, len(program_generation_tasks)))
    generated_program_results = await asyncio.gather(*program_generation_tasks)
    logger.info("Generation %s: All %s program generation tasks completed." % (gen, len(generated_program_results)))

    programs_processed_count = 0
    for idx, result_tuple in enumerate(generated_program_results):
        child_code_string, prompt_used_for_child = result_tuple
        current_parent_info = program_parent_info_for_tasks[idx]
        parent_id_for_child = current_parent_info['parent_id']
        prompt_id_for_child = current_parent_info['prompt_id']
        programs_processed_count +=1

        if child_code_string is None:
            logger.warning("    Program generation %s/%s for parent %s failed (LLM returned None or bad format). Prompt used (first 100 chars): %s..." % (programs_processed_count, len(generated_program_results), parent_id_for_child[:8] if parent_id_for_child else 'N/A', prompt_used_for_child[:100]))
            continue
        
        logger.log(VERBOSE_LEVEL_NUM, "Initial Child Program %s/%s (Parent: %s):\nPrompt (first 300 chars):\n%s...\nChild Code:\n%s",
                        programs_processed_count, len(generated_program_results),
                        parent_id_for_child[:8] if parent_id_for_child else 'N/A',
                        prompt_used_for_child[:300], child_code_string)
        logger.debug("    Child Program %s/%s generated for parent %s. Code (first 200 chars):\n%s..." % (programs_processed_count, len(generated_program_results), parent_id_for_child[:8] if parent_id_for_child else 'N/A', child_code_string[:200]))

        current_code_to_evaluate = child_code_string
        current_eval_results = evaluator.evaluate(current_code_to_evaluate, main_cfg, problem_cfg, current_problem_dir, current_evaluator_data['evaluator_code_string'])
        current_prompt_for_db = prompt_used_for_child

        llm_settings = main_cfg.get('llm', {})
        enable_self_correction = llm_settings.get('enable_self_correction', False)
        max_correction_attempts = llm_settings.get('max_correction_attempts', 3)

        if enable_self_correction and not current_eval_results.get('is_valid', False):
            error_message = current_eval_results.get('error_message', '')
            is_correctable_error_type = (
                "SyntaxError:" in error_message or
                "Error during program execution/setup:" in error_message or
                ("Error in problem-specific evaluator" in error_message and "Function 'solve' not found" in error_message)
            )

            if is_correctable_error_type:
                logger.info("Child Program %s/%s (Parent: %s) has a correctable error: '%s'. Attempting self-correction." %
                                 (programs_processed_count, len(generated_program_results), parent_id_for_child[:8] if parent_id_for_child else 'N/A', error_message.splitlines()[0]))
                
                for attempt in range(max_correction_attempts):
                    logger.info("  Self-correction attempt %s/%s..." % (attempt + 1, max_correction_attempts))
                    correction_context = {
                        'parent_code': current_code_to_evaluate,
                        'previous_error_feedback': error_message
                    }
                    corrected_code_string, correction_prompt_used = await program_llm.generate_code_variant(
                        context_from_evolution_loop=correction_context,
                        client=client,
                        current_prompt_string=current_prompt_data['prompt_string'],
                        current_delegation_depth=0
                    )

                    if corrected_code_string is None:
                        logger.warning("    Self-correction attempt %s/%s: LLM returned None. Aborting correction for this child." % (attempt + 1, max_correction_attempts))
                        break
                    
                    logger.log(VERBOSE_LEVEL_NUM, "Self-Correction Attempt %s/%s - Program (Parent: %s):\nCorrection Prompt (first 300 chars):\n%s...\nCorrected Code:\n%s",
                                    attempt + 1, max_correction_attempts, parent_id_for_child[:8] if parent_id_for_child else 'N/A',
                                    correction_prompt_used[:300], corrected_code_string)
                    logger.debug("    Self-correction attempt %s/%s: Received new code (first 200 chars):\n%s..." % (attempt + 1, max_correction_attempts, corrected_code_string[:200]))
                    temp_eval_results = evaluator.evaluate(corrected_code_string, main_cfg, problem_cfg, current_problem_dir, current_evaluator_data['evaluator_code_string'])

                    current_code_to_evaluate = corrected_code_string
                    current_eval_results = temp_eval_results
                    current_prompt_for_db = correction_prompt_used
                    
                    if temp_eval_results.get('is_valid', False):
                        logger.info("    Self-correction attempt %s/%s successful: Program is now valid." % (attempt + 1, max_correction_attempts))
                        break
                    else:
                        new_error_message = temp_eval_results.get('error_message', '')
                        if new_error_message == error_message:
                            logger.warning("    Self-correction attempt %s/%s: Error persisted unchanged: '%s'." %
                                                (attempt + 1, max_correction_attempts, new_error_message.splitlines()[0]))
                        else:
                            logger.info("    Self-correction attempt %s/%s: Error changed to: '%s'. Continuing if attempts left." %
                                             (attempt + 1, max_correction_attempts, new_error_message.splitlines()[0]))
                            error_message = new_error_message
                        
                        if attempt == max_correction_attempts - 1:
                            logger.warning("    Self-correction failed after %s attempts. Using last generated code for DB entry." % max_correction_attempts)
        
        if current_eval_results.get('is_valid'):
            child_code_length = current_eval_results.get('code_length')
            child_num_function_calls = current_eval_results.get('num_function_calls')

            child_program_id = program_db.add_program(
                code_string=current_code_to_evaluate,
                score=current_eval_results.get('score'),
                is_valid=current_eval_results.get('is_valid'),
                generation_discovered=gen,
                parent_id=parent_id_for_child,
                llm_prompt=current_prompt_for_db,
                evaluation_results=current_eval_results,
                descriptor_1=child_code_length,
                descriptor_2=child_num_function_calls,
                prompt_id=prompt_id_for_child,
                db_path=program_db_path
            )
            if child_program_id:
                logger.info("    Added new valid child program to DB: ID=%s, Score=%.4f" % (child_program_id[:8] if child_program_id else 'N/A', current_eval_results.get('score',0.0)))
                
                if program_db.get_map_elites_config().get('enable', False):
                    program_db.add_program_to_map_elites(
                        program_id=child_program_id,
                        score=current_eval_results.get('score'),
                        descriptor_values=[child_code_length, child_num_function_calls],
                        db_path=program_db_path
                    )
                
                prompt_data_for_update = prompt_db.get_prompt(prompt_id_for_child, db_path=prompt_db_path) # Corrected db_path
                if prompt_data_for_update:
                    current_prompt_score = prompt_data_for_update.get('score', 0.0)
                    new_prompt_score = max(current_prompt_score, current_eval_results.get('score', 0.0))
                    # prompt_db.update_prompt_score is not implemented yet, will need to add this.
                    # For now, we'll just add the prompt to map_elites if its score improves.
                    # The score update logic will be handled by the add_prompt_to_map_elites if it's a new elite.
                    logger.info("    Prompt %s score (current: %.4f, new program score: %.4f)" % (prompt_id_for_child[:8], current_prompt_score, current_eval_results.get('score', 0.0)))
                    if prompt_db.get_map_elites_config().get('enable', False):
                        prompt_db.add_prompt_to_map_elites(
                            prompt_id=prompt_id_for_child,
                            score=current_eval_results.get('score', 0.0), # Use the program's score to update prompt's score
                            descriptor_values=[prompt_data_for_update['descriptor_1'], prompt_data_for_update['descriptor_2']],
                            db_path=prompt_db_path # Corrected db_path
                        )

            else: # This else corresponds to 'if child_program_id:'
                logger.info("    Valid child program was a duplicate (by normalized hash), not added. Score=%.4f" % current_eval_results.get('score',0.0))
        else: # This else corresponds to 'if current_eval_results.get('is_valid'):'
            logger.info("    Child program code (after potential corrections) was invalid or failed. Not adding to DB. Error: %s" % current_eval_results.get('error_message', 'N/A'))

async def run_evaluator_evolution_phase(
    gen: int,
    main_cfg: Dict[str, Any],
    problem_cfg: Dict[str, Any],
    current_problem_dir: str,
    seed_evaluator_code_str: str,
    num_children_per_parent: int,
    num_parents_to_select: int,
    top_k_parents_pool_size: int,
    program_db_path: str,
    evaluator_db_path: str,
    client: httpx.AsyncClient
) -> None:
    """
    Executes the evaluator evolution phase for a given generation.
    Generates new evaluators, tests them against best programs, and adds them to the evaluator database.
    """
    logger = get_logger("EvolutionLoop.EvaluatorPhase")
    logger.info("\n--- Evaluator Evolution Phase ---")

    selected_parent_evaluators = selection.select_evaluators(
        num_evaluators=num_parents_to_select,
        top_k_to_consider=top_k_parents_pool_size,
        db_path=evaluator_db_path
    )
    if not selected_parent_evaluators:
        logger.warning("Generation %s: No parent evaluators selected. Using seed evaluator as current." % gen)
        current_evaluator_data = {"evaluator_code_string": seed_evaluator_code_str, "evaluator_id": "seed_evaluator", "challenge_score": 0.0}
    else:
        current_evaluator_data = selected_parent_evaluators[0]
        logger.info("Selected parent evaluator (ID: %s, Score: %.4f)" % (current_evaluator_data['evaluator_id'][:8], current_evaluator_data['challenge_score']))

    evaluator_generation_tasks = []
    best_programs_for_evaluator_challenge = program_db.get_best_n_programs(n=top_k_parents_pool_size, db_path=program_db_path)

    for i in range(num_children_per_parent):
        dynamic_evaluator_context = {
            'parent_evaluator_code': current_evaluator_data['evaluator_code_string'],
            'previous_error_feedback': None,
            'best_programs_from_generator': best_programs_for_evaluator_challenge
        }
        task = evaluator_llm.generate_evaluator_logic(
            context_from_evolution_loop=dynamic_evaluator_context,
            client=client,
            current_delegation_depth=0
        )
        evaluator_generation_tasks.append(task)
    
    generated_evaluator_results = await asyncio.gather(*evaluator_generation_tasks)

    for new_evaluator_code_string, prompt_used_for_evaluator in generated_evaluator_results:
        if new_evaluator_code_string:
            challenge_scores = []
            for program_data in best_programs_for_evaluator_challenge:
                program_code = program_data['code_string']
                eval_res = evaluator.evaluate(program_code, main_cfg, problem_cfg, current_problem_dir, new_evaluator_code_string)
                challenge_scores.append(1.0 - eval_res.get('score', 0.0))
            
            avg_challenge_score = sum(challenge_scores) / len(challenge_scores) if challenge_scores else 0.0

            evaluator_length = len(new_evaluator_code_string.splitlines())
            evaluator_test_cases = new_evaluator_code_string.count("test_case")

            new_evaluator_id = evaluator_db.add_evaluator(
                evaluator_code_string=new_evaluator_code_string,
                challenge_score=avg_challenge_score,
                generation_discovered=gen,
                parent_evaluator_id=current_evaluator_data['evaluator_id'],
                llm_prompt=prompt_used_for_evaluator,
                evaluation_results={"challenge_score_details": challenge_scores},
                descriptor_1=evaluator_length,
                descriptor_2=evaluator_test_cases,
                db_path=evaluator_db_path
            )
            if new_evaluator_id:
                logger.info("Added new evaluator to DB: ID=%s, Challenge Score: %.4f" % (new_evaluator_id[:8], avg_challenge_score))
                if evaluator_db.get_map_elites_config().get('enable', False):
                    evaluator_db.add_evaluator_to_map_elites(
                        evaluator_id=new_evaluator_id,
                        challenge_score=avg_challenge_score,
                        descriptor_values=[evaluator_length, evaluator_test_cases],
                        db_path=evaluator_db_path
                    )
        else:
            logger.warning("Evaluator generation failed.")
            # Log the content that failed to be generated
            if new_evaluator_code_string:
                logger.warning(f"Failed evaluator code content:\n{new_evaluator_code_string[:500]}...")
            else:
                logger.warning("Evaluator generation failed: LLM returned no content.")