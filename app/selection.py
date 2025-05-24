import random
import sqlite3 # Added for direct DB access for MAP-Elites
from . import program_db
from . import prompt_db # New: Import prompt_db
from . import evaluator_db # New: Import evaluator_db
from .logger_setup import get_logger # Import logger

# --- Configuration (can be moved to config.yaml later if needed) ---
# These are example values. The actual values will typically be read from a central config.
DEFAULT_NUM_PARENTS_TO_SELECT = 2
DEFAULT_TOP_K_TO_CONSIDER = 10 

def select_parents(num_parents: int = DEFAULT_NUM_PARENTS_TO_SELECT, 
                   top_k_to_consider: int = DEFAULT_TOP_K_TO_CONSIDER,
                   db_path: str | None = None) -> list[dict]:
    """Selects a list of parent programs for the next generation.

    Strategy: Truncation + Random Selection
    1. Fetches `top_k_to_consider` best unique *valid* programs from the database.
    2. If no valid programs are found, it falls back to fetching the top `top_k_to_consider`
       best unique programs regardless of validity.
    3. Randomly selects `num_parents` from this pool.
    
    Args:
        num_parents: The number of parent programs to select.
        top_k_to_consider: The size of the pool of best unique programs to draw from.
        db_path: Optional path to the database. If None, uses default from program_db.

    Returns:
        A list of program dictionaries, each representing a selected parent.
        Returns an empty list if no suitable parents can be found even with fallback.
    """
    logger = get_logger("Selection") # Get logger instance
    if db_path is None:
        db_path = program_db.get_database_path()

    map_elites_enabled = program_db.get_map_elites_config().get('enable', False)

    if map_elites_enabled:
        logger.info("MAP-Elites enabled. Selecting parents from the MAP-Elites grid.")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Fetch all program_ids from the map_elites_grid
            cursor.execute("SELECT program_id FROM map_elites_grid")
            elite_ids = [row['program_id'] for row in cursor.fetchall()]
            
            candidate_pool = []
            for prog_id in elite_ids:
                program_data = program_db.get_program(prog_id, db_path=db_path)
                if program_data and program_data.get('is_valid'): # Only consider valid elites for selection
                    candidate_pool.append(program_data)
            
            logger.info("Found %s valid elites in the MAP-Elites grid." % len(candidate_pool))

        except sqlite3.OperationalError as e:
            logger.error(f"Error querying map_elites_grid: {e}. Falling back to score-based selection.")
            map_elites_enabled = False # Disable MAP-Elites for this run if table not found
        finally:
            conn.close()

    if not map_elites_enabled: # Fallback to original selection if MAP-Elites not enabled or failed
        logger.info("MAP-Elites not enabled or failed. Selecting parents based on score.")
        # Attempt to get a pool of high-quality, unique, *valid* programs first
        candidate_pool = program_db.get_unique_programs_from_top_k(
            k_to_select=top_k_to_consider,
            top_n_to_consider=top_k_to_consider * 2, # Look deeper to ensure we find enough unique valid ones
            db_path=db_path,
            only_valid=True # Explicitly request only valid programs
        )

        if not candidate_pool:
            logger.warning("No *valid* candidate programs found. Attempting to select from all programs (including invalid).")
            # Fallback: try to get programs regardless of validity
            candidate_pool = program_db.get_unique_programs_from_top_k(
                k_to_select=top_k_to_consider,
                top_n_to_consider=top_k_to_consider * 2,
                db_path=db_path,
                only_valid=False # Request programs regardless of validity
            )
            if not candidate_pool:
                logger.warning("No programs found in the database at all for parent selection (even invalid ones).")
                return []
            else:
                logger.info("Found %s programs (including invalid) for parent selection pool." % len(candidate_pool))

    if not candidate_pool: # Should not happen if the second call worked, but as a safeguard
        logger.error("Unexpected: No candidate programs found even after fallback. This indicates an empty DB or DB error.")
        return []

    # If the pool is smaller than the number of parents we want to select,
    # we return the entire pool (shuffled for some randomness if a choice has to be made by caller).
    if len(candidate_pool) < num_parents:
        logger.warning("Candidate pool size (%s) is less than num_parents (%s). Returning all candidates." % (len(candidate_pool), num_parents))
        random.shuffle(candidate_pool) # Shuffle to ensure random pick if caller takes a subset
        return candidate_pool
    
    # Randomly select num_parents from the candidate pool
    selected_parents = random.sample(candidate_pool, num_parents)
    logger.info("Selected %s parents from a pool of %s candidates." % (len(selected_parents), len(candidate_pool)))
    
    return selected_parents

def select_prompts(num_prompts: int = DEFAULT_NUM_PARENTS_TO_SELECT,
                   top_k_to_consider: int = DEFAULT_TOP_K_TO_CONSIDER,
                   db_path: str | None = None) -> list[dict]:
    """Selects a list of parent prompts for the next generation.

    Strategy: Truncation + Random Selection
    1. Fetches `top_k_to_consider` best unique prompts from the database.
    2. Randomly selects `num_prompts` from this pool.
    
    Args:
        num_prompts: The number of parent prompts to select.
        top_k_to_consider: The size of the pool of best unique prompts to draw from.
        db_path: Optional path to the database. If None, uses default from prompt_db.

    Returns:
        A list of prompt dictionaries, each representing a selected parent.
        Returns an empty list if no suitable prompts can be found.
    """
    logger = get_logger("Selection.Prompts")
    if db_path is None:
        db_path = prompt_db.get_database_path()

    map_elites_enabled = prompt_db.get_map_elites_config().get('enable', False)

    if map_elites_enabled:
        logger.info("MAP-Elites enabled. Selecting prompts from the MAP-Elites grid.")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT prompt_id FROM map_elites_grid")
            elite_ids = [row['prompt_id'] for row in cursor.fetchall()]
            
            candidate_pool = []
            for prompt_id in elite_ids:
                prompt_data = prompt_db.get_prompt(prompt_id, db_path=db_path)
                if prompt_data: # All prompts are "valid" in a sense, just have scores
                    candidate_pool.append(prompt_data)
            
            logger.info("Found %s elites in the MAP-Elites grid for prompts." % len(candidate_pool))

        except sqlite3.OperationalError as e:
            logger.error(f"Error querying map_elites_grid for prompts: {e}. Falling back to score-based selection.")
            map_elites_enabled = False
        finally:
            conn.close()

    if not map_elites_enabled:
        logger.info("MAP-Elites not enabled or failed. Selecting prompts based on score.")
        candidate_pool = prompt_db.get_unique_prompts_from_top_k(
            k_to_select=top_k_to_consider,
            top_n_to_consider=top_k_to_consider * 2,
            db_path=db_path
        )
        if not candidate_pool:
            logger.warning("No prompts found in the database at all for parent selection.")
            return []
        else:
            logger.info("Found %s prompts for parent selection pool." % len(candidate_pool))

    if not candidate_pool:
        logger.error("Unexpected: No candidate prompts found even after fallback. This indicates an empty DB or DB error.")
        return []

    if len(candidate_pool) < num_prompts:
        logger.warning("Candidate prompt pool size (%s) is less than num_prompts (%s). Returning all candidates." % (len(candidate_pool), num_prompts))
        random.shuffle(candidate_pool)
        return candidate_pool
    
    selected_prompts = random.sample(candidate_pool, num_prompts)
    logger.info("Selected %s prompts from a pool of %s candidates." % (len(selected_prompts), len(candidate_pool)))
    
def select_evaluators(num_evaluators: int = DEFAULT_NUM_PARENTS_TO_SELECT,
                      top_k_to_consider: int = DEFAULT_TOP_K_TO_CONSIDER,
                      db_path: str | None = None) -> list[dict]:
    """Selects a list of parent evaluators for the next generation.

    Strategy: Truncation + Random Selection
    1. Fetches `top_k_to_consider` best unique evaluators from the database.
    2. Randomly selects `num_evaluators` from this pool.
    
    Args:
        num_evaluators: The number of parent evaluators to select.
        top_k_to_consider: The size of the pool of best unique evaluators to draw from.
        db_path: Optional path to the database. If None, uses default from evaluator_db.

    Returns:
        A list of evaluator dictionaries, each representing a selected parent.
        Returns an empty list if no suitable evaluators can be found.
    """
    logger = get_logger("Selection.Evaluators")
    if db_path is None:
        db_path = evaluator_db.get_database_path()

    map_elites_enabled = evaluator_db.get_map_elites_config().get('enable', False)

    if map_elites_enabled:
        logger.info("MAP-Elites enabled. Selecting evaluators from the MAP-Elites grid.")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT evaluator_id FROM map_elites_grid")
            elite_ids = [row['evaluator_id'] for row in cursor.fetchall()]
            
            candidate_pool = []
            for evaluator_id in elite_ids:
                evaluator_data = evaluator_db.get_evaluator(evaluator_id, db_path=db_path)
                if evaluator_data:
                    candidate_pool.append(evaluator_data)
            
            logger.info("Found %s elites in the MAP-Elites grid for evaluators." % len(candidate_pool))

        except sqlite3.OperationalError as e:
            logger.error(f"Error querying map_elites_grid for evaluators: {e}. Falling back to score-based selection.")
            map_elites_enabled = False
        finally:
            conn.close()

    if not map_elites_enabled:
        logger.info("MAP-Elites not enabled or failed. Selecting evaluators based on score.")
        candidate_pool = evaluator_db.get_unique_evaluators_from_top_k(
            k_to_select=top_k_to_consider,
            top_n_to_consider=top_k_to_consider * 2,
            db_path=db_path
        )
        if not candidate_pool:
            logger.warning("No evaluators found in the database at all for parent selection.")
            return []
        else:
            logger.info("Found %s evaluators for parent selection pool." % len(candidate_pool))

    if not candidate_pool:
        logger.error("Unexpected: No candidate evaluators found even after fallback. This indicates an empty DB or DB error.")
        return []

    if len(candidate_pool) < num_evaluators:
        logger.warning("Candidate evaluator pool size (%s) is less than num_evaluators (%s). Returning all candidates." % (len(candidate_pool), num_evaluators))
        random.shuffle(candidate_pool)
        return candidate_pool
    
    selected_evaluators = random.sample(candidate_pool, num_evaluators)
    logger.info("Selected %s evaluators from a pool of %s candidates." % (len(selected_evaluators), len(candidate_pool)))
    
    return selected_evaluators


# --- Main for Testing ---
if __name__ == "__main__":
    import os
    from . import prompt_db # Import prompt_db for testing
    from . import evaluator_db # New: Import evaluator_db for testing
    print("Running tests for selection.py...")
    
    # Ensure a clean database for programs
    TEST_PROGRAM_DB_PATH = "test_selection_program_db.sqlite"
    if os.path.exists(TEST_PROGRAM_DB_PATH):
        os.remove(TEST_PROGRAM_DB_PATH)
    program_db.init_db(db_path=TEST_PROGRAM_DB_PATH)

    # Ensure a clean database for prompts
    TEST_PROMPT_DB_PATH = "test_selection_prompt_db.sqlite"
    if os.path.exists(TEST_PROMPT_DB_PATH):
        os.remove(TEST_PROMPT_DB_PATH)
    prompt_db.init_db(db_path=TEST_PROMPT_DB_PATH)

    # Ensure a clean database for evaluators
    TEST_EVALUATOR_DB_PATH = "test_selection_evaluator_db.sqlite"
    if os.path.exists(TEST_EVALUATOR_DB_PATH):
        os.remove(TEST_EVALUATOR_DB_PATH)
    evaluator_db.init_db(db_path=TEST_EVALUATOR_DB_PATH)

    # Add some sample programs for testing select_parents
    programs_to_add = [
        ("def p1(): return 0.9", 0.9, True, 1, None, "prompt1", {"fitness": 0.9}),
        ("def p2(): return 0.8", 0.8, True, 1, None, "prompt2", {"fitness": 0.8}),
        ("def p3(): return 0.7", 0.7, True, 1, None, "prompt3", {"fitness": 0.7}),
        ("def p4(): return 0.6", 0.6, True, 1, None, "prompt4", {"fitness": 0.6}),
        ("def p5(): return 0.5", 0.5, True, 1, None, "prompt5", {"fitness": 0.5}),
        ("def p1_dup(): return 0.9", 0.9, True, 1, None, "prompt1_dup", {"fitness": 0.9}),
        ("def p6_invalid(): return 0.4", 0.0, False, 1, None, "prompt6", {"fitness": 0.0, "error": "invalid"}),
    ]

    print(f"\nPopulating test program database: {TEST_PROGRAM_DB_PATH}")
    added_program_ids = []
    for i, p_data in enumerate(programs_to_add):
        pid = program_db.add_program(
            code_string=p_data[0],
            score=p_data[1],
            is_valid=p_data[2],
            generation_discovered=p_data[3],
            parent_id=p_data[4],
            llm_prompt=p_data[5],
            evaluation_results=p_data[6],
            db_path=TEST_PROGRAM_DB_PATH
        )
        if pid:
            print(f"Added: {p_data[0][:15]}... Score: {p_data[1]}, Valid: {p_data[2]}, ID: {pid[:8]}")
            added_program_ids.append(pid)
        else:
            print(f"NOT Added (likely duplicate): {p_data[0][:15]}... Score: {p_data[1]}")
    
    print("\n--- Test Case: select_parents (for programs) ---")
    parents1 = select_parents(num_parents=2, top_k_to_consider=5, db_path=TEST_PROGRAM_DB_PATH)
    print(f"Selected {len(parents1)} program parents:")
    for p in parents1:
        print(f"  ID: {p['program_id'][:8]}, Score: {p['score']}, Code: {p['code_string']}")
    assert len(parents1) == 2
    parent_scores1 = sorted([p['score'] for p in parents1], reverse=True)
    assert parent_scores1[0] <= 0.9 and parent_scores1[0] >= 0.5
    assert parent_scores1[1] <= 0.9 and parent_scores1[1] >= 0.5

    # Add some sample prompts for testing select_prompts
    prompts_to_add = [
        ("Prompt A: Sort list.", 0.9, 1, None, "llm_gen_A", {"eval": "good"}),
        ("Prompt B: Find max element.", 0.8, 1, None, "llm_gen_B", {"eval": "ok"}),
        ("Prompt C: Reverse string.", 0.7, 1, None, "llm_gen_C", {"eval": "avg"}),
        ("Prompt A: Sort list.", 0.9, 2, None, "llm_gen_A_dup", {"eval": "good"}), # Duplicate
    ]

    print(f"\nPopulating test prompt database: {TEST_PROMPT_DB_PATH}")
    added_prompt_ids = []
    for i, p_data in enumerate(prompts_to_add):
        pid = prompt_db.add_prompt(
            prompt_string=p_data[0],
            score=p_data[1],
            generation_discovered=p_data[2],
            parent_prompt_id=p_data[3],
            llm_prompt=p_data[4],
            evaluation_results=p_data[5],
            db_path=TEST_PROMPT_DB_PATH
        )
        if pid:
            print(f"Added: {p_data[0][:15]}... Score: {p_data[1]}, ID: {pid[:8]}")
            added_prompt_ids.append(pid)
        else:
            print(f"NOT Added (likely duplicate): {p_data[0][:15]}... Score: {p_data[1]}")

    print("\n--- Test Case: select_prompts ---")
    selected_prompts = select_prompts(num_prompts=2, top_k_to_consider=5, db_path=TEST_PROMPT_DB_PATH)
    print(f"Selected {len(selected_prompts)} prompt parents:")
    for p in selected_prompts:
        print(f"  ID: {p['prompt_id'][:8]}, Score: {p['score']}, Prompt: {p['prompt_string']}")
    assert len(selected_prompts) == 2
    selected_prompt_scores = sorted([p['score'] for p in selected_prompts], reverse=True)
    assert selected_prompt_scores[0] == 0.9
    assert selected_prompt_scores[1] == 0.8

    # Add some sample evaluators for testing select_evaluators
    evaluators_to_add = [
        ("def eval1(): return 0.9", 0.9, 1, None, "llm_eval_1", {"eval": "hard"}),
        ("def eval2(): return 0.8", 0.8, 1, None, "llm_eval_2", {"eval": "medium"}),
        ("def eval3(): return 0.7", 0.7, 1, None, "llm_eval_3", {"eval": "easy"}),
        ("def eval1(): return 0.9", 0.9, 2, None, "llm_eval_1_dup", {"eval": "hard"}), # Duplicate
    ]

    print(f"\nPopulating test evaluator database: {TEST_EVALUATOR_DB_PATH}")
    added_evaluator_ids = []
    for i, e_data in enumerate(evaluators_to_add):
        eid = evaluator_db.add_evaluator(
            evaluator_code_string=e_data[0],
            challenge_score=e_data[1],
            generation_discovered=e_data[2],
            parent_evaluator_id=e_data[3],
            llm_prompt=e_data[4],
            evaluation_results=e_data[5],
            db_path=TEST_EVALUATOR_DB_PATH
        )
        if eid:
            print(f"Added: {e_data[0][:15]}... Score: {e_data[1]}, ID: {eid[:8]}")
            added_evaluator_ids.append(eid)
        else:
            print(f"NOT Added (likely duplicate): {e_data[0][:15]}... Score: {e_data[1]}")

    print("\n--- Test Case: select_evaluators ---")
    selected_evaluators = select_evaluators(num_evaluators=2, top_k_to_consider=5, db_path=TEST_EVALUATOR_DB_PATH)
    print(f"Selected {len(selected_evaluators)} evaluator parents:")
    for e in selected_evaluators:
        print(f"  ID: {e['evaluator_id'][:8]}, Score: {e['challenge_score']}, Evaluator: {e['evaluator_code_string']}")
    assert len(selected_evaluators) == 2
    selected_evaluator_scores = sorted([e['challenge_score'] for e in selected_evaluators], reverse=True)
    assert selected_evaluator_scores[0] == 0.9
    assert selected_evaluator_scores[1] == 0.8

    # Clean up all test databases
    if os.path.exists(TEST_PROGRAM_DB_PATH):
        os.remove(TEST_PROGRAM_DB_PATH)
    if os.path.exists(TEST_PROMPT_DB_PATH):
        os.remove(TEST_PROMPT_DB_PATH)
    if os.path.exists(TEST_EVALUATOR_DB_PATH):
        os.remove(TEST_EVALUATOR_DB_PATH)
    print(f"\nCleaned up test databases: {TEST_PROGRAM_DB_PATH}, {TEST_PROMPT_DB_PATH}, {TEST_EVALUATOR_DB_PATH}")
    print("\nselection.py tests finished.")
