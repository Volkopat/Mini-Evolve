import random
from . import program_db
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

# --- Main for Testing ---
if __name__ == "__main__":
    import os
    print("Running tests for selection.py...")
    
    # Ensure a clean database for testing
    TEST_DB_PATH = "test_selection_db.sqlite"
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    program_db.init_db(db_path=TEST_DB_PATH)

    # Add some sample programs for testing
    # program_id, code_string, normalized_code_string, normalized_code_hash, score, is_valid, generation, parent_id, llm_prompt, eval_results_json, timestamp
    programs_to_add = [
        ("def p1(): return 0.9", 0.9, True, 1, None, "prompt1", {"fitness": 0.9}),
        ("def p2(): return 0.8", 0.8, True, 1, None, "prompt2", {"fitness": 0.8}),
        ("def p3(): return 0.7", 0.7, True, 1, None, "prompt3", {"fitness": 0.7}),
        ("def p4(): return 0.6", 0.6, True, 1, None, "prompt4", {"fitness": 0.6}),
        ("def p5(): return 0.5", 0.5, True, 1, None, "prompt5", {"fitness": 0.5}),
        ("def p1_dup(): return 0.9", 0.9, True, 1, None, "prompt1_dup", {"fitness": 0.9}), # Duplicate normalized hash of p1
        ("def p6_invalid(): return 0.4", 0.0, False, 1, None, "prompt6", {"fitness": 0.0, "error": "invalid"}), # Invalid
    ]

    print(f"\nPopulating test database: {TEST_DB_PATH}")
    added_ids = []
    for i, p_data in enumerate(programs_to_add):
        pid = program_db.add_program(
            code_string=p_data[0],
            score=p_data[1],
            is_valid=p_data[2],
            generation_discovered=p_data[3],
            parent_id=p_data[4],
            llm_prompt=p_data[5],
            evaluation_results=p_data[6],
            db_path=TEST_DB_PATH
        )
        if pid:
            print(f"Added: {p_data[0][:15]}... Score: {p_data[1]}, Valid: {p_data[2]}, ID: {pid[:8]}")
            added_ids.append(pid)
        else:
            print(f"NOT Added (likely duplicate): {p_data[0][:15]}... Score: {p_data[1]}")
    
    # Expected unique valid programs: p1 (0.9), p2 (0.8), p3 (0.7), p4 (0.6), p5 (0.5)
    # Total 5 unique valid programs

    print("\n--- Test Case 1: Select 2 parents from top 5 consideration pool ---")
    parents1 = select_parents(num_parents=2, top_k_to_consider=5, db_path=TEST_DB_PATH)
    print(f"Selected {len(parents1)} parents:")
    for p in parents1:
        print(f"  ID: {p['program_id'][:8]}, Score: {p['score']}, Code: {p['code_string']}")
    assert len(parents1) == 2
    # Check scores are from the top candidates (though random.sample won't guarantee order)
    parent_scores1 = sorted([p['score'] for p in parents1], reverse=True)
    assert parent_scores1[0] <= 0.9 and parent_scores1[0] >= 0.5 
    assert parent_scores1[1] <= 0.9 and parent_scores1[1] >= 0.5

    print("\n--- Test Case 2: Select 3 parents from top 3 consideration pool ---")
    # Pool should contain p1(0.9), p2(0.8), p3(0.7)
    parents2 = select_parents(num_parents=3, top_k_to_consider=3, db_path=TEST_DB_PATH)
    print(f"Selected {len(parents2)} parents:")
    for p in parents2:
        print(f"  ID: {p['program_id'][:8]}, Score: {p['score']}, Code: {p['code_string']}")
    assert len(parents2) == 3
    parent_scores2 = sorted([p['score'] for p in parents2], reverse=True)
    # Expected scores from p1 (0.9), p1_dup (0.9), and p2 (0.8) if top_k_to_consider=3 includes them
    assert parent_scores2 == [0.9, 0.9, 0.8]

    print("\n--- Test Case 3: Select more parents than available unique programs ---")
    # We have 5 unique valid programs. Try to select 7.
    parents3 = select_parents(num_parents=7, top_k_to_consider=10, db_path=TEST_DB_PATH)
    print(f"Selected {len(parents3)} parents (requested 7, pool had <7 unique valid):")
    for p in parents3:
        print(f"  ID: {p['program_id'][:8]}, Score: {p['score']}, Code: {p['code_string']}")
    assert len(parents3) == 6 # Should return all 6 unique valid programs

    print("\n--- Test Case 4: Request 0 parents ---")
    parents4 = select_parents(num_parents=0, top_k_to_consider=5, db_path=TEST_DB_PATH)
    print(f"Selected {len(parents4)} parents (requested 0):")
    assert len(parents4) == 0

    print("\n--- Test Case 5: Empty Database ---")
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    program_db.init_db(db_path=TEST_DB_PATH) # Fresh empty DB
    parents5 = select_parents(num_parents=2, top_k_to_consider=5, db_path=TEST_DB_PATH)
    print(f"Selected {len(parents5)} parents from empty DB:")
    assert len(parents5) == 0

    # Clean up test database
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    print(f"\nCleaned up test database: {TEST_DB_PATH}")
    print("\nselection.py tests finished.") 