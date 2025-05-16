import sqlite3
import hashlib
import ast
import time
import json # For storing dicts like evaluation_metrics
import uuid
import os # Added for deleting DB file in test

# --- Configuration (could also be moved to main config.yaml if preferred) ---
# Load from main config.yaml instead
import yaml
CONFIG_FILE = "config/config.yaml"

def load_db_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('database', {})
    except FileNotFoundError:
        print(f"Warning: Main configuration file {CONFIG_FILE} not found. Using default DB path.")
        return {}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing YAML configuration: {e}. Using default DB path.")
        return {}

db_config = load_db_config()
DATABASE_PATH = db_config.get('path', 'program_database.db')

# --- Code Normalization ---
def normalize_code(code_string: str) -> str:
    """Normalizes Python code by parsing and unparsing it using AST."""
    try:
        parsed_ast = ast.parse(code_string)
        return ast.unparse(parsed_ast) # Requires Python 3.9+
    except SyntaxError:
        # If code is not valid Python, return it as is (or handle error)
        # Hashing will still be different for syntactically incorrect but textually different strings.
        return code_string 
    except Exception as e:
        print(f"Warning: AST normalization failed: {e}. Returning original code string.")
        return code_string

def get_code_hash(code_string: str) -> str:
    """Generates a SHA256 hash for a code string."""
    return hashlib.sha256(code_string.encode('utf-8')).hexdigest()

# --- Database Initialization ---
def init_db(db_path=DATABASE_PATH):
    """Initializes the database and creates the programs table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS programs (
        program_id TEXT PRIMARY KEY,
        code_string TEXT NOT NULL,
        normalized_code_string TEXT NOT NULL, 
        normalized_code_hash TEXT NOT NULL UNIQUE, -- Ensures uniqueness based on normalized code
        score REAL,
        is_valid BOOLEAN,
        generation_discovered INTEGER,
        parent_id TEXT, -- Can be NULL for seed programs
        llm_prompt TEXT, -- The full prompt sent to the LLM
        evaluation_results_json TEXT, -- Store the full dict from evaluator.py
        timestamp_added REAL NOT NULL
    )
    """)
    # Add index for faster lookups on normalized_code_hash
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_normalized_code_hash ON programs (normalized_code_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_score ON programs (score DESC)") # For get_best_n
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

# --- API Functions ---
def add_program(code_string: str,
                score: float | None,
                is_valid: bool,
                generation_discovered: int | None,
                parent_id: str | None,
                llm_prompt: str | None, # The prompt used to generate this program
                evaluation_results: dict, # Full results from evaluator.py
                db_path=DATABASE_PATH) -> str | None:
    """Adds a new program to the database if its normalized form doesn't already exist.
    Returns the program_id if added, None otherwise (e.g., if duplicate or error).
    """
    
    original_code_string = code_string # Keep a copy before normalization for storage
    normalized_code = normalize_code(code_string)
    normalized_hash = get_code_hash(normalized_code)

    if check_if_exists(normalized_hash, db_path=db_path):
        # print(f"Program with hash {normalized_hash} already exists. Not adding.")
        return None # Indicate that the program was a duplicate by normalized hash

    program_id = str(uuid.uuid4())
    timestamp = time.time()
    evaluation_results_str = json.dumps(evaluation_results) # Serialize the dict to JSON string

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO programs (
            program_id, code_string, normalized_code_string, normalized_code_hash, 
            score, is_valid, generation_discovered, parent_id, llm_prompt, 
            evaluation_results_json, timestamp_added
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (program_id, original_code_string, normalized_code, normalized_hash, 
              score, is_valid, generation_discovered, parent_id, llm_prompt,
              evaluation_results_str, timestamp))
        conn.commit()
        # print(f"Added program {program_id} with hash {normalized_hash}")
        return program_id
    except sqlite3.IntegrityError as e:
        # This could happen if by some race condition, another process inserted the same hash
        # between the check_if_exists and the insert. Or if program_id wasn't unique (unlikely with UUID4).
        print(f"SQLite IntegrityError while adding program: {e}. Hash: {normalized_hash}")
        return None 
    except Exception as e:
        print(f"Error adding program to DB: {e}")
        conn.rollback() # Rollback any partial changes on other errors
        return None
    finally:
        conn.close()

def check_if_exists(normalized_code_hash: str, db_path=DATABASE_PATH) -> bool:
    """Checks if a program with the given normalized_code_hash already exists."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM programs WHERE normalized_code_hash = ?", (normalized_code_hash,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_program(program_id: str, db_path=DATABASE_PATH) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row # Access columns by name
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM programs WHERE program_id = ?", (program_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        program_data = dict(row)
        # Explicitly convert is_valid from integer (0 or 1) to boolean
        if 'is_valid' in program_data and isinstance(program_data['is_valid'], int):
            program_data['is_valid'] = bool(program_data['is_valid'])
        
        try: # Deserialize JSON fields
            program_data['evaluation_results'] = json.loads(program_data['evaluation_results_json'])
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not parse evaluation_results_json for program {program_id}: {e}")
            program_data['evaluation_results'] = {} # Fallback to empty dict
        return program_data
    return None

# --- Additional Query Functions for Selection ---

def get_best_n_programs(n: int, min_score: float | None = None, db_path=DATABASE_PATH) -> list[dict]:
    """Retrieves the top N programs, optionally above a minimum score, ordered by score DESC.
    Returns a list of program dictionaries.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM programs WHERE is_valid = 1" # Only consider valid programs
    params = []
    
    if min_score is not None:
        query += " AND score >= ?"
        params.append(min_score)
        
    query += " ORDER BY score DESC LIMIT ?"
    params.append(n)
    
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    
    programs = []
    for row in rows:
        program_data = dict(row)
        if 'is_valid' in program_data and isinstance(program_data['is_valid'], int):
            program_data['is_valid'] = bool(program_data['is_valid'])
        try:
            program_data['evaluation_results'] = json.loads(program_data['evaluation_results_json'])
        except (json.JSONDecodeError, TypeError):
            program_data['evaluation_results'] = {}
        programs.append(program_data)
    return programs

def get_unique_programs_from_top_k(k_to_select: int, top_n_to_consider: int, db_path=DATABASE_PATH, only_valid: bool = True) -> list[dict]:
    """
    Retrieves k_to_select unique programs (based on normalized_code_hash)
    by first considering the top_n_to_consider best scoring programs.
    If only_valid is True, only considers valid programs.
    This aims to get diverse, high-quality programs.
    Returns a list of program dictionaries.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    base_query = "SELECT * FROM programs"
    params = []

    if only_valid:
        base_query += " WHERE is_valid = 1"
    
    base_query += " ORDER BY score DESC, timestamp_added DESC LIMIT ?"
    params.append(top_n_to_consider)
    
    cursor.execute(base_query, tuple(params))
    
    potential_programs_rows = cursor.fetchall()
    conn.close()

    selected_programs = []
    seen_normalized_hashes = set()

    for row in potential_programs_rows:
        if len(selected_programs) >= k_to_select:
            break # Found enough unique programs

        program_data = dict(row)
        normalized_hash = program_data['normalized_code_hash']

        if normalized_hash not in seen_normalized_hashes:
            if 'is_valid' in program_data and isinstance(program_data['is_valid'], int):
                program_data['is_valid'] = bool(program_data['is_valid'])
            try:
                program_data['evaluation_results'] = json.loads(program_data['evaluation_results_json'])
            except (json.JSONDecodeError, TypeError):
                program_data['evaluation_results'] = {}
            
            selected_programs.append(program_data)
            seen_normalized_hashes.add(normalized_hash)
            
    return selected_programs

# --- Helper to get config for other modules that might need DB path ---
def get_database_path() -> str:
    return DATABASE_PATH

# --- Main for Testing ---
if __name__ == "__main__":
    # Ensure a clean slate for testing by removing the old DB if it exists
    if os.path.exists(DATABASE_PATH):
        try:
            os.remove(DATABASE_PATH)
            print(f"Removed existing database file: {DATABASE_PATH}")
        except OSError as e:
            print(f"Error removing database file {DATABASE_PATH}: {e}. Please check permissions or close other connections.")
            # Depending on desired behavior, you might want to exit or raise here

    print(f"Using database at: {DATABASE_PATH}")
    init_db() # Ensure table is created

    # Test Data
    test_code_1 = """def solve():
    x = 1
    y = 2
    z = 3
    return (x*y)+z # Result 5"""
    test_code_1_slightly_different_formatting = """def solve():
    x=1
    y=2
    z=3
    return (x * y) + z # Still 5"""
    test_code_2 = """def solve():
    x = 10
    y = 20
    z = 30
    return (x*y)+z # Result 230"""
    broken_code = """def solve():
    x=1
    y=2
    z=3
    return x y z"""

    eval_res_1 = {'score': 0.1, 'is_valid': True, 'error_message': None, 'execution_time_ms': 10.0}
    eval_res_2 = {'score': 0.5, 'is_valid': True, 'error_message': None, 'execution_time_ms': 12.0}
    eval_res_broken = {'score': 0.0, 'is_valid': False, 'error_message': 'SyntaxError', 'execution_time_ms': 5.0}

    print("\n--- Testing add_program and check_if_exists ---")
    # Add first program
    prog1_id = add_program(test_code_1, eval_res_1['score'], eval_res_1['is_valid'], 0, None, "prompt1", eval_res_1)
    print(f"Added program 1: ID = {prog1_id}")
    assert prog1_id is not None

    # Try adding same program with different formatting (should be detected as duplicate)
    prog1_dup_id = add_program(test_code_1_slightly_different_formatting, eval_res_1['score'], eval_res_1['is_valid'], 0, None, "prompt1_dup", eval_res_1)
    print(f"Attempted to add duplicate of program 1 (diff format): ID = {prog1_dup_id}")
    assert prog1_dup_id is None # Should be None because it's a duplicate by normalized hash

    # Add second, different program
    prog2_id = add_program(test_code_2, eval_res_2['score'], eval_res_2['is_valid'], 0, prog1_id, "prompt2", eval_res_2)
    print(f"Added program 2: ID = {prog2_id}")
    assert prog2_id is not None

    # Add broken program
    prog_broken_id = add_program(broken_code, eval_res_broken['score'], eval_res_broken['is_valid'], 1, None, "prompt_broken", eval_res_broken)
    print(f"Added broken program: ID = {prog_broken_id}")
    assert prog_broken_id is not None # Should add fine, just has low score/is_invalid
    
    # Test check_if_exists directly
    norm_hash_1 = get_code_hash(normalize_code(test_code_1))
    assert check_if_exists(norm_hash_1) is True
    norm_hash_non_existent = get_code_hash(normalize_code("def non_existent(): pass"))
    assert check_if_exists(norm_hash_non_existent) is False
    print("check_if_exists tests passed.")

    print("\n--- Testing get_program ---")
    if prog1_id:
        retrieved_prog1 = get_program(prog1_id)
        print(f"Retrieved program 1: {retrieved_prog1['program_id']}, Score: {retrieved_prog1['score']}")
        assert retrieved_prog1 is not None
        assert retrieved_prog1['code_string'] == test_code_1
        assert retrieved_prog1['evaluation_results']['score'] == eval_res_1['score']

    if prog_broken_id:
        retrieved_broken = get_program(prog_broken_id)
        print(f"Retrieved broken program: {retrieved_broken['program_id']}, Valid: {retrieved_broken['is_valid']}")
        assert retrieved_broken['is_valid'] is False
        assert retrieved_broken['evaluation_results']['error_message'] == 'SyntaxError'

    print("\n--- Testing get_best_n_programs and get_unique_programs_from_top_k ---")

    # Re-add some programs for these tests, ensuring some variety in scores and some duplicates
    # To make tests deterministic, we'll clear and re-init for this section too.
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
    init_db()

    prog_A_s08_vT_id = add_program("def solveA(): return 8", 0.8, True, 1, None, "pA", {"detail": "A"})
    prog_B_s09_vT_id = add_program("def solveB(): return 9", 0.9, True, 1, None, "pB", {"detail": "B"})
    prog_C_s07_vT_id = add_program("def solveC(): return 7", 0.7, True, 1, None, "pC", {"detail": "C"})
    prog_D_s09_vT_id_dupB = add_program("def solveB(): return 9 # slightly different", 0.9, True, 1, None, "pB_dup", {"detail": "B_dup"}) # Duplicate normalized
    prog_E_s06_vT_id = add_program("def solveE(): return 6", 0.6, True, 2, None, "pE", {"detail": "E"})
    prog_F_s08_vT_id_dupA = add_program("def solveA(): return 8 # different again", 0.8, True, 2, None, "pA_dup", {"detail": "A_dup"}) # Duplicate normalized
    prog_G_s05_vF_id = add_program("def solveG(): error", 0.05, False, 2, None, "pG", {"detail": "G_invalid"}) # Invalid

    assert prog_D_s09_vT_id_dupB is None # Check that duplicate was not added
    assert prog_F_s08_vT_id_dupA is None

    print("\nTesting get_best_n_programs:")
    top_3 = get_best_n_programs(n=3)
    print(f"Top 3 programs: {[p['program_id'][:8] + ' (score: ' + str(p['score']) +')' for p in top_3]}")
    assert len(top_3) == 3
    assert top_3[0]['score'] == 0.9 # prog_B
    assert top_3[1]['score'] == 0.8 # prog_A
    assert top_3[2]['score'] == 0.7 # prog_C
    
    top_5_min075 = get_best_n_programs(n=5, min_score=0.75)
    print(f"Top 5 programs (min_score 0.75): {[p['program_id'][:8] + ' (score: ' + str(p['score']) +')' for p in top_5_min075]}")
    assert len(top_5_min075) == 2 # B (0.9) and A (0.8)
    assert top_5_min075[0]['score'] == 0.9
    assert top_5_min075[1]['score'] == 0.8

    print("\nTesting get_unique_programs_from_top_k:")
    # We have B(0.9), A(0.8), C(0.7), E(0.6) as unique valid programs. G is invalid.
    # Duplicates of B and A were not added.

    unique_2_from_top_3 = get_unique_programs_from_top_k(k_to_select=2, top_n_to_consider=3)
    # Top 3 to consider are B(0.9), A(0.8), C(0.7). All are unique. So it should pick first 2.
    print(f"Unique 2 from top 3: {[p['program_id'][:8] + ' (score: ' + str(p['score']) +')' for p in unique_2_from_top_3]}")
    assert len(unique_2_from_top_3) == 2
    assert unique_2_from_top_3[0]['score'] == 0.9 # B
    assert unique_2_from_top_3[1]['score'] == 0.8 # A

    unique_3_from_top_10 = get_unique_programs_from_top_k(k_to_select=3, top_n_to_consider=10)
    # Should get B, A, C
    print(f"Unique 3 from top 10: {[p['program_id'][:8] + ' (score: ' + str(p['score']) +')' for p in unique_3_from_top_10]}")
    assert len(unique_3_from_top_10) == 3
    assert unique_3_from_top_10[0]['score'] == 0.9 # B
    assert unique_3_from_top_10[1]['score'] == 0.8 # A
    assert unique_3_from_top_10[2]['score'] == 0.7 # C
    
    unique_5_from_top_10 = get_unique_programs_from_top_k(k_to_select=5, top_n_to_consider=10)
    # We only have 4 unique valid programs: B,A,C,E
    print(f"Unique 5 from top 10: {[p['program_id'][:8] + ' (score: ' + str(p['score']) +')' for p in unique_5_from_top_10]}")
    assert len(unique_5_from_top_10) == 4 

    print("\nProgram DB query tests finished.")
    print(f"Database located at: {DATABASE_PATH}") 