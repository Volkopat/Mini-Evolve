import sqlite3
import hashlib
import ast
import time
import json
import uuid
import os
import yaml
from .logger_setup import get_logger

# --- Configuration ---
CONFIG_FILE = "config/config.yaml"

def load_db_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        get_logger("EvaluatorDB").warning(f"Warning: Main configuration file {CONFIG_FILE} not found. Using default DB path and no MAP-Elites config.")
        return {}
    except yaml.YAMLError as e:
        get_logger("EvaluatorDB").warning(f"Warning: Error parsing YAML configuration: {e}. Using default DB path and no MAP-Elites config.")
        return {}

_map_elites_config_internal = {}

def set_map_elites_config(config: dict):
    global _map_elites_config_internal
    _map_elites_config_internal = config

# Initial load
full_config_initial = load_db_config()
db_config = full_config_initial.get('database', {}).get('evaluator_database', {})
set_map_elites_config(full_config_initial.get('map_elites', {}))

DATABASE_PATH = db_config.get('path', 'evaluator_database.db')

# --- Code Normalization (reused for evaluators) ---
def normalize_code(code_string: str) -> str:
    """Normalizes Python code by parsing and unparsing it using AST."""
    try:
        parsed_ast = ast.parse(code_string)
        return ast.unparse(parsed_ast)
    except SyntaxError:
        return code_string
    except Exception as e:
        get_logger("EvaluatorDB").warning(f"Warning: AST normalization failed: {e}. Returning original code string.")
        return code_string

def get_code_hash(code_string: str) -> str:
    """Generates a SHA256 hash for a code string."""
    return hashlib.sha256(code_string.encode('utf-8')).hexdigest()

# --- Database Initialization ---
def init_db(db_path=DATABASE_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluators (
        evaluator_id TEXT PRIMARY KEY,
        evaluator_code_string TEXT NOT NULL,
        normalized_evaluator_code_string TEXT NOT NULL, 
        normalized_evaluator_code_hash TEXT NOT NULL UNIQUE,
        challenge_score REAL,
        generation_discovered INTEGER,
        parent_evaluator_id TEXT,
        llm_prompt TEXT,
        evaluation_results_json TEXT,
        descriptor_1 REAL,
        descriptor_2 REAL,
        timestamp_added REAL NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS map_elites_grid (
        descriptor_1_bin INTEGER NOT NULL,
        descriptor_2_bin INTEGER NOT NULL,
        evaluator_id TEXT NOT NULL,
        PRIMARY KEY (descriptor_1_bin, descriptor_2_bin),
        FOREIGN KEY (evaluator_id) REFERENCES evaluators (evaluator_id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_normalized_evaluator_hash ON evaluators (normalized_evaluator_code_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_challenge_score ON evaluators (challenge_score DESC)")
    conn.commit()
    conn.close()
    get_logger("EvaluatorDB").info(f"Evaluator database initialized at {db_path}")

# --- API Functions ---
def add_evaluator(evaluator_code_string: str,
                  challenge_score: float | None,
                  generation_discovered: int | None,
                  parent_evaluator_id: str | None,
                  llm_prompt: str | None,
                  evaluation_results: dict,
                  descriptor_1: float | None = None,
                  descriptor_2: float | None = None,
                  db_path=DATABASE_PATH) -> str | None:
    
    original_evaluator_code_string = evaluator_code_string
    normalized_evaluator = normalize_code(evaluator_code_string)
    normalized_hash = get_code_hash(normalized_evaluator)

    if check_if_exists(normalized_hash, db_path=db_path):
        return None

    evaluator_id = str(uuid.uuid4())
    timestamp = time.time()
    evaluation_results_str = json.dumps(evaluation_results)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO evaluators (
            evaluator_id, evaluator_code_string, normalized_evaluator_code_string, normalized_evaluator_code_hash,
            challenge_score, generation_discovered, parent_evaluator_id, llm_prompt,
            evaluation_results_json, descriptor_1, descriptor_2, timestamp_added
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (evaluator_id, original_evaluator_code_string, normalized_evaluator, normalized_hash,
              challenge_score, generation_discovered, parent_evaluator_id, llm_prompt,
              evaluation_results_str, descriptor_1, descriptor_2, timestamp))
        conn.commit()
        return evaluator_id
    except sqlite3.IntegrityError as e:
        get_logger("EvaluatorDB").error(f"SQLite IntegrityError while adding evaluator: {e}. Hash: {normalized_hash}")
        return None 
    except Exception as e:
        get_logger("EvaluatorDB").error(f"Error adding evaluator to DB: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def check_if_exists(normalized_evaluator_code_hash: str, db_path=DATABASE_PATH) -> bool:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM evaluators WHERE normalized_evaluator_code_hash = ?", (normalized_evaluator_code_hash,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_evaluator(evaluator_id: str, db_path=DATABASE_PATH) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluators WHERE evaluator_id = ?", (evaluator_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        evaluator_data = dict(row)
        try:
            evaluator_data['evaluation_results'] = json.loads(evaluator_data['evaluation_results_json'])
        except (json.JSONDecodeError, TypeError) as e:
            get_logger("EvaluatorDB").warning(f"Warning: Could not parse evaluation_results_json for evaluator {evaluator_id}: {e}")
            evaluator_data['evaluation_results'] = {}
        return evaluator_data
    return None

def _get_bin_coordinates(descriptor_values: list[float], grid_resolution: list[int]) -> tuple[int, ...]:
    bins = []
    divisors = [10, 1] # Example: 10 for number of test cases, 1 for complexity
    for i, value in enumerate(descriptor_values):
        if i < len(grid_resolution) and i < len(divisors):
            bin_coord = int(value / divisors[i])
            bin_coord = max(0, min(bin_coord, grid_resolution[i] - 1))
            bins.append(bin_coord)
        else:
            bins.append(0)
    return tuple(bins)

def add_evaluator_to_map_elites(evaluator_id: str, challenge_score: float, descriptor_values: list[float], db_path=DATABASE_PATH) -> bool:
    logger = get_logger("EvaluatorDB.MAPElites")
    logger.debug(f"Attempting to add evaluator {evaluator_id} to MAP-Elites grid.")

    map_elites_config = get_map_elites_config()

    if not map_elites_config.get('enable', False):
        logger.debug("MAP-Elites not enabled in config.")
        return False

    grid_resolution = map_elites_config.get('grid_resolution')
    if not grid_resolution or len(descriptor_values) != len(grid_resolution):
        logger.warning(f"MAP-Elites grid_resolution not configured correctly ({grid_resolution}) or mismatch with descriptor values ({descriptor_values}).")
        return False

    bin_coords = _get_bin_coordinates(descriptor_values, grid_resolution)
    logger.debug(f"Evaluator {evaluator_id} with descriptors {descriptor_values} mapped to bin {bin_coords}.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(f"""
            SELECT T1.evaluator_id, T1.challenge_score
            FROM evaluators AS T1
            JOIN map_elites_grid AS T2 ON T1.evaluator_id = T2.evaluator_id
            WHERE T2.descriptor_1_bin = ? AND T2.descriptor_2_bin = ?
        """, bin_coords)
        
        existing_elite = cursor.fetchone()

        if existing_elite:
            existing_elite_id, existing_elite_score = existing_elite
            logger.debug(f"Existing elite found for bin {bin_coords}: {existing_elite_id} (score: {existing_elite_score}). New evaluator score: {challenge_score}.")
            if challenge_score > existing_elite_score:
                cursor.execute(f"""
                    UPDATE map_elites_grid
                    SET evaluator_id = ?
                    WHERE descriptor_1_bin = ? AND descriptor_2_bin = ?
                """, (evaluator_id, *bin_coords))
                conn.commit()
                logger.info(f"Updated elite for bin {bin_coords}: Evaluator {evaluator_id} (score: {challenge_score}) replaced {existing_elite_id} (score: {existing_elite_score})")
                return True
            else:
                logger.debug(f"Existing elite {existing_elite_id} (score: {existing_elite_score}) is better or equal. New evaluator not added as elite.")
                return False
        else:
            cursor.execute(f"""
                INSERT INTO map_elites_grid (descriptor_1_bin, descriptor_2_bin, evaluator_id)
                VALUES (?, ?, ?)
            """, (*bin_coords, evaluator_id))
            conn.commit()
            logger.info(f"New elite for bin {bin_coords}: Evaluator {evaluator_id} (score: {challenge_score})")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error adding evaluator to MAP-Elites grid: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_best_n_evaluators(n: int, min_score: float | None = None, db_path=DATABASE_PATH) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM evaluators"
    params = []
    
    if min_score is not None:
        query += " WHERE challenge_score >= ?"
        params.append(min_score)
        
    query += " ORDER BY challenge_score DESC LIMIT ?"
    params.append(n)
    
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    
    evaluators = []
    for row in rows:
        evaluator_data = dict(row)
        try:
            evaluator_data['evaluation_results'] = json.loads(evaluator_data['evaluation_results_json'])
        except (json.JSONDecodeError, TypeError):
            evaluator_data['evaluation_results'] = {}
        evaluators.append(evaluator_data)
    return evaluators

def get_unique_evaluators_from_top_k(k_to_select: int, top_n_to_consider: int, db_path=DATABASE_PATH) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    base_query = "SELECT * FROM evaluators ORDER BY challenge_score DESC, timestamp_added DESC LIMIT ?"
    params = [top_n_to_consider]
    
    cursor.execute(base_query, tuple(params))
    
    potential_evaluators_rows = cursor.fetchall()
    conn.close()

    selected_evaluators = []
    seen_normalized_hashes = set()

    for row in potential_evaluators_rows:
        if len(selected_evaluators) >= k_to_select:
            break

        evaluator_data = dict(row)
        normalized_hash = evaluator_data['normalized_evaluator_code_hash']

        if normalized_hash not in seen_normalized_hashes:
            try:
                evaluator_data['evaluation_results'] = json.loads(evaluator_data['evaluation_results_json'])
            except (json.JSONDecodeError, TypeError):
                evaluator_data['evaluation_results'] = {}
            
            selected_evaluators.append(evaluator_data)
            seen_normalized_hashes.add(normalized_hash)
            
    return selected_evaluators

def get_database_path() -> str:
    return DATABASE_PATH

def get_map_elites_config() -> dict:
    return _map_elites_config_internal

if __name__ == "__main__":
    # Test the evaluator_db module
    test_db_path = "test_evaluator_database.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    init_db(db_path=test_db_path)
    
    logger = get_logger("EvaluatorDB.Test")
    logger.info("Starting evaluator_db tests...")

    # Test add_evaluator
    evaluator_code_1 = """def evaluate_program(program_module, problem_config, main_config):
    # Simple evaluator: always returns 1.0
    return {'score': 1.0, 'is_valid': True, 'error_message': None}
"""
    eval_res_1 = {"test_results": "all_pass"}
    desc_1 = [1, 1] # Example descriptors: num_test_cases, complexity
    
    evaluator_id_1 = add_evaluator(evaluator_code_1, 0.5, 0, None, "initial_evaluator_llm_call", eval_res_1, desc_1, db_path=test_db_path)
    logger.info(f"Added evaluator 1: {evaluator_id_1}")
    assert evaluator_id_1 is not None

    # Test duplicate evaluator (should not add)
    evaluator_id_dup = add_evaluator(evaluator_code_1, 0.6, 1, evaluator_id_1, "llm_call_dup", eval_res_1, desc_1, db_path=test_db_path)
    logger.info(f"Added duplicate evaluator: {evaluator_id_dup}")
    assert evaluator_id_dup is None

    # Test add_evaluator_to_map_elites
    _map_elites_config_internal['enable'] = True
    _map_elites_config_internal['grid_resolution'] = [10, 10]

    add_evaluator_to_map_elites(evaluator_id_1, 0.5, desc_1, db_path=test_db_path)
    logger.info(f"Added evaluator {evaluator_id_1} to MAP-Elites grid.")

    # Add a better evaluator for the same bin
    evaluator_code_2 = """def evaluate_program(program_module, problem_config, main_config):
    # More complex evaluator: checks for specific edge cases
    if hasattr(program_module, 'solve') and program_module.solve(0,0) == 1:
        return {'score': 0.1, 'is_valid': True, 'error_message': 'Failed edge case'}
    return {'score': 0.9, 'is_valid': True, 'error_message': None}
"""
    eval_res_2 = {"test_results": "some_fail"}
    desc_2 = [2, 2] # Assume same bin for test
    
    evaluator_id_2 = add_evaluator(evaluator_code_2, 0.9, 1, evaluator_id_1, "llm_call_2", eval_res_2, desc_2, db_path=test_db_path)
    logger.info(f"Added evaluator 2: {evaluator_id_2}")
    assert evaluator_id_2 is not None

    # This should update the elite for the bin
    add_evaluator_to_map_elites(evaluator_id_2, 0.9, desc_2, db_path=test_db_path)
    logger.info(f"Added evaluator {evaluator_id_2} to MAP-Elites grid (should replace previous).")

    # Test get_evaluator
    retrieved_evaluator_1 = get_evaluator(evaluator_id_1, db_path=test_db_path)
    logger.info(f"Retrieved evaluator 1: {retrieved_evaluator_1['evaluator_code_string'][:50]}...")
    assert retrieved_evaluator_1 is not None
    assert retrieved_evaluator_1['challenge_score'] == 0.5

    # Test get_best_n_evaluators
    best_evaluators = get_best_n_evaluators(1, db_path=test_db_path)
    logger.info(f"Best evaluator score: {best_evaluators[0]['challenge_score']}")
    assert best_evaluators[0]['challenge_score'] == 0.9

    # Test get_unique_evaluators_from_top_k
    unique_evaluators = get_unique_evaluators_from_top_k(1, 10, db_path=test_db_path)
    logger.info(f"Unique evaluator count: {len(unique_evaluators)}")
    assert len(unique_evaluators) == 2 # evaluator_id_1 and evaluator_id_2 are unique by content

    logger.info("All evaluator_db tests passed.")
    os.remove(test_db_path)