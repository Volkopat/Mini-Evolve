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
import sys
# Adjust Python path to include the project root for config loading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CONFIG_FILE = "config/config.yaml"

# Global variable to hold the main config once loaded
_main_config_cache = None

def load_main_config_for_db():
    global _main_config_cache
    if _main_config_cache is None:
        try:
            with open(CONFIG_FILE, 'r') as f:
                _main_config_cache = yaml.safe_load(f)
        except FileNotFoundError:
            get_logger("PromptDB").critical(f"CRITICAL: Main configuration file {CONFIG_FILE} not found.")
            raise
        except yaml.YAMLError as e:
            get_logger("PromptDB").critical(f"CRITICAL: Error parsing main YAML configuration: {e}.")
            raise
    return _main_config_cache

_map_elites_config_internal = {}

def set_map_elites_config(config: dict):
    global _map_elites_config_internal
    _map_elites_config_internal = config

# No initial load of DATABASE_PATH here. It will be fetched via get_database_path()
# when needed, ensuring main_config is loaded.

# --- Code Normalization (reused for prompts) ---
def normalize_code(code_string: str) -> str:
    """Normalizes Python code by parsing and unparsing it using AST."""
    try:
        parsed_ast = ast.parse(code_string)
        return ast.unparse(parsed_ast)
    except SyntaxError:
        return code_string
    except Exception as e:
        get_logger("PromptDB").warning(f"Warning: AST normalization failed: {e}. Returning original code string.")
        return code_string

def get_code_hash(code_string: str) -> str:
    """Generates a SHA256 hash for a code string."""
    return hashlib.sha256(code_string.encode('utf-8')).hexdigest()

# --- Database Initialization ---
def init_db(db_path=None):
    if db_path is None:
        db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prompts (
        prompt_id TEXT PRIMARY KEY,
        prompt_string TEXT NOT NULL,
        normalized_prompt_string TEXT NOT NULL, 
        normalized_prompt_hash TEXT NOT NULL UNIQUE,
        score REAL,
        generation_discovered INTEGER,
        parent_prompt_id TEXT,
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
        prompt_id TEXT NOT NULL,
        PRIMARY KEY (descriptor_1_bin, descriptor_2_bin),
        FOREIGN KEY (prompt_id) REFERENCES prompts (prompt_id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_normalized_prompt_hash ON prompts (normalized_prompt_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_score ON prompts (score DESC)")
    conn.commit()
    conn.close()
    get_logger("PromptDB").info(f"Prompt database initialized at {db_path}")

# --- API Functions ---
def add_prompt(prompt_string: str,
               score: float | None,
               generation_discovered: int | None,
               parent_prompt_id: str | None,
               llm_prompt: str | None,
               evaluation_results: dict,
               descriptor_1: float | None = None,
               descriptor_2: float | None = None,
               db_path=None) -> str | None:
    if db_path is None:
        db_path = get_database_path()
    
    original_prompt_string = prompt_string
    normalized_prompt = normalize_code(prompt_string)
    normalized_hash = get_code_hash(normalized_prompt)

    if check_if_exists(normalized_hash, db_path=db_path):
        return None

    prompt_id = str(uuid.uuid4())
    timestamp = time.time()
    evaluation_results_str = json.dumps(evaluation_results)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO prompts (
            prompt_id, prompt_string, normalized_prompt_string, normalized_prompt_hash,
            score, generation_discovered, parent_prompt_id, llm_prompt,
            evaluation_results_json, descriptor_1, descriptor_2, timestamp_added
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (prompt_id, original_prompt_string, normalized_prompt, normalized_hash,
              score, generation_discovered, parent_prompt_id, llm_prompt,
              evaluation_results_str, descriptor_1, descriptor_2, timestamp))
        conn.commit()
        return prompt_id
    except sqlite3.IntegrityError as e:
        get_logger("PromptDB").error(f"SQLite IntegrityError while adding prompt: {e}. Hash: {normalized_hash}")
        return None 
    except Exception as e:
        get_logger("PromptDB").error(f"Error adding prompt to DB: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def check_if_exists(normalized_prompt_hash: str, db_path=None) -> bool:
    if db_path is None:
        db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM prompts WHERE normalized_prompt_hash = ?", (normalized_prompt_hash,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_prompt(prompt_id: str, db_path=None) -> dict | None:
    if db_path is None:
        db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prompts WHERE prompt_id = ?", (prompt_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        prompt_data = dict(row)
        try:
            prompt_data['evaluation_results'] = json.loads(prompt_data['evaluation_results_json'])
        except (json.JSONDecodeError, TypeError) as e:
            get_logger("PromptDB").warning(f"Warning: Could not parse evaluation_results_json for prompt {prompt_id}: {e}")
            prompt_data['evaluation_results'] = {}
        return prompt_data
    return None

def _get_bin_coordinates(descriptor_values: list[float], grid_resolution: list[int]) -> tuple[int, ...]:
    bins = []
    divisors = [50, 5] # Example: 50 for length, 5 for complexity/keywords
    for i, value in enumerate(descriptor_values):
        if i < len(grid_resolution) and i < len(divisors):
            bin_coord = int(value / divisors[i])
            bin_coord = max(0, min(bin_coord, grid_resolution[i] - 1))
            bins.append(bin_coord)
        else:
            bins.append(0)
    return tuple(bins)

def add_prompt_to_map_elites(prompt_id: str, score: float, descriptor_values: list[float], db_path=None) -> bool:
    if db_path is None:
        db_path = get_database_path()
    logger = get_logger("PromptDB.MAPElites")
    logger.debug(f"Attempting to add prompt {prompt_id} to MAP-Elites grid.")

    map_elites_config = get_map_elites_config()

    if not map_elites_config.get('enable', False):
        logger.debug("MAP-Elites not enabled in config.")
        return False

    grid_resolution = map_elites_config.get('grid_resolution')
    if not grid_resolution or len(descriptor_values) != len(grid_resolution):
        logger.warning(f"MAP-Elites grid_resolution not configured correctly ({grid_resolution}) or mismatch with descriptor values ({descriptor_values}).")
        return False

    bin_coords = _get_bin_coordinates(descriptor_values, grid_resolution)
    logger.debug(f"Prompt {prompt_id} with descriptors {descriptor_values} mapped to bin {bin_coords}.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(f"""
            SELECT T1.prompt_id, T1.score
            FROM prompts AS T1
            JOIN map_elites_grid AS T2 ON T1.prompt_id = T2.prompt_id
            WHERE T2.descriptor_1_bin = ? AND T2.descriptor_2_bin = ?
        """, bin_coords)
        
        existing_elite = cursor.fetchone()

        if existing_elite:
            existing_elite_id, existing_elite_score = existing_elite
            logger.debug(f"Existing elite found for bin {bin_coords}: {existing_elite_id} (score: {existing_elite_score}). New prompt score: {score}.")
            if score > existing_elite_score:
                cursor.execute(f"""
                    UPDATE map_elites_grid
                    SET prompt_id = ?
                    WHERE descriptor_1_bin = ? AND descriptor_2_bin = ?
                """, (prompt_id, *bin_coords))
                conn.commit()
                logger.info(f"Updated elite for bin {bin_coords}: Prompt {prompt_id} (score: {score}) replaced {existing_elite_id} (score: {existing_elite_score})")
                return True
            else:
                logger.debug(f"Existing elite {existing_elite_id} (score: {existing_elite_score}) is better or equal. New prompt not added as elite.")
                return False
        else:
            cursor.execute(f"""
                INSERT INTO map_elites_grid (descriptor_1_bin, descriptor_2_bin, prompt_id)
                VALUES (?, ?, ?)
            """, (*bin_coords, prompt_id))
            conn.commit()
            logger.info(f"New elite for bin {bin_coords}: Prompt {prompt_id} (score: {score})")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error adding prompt to MAP-Elites grid: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_best_n_prompts(n: int, min_score: float | None = None, db_path=None) -> list[dict]:
    if db_path is None:
        db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM prompts"
    params = []
    
    if min_score is not None:
        query += " WHERE score >= ?"
        params.append(min_score)
        
    query += " ORDER BY score DESC LIMIT ?"
    params.append(n)
    
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    
    prompts = []
    for row in rows:
        prompt_data = dict(row)
        try:
            prompt_data['evaluation_results'] = json.loads(prompt_data['evaluation_results_json'])
        except (json.JSONDecodeError, TypeError):
            prompt_data['evaluation_results'] = {}
        prompts.append(prompt_data)
    return prompts

def get_unique_prompts_from_top_k(k_to_select: int, top_n_to_consider: int, db_path=None) -> list[dict]:
    if db_path is None:
        db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    base_query = "SELECT * FROM prompts ORDER BY score DESC, timestamp_added DESC LIMIT ?"
    params = [top_n_to_consider]
    
    cursor.execute(base_query, tuple(params))
    
    potential_prompts_rows = cursor.fetchall()
    conn.close()

    selected_prompts = []
    seen_normalized_hashes = set()

    for row in potential_prompts_rows:
        if len(selected_prompts) >= k_to_select:
            break

        prompt_data = dict(row)
        normalized_hash = prompt_data['normalized_prompt_hash']

        if normalized_hash not in seen_normalized_hashes:
            try:
                prompt_data['evaluation_results'] = json.loads(prompt_data['evaluation_results_json'])
            except (json.JSONDecodeError, TypeError):
                prompt_data['evaluation_results'] = {}
            
            selected_prompts.append(prompt_data)
            seen_normalized_hashes.add(normalized_hash)
            
    return selected_prompts

def get_database_path() -> str:
    main_config = load_main_config_for_db()
    db_config = main_config.get('database', {}).get('prompt_database', {})
    return db_config.get('path', 'prompt_database.db')

# Define DATABASE_PATH globally after get_database_path is defined

def get_map_elites_config() -> dict:
    return _map_elites_config_internal

if __name__ == "__main__":
    # Test the prompt_db module
    test_db_path = "test_prompt_database.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    init_db(db_path=test_db_path)
    
    logger = get_logger("PromptDB.Test")
    logger.info("Starting prompt_db tests...")

    # Test add_prompt
    prompt_str_1 = "Generate a Python function that sorts a list of integers in ascending order."
    eval_res_1 = {"program_score": 0.8, "valid": True}
    desc_1 = [len(prompt_str_1.splitlines()), prompt_str_1.count("function")]
    
    prompt_id_1 = add_prompt(prompt_str_1, 0.8, 0, None, "initial_prompt_llm_call", eval_res_1, desc_1, db_path=test_db_path)
    logger.info(f"Added prompt 1: {prompt_id_1}")
    assert prompt_id_1 is not None

    # Test duplicate prompt (should not add)
    prompt_id_dup = add_prompt(prompt_str_1, 0.9, 1, prompt_id_1, "llm_call_dup", eval_res_1, desc_1, db_path=test_db_path)
    logger.info(f"Added duplicate prompt: {prompt_id_dup}")
    assert prompt_id_dup is None

    # Test add_prompt_to_map_elites
    # Ensure map_elites is enabled in config for this test
    _map_elites_config_internal['enable'] = True
    _map_elites_config_internal['grid_resolution'] = [10, 10] # Example resolution

    add_prompt_to_map_elites(prompt_id_1, 0.8, desc_1, db_path=test_db_path)
    logger.info(f"Added prompt {prompt_id_1} to MAP-Elites grid.")

    # Add a better prompt for the same bin
    prompt_str_2 = "Generate a highly optimized Python function for quicksorting a list of integers."
    eval_res_2 = {"program_score": 0.95, "valid": True}
    desc_2 = [len(prompt_str_2.splitlines()), prompt_str_2.count("function")] # Assume same bin for test
    
    prompt_id_2 = add_prompt(prompt_str_2, 0.95, 1, prompt_id_1, "llm_call_2", eval_res_2, desc_2, db_path=test_db_path)
    logger.info(f"Added prompt 2: {prompt_id_2}")
    assert prompt_id_2 is not None

    # This should update the elite for the bin
    add_prompt_to_map_elites(prompt_id_2, 0.95, desc_2, db_path=test_db_path)
    logger.info(f"Added prompt {prompt_id_2} to MAP-Elites grid (should replace previous).")

    # Test get_prompt
    retrieved_prompt_1 = get_prompt(prompt_id_1, db_path=test_db_path)
    logger.info(f"Retrieved prompt 1: {retrieved_prompt_1['prompt_string'][:50]}...")
    assert retrieved_prompt_1 is not None
    assert retrieved_prompt_1['score'] == 0.8

    # Test get_best_n_prompts
    best_prompts = get_best_n_prompts(1, db_path=test_db_path)
    logger.info(f"Best prompt score: {best_prompts[0]['score']}")
    assert best_prompts[0]['score'] == 0.95

    # Test get_unique_prompts_from_top_k
    unique_prompts = get_unique_prompts_from_top_k(1, 10, db_path=test_db_path)
    logger.info(f"Unique prompt count: {len(unique_prompts)}")
    assert len(unique_prompts) == 2 # prompt_id_1 and prompt_id_2 are unique by content

    logger.info("All prompt_db tests passed.")
    os.remove(test_db_path)