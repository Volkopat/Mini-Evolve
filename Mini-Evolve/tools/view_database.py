import sqlite3
import json
import os
import sys # Added sys

# Adjust Python path to include the project root (parent of 'app' and 'tools')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.program_db import get_database_path # Changed import

def display_programs(db_path, limit=None, sort_by='score_desc'):
    """
    Connects to the SQLite database and displays program information.

    Args:
        db_path (str): Path to the SQLite database file.
        limit (int, optional): Number of programs to display. Displays all if None.
        sort_by (str, optional): How to sort programs. Options:
                                 'score_desc', 'score_asc',
                                 'gen_desc', 'gen_asc',
                                 'time_desc', 'time_asc' (default)
    """
    if not os.path.exists(db_path):
        print("Database file '%s' not found." % db_path)
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row # Access columns by name
    cursor = conn.cursor()

    query = "SELECT program_id, code_string, score, is_valid, generation_discovered, parent_id, timestamp_added, evaluation_results_json FROM programs"

    if sort_by == 'score_desc':
        query += " ORDER BY score DESC, timestamp_added DESC"
    elif sort_by == 'score_asc':
        query += " ORDER BY score ASC, timestamp_added ASC"
    elif sort_by == 'gen_desc':
        query += " ORDER BY generation_discovered DESC, score DESC"
    elif sort_by == 'gen_asc':
        query += " ORDER BY generation_discovered ASC, score DESC"
    elif sort_by == 'time_asc':
        query += " ORDER BY timestamp_added ASC"
    else: # Default to time_desc
        query += " ORDER BY timestamp_added DESC"

    if limit is not None:
        query += " LIMIT %d" % int(limit)

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print("Error querying database: %s" % e)
        print("Ensure the database schema is correct and the file is a valid SQLite DB.")
        conn.close()
        return
    
    conn.close()

    if not rows:
        print("No programs found in the database.")
        return

    print("\n--- Program Database Contents (Sorted by: %s, Limit: %s) ---" % (sort_by, limit if limit is not None else 'All'))
    for i, row in enumerate(rows):
        print("\n--- Program %d ---" % (i + 1))
        print("ID: %s" % row['program_id'])
        print("Generation: %s" % row['generation_discovered'])
        print("Score: %.4f" % (row['score'] if row['score'] is not None else -1))
        print("Is Valid: %s" % bool(row['is_valid']))
        print("Parent ID: %s" % row['parent_id'])
        print("Timestamp: %.2f" % row['timestamp_added'])
        
        try:
            eval_results = json.loads(row['evaluation_results_json'])
            print("Evaluation Output: %s" % eval_results.get('output'))
            if eval_results.get('error_message'):
                print("Evaluation Error: %s" % eval_results.get('error_message'))
        except (json.JSONDecodeError, TypeError):
            print("Evaluation Results: Could not parse JSON or data missing.")

        print("Code:\n%s" % row['code_string'])
        print("--------------------")

if __name__ == "__main__":
    db_file_path = get_database_path()
    
    # Basic usage: Display top 10 by score
    # display_programs(db_file_path, limit=10, sort_by='score_desc')
    
    # Display all programs, sorted by generation discovered (ascending) then score
    display_programs(db_file_path, sort_by='gen_asc')

    # To display programs by time added (oldest first):
    # display_programs(db_file_path, sort_by='time_asc')

    # To display all programs, latest first (default if no sort_by is given effectively):
    # display_programs(db_file_path, sort_by='time_desc') 