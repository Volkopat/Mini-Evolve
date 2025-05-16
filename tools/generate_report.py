import sqlite3
import json
import os
import sys
import datetime
import yaml

# Adjust Python path to include the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.program_db import get_database_path, normalize_code # Assuming normalize_code might be useful for display

REPORTS_DIR = "reports"
MAIN_CONFIG_FILE = "config/config.yaml"

def get_current_problem_from_config():
    try:
        with open(MAIN_CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('current_problem_directory', 'Unknown Problem')
    except Exception:
        return "Unknown Problem (Error reading config)"

def generate_report(db_path, report_filename=None):
    if not os.path.exists(db_path):
        print("Database file '%s' not found." % db_path)
        return

    problem_name = get_current_problem_from_config()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    report_lines = []
    report_lines.append("# Mini-Evolve Run Report")
    report_lines.append("Generated: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report_lines.append("Problem: %s" % os.path.basename(problem_name)) # Get just the dir name
    report_lines.append("Database: %s" % db_path)
    report_lines.append("\n---")

    # --- Overall Statistics ---
    report_lines.append("\n## I. Overall Statistics")
    try:
        cursor.execute("SELECT COUNT(*) as total_programs FROM programs")
        total_programs = cursor.fetchone()['total_programs']
        report_lines.append("- Total programs in database: %s" % total_programs)

        cursor.execute("SELECT COUNT(*) as valid_programs FROM programs WHERE is_valid = 1")
        valid_programs = cursor.fetchone()['valid_programs']
        report_lines.append("- Valid programs: %s" % valid_programs)

        cursor.execute("SELECT COUNT(*) as invalid_programs FROM programs WHERE is_valid = 0")
        invalid_programs = cursor.fetchone()['invalid_programs']
        report_lines.append("- Invalid programs: %s" % invalid_programs)

        if total_programs > 0:
            report_lines.append("- Percentage valid: %.2f%%" % ((valid_programs / total_programs) * 100 if total_programs > 0 else 0))

        cursor.execute("SELECT MIN(score) as min_score, MAX(score) as max_score, AVG(score) as avg_score FROM programs WHERE is_valid = 1")
        score_stats = cursor.fetchone()
        if score_stats and score_stats['max_score'] is not None:
            report_lines.append("- Max score (valid programs): %.4f" % score_stats['max_score'])
            report_lines.append("- Min score (valid programs): %.4f" % score_stats['min_score'])
            report_lines.append("- Average score (valid programs): %.4f" % score_stats['avg_score'])
        else:
            report_lines.append("- No valid programs with scores found for stats.")
            
        cursor.execute("SELECT MIN(generation_discovered) as first_gen, MAX(generation_discovered) as last_gen FROM programs")
        gen_stats = cursor.fetchone()
        if gen_stats and gen_stats['first_gen'] is not None:
            report_lines.append("- Generations spanned: %s to %s" % (gen_stats['first_gen'], gen_stats['last_gen']))

    except sqlite3.Error as e:
        report_lines.append("- Error fetching overall statistics: %s" % e)


    # --- Best Program(s) ---
    report_lines.append("\n## II. Best Program(s)")
    try:
        # Get the program(s) with the highest score
        # If multiple have the same highest score, this will pick one by timestamp
        cursor.execute("SELECT * FROM programs WHERE is_valid = 1 ORDER BY score DESC, timestamp_added DESC LIMIT 1")
        best_program = cursor.fetchone()

        if best_program:
            report_lines.append("### Top Scorer:")
            report_lines.append("- Program ID: %s" % best_program['program_id'])
            report_lines.append("- Score: %.4f" % best_program['score'])
            report_lines.append("- Generation Discovered: %s" % best_program['generation_discovered'])
            report_lines.append("- Parent ID: %s" % best_program['parent_id'])
            try:
                eval_res = json.loads(best_program['evaluation_results_json'])
                report_lines.append("- Evaluation Details: `%s`" % json.dumps(eval_res))
            except:
                report_lines.append("- Evaluation Details: Error parsing JSON.")
            report_lines.append("```python")
            report_lines.append(best_program['code_string'])
            report_lines.append("```")
        else:
            report_lines.append("No valid programs found in the database.")
    except sqlite3.Error as e:
        report_lines.append("- Error fetching best program: %s" % e)

    # --- Top N Programs ---
    TOP_N_COUNT = 5
    report_lines.append("\n## III. Top %s Programs (by Score)" % TOP_N_COUNT)
    try:
        cursor.execute("SELECT * FROM programs WHERE is_valid = 1 ORDER BY score DESC, timestamp_added DESC LIMIT ?", (TOP_N_COUNT,))
        top_programs = cursor.fetchall()
        if top_programs:
            for i, prog in enumerate(top_programs):
                report_lines.append("\n### %s. Program ID: %s" % (i + 1, prog['program_id']))
                report_lines.append("    - Score: %.4f" % prog['score'])
                report_lines.append("    - Generation: %s" % prog['generation_discovered'])
                report_lines.append("    - Parent ID: %s" % prog['parent_id'])
                try:
                    eval_res_top = json.loads(prog['evaluation_results_json'])
                    report_lines.append("    - Evaluation Details: `%s`" % json.dumps(eval_res_top))
                except:
                    report_lines.append("    - Evaluation Details: Error parsing JSON.")
                report_lines.append("    - Code:")
                report_lines.append("    ```python")
                # Indent each line of the code block for proper markdown formatting within a list item
                for code_line in prog['code_string'].split('\n'):
                    report_lines.append("    " + code_line)
                report_lines.append("    ```")
        else:
            report_lines.append("No valid programs to display in top %s." % TOP_N_COUNT)
    except sqlite3.Error as e:
        report_lines.append("- Error fetching top N programs: %s" % e)

    # --- Section IV: Evolutionary Tree ---
    report_lines.append("\n## IV. Evolutionary Lineage (Parent-Child)")
    try:
        cursor.execute("SELECT program_id, parent_id, generation_discovered, score, is_valid FROM programs ORDER BY generation_discovered ASC, timestamp_added ASC")
        all_programs_for_tree = cursor.fetchall()
        
        programs_by_id = {p['program_id']: p for p in all_programs_for_tree}
        children_map = {}
        for p in all_programs_for_tree:
            if p['parent_id']:
                if p['parent_id'] not in children_map:
                    children_map[p['parent_id']] = []
                children_map[p['parent_id']].append(p['program_id'])

        added_to_report = set()
        def add_program_to_lineage_report(prog_id, indent_level):
            if prog_id in added_to_report:
                return
            added_to_report.add(prog_id)
            prog = programs_by_id[prog_id]
            indent = "    " * indent_level
            valid_char = 'V' if prog['is_valid'] else 'I'
            score_str = "%.3f" % prog['score'] if prog['score'] is not None else 'N/A'
            report_lines.append("%s- Gen: %s, ID: %s (Score: %s, %s)" % \
                                (indent, prog['generation_discovered'], prog_id[:8], score_str, valid_char))
            if prog_id in children_map:
                for child_id in children_map[prog_id]:
                    add_program_to_lineage_report(child_id, indent_level + 1)

        # Start with seed programs (no parent_id)
        seed_programs = [p for p in all_programs_for_tree if p['parent_id'] is None]
        if not seed_programs and all_programs_for_tree: # Fallback if no explicit seeds but programs exist
            # This might happen if DB was populated manually or if a run started with non-seed parents someotherhow
            # report_lines.append("- Warning: No seed programs (null parent_id) found. Displaying all as roots if unrelated.")
            # For simplicity, we'll just iterate through all and let the map build the tree from available roots
            processed_for_roots = set()
            for p_id in programs_by_id.keys():
                # check if it is a root (or effectively a root for this display)
                is_root_for_display = True
                if programs_by_id[p_id]['parent_id'] and programs_by_id[p_id]['parent_id'] in programs_by_id:
                    is_root_for_display = False 
                if is_root_for_display and p_id not in added_to_report:
                     add_program_to_lineage_report(p_id, 0)
        elif seed_programs:
            for seed in seed_programs:
                add_program_to_lineage_report(seed['program_id'], 0)
        else:
            report_lines.append("- No programs found for lineage tree.")

    except sqlite3.Error as e:
        report_lines.append("- Error generating evolutionary lineage: %s" % e)

    # --- Section V: Sequential List of Programs by Generation ---
    report_lines.append("\n## V. All Programs by Generation & Timestamp")
    try:
        cursor.execute("SELECT program_id, generation_discovered, score, is_valid, parent_id, timestamp_added, code_string FROM programs ORDER BY generation_discovered ASC, timestamp_added ASC")
        all_programs_sequential = cursor.fetchall()
        if all_programs_sequential:
            for i, prog in enumerate(all_programs_sequential):
                report_lines.append("\n### %s. Program ID: %s (Gen: %s)" % (i + 1, prog['program_id'], prog['generation_discovered']))
                report_lines.append("    - Score: %.4f" % (prog['score'] if prog['score'] is not None else -1))
                report_lines.append("    - Valid: %s" % bool(prog['is_valid']))
                report_lines.append("    - Parent ID: %s" % prog['parent_id'])
                report_lines.append("    - Timestamp: %.2f" % prog['timestamp_added'])
                # Display full code for Section V programs
                report_lines.append("    - Code:")
                report_lines.append("    ```python")
                for code_line in prog['code_string'].split('\n'):
                    report_lines.append("    " + code_line)
                report_lines.append("    ```")
        else:
            report_lines.append("- No programs found to list sequentially.")
    except sqlite3.Error as e:
        report_lines.append("- Error fetching sequential program list: %s" % e)
        
    conn.close()

    # --- Write Report ---
    if not os.path.exists(REPORTS_DIR):
        try:
            os.makedirs(REPORTS_DIR)
        except OSError as e:
            print("Error creating reports directory '%s': %s" % (REPORTS_DIR, e))
            return

    if report_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize problem_name for use in filename
        problem_name_for_file = os.path.basename(problem_name) # Get just dir name
        problem_name_for_file = problem_name_for_file.replace(' ', '_').replace('/', '-') # Replace spaces and slashes
        problem_name_for_file = "".join(c for c in problem_name_for_file if c.isalnum() or c in ('_', '-')).strip()
        if not problem_name_for_file: # Fallback if somehow empty
            problem_name_for_file = "problem"
        
        report_filename = os.path.join(REPORTS_DIR, "report_%s_%s.md" % (problem_name_for_file, timestamp))
    else:
        # If a specific filename is provided, use it directly (could also prepend problem name if desired)
        report_filename = os.path.join(REPORTS_DIR, report_filename)

    try:
        with open(report_filename, 'w') as f:
            f.write("\n".join(report_lines))
        print("Report generated successfully: %s" % report_filename)
    except IOError as e:
        print("Error writing report file '%s': %s" % (report_filename, e))

if __name__ == "__main__":
    db_file = get_database_path()
    print("Generating report from database: %s" % db_file)
    generate_report(db_file)
    print("\nTo view the report, open the .md file in the '%s' directory." % REPORTS_DIR) 