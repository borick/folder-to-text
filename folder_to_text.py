import argparse
import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
from collections import defaultdict

# --- Configuration ---
# Version 4: Added robust, pattern-based test file filtering for frontend projects.
# Version 3: Added logic to select only the latest .sql file from 'backup' directories.
CONFIG = {
    "EXCLUDE_DIRS_DEFAULT": {
        '.git', 'venv', '__pycache__', 'node_modules', '.pytest_cache', 'build', 'dist',
        'htmlcov', '.mypy_cache', 'reports', 'data',  # 'backup' is removed to allow discovery
    },
    "EXCLUDE_FILES_DEFAULT": {
        '.DS_Store', '.gitignore', 'package-lock.json', 'yarn.lock',
        '.env', '.env.test', '.env.example',
        'jest_detailed_results.json', 'coverage.xml', '.coverage',
    },
    "EXCLUDE_PATTERNS_DEFAULT": [
        re.compile(r'.*_tokens.*\.txt$'),
        re.compile(r'\.log$'),
        re.compile(r'\.tmp$'),
        re.compile(r'\.swp$'),
    ],
    "INCLUDE_EXT_DEFAULT": {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.css', '.scss', '.md', '.json', '.sql',
        '.ini', '.cfg', '.toml', '.yaml', '.yml', '.sh', '.bat', '.txt',
        '.mako', 'Dockerfile'
    },
    "JSON_SAMPLE_CONFIG": {
        "max_lines": 50,
        "max_chars": 3000
    },
    # New in v4: A pattern to identify common test file naming conventions.
    "TEST_FILE_PATTERN": re.compile(r'([._])(test|spec)\.(js|jsx|ts|tsx)$', re.IGNORECASE),
    "COMPLEX_PATTERNS": {
        '.sql': [
            # Pattern 1: Collapse all COPY data blocks (the biggest token saver)
            (
                re.compile(r'(^COPY public\..*? FROM stdin;\n)(.*?)(\n\\\.$\n)', re.DOTALL | re.MULTILINE),
                lambda m: f"{m.group(1)}-- [SQL DATA OMITTED FOR BREVITY]\n{m.group(3)}",
                "SQL_COPY_DATA"
            ),
            # Pattern 2: Remove all SQL comments
            (
                re.compile(r'^--.*?\n', re.MULTILINE),
                "",
                "SQL_COMMENTS"
            ),
            # Pattern 3: Remove all boilerplate SET statements
            (
                re.compile(r'^SET .*?;\n', re.MULTILINE),
                "",
                "SQL_SET_STATEMENTS"
            ),
            # Pattern 4: Remove sequence ownership and value setting
            (
                re.compile(r'^SELECT pg_catalog\.setval.*?;', re.MULTILINE | re.IGNORECASE),
                "",
                "SQL_SETVAL"
            ),
             # Pattern 5: Remove noisy ALTER TABLE ... ADD CONSTRAINT statements
            (
                re.compile(r'^ALTER TABLE .*? ADD CONSTRAINT .*?;\n', re.MULTILINE),
                "",
                "SQL_ADD_CONSTRAINT"
            ),
            # Pattern 6: Remove noisy CREATE INDEX statements
            (
                re.compile(r'^CREATE (UNIQUE )?INDEX .*?;\n', re.MULTILINE),
                "",
                "SQL_CREATE_INDEX"
            ),
            # Pattern 7: Remove noisy ALTER TABLE ... OWNER TO ... statements
            (
                 re.compile(r'^ALTER (TABLE|TYPE|SEQUENCE) .*? OWNER TO .*?;\n', re.MULTILINE),
                 "",
                 "SQL_OWNER_TO"
            ),
             # Pattern 8: Remove sequence anmespace setting
            (
                 re.compile(r'^ALTER SEQUENCE .*? OWNED BY .*?;\n', re.MULTILINE),
                 "",
                 "SQL_SEQUENCE_OWNED_BY"
            )
        ]
    }
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def log_info(message): logger.info(message)
def log_warn(message): logger.warning(message)
def log_error(message): logger.error(message)

def process_file_content(filepath: Path, content: str, args: argparse.Namespace, config: dict) -> str:
    """Processes file content based on file type and command-line arguments."""
    ext = filepath.suffix.lower()

    # Minify lines unless disabled
    if not args.no_minify_lines:
        lines = [line.strip() for line in content.splitlines()]
        content = '\n'.join(filter(None, lines))

    if ext == '.json' and not args.no_json_sample:
        try:
            if len(content.splitlines()) > config["JSON_SAMPLE_CONFIG"]["max_lines"] or len(content) > config["JSON_SAMPLE_CONFIG"]["max_chars"]:
                json.loads(content)
                sampled_content = '\n'.join(content.splitlines()[:config["JSON_SAMPLE_CONFIG"]["max_lines"]])
                return f"// Sample representation of the original JSON file.\n{sampled_content}"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if ext in config["COMPLEX_PATTERNS"]:
        for pattern, replacement, name in config["COMPLEX_PATTERNS"][ext]:
            content, count = pattern.subn(replacement, content)
            if count > 0:
                log_info(f"  -> Applied pattern '{name}' and made {count} replacement(s).")
    
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content

def analyze_output(output_content: str):
    """Provides a report on the generated content."""
    analysis = {"total_chars": len(output_content), "by_type": defaultdict(lambda: {"chars": 0, "count": 0})}
    file_pattern = re.compile(r"--- File: (.*?) ---\n(.*?)(?=\n--- End File:|\Z)", re.DOTALL)
    
    for match in file_pattern.finditer(output_content):
        file_path_str, file_content = match.groups()
        ext = f".{file_path_str.split('.')[-1]}" if '.' in file_path_str else ".other"
        analysis["by_type"][ext]["chars"] += len(file_content)
        analysis["by_type"][ext]["count"] += 1
        
    print("\n" + "-"*40, file=sys.stderr)
    print("Content Analysis Report (on final output):", file=sys.stderr)
    print(f"Total Output Characters: {analysis['total_chars']:,}", file=sys.stderr)
    print("-"*40, file=sys.stderr)
    sorted_types = sorted(analysis["by_type"].items(), key=lambda item: item[1]['chars'], reverse=True)
    for ext, data in sorted_types:
        chars, count = data['chars'], data['count']
        percentage = (chars / analysis["total_chars"]) * 100 if analysis["total_chars"] > 0 else 0
        print(f"  - {ext:<12s}: {chars:>10,d} chars ({percentage:5.2f}%) [{count} file(s)]", file=sys.stderr)
    print("-"*40, file=sys.stderr)

def discover_files(path: Path, config: dict, args: argparse.Namespace, output_abs_path: Path) -> list[Path]:
    """
    Recursively discovers all files in a path that match the include/exclude criteria.
    Returns a list of Path objects.
    """
    discovered_paths = []
    
    if path.is_file():
        # Handle the case where the root_path is a single file
        if path.resolve() != output_abs_path and path.suffix.lower() in config["INCLUDE_EXT_DEFAULT"]:
            discovered_paths.append(path)
        return discovered_paths

    test_file_pattern = config.get("TEST_FILE_PATTERN")

    for root, dirs, files in os.walk(path, topdown=True):
        current_dir = Path(root)
        
        # --- Directory exclusion logic ---
        dirs[:] = [d for d in dirs if d not in config["EXCLUDE_DIRS_DEFAULT"] and d not in args.ignore_dir]
        # 1. Old method: Exclude any directory named 'tests'
        if not args.include_tests and 'tests' in dirs:
            dirs.remove('tests')
        
        for f in files:
            # --- File exclusion logic ---
            
            # 2. New method: Exclude files matching test pattern (e.g., *.test.js)
            if not args.include_tests and test_file_pattern and test_file_pattern.search(f):
                continue
            
            file_path = current_dir / f
            ext = file_path.suffix.lower()
            
            if file_path.resolve() == output_abs_path: continue
            if args.ignore_all_txt and ext == '.txt' and file_path.name.lower() != 'requirements.txt': continue
            if file_path.name in config["EXCLUDE_FILES_DEFAULT"] or file_path.name in args.ignore_file: continue
            if any(p.match(f) for p in config["EXCLUDE_PATTERNS_DEFAULT"]): continue
            if args.ignore_extension and ext in args.ignore_extension: continue
            if ext not in config["INCLUDE_EXT_DEFAULT"]: continue
            
            discovered_paths.append(file_path)
            
    return discovered_paths

def main():
    parser = argparse.ArgumentParser(
        description="A powerful tool to concatenate and process project files into a single text file for LLM context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("root_path", type=Path, help="The root directory or specific file to process.")
    parser.add_argument("-o", "--output", type=Path, default="project_tokens.txt", help="Output file name.")
    parser.add_argument(
        "--additional-context-folder", type=Path, default=None,
        help="Optional: A path to an additional folder to include in the context."
    )
    
    parser.add_argument("--include-tests", action="store_true", help="Include test files and 'tests' directories in the output.")
    parser.add_argument("--ignore-dir", action="append", default=[], help="Specify directory names to ignore.")
    parser.add_argument("--ignore-file", action="append", default=[], help="Specify file names to ignore.")
    parser.add_argument("--ignore-extension", action="append", default=[], help="Specify file extensions to ignore (e.g., .log).")
    
    parser.add_argument("--ignore-all-txt", action="store_true", help="Ignore all .txt files except for 'requirements.txt'.")
    
    parser.add_argument("--no-minify-lines", action="store_true", help="Do not remove blank lines or strip whitespace from lines.")
    parser.add_argument("--no-json-sample", action="store_true", help="Include the full content of large JSON files.")
    
    args = parser.parse_args()

    if not args.root_path.exists():
        log_error(f"Path '{args.root_path}' does not exist."); return

    output_abs_path = args.output.resolve()
    output_abs_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Discover all potential files ---
    all_discovered_files = set()
    
    log_info(f"--- Discovering files in Primary Path: {args.root_path} ---")
    all_discovered_files.update(discover_files(args.root_path, CONFIG, args, output_abs_path))

    if args.additional_context_folder and args.additional_context_folder.is_dir():
        log_info(f"--- Discovering files in Additional Context Folder: {args.additional_context_folder} ---")
        all_discovered_files.update(discover_files(args.additional_context_folder, CONFIG, args, output_abs_path))
    elif args.additional_context_folder:
        log_warn(f"Additional context folder specified but not found: {args.additional_context_folder}")

    # --- 2. Apply special filtering for backup files ---
    backup_sql_files = []
    other_files = []
    for file_path in all_discovered_files:
        if file_path.parent.name == 'backup' and file_path.suffix.lower() == '.sql':
            backup_sql_files.append(file_path)
        else:
            other_files.append(file_path)

    final_files_to_process = other_files
    if backup_sql_files:
        # Find the latest .sql file in the backup directory by modification time
        latest_backup_file = max(backup_sql_files, key=os.path.getmtime)
        log_info(f"Found {len(backup_sql_files)} SQL backup files. Selecting the latest: {latest_backup_file.name}")
        final_files_to_process.append(latest_backup_file)
        
    # Sort the final list for consistent output
    final_files_to_process.sort()

    # --- 3. Process the final, filtered list of files ---
    output_content_parts = [f"--- START OF FILE {args.output.name} ---\n\n"]
    
    primary_description = "folder" if args.root_path.is_dir() else "file"
    output_content_parts.append(f"# Compressed Representation of Primary {primary_description}: {args.root_path}\n\n")

    for file_path in final_files_to_process:
        try:
            # Safely create a relative path string
            try:
                relative_file_path_str = file_path.relative_to(args.root_path).as_posix()
            except ValueError:
                relative_file_path_str = str(file_path)
                
            log_info(f"Processing file: {relative_file_path_str}")
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if content.strip():
                processed_content = process_file_content(file_path, content, args, CONFIG)
                output_content_parts.append(f"--- File: {relative_file_path_str} ---\n")
                output_content_parts.append(processed_content)
                output_content_parts.append(f"\n--- End File: {relative_file_path_str} ---\n")
        except Exception as e:
            log_error(f"Could not process file {file_path}: {e}")

    output_content_parts.append("\n--- END OF FILE ---")
    final_output_content = "".join(output_content_parts)
    
    try:
        args.output.write_text(final_output_content, encoding='utf-8')
        log_info(f"\nProcessing complete. Output written to {args.output}")
        analyze_output(final_output_content)
    except Exception as e:
        log_error(f"Failed to write to output file {args.output}: {e}")

if __name__ == '__main__':
    main()