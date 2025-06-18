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
# Version 2: Added more aggressive SQL cleaning and better defaults.
CONFIG = {
    "EXCLUDE_DIRS_DEFAULT": {
        '.git', 'venv', '__pycache__', 'node_modules', '.pytest_cache', 'build', 'dist',
        'htmlcov', '.mypy_cache', 'reports', 'backup', 'data',
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
        '.py', '.js', '.jsx', '.css', '.scss', '.md', '.json', '.sql',
        '.ini', '.cfg', '.toml', '.yaml', '.yml', '.sh', '.bat', '.txt',
        '.mako', 'Dockerfile'
    },
    "JSON_SAMPLE_CONFIG": {
        "max_lines": 50,
        "max_chars": 3000
    },
    "COMPLEX_PATTERNS": {
        '.sql': [
            # Pattern 1: Collapse all COPY data blocks (the biggest token saver)
            (
                re.compile(r'(^COPY public\..*? FROM stdin;\n)(.*?)(\n\\\.$\n)', re.DOTALL | re.MULTILINE),
                lambda m: f"{m.group(1)}-- [SQL DATA OMITTED FOR BREVITY]\n{m.group(3)}",
                "SQL_COPY_DATA"
            ),
            # NEW Pattern 2: Remove all SQL comments
            (
                re.compile(r'^--.*?\n', re.MULTILINE),
                "",
                "SQL_COMMENTS"
            ),
            # NEW Pattern 3: Remove all boilerplate SET statements
            (
                re.compile(r'^SET .*?;\n', re.MULTILINE),
                "",
                "SQL_SET_STATEMENTS"
            ),
            # NEW Pattern 4: Remove sequence ownership and value setting
            (
                re.compile(r'^SELECT pg_catalog\.setval.*?;', re.MULTILINE | re.IGNORECASE),
                "",
                "SQL_SETVAL"
            ),
             # NEW Pattern 5: Remove noisy ALTER TABLE ... ADD CONSTRAINT statements
            (
                re.compile(r'^ALTER TABLE .*? ADD CONSTRAINT .*?;\n', re.MULTILINE),
                "",
                "SQL_ADD_CONSTRAINT"
            ),
            # NEW Pattern 6: Remove noisy CREATE INDEX statements
            (
                re.compile(r'^CREATE (UNIQUE )?INDEX .*?;\n', re.MULTILINE),
                "",
                "SQL_CREATE_INDEX"
            ),
            # NEW Pattern 7: Remove noisy ALTER TABLE ... OWNER TO ... statements
            (
                 re.compile(r'^ALTER (TABLE|TYPE|SEQUENCE) .*? OWNER TO .*?;\n', re.MULTILINE),
                 "",
                 "SQL_OWNER_TO"
            ),
             # NEW Pattern 8: Remove sequence anmespace setting
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

def process_path(path: Path, config: dict, args: argparse.Namespace, processed_files: set, output_abs_path: Path):
    """Recursively processes a directory or a single file and returns content parts."""
    content_parts = []
    
    if path.is_file():
        try:
            relative_file_path_str = path.relative_to(args.root_path).as_posix()
        except ValueError:
            relative_file_path_str = str(path)

        if relative_file_path_str in processed_files:
            return []
        processed_files.add(relative_file_path_str)
        log_info(f"Processing single file: {relative_file_path_str}")

        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            if content.strip():
                processed_content = process_file_content(path, content, args, config)
                content_parts.append(f"--- File: {relative_file_path_str} ---\n")
                content_parts.append(processed_content)
                content_parts.append(f"\n--- End File: {relative_file_path_str} ---\n")
        except Exception as e:
            log_error(f"Could not process file {path}: {e}")
        return content_parts

    for root, dirs, files in os.walk(path, topdown=True):
        current_dir = Path(root)
        dirs[:] = [d for d in dirs if d not in config["EXCLUDE_DIRS_DEFAULT"] and d not in args.ignore_dir]
        if not args.include_tests and 'tests' in dirs:
            dirs.remove('tests')
        
        try:
            relative_dir = current_dir.relative_to(path)
        except ValueError:
            relative_dir = current_dir

        if str(relative_dir) != '.':
            content_parts.append(f"========== Directory: {relative_dir.as_posix()} ==========\n")

        files_to_process = []
        for f in files:
            file_path = current_dir / f
            ext = file_path.suffix.lower()
            
            if file_path.resolve() == output_abs_path: continue
            if args.ignore_all_txt and ext == '.txt' and file_path.name.lower() != 'requirements.txt': continue
            if file_path.name in config["EXCLUDE_FILES_DEFAULT"] or file_path.name in args.ignore_file: continue
            if any(p.match(f) for p in config["EXCLUDE_PATTERNS_DEFAULT"]): continue
            if args.ignore_extension and ext in args.ignore_extension: continue
            if ext not in config["INCLUDE_EXT_DEFAULT"]: continue
            
            files_to_process.append(f)
        
        for filename in sorted(files_to_process):
            file_path = current_dir / filename
            try:
                relative_file_path_str = file_path.relative_to(args.root_path).as_posix()
            except ValueError:
                 relative_file_path_str = str(file_path)

            if relative_file_path_str in processed_files: continue
            processed_files.add(relative_file_path_str)

            try:
                log_info(f"Processing file: {relative_file_path_str}")
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if content.strip():
                    processed_content = process_file_content(file_path, content, args, config)
                    content_parts.append(f"--- File: {relative_file_path_str} ---\n")
                    content_parts.append(processed_content)
                    content_parts.append(f"\n--- End File: {relative_file_path_str} ---\n")
            except Exception as e:
                log_error(f"Could not process file {file_path}: {e}")
    
    return content_parts

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
    
    parser.add_argument("--include-tests", action="store_true", help="Include the 'tests' directory in the output.")
    parser.add_argument("--ignore-dir", action="append", default=[], help="Specify directory names to ignore.")
    parser.add_argument("--ignore-file", action="append", default=[], help="Specify file names to ignore.")
    parser.add_argument("--ignore-extension", action="append", default=[], help="Specify file extensions to ignore (e.g., .log).")
    
    # NEW FLAG for ignoring .txt files
    parser.add_argument("--ignore-all-txt", action="store_true", help="Ignore all .txt files except for 'requirements.txt'.")
    
    parser.add_argument("--no-minify-lines", action="store_true", help="Do not remove blank lines or strip whitespace from lines.")
    parser.add_argument("--no-json-sample", action="store_true", help="Include the full content of large JSON files.")
    
    args = parser.parse_args()

    if not args.root_path.exists():
        log_error(f"Path '{args.root_path}' does not exist."); return

    output_abs_path = args.output.resolve()
    output_abs_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_files = set()
    output_content_parts = []
    
    output_content_parts.append(f"--- START OF FILE {args.output.name} ---\n\n")
    
    primary_description = "folder" if args.root_path.is_dir() else "file"
    log_info(f"--- Processing Primary {primary_description.upper()}: {args.root_path} ---")
    output_content_parts.append(f"# Compressed Representation of Primary {primary_description}: {args.root_path}\n\n")
    output_content_parts.extend(
        process_path(args.root_path, CONFIG, args, processed_files, output_abs_path)
    )

    if args.additional_context_folder and args.additional_context_folder.is_dir():
        log_info(f"--- Processing Additional Context Folder: {args.additional_context_folder} ---")
        output_content_parts.append(f"\n# Additional Context from folder: {args.additional_context_folder}\n\n")
        output_content_parts.extend(
            process_path(args.additional_context_folder, CONFIG, args, processed_files, output_abs_path)
        )
    elif args.additional_context_folder:
        log_warn(f"Additional context folder specified but not found: {args.additional_context_folder}")

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