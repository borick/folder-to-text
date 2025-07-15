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
# Version 5.2: Excluded .sql files by default for better compression.

def _summarize_package_json(content: str) -> str:
    """Summarizes package.json to its most essential parts for LLM context."""
    try:
        data = json.loads(content)
        summary = {
            "name": data.get("name"),
            "version": data.get("version"),
            "scripts": data.get("scripts"),
        }
        return (
            f"// Summarized package.json. Key details:\n"
            f"{json.dumps(summary, indent=2)}\n"
            f"// Dependencies are omitted for brevity."
        )
    except json.JSONDecodeError:
        return "// Could not parse package.json, showing raw content.\n" + content

def _summarize_requirements_txt(content: str) -> str:
    """Summarizes requirements.txt to a single line."""
    line_count = len(content.strip().splitlines())
    return f"// Contains {line_count} dependencies, including Flask, SQLAlchemy, etc."


CONFIG = {
    "EXCLUDE_DIRS_DEFAULT": {
        '.git', 'venv', '__pycache__', 'node_modules', '.pytest_cache', 'build', 'dist',
        'htmlcov', '.mypy_cache', 'reports', 'data', 'ui_screenshots', 'backup', # Exclude backup dir entirely
    },
    "EXCLUDE_FILES_DEFAULT": {
        '.DS_Store', '.gitignore', 'package-lock.json', 'yarn.lock',
        '.env', '.env.test', '.env.example', 'jest_detailed_results.json',
        'coverage.xml', '.coverage',
    },
    "EXCLUDE_PATTERNS_DEFAULT": [
        re.compile(r'.*_tokens.*\.txt$'),
        re.compile(r'\.log$'), re.compile(r'\.tmp$'), re.compile(r'\.swp$'),
        re.compile(r'\.zip$'), re.compile(r'\.tar\.gz$'), re.compile(r'\.rar$'),
    ],
    # === THE FIX IS HERE: .sql has been REMOVED from the default includes ===
    "INCLUDE_EXT_DEFAULT": {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.css', '.scss', '.md', '.json',
        '.ini', '.cfg', '.toml', '.yaml', '.yml', '.sh', '.bat', '.txt',
        '.mako', 'Dockerfile'
    },
    "TEST_FILE_PATTERN": re.compile(r'([._])(test|spec)\.(js|jsx|ts|tsx)$|^(test_)|(_test)\.py$', re.IGNORECASE),
    "SUMMARIZE_FILES": {
        'package.json': _summarize_package_json,
        'requirements.txt': _summarize_requirements_txt,
    },
    "JSON_SAMPLE_CONFIG": {
        "max_lines": 50,
        "max_chars": 3000
    },
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
    filename = filepath.name
    ext = filepath.suffix.lower()

    if filename in config["SUMMARIZE_FILES"]:
        return config["SUMMARIZE_FILES"][filename](content)

    if ext == '.json' and not args.no_json_sample:
        try:
            if len(content.splitlines()) > config["JSON_SAMPLE_CONFIG"]["max_lines"] or len(content) > config["JSON_SAMPLE_CONFIG"]["max_chars"]:
                json.loads(content)
                sampled_content = '\n'.join(content.splitlines()[:config["JSON_SAMPLE_CONFIG"]["max_lines"]])
                return f"// Sample representation of the original JSON file.\n{sampled_content}"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if not args.no_minify_lines:
        lines = [line.strip() for line in content.splitlines()]
        content = '\n'.join(filter(None, lines))
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

def _should_exclude(path: Path, args: argparse.Namespace, config: dict, output_abs_path: Path) -> bool:
    """Helper function to determine if a file or directory should be excluded."""
    # Check for forced inclusion first
    if args.force_include:
        for pattern in args.force_include:
            if Path(pattern).match(str(path)):
                return False # Do not exclude, force include

    # Directory checks
    for part in path.parts:
        if part in config["EXCLUDE_DIRS_DEFAULT"] or part in args.ignore_dir:
            return True

    # Test file/directory checks
    if not args.include_tests:
        if any(part == 'tests' for part in path.parts):
            return True
        if config["TEST_FILE_PATTERN"] and config["TEST_FILE_PATTERN"].search(path.name):
            return True

    # Specific file checks
    if path.name in config["EXCLUDE_FILES_DEFAULT"] or path.name in args.ignore_file:
        return True
    if path.resolve() == output_abs_path:
        return True
    if any(p.search(path.name) for p in config["EXCLUDE_PATTERNS_DEFAULT"]):
        return True
        
    # Extension-based checks should ONLY apply to files.
    if path.is_file():
        ext = path.suffix.lower()
        if args.ignore_all_txt and ext == '.txt' and path.name.lower() != 'requirements.txt':
            return True
        if args.ignore_extension and ext in args.ignore_extension:
            return True
        if ext not in config["INCLUDE_EXT_DEFAULT"]:
            return True

    return False

def discover_files(path: Path, config: dict, args: argparse.Namespace, output_abs_path: Path) -> list[Path]:
    """Recursively discovers all files in a path that match the include/exclude criteria."""
    discovered_paths = []
    
    if path.is_file():
        if not _should_exclude(path, args, config, output_abs_path):
            discovered_paths.append(path)
        return discovered_paths

    for root, dirs, files in os.walk(path, topdown=True):
        current_dir = Path(root)
        
        # This is more efficient: filter dirs in-place to prevent os.walk from descending further.
        dirs[:] = [d for d in dirs if not _should_exclude(current_dir / d, args, config, output_abs_path)]
        
        for f in files:
            file_path = current_dir / f
            if not _should_exclude(file_path, args, config, output_abs_path):
                discovered_paths.append(file_path)
            
    return discovered_paths

def main():
    parser = argparse.ArgumentParser(
        description="v5.2: A powerful tool to concatenate project files into a single text file for LLM context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    path_group = parser.add_argument_group("Path Control")
    path_group.add_argument("root_path", type=Path, help="The root directory or specific file to process.")
    path_group.add_argument("-o", "--output", type=Path, default="project_tokens.txt", help="Output file name.")
    path_group.add_argument("--additional-context-folder", type=Path, default=None, help="Optional: A path to an additional folder to include in the context.")

    filter_group = parser.add_argument_group("Filtering Options")
    filter_group.add_argument("--include-tests", action="store_true", help="Include test files (e.g., *_test.py, *.test.js) and 'tests' directories.")
    filter_group.add_argument("--force-include", action="append", default=[], help="File patterns to include even if they would normally be excluded (e.g., '*/important.log').")
    filter_group.add_argument("--ignore-dir", action="append", default=[], help="Additional directory names to ignore.")
    filter_group.add_argument("--ignore-file", action="append", default=[], help="Additional file names to ignore.")
    filter_group.add_argument("--ignore-extension", action="append", default=[], help="File extensions to ignore (e.g., .log).")
    filter_group.add_argument("--ignore-all-txt", action="store_true", help="Ignore all .txt files except for 'requirements.txt'.")
    
    format_group = parser.add_argument_group("Output Formatting")
    format_group.add_argument("--no-minify-lines", action="store_true", help="Do not remove blank lines or strip whitespace from lines.")
    format_group.add_argument("--no-json-sample", action="store_true", help="Include the full content of large JSON files.")

    args = parser.parse_args()

    if not args.root_path.exists():
        log_error(f"Path '{args.root_path}' does not exist."); return

    output_abs_path = args.output.resolve()
    output_abs_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_discovered_files = set()
    log_info(f"--- Discovering files in Primary Path: {args.root_path} ---")
    all_discovered_files.update(discover_files(args.root_path, CONFIG, args, output_abs_path))

    if args.additional_context_folder:
        if args.additional_context_folder.exists():
            log_info(f"--- Discovering files in Additional Context Folder: {args.additional_context_folder} ---")
            all_discovered_files.update(discover_files(args.additional_context_folder, CONFIG, args, output_abs_path))
        else:
            log_warn(f"Additional context folder specified but not found: {args.additional_context_folder}")

    # No longer need special handling for backups, they are excluded by default.
    final_files_to_process = sorted(list(all_discovered_files))
        
    with args.output.open('w', encoding='utf-8') as f_out:
        f_out.write(f"--- START OF FILE {args.output.name} ---\n\n")
        primary_desc = "folder" if args.root_path.is_dir() else "file"
        f_out.write(f"# Compressed Representation of Primary {primary_desc}: {args.root_path}\n\n")

        for file_path in final_files_to_process:
            try:
                relative_path_str = str(file_path)
                try:
                    # Attempt to make path relative to CWD for cleaner output if possible
                    relative_path_str = str(file_path.relative_to(Path.cwd()))
                except ValueError:
                    # If it's on a different drive or path, use the absolute path
                    pass

                log_info(f"Processing file: {relative_path_str}")
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if content.strip():
                    processed_content = process_file_content(file_path, content, args, CONFIG)
                    f_out.write(f"--- File: {relative_path_str} ---\n")
                    f_out.write(processed_content)
                    f_out.write(f"\n--- End File: {relative_path_str} ---\n")
            except Exception as e:
                log_error(f"Could not process file {file_path}: {e}")

        f_out.write("\n--- END OF FILE ---")

    log_info(f"\nProcessing complete. Output written to {args.output}")
    with args.output.open('r', encoding='utf-8') as f_in:
        analyze_output(f_in.read())

if __name__ == '__main__':
    main()