#!/usr/bin/env python3

# --- folder_to_text.py ---
# Processes a folder, simplifying text files (standard mode), summarizing others,
# compressing repetitive patterns/lines (standard mode), or dumping raw content.
# Includes standard mode, raw dump (all files), and raw source dump (source files only).
# Provides summary reports both to stderr and *within* the output file.
# Includes approximate origin tracking for post-process cleanup.
# Explicitly ignores .env files. Allows including test files/dirs via flag.

# --- Example Usage ---

# 1. Default (Max Reduction - Recommended, Excludes Tests):
#    python folder_to_text.py /path/to/project -o max_compressed_cleaned.txt

# 2. Include Tests (Standard Mode):
#    python folder_to_text.py /path/to/project --include-tests -o standard_with_tests.txt

# 3. Raw Source Dump (Including Tests):
#    python folder_to_text.py /path/to/project --raw-dump-source --include-tests -o raw_source_with_tests.txt

# 4. Raw Dump (Concatenate ALL files verbatim - Ignores --include-tests as it dumps all):
#    python folder_to_text.py /path/to/project --raw-dump -o raw_dump.txt



import os
import re
import argparse
import sys
import string
from pathlib import Path
from collections import Counter
import logging
import hashlib
import io
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

FILE_HEADER_RE = re.compile(r"^--- File: (.*) ---$")

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(funcName)s: %(message)s', stream=sys.stderr)
log = logging.getLogger(__name__)


# --- Configuration ---

# Base ignore patterns (generally excluded)
BASE_IGNORE_PATTERNS = {
    '.git', '__pycache__', '.svn', '.hg', '.idea', '.vscode', 'node_modules',
    'build', 'dist', 'target', 'venv', '.venv', 'env', 'envs', 'conda', 'anaconda3', 'AppData',
    '.DS_Store', 'Thumbs.db',
    # Common binary/archive/media types
    '*.zip', '*.gz', '*.tar', '*.rar', '*.7z',
    '*.class', '*.jar', '*.exe', '*.dll', '*.so', '*.o', '*.a', '*.lib', '*.pyc', '*.pyo',
    '*.swp',
    '*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg', '*.ico', '*.bmp', '*.tif', '*.tiff',
    '*.pdf', '*.doc', '*.docx', '*.xls', '*.xlsx', '*.ppt', '*.pptx',
    '*.mp3', '*.mp4', '*.avi', '*.mov', '*.wmv', '*.flv', '*.wav', '*.ogg', '*.opus',
    # Lock files
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'composer.lock', 'go.sum', 'Gemfile.lock',
    # Minified files
    '*.min.js', '*.min.css',
    # Sensitive/local config
    '.env',
    # Temporary/backup files
    '*.bak', '*.old', '*.tmp', '*~',
    # Logs
    '*.log',
}

# Test-related patterns (conditionally ignored)
TEST_IGNORE_PATTERNS = {
    'test', 'tests', '__tests__', 'test_data', 'fixtures', # Common directory names
    '*.test.*', 'test_*.py', '*_test.py', '*.spec.*', # Common file patterns (py, js, ts, etc.)
    'Test*.java', '*Test.java', '*Tests.java', # Java patterns
    '*Test.kt', '*Tests.kt', # Kotlin patterns
    '*Tests.cs', # C# pattern
    '*_test.go', # Go pattern
}

# Combine base and test patterns for the initial default full ignore set
DEFAULT_IGNORE_PATTERNS = BASE_IGNORE_PATTERNS.union(TEST_IGNORE_PATTERNS)

# Files considered source code or important text config
CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.kt', '.swift',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rb', '.php', '.sh',
    '.bash', '.zsh', '.ps1', '.psm1',
    '.css', '.scss', '.less', '.sass',
    '.html', '.htm', '.xml', '.json', '.yaml', '.yml', '.toml',
    '.sql', '.md', '.rst', '.tex',
    '.dockerfile', 'Dockerfile',
    '.r', '.pl', '.pm', '.lua',
    '.gradle', '.tf', '.tfvars', '.hcl',
    '.conf', '.ini', '.cfg', '.properties', '.config', '.settings',
    'Makefile', 'Jenkinsfile', 'Gemfile', 'pom.xml', 'build.gradle', 'settings.gradle',
    '.ipynb', '.env.example',
}
# Specific filenames often containing important config or documentation
INTERESTING_FILENAMES = {
    'README', 'LICENSE', 'CONTRIBUTING', 'CHANGELOG', 'SECURITY', 'AUTHORS', 'COPYING',
    'requirements.txt', 'Pipfile', 'setup.py', 'pyproject.toml',
    'docker-compose.yml', 'docker-compose.yaml', 'compose.yaml', 'compose.yml',
    'package.json', 'tsconfig.json', 'angular.json', 'vue.config.js', 'webpack.config.js',
    'go.mod', 'go.work',
    '.gitignore', '.dockerignore', '.editorconfig', '.gitattributes', '.gitlab-ci.yml', '.travis.yml',
}
BINARY_CHECK_BYTES = 1024
BINARY_NON_PRINTABLE_THRESHOLD = 0.15

# --- Config specific to Standard Mode ---
SIMPLIFICATION_PATTERNS = [
    (re.compile(r'\d{8,}'), '*NUM_LONG*'),
    (re.compile(r'\b[a-fA-F0-9]{12,}\b'), '*HEX_LONG*'),
    (re.compile(r'[a-zA-Z0-9+/=]{30,}'), '*BASE64LIKE_LONG*'),
]
POST_SIMPLIFICATION_PATTERNS = [
    (re.compile(r"""(['"])\b([a-fA-F0-9]{8}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{12})\b\1"""), r'"*UUID*"'),
    (re.compile(r"""(['"])(https?://[^/'"]+)(/[^'"\s]+/?)(\w+\.(?:py|js|html|css|png|jpg|gif|svg|json|xml|yaml|yml))\1"""), r'"\g<2>/*PATH*/\g<4>"'),
    (re.compile(r"""(['"])(https?://[^/'"]+)(/[^'"\s]+)\1"""), r'"\g<2>/*PATH*"'),
    (re.compile(r"""(['"])((?:\\.|[^\\\1])*?)\1"""), lambda m: '...' if len(m.group(2)) > 10 else m.group(0)),
]
SINGLE_VOICE_ID_FINDER = re.compile(r"""("([a-z]{2,3}-[A-Z][a-zA-Z0-9]{1,3}(?:-(?:[A-Z]{2}|[a-zA-Z0-9]+))?-[A-Z][a-zA-Z0-9]+Neural)",?)""", re.VERBOSE)
BLOCK_COMPRESSION_PATTERNS = {
    "VOICE_ID": re.compile(r"""^\s*"([a-z]{2,3}-[A-Z][a-zA-Z0-9]{1,3}(?:-(?:[A-Z]{2}|[a-zA-Z0-9]+))?-[A-Z][a-zA-Z0-9]+Neural)",?\s*$""", re.VERBOSE),
    "QUOTED_UUID": re.compile(r"""^\s*(['"])[a-fA-F0-9]{8}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{12}\1,?$"""),
}
DEFAULT_MIN_CONSECUTIVE_LINES = 3
DEFINITION_SIMPLIFICATION_PATTERNS = [
    (re.compile(r'\b\d{2,}\.\d{1,}|\d{1,}\.\d{2,}\b'), '*FLOAT*'),
    (re.compile(r'\b\d{3,}\b'), '*INT*'),
    (re.compile(r"""(['"])((?:\\.|[^\\\1])*?)\1"""), '...'),
    (re.compile(r'\b[a-fA-F0-9]{10,}\b'), '*HEX*'),
    (re.compile(r'[a-zA-Z0-9+/=]{20,}'), '*BASE64LIKE*'),
]

DEFAULT_LARGE_LITERAL_THRESHOLD = 5

ALL_PLACEHOLDERS = [
    r'\*...\*', r'\*INT\*', r'\*FLOAT\*', r'\*UUID\*',
    r'\*NUM_LONG\*', r'\*HEX_LONG\*', r'\*BASE64LIKE_LONG\*',
    r'\*HEX\*', r'\*BASE64LIKE\*',
    r'\/\*PATH\*\/?'
]
PLACEHOLDER_CLEANUP_PATTERN = re.compile(
    r"""^                 # Start of line
       \s*                # Optional leading whitespace
       (?:                # Non-capturing group for the allowed elements
          (?:{placeholder_group})  # Any of the defined placeholders
          |                 # OR
          [,\[\]{{}}]        # Commas or brackets/braces
          |                 # OR
          \s                # Whitespace itself
       )+                 # One or more of these allowed elements
       \s*                # Optional trailing whitespace
       $                  # End of line
    """.format(placeholder_group='|'.join(ALL_PLACEHOLDERS)),
    re.VERBOSE
)

# --- Globals ---
seen_content_hashes = set()

# --- Helper Functions ---
def is_likely_binary(filepath: Path) -> bool:
    try:
        with open(filepath, 'rb') as f: chunk = f.read(BINARY_CHECK_BYTES)
        if not chunk: return False
        if b'\0' in chunk: return True
        printable_bytes = set(bytes(string.printable, 'ascii'))
        non_printable_count = sum(1 for byte in chunk if byte not in printable_bytes)
        chunk_len = len(chunk); non_printable_ratio = (non_printable_count / chunk_len) if chunk_len > 0 else 0
        return non_printable_ratio > BINARY_NON_PRINTABLE_THRESHOLD
    except Exception as e: log.warning(f"Binary check error for {filepath}: {e}"); return False

def get_file_id_from_path_str(path_str: str) -> str:
    try: p = Path(path_str); suffix = p.suffix.lower(); name = p.name.lower(); return suffix if suffix else name
    except Exception: log.warning(f"Could not parse file ID from path: {path_str}"); return "<unknown>"

def summarize_other_files(filenames: list[str], code_extensions: set, interesting_filenames: set) -> str:
    if not filenames: return ""
    ext_counts = Counter(); explicit_files = set(); interesting_lower = {fn.lower() for fn in interesting_filenames}; code_ext_lower = {ext.lower() for ext in code_extensions}
    for fname in filenames:
        base, ext = os.path.splitext(fname); fname_lower = fname.lower(); ext_lower = ext.lower()
        is_interesting = (fname_lower in interesting_lower or
                          base.lower() in interesting_lower or
                          fname in code_extensions or
                          base in code_extensions)
        if is_interesting: explicit_files.add(fname)
        elif ext_lower and ext_lower not in code_ext_lower: ext_counts[ext_lower] += 1
        elif not ext and base and fname_lower not in interesting_lower and base.lower() not in interesting_lower: ext_counts['<no_ext>'] += 1
    summary_parts = []; summary_parts.extend(sorted(list(explicit_files)))
    sorted_ext_counts = sorted([item for item in ext_counts.items() if item[0] not in code_ext_lower], key=lambda x: x[0])
    for ext, count in sorted_ext_counts: summary_parts.append(f"{count}x {ext}" if count > 1 else f"1x {ext}")
    if not summary_parts: return f"{len(filenames)} other file(s)"
    return ", ".join(summary_parts)

def generate_output_file_summary(report_data: Dict[str, Counter], mode: str) -> str:
    """ Generates a formatted string block summarizing included file types for the output file header. """
    summary_lines = []; total_files = 0; included_types = {}
    if mode == "Standard":
        summary_lines.append("# --- Included File Types Summary (Standard Mode) ---")
        for file_id, stats in report_data.items():
            count = stats.get('contributed', 0)
            if count > 0: included_types[file_id] = count; total_files += count
    elif mode in ["Raw Source Dump", "Raw Dump (All Files)"]:
        header_text = "Source File" if mode == "Raw Source Dump" else "File"
        summary_lines.append(f"# --- Included {header_text} Types Summary ({mode}) ---")
        for file_id, stats in report_data.items():
            count = stats.get('dumped', 0)
            if count > 0: included_types[file_id] = count; total_files += count
    else: return ""
    if not included_types: summary_lines.append("#   (No files included in output)")
    else:
        sorted_types = sorted(included_types.items()); max_len = max((len(fid) for fid in included_types.keys()), default=10)
        for file_id, count in sorted_types:
            display_id = file_id if file_id else '<no_ext>'
            summary_lines.append(f"#   - {display_id:<{max_len}} : {count:>4}")
        summary_lines.append(f"# Total Files Included: {total_files}")
    summary_lines.append("# --- End Summary ---")
    return "\n".join(summary_lines) + "\n\n"

# <<< Helper Function for finding file context (used by minification & cleanup reports) >>>
def create_file_context_finder(lines_list: List[str]) -> Callable[[int], str]:
    """Creates a closure to find the preceding file context for a given line index."""
    file_context_cache: Dict[int, str] = {}
    lines_ref = lines_list # Keep a reference to the list

    def find_file_context(target_index: int) -> str:
        if target_index < 0 or target_index >= len(lines_ref):
             log.warning(f"Index {target_index+1} out of bounds for context finding.")
             return "<index_error>"
        cached_id = file_context_cache.get(target_index)
        if cached_id: return cached_id
        # Search backwards from the target index
        for k in range(target_index, -1, -1):
            if k >= len(lines_ref): continue # Should not happen, but safety check
            line_to_check = lines_ref[k]
            match = FILE_HEADER_RE.match(line_to_check)
            if match:
                filepath_str = match.group(1).strip()
                file_id = get_file_id_from_path_str(filepath_str)
                # Cache for the target index and potentially intermediate indices
                file_context_cache[target_index] = file_id
                return file_id
            # Optimization: If we hit another cached entry while searching backwards, use it
            # (Avoids redundant searches for lines within the same file block)
            cached_id_backward = file_context_cache.get(k)
            if cached_id_backward:
                 file_context_cache[target_index] = cached_id_backward
                 return cached_id_backward

        log.warning(f"Could not find file header preceding line index {target_index+1}")
        file_context_cache[target_index] = "<unknown_context>" # Cache unknown result
        return "<unknown_context>"

    return find_file_context


# --- Core Processing Functions ---

def simplify_source_code(
    content: str,
    strip_logging: bool,
    large_literal_threshold: int,
    disable_literal_compression: bool
) -> str:
    if not content.strip(): return ""
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'"""(?:.|\n)*?"""', '', content, flags=re.MULTILINE)
    content = re.sub(r"'''(?:.|\n)*?'''", '', content, flags=re.MULTILINE)

    lines = content.splitlines()
    simplified_lines = []
    in_literal_block = False
    literal_block_start_index = -1
    literal_line_count = 0
    literal_line_pattern = re.compile(r"""^\s*(?:(?:r|u|f|b)?(['"])(?:(?=(\\?))\2.)*?\1|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|True|False|None|\*NUM_LONG\*|\*HEX_LONG\*|\*BASE64LIKE_LONG\*|\*UUID\*|\*INT\*|\*FLOAT\*|\*...\*)\s*[,\]\}]?\s*$""", re.VERBOSE)
    list_dict_start_pattern = re.compile(r'[:=]\s*(\[|\{)\s*$')
    for i, line in enumerate(lines):
        original_line = line
        line = re.sub(r'(?<![:/])#.*$', '', line)
        line = re.sub(r'\s+//.*$', '', line)
        line = re.sub(r'^\s*//.*$', '', line)
        line = re.sub(r'\s+--.*$', '', line)
        line = re.sub(r'^\s*--.*$', '', line)
        stripped_line = line.strip()
        if strip_logging and stripped_line:
            if re.match(r'^(?:log|logger|logging|console|print|fmt\.Print|System\.out\.print|TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\b', stripped_line, re.IGNORECASE):
                 if '(' in stripped_line and ')' in stripped_line: continue
        is_potential_start = list_dict_start_pattern.search(line) is not None
        is_end = stripped_line.startswith(']') or stripped_line.startswith('}')
        current_line_index = len(simplified_lines)
        if is_potential_start and not in_literal_block:
            in_literal_block = True; literal_block_start_index = current_line_index; literal_line_count = 0; simplified_lines.append(original_line)
        elif in_literal_block:
            is_simple_literal_line = literal_line_pattern.match(stripped_line) is not None
            if is_simple_literal_line: literal_line_count += 1; simplified_lines.append(original_line)
            elif is_end:
                simplified_lines.append(original_line)
                if not disable_literal_compression and literal_line_count >= large_literal_threshold and literal_block_start_index >= 0:
                    start_slice_idx = literal_block_start_index + 1; end_slice_idx = start_slice_idx + literal_line_count
                    if start_slice_idx < end_slice_idx <= len(simplified_lines)-1:
                         try: indent_line = simplified_lines[start_slice_idx] if start_slice_idx < len(simplified_lines) else simplified_lines[literal_block_start_index]; indent_level = indent_line.find(indent_line.lstrip()) if indent_line.strip() else 2
                         except IndexError: indent_level = 2
                         indent = " " * (indent_level); placeholder_line = f"{indent}# --- Large literal collection compressed ({literal_line_count} lines) ---"
                         del simplified_lines[start_slice_idx:end_slice_idx]; simplified_lines.insert(start_slice_idx, placeholder_line); log.debug(f"Compressed literal block, {literal_line_count} lines.")
                    else: log.warning(f"Compression slice error: start={start_slice_idx}, end={end_slice_idx}, len={len(simplified_lines)}. Skipping.")
                in_literal_block = False; literal_block_start_index = -1; literal_line_count = 0
            else:
                log.debug(f"Complex line ended potential literal block compression."); simplified_lines.append(original_line)
                in_literal_block = False; literal_block_start_index = -1; literal_line_count = 0
        elif stripped_line: simplified_lines.append(line)
        elif simplified_lines and simplified_lines[-1].strip(): simplified_lines.append("")
    processed_content = "\n".join(simplified_lines).strip()
    if processed_content: processed_content += "\n"
    for pattern, replacement in SIMPLIFICATION_PATTERNS: processed_content = pattern.sub(replacement, processed_content)
    processed_content = re.sub(r'\n{3,}', '\n\n', processed_content)
    return processed_content

def process_folder(
    target_folder: str, buffer: io.StringIO, ignore_set: set, code_extensions_set: set, interesting_filenames_set: set,
    skip_empty: bool, strip_logging: bool, skip_duplicates: bool, large_literal_threshold: int, compress_patterns_enabled: bool,
    report_data: Dict[str, Counter]
):
    """ Processes the folder using simplification, filtering, etc. (Standard Mode). """
    global seen_content_hashes; target_path = Path(target_folder).resolve()
    def write_to_buffer(text): buffer.write(text + "\n")
    log.debug(f"Starting folder walk: {target_path} (Standard Mode)"); log.debug(f"Skips: empty={skip_empty}, duplicates={skip_duplicates}")
    for dirpath, dirnames, filenames in os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} - {e}")):
        current_path = Path(dirpath)
        try: current_rel_path = current_path.relative_to(target_path)
        except ValueError: log.warning(f"Cannot get relative path for {current_path}. Skipping."); dirnames[:] = []; continue
        original_dir_count = len(dirnames)
        dirnames[:] = [d for d in dirnames if d not in ignore_set and not d.startswith('.') and not any(Path(d).match(p) for p in ignore_set if '*' in p or '?' in p)]
        if len(dirnames) < original_dir_count: log.debug(f"Filtered {original_dir_count - len(dirnames)} subdirectories in {current_rel_path}")
        log.debug(f"Processing directory: {current_rel_path} (Files: {len(filenames)}, Subdirs: {len(dirnames)})")
        filenames.sort(); dirnames.sort()
        source_files_to_process = []; other_text_filenames = []
        for filename in filenames:
            if filename in ignore_set or any(Path(filename).match(p) for p in ignore_set if '*' in p or '?' in p): continue
            if filename.startswith('.') and filename.lower() not in interesting_filenames_set and filename not in code_extensions_set: continue
            file_path = current_path / filename
            if not file_path.is_file(): continue
            if is_likely_binary(file_path): log.debug(f"Skipping binary file: {file_path.relative_to(target_path)}"); continue
            base, ext = os.path.splitext(filename); ext_lower = ext.lower(); fname_lower = filename.lower()
            is_source = (ext_lower in code_extensions_set or filename in code_extensions_set or fname_lower in code_extensions_set)
            relative_file_path = file_path.relative_to(target_path)
            if is_source: source_files_to_process.append((relative_file_path, file_path, ext_lower))
            else: other_text_filenames.append(filename)
        if source_files_to_process:
            dir_header = f"\n{'=' * 10} Directory: {current_rel_path} {'=' * 10}\n" if str(current_rel_path) != '.' else ""
            if dir_header: write_to_buffer(dir_header)
            for rel_path, full_path, file_ext in source_files_to_process:
                file_info_prefix = f"--- File: {rel_path}"; file_id_for_report = get_file_id_from_path_str(rel_path.name)
                if file_id_for_report not in report_data: report_data[file_id_for_report] = Counter()
                report_data[file_id_for_report]['processed'] += 1
                try:
                    log.debug(f"Processing source file: {rel_path}")
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                    hash_before = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, compress_patterns_enabled)
                    hash_after = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest()
                    is_simplified = hash_before != hash_after; is_empty_after_simplification = not simplified_content.strip()
                    content_hash = hash_after; is_duplicate = False
                    if is_simplified and not is_empty_after_simplification: report_data[file_id_for_report]['simplified'] += 1
                    if skip_empty and is_empty_after_simplification: log.info(f"Skipping empty file: {rel_path}"); report_data[file_id_for_report]['skipped_empty'] += 1; continue
                    if skip_duplicates and not is_empty_after_simplification:
                        if content_hash in seen_content_hashes: is_duplicate = True; log.info(f"Skipping duplicate file: {rel_path}"); report_data[file_id_for_report]['skipped_duplicate'] += 1; continue
                        else: seen_content_hashes.add(content_hash); log.debug(f"Adding new hash: {content_hash[:8]}... for {rel_path}")
                    report_data[file_id_for_report]['contributed'] += 1
                    write_to_buffer(file_info_prefix + " ---")
                    if is_empty_after_simplification: write_to_buffer("# (File empty after simplification)")
                    else: write_to_buffer(simplified_content.strip())
                    write_to_buffer(f"--- End File: {rel_path} ---"); write_to_buffer("")
                except OSError as e: log.error(f"Read error: {rel_path}: {e}"); report_data[file_id_for_report]['read_error'] += 1; write_to_buffer(f"{file_info_prefix} Error Reading ---\n### Error: {e} ###\n--- End File: {rel_path} ---\n")
                except Exception as e: log.error(f"Processing error: {rel_path}: {e}", exc_info=True); report_data[file_id_for_report]['processing_error'] += 1; write_to_buffer(f"{file_info_prefix} Error Processing ---\n### Error: {e} ###\n--- End File: {rel_path} ---\n")
        if other_text_filenames:
            summary = summarize_other_files(other_text_filenames, code_extensions_set, interesting_filenames_set)
            if summary:
                indent_level = len(current_rel_path.parts); indent = "  " * indent_level if indent_level > 0 else ""
                summary_line = f"{indent}# Other files in '{current_rel_path}': {summary}"
                log.debug(f"Adding summary: {summary_line}"); buffer.seek(0, io.SEEK_END); current_pos = buffer.tell()
                if current_pos > 0:
                    buffer.seek(max(0, current_pos - 200)); last_part = buffer.read()
                    if not last_part.strip().endswith(f"Directory: {current_rel_path} {'=' * 10}") and not last_part.endswith("\n\n"): write_to_buffer("")
                write_to_buffer(summary_line); write_to_buffer("")

def process_folder_raw(target_folder: str, buffer: io.StringIO, report_data: Dict[str, Counter]) -> int:
    """ Walks and dumps all files verbatim. Populates report_data. Returns file count. """
    target_path = Path(target_folder).resolve(); log.debug(f"Starting RAW folder walk: {target_path}")
    file_count = 0; walk_results = sorted(list(os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Access error {e.filename}: {e}"))), key=lambda x: x[0])
    for dirpath, dirnames, filenames in walk_results:
        dirnames.sort(); filenames.sort(); current_path = Path(dirpath)
        try: current_rel_path = current_path.relative_to(target_path)
        except ValueError: log.warning(f"Relative path error for {current_path}. Skipping."); continue
        log.debug(f"Processing directory (RAW): {current_rel_path} (Files: {len(filenames)})")
        for filename in filenames:
            file_path = current_path / filename; relative_file_path = file_path.relative_to(target_path)
            if not file_path.is_file(): continue
            file_count += 1; file_id = get_file_id_from_path_str(filename)
            if file_id not in report_data: report_data[file_id] = Counter()
            report_data[file_id]['dumped'] += 1
            buffer.write(f"\n{'=' * 20} START FILE: {relative_file_path} {'=' * 20}\n")
            log.info(f"Dumping file ({file_count}): {relative_file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                if not content.endswith('\n'): content += '\n'; buffer.write(content)
            except OSError as e: log.error(f"Read error: {relative_file_path}: {e}"); report_data[file_id]['read_error'] += 1; buffer.write(f"### Error reading file: {e} ###\n")
            except Exception as e: log.error(f"Processing error: {relative_file_path}: {e}", exc_info=True); report_data[file_id]['processing_error'] += 1; buffer.write(f"### Error processing file: {e} ###\n")
            buffer.write(f"{'=' * 20} END FILE: {relative_file_path} {'=' * 20}\n")
    log.info(f"Finished raw dump. Processed {file_count} files.")
    return file_count

def process_folder_raw_source(
    target_folder: str, buffer: io.StringIO, ignore_set: set, code_extensions_set: set, interesting_filenames_set: set,
    report_data: Dict[str, Counter]
) -> int:
    """ Walks, filters for source files, dumps verbatim. Populates report_data. Returns file count. """
    target_path = Path(target_folder).resolve(); log.debug(f"Starting RAW SOURCE folder walk: {target_path}")
    source_file_count = 0; walk_results = sorted(list(os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Access error {e.filename}: {e}"))), key=lambda x: x[0])
    for dirpath, dirnames, filenames in walk_results:
        current_path = Path(dirpath)
        try: current_rel_path = current_path.relative_to(target_path)
        except ValueError: log.warning(f"Relative path error for {current_path}. Skipping."); dirnames[:]= []; continue
        original_dir_count = len(dirnames)
        dirnames[:] = [d for d in dirnames if d not in ignore_set and not d.startswith('.') and not any(Path(d).match(p) for p in ignore_set if '*' in p or '?' in p)]
        if len(dirnames) < original_dir_count: log.debug(f"Filtered {original_dir_count - len(dirnames)} subdirectories in {current_rel_path}")
        log.debug(f"Processing directory (RAW SOURCE): {current_rel_path} (Files: {len(filenames)}, Subdirs: {len(dirnames)})")
        filenames.sort(); dirnames.sort()
        for filename in filenames:
            if filename in ignore_set or any(Path(filename).match(p) for p in ignore_set if '*' in p or '?' in p): continue
            if filename.startswith('.') and filename.lower() not in interesting_filenames_set and filename not in code_extensions_set: continue
            file_path = current_path / filename
            if not file_path.is_file(): continue
            if is_likely_binary(file_path): log.debug(f"Skipping binary file: {file_path.relative_to(target_path)}"); continue
            base, ext = os.path.splitext(filename); ext_lower = ext.lower(); fname_lower = filename.lower()
            is_source = (ext_lower in code_extensions_set or filename in code_extensions_set or fname_lower in code_extensions_set)
            relative_file_path_in_scan = file_path_abs.relative_to(folder_to_scan)
            if is_source: source_files_to_process.append((relative_file_path_in_scan, file_path_abs))
            else: other_text_filenames.append(filename)
        if source_files_to_process:
            dir_header_display_path = str(current_rel_path_in_scan) if str(current_rel_path_in_scan) != '.' else folder_to_scan.name
            dir_header = f"\n{'=' * 10} Directory ({log_prefix.strip(': ')}): {dir_header_display_path} {'=' * 10}\n"
            if dir_header: write_to_buffer(dir_header)
            for rel_path_in_scan, full_path_abs in source_files_to_process:
                display_path_for_marker = f"{folder_to_scan.name}/{rel_path_in_scan}" if is_additional_context and str(folder_to_scan.name) != "." else str(rel_path_in_scan)
                file_info_prefix = f"--- File: {display_path_for_marker}"
                try:
                    log.debug(f"{log_prefix}Processing source file: {full_path_abs}")
                    with open(full_path_abs, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                    simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, disable_literal_compression_in_simplify)
                    is_empty_after_simplification = not simplified_content.strip()
                    if skip_empty and is_empty_after_simplification: log.info(f"{log_prefix}Skipping empty file (after simplification): {full_path_abs}"); continue 
                    if skip_duplicates and not is_empty_after_simplification:
                        content_hash = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest()
                        if content_hash in seen_content_hashes: log.info(f"{log_prefix}Skipping duplicate content file: {full_path_abs} (Hash: {content_hash[:8]}...)"); continue 
                        seen_content_hashes.add(content_hash); log.debug(f"{log_prefix}Adding new content hash: {content_hash[:8]}... for {full_path_abs}")
                    write_to_buffer(file_info_prefix + " ---")
                    if is_empty_after_simplification: write_to_buffer("# (File is empty or contained only comments/whitespace/logging)")
                    else: write_to_buffer(simplified_content.strip())
                    write_to_buffer(f"--- End File: {display_path_for_marker} ---"); write_to_buffer("") 
                except OSError as e:
                    log.error(f"{log_prefix}Error reading file {full_path_abs}: {e}")
                    write_to_buffer(file_info_prefix + " Error Reading ---"); write_to_buffer(f"### Error reading file: {type(e).__name__}: {e} ###"); write_to_buffer(f"--- End File: {display_path_for_marker} ---"); write_to_buffer("")
                except Exception as e:
                    log.error(f"{log_prefix}Error processing file {full_path_abs}: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
                    write_to_buffer(file_info_prefix + " Error Processing ---"); write_to_buffer(f"### Error processing file: {type(e).__name__}: {e} ###"); write_to_buffer(f"--- End File: {display_path_for_marker} ---"); write_to_buffer("")
        if other_text_filenames:
            summary = summarize_other_files(other_text_filenames, code_extensions_set, interesting_filenames_set)
            if summary:
                indent_level = len(current_rel_path_in_scan.parts) if str(current_rel_path_in_scan) != '.' else 0
                indent = "  " * indent_level if indent_level > 0 else ""
                summary_line = f"{indent}# Other files in '{current_rel_path_in_scan}': {summary}"
                log.debug(f"{log_prefix}Adding summary for {current_rel_path_in_scan}: {summary}")
                buffer.seek(0, io.SEEK_END); current_pos = buffer.tell()
                if current_pos > 0:
                    buffer.seek(max(0, current_pos - 300)); last_part = buffer.read()
                    dir_header_display_path = str(current_rel_path_in_scan) if str(current_rel_path_in_scan) != '.' else folder_to_scan.name
                    if not last_part.strip().endswith(f"Directory ({log_prefix.strip(': ')}): {dir_header_display_path} {'=' * 10}") and not last_part.endswith("\n\n"): write_to_buffer("") 
                write_to_buffer(summary_line); write_to_buffer("")

def process_folder_raw(target_input_path_str: str, buffer: io.StringIO, additional_context_folder_str: Optional[str] = None):
    paths_to_process = [(Path(target_input_path_str).resolve(), "Primary Input")]
    if additional_context_folder_str:
        additional_path = Path(additional_context_folder_str).resolve()
        if additional_path.is_dir(): paths_to_process.append((additional_path, "Additional Context Folder"))
        else: log.warning(f"Raw Dump: Additional context path '{additional_path}' is not a directory, skipping.")

    for current_target_path_obj, current_target_desc in paths_to_process:
        log.debug(f"Starting RAW dump for {current_target_desc}: {current_target_path_obj}"); file_count_for_target = 0
        if current_target_path_obj.is_file():
            buffer.write(f"\n{'=' * 20} START FILE ({current_target_desc}): {current_target_path_obj.name} {'=' * 20}\n")
            try:
                with open(current_target_path_obj, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                if content and not content.endswith('\n'): content += '\n'
                buffer.write(content); file_count_for_target = 1
            except Exception as e: log.error(f"Error reading/writing file {current_target_path_obj.name} for raw dump: {e}"); buffer.write(f"### Error during raw dump of {current_target_path_obj.name}: {e} ###\n")
            buffer.write(f"{'=' * 20} END FILE ({current_target_desc}): {current_target_path_obj.name} {'=' * 20}\n")
        elif current_target_path_obj.is_dir():
            buffer.write(f"\n{'=' * 10} STARTING RAW DUMP OF {current_target_desc.upper()}: {current_target_path_obj.name} {'=' * 10}\n")
            walk_results = sorted(list(os.walk(current_target_path_obj, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} - {e}"))), key=lambda x: x[0])
            for dirpath, dirnames, filenames in walk_results:
                dirnames.sort(); filenames.sort(); current_path_in_walk = Path(dirpath)
                for filename_in_walk in filenames:
                    file_path_in_walk = current_path_in_walk / filename_in_walk
                    relative_file_path_display = file_path_in_walk.relative_to(current_target_path_obj)
                    if not file_path_in_walk.is_file(): continue
                    file_count_for_target +=1
                    buffer.write(f"\n{'=' * 20} START FILE ({current_target_desc}/{relative_file_path_display}): {filename_in_walk} {'=' * 20}\n")
                    try:
                        with open(file_path_in_walk, 'r', encoding='utf-8', errors='ignore') as f: content_f = f.read()
                        if content_f and not content_f.endswith('\n'): content_f += '\n'
                        buffer.write(content_f)
                    except Exception as e: log.error(f"Error reading/writing file {file_path_in_walk} for raw dump: {e}"); buffer.write(f"### Error during raw dump of {file_path_in_walk}: {e} ###\n")
                    buffer.write(f"{'=' * 20} END FILE ({current_target_desc}/{relative_file_path_display}): {filename_in_walk} {'=' * 20}\n")
            buffer.write(f"\n{'=' * 10} FINISHED RAW DUMP OF {current_target_desc.upper()}: {current_target_path_obj.name} ({file_count_for_target} files) {'=' * 10}\n")
        else: log.error(f"Raw dump target '{current_target_path_obj}' from {current_target_desc} is neither a file nor a directory."); buffer.write(f"### ERROR: Raw dump target '{current_target_path_obj}' is not valid. ###\n")
        log.info(f"Finished raw dump for {current_target_desc}. Processed {file_count_for_target} file(s).")

# --- Post-Processing Functions ---
def apply_post_simplification_patterns(content: str, patterns: list[tuple[re.Pattern, Any]]) -> tuple[str, int]:
    """ Applies post-simplification regex patterns. (Standard Mode) """
    total_replacements = 0; lines = content.splitlines(keepends=True); output_lines = []
    log.debug(f"Applying {len(patterns)} post-simplification patterns..."); pattern_counts = Counter()
    for line in lines:
        modified_line = line
        for i, (pattern, replacement) in enumerate(patterns):
            try:
                 original_line_segment = modified_line
                 if callable(replacement): modified_line_new, count = pattern.subn(replacement, modified_line)
                 else: modified_line_new, count = pattern.subn(replacement, modified_line)
                 if count > 0:
                     total_replacements += count; pattern_counts[i] += count
                     if log.getEffectiveLevel() <= logging.DEBUG:
                         diff_start = -1
                         for k in range(min(len(original_line_segment), len(modified_line_new))):
                             if original_line_segment[k] != modified_line_new[k]: diff_start = k; break
                         log.debug(f"  Pattern {i} ({pattern.pattern[:30]}...) matched {count} time(s) on line: ...{original_line_segment[max(0,diff_start-10):diff_start+10]}... -> ...{modified_line_new[max(0,diff_start-10):diff_start+10]}...")
                     modified_line = modified_line_new
            except Exception as e: log.error(f"Error applying post-simplification pattern {i} ({pattern.pattern}) to line: {e}"); log.debug(f"Problematic line: {line.strip()[:100]}")
        output_lines.append(modified_line)
    modified_content = "".join(output_lines)
    if pattern_counts: log.debug(f"Post-simplification counts: {dict(pattern_counts)}")
    log.info(f"Applied {len(patterns)} post-simplification patterns, making {total_replacements} replacements.")
    return modified_content, total_replacements

def expand_multi_pattern_lines(content: str, finder_pattern: re.Pattern, pattern_name_for_log: str = "PATTERN") -> tuple[str, int]:
    """ Expands lines with multiple pattern instances. (Standard Mode) """
    lines = content.splitlines(keepends=False); output_lines = []; lines_expanded = 0
    log.debug(f"--- expand_multi_pattern_lines: Scanning for '{pattern_name_for_log}' ---")
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith(("#", "---", "##", "==", "*LINE_REF_")): output_lines.append(line + "\n"); continue
        try: matches = finder_pattern.findall(line)
        except Exception as e: log.error(f"Regex error finding patterns on line {i+1}: {e}"); log.debug(f"Problematic line: {line}"); matches = []
        if len(matches) > 1:
             combined_match_len = sum(len(str(m).strip(',').strip()) for m in matches)
             stripped_len_approx = len(re.sub(r'\s+|,', '', stripped_line))
             if combined_match_len >= stripped_len_approx * 0.8:
                log.info(f"Expanding line {i+1} containing {len(matches)} instances of '{pattern_name_for_log}'."); log.debug(f"Original line {i+1}: {line.strip()}")
                lines_expanded += 1; indent_level = line.find(stripped_line[0]) if stripped_line else 0; indent = " " * indent_level
                for match_item in matches: match_str = str(match_item).strip(); output_lines.append(f"{indent}{match_str}\n"); log.debug(f"  Expanded to: {indent}{match_str}")
                continue
        output_lines.append(line + "\n")
    log.info(f"Finished line expansion pre-processing. Expanded {lines_expanded} lines containing multiple '{pattern_name_for_log}' instances.")
    return "".join(output_lines), lines_expanded

def compress_pattern_blocks(content: str, patterns_to_compress: dict[str, re.Pattern], min_consecutive: int) -> tuple[str, int]:
    lines = content.splitlines(keepends=True); output_lines = []; total_blocks_compressed = 0; i = 0
    log.debug(f"--- compress_pattern_blocks: Starting scan (min_consecutive={min_consecutive}) ---")
    while i < len(lines):
        current_line, stripped_line = lines[i], lines[i].strip(); matched_pattern_name = None
        is_ignorable = (stripped_line.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or not stripped_line)
        if not is_ignorable:
            for name, pattern in patterns_to_compress.items():
                if pattern.match(stripped_line): matched_pattern_name = name; log.debug(f"Line {i+1} potentially starts block '{name}': {stripped_line[:60]}..."); break
        if matched_pattern_name:
            block_pattern_name, block_start_index, block_lines_indices = matched_pattern_name, i, [i]; j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                next_is_ignorable = (next_stripped.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or not next_stripped)
                if next_is_ignorable: log.debug(f"  Block '{block_pattern_name}' interrupted at line {j+1} by ignorable line."); break
                if patterns_to_compress[block_pattern_name].match(next_stripped): block_lines_indices.append(j); log.debug(f"  Line {j+1} continues block '{block_pattern_name}'."); j += 1
                else: log.debug(f"  Block '{block_pattern_name}' ended at line {j+1}."); break
            block_count = len(block_lines_indices)
            if block_count >= min_consecutive:
                first_line_in_block, first_line_stripped = lines[block_start_index], lines[block_start_index].strip(); indent = ""
                if len(first_line_in_block) > len(first_line_stripped): indent = first_line_in_block[:len(first_line_in_block) - len(first_line_stripped)]
                summary_line = f"{indent}## [Compressed Block: {block_count} lines matching pattern '{block_pattern_name}'] ##\n"
                output_lines.append(summary_line); log.info(f"Compressed {block_count} lines (Indices {block_start_index+1}-{j}) matching '{block_pattern_name}'.")
                total_blocks_compressed += 1; i = j
            else:
                log.debug(f"  Block '{block_pattern_name}' starting at line {block_start_index+1} had {block_count} lines (min={min_consecutive}). Not compressing.")
                for block_line_index in block_lines_indices: output_lines.append(lines[block_line_index])
                i = j
        else: output_lines.append(current_line); i += 1
    log.info(f"Pattern block compression: Compressed {total_blocks_compressed} blocks (min consecutive: {min_consecutive}).")
    return "".join(output_lines), total_blocks_compressed

def minify_repeated_lines(content: str, min_length: int, min_repetitions: int) -> tuple[str, int]:
    global DEFINITION_SIMPLIFICATION_PATTERNS
    lines = content.splitlines(keepends=True); line_counts = Counter(); placeholder_template = "*LINE_REF_{}*"
    meaningful_lines_indices = {}; log.debug(f"--- minify_repeated_lines: Scan lines >= {min_length} chars, repeated >= {min_repetitions}x ---")
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        is_structural_or_placeholder = (stripped_line.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or
            any(re.match(pat, stripped_line) for pat in (r'^\*VOICE:.*\*$', r'^\*UUID\*$', r'^\*...\*$')) or not stripped_line)
        if len(stripped_line) >= min_length and not is_structural_or_placeholder:
            line_content_key = stripped_line.rstrip() # Use the stripped line as the key
            line_counts[line_content_key] += 1
            if line_content_key not in meaningful_lines_indices: meaningful_lines_indices[line_content_key] = []
            meaningful_lines_indices[line_content_key].append(i)
    replacement_map: Dict[str, str] = {}; minified_line_origins: Dict[str, Counter] = {}; definition_lines = []; placeholder_counter = 1
    placeholder_template = "*LINE_REF_{}*"; repeated_lines = sorted([(line, count) for line, count in line_counts.items() if count >= min_repetitions], key=lambda item: (-item[1], item[0]))
    # Create a context finder specific to this list of lines
    find_file_context = create_file_context_finder(lines)
    for line_content, count in repeated_lines:
        if line_content not in replacement_map:
            placeholder = placeholder_template.format(placeholder_counter); replacement_map[line_content] = placeholder
            log.debug(f"  Creating def {placeholder} for line repeated {count} times: {line_content.strip()[:80]}...")
            indices = meaningful_lines_indices.get(line_content, [])
            for index in indices:
                file_id = find_file_context(index) # Use the created finder
                if file_id not in minified_line_origins: minified_line_origins[file_id] = Counter()
                minified_line_origins[file_id]['placeholder_instances'] += 1
            simplified_definition_content = line_content.rstrip()
            for pattern, replacement in DEFINITION_SIMPLIFICATION_PATTERNS:
                 try:
                     if callable(replacement): simplified_definition_content, _ = pattern.subn(replacement, simplified_definition_content)
                     else: simplified_definition_content = pattern.sub(replacement, simplified_definition_content)
                 except Exception as e: log.warning(f"Error simplifying definition pattern {pattern.pattern}: {e}")
            definition_lines.append(f"{placeholder} = {simplified_definition_content}"); placeholder_counter += 1
    if not replacement_map: log.info("Line minification: No lines met criteria."); return content, 0, {}
    new_lines = lines[:]; num_actual_replacements = 0
    for original_line, placeholder in replacement_map.items():
        indices_to_replace = meaningful_lines_indices.get(original_line, [])
        for index in indices_to_replace:
            if index < len(new_lines) and new_lines[index] == original_line:
                new_lines[index] = placeholder + ("\n" if original_line.endswith("\n") else ""); num_actual_replacements += 1
            else: log.warning(f"Skipped replacing line at index {index+1} due to content/index mismatch.")
    minified_content = "".join(new_lines)
    if definition_lines:
        definition_header = ["", "=" * 40, f"# Line Minification Definitions ({len(definition_lines)}):", f"# (Lines >= {min_length} chars, >= {min_repetitions} reps, content simplified)", "=" * 40]
        definition_block = "\n".join(definition_header + definition_lines) + "\n\n"
        log.info(f"Line minification: Replaced {num_actual_replacements} occurrences of {len(definition_lines)} unique lines.")
        return definition_block + minified_content, num_actual_replacements, minified_line_origins
    else: return content, 0, {}

# <<< MODIFIED to track approximate origins >>>
def post_process_cleanup(content: str, cleanup_pattern: re.Pattern) -> tuple[str, int, Dict[str, Counter]]:
    """
    Removes lines consisting primarily of placeholders/structure. (Standard Mode)
    Returns: (cleaned_content, lines_removed_count, cleanup_origins_report)
    """
    lines = content.splitlines(keepends=True); output_lines = []; lines_removed = 0
    cleanup_origins_report: Dict[str, Counter] = {}
    # Create a context finder for the lines *before* cleanup
    find_file_context = create_file_context_finder(lines)

    log.debug(f"--- post_process_cleanup: Starting ---")
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Keep definition lines, block markers, file markers, and comments explicitly
        if (line.startswith(("*LINE_REF_", "## [Compressed Block:", "--- File:", "--- End File:", "===")) or stripped_line.startswith("#")):
            output_lines.append(line); continue
        # Check if the line matches the cleanup pattern
        if cleanup_pattern.match(line):
            log.debug(f"Post-cleanup removing line {i+1}: {line.strip()[:80]}...")
            lines_removed += 1
            # <<< ADDED: Track approximate origin >>>
            file_id = find_file_context(i)
            if file_id not in cleanup_origins_report: cleanup_origins_report[file_id] = Counter()
            cleanup_origins_report[file_id]['removed_by_cleanup'] += 1
        else: output_lines.append(line) # Keep the line

    log.info(f"Post-processing cleanup removed {lines_removed} lines.")
    cleaned_content = "".join(output_lines); cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    return cleaned_content, lines_removed, cleanup_origins_report

# --- Main Function ---
def main():
    global DEFAULT_IGNORE_PATTERNS, BASE_IGNORE_PATTERNS, TEST_IGNORE_PATTERNS, CODE_EXTENSIONS, INTERESTING_FILENAMES # Allow access
    DEFAULT_MIN_LINE_LENGTH_REC, DEFAULT_MIN_REPETITIONS_REC, DEFAULT_MIN_CONSECUTIVE_REC = 140, 2, 4

    parser = argparse.ArgumentParser( description="Generate a representation of a folder's text content.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("folder_path", help="Path to the target folder.")
    parser.add_argument("-o", "--output", help="Output file path (optional, defaults to stdout).")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--raw-dump", action="store_true", default=False, help="Dump ALL files verbatim.")
    mode_group.add_argument("--raw-dump-source", action="store_true", default=False, help="Dump only SOURCE files verbatim.")

    sel_group = parser.add_argument_group('File Selection Options (Standard & Raw Source Modes)')
    sel_group.add_argument("--ignore", nargs='+', default=[], help="Additional names/patterns to ignore.")
    sel_group.add_argument("--source-ext", nargs='+', default=[], help="Additional extensions/filenames for source code.")
    sel_group.add_argument("--interesting-files", nargs='+', default=[], help="Additional notable filenames for summaries/selection.")
    sel_group.add_argument("--include-tests", action="store_true", default=False, help="Include files/dirs matching test patterns.")

    proc_group = parser.add_argument_group('Standard Mode Processing Options (ignored in raw modes)')
    proc_group.add_argument("--keep-empty", action="store_true", default=False, help="Keep files even if empty after simplification.")
    proc_group.add_argument("--keep-duplicates", action="store_true", default=False, help="Keep files even if simplified content is duplicated.")
    proc_group.add_argument("--no-preprocess-split-lines", action="store_false", dest="preprocess_split_lines", default=True, help="Disable pre-processing split of multi-pattern lines.")
    proc_group.add_argument("--no-compress-patterns", action="store_false", dest="compress_patterns", default=True, help="Disable compression of consecutive pattern lines.")
    proc_group.add_argument("--no-minify-lines", action="store_false", dest="minify_lines", default=True, help="Disable repeated identical long line minification.")
    proc_group.add_argument("--no-post-cleanup", action="store_false", dest="post_cleanup", default=True, help="Disable final removal of placeholder-only lines.")
    proc_group.add_argument("--strip-logging", action="store_true", default=False, help="Attempt to remove common logging statements.")
    proc_group.add_argument("--apply-patterns", action="store_true", default=False, help="Apply detailed post-simplification patterns (UUIDs, URLs, etc.).")
    proc_group.add_argument("--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE_REC, help="Min consecutive lines for pattern block compression.")
    proc_group.add_argument("--min-line-length", type=int, default=DEFAULT_MIN_LINE_LENGTH_REC, help="Min length for identical line minification.")
    proc_group.add_argument("--min-repetitions", type=int, default=DEFAULT_MIN_REPETITIONS_REC, help="Min repetitions for identical line minification.")
    proc_group.add_argument("--large-literal-threshold", type=int, default=DEFAULT_LARGE_LITERAL_THRESHOLD, help="Min lines in list/dict for literal compression (skipped if pattern comp enabled).")

    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level.")
    args = parser.parse_args()

    log.setLevel(args.log_level.upper()); logging.getLogger().setLevel(args.log_level.upper())
    seen_content_hashes.clear(); total_bytes_written = 0
    process_report_data: Dict[str, Counter] = {}
    minification_origin_report: Dict[str, Counter] = {}
    # <<< ADDED for cleanup origins >>>
    cleanup_origin_report: Dict[str, Counter] = {}
    files_processed_count = 0

    target_folder_path = Path(args.folder_path)
    if not target_folder_path.is_dir(): log.critical(f"Error: Folder not found: '{args.folder_path}'"); sys.exit(1)
    resolved_path = target_folder_path.resolve()

    run_mode = "Standard"; effective_test_inclusion = args.include_tests
    if args.raw_dump: run_mode = "Raw Dump (All Files)"; effective_test_inclusion = False
    elif args.raw_dump_source: run_mode = "Raw Source Dump"; effective_test_inclusion = args.include_tests

    current_ignore_patterns = BASE_IGNORE_PATTERNS.copy()
    if not effective_test_inclusion and run_mode != "Raw Dump (All Files)":
        log.info("Excluding test files/directories (default). Use --include-tests to override.")
        current_ignore_patterns.update(TEST_IGNORE_PATTERNS)
    elif effective_test_inclusion: log.info("Including test files/directories based on --include-tests flag.")
    current_ignore_patterns.update(args.ignore)

    current_code_extensions = CODE_EXTENSIONS.copy(); current_code_extensions.update(args.source_ext)
    current_interesting_files = INTERESTING_FILENAMES.copy(); current_interesting_files.update(args.interesting_files)

    print("-" * 40, file=sys.stderr); print(f"Processing folder: {resolved_path}", file=sys.stderr)
    print(f"Output target: {'Stdout' if not args.output else args.output}", file=sys.stderr)
    print(f"Log Level: {args.log_level.upper()}", file=sys.stderr); print(f"Mode: {run_mode}", file=sys.stderr)
    if run_mode != "Raw Dump (All Files)": print(f"  Include Tests: {effective_test_inclusion} ({'--include-tests' if effective_test_inclusion else 'DEFAULT'})", file=sys.stderr)
    else: print("  Include Tests: N/A (Raw Dump includes all files regardless)", file=sys.stderr)
    if run_mode == "Standard":
        effective_skip_empty = not args.keep_empty; effective_skip_duplicates = not args.keep_duplicates
        print(f"  Skip empty: {effective_skip_empty}", file=sys.stderr); print(f"  Skip duplicates: {effective_skip_duplicates}", file=sys.stderr)
        print(f"  Strip logging: {args.strip_logging}", file=sys.stderr); print(f"  Pre-process split lines: {args.preprocess_split_lines}", file=sys.stderr)
        print(f"  Compress pattern blocks: {args.compress_patterns}", file=sys.stderr); print(f"    Min consecutive lines: {args.min_consecutive}", file=sys.stderr)
        print(f"  Apply detailed patterns: {args.apply_patterns}", file=sys.stderr); print(f"  Minify identical lines: {args.minify_lines}", file=sys.stderr)
        if args.minify_lines: print(f"    Min line length: {args.min_line_length}, Min repetitions: {args.min_repetitions}", file=sys.stderr)
        print(f"  Post-process cleanup: {args.post_cleanup}", file=sys.stderr)
        literal_compression_status = 'DISABLED (Pattern comp enabled)' if args.compress_patterns else f'ENABLED (thresh: {args.large_literal_threshold})'
        print(f"  Large literal compression: {literal_compression_status}", file=sys.stderr)
        print(f"  (User Ignore/Source/Interesting patterns affect processing/reporting)", file=sys.stderr)
    elif run_mode == "Raw Source Dump": print("  (Standard mode compression/simplification ignored; User Ignore/Source/Interesting affect selection)", file=sys.stderr)
    else: print("  (Standard mode options & Ignore/Source/Interesting ignored)", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    output_handle = None; output_path = None; content_buffer = io.StringIO(); initial_header_str = ""; output_content_to_write = ""
    num_lines_expanded, num_pattern_replacements = 0, 0; num_blocks_compressed, num_lines_minified, num_lines_cleaned_up = 0, 0, 0
    main_content = ""

    try:
        header_lines = []
        if run_mode == "Raw Dump (All Files)": header_lines = [f"# RAW DUMP: {resolved_path}", f"# Mode: --raw-dump", "="*40, ""]
        elif run_mode == "Raw Source Dump": header_lines = [f"# RAW SOURCE DUMP: {resolved_path}", f"# Mode: --raw-dump-source", f"# Include Tests: {effective_test_inclusion}", "="*40, ""]
        else:
             effective_skip_empty = not args.keep_empty; effective_skip_duplicates = not args.keep_duplicates
             header_lines = [f"# Compressed Representation: {resolved_path}", f"# Mode: Standard", f"# Include Tests: {effective_test_inclusion}", f"# Options: skip_empty={effective_skip_empty}, skip_duplicates={effective_skip_duplicates}, preprocess_split={args.preprocess_split_lines}, compress_patterns={args.compress_patterns}({args.min_consecutive}), apply_patterns={args.apply_patterns}, minify_lines={args.minify_lines}, post_cleanup={args.post_cleanup}", "="*40, ""]
        initial_header_str = "\n".join(header_lines) + "\n"

        if run_mode == "Raw Dump (All Files)":
            log.info("Starting raw folder dump..."); files_processed_count = process_folder_raw(str(resolved_path), content_buffer, process_report_data); log.info("Finished raw folder dump.")
            content_buffer.seek(0); main_content = content_buffer.getvalue()
        elif run_mode == "Raw Source Dump":
            log.info("Starting raw source folder dump...")
            files_processed_count = process_folder_raw_source(str(resolved_path), content_buffer, current_ignore_patterns, current_code_extensions, current_interesting_files, process_report_data)
            log.info("Finished raw source folder dump.")
            content_buffer.seek(0); main_content = content_buffer.getvalue()
        else: # Standard Mode
            log.info("Starting standard folder processing (Step 1)...")
            effective_skip_empty = not args.keep_empty; effective_skip_duplicates = not args.keep_duplicates
            process_folder(str(resolved_path), content_buffer, current_ignore_patterns, current_code_extensions, current_interesting_files, effective_skip_empty, args.strip_logging, effective_skip_duplicates, args.large_literal_threshold, args.compress_patterns, process_report_data)
            log.info("Finished initial folder processing.")
            content_buffer.seek(0); processed_content = content_buffer.getvalue(); final_output_content = processed_content
            if args.preprocess_split_lines and final_output_content.strip(): log.info("Step 2: Expanding multi-pattern lines..."); final_output_content, num_lines_expanded = expand_multi_pattern_lines(final_output_content, SINGLE_VOICE_ID_FINDER, "VOICE_ID"); log.info("Finished Step 2.")
            if args.compress_patterns and final_output_content.strip(): log.info("Step 3: Compressing pattern blocks..."); final_output_content, num_blocks_compressed = compress_pattern_blocks(final_output_content, BLOCK_COMPRESSION_PATTERNS, args.min_consecutive); log.info("Finished Step 3.")
            if args.apply_patterns and final_output_content.strip(): log.info("Step 4: Applying detailed patterns..."); final_output_content, num_pattern_replacements = apply_post_simplification_patterns(final_output_content, POST_SIMPLIFICATION_PATTERNS); log.info("Finished Step 4.")
            if args.minify_lines and final_output_content.strip(): log.info("Step 5: Minifying identical lines..."); final_output_content, num_lines_minified, minification_origin_report = minify_repeated_lines(final_output_content, args.min_line_length, args.min_repetitions); log.info("Finished Step 5.")
            if args.post_cleanup and final_output_content.strip():
                log.info("Step 6: Applying post-processing cleanup...")
                # <<< MODIFIED: Capture cleanup origins >>>
                final_output_content, num_lines_cleaned_up, cleanup_origin_report = post_process_cleanup(final_output_content, PLACEHOLDER_CLEANUP_PATTERN)
                log.info("Finished Step 6.")
            else: log.debug("Skipping Step 6: Post-process cleanup")
            main_content = final_output_content

        file_summary_block = generate_output_file_summary(process_report_data, run_mode)
        output_content_to_write = initial_header_str + file_summary_block + main_content

        log.info("Writing final output...")
        if args.output:
            output_path = Path(args.output).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = open(output_path, 'w', encoding='utf-8'); log.debug(f"Opening output file: {output_path}")
        else: output_handle = sys.stdout; log.debug("Using stdout for output.")
        output_handle.write(output_content_to_write); total_bytes_written = len(output_content_to_write.encode('utf-8')); log.info("Finished writing output.")

        # --- Final Summary (to stderr) ---
        print("-" * 40, file=sys.stderr); print("Processing complete.", file=sys.stderr); print(f"Mode: {run_mode}", file=sys.stderr)
        if run_mode != "Raw Dump (All Files)": print(f"Include Tests: {effective_test_inclusion}", file=sys.stderr)

        if run_mode == "Standard":
             print("\n--- Overall Post-Processing Stage Summary (Standard Mode) ---", file=sys.stderr)
             print(f"  Line expansion pre-processing expanded {num_lines_expanded} lines.", file=sys.stderr); print(f"  Pattern block compression created {num_blocks_compressed} summary lines.", file=sys.stderr)
             print(f"  Detailed pattern application made {num_pattern_replacements} replacements.", file=sys.stderr); print(f"  Identical line minification created {len(minification_origin_report)} definition(s) replacing {num_lines_minified} original line instances.", file=sys.stderr)
             # <<< MODIFIED: Updated cleanup line description >>>
             print(f"  Post-processing cleanup removed {num_lines_cleaned_up} lines (see details below).", file=sys.stderr)

             if args.minify_lines and minification_origin_report:
                 print("\n--- Minification Origin Report (Placeholder Instances Created per File Type) ---", file=sys.stderr)
                 total_minified_instances = 0; sorted_minify_origins = sorted(minification_origin_report.items())
                 for file_id, counts in sorted_minify_origins: instances = counts.get('placeholder_instances', 0); print(f"  {file_id if file_id else '<unknown_context>'}: {instances} instances", file=sys.stderr); total_minified_instances += instances
                 print(f"  --- \n  TOTAL Minified Instances: {total_minified_instances}", file=sys.stderr)
                 if total_minified_instances != num_lines_minified: log.warning(f"Mismatch between reported minified instances ({total_minified_instances}) and replacements ({num_lines_minified}).")

             # <<< ADDED: Cleanup Origin Report >>>
             if args.post_cleanup and cleanup_origin_report:
                 print("\n--- Cleanup Origin Report (Approximate Lines Removed per File Type) ---", file=sys.stderr)
                 print("# NOTE: Attribution is based on the nearest preceding file header and may be inaccurate", file=sys.stderr)
                 print("#       if minification significantly reordered the content.", file=sys.stderr)
                 total_cleaned_up_reported = 0
                 sorted_cleanup_origins = sorted(cleanup_origin_report.items())
                 for file_id, counts in sorted_cleanup_origins:
                     removed_count = counts.get('removed_by_cleanup', 0)
                     print(f"  {file_id if file_id else '<unknown_context>'}: {removed_count} lines", file=sys.stderr)
                     total_cleaned_up_reported += removed_count
                 print(f"  --- \n  TOTAL Cleaned Up Lines (Reported): {total_cleaned_up_reported}", file=sys.stderr)
                 if total_cleaned_up_reported != num_lines_cleaned_up:
                      log.warning(f"Mismatch between reported cleaned lines ({total_cleaned_up_reported}) and total removed ({num_lines_cleaned_up}). Check logic.")


             print("\n--- File Type Processing Report (Initial Processing & Contribution) ---", file=sys.stderr)
             if not process_report_data: print("  No source files were processed or met inclusion criteria.", file=sys.stderr)
             else:
                 sorted_file_ids = sorted(process_report_data.keys())
                 total_processed, total_simplified, total_skipped_empty, total_skipped_duplicate, total_contributed = 0, 0, 0, 0, 0
                 for file_id in sorted_file_ids:
                     stats = process_report_data[file_id]; processed, simplified, skipped_e, skipped_d, contributed = stats.get('processed', 0), stats.get('simplified', 0), stats.get('skipped_empty', 0), stats.get('skipped_duplicate', 0), stats.get('contributed', 0)
                     total_processed += processed; total_simplified += simplified; total_skipped_empty += skipped_e; total_skipped_duplicate += skipped_d; total_contributed += contributed
                     print(f"  {file_id if file_id else '<no_ext>'}:", file=sys.stderr); print(f"    - Processed: {processed}", file=sys.stderr)
                     if simplified > 0: print(f"    - Simplified (Initial): {simplified}", file=sys.stderr)
                     if skipped_e > 0: print(f"    - Skipped (Empty): {skipped_e}", file=sys.stderr);
                     if skipped_d > 0: print(f"    - Skipped (Duplicate): {skipped_d}", file=sys.stderr)
                     if contributed > 0: print(f"    - Contributed Content: {contributed}", file=sys.stderr)
                     if stats.get('read_error', 0) > 0: print(f"    - Read Errors: {stats['read_error']}", file=sys.stderr)
                     if stats.get('processing_error', 0) > 0: print(f"    - Processing Errors: {stats['processing_error']}", file=sys.stderr)
                 print("  ---", file=sys.stderr); print(f"  TOTALS:", file=sys.stderr); print(f"    - Processed: {total_processed}", file=sys.stderr); print(f"    - Simplified (Initial): {total_simplified}", file=sys.stderr)
                 print(f"    - Skipped (Empty): {total_skipped_empty}", file=sys.stderr); print(f"    - Skipped (Duplicate): {total_skipped_duplicate}", file=sys.stderr); print(f"    - Contributed Content: {total_contributed}", file=sys.stderr)
        elif run_mode == "Raw Source Dump": print(f"  Dumped content of {files_processed_count} source files.", file=sys.stderr)
        else: print(f"  Dumped content of {files_processed_count} files.", file=sys.stderr)

        print("-" * 40, file=sys.stderr)
        if args.output and output_path: print(f"Total bytes written to {output_path}: {total_bytes_written}", file=sys.stderr)
        else: print(f"Total bytes written to stdout: {total_bytes_written}", file=sys.stderr)
        print("-" * 40, file=sys.stderr)

    except IOError as e: log.critical(f"I/O Error: {e}"); sys.exit(1)
    except KeyboardInterrupt: log.warning("Processing interrupted (Ctrl+C)."); sys.exit(1)
    except Exception as e: log.critical(f"Unexpected error: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG)); sys.exit(1)
    finally:
        if content_buffer: content_buffer.close()
        if args.output and output_handle and output_handle is not sys.stdout:
            try: output_handle.close(); log.debug(f"Closed output file: {output_path}")
            except Exception as e: log.error(f"Error closing output file '{args.output}': {e}")

if __name__ == "__main__":
    main()
