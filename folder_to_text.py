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
import json # Added for JSON processing
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

# --- Constants and Patterns (many are unchanged, will be included in the full script) ---
FILE_HEADER_RE = re.compile(r"^--- File: (.*) ---$")

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(funcName)s: %(message)s', stream=sys.stderr)
log = logging.getLogger(__name__)

# --- Default Ignore Patterns ---
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
    # Lock files, minified files, env files
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'composer.lock', 'go.sum', 'Gemfile.lock',
    '*.min.js', '*.min.css',
    '.env', # Actual .env files
    # Backup/temp files
    '*.bak', '*.old', '*.tmp', '*~',
    '*.log', # Log files by default
}

TEST_IGNORE_PATTERNS = {
    'test', 'tests', '__tests__', 'test_data', 'fixtures', # Common directory names
    '*.test.*', 'test_*.py', '*_test.py', '*.spec.*', # Common file patterns (py, js, ts, etc.)
    'Test*.java', '*Test.java', '*Tests.java', # Java patterns
    '*Test.kt', '*Tests.kt', # Kotlin patterns
    '*Tests.cs', # C# pattern
    '*_test.go', # Go pattern
}
DEFAULT_IGNORE_PATTERNS = BASE_IGNORE_PATTERNS.union(TEST_IGNORE_PATTERNS)

# --- Code Extensions ---
CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.kt', '.swift',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rb', '.php', '.sh',
    '.bash', '.zsh', '.ps1', '.psm1',
    '.css', '.scss', '.less', '.sass',
    '.html', '.htm', '.xml', '.json', '.yaml', '.yml', '.toml', # .json is key here
    '.sql', '.md', '.rst', '.tex',
    '.dockerfile', 'Dockerfile',
    '.r', '.pl', '.pm', '.lua',
    '.gradle', '.tf', '.tfvars', '.hcl',
    '.conf', '.ini', '.cfg', '.properties', '.config', '.settings',
    'Makefile', 'Jenkinsfile', 'Gemfile', 'pom.xml', 'build.gradle', 'settings.gradle',
    '.ipynb', '.env.example', # example env files are often useful
}

INTERESTING_FILENAMES = {
    'README', 'LICENSE', 'CONTRIBUTING', 'CHANGELOG', 'SECURITY', 'AUTHORS', 'COPYING',
    'requirements.txt', 'Pipfile', 'setup.py', 'pyproject.toml',
    'docker-compose.yml', 'docker-compose.yaml', 'compose.yaml', 'compose.yml',
    'package.json', 'tsconfig.json', 'angular.json', 'vue.config.js', 'webpack.config.js',
    'go.mod', 'go.work',
    '.gitignore', '.dockerignore', '.editorconfig', '.gitattributes', '.gitlab-ci.yml', '.travis.yml',
}

# --- Binary File Check ---
BINARY_CHECK_BYTES = 1024
BINARY_NON_PRINTABLE_THRESHOLD = 0.15 # Percentage of non-printable chars

# --- Simplification and Compression Patterns ---
SIMPLIFICATION_PATTERNS = [
    # Obfuscate very long numbers (potential IDs, timestamps if not dates)
    (re.compile(r'\d{8,}'), '*NUM_LONG*'),
    # Obfuscate long hex strings (potential hashes, IDs)
    (re.compile(r'\b[a-fA-F0-9]{12,}\b'), '*HEX_LONG*'),
    # Obfuscate long Base64-like strings
    (re.compile(r'[a-zA-Z0-9+/=]{30,}'), '*BASE64LIKE_LONG*'),
    # Common float and int patterns (generic)
    (re.compile(r'\b\d+\.\d+\b'), '*FLOAT*'),
    (re.compile(r'\b\d+\b'), '*INT*'),
]

POST_SIMPLIFICATION_PATTERNS = [
    # UUIDs (quoted and unquoted)
    (re.compile(r'["\']?([a-fA-F0-9]{8}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{12})["\']?'), r'"*UUID*"'),
    # URLs (simple version)
    (re.compile(r'(https?://(?:www\.)?[\w\.-]+(?:/[\w\./\-\?%=\&\#\+]*)*)'), r'"*URL_SIMPLE*"'),
    # Email addresses
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), r'"*EMAIL*"'),
    # AWS Access Key IDs (example)
    (re.compile(r'\b(AKIA|ASIA)[A-Z0-9]{16}\b'), r'"*AWS_ACCESS_KEY_ID*"'),
    # Generic long strings in quotes (e.g., long paths, messages)
    (re.compile(r'([\'"])(.{80,})\1'), r'\1*STRING_VERY_LONG*\1'), # Over 80 chars
    (re.compile(r'([\'"])(.{40,})\1'), r'\1*STRING_LONG*\1'),    # Over 40 chars
]

# For --preprocess-split-lines (e.g., Google Cloud Voice IDs)
# Example: "en-US-Standard-A", "en-US-Wavenet-D Neural"
SINGLE_VOICE_ID_PATTERN_STR = r'"([a-z]{2}-[A-Z]{2}-(?:Standard|Wavenet|News|Studio|Polyglot)-[A-Z](?:(?: Male| Female)? Neural)?)"'
SINGLE_VOICE_ID_FINDER = re.compile(SINGLE_VOICE_ID_PATTERN_STR)

BLOCK_COMPRESSION_PATTERNS = {
    "VOICE_ID_LIST": re.compile(r'^\s*' + SINGLE_VOICE_ID_PATTERN_STR + r',?\s*$'),
    "QUOTED_UUID_LIST": re.compile(r'^\s*"\*UUID\*",?\s*$'),
    "SIMPLE_KEY_VALUE": re.compile(r'^\s*[\'"][\w\s\-]+[\'"]\s*:\s*[\'"].+[\'"],?\s*$'),
    # Add more patterns as needed
}

DEFINITION_SIMPLIFICATION_PATTERNS = [
    (re.compile(r'\s{2,}'), ' '), # Collapse multiple spaces
    (re.compile(r'.{100}(.*)'), lambda m: m.group(0)[:100] + '... (truncated)'), # Truncate very long lines
]

# For --post-cleanup
# Matches lines that are effectively empty or only contain placeholders/simple punctuation
PLACEHOLDER_CLEANUP_PATTERN = re.compile(
    r'^\s*('
    r'(\*NUM_LONG\*|\*HEX_LONG\*|\*BASE64LIKE_LONG\*|\*FLOAT\*|\*INT\*|\*UUID\*|\*URL_SIMPLE\*|\*EMAIL\*|\*STRING_LONG\*|\*STRING_VERY_LONG\*|\*AWS_ACCESS_KEY_ID\*)'
    r'|[,\[\]\{\}\(\)\s]|\.\.\.' # Common placeholders and structural chars
    r')*\s*$'
)
# Default threshold for compressing large list/dict literals
DEFAULT_LARGE_LITERAL_THRESHOLD = 10

seen_content_hashes = set() # Global set for duplicate detection

# --- Utility Functions ---
def get_file_id_from_path_str(path_str: Union[str, Path]) -> str:
    name = Path(path_str).name
    base, ext = os.path.splitext(name)
    if not ext and base: return base # e.g. Makefile
    return ext.lower() if ext else "<no_ext>"

def is_likely_binary(file_path: Path) -> bool:
    """Checks if a file is likely binary by reading a small chunk."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(BINARY_CHECK_BYTES)
        if not chunk: return False # Empty file is not binary for our purposes
        # Count non-printable characters (excluding common whitespace like tab, newline, CR)
        non_printable_chars = sum(1 for byte in chunk if chr(byte) not in string.printable and chr(byte) not in '\t\n\r')
        ratio = non_printable_chars / len(chunk)
        return ratio > BINARY_NON_PRINTABLE_THRESHOLD
    except OSError as e:
        log.warning(f"Error checking binary status for {file_path}: {e}")
        return True # Assume binary on error to be safe

def summarize_other_files(filenames: List[str], code_ext_set: set, interesting_names_set: set) -> str:
    """Generates a summary line for non-code text files in a directory."""
    counts = Counter()
    interesting_found = []
    for fname in filenames:
        base, ext = os.path.splitext(fname)
        ext_lower = ext.lower()
        if base in interesting_names_set or fname in interesting_names_set:
            interesting_found.append(fname)
        elif ext_lower and ext_lower not in code_ext_set: # Avoid double-counting if something is miscategorized
            counts[ext_lower] += 1
    
    summary_parts = []
    if interesting_found:
        summary_parts.append(", ".join(sorted(interesting_found)))
    
    # Sort extensions by count, then alphabetically for stable output
    sorted_ext_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    for ext, num in sorted_ext_counts:
        summary_parts.append(f"{num}x {ext}")
        
    if not summary_parts: return ""
    return f"# Other files in '.': {', '.join(summary_parts)}"

def create_file_context_finder(all_lines: List[str]) -> Callable[[int], str]:
    """Creates a function to find the file context for a given line index."""
    file_boundaries: List[Tuple[int, str]] = []
    for i, line_content in enumerate(all_lines):
        match = FILE_HEADER_RE.match(line_content)
        if match:
            file_path_str = match.group(1).split(" (")[0] # Get path before any parenthetical like (Sample Representation)
            file_boundaries.append((i, get_file_id_from_path_str(file_path_str)))

    def find_context(line_index: int) -> str:
        current_file_id = "<unknown_context>"
        for start_index, file_id in file_boundaries:
            if line_index >= start_index:
                current_file_id = file_id
            else:
                break
        return current_file_id
    return find_context

# --- Core Simplification and Processing Functions ---

def generate_json_sample(json_content: str) -> Tuple[Optional[str], str]:
    """
    Tries to generate a sample representation of JSON content.
    Returns a tuple: (sample_string_or_None, status_message)
    """
    try:
        data = json.loads(json_content)
        sample_comment = "// Sample representation of the original JSON file.\n"

        if isinstance(data, list):
            if not data:
                return sample_comment + "// JSON array is empty.", "empty_array"
            # Take the first element as a sample
            sample = [data[0]] # Keep it as a list with one item for consistent structure
            return sample_comment + json.dumps(sample, indent=2), "list_sample"
        elif isinstance(data, dict):
            if not data:
                return sample_comment + "// JSON object is empty.", "empty_dict"
            # Take the first key-value pair
            first_key = next(iter(data))
            sample = {first_key: data[first_key]}
            return sample_comment + json.dumps(sample, indent=2), "dict_sample"
        elif isinstance(data, (str, int, float, bool)) or data is None:
            # For simple scalar types, just show them
            return sample_comment + json.dumps(data, indent=2), "scalar_sample"
        else:
            return None, "unknown_json_structure" # Should not happen with valid JSON
    except json.JSONDecodeError:
        return None, "json_decode_error"
    except Exception as e:
        log.warning(f"Unexpected error generating JSON sample: {e}")
        return None, "unexpected_error_sampling_json"


def simplify_source_code(
    content: str,
    strip_logging: bool,
    large_literal_threshold: int,
    disable_literal_compression: bool = False # Usually true if --compress-patterns is on
) -> str:
    """Simplifies source code content by removing comments, obfuscating, etc."""
    # Multi-line comments (/* ... */, """...""", '''...''')
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'"""(?:.|\n)*?"""', '', content, flags=re.DOTALL) # Python docstrings
    content = re.sub(r"'''(?:.|\n)*?'''", '', content, flags=re.DOTALL) # Python docstrings

    # Single-line comments (#, //, --)
    # Order matters: specific first (e.g., not http://)
    content = re.sub(r'(?<![:/])#.*$', '', content, flags=re.MULTILINE) # Python/Ruby/Shell comments (careful with URLs)
    content = re.sub(r'\s*//.*$', '', content, flags=re.MULTILINE)     # JS/Java/C++ comments
    content = re.sub(r'^\s*//.*$', '', content, flags=re.MULTILINE)    # JS/Java/C++ comments at line start
    content = re.sub(r'\s+--.*$', '', content, flags=re.MULTILINE)     # SQL comments
    content = re.sub(r'^\s*--.*$', '', content, flags=re.MULTILINE)    # SQL comments at line start

    lines = content.splitlines()
    simplified_lines = []
    
    in_literal_block = False
    literal_block_start_index = -1 # Index in simplified_lines where block started
    literal_line_count = 0 # Number of simple literal lines within the block

    # Pattern for simple literal lines (e.g., "item", 123, true,)
    # Allows basic types, quoted strings, simple dicts/lists if short.
    literal_line_pattern = re.compile(
        r'^\s*('
        r'([\'"]).*?\2'                 # "string" or 'string'
        r'|true|false|null'             # booleans, null
        r'|\-?\d+(\.\d+)?([eE][+\-]?\d+)?' # numbers (int, float, scientific)
        r'|\[.*?\]'                     # short inline list
        r'|\{.*?\}'                     # short inline dict
        r')\s*,?\s*$', re.VERBOSE
    )
    # Pattern to detect start of a list or dict assignment/declaration
    list_dict_start_pattern = re.compile(r'[:=]\s*(\[|\{)\s*$')

    for i, line in enumerate(lines):
        original_line = line # Keep original for appending if block logic dictates
        # Re-apply single line comment stripping to be sure (might have been part of a multi-line before)
        line = re.sub(r'(?<![:/])#.*$', '', line) 
        line = re.sub(r'\s+//.*$', '', line)
        line = re.sub(r'^\s*//.*$', '', line)
        line = re.sub(r'\s+--.*$', '', line)
        line = re.sub(r'^\s*--.*$', '', line)
        
        stripped_line = line.strip()

        if strip_logging and stripped_line:
            # Basic logging pattern removal (extend as needed)
            # Looks for common log keywords followed by ( ... )
            if re.match(r'^(?:log|logger|logging|console|print|fmt\.Print|System\.out\.print|TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\b', stripped_line, re.IGNORECASE):
                 if '(' in stripped_line and ')' in stripped_line: # Basic check for function call
                    # log.debug(f"Stripping logging line: {stripped_line[:80]}")
                    continue # Skip this line

        is_potential_start = list_dict_start_pattern.search(line) is not None
        is_end = stripped_line.startswith(']') or stripped_line.startswith('}')
        current_line_index = len(simplified_lines) # Where this line would go

        if is_potential_start and not in_literal_block:
            in_literal_block = True
            literal_block_start_index = current_line_index # This line starts the block
            literal_line_count = 0
            simplified_lines.append(original_line) # Add the line that starts the block
        elif in_literal_block:
            is_simple_literal_line = literal_line_pattern.match(stripped_line) is not None
            
            if is_simple_literal_line:
                literal_line_count += 1
                simplified_lines.append(original_line) # Add this simple literal line
            elif is_end: # End of the block
                simplified_lines.append(original_line) # Add the closing bracket line
                if not disable_literal_compression and literal_line_count >= large_literal_threshold and literal_block_start_index >= 0:
                    # Compress the collected simple literal lines
                    start_slice_idx = literal_block_start_index + 1 # After the opening line
                    end_slice_idx = start_slice_idx + literal_line_count # Up to, but not including, the closing line
                    
                    if start_slice_idx < end_slice_idx <= len(simplified_lines)-1 : # Ensure valid slice and we have a closing line
                         try:
                             # Try to get indentation from the first line of the literal block content
                             indent_line = simplified_lines[start_slice_idx] if start_slice_idx < len(simplified_lines) else simplified_lines[literal_block_start_index]
                             indent_level = indent_line.find(indent_line.lstrip()) if indent_line.strip() else 2
                         except IndexError:
                             indent_level = 2 # Default indent if something goes wrong
                         
                         indent = " " * (indent_level)
                         placeholder_line = f"{indent}# ... {literal_line_count} similar lines compressed ..."
                         
                         del simplified_lines[start_slice_idx:end_slice_idx]
                         simplified_lines.insert(start_slice_idx, placeholder_line)
                         log.debug(f"Compressed literal block, {literal_line_count} lines replaced with placeholder.")
                    else:
                         log.warning(f"Could not compress literal block: slice error (start={start_slice_idx}, end={end_slice_idx}, len_sl={len(simplified_lines)}). Skipping compression for this block.")
                
                in_literal_block = False
                literal_block_start_index = -1
                literal_line_count = 0
            else: # A complex line inside a literal block, ends compression attempt for this block
                log.debug(f"Complex line '{stripped_line[:50]}...' ended potential literal block compression.")
                simplified_lines.append(original_line) # Add the complex line
                in_literal_block = False # Reset for next potential block
                literal_block_start_index = -1
                literal_line_count = 0
        
        elif stripped_line: # Not in a literal block, and line is not empty
            simplified_lines.append(line)
        elif simplified_lines and simplified_lines[-1].strip(): # Add a single blank line if previous wasn't blank
            simplified_lines.append("")

    processed_content = "\n".join(simplified_lines).strip()
    if processed_content: # Ensure a final newline if there's content
        processed_content += "\n"

    # Apply regex-based simplifications (numbers, hex, etc.)
    for pattern, replacement in SIMPLIFICATION_PATTERNS:
        processed_content = pattern.sub(replacement, processed_content)

    # Remove excessive blank lines (max 1 blank line)
    processed_content = re.sub(r'\n{3,}', '\n\n', processed_content)
    
    return processed_content

def process_folder(
    target_folder: str,
    buffer: io.StringIO,
    ignore_set: set,
    code_extensions_set: set,
    interesting_filenames_set: set,
    skip_empty: bool,
    strip_logging: bool,
    skip_duplicates: bool,
    large_literal_threshold: int,
    compress_patterns_enabled: bool, # This implies literal compression should be disabled
    report_data: Dict[str, Counter]
):
    global seen_content_hashes
    target_path = Path(target_folder).resolve()

    def write_to_buffer(text):
        buffer.write(text + "\n")

    log.debug(f"Starting folder walk: {target_path} (Standard Mode)")
    log.debug(f"Skips: empty={skip_empty}, duplicates={skip_duplicates}")
    log.debug(f"Strip logging: {strip_logging}")
    log.debug(f"Compress patterns enabled: {compress_patterns_enabled} (disables large literal compression if true)")

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} - {e}")):
        current_path = Path(dirpath)
        try:
            current_rel_path = current_path.relative_to(target_path)
        except ValueError: # pragma: no cover (should not happen if target_path is an ancestor)
            log.warning(f"Cannot get relative path for {current_path}. Skipping.")
            dirnames[:] = [] # Don't recurse further into this path
            continue

        # Filter ignored directories (modify dirnames in place)
        original_dir_count = len(dirnames)
        # *LINE_REF_1* (from original input, slightly different context but similar logic)
        dirnames[:] = [d for d in dirnames if d not in ignore_set and not d.startswith('.') and not any(Path(d).match(p) for p in ignore_set if '*' in p or '?' in p)]
        if len(dirnames) < original_dir_count:
            log.debug(f"Filtered {original_dir_count - len(dirnames)} subdirectories in {current_rel_path}")

        log.debug(f"Processing directory: {current_rel_path} (Files: {len(filenames)}, Subdirs: {len(dirnames)})")

        filenames.sort()
        dirnames.sort() # Ensure deterministic order

        source_files_to_process: List[Tuple[Path, Path, str]] = [] # rel_path, full_path, extension
        other_text_filenames: List[str] = []

        for filename in filenames:
            # Basic ignore checks
            if filename in ignore_set or any(Path(filename).match(p) for p in ignore_set if '*' in p or '?' in p):
                continue
            # Skip hidden files unless explicitly interesting or code
            if filename.startswith('.') and filename.lower() not in interesting_filenames_set and filename not in code_extensions_set:
                continue

            file_path = current_path / filename
            if not file_path.is_file(): # Should not happen with os.walk files, but good check
                continue

            # Skip likely binary files
            if is_likely_binary(file_path):
                log.debug(f"Skipping binary file: {file_path.relative_to(target_path)}")
                continue

            base, ext = os.path.splitext(filename)
            ext_lower = ext.lower()
            fname_lower = filename.lower() # For full filename matches in CODE_EXTENSIONS

            is_source = (ext_lower in code_extensions_set or 
                         filename in code_extensions_set or # Match full filename e.g. 'Makefile'
                         fname_lower in code_extensions_set) # Match case-insensitive full filename

            relative_file_path = file_path.relative_to(target_path)
            if is_source:
                source_files_to_process.append((relative_file_path, file_path, ext_lower))
            else:
                other_text_filenames.append(filename)
        
        # Process source files
        if source_files_to_process:
            # Add a directory header only if it's not the root and has content
            dir_header = f"\n{'=' * 10} Directory: {current_rel_path} {'=' * 10}\n" if str(current_rel_path) != '.' else ""
            has_written_dir_header = False

            for rel_path, full_path, file_ext in source_files_to_process:
                file_id_for_report = get_file_id_from_path_str(rel_path.name) # Use name for consistency
                if file_id_for_report not in report_data: report_data[file_id_for_report] = Counter()
                report_data[file_id_for_report]['processed'] += 1
                
                file_info_prefix = f"--- File: {rel_path}"
                is_json_sample = False

                try:
                    log.debug(f"Processing source file: {rel_path}")
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    hash_before = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    
                    # --- MODIFICATION: Special handling for JSON files ---
                    if file_ext == '.json':
                        log.debug(f"Attempting to generate sample for JSON file: {rel_path}")
                        sample_str, status = generate_json_sample(content)
                        if sample_str:
                            simplified_content = sample_str
                            file_info_prefix += " (Sample Representation)"
                            is_json_sample = True
                            log.info(f"Generated sample for {rel_path} (status: {status})")
                            report_data[file_id_for_report]['json_sampled'] += 1
                        else:
                            log.warning(f"Could not generate sample for {rel_path} (status: {status}). Falling back to standard simplification.")
                            # Fallback: process as regular source code
                            simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, compress_patterns_enabled)
                    else:
                        # Standard simplification for other source files
                        simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, compress_patterns_enabled)
                    # --- END MODIFICATION ---

                    hash_after = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest()
                    
                    is_simplified = hash_before != hash_after
                    is_empty_after_simplification = not simplified_content.strip()
                    
                    content_hash = hash_after # Use hash of the (potentially sampled) content
                    is_duplicate = False

                    if is_simplified and not is_empty_after_simplification and not is_json_sample: # Don't count JSON sampling as "simplification" in this metric
                        report_data[file_id_for_report]['simplified'] += 1
                    
                    if skip_empty and is_empty_after_simplification:
                        log.info(f"Skipping empty file (after simplification/sampling): {rel_path}")
                        report_data[file_id_for_report]['skipped_empty'] += 1
                        continue # Skip to next file

                    if skip_duplicates and not is_empty_after_simplification:
                        if content_hash in seen_content_hashes:
                            is_duplicate = True
                            log.info(f"Skipping duplicate file (based on simplified/sampled content): {rel_path}")
                            report_data[file_id_for_report]['skipped_duplicate'] += 1
                            continue # Skip to next file
                        else:
                            seen_content_hashes.add(content_hash)
                            log.debug(f"Adding new content hash: {content_hash[:8]}... for {rel_path}")
                    
                    # If we reach here, the file content will be included
                    if dir_header and not has_written_dir_header:
                        write_to_buffer(dir_header.strip("\n")) # remove then re-add \n via write_to_buffer
                        has_written_dir_header = True
                    
                    report_data[file_id_for_report]['contributed'] += 1
                    write_to_buffer(file_info_prefix + " ---") # Standard header format
                    if is_empty_after_simplification: # Should only happen if skip_empty is false
                        write_to_buffer("### This file became empty after simplification. ###")
                    else:
                        write_to_buffer(simplified_content.strip()) # Ensure no leading/trailing newlines from content itself
                    write_to_buffer(f"--- End File: {rel_path} ---")

                except OSError as e:
                    log.error(f"Read error during processing of {rel_path}: {e}")
                    report_data[file_id_for_report]['read_error'] += 1
                    if dir_header and not has_written_dir_header: write_to_buffer(dir_header.strip("\n")); has_written_dir_header = True
                    write_to_buffer(f"{file_info_prefix} --- Error Reading File ---")
                except Exception as e:
                    log.error(f"Unexpected error processing file {rel_path}: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
                    report_data[file_id_for_report]['processing_error'] += 1
                    if dir_header and not has_written_dir_header: write_to_buffer(dir_header.strip("\n")); has_written_dir_header = True
                    write_to_buffer(f"{file_info_prefix} --- Error Processing File ---")

        # Summarize other text files in this directory
        if other_text_filenames:
            summary = summarize_other_files(other_text_filenames, code_extensions_set, interesting_filenames_set)
            if summary:
                # Determine proper indentation based on directory depth
                indent_level = len(current_rel_path.parts) if str(current_rel_path) != '.' else 0
                indent = "  " * indent_level if indent_level > 0 else ""
                
                summary_line = f"{indent}{summary}"
                log.debug(f"Adding summary for {current_rel_path}: {summary_line}")

                # Ensure there's a blank line before a summary if it follows file content
                # or another summary, but not if it's right after a directory header.
                buffer.seek(0, io.SEEK_END)
                current_pos = buffer.tell()
                if current_pos > 0: # Check if buffer is not empty
                    buffer.seek(max(0, current_pos - 200)) # Read last part to check context
                    last_part = buffer.read()
                    # Don't add extra newline if previous line was already a dir header for this dir or empty
                    if not last_part.strip().endswith(f"Directory: {current_rel_path} {'=' * 10}") and \
                       not last_part.endswith("\n\n"):
                        write_to_buffer("") # Add a separating newline

                write_to_buffer(summary_line)
                # Add a blank line after the summary for better separation,
                # unless it's the very end of processing (which is hard to tell here)
                write_to_buffer("")


# --- Raw Dump Functions (largely unchanged, included for completeness) ---
def process_folder_raw(
    target_input_path_str: str,
    buffer: io.StringIO,
    # report_data: Dict[str, Counter] # report_data not used by this version of raw dump
    additional_context_folder_str: Optional[str] = None
) -> int: # Returns total files dumped
    """Dumps all files verbatim from target_input_path (file or folder) and optionally an additional context folder."""
    
    paths_to_process: List[Tuple[Path, str]] = [] # List of (Path object, description string)
    
    main_path_obj = Path(target_input_path_str).resolve()
    paths_to_process.append((main_path_obj, "Primary Input"))

    if additional_context_folder_str:
        additional_path_obj = Path(additional_context_folder_str).resolve()
        if additional_path_obj.is_dir():
            paths_to_process.append((additional_path_obj, "Additional Context Folder"))
        elif additional_path_obj.is_file(): # Allow additional context to be a single file too
            paths_to_process.append((additional_path_obj, "Additional Context File"))
        else:
            log.warning(f"Raw Dump: Additional context path '{additional_path_obj}' is not a valid file or directory, skipping.")

    total_files_dumped_across_all_targets = 0

    for current_target_path_obj, current_target_desc in paths_to_process:
        log.debug(f"Starting RAW dump for {current_target_desc}: {current_target_path_obj}")
        file_count_for_this_target = 0

        if current_target_path_obj.is_file():
            # Handle single file input
            buffer.write(f"\n{'=' * 20} START FILE ({current_target_desc}): {current_target_path_obj.name} {'=' * 20}\n")
            try:
                with open(current_target_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if content and not content.endswith('\n'): # Ensure trailing newline
                    content += '\n'
                buffer.write(content)
                file_count_for_this_target = 1
            except Exception as e:
                log.error(f"Error reading/writing file {current_target_path_obj.name} for raw dump ({current_target_desc}): {e}")
                buffer.write(f"### ERROR READING FILE: {e} ###\n")
            buffer.write(f"{'=' * 20} END FILE ({current_target_desc}): {current_target_path_obj.name} {'=' * 20}\n")
        
        elif current_target_path_obj.is_dir():
            # Handle directory input
            buffer.write(f"\n{'=' * 10} STARTING RAW DUMP OF {current_target_desc.upper()}: {current_target_path_obj.name} {'=' * 10}\n")
            # Sort os.walk results for deterministic output
            walk_results = sorted(list(os.walk(current_target_path_obj, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} during raw dump - {e}"))), key=lambda x: x[0])
            
            for dirpath, dirnames, filenames in walk_results:
                dirnames.sort() # Sort dirnames for consistent traversal order
                filenames.sort() # Sort filenames for consistent dump order
                
                current_path_in_walk = Path(dirpath)
                for filename_in_walk in filenames:
                    file_path_in_walk = current_path_in_walk / filename_in_walk
                    try:
                        # Get relative path for display, fallback if current_target_path_obj is not an ancestor (should not happen with os.walk from it)
                        relative_file_path_display = file_path_in_walk.relative_to(current_target_path_obj)
                    except ValueError: # pragma: no cover
                        relative_file_path_display = file_path_in_walk # Fallback to absolute path display
                    
                    if not file_path_in_walk.is_file(): # Should be filtered by os.walk but good practice
                        continue
                    
                    file_count_for_this_target +=1
                    buffer.write(f"\n{'=' * 20} START FILE ({current_target_desc}/{relative_file_path_display}): {filename_in_walk} {'=' * 20}\n")
                    log.info(f"Dumping file ({file_count_for_this_target} for {current_target_desc}): {relative_file_path_display}")
                    try:
                        with open(file_path_in_walk, 'r', encoding='utf-8', errors='ignore') as f:
                            content_f = f.read()
                        if content_f and not content_f.endswith('\n'): # Ensure trailing newline
                            content_f += '\n'
                        buffer.write(content_f)
                    except Exception as e:
                        log.error(f"Error reading/writing file {file_path_in_walk} for raw dump: {e}")
                        buffer.write(f"### ERROR READING FILE: {e} ###\n")
                    buffer.write(f"{'=' * 20} END FILE ({current_target_desc}/{relative_file_path_display}): {filename_in_walk} {'=' * 20}\n")
            buffer.write(f"\n{'=' * 10} FINISHED RAW DUMP OF {current_target_desc.upper()}: {current_target_path_obj.name} ({file_count_for_this_target} files) {'=' * 10}\n")
        else:
            log.error(f"Raw dump target '{current_target_path_obj}' from {current_target_desc} is neither a file nor a directory. Skipping.")
            buffer.write(f"### ERROR: Raw dump target '{current_target_path_obj}' not found or invalid. ###\n")
        
        log.info(f"Finished raw dump for {current_target_desc}. Processed {file_count_for_this_target} file(s).")
        total_files_dumped_across_all_targets += file_count_for_this_target
        
    return total_files_dumped_across_all_targets


def process_folder_raw_source(
    target_folder: str,
    buffer: io.StringIO,
    ignore_set: set,
    code_extensions_set: set,
    interesting_filenames_set: set, # Though not directly used for filtering content, can be logged or used for future features
    report_data: Dict[str, Counter] # For reporting counts
) -> int: # Returns number of source files dumped
    """Dumps only recognized 'source' files verbatim, respecting ignore patterns."""
    target_path = Path(target_folder).resolve()
    log.debug(f"Starting RAW SOURCE folder walk: {target_path}")
    
    source_file_count = 0
    
    # Sort os.walk results for deterministic output
    walk_results = sorted(list(os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Access error {e.filename}: {e}"))), key=lambda x: x[0])

    for dirpath, dirnames, filenames in walk_results:
        current_path = Path(dirpath)
        try:
            current_rel_path = current_path.relative_to(target_path)
        except ValueError: # pragma: no cover
            log.warning(f"Relative path error for {current_path} in raw_source mode. Skipping.")
            dirnames[:] = [] # Don't recurse
            continue

        # Filter ignored directories (modify dirnames in place)
        original_dir_count = len(dirnames)
        dirnames[:] = [d for d in dirnames if d not in ignore_set and not d.startswith('.') and not any(Path(d).match(p) for p in ignore_set if '*' in p or '?' in p)]
        if len(dirnames) < original_dir_count:
            log.debug(f"Filtered {original_dir_count - len(dirnames)} subdirectories in {current_rel_path} (Raw Source Mode)")

        log.debug(f"Processing directory (RAW SOURCE): {current_rel_path} (Files: {len(filenames)}, Subdirs: {len(dirnames)})")
        
        dirnames.sort() # Sort for consistent traversal
        filenames.sort() # Sort for consistent processing

        for filename in filenames:
            # Basic ignore checks
            if filename in ignore_set or any(Path(filename).match(p) for p in ignore_set if '*' in p or '?' in p):
                continue
            if filename.startswith('.') and filename.lower() not in interesting_filenames_set and filename not in code_extensions_set:
                continue # Skip hidden unless interesting or code type

            file_path = current_path / filename
            if not file_path.is_file():
                continue

            # Skip likely binary files (even in raw source, we usually want text)
            if is_likely_binary(file_path):
                log.debug(f"Skipping binary file in Raw Source mode: {file_path.relative_to(target_path)}")
                continue
                
            base, ext = os.path.splitext(filename)
            ext_lower = ext.lower()
            fname_lower = filename.lower()

            is_source = (ext_lower in code_extensions_set or 
                         filename in code_extensions_set or 
                         fname_lower in code_extensions_set)
            
            relative_file_path = file_path.relative_to(target_path)
            file_id = get_file_id_from_path_str(filename) # For reporting

            if is_source:
                if file_id not in report_data: report_data[file_id] = Counter()
                report_data[file_id]['dumped_raw_source'] += 1
                source_file_count += 1
                
                buffer.write(f"\n{'=' * 20} START SOURCE FILE (RAW): {relative_file_path} {'=' * 20}\n")
                log.info(f"Dumping source file (RAW) ({source_file_count}): {relative_file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if content and not content.endswith('\n'): # Ensure trailing newline
                        content += '\n'
                    buffer.write(content)
                except OSError as e:
                    log.error(f"Read error (RAW SOURCE): {relative_file_path}: {e}")
                    report_data[file_id]['read_error_raw_source'] += 1
                    buffer.write(f"### ERROR READING FILE (RAW SOURCE): {e} ###\n")
                except Exception as e: # pragma: no cover
                    log.error(f"Processing error (RAW SOURCE): {relative_file_path}: {e}", exc_info=True)
                    report_data[file_id]['processing_error_raw_source'] += 1
                    buffer.write(f"### ERROR PROCESSING FILE (RAW SOURCE): {e} ###\n")
                buffer.write(f"{'=' * 20} END SOURCE FILE (RAW): {relative_file_path} {'=' * 20}\n")
            # else: (not a source file, so we ignore it in raw_source mode)

    log.info(f"Finished raw source dump. Processed and dumped {source_file_count} source files.")
    return source_file_count

# --- Post-Processing Functions (largely unchanged) ---

def apply_post_simplification_patterns(content: str, patterns: list[tuple[re.Pattern, Any]]) -> tuple[str, int]:
    """Applies a list of regex patterns for further detailed simplification."""
    total_replacements = 0
    lines = content.splitlines(keepends=True) # Keep newlines to preserve structure
    output_lines = []
    log.debug(f"Applying {len(patterns)} post-simplification patterns...")
    pattern_counts = Counter() # To count replacements per pattern

    for line_idx, line in enumerate(lines):
        modified_line = line
        for i, (pattern, replacement) in enumerate(patterns):
            try:
                 original_line_segment = modified_line # For logging diffs
                 if callable(replacement):
                     # Pass match object to callable
                     modified_line_new, count = pattern.subn(replacement, modified_line)
                 else:
                     modified_line_new, count = pattern.subn(replacement, modified_line)
                 
                 if count > 0:
                     total_replacements += count
                     pattern_counts[i] += count
                     # Detailed logging for DEBUG level
                     if log.getEffectiveLevel() <= logging.DEBUG:
                         # Try to find where the change happened for a snippet
                         diff_start = -1
                         for k in range(min(len(original_line_segment), len(modified_line_new))):
                             if original_line_segment[k] != modified_line_new[k]:
                                 diff_start = k
                                 break
                         log_orig_snip = original_line_segment[max(0,diff_start-10):diff_start+20].strip().replace('\n', '\\n')
                         log_new_snip = modified_line_new[max(0,diff_start-10):diff_start+20].strip().replace('\n', '\\n')
                         log.debug(f"  Pattern {i} ({pattern.pattern[:30]}...) matched {count}x on line {line_idx+1}: ...{log_orig_snip}... -> ...{log_new_snip}...")
                     modified_line = modified_line_new
            except Exception as e: # pragma: no cover
                log.error(f"Error applying post-simplification pattern {i} ('{pattern.pattern}') to line: {e}")
                log.debug(f"Problematic line (original): {line.strip()[:100]}")
        output_lines.append(modified_line)
    
    modified_content = "".join(output_lines)
    if pattern_counts:
        log.debug(f"Post-simplification pattern application counts: {dict(pattern_counts)}")
    log.info(f"Applied {len(patterns)} post-simplification patterns, making {total_replacements} replacements.")
    return modified_content, total_replacements


def expand_multi_pattern_lines(content: str, finder_pattern: re.Pattern, pattern_name_for_log: str = "PATTERN") -> tuple[str, int]:
    """
    Expands lines containing multiple instances of a pattern (e.g., multiple Voice IDs on one line)
    onto separate lines. This helps subsequent block compression.
    """
    lines = content.splitlines(keepends=False) # Process without newlines initially
    output_lines = []
    lines_expanded = 0
    log.debug(f"--- Starting expand_multi_pattern_lines: Scanning for multiple '{pattern_name_for_log}' instances per line ---")

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Avoid processing structural lines or already compressed blocks
        if not stripped_line or \
           stripped_line.startswith(("--- File:", "--- End File:", "===")) or \
           re.match(r"^\s*# \.\.\. \d+ similar lines compressed \.\.\.", stripped_line) or \
           re.match(r"^\s*\/\/\s*Sample representation", stripped_line) :
            output_lines.append(line + "\n")
            continue
        
        try:
            matches = finder_pattern.findall(line)
        except Exception as e: # pragma: no cover (regex should be pre-compiled and tested)
            log.error(f"Regex error during expand_multi_pattern_lines on line {i+1}: {e}")
            log.debug(f"Problematic line: {line}")
            matches = [] # Treat as no matches

        if len(matches) > 1:
            # Heuristic: if the combined length of found patterns is a large portion of the line,
            # it's likely a deliberate multi-pattern line rather than coincidental short matches.
            combined_match_len = sum(len(str(m).strip(',').strip()) for m in matches) # m can be tuple from findall if groups are used
            # Approximate stripped length (remove whitespace and commas which might be separators)
            stripped_len_approx = len(re.sub(r'\s+|,', '', stripped_line))

            if stripped_len_approx > 0 and combined_match_len >= stripped_len_approx * 0.7: # If matches make up 70% of content
                log.info(f"Expanding line {i+1} containing {len(matches)} instances of '{pattern_name_for_log}'.")
                log.debug(f"Original line {i+1}: {line.strip()}")
                lines_expanded += 1
                
                indent_level = line.find(stripped_line[0]) if stripped_line else 0
                indent = " " * indent_level
                
                for match_item in matches:
                    # If finder_pattern has groups, match_item might be a tuple.
                    # We assume the relevant part is the first element or the whole string if not tuple.
                    match_str = str(match_item[0] if isinstance(match_item, tuple) else match_item).strip()
                    # Ensure the pattern itself is re-added if findall captured content inside it (e.g. quotes)
                    # This depends heavily on how SINGLE_VOICE_ID_FINDER is defined.
                    # For the given SINGLE_VOICE_ID_PATTERN_STR, it captures the content *inside* quotes.
                    # So we need to re-add quotes if they were part of the original pattern structure.
                    # Assuming the pattern is like '"(content)"'.
                    if SINGLE_VOICE_ID_FINDER.pattern.startswith('"') and SINGLE_VOICE_ID_FINDER.pattern.endswith('"'):
                         expanded_line_content = f'"{match_str}"' # Re-add quotes
                    else:
                         expanded_line_content = match_str

                    output_lines.append(f"{indent}{expanded_line_content}\n")
                    log.debug(f"  Expanded to: {indent}{expanded_line_content}")
                continue # Skip appending the original line
        
        output_lines.append(line + "\n") # Add original line (or if no expansion happened)

    log.info(f"Finished line expansion pre-processing. Expanded {lines_expanded} lines containing multiple '{pattern_name_for_log}' instances.")
    return "".join(output_lines), lines_expanded


def compress_pattern_blocks(content: str, patterns_to_compress: dict[str, re.Pattern], min_consecutive: int) -> tuple[str, int]:
    """Compresses blocks of consecutive lines matching predefined patterns."""
    lines = content.splitlines(keepends=True)
    output_lines = []
    total_blocks_compressed = 0
    i = 0
    log.debug(f"--- Starting compress_pattern_blocks: Scanning for blocks (min_consecutive={min_consecutive}) ---")

    while i < len(lines):
        current_line = lines[i]
        stripped_line = current_line.strip()
        matched_pattern_name = None

        # Skip structural lines, comments, or placeholder lines from being block starters
        is_ignorable_starter = (
            stripped_line.startswith(("--- File:", "--- End File:", "===")) or
            stripped_line.startswith(("#", "//", "/*", "*LINE_REF_")) or # Comments, line refs
            re.match(r"^\s*\/\/\s*Sample representation", stripped_line) or
            re.match(r"^\s*# \.\.\. \d+ similar lines compressed \.\.\.", stripped_line)
        )

        if not is_ignorable_starter:
            for name, pattern in patterns_to_compress.items():
                if pattern.match(stripped_line):
                    matched_pattern_name = name
                    log.debug(f"Line {i+1} potentially starts a block of '{name}': {stripped_line[:60]}...")
                    break
        
        if matched_pattern_name:
            block_pattern_name = matched_pattern_name
            block_start_index = i
            block_lines_indices = [i] # Store indices of lines in the current block
            
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                # Check if the next line is ignorable (e.g., file header, comment) - this would break a block.
                # For block compression, we generally want contiguous data lines.
                next_is_ignorable_continuation = (
                    next_stripped.startswith(("--- File:", "--- End File:", "===")) or
                    next_stripped.startswith(("#", "//", "/*")) # Comments might break data blocks
                )
                if next_is_ignorable_continuation:
                    log.debug(f"  Block '{block_pattern_name}' interrupted at line {j+1} by non-data line: {next_stripped[:60]}...")
                    break 

                if patterns_to_compress[block_pattern_name].match(next_stripped):
                    block_lines_indices.append(j)
                    log.debug(f"  Line {j+1} continues block '{block_pattern_name}'.")
                    j += 1
                else:
                    log.debug(f"  Block '{block_pattern_name}' ended at line {j+1} (no match).")
                    break
            
            block_count = len(block_lines_indices)
            if block_count >= min_consecutive:
                first_line_in_block = lines[block_start_index]
                first_line_stripped = first_line_in_block.strip()
                
                indent = ""
                if len(first_line_in_block) > len(first_line_stripped): # Calculate indent from the first line
                    indent = first_line_in_block[:len(first_line_in_block) - len(first_line_stripped)]
                
                summary_line = f"{indent}# ... {block_count} lines matching '{block_pattern_name}' pattern compressed ...\n"
                output_lines.append(summary_line)
                log.info(f"Compressed {block_count} lines (Indices {block_start_index+1}-{j}) matching '{block_pattern_name}'.")
                total_blocks_compressed += 1
                i = j # Move main index past this compressed block
            else:
                # Block not long enough, append original lines
                log.debug(f"  Block '{block_pattern_name}' starting at line {block_start_index+1} had only {block_count} lines (min={min_consecutive}). Not compressing.")
                for block_line_index in block_lines_indices:
                    output_lines.append(lines[block_line_index])
                i = j # Move main index past this processed (but not compressed) block
        else:
            # No pattern matched, or line was ignorable starter
            output_lines.append(current_line)
            i += 1
            
    log.info(f"Pattern block compression: Compressed {total_blocks_compressed} blocks (min consecutive: {min_consecutive}).")
    return "".join(output_lines), total_blocks_compressed


def minify_repeated_lines(content: str, min_length: int, min_repetitions: int) -> tuple[str, int, Dict[str, Counter]]:
    """Identifies and replaces frequently repeated long lines with placeholders."""
    global DEFINITION_SIMPLIFICATION_PATTERNS # Uses global for simplification patterns
    
    lines = content.splitlines(keepends=True) # Keep newlines for accurate replacement
    line_counts = Counter()
    meaningful_lines_indices: Dict[str, List[int]] = {} # Stores line_content_key -> list of indices

    log.debug(f"--- Starting minify_repeated_lines: Scanning lines >= {min_length} chars, repeated >= {min_repetitions}x ---")

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Heuristic to avoid structural lines, short placeholders, or already compressed lines
        is_structural_or_placeholder = (
            stripped_line.startswith(("--- File:", "--- End File:", "===")) or
            stripped_line.startswith(("*LINE_REF_", "# ...", "// ...")) or # Existing refs, block compressions
            len(stripped_line) < min_length or # Too short even if not caught by pattern
            re.match(r'^\s*(\*([A-Z0-9_]+)\*[,;\s]*)+\s*$', stripped_line) or # Line of only placeholders
            not stripped_line # Empty line
        )
        
        if not is_structural_or_placeholder:
            line_content_key = stripped_line.rstrip() # Use rstrip to normalize trailing whitespace for keying
            line_counts[line_content_key] += 1
            if line_content_key not in meaningful_lines_indices:
                meaningful_lines_indices[line_content_key] = []
            meaningful_lines_indices[line_content_key].append(i)

    replacement_map: Dict[str, str] = {} # Maps original line content (key) to placeholder string
    minified_line_origins: Dict[str, Counter] = {} # Tracks where minified lines came from (file type)
    definition_lines: List[str] = []
    placeholder_counter = 1
    placeholder_template = "*LINE_REF_{}*"

    # Sort by count (desc) then by line content (asc) for deterministic placeholder assignment
    repeated_lines = sorted(
        [(line_content, count) for line_content, count in line_counts.items() if count >= min_repetitions],
        key=lambda item: (-item[1], item[0])
    )
    
    if not repeated_lines:
        log.info("Line minification: No lines met the repetition and length criteria.")
        return content, 0, {}

    # Create a context finder once for all lookups
    find_file_context_for_line = create_file_context_finder(lines)

    for line_content, count in repeated_lines:
        # This check is mostly redundant due to sorting and single pass, but safe
        if line_content not in replacement_map: 
            placeholder = placeholder_template.format(placeholder_counter)
            replacement_map[line_content] = placeholder
            
            log.debug(f"  Creating definition {placeholder} for line repeated {count} times (len {len(line_content)}): {line_content.strip()[:80]}...")
            
            # Track origins for reporting
            indices_of_this_line = meaningful_lines_indices.get(line_content, [])
            for index_of_occurrence in indices_of_this_line:
                file_id = find_file_context_for_line(index_of_occurrence)
                if file_id not in minified_line_origins:
                    minified_line_origins[file_id] = Counter()
                minified_line_origins[file_id]['placeholder_instances_created'] += 1 # Count each instance that will be replaced

            # Simplify the definition line itself for the header
            simplified_definition_content = line_content.rstrip() # Already rstripped for key
            for pattern, replacement_func_or_str in DEFINITION_SIMPLIFICATION_PATTERNS:
                 try:
                     if callable(replacement_func_or_str):
                         simplified_definition_content, _ = pattern.subn(replacement_func_or_str, simplified_definition_content)
                     else:
                         simplified_definition_content = pattern.sub(replacement_func_or_str, simplified_definition_content)
                 except Exception as e: # pragma: no cover
                     log.warning(f"Error simplifying definition content with pattern '{pattern.pattern}': {e}")
            
            definition_lines.append(f"{placeholder} = {simplified_definition_content}")
            placeholder_counter += 1

    if not replacement_map: # Should be caught by `if not repeated_lines` earlier
        log.info("Line minification: No lines were ultimately chosen for replacement.")
        return content, 0, {}

    # Perform replacements
    new_lines = list(lines) # Make a mutable copy
    num_actual_replacements = 0
    for original_line_content_key, placeholder in replacement_map.items():
        indices_to_replace = meaningful_lines_indices.get(original_line_content_key, [])
        for index_val in indices_to_replace:
            # Double check the line at index still matches what we expect (it should)
            # We need to compare the stripped, rstripped version with original_line_content_key
            # And replace the original line new_lines[index_val]
            if index_val < len(new_lines) and new_lines[index_val].strip().rstrip() == original_line_content_key:
                # Preserve original line's whitespace and newline char
                original_full_line = new_lines[index_val]
                leading_whitespace = original_full_line[:-len(original_full_line.lstrip())]
                trailing_newline = "\n" if original_full_line.endswith("\n") else "" # Should always be \n due to splitlines(keepends=True)
                
                new_lines[index_val] = leading_whitespace + placeholder + trailing_newline
                num_actual_replacements += 1
            else: # pragma: no cover (should ideally not happen if logic is sound)
                log.warning(f"Skipped replacing line at index {index_val+1} due to content/index mismatch. Expected: '{original_line_content_key[:50]}...', Found: '{new_lines[index_val].strip()[:50]}...' This might indicate an issue in indexing or prior modification.")

    minified_content = "".join(new_lines)
    
    if definition_lines:
        definition_header = [
            "", "=" * 40,
            f"# Line Minification Definitions ({len(definition_lines)}):",
            "=" * 40
        ]
        definition_block_str = "\n".join(definition_header + definition_lines) + "\n\n" # Ensure trailing newlines
        log.info(f"Line minification: Replaced {num_actual_replacements} occurrences of {len(definition_lines)} unique lines.")
        return definition_block_str + minified_content, num_actual_replacements, minified_line_origins
    else: # Should not happen if replacement_map is populated
        return content, 0, {}


def post_process_cleanup(content: str, cleanup_pattern: re.Pattern) -> tuple[str, int, Dict[str, Counter]]:
    """Removes lines that consist *only* of placeholders, commas, brackets, etc., after all other processing."""
    lines = content.splitlines(keepends=True)
    output_lines = []
    lines_removed = 0
    cleanup_origins_report: Dict[str, Counter] = {} # File type -> count of lines removed

    # Create context finder to attribute removed lines to their original files
    find_file_context_for_cleanup = create_file_context_finder(lines)
    
    log.debug(f"--- Starting post_process_cleanup: Scanning for lines matching cleanup_pattern ---")
    log.debug(f"Cleanup pattern: {cleanup_pattern.pattern}")

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Always keep definition block lines and file/directory markers
        if line.startswith(("*LINE_REF_", "# Line Minification Definitions")) or \
           line.startswith("===") or \
           line.startswith("--- File:") or \
           line.startswith("--- End File:") or \
           re.match(r"^\s*# Other files in", stripped_line) or \
           re.match(r"^\s*# \.\.\. \d+ lines matching", stripped_line) or \
           re.match(r"^\s*\/\/\s*Sample representation", stripped_line) :
            output_lines.append(line)
            continue

        if cleanup_pattern.match(stripped_line): # Match on stripped line
            log.debug(f"Post-cleanup removing line {i+1}: {stripped_line[:80]}...")
            lines_removed += 1
            
            # Attribute removal to file type
            file_id = find_file_context_for_cleanup(i)
            if file_id not in cleanup_origins_report:
                cleanup_origins_report[file_id] = Counter()
            cleanup_origins_report[file_id]['removed_by_cleanup'] += 1
        else:
            output_lines.append(line) 
            
    cleaned_content = "".join(output_lines)
    # Final pass to reduce multiple blank lines that might result from removals
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content) 
    
    log.info(f"Post-processing cleanup removed {lines_removed} lines.")
    return cleaned_content, lines_removed, cleanup_origins_report


def generate_output_file_summary(report: Dict[str, Counter], mode: str) -> str:
    """Generates a summary of file types processed and their status for the output header."""
    if not report:
        return "# No files were processed or met inclusion criteria based on current settings.\n"

    summary_lines = ["# --- Included File Types Summary (" + mode + " Mode) ---"]
    total_files_contributed = 0
    
    # Sort by file extension/ID for consistent output
    sorted_file_ids = sorted(report.keys())

    for file_id in sorted_file_ids:
        stats = report[file_id]
        # Determine the primary count to display for this file type
        # In standard mode, 'contributed' is key. In raw modes, 'dumped' or 'dumped_raw_source'.
        count = 0
        if 'contributed' in stats: # Standard mode
            count = stats['contributed']
        elif 'dumped_raw_source' in stats: # Raw source mode
            count = stats['dumped_raw_source']
        elif 'dumped' in stats: # Raw dump all mode
             count = stats['dumped'] # This might not be populated if raw dump doesn't use report_data

        if count > 0: # Only list file types that contributed content
            summary_lines.append(f"#   - {file_id} : {count:4d}") # Align counts
            total_files_contributed += count
        elif stats.get('processed',0) > 0 and mode == "Standard": # If processed but not contributed (e.g. all skipped)
             summary_lines.append(f"#   - {file_id} : {count:4d} (all instances skipped or empty)")


    summary_lines.append(f"# Total Files Contributed to Output: {total_files_contributed}")
    summary_lines.append("# --- End Summary ---")
    return "\n".join(summary_lines) + "\n"


# --- Main Function ---
def main():
    global DEFAULT_IGNORE_PATTERNS, BASE_IGNORE_PATTERNS, TEST_IGNORE_PATTERNS
    global CODE_EXTENSIONS, INTERESTING_FILENAMES, DEFAULT_LARGE_LITERAL_THRESHOLD
    global seen_content_hashes # Ensure it's accessible

    # Define recommended defaults for some compression args if used with --recommended-max-reduction
    # These are now the defaults for the arguments themselves.
    DEFAULT_MIN_LINE_LENGTH_REC = 50 # Reduced from 140 for more potential minification
    DEFAULT_MIN_REPETITIONS_REC = 3  # Reduced from 2, as 2 is very common.
    DEFAULT_MIN_CONSECUTIVE_REC = 3  # Default for block compression

    parser = argparse.ArgumentParser(
        description="Generate a concise text representation of a folder's content, suitable for LLM context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help
    )

    # --- Mode Selection ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode", default="standard", choices=["standard", "raw_dump", "raw_source_dump"],
        help="Processing mode: 'standard' (default, simplify and compress), "
             "'raw_dump' (dump all files verbatim, ignores most other settings), "
             "'raw_source_dump' (dump only recognized source files verbatim, respects ignores)."
    )

    # --- Core Path Arguments ---
    parser.add_argument("folder_path", help="Path to the target folder or file (for raw_dump mode).")
    parser.add_argument(
        "-o", "--output",
        help="Output file path. If not provided, output goes to stdout."
    )
    parser.add_argument(
        "--additional-context-path", default=None,
        help="(Raw Dump Mode Only) Path to an additional folder or file to also dump verbatim. "
             "Content will be appended after the primary folder_path content."
    )


    # --- File Selection Options (for Standard and Raw Source Dump modes) ---
    sel_group = parser.add_argument_group('File Selection Options (Standard & Raw Source Dump Modes)')
    sel_group.add_argument(
        "--ignore", nargs='+', default=[],
        help="Additional file or directory names/patterns to ignore (e.g., '*.log', 'temp_folder'). Supports glob patterns."
    )
    sel_group.add_argument(
        "--source-ext", nargs='+', default=[],
        help="Additional file extensions or full filenames to treat as source code (e.g., '.my_lang', 'CustomBuildScript')."
    )
    sel_group.add_argument(
        "--interesting-files", nargs='+', default=[],
        help="Additional filenames (without extension, or full name) to explicitly list in summaries or "
             "prevent from being skipped if hidden (e.g., 'CONFIG', '.env.template')."
    )
    sel_group.add_argument(
        "--include-tests", action="store_true", default=False,
        help="Include files and directories that typically match test patterns (e.g., 'tests/', '*_test.py'). Default is to exclude them."
    )

    # --- Standard Mode Processing Options ---
    proc_group = parser.add_argument_group('Standard Mode Processing Options (ignored in raw modes)')
    proc_group.add_argument(
        "--strip-logging", action="store_true", default=False,
        help="Attempt to remove common logging/print statements (e.g., log.info(...), print(...)). Use with caution."
    )
    # Changed to --keep-X to make default behavior 'skip' (more common use case for reduction)
    proc_group.add_argument(
        "--keep-empty", action="store_true", default=False,
        help="Keep files even if they become empty after simplification. Default is to skip them."
    )
    proc_group.add_argument(
        "--keep-duplicates", action="store_true", default=False,
        help="Keep files even if their simplified content is identical to a previously processed file. Default is to skip them."
    )
    proc_group.add_argument(
        "--large-literal-threshold", type=int, default=DEFAULT_LARGE_LITERAL_THRESHOLD,
        help="Min number of simple lines in a list/dict for it to be considered for placeholder compression. "
             "This is automatically disabled if --no-compress-patterns is NOT used (i.e., pattern compression is on)."
    )

    # --- Standard Mode Compression Steps (all default to True, allow disabling) ---
    # These are powerful reduction techniques, good for LLMs.
    comp_group = parser.add_argument_group('Standard Mode Compression Steps (On by default, use --no-X to disable)')
    comp_group.add_argument(
        "--no-preprocess-split-lines", action="store_false", dest="preprocess_split_lines", default=True,
        help="Disable pre-processing that splits lines with multiple identifiable patterns (e.g., Voice IDs) "
             "onto separate lines. Splitting improves block compression."
    )
    comp_group.add_argument(
        "--no-compress-patterns", action="store_false", dest="compress_patterns", default=True,
        help="Disable compression of consecutive lines matching predefined patterns (e.g., lists of Voice IDs, UUIDs) "
             "into summary lines like '# ... N lines matching PATTERN ...'. Enabling this disables large-literal-threshold."
    )
    comp_group.add_argument(
        "--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE_REC,
        help="Min number of consecutive lines matching a pattern to trigger block compression (if --compress-patterns is enabled)."
    )
    comp_group.add_argument(
        "--apply-patterns", action="store_true", default=False, # Default to False, as it can be aggressive
        help="Apply more detailed regex substitutions *after* initial simplification and block compression "
             "(e.g., further UUID formatting, URL simplification, long string shortening). See POST_SIMPLIFICATION_PATTERNS."
    )
    comp_group.add_argument(
        "--no-minify-lines", action="store_false", dest="minify_lines", default=True,
        help="Disable minification of identical long lines. This replaces repeated lines with placeholders (*LINE_REF_N*) "
             "and adds a definition block at the start."
    )
    comp_group.add_argument(
        "--min-line-length", type=int, default=DEFAULT_MIN_LINE_LENGTH_REC,
        help="Min character length for a line to be considered for minification (if --minify-lines is enabled)."
    )
    comp_group.add_argument(
        "--min-repetitions", type=int, default=DEFAULT_MIN_REPETITIONS_REC,
        help="Min number of times an identical line must appear to be minified (if --minify-lines is enabled)."
    )
    comp_group.add_argument(
        "--no-post-cleanup", action="store_false", dest="post_cleanup", default=True,
        help="Disable final cleanup step that removes lines consisting *only* of placeholders, commas, brackets, and whitespace."
    )
    
    # --- General Options ---
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity. DEBUG provides detailed step-by-step info."
    )

    args = parser.parse_args()

    # --- Setup Logging ---
    try:
        log.setLevel(args.log_level.upper())
        logging.getLogger().setLevel(args.log_level.upper()) # Set root logger level as well
        log.info(f"Log level set to {args.log_level.upper()}")
    except ValueError: # pragma: no cover
        log.setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)
        log.warning(f"Invalid log level '{args.log_level}'. Defaulting to WARNING.")

    # --- Initialize Globals ---
    seen_content_hashes.clear()
    total_bytes_written = 0
    process_report_data: Dict[str, Counter] = {} # For file processing stats
    minification_origin_report: Dict[str, Counter] = {} # For line minification stats
    cleanup_origin_report: Dict[str, Counter] = {} # For post-cleanup stats
    
    # --- Validate folder_path based on mode ---
    target_path_obj = Path(args.folder_path)
    if args.mode == "raw_dump": # Can be file or dir
        if not target_path_obj.exists():
            log.critical(f"Error: Raw dump target not found: '{args.folder_path}'")
            sys.exit(1)
    else: # Must be a directory for standard and raw_source_dump
        if not target_path_obj.is_dir():
            log.critical(f"Error: Folder not found or not a directory: '{args.folder_path}' (Required for mode: {args.mode})")
            sys.exit(1)
    
    resolved_path_str = str(target_path_obj.resolve())

    # --- Determine effective settings and print config summary ---
    run_mode_display = args.mode.replace("_", " ").title()

    # Build ignore set
    current_ignore_patterns = BASE_IGNORE_PATTERNS.copy()
    effective_test_inclusion = args.include_tests
    if args.mode == "raw_dump": # Raw dump ignores test filtering logic, dumps all
        effective_test_inclusion = True # Effectively, as it doesn't filter by test patterns
        log.info("Raw Dump mode: Ignoring --include-tests logic, will attempt to dump all files.")
    elif not args.include_tests:
        log.info("Excluding test files/directories by default. Use --include-tests to override.")
        current_ignore_patterns.update(TEST_IGNORE_PATTERNS)
    else: # --include-tests is true and not raw_dump
        log.info("Including test files/directories as per --include-tests.")
    
    current_ignore_patterns.update(args.ignore) # Add user-specified ignores

    # Build code extensions and interesting files sets
    current_code_extensions = CODE_EXTENSIONS.copy()
    current_code_extensions.update(args.source_ext)
    current_interesting_files = INTERESTING_FILENAMES.copy()
    current_interesting_files.update(args.interesting_files)

    # Print run configuration summary to stderr
    print("-" * 40, file=sys.stderr)
    print(f"Starting Folder-to-Text Processing", file=sys.stderr)
    print(f"Target Path: {resolved_path_str}", file=sys.stderr)
    print(f"Output Target: {'Stdout' if not args.output else args.output}", file=sys.stderr)
    print(f"Mode: {run_mode_display}", file=sys.stderr)
    print(f"Log Level: {args.log_level.upper()}", file=sys.stderr)

    if args.mode != "raw_dump":
        print(f"  Include Tests: {effective_test_inclusion} ({'--include-tests flag used' if args.include_tests else ('Default: Exclude tests' if not args.include_tests else 'N/A')})", file=sys.stderr)
        print(f"  User Ignore Patterns: {args.ignore if args.ignore else 'None'}", file=sys.stderr)
        print(f"  User Source Extensions: {args.source_ext if args.source_ext else 'None'}", file=sys.stderr)
        print(f"  User Interesting Files: {args.interesting_files if args.interesting_files else 'None'}", file=sys.stderr)
    else: # Raw Dump mode
        print("  (File selection options like --ignore, --include-tests are mostly bypassed in Raw Dump mode)", file=sys.stderr)
        if args.additional_context_path:
            print(f"  Additional Raw Dump Path: {Path(args.additional_context_path).resolve()}", file=sys.stderr)


    if args.mode == "standard":
        effective_skip_empty = not args.keep_empty
        effective_skip_duplicates = not args.keep_duplicates
        print(f"  Skip Empty Files (after simplify): {effective_skip_empty}", file=sys.stderr)
        print(f"  Skip Duplicate Files (after simplify): {effective_skip_duplicates}", file=sys.stderr)
        print(f"  Strip Logging: {args.strip_logging}", file=sys.stderr)
        
        # Literal compression status depends on --compress-patterns
        # If --compress-patterns is ON (i.e., --no-compress-patterns is FALSE), then literal compression is off.
        literal_comp_disabled_due_to_pattern_comp = args.compress_patterns
        if literal_comp_disabled_due_to_pattern_comp:
            print(f"  Large Literal Compression: DISABLED (due to Pattern Block Compression being active)", file=sys.stderr)
        else:
            print(f"  Large Literal Compression Threshold: {args.large_literal_threshold} lines (Active if Pattern Block Compression is disabled)", file=sys.stderr)
        
        print(f"  Compression - Pre-process Split Lines: {args.preprocess_split_lines}", file=sys.stderr)
        print(f"  Compression - Compress Pattern Blocks: {args.compress_patterns}", file=sys.stderr)
        if args.compress_patterns:
            print(f"    Min Consecutive Lines for Block: {args.min_consecutive}", file=sys.stderr)
        print(f"  Compression - Apply Detailed Patterns: {args.apply_patterns}", file=sys.stderr)
        print(f"  Compression - Minify Identical Lines: {args.minify_lines}", file=sys.stderr)
        if args.minify_lines:
            print(f"    Min Line Length for Minify: {args.min_line_length}", file=sys.stderr)
            print(f"    Min Repetitions for Minify: {args.min_repetitions}", file=sys.stderr)
        print(f"  Compression - Post-process Cleanup: {args.post_cleanup}", file=sys.stderr)
    
    print("-" * 40, file=sys.stderr)

    # --- Prepare Output ---
    output_handle: Union[io.TextIOWrapper, io.StringIO] = sys.stdout
    output_path_obj: Optional[Path] = None
    content_buffer = io.StringIO() # Intermediate buffer for all content generation
    
    # Variables for stats
    files_processed_count = 0 # General counter, specific counts in report_data
    num_lines_expanded, num_pattern_replacements = 0, 0
    num_blocks_compressed, num_lines_minified, num_lines_cleaned_up = 0, 0, 0
    main_content_str = "" # Holds the primary content after processing steps

    try:
        # --- Generate Initial Header for the output file ---
        header_info_lines = [
            f"# Compressed Representation: {resolved_path_str}",
            f"# Mode: {run_mode_display}"
        ]
        if args.mode == "standard":
            header_info_lines.append(f"# Include Tests: {effective_test_inclusion}")
            options_summary = []
            if effective_skip_empty: options_summary.append("skip_empty=True")
            if effective_skip_duplicates: options_summary.append("skip_duplicates=True")
            if args.preprocess_split_lines: options_summary.append("preprocess_split=True")
            if args.compress_patterns: options_summary.append(f"compress_patterns=True({args.min_consecutive})")
            if args.apply_patterns: options_summary.append("apply_patterns=True")
            if args.minify_lines: options_summary.append(f"minify_lines=True({args.min_line_length},{args.min_repetitions})")
            if args.post_cleanup: options_summary.append("post_cleanup=True")
            if args.strip_logging: options_summary.append("strip_logging=True")
            header_info_lines.append(f"# Options: {', '.join(options_summary) if options_summary else 'Default processing'}")
        elif args.mode == "raw_source_dump":
             header_info_lines.append(f"# Include Tests: {effective_test_inclusion}")
        initial_header_str = "\n".join(header_info_lines) + "\n" + "=" * 40 + "\n"
        
        # --- Main Processing Logic based on mode ---
        if args.mode == "raw_dump":
            log.info(f"Starting raw dump for '{resolved_path_str}'...")
            if args.additional_context_path and not Path(args.additional_context_path).exists():
                log.warning(f"Additional context path '{args.additional_context_path}' not found. Dumping primary path only.")
                files_processed_count = process_folder_raw(resolved_path_str, content_buffer, None)
            else:
                files_processed_count = process_folder_raw(resolved_path_str, content_buffer, args.additional_context_path)
            log.info(f"Finished raw dump. Dumped {files_processed_count} file(s).")
            content_buffer.seek(0)
            main_content_str = content_buffer.getvalue()

        elif args.mode == "raw_source_dump":
            log.info(f"Starting raw source folder dump for '{resolved_path_str}'...")
            files_processed_count = process_folder_raw_source(
                resolved_path_str, content_buffer,
                current_ignore_patterns, current_code_extensions, current_interesting_files,
                process_report_data
            )
            log.info(f"Finished raw source folder dump. Dumped {files_processed_count} source files.")
            content_buffer.seek(0)
            main_content_str = content_buffer.getvalue()

        else: # Standard mode
            log.info("Starting standard folder processing (Step 1: File Traversal & Initial Simplification)...")
            effective_skip_empty = not args.keep_empty
            effective_skip_duplicates = not args.keep_duplicates
            # Determine if large literal compression should be active
            disable_large_literal_comp = args.compress_patterns
            
            process_folder(
                resolved_path_str, content_buffer,
                current_ignore_patterns, current_code_extensions, current_interesting_files,
                effective_skip_empty, args.strip_logging, effective_skip_duplicates,
                args.large_literal_threshold, disable_large_literal_comp,
                process_report_data
            )
            log.info("Finished initial folder processing and simplification.")
            content_buffer.seek(0)
            processed_content_after_step1 = content_buffer.getvalue()
            
            # --- Apply Optional Compression/Cleanup Steps (Standard Mode) ---
            current_processed_content = processed_content_after_step1
            
            if args.preprocess_split_lines and current_processed_content.strip():
                log.info("Step 2: Expanding multi-pattern lines...")
                current_processed_content, num_lines_expanded = expand_multi_pattern_lines(current_processed_content, SINGLE_VOICE_ID_FINDER, "VOICE_ID")
                log.info(f"Finished Step 2 (Line Expansion). Expanded {num_lines_expanded} lines.")

            if args.compress_patterns and current_processed_content.strip():
                log.info("Step 3: Compressing pattern blocks...")
                current_processed_content, num_blocks_compressed = compress_pattern_blocks(current_processed_content, BLOCK_COMPRESSION_PATTERNS, args.min_consecutive)
                log.info(f"Finished Step 3 (Pattern Block Compression). Compressed {num_blocks_compressed} blocks.")

            if args.apply_patterns and current_processed_content.strip():
                log.info("Step 4: Applying detailed post-simplification patterns...")
                current_processed_content, num_pattern_replacements = apply_post_simplification_patterns(current_processed_content, POST_SIMPLIFICATION_PATTERNS)
                log.info(f"Finished Step 4 (Detailed Patterns). Made {num_pattern_replacements} replacements.")

            if args.minify_lines and current_processed_content.strip():
                log.info("Step 5: Minifying identical long lines...")
                current_processed_content, num_lines_minified, minification_origin_report = minify_repeated_lines(current_processed_content, args.min_line_length, args.min_repetitions)
                log.info(f"Finished Step 5 (Line Minification). Minified {num_lines_minified} line instances into {len(minification_origin_report)} definitions.")
            
            if args.post_cleanup and current_processed_content.strip():
                log.info("Step 6: Applying post-processing cleanup...")
                current_processed_content, num_lines_cleaned_up, cleanup_origin_report = post_process_cleanup(current_processed_content, PLACEHOLDER_CLEANUP_PATTERN)
                log.info(f"Finished Step 6 (Post-Cleanup). Removed {num_lines_cleaned_up} lines.")
            else:
                log.debug("Skipping Step 6: Post-process cleanup (disabled or no content).")
            
            main_content_str = current_processed_content
        
        # --- Generate File Summary Block (after all processing) ---
        file_summary_block_str = ""
        if args.mode != "raw_dump": # Raw dump doesn't use process_report_data in this version
            file_summary_block_str = generate_output_file_summary(process_report_data, run_mode_display)
        
        # --- Combine Header, Summary, and Main Content ---
        final_output_to_write = initial_header_str + file_summary_block_str + main_content_str

        # --- Write to Output ---
        log.info("Preparing to write final output...")
        if args.output:
            output_path_obj = Path(args.output).resolve()
            output_path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
            output_handle = open(output_path_obj, 'w', encoding='utf-8')
            log.debug(f"Opening output file for writing: {output_path_obj}")
        else: # output_handle is already sys.stdout
            log.debug("Using stdout for output.")
            
        output_handle.write(final_output_to_write)
        total_bytes_written = len(final_output_to_write.encode('utf-8')) # Calculate bytes from string
        log.info(f"Finished writing output. Total bytes written: {total_bytes_written}.")

        # --- Final Summary to Stderr ---
        print("-" * 40, file=sys.stderr)
        print("Processing Complete.", file=sys.stderr)
        print(f"Mode: {run_mode_display}", file=sys.stderr)
        if args.mode != "raw_dump":
            print(f"Include Tests Setting: {effective_test_inclusion}", file=sys.stderr)

        if args.mode == "standard":
            print("\n--- Overall Post-Processing Stage Summary (Standard Mode) ---", file=sys.stderr)
            if args.preprocess_split_lines: print(f"  Line Expansion Pre-processing: Expanded {num_lines_expanded} lines.", file=sys.stderr)
            if args.compress_patterns: print(f"  Pattern Block Compression: Created {num_blocks_compressed} summary lines (from blocks of {args.min_consecutive}+).", file=sys.stderr)
            if args.apply_patterns: print(f"  Detailed Pattern Application: Made {num_pattern_replacements} replacements.", file=sys.stderr)
            if args.minify_lines: print(f"  Identical Line Minification: Created {len(minification_origin_report)} definition(s) replacing {num_lines_minified} original line instances.", file=sys.stderr)
            if args.post_cleanup: print(f"  Post-processing Cleanup: Removed {num_lines_cleaned_up} lines.", file=sys.stderr)
            
            if args.minify_lines and minification_origin_report:
                print("\n--- Minification Origin Report (Placeholder Instances Created per File Type) ---", file=sys.stderr)
                total_minified_instances_reported = 0
                sorted_minify_origins = sorted(minification_origin_report.items())
                for file_id, counts in sorted_minify_origins:
                    instances = counts.get('placeholder_instances_created', 0)
                    if instances > 0:
                        print(f"  {file_id if file_id else '<unknown_context>'}: {instances} instances replaced by placeholders", file=sys.stderr)
                        total_minified_instances_reported += instances
                print(f"  --------------------------------------------------", file=sys.stderr)
                print(f"  Total placeholder instances created: {total_minified_instances_reported} (should match total lines minified: {num_lines_minified})", file=sys.stderr)
                if total_minified_instances_reported != num_lines_minified:
                     log.warning(f"Mismatch: Reported minified instances ({total_minified_instances_reported}) vs replacements made ({num_lines_minified}).")

            if args.post_cleanup and cleanup_origin_report:
                print("\n--- Cleanup Origin Report (Approximate Lines Removed per File Type) ---", file=sys.stderr)
                print("  (Note: Line attribution is to the file context where the line existed before removal)", file=sys.stderr)
                print("  (This can be imprecise if file markers themselves were removed or content shifted significantly)", file=sys.stderr)
                total_cleaned_up_reported = 0
                sorted_cleanup_origins = sorted(cleanup_origin_report.items())
                for file_id, counts in sorted_cleanup_origins:
                    removed_count = counts.get('removed_by_cleanup', 0)
                    if removed_count > 0:
                        print(f"  {file_id if file_id else '<unknown_context>'}: {removed_count} lines removed", file=sys.stderr)
                        total_cleaned_up_reported += removed_count
                print(f"  --------------------------------------------------", file=sys.stderr)
                print(f"  Total lines reported as cleaned: {total_cleaned_up_reported} (should match total lines cleaned: {num_lines_cleaned_up})", file=sys.stderr)
                if total_cleaned_up_reported != num_lines_cleaned_up:
                     log.warning(f"Mismatch: Reported cleaned lines ({total_cleaned_up_reported}) vs total removed ({num_lines_cleaned_up}).")
            
            print("\n--- File Type Processing Report (Initial Processing & Contribution - Standard Mode) ---", file=sys.stderr)
            if not process_report_data:
                print("  No files were processed or met inclusion criteria.", file=sys.stderr)
            else:
                sorted_file_ids = sorted(process_report_data.keys())
                totals = Counter()
                for file_id in sorted_file_ids:
                    stats = process_report_data[file_id]
                    print(f"  {file_id if file_id else '<no_ext>'}:", file=sys.stderr)
                    for key, val in stats.items():
                        if val > 0: print(f"    - {key.replace('_', ' ').capitalize()}: {val}", file=sys.stderr)
                        totals[key] += val # Aggregate totals
                print("  --------------------------------------------------", file=sys.stderr)
                print("  Overall Totals:", file=sys.stderr)
                for key, val in sorted(totals.items()):
                     if val > 0: print(f"    - {key.replace('_', ' ').capitalize()}: {val}", file=sys.stderr)

        elif args.mode == "raw_source_dump":
            print(f"  Dumped content of {files_processed_count} source files.", file=sys.stderr)
            # Optionally print report_data for raw_source_dump too
            if process_report_data:
                print("\n--- File Type Dump Report (Raw Source Mode) ---", file=sys.stderr)
                for file_id, counts in sorted(process_report_data.items()):
                    dump_count = counts.get('dumped_raw_source', 0)
                    if dump_count > 0: print(f"  {file_id}: Dumped {dump_count} file(s)", file=sys.stderr)
        else: # Raw Dump (all files)
            print(f"  Dumped content of {files_processed_count} files.", file=sys.stderr)
        
        print("-" * 40, file=sys.stderr)
        if args.output and output_path_obj:
            print(f"Total bytes written to {output_path_obj}: {total_bytes_written}", file=sys.stderr)
        else:
            print(f"Total bytes written to stdout: {total_bytes_written}", file=sys.stderr)
        print("-" * 40, file=sys.stderr)

    except IOError as e: # pragma: no cover
        log.critical(f"I/O Error: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
        sys.exit(1)
    except KeyboardInterrupt: # pragma: no cover
        log.warning("\nProcessing interrupted by user (Ctrl+C). Exiting.")
        sys.exit(1)
    except Exception as e: # pragma: no cover
        log.critical(f"An unexpected error occurred: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
        sys.exit(1)
    finally:
        if content_buffer:
            content_buffer.close()
        if args.output and output_handle is not sys.stdout and output_handle: # Check if it's a file handle
            try:
                output_handle.close()
                log.debug(f"Closed output file: {args.output}")
            except Exception as e: # pragma: no cover
                log.error(f"Error closing output file '{args.output}': {e}")

if __name__ == "__main__":
    main()