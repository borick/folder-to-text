#!/usr/bin/env python3

# --- folder_to_text.py ---
# Processes a primary input (folder or file), and optionally an additional context folder.
# Simplifies text files, summarizes others, compresses patterns/lines. Includes raw dump.
# Provides a content analysis report of the final output, with Code/Text broken down by extension.

import os
import re
import argparse
import sys
import string
from pathlib import Path
from collections import Counter, defaultdict
import logging
import hashlib
import io
import json # Added for JSON processing
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

# --- Constants and Patterns (many are unchanged, will be included in the full script) ---
FILE_HEADER_RE = re.compile(r"^--- File: (.*) ---$")

# Configure logging FIRST, then get the logger instance.
# The level set here will be the default if --log-level is not provided or if other modules log.
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(name)s:%(funcName)s:%(lineno)d: %(message)s', stream=sys.stderr)
log = logging.getLogger(__name__) # Get a logger specific to this module

# --- Default Ignore Patterns ---
BASE_IGNORE_PATTERNS = {
    '.git', '__pycache__', '.svn', '.hg', '.idea', '.vscode', 'node_modules',
    'build', 'dist', 'target', 'venv', '.venv', 'env', 'envs', 'conda', 'anaconda3', 'AppData',
    '.DS_Store', 'Thumbs.db',
    'discord', # <--- THIS IS THE KEY IGNORE FOR THE USER'S PROBLEM
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
    #(re.compile(r'\d{8,}'), '*NUM_LONG*'),
    # Obfuscate long hex strings (potential hashes, IDs)
    #(re.compile(r'\b[a-fA-F0-9]{12,}\b'), '*HEX_LONG*'),
    # Obfuscate long Base64-like strings
    #(re.compile(r'[a-zA-Z0-9+/=]{30,}'), '*BASE64LIKE_LONG*'),
    # Common float and int patterns (generic)
    #(re.compile(r'\b\d+\.\d+\b'), '*FLOAT*'),
    #(re.compile(r'\b\d+\b'), '*INT*'),
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

# --- Content Report Patterns (for character count analysis) ---
# Order matters: More specific patterns should come before general ones.
REPORT_STRUCTURAL_PATTERNS = [
    re.compile(r"^\s*--- File:.*---$"),                         # Std mode file start (pattern 0 for ext tracking)
    re.compile(r"^\s*--- End File:.*---$"),
    re.compile(r"^\s*={10,}\s*Directory.*={10,}$"),
    re.compile(r"^\s*={10,}\s*STARTING RAW DUMP.*={10,}$"),
    re.compile(r"^\s*={10,}\s*FINISHED RAW DUMP.*={10,}$"),
    re.compile(r"^\s*={10,}\s*STARTING ADDITIONAL CONTEXT.*={10,}$"),
    re.compile(r"^\s*={10,}\s*FINISHED ADDITIONAL CONTEXT.*={10,}$"),
    re.compile(r"^\s*={20,}\s*START FILE.*={20,}$"),            # Raw dump file start (pattern 7 for ext tracking)
    re.compile(r"^\s*={20,}\s*END FILE.*={20,}$"),              # Raw dump file end
    re.compile(r"^\s*# (RAW DUMP|Compressed Representation) of.*"), # Script header
    re.compile(r"^\s*# Generated by folder_to_text\.py.*"),      # Script header
    re.compile(r"^\s*# Options:.*"),                             # Script header
    re.compile(r"^\s*={40,}$"),                                  # Separator line (====...)
    re.compile(r"^\s*# Line Minification Definitions.*"),         # Minification header
    re.compile(r"^\s*\*LINE_REF_\d+\*\s*=.*")                   # Minification definition line
]

REPORT_PLACEHOLDER_SUMMARY_PATTERNS = [
    re.compile(r"^\s*## \[Compressed Block:.*\] ##\s*$"),            # Compressed block summary
    re.compile(r"^\s*# --- Large literal collection compressed.*---\s*$"), # Literal compression summary
    re.compile(r"^\s*# \(File is empty or contained only comments/whitespace/logging.*\)\s*$"), # Empty file summary
    re.compile(r"^\s*# Other files in.*"), # Summary of other files
    re.compile(r"^\s*\*LINE_REF_\d+\*\s*$"),                        # Minification placeholder usage
    # PLACEHOLDER_CLEANUP_PATTERN is also used dynamically for lines with only placeholders
]

REPORT_COMMENT_PATTERNS = [
    re.compile(r"^\s*#.*"), # Catches general comments. Specific # lines are caught by Structural/Placeholder patterns first.
    re.compile(r"^\s*//.*"),
    re.compile(r"^\s*--.*"),
]

# Regexes for extracting filenames from markers for extension detection in report
STD_FILE_MARKER_EXTRACT_RE = re.compile(r"^\s*--- File(?: \([^)]+\))?: ([^ ]+)")
RAW_FILE_MARKER_EXTRACT_RE = re.compile(r"^\s*={10,}\s*START FILE \((?:(?:[^/]+)/)?([^)]+)\): ([^=]+)\s*={10,}")


# --- Globals ---
seen_content_hashes = set()

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
    disable_literal_compression: bool = False
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

        # Strip logging statements
        if strip_logging and stripped_line:
            # Basic logging pattern: log.info(...), console.log(...), print(...), System.out.println(...) etc.
            if re.match(r'^(?:log(?:ger|ging)?|console|print(?:ln|f|Error)?|fmt\.Print|System\.(?:out|err)\.print|TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|EXCEPTION|CRITICAL)\b', stripped_line, re.IGNORECASE):
                 if '(' in stripped_line and stripped_line.endswith(')'): # Simplistic check for function call
                    # Avoid stripping if it's complex, e.g. print(a) + print(b)
                    if stripped_line.count('(') == 1 and stripped_line.count(')') == 1:
                        if not re.search(r'\)\s*\+\s*\(', stripped_line): # check for `)... + ...(`
                             log.debug(f"Stripping logging line: {original_line.strip()}")
                             continue # Skip this line
        
        # Literal block compression logic
        is_potential_start = list_dict_start_pattern.search(line) is not None
        is_end_char = stripped_line.endswith(']') or stripped_line.endswith('}') or \
                      stripped_line.endswith('],') or stripped_line.endswith('},')
        
        current_line_index = len(simplified_lines) # Index in the *output* lines

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
            elif is_end_char: # End of the block (Fixed: was is_end)
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

def process_single_file_standard(
    target_file_path_str: str,
    buffer: io.StringIO,
    strip_logging: bool,
    large_literal_threshold: int,
    disable_literal_compression: bool, # True if pattern compression is on
    report_data: Dict[str, Counter],
    is_additional_context: bool = False # Added parameter
):
    """Processes a single file in standard mode."""
    def write_to_buffer(text): buffer.write(text + "\n")

    target_file_path = Path(target_file_path_str)
    log_prefix = "Additional Context File: " if is_additional_context else "Primary File: "
    log.debug(f"{log_prefix}Processing: {target_file_path} (Standard Mode)")

    try:
        # Try to get a relative path for display if CWD is an ancestor
        if Path.cwd() in target_file_path.parents or Path.cwd() == target_file_path.parent:
            display_path_str = str(target_file_path.relative_to(Path.cwd()))
        else:
            display_path_str = str(target_file_path)
    except ValueError:
        display_path_str = str(target_file_path)

    file_info_prefix = f"--- File: {display_path_str}"
    if is_additional_context:
        file_info_prefix = f"--- File (from additional context): {display_path_str}"
    
    file_id_for_report = get_file_id_from_path_str(target_file_path.name)
    if file_id_for_report not in report_data:
        report_data[file_id_for_report] = Counter()
    report_data[file_id_for_report]['processed'] += 1


    if is_likely_binary(target_file_path):
        log.warning(f"{log_prefix}Skipping likely binary: {target_file_path}")
        write_to_buffer(file_info_prefix + " --- SKIPPED (BINARY) ---"); write_to_buffer("")
        report_data[file_id_for_report]['skipped_binary'] += 1
        return

    try:
        log.debug(f"{log_prefix}Reading and simplifying source file: {target_file_path}")
        with open(target_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        hash_before = hashlib.sha256(content.encode('utf-8')).hexdigest()
        is_json_sample = False
        file_ext = target_file_path.suffix.lower()

        if file_ext == '.json':
            log.debug(f"Attempting to generate sample for JSON file: {target_file_path}")
            sample_str, status = generate_json_sample(content)
            if sample_str:
                simplified_content = sample_str
                file_info_prefix_display = file_info_prefix + " (Sample Representation)"
                is_json_sample = True
                log.info(f"Generated sample for {target_file_path} (status: {status})")
                report_data[file_id_for_report]['json_sampled'] += 1
            else:
                log.warning(f"Could not generate sample for {target_file_path} (status: {status}). Falling back to standard simplification.")
                simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, disable_literal_compression)
                file_info_prefix_display = file_info_prefix
        else:
            simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, disable_literal_compression)
            file_info_prefix_display = file_info_prefix
        
        hash_after = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest()
        is_simplified = hash_before != hash_after
        is_empty_after_simplification = not simplified_content.strip()

        if is_simplified and not is_empty_after_simplification and not is_json_sample:
            report_data[file_id_for_report]['simplified'] += 1

        # Single file processing doesn't typically use skip_empty or skip_duplicates logic
        # It's usually about processing THIS specific file. If that behavior is desired,
        # it would need to be passed in or handled differently.

        write_to_buffer(file_info_prefix_display + " ---")
        if is_empty_after_simplification:
            write_to_buffer("# (File is empty or contained only comments/whitespace/logging after simplification)")
            report_data[file_id_for_report]['empty_after_simplification'] += 1
        else:
            write_to_buffer(simplified_content.strip())
        write_to_buffer(f"--- End File: {display_path_str} ---"); write_to_buffer("")

    except OSError as e:
        log.error(f"{log_prefix}Error reading file {target_file_path}: {e}")
        write_to_buffer(file_info_prefix + " Error Reading ---"); write_to_buffer(f"### Error reading file: {type(e).__name__}: {e} ###"); write_to_buffer(f"--- End File: {display_path_str} ---"); write_to_buffer("")
        report_data[file_id_for_report]['read_error'] += 1
    except Exception as e:
        log.error(f"{log_prefix}Error processing file {target_file_path}: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
        write_to_buffer(file_info_prefix + " Error Processing ---"); write_to_buffer(f"### Error processing file: {type(e).__name__}: {e} ###"); write_to_buffer(f"--- End File: {display_path_str} ---"); write_to_buffer("")
        report_data[file_id_for_report]['processing_error'] += 1


def process_folder(
    target_folder_str: str,
    buffer: io.StringIO,
    ignore_set: set,
    code_extensions_set: set,
    interesting_filenames_set: set,
    skip_empty: bool,
    strip_logging: bool,
    skip_duplicates: bool,
    large_literal_threshold: int,
    disable_literal_compression: bool,
    report_data: Dict[str, Counter]
):
    def write_to_buffer(text): buffer.write(text + "\n")

    target_path = Path(target_folder_str).resolve()
    if not target_path.is_dir():
        log.error(f"Input to process_folder is not a directory: {target_path}")
        return

    log.debug(f"--- Starting Standard Folder Walk ---")
    log.debug(f"Root: {target_path}")
    log.debug(f"Effective Ignore Set: {sorted(list(ignore_set))}")
    log.debug(f"Effective Code Extensions: {code_extensions_set}")
    log.debug(f"Skips: empty={skip_empty}, duplicates={skip_duplicates}")
    log.debug(f"Strip logging: {strip_logging}")
    log.debug(f"Disable large literal compression (if pattern comp is on): {disable_literal_compression}")

    for dirpath_str, dirnames_orig_iter, filenames_orig_iter in os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"OS Walk Error: Cannot access {e.filename} - {e}")):
        current_walk_path = Path(dirpath_str)
        dirnames_orig = sorted(list(dirnames_orig_iter)) # Sort for consistent processing and logging
        filenames_orig = sorted(list(filenames_orig_iter))

        try:
            current_rel_path_from_target = current_walk_path.relative_to(target_path)
        except ValueError:
            log.warning(f"Path Error: Cannot get relative path for '{current_walk_path}' "
                        f"relative to walk root '{target_path}'. Skipping this path.")
            dirnames_orig_iter[:] = [] # Modify the list os.walk uses
            continue

        log.debug(f"\nDIR_WALK: Processing directory '{current_rel_path_from_target}' (Abs: '{current_walk_path}')")
        log.debug(f"DIR_WALK: Original subdirectories before filtering: {dirnames_orig}")

        dirs_to_remove_from_walk = []
        for d_name in dirnames_orig:
            potential_subdir_rel_to_target = current_rel_path_from_target / d_name

            if d_name in ignore_set:
                log.debug(f"  DIR_IGNORE (Exact Name): '{d_name}' in '{current_rel_path_from_target}'. Marking for removal from walk.")
                dirs_to_remove_from_walk.append(d_name)
                continue
            if d_name.startswith('.') and d_name not in interesting_filenames_set:
                log.debug(f"  DIR_IGNORE (Hidden): '{d_name}' in '{current_rel_path_from_target}'. Marking for removal from walk.")
                dirs_to_remove_from_walk.append(d_name)
                continue

            is_glob_ignored_path = False
            for p_glob in ignore_set:
                if not ('*' in p_glob or '?' in p_glob or '[' in p_glob): continue
                if potential_subdir_rel_to_target.match(p_glob):
                    log.debug(f"  DIR_IGNORE (Glob on RelPath): '{potential_subdir_rel_to_target}' matched by glob '{p_glob}'. Marking for removal from walk.")
                    is_glob_ignored_path = True
                    break
            if is_glob_ignored_path:
                dirs_to_remove_from_walk.append(d_name)
                continue
            
            log.debug(f"  DIR_KEEP: '{d_name}' in '{current_rel_path_from_target}'. Will be explored by os.walk if not removed by name.")

        # Modify the list that os.walk uses for recursion
        for d_to_remove in dirs_to_remove_from_walk:
            if d_to_remove in dirnames_orig_iter: # Check if still present before removing
                 dirnames_orig_iter.remove(d_to_remove)
        
        log.debug(f"DIR_WALK: Subdirectories os.walk will explore next in '{current_rel_path_from_target}': {sorted(list(dirnames_orig_iter))}")


        source_files_to_process: List[Tuple[Path, Path, str]] = [] 
        other_text_filenames_in_curr_dir: List[str] = []

        log.debug(f"DIR_WALK: Considering files in '{current_rel_path_from_target}': {filenames_orig}")
        for filename in filenames_orig:
            file_path_abs = current_walk_path / filename
            file_rel_to_target = file_path_abs.relative_to(target_path)

            log.debug(f"  FILE_CONSIDER: '{filename}' (Full Rel: '{file_rel_to_target}')")

            if filename in ignore_set:
                log.debug(f"    FILE_IGNORE (Exact Name): '{filename}'. SKIPPING.")
                continue
            
            if filename.startswith('.') and \
               filename.lower() not in interesting_filenames_set and \
               filename not in code_extensions_set and \
               Path(filename).suffix.lower() not in code_extensions_set:
                log.debug(f"    FILE_IGNORE (Hidden): '{filename}'. SKIPPING.")
                continue

            is_filename_glob_ignored = False
            for p_glob in ignore_set:
                if '*' not in p_glob and '?' not in p_glob and '[' not in p_glob: continue
                if Path(filename).match(p_glob):
                    log.debug(f"    FILE_IGNORE (Glob on Filename): '{filename}' matched by glob '{p_glob}'. SKIPPING.")
                    is_filename_glob_ignored = True
                    break
            if is_filename_glob_ignored:
                continue
            
            # This check is slightly redundant if directory pruning is perfect, but good for safety.
            is_parent_path_glob_ignored = False
            # Check against the relative path from the *target_path* (initial walk root)
            for p_glob_for_parent_check in ignore_set:
                if not ('*' in p_glob_for_parent_check or '?' in p_glob_for_parent_check or '[' in p_glob_for_parent_check): continue
                # Check if any parent component of current_rel_path_from_target matches the glob
                # This correctly checks if 'discord/file.py' should be skipped if 'discord' is a glob ignore
                for i_parent in range(len(current_rel_path_from_target.parts)):
                    path_component_to_check = Path(*current_rel_path_from_target.parts[:i_parent+1])
                    if path_component_to_check.match(p_glob_for_parent_check):
                        log.debug(f"    FILE_IGNORE (Parent Path Glob): Parent '{path_component_to_check}' of file '{file_rel_to_target}' "
                                  f"matched glob '{p_glob_for_parent_check}'. SKIPPING.")
                        is_parent_path_glob_ignored = True
                        break
                if is_parent_path_glob_ignored:
                    break
            if is_parent_path_glob_ignored:
                continue

            if not file_path_abs.is_file(): 
                log.debug(f"    FILE_SKIP (Not a File): '{filename}'.")
                continue

            if is_likely_binary(file_path_abs):
                log.debug(f"    FILE_SKIP (Binary): '{filename}'.")
                report_data.setdefault(get_file_id_from_path_str(filename), Counter())['skipped_binary_in_folder'] += 1
                continue

            base, ext = os.path.splitext(filename)
            ext_lower = ext.lower()
            is_source = (ext_lower in code_extensions_set or 
                         filename in code_extensions_set or
                         filename.lower() in code_extensions_set)

            if is_source:
                log.debug(f"    FILE_ADD (Source): '{filename}' (Ext: '{ext_lower}').")
                source_files_to_process.append((file_rel_to_target, file_path_abs, ext_lower))
            else:
                log.debug(f"    FILE_ADD (Other Text): '{filename}'.")
                other_text_filenames_in_curr_dir.append(filename)
        
        if source_files_to_process:
            dir_header = f"\n{'=' * 10} Directory: {current_rel_path_from_target} {'=' * 10}\n" if str(current_rel_path_from_target) != '.' else ""
            has_written_dir_header = False

            for rel_path, full_path, file_ext in source_files_to_process:
                file_id_for_report = get_file_id_from_path_str(rel_path.name) 
                if file_id_for_report not in report_data: report_data[file_id_for_report] = Counter()
                report_data[file_id_for_report]['processed'] += 1
                
                file_info_prefix = f"--- File: {rel_path}"
                is_json_sample = False

                try:
                    log.debug(f"Opening and simplifying source file: {rel_path}")
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    hash_before = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    
                    if file_ext == '.json':
                        log.debug(f"Attempting to generate sample for JSON file: {rel_path}")
                        sample_str, status = generate_json_sample(content)
                        if sample_str:
                            simplified_content = sample_str
                            file_info_prefix_display = file_info_prefix + " (Sample Representation)"
                            is_json_sample = True
                            log.info(f"Generated sample for {rel_path} (status: {status})")
                            report_data[file_id_for_report]['json_sampled'] += 1
                        else:
                            log.warning(f"Could not generate sample for {rel_path} (status: {status}). Falling back to standard simplification.")
                            simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, disable_literal_compression)
                            file_info_prefix_display = file_info_prefix
                    else:
                        simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, disable_literal_compression)
                        file_info_prefix_display = file_info_prefix

                    hash_after = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest()
                    
                    is_simplified = hash_before != hash_after
                    is_empty_after_simplification = not simplified_content.strip()
                    
                    if is_simplified and not is_empty_after_simplification and not is_json_sample: 
                        report_data[file_id_for_report]['simplified'] += 1
                    
                    if skip_empty and is_empty_after_simplification:
                        log.info(f"Skipping empty file (after simplification/sampling): {rel_path}")
                        report_data[file_id_for_report]['skipped_empty'] += 1
                        continue 

                    if skip_duplicates and not is_empty_after_simplification:
                        content_hash = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest()
                        if content_hash in seen_content_hashes:
                            log.info(f"Skipping duplicate content file: {full_path} (Hash: {content_hash[:8]}...)")
                            report_data[file_id_for_report]['skipped_duplicate'] += 1
                            continue 
                        seen_content_hashes.add(content_hash)
                        log.debug(f"Adding new content hash: {content_hash[:8]}... for {full_path}")

                    if dir_header and not has_written_dir_header:
                        write_to_buffer(dir_header.strip("\n"))
                        has_written_dir_header = True
                    
                    write_to_buffer(file_info_prefix_display + " ---")
                    if is_empty_after_simplification:
                        write_to_buffer("# (File is empty or contained only comments/whitespace/logging after simplification)")
                        if not skip_empty: 
                             report_data[file_id_for_report]['kept_empty'] +=1
                    else:
                        write_to_buffer(simplified_content.strip())
                    write_to_buffer(f"--- End File: {rel_path} ---"); write_to_buffer("") 
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

        if other_text_filenames_in_curr_dir:
            summary = summarize_other_files(other_text_filenames_in_curr_dir, code_extensions_set, interesting_filenames_set)
            if summary:
                indent_level = len(current_rel_path_from_target.parts) if str(current_rel_path_from_target) != '.' else 0
                indent = "  " * indent_level if indent_level > 0 else ""
                summary_line = f"{indent}{summary}"
                log.debug(f"Adding summary for '{current_rel_path_from_target}': {summary_line}")

                buffer.seek(0, io.SEEK_END)
                original_end_pos = buffer.tell()
                last_part_to_check = ""
                if original_end_pos > 0:
                    buffer.seek(max(0, original_end_pos - 200))
                    last_part_to_check = buffer.read()
                    buffer.seek(original_end_pos) 

                if original_end_pos > 0 and \
                   not last_part_to_check.endswith("\n\n") and \
                   not (dir_header and last_part_to_check.strip().endswith(dir_header.strip())):
                    write_to_buffer("") 
                write_to_buffer(summary_line)
                write_to_buffer("")
    log.debug(f"--- Finished Standard Folder Walk for {target_path} ---")


# --- Raw Dump Functions (largely unchanged, included for completeness) ---
def process_folder_raw(
    target_input_path_str: str,
    buffer: io.StringIO,
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
        elif additional_path_obj.is_file(): 
            paths_to_process.append((additional_path_obj, "Additional Context File"))
        else:
            log.warning(f"Raw Dump: Additional context path '{additional_path_obj}' is not a valid file or directory, skipping.")

    total_files_dumped_across_all_targets = 0

    for current_target_path_obj, current_target_desc in paths_to_process:
        log.debug(f"Starting RAW dump for {current_target_desc}: {current_target_path_obj}")
        file_count_for_this_target = 0

        if current_target_path_obj.is_file():
            buffer.write(f"\n{'=' * 20} START FILE ({current_target_desc}): {current_target_path_obj.name} {'=' * 20}\n")
            try:
                with open(current_target_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if content and not content.endswith('\n'): 
                    content += '\n'
                buffer.write(content)
                file_count_for_this_target = 1
            except Exception as e:
                log.error(f"Error reading/writing file {current_target_path_obj.name} for raw dump ({current_target_desc}): {e}")
                buffer.write(f"### ERROR READING FILE: {e} ###\n")
            buffer.write(f"{'=' * 20} END FILE ({current_target_desc}): {current_target_path_obj.name} {'=' * 20}\n")
        
        elif current_target_path_obj.is_dir():
            buffer.write(f"\n{'=' * 10} STARTING RAW DUMP OF {current_target_desc.upper()}: {current_target_path_obj.name} {'=' * 10}\n")
            walk_results = sorted(list(os.walk(current_target_path_obj, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} during raw dump - {e}"))), key=lambda x: x[0])
            
            for dirpath, dirnames, filenames in walk_results:
                dirnames.sort() 
                filenames.sort() 
                
                current_path_in_walk = Path(dirpath)
                for filename_in_walk in filenames:
                    file_path_in_walk = current_path_in_walk / filename_in_walk
                    relative_file_path_display = file_path_in_walk.relative_to(current_target_path_obj)
                    if not file_path_in_walk.is_file(): continue
                    file_count_for_this_target +=1 # Corrected variable name
                    buffer.write(f"\n{'=' * 20} START FILE ({current_target_desc}/{relative_file_path_display}): {filename_in_walk} {'=' * 20}\n")
                    log.info(f"Dumping file ({file_count_for_this_target} for {current_target_desc}): {relative_file_path_display}")
                    try:
                        with open(file_path_in_walk, 'r', encoding='utf-8', errors='ignore') as f:
                            content_f = f.read()
                        if content_f and not content_f.endswith('\n'): 
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
    target_folder_str: str, # Renamed for clarity
    buffer: io.StringIO,
    ignore_set: set,
    code_extensions_set: set,
    interesting_filenames_set: set, 
    report_data: Dict[str, Counter] 
) -> int: 
    """Dumps only recognized 'source' files verbatim, respecting ignore patterns."""
    target_path = Path(target_folder_str).resolve()
    log.debug(f"--- Starting Raw Source Folder Walk ---")
    log.debug(f"Root: {target_path}")
    log.debug(f"Effective Ignore Set: {sorted(list(ignore_set))}") # Added
    
    source_file_count = 0
    
    walk_results = sorted(list(os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Access error {e.filename}: {e}"))), key=lambda x: x[0])

    for dirpath_str, dirnames_orig_iter, filenames_orig_iter in walk_results:
        current_walk_path = Path(dirpath_str)
        dirnames_orig = sorted(list(dirnames_orig_iter))
        filenames_orig = sorted(list(filenames_orig_iter))
        
        try:
            current_rel_path_from_target = current_walk_path.relative_to(target_path)
        except ValueError: # pragma: no cover
            log.warning(f"Path Error: Cannot get relative path for '{current_walk_path}' "
                        f"relative to walk root '{target_path}'. Skipping this path.")
            dirnames_orig_iter[:] = [] 
            continue
        
        log.debug(f"\nRAW_SOURCE_DIR_WALK: Processing directory '{current_rel_path_from_target}' (Abs: '{current_walk_path}')")
        log.debug(f"RAW_SOURCE_DIR_WALK: Original subdirectories before filtering: {dirnames_orig}")

        dirs_to_remove_from_walk = []
        for d_name in dirnames_orig:
            potential_subdir_rel_to_target = current_rel_path_from_target / d_name
            if d_name in ignore_set:
                log.debug(f"  RAW_SOURCE_DIR_IGNORE (Exact Name): '{d_name}' in '{current_rel_path_from_target}'. Marking for removal from walk.")
                dirs_to_remove_from_walk.append(d_name)
                continue
            if d_name.startswith('.') and d_name not in interesting_filenames_set and d_name not in code_extensions_set:
                log.debug(f"  RAW_SOURCE_DIR_IGNORE (Hidden): '{d_name}' in '{current_rel_path_from_target}'. Marking for removal from walk.")
                dirs_to_remove_from_walk.append(d_name)
                continue
            is_glob_ignored = False
            for p_glob in ignore_set:
                if not ('*' in p_glob or '?' in p_glob or '[' in p_glob): continue
                if potential_subdir_rel_to_target.match(p_glob):
                    log.debug(f"  RAW_SOURCE_DIR_IGNORE (Glob on RelPath): '{potential_subdir_rel_to_target}' matched by glob '{p_glob}'. Marking for removal from walk.")
                    is_glob_ignored = True; break
            if is_glob_ignored:
                dirs_to_remove_from_walk.append(d_name)
                continue
            log.debug(f"  RAW_SOURCE_DIR_KEEP: '{d_name}' in '{current_rel_path_from_target}'. Will be explored by os.walk if not removed by name.")
        
        for d_to_remove in dirs_to_remove_from_walk:
            if d_to_remove in dirnames_orig_iter:
                 dirnames_orig_iter.remove(d_to_remove)
        
        log.debug(f"RAW_SOURCE_DIR_WALK: Subdirectories os.walk will explore next in '{current_rel_path_from_target}': {sorted(list(dirnames_orig_iter))}")
        
        log.debug(f"RAW_SOURCE_DIR_WALK: Considering files in '{current_rel_path_from_target}': {filenames_orig}")
        for filename in filenames_orig:
            file_path_abs = current_walk_path / filename
            file_rel_to_target = file_path_abs.relative_to(target_path)
            log.debug(f"  RAW_SOURCE_FILE_CONSIDER: '{filename}' (Full Rel: '{file_rel_to_target}')")

            if filename in ignore_set:
                log.debug(f"    RAW_SOURCE_FILE_IGNORE (Exact Name): '{filename}'. SKIPPING.")
                continue
            if filename.startswith('.') and filename.lower() not in interesting_filenames_set and filename not in code_extensions_set and Path(filename).suffix.lower() not in code_extensions_set:
                log.debug(f"    RAW_SOURCE_FILE_IGNORE (Hidden): '{filename}'. SKIPPING.")
                continue
            is_filename_glob_ignored = False
            for p_glob in ignore_set:
                if '*' not in p_glob and '?' not in p_glob and '[' not in p_glob: continue
                if Path(filename).match(p_glob):
                    log.debug(f"    RAW_SOURCE_FILE_IGNORE (Glob on Filename): '{filename}' matched by glob '{p_glob}'. SKIPPING.")
                    is_filename_glob_ignored = True; break
            if is_filename_glob_ignored: continue
            is_parent_path_glob_ignored = False
            for p_glob_for_parent_check in ignore_set:
                if not ('*' in p_glob_for_parent_check or '?' in p_glob_for_parent_check or '[' in p_glob_for_parent_check): continue
                for i_parent in range(len(current_rel_path_from_target.parts)): # Check current_rel_path_from_target
                    path_component_to_check = Path(*current_rel_path_from_target.parts[:i_parent+1])
                    if path_component_to_check.match(p_glob_for_parent_check):
                        log.debug(f"    RAW_SOURCE_FILE_IGNORE (Parent Path Glob): Parent '{path_component_to_check}' of file in dir '{current_rel_path_from_target}' "
                                  f"matched glob '{p_glob_for_parent_check}'. SKIPPING file '{filename}'.")
                        is_parent_path_glob_ignored = True; break
                if is_parent_path_glob_ignored: break
            if is_parent_path_glob_ignored: continue
            if not file_path_abs.is_file():
                log.debug(f"    RAW_SOURCE_FILE_SKIP (Not a File): '{filename}'.")
                continue
            if is_likely_binary(file_path_abs):
                log.debug(f"    RAW_SOURCE_FILE_SKIP (Binary): '{filename}'.")
                continue
                
            base, ext = os.path.splitext(filename)
            ext_lower = ext.lower(); fname_lower = filename.lower()
            is_source = (ext_lower in code_extensions_set or filename in code_extensions_set or fname_lower in code_extensions_set)
            file_id = get_file_id_from_path_str(filename) 

            if is_source:
                log.debug(f"    RAW_SOURCE_FILE_DUMP: '{filename}' is source. Dumping.")
                if file_id not in report_data: report_data[file_id] = Counter()
                report_data[file_id]['dumped_raw_source'] += 1
                source_file_count += 1
                
                buffer.write(f"\n{'=' * 20} START SOURCE FILE (RAW): {file_rel_to_target} {'=' * 20}\n")
                log.info(f"Dumping source file (RAW) ({source_file_count}): {file_rel_to_target}")
                try:
                    with open(file_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if content and not content.endswith('\n'): 
                        content += '\n'
                    buffer.write(content)
                except OSError as e:
                    log.error(f"Read error (RAW SOURCE): {file_rel_to_target}: {e}")
                    report_data[file_id]['read_error_raw_source'] += 1
                    buffer.write(f"### ERROR READING FILE (RAW SOURCE): {e} ###\n")
                except Exception as e: # pragma: no cover
                    log.error(f"Processing error (RAW SOURCE): {file_rel_to_target}: {e}", exc_info=True)
                    report_data[file_id]['processing_error_raw_source'] += 1
                    buffer.write(f"### ERROR PROCESSING FILE (RAW SOURCE): {e} ###\n")
                buffer.write(f"{'=' * 20} END SOURCE FILE (RAW): {file_rel_to_target} {'=' * 20}\n")
            else:
                 log.debug(f"    RAW_SOURCE_FILE_SKIP (Not Source): '{filename}'.")

    log.info(f"Finished raw source dump. Processed and dumped {source_file_count} source files.")
    return source_file_count

# --- Post-Processing Functions (largely unchanged) ---

def apply_post_simplification_patterns(content: str, patterns: list[tuple[re.Pattern, Any]]) -> tuple[str, int]:
    """Applies a list of regex patterns for further detailed simplification."""
    total_replacements = 0
    lines = content.splitlines(keepends=True) 
    output_lines = []
    log.debug(f"Applying {len(patterns)} post-simplification patterns...")
    pattern_counts = Counter()

    for line_idx, line in enumerate(lines):
        modified_line = line
        for i, (pattern, replacement) in enumerate(patterns):
            try:
                 original_line_segment = modified_line 
                 if callable(replacement):
                     modified_line_new, count = pattern.subn(replacement, modified_line)
                 else:
                     modified_line_new, count = pattern.subn(replacement, modified_line)
                 
                 if count > 0:
                     total_replacements += count
                     pattern_counts[i] += count
                     if log.getEffectiveLevel() <= logging.DEBUG:
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
    lines = content.splitlines(keepends=False) 
    output_lines = []
    lines_expanded = 0
    log.debug(f"--- Starting expand_multi_pattern_lines: Scanning for multiple '{pattern_name_for_log}' instances per line ---")

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line or \
           stripped_line.startswith(("--- File:", "--- End File:", "===")) or \
           re.match(r"^\s*# \.\.\. \d+ similar lines compressed \.\.\.", stripped_line) or \
           re.match(r"^\s*\/\/\s*Sample representation", stripped_line) :
            output_lines.append(line + "\n")
            continue
        
        try:
            matches = finder_pattern.findall(line)
        except Exception as e: # pragma: no cover 
            log.error(f"Regex error during expand_multi_pattern_lines on line {i+1}: {e}")
            log.debug(f"Problematic line: {line}")
            matches = [] 

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

        is_ignorable_starter = (
            stripped_line.startswith(("--- File:", "--- End File:", "===")) or
            stripped_line.startswith(("#", "//", "/*", "*LINE_REF_")) or 
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
            block_lines_indices = [i] 
            
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                next_is_ignorable_continuation = (
                    next_stripped.startswith(("--- File:", "--- End File:", "===")) or
                    next_stripped.startswith(("#", "//", "/*")) 
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
                if len(first_line_in_block) > len(first_line_stripped): 
                    indent = first_line_in_block[:len(first_line_in_block) - len(first_line_stripped)]
                
                summary_line = f"{indent}# ... {block_count} lines matching '{block_pattern_name}' pattern compressed ...\n"
                output_lines.append(summary_line)
                log.info(f"Compressed {block_count} lines (Indices {block_start_index+1}-{j}) matching '{block_pattern_name}'.")
                total_blocks_compressed += 1
                i = j 
            else:
                log.debug(f"  Block '{block_pattern_name}' starting at line {block_start_index+1} had only {block_count} lines (min={min_consecutive}). Not compressing.")
                for block_line_index in block_lines_indices:
                    output_lines.append(lines[block_line_index])
                i = j 
        else:
            output_lines.append(current_line)
            i += 1
            
    log.info(f"Pattern block compression: Compressed {total_blocks_compressed} blocks (min consecutive: {min_consecutive}).")
    return "".join(output_lines), total_blocks_compressed


def minify_repeated_lines(content: str, min_length: int, min_repetitions: int) -> tuple[str, int, Dict[str, Counter]]:
    """Identifies and replaces frequently repeated long lines with placeholders."""
    global DEFINITION_SIMPLIFICATION_PATTERNS
    
    lines = content.splitlines(keepends=True)
    line_counts = Counter()
    meaningful_lines_indices: Dict[str, List[int]] = defaultdict(list)

    log.debug(f"--- Starting minify_repeated_lines: Scanning lines >= {min_length} chars, repeated >= {min_repetitions}x ---")

    for i, line_val in enumerate(lines):
        stripped_line = line_val.strip()
        is_structural_or_placeholder = (
            stripped_line.startswith(("--- File:", "--- End File:", "===")) or
            stripped_line.startswith(("*LINE_REF_", "# ...", "// ...")) or
            len(stripped_line) < min_length or
            re.match(r'^\s*(\*([A-Z0-9_]+)\*[,;\s]*)+\s*$', stripped_line) or
            not stripped_line
        )
        
        if not is_structural_or_placeholder:
            line_content_key = stripped_line.rstrip() # Normalize trailing whitespace for keying
            line_counts[line_content_key] += 1
            meaningful_lines_indices[line_content_key].append(i)

    replacement_map: Dict[str, str] = {}
    minified_line_origins: Dict[str, Counter] = {} # file_type -> Counter({'placeholder_instances_created': count})
    definition_lines: List[str] = []
    placeholder_counter = 1
    placeholder_template = "*LINE_REF_{}*"

    repeated_lines_candidates = sorted(
        [(line_content, count) for line_content, count in line_counts.items() if count >= min_repetitions],
        key=lambda item: (-item[1], item[0])
    )
    
    if not repeated_lines_candidates:
        log.info("Line minification: No lines met the repetition and length criteria.")
        return content, 0, {}

    # Create context finder AFTER initial lines are established, BEFORE modification
    find_file_context = create_file_context_finder(lines) 
    
    for original_line_key, count in repeated_lines_candidates:
        placeholder_str = placeholder_template.format(placeholder_counter)
        placeholder_counter += 1
        replacement_map[original_line_key] = placeholder_str

        simplified_def_line = original_line_key
        for def_pattern, def_repl in DEFINITION_SIMPLIFICATION_PATTERNS:
            try:
                if callable(def_repl):
                    simplified_def_line = def_pattern.sub(lambda mobj: def_repl(mobj) or "", simplified_def_line)
                else:
                    simplified_def_line = def_pattern.sub(def_repl, simplified_def_line)
            except Exception as e: # pragma: no cover
                log.warning(f"Error simplifying definition line '{original_line_key[:50]}...' with pattern '{def_pattern.pattern}': {e}")
        definition_lines.append(f"{placeholder_str} = {simplified_def_line.strip()}")

        # Attribute origins based on all occurrences that will be replaced
        if original_line_key in meaningful_lines_indices:
            for idx in meaningful_lines_indices[original_line_key]:
                file_id = find_file_context(idx)
                if file_id not in minified_line_origins:
                    minified_line_origins[file_id] = Counter()
                minified_line_origins[file_id]['placeholder_instances_created'] += 1
    
    new_lines = list(lines) # Make a mutable copy
    num_actual_replacements_done = 0
    if replacement_map:
        for i, line_val in enumerate(lines): # Iterate original lines to find what to replace
            stripped_content_for_match = line_val.strip().rstrip() # Match key format
            if stripped_content_for_match in replacement_map:
                placeholder = replacement_map[stripped_content_for_match]
                # Preserve original line's ending (newline or not)
                trailing_newline = "\n" if line_val.endswith("\n") else ("\r\n" if line_val.endswith("\r\n") else "")
                
                # Construct the replacement line, ensuring it maintains the original line's indent
                original_indent_len = len(line_val) - len(line_val.lstrip())
                indent_str = line_val[:original_indent_len]
                new_lines[i] = indent_str + placeholder + trailing_newline
                num_actual_replacements_done += 1
    
    minified_content = "".join(new_lines)
    
    if definition_lines:
        definition_header = [
            "", "=" * 40,
            f"# Line Minification Definitions ({len(definition_lines)}):",
            "=" * 40
        ]
        definition_block_str = "\n".join(definition_header + definition_lines) + "\n\n"
        log.info(f"Line minification: Replaced {num_actual_replacements_done} occurrences of {len(definition_lines)} unique lines.")
        return definition_block_str + minified_content, num_actual_replacements_done, minified_line_origins
    else:
        return content, 0, {}


def post_process_cleanup(content: str, cleanup_pattern: re.Pattern) -> tuple[str, int, Dict[str, Counter]]:
    """Removes lines that consist *only* of placeholders, commas, brackets, etc., after all other processing."""
    lines = content.splitlines(keepends=True)
    output_lines = []
    lines_removed = 0
    cleanup_origins_report: Dict[str, Counter] = defaultdict(Counter)

    find_file_context_for_cleanup = create_file_context_finder(lines)
    
    log.debug(f"--- Starting post_process_cleanup: Scanning for lines matching cleanup_pattern ---")
    log.debug(f"Cleanup pattern: {cleanup_pattern.pattern}")

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        if line.startswith(("*LINE_REF_", "# Line Minification Definitions")) or \
           line.startswith("===") or \
           line.startswith("--- File:") or \
           line.startswith("--- End File:") or \
           re.match(r"^\s*# Other files in", stripped_line) or \
           re.match(r"^\s*# \.\.\. \d+ lines matching", stripped_line) or \
           re.match(r"^\s*# \.\.\. \d+ similar lines compressed", stripped_line) or \
           re.match(r"^\s*\/\/\s*Sample representation", stripped_line) :
            output_lines.append(line)
            continue

        if cleanup_pattern.match(stripped_line): 
            log.debug(f"Post-cleanup removing line {i+1}: {stripped_line[:80]}...")
            lines_removed += 1
            file_id = find_file_context_for_cleanup(i)
            cleanup_origins_report[file_id]['removed_by_cleanup'] += 1
        else:
            output_lines.append(line) # Keep the line if it doesn't match cleanup

    cleaned_content = "".join(output_lines)
    log.info(f"Post-processing cleanup: Removed {lines_removed} lines.")
    return cleaned_content, lines_removed, dict(cleanup_origins_report) # Convert back to dict for consistency

# Placeholder for generate_output_file_summary
def generate_output_file_summary(report_data: Dict[str, Counter], mode_display: str) -> str:
    """Generates a summary block about processed files for the output."""
    log.debug("Generating output file summary (stub implementation).")
    if not report_data:
        return "# No file processing data to summarize.\n\n"

    summary_lines = [
        "=" * 40,
        f"# File Processing Summary ({mode_display} Mode)",
        "=" * 40,
    ]
    
    grand_totals = Counter()
    sorted_file_ids = sorted(report_data.keys())

    for file_id in sorted_file_ids:
        stats = report_data[file_id]
        if not stats: continue # Skip if no stats for this ID

        summary_lines.append(f"# Type: {file_id if file_id else '<no_ext>'}")
        for key, val in sorted(stats.items()):
            if val > 0:
                summary_lines.append(f"#   - {key.replace('_', ' ').capitalize()}: {val}")
                grand_totals[key] += val
    
    if grand_totals:
        summary_lines.append("# " + "-" * 38)
        summary_lines.append("# Overall Totals:")
        for key, val in sorted(grand_totals.items()):
            if val > 0:
                summary_lines.append(f"#   - {key.replace('_', ' ').capitalize()}: {val}")

    summary_lines.append("=" * 40)
    summary_lines.append("") # Trailing newline
    return "\n".join(summary_lines) + "\n"


# Placeholder for generate_content_report
def generate_content_report(final_content: str) -> Dict[str, Any]:
    """
    Analyzes the character content of the final output.
    This is a more complex function that categorizes lines based on various patterns.
    """
    log.debug("Generating content report (stub implementation).")
    total_chars = len(final_content)
    char_counts: Dict[str, Any] = { # Using Any for the Code/Text breakdown by extension
        'Structural Elements': 0,
        'Placeholders & Summaries': 0,
        'Comments': 0,
        'Code/Text': Counter(), # Stores counts per extension (e.g., '.py': 1000)
        'Whitespace/Empty Lines': 0,
    }
    
    current_file_ext = "<unknown_file>" # Track current file type for Code/Text attribution

    for line_num, original_line_text_for_match in enumerate(final_content.splitlines()):
        current_segment_len = len(original_line_text_for_match) + 1 # +1 for newline
        
        if not original_line_text_for_match.strip():
            char_counts['Whitespace/Empty Lines'] += current_segment_len
            continue

        category_assigned = None
        
        is_structural_and_file_marker = False
        for pattern_idx, pattern in enumerate(REPORT_STRUCTURAL_PATTERNS):
            if pattern.match(original_line_text_for_match):
                category_assigned = 'Structural Elements'
                
                std_match = STD_FILE_MARKER_EXTRACT_RE.match(original_line_text_for_match)
                if std_match:
                    is_structural_and_file_marker = True
                    if "SKIPPED (BINARY)" in original_line_text_for_match:
                        current_file_ext = "<binary>"
                    else:
                        filename_in_marker = std_match.group(1)
                        current_file_ext = get_file_id_from_path_str(filename_in_marker)
                    break 

                raw_match = RAW_FILE_MARKER_EXTRACT_RE.match(original_line_text_for_match)
                if raw_match:
                    is_structural_and_file_marker = True
                    filename_in_marker = raw_match.group(2).strip() # group(2) is filename
                    current_file_ext = get_file_id_from_path_str(filename_in_marker)
                    break 
                if category_assigned: break
        
        if category_assigned:
            char_counts[category_assigned] += current_segment_len
            continue

        for pattern in REPORT_PLACEHOLDER_SUMMARY_PATTERNS:
            if pattern.match(original_line_text_for_match):
                category_assigned = 'Placeholders & Summaries'; break
        if category_assigned:
            char_counts[category_assigned] += current_segment_len
            continue
        
        if PLACEHOLDER_CLEANUP_PATTERN.match(original_line_text_for_match):
            category_assigned = 'Placeholders & Summaries'
            char_counts[category_assigned] += current_segment_len
            continue

        for pattern in REPORT_COMMENT_PATTERNS:
            if pattern.match(original_line_text_for_match):
                category_assigned = 'Comments'; break
        if category_assigned:
            char_counts[category_assigned] += current_segment_len
            continue
            
        category_assigned = 'Code/Text' # Default
        char_counts['Code/Text'][current_file_ext] += current_segment_len

    report_breakdown = {}
    total_code_text_chars = sum(char_counts['Code/Text'].values())

    for category_name, count_value in char_counts.items():
        if category_name == 'Code/Text': continue

        percentage = (count_value / total_chars) * 100 if total_chars > 0 else 0
        report_breakdown[category_name] = {'chars': count_value, 'percentage': percentage}

    if total_code_text_chars > 0 or 'Code/Text' in char_counts : # ensure entry even if 0
        percentage_total_code_text = (total_code_text_chars / total_chars) * 100 if total_chars > 0 else 0
        code_text_entry: Dict[str, Any] = {
            'chars': total_code_text_chars,
            'percentage': percentage_total_code_text,
            'sub_categories': {}
        }
        # Sort extensions by character count descending
        sorted_ext_breakdown = sorted(char_counts['Code/Text'].items(), key=lambda item: item[1], reverse=True)
        for ext, count in sorted_ext_breakdown:
            percentage_of_total = (count / total_chars) * 100 if total_chars > 0 else 0
            percentage_of_code_text = (count / total_code_text_chars) * 100 if total_code_text_chars > 0 else 0
            code_text_entry['sub_categories'][ext] = {
                'chars': count,
                'percentage_of_total': percentage_of_total,
                'percentage_of_code_text': percentage_of_code_text
            }
        report_breakdown['Code/Text (Total)'] = code_text_entry
    
    return {"Total Characters": total_chars, "Breakdown": report_breakdown}


# --- Main Function ---
def main():
    global DEFAULT_IGNORE_PATTERNS, BASE_IGNORE_PATTERNS, TEST_IGNORE_PATTERNS
    global CODE_EXTENSIONS, INTERESTING_FILENAMES, DEFAULT_LARGE_LITERAL_THRESHOLD
    global seen_content_hashes 

    DEFAULT_MIN_LINE_LENGTH_REC = 50 
    DEFAULT_MIN_REPETITIONS_REC = 3  
    DEFAULT_MIN_CONSECUTIVE_REC = 3 

    parser = argparse.ArgumentParser(
        description="Generate a concise text representation of a folder's content, suitable for LLM context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode", default="standard", choices=["standard", "raw_dump", "raw_source_dump"],
        help="Processing mode: 'standard' (default, simplify and compress), "
             "'raw_dump' (dump all files verbatim, ignores most other settings), "
             "'raw_source_dump' (dump only recognized source files verbatim, respects ignores)."
    )

    parser.add_argument("folder_path", help="Path to the target folder or file.") # Changed from folder_path
    parser.add_argument(
        "-o", "--output",
        help="Output file path. If not provided, output goes to stdout."
    )
    parser.add_argument(
        "--additional-context-path", default=None, # Changed from --additional-context-folder
        help="(Raw Dump Mode Only) Path to an additional folder or file to also dump verbatim. "
             "Content will be appended after the primary folder_path content."
    )

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

    proc_group = parser.add_argument_group('Standard Mode Processing Options (ignored in raw modes)')
    proc_group.add_argument(
        "--strip-logging", action="store_true", default=False,
        help="Attempt to remove common logging/print statements (e.g., log.info(...), print(...)). Use with caution."
    )
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
             "This is active if Pattern Block Compression (--no-compress-patterns) is disabled."
    )

    comp_group = parser.add_argument_group('Standard Mode Compression Steps (On by default, use --no-X to disable)')
    comp_group.add_argument(
        "--no-preprocess-split-lines", action="store_false", dest="preprocess_split_lines", default=True,
        help="Disable pre-processing that splits lines with multiple identifiable patterns (e.g., Voice IDs) "
             "onto separate lines. Splitting improves block compression."
    )
    comp_group.add_argument(
        "--no-compress-patterns", action="store_false", dest="compress_patterns", default=True,
        help="Disable compression of consecutive lines matching predefined patterns (e.g., lists of Voice IDs, UUIDs) "
             "into summary lines like '# ... N lines matching PATTERN ...'. Disabling this enables large-literal-threshold based compression."
    )
    comp_group.add_argument(
        "--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE_REC,
        help="Min number of consecutive lines matching a pattern to trigger block compression (if --compress-patterns is enabled)."
    )
    comp_group.add_argument(
        "--apply-patterns", action="store_true", default=False, 
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
    
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity. DEBUG provides detailed step-by-step info."
    )

    args = parser.parse_args()

    # Set log level for THIS script's logger AND the root logger (to affect other libraries if they use root)
    # This must be done AFTER parsing args.
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    log.setLevel(numeric_level)
    logging.getLogger().setLevel(numeric_level) # Set root logger level
    
    # --- Initialize Globals ---
    seen_content_hashes.clear()
    total_bytes_written = 0
    process_report_data: Dict[str, Counter] = defaultdict(Counter) 
    minification_origin_report: Dict[str, Counter] = {} 
    cleanup_origin_report: Dict[str, Counter] = {} 
    
    primary_input_path_obj = Path(args.folder_path) # Corrected from args.input_path
    if not primary_input_path_obj.exists():
        log.critical(f"Error: Primary input path not found: '{args.folder_path}'"); sys.exit(1)
    if not (primary_input_path_obj.is_dir() or primary_input_path_obj.is_file()):
        log.critical(f"Error: Primary input path '{args.folder_path}' is neither a directory nor a file."); sys.exit(1)
    
    resolved_primary_path = primary_input_path_obj.resolve()
    is_primary_input_directory = resolved_primary_path.is_dir()
    resolved_path_str = str(resolved_primary_path) # For use where string path is needed

    resolved_additional_context_path_obj = None
    if args.additional_context_path: # Corrected from args.additional_context_folder
        additional_context_path_obj_temp = Path(args.additional_context_path)
        if not (additional_context_path_obj_temp.is_dir() or additional_context_path_obj_temp.is_file()):
            log.warning(f"Warning: Additional context path '{args.additional_context_path}' is not a valid directory or file. It will be ignored.")
        else:
            resolved_additional_context_path_obj = additional_context_path_obj_temp.resolve()
            if resolved_additional_context_path_obj == resolved_primary_path:
                log.warning(f"Warning: Additional context path is the same as the primary input path. It will be processed once as primary."); resolved_additional_context_path_obj = None

    # --- Determine effective settings and print config summary ---
    run_mode_display = args.mode.replace("_", " ").title()

    current_ignore_patterns = BASE_IGNORE_PATTERNS.copy()
    effective_test_inclusion = args.include_tests
    if args.mode == "raw_dump":
        effective_test_inclusion = True 
        log.info("Raw Dump mode: Ignoring --include-tests logic, will attempt to dump all files.")
    elif not args.include_tests:
        log.info("Excluding test files/directories by default. Use --include-tests to override.")
        current_ignore_patterns.update(TEST_IGNORE_PATTERNS)
    else: 
        log.info("Including test files/directories as per --include-tests.")
    
    current_ignore_patterns.update(args.ignore)

    current_code_extensions = CODE_EXTENSIONS.copy()
    current_code_extensions.update(args.source_ext)
    current_interesting_files = INTERESTING_FILENAMES.copy()
    current_interesting_files.update(args.interesting_files)

    print("-" * 40, file=sys.stderr)
    print(f"Starting Folder-to-Text Processing", file=sys.stderr)
    print(f"Target Path: {resolved_path_str}", file=sys.stderr)
    print(f"Output Target: {'Stdout' if not args.output else args.output}", file=sys.stderr)
    print(f"Mode: {run_mode_display}", file=sys.stderr)
    print(f"Log Level: {args.log_level.upper()}", file=sys.stderr)
    if args.mode != "raw_dump": # Print these for standard and raw_source_dump
        print(f"  Include Tests: {effective_test_inclusion}", file=sys.stderr)
        print(f"  Effective Ignore Patterns: {sorted(list(current_ignore_patterns))}", file=sys.stderr) # Log the full set
        print(f"  Effective Source Extensions: {sorted(list(current_code_extensions))}", file=sys.stderr)
        print(f"  Effective Interesting Files: {sorted(list(current_interesting_files))}", file=sys.stderr)
    else: 
        print("  (File selection options like --ignore, --include-tests are mostly bypassed in Raw Dump mode)", file=sys.stderr)
    if resolved_additional_context_path_obj:
        print(f"  Additional Context Path: {resolved_additional_context_path_obj}", file=sys.stderr)

    # CRITICAL CHECK FOR ROOT FOLDER IGNORING - Moved here after current_ignore_patterns is finalized
    if is_primary_input_directory and resolved_primary_path.name in current_ignore_patterns:
        log.critical(
            f"CRITICAL WARNING: The root processing directory '{resolved_primary_path.name}' "
            f"is itself in the ignore patterns. This usually means the script will output "
            "nothing from the primary path because the ignore logic in os.walk "
            "will prune it immediately. "
            "If you intend to process this folder, remove '{resolved_primary_path.name}' from "
            "your ignore patterns or use a more specific glob pattern for subdirectories "
            " (e.g., '*/discord' instead of just 'discord' if 'discord' is your root)."
        )

    if args.mode == "standard":
        effective_skip_empty = not args.keep_empty
        effective_skip_duplicates = not args.keep_duplicates
        print(f"  Skip Empty Files (after simplify): {effective_skip_empty}", file=sys.stderr)
        print(f"  Skip Duplicate Files (after simplify): {effective_skip_duplicates}", file=sys.stderr)
        print(f"  Strip Logging: {args.strip_logging}", file=sys.stderr)
        
        literal_comp_active = not args.compress_patterns 
        if literal_comp_active:
            print(f"  Large Literal Compression: ACTIVE (threshold: {args.large_literal_threshold} lines)", file=sys.stderr)
        else:
            print(f"  Large Literal Compression: DISABLED (Pattern Block Compression is active)", file=sys.stderr)
        
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
    output_handle: Union[io.TextIOWrapper, io.StringIO] = sys.stdout # Default
    output_path_obj: Optional[Path] = None
    content_buffer = io.StringIO() 
    
    files_processed_count = 0 
    num_lines_expanded, num_pattern_replacements = 0, 0
    num_blocks_compressed, num_lines_minified, num_lines_cleaned_up = 0, 0, 0
    main_content_str = "" 

    try:
        # --- Generate Initial Header for the output file ---
        header_desc = f"Primary {'folder' if is_primary_input_directory else 'file'}: {resolved_primary_path}"
        if resolved_additional_context_path_obj: header_desc += f" | Additional Context: {resolved_additional_context_path_obj}"
        
        header_info_lines = [
            f"# {'RAW DUMP' if args.mode == 'raw_dump' else ('RAW SOURCE DUMP' if args.mode == 'raw_source_dump' else 'Compressed Representation')} of {header_desc}",
            f"# Generated by folder_to_text.py (Mode: {run_mode_display})"
        ]
        
        options_list = []
        if args.mode == "standard":
            options_list.extend([
                f"strip_log={args.strip_logging}", f"apply_pat={args.apply_patterns}",
                f"minify={args.minify_lines}({args.min_line_length}c,{args.min_repetitions}x)",
                f"large_lit_thresh={args.large_literal_threshold}(active if pat_comp_disabled)",
                f"split_lines={args.preprocess_split_lines}", f"comp_pat={args.compress_patterns}({args.min_consecutive}l)",
                f"cleanup={args.post_cleanup}"
            ])
            if effective_skip_empty: options_list.append("skip_empty=True")
            if effective_skip_duplicates: options_list.append("skip_duplicates=True")
        elif args.mode == "raw_source_dump":
            options_list.append(f"include_tests={effective_test_inclusion}")

        if options_list:
            header_info_lines.append(f"# Options: {', '.join(options_list)}")
        
        initial_header_str = "\n".join(header_info_lines) + "\n" + "=" * 40 + "\n\n"
        
        # --- Main Processing Logic based on mode ---
        if args.mode == "raw_dump":
            log.info(f"Starting raw dump for '{resolved_path_str}'...")
            additional_path_str_for_raw = str(resolved_additional_context_path_obj) if resolved_additional_context_path_obj else None
            files_processed_count = process_folder_raw(resolved_path_str, content_buffer, additional_path_str_for_raw)
            log.info(f"Finished raw dump. Dumped {files_processed_count} file(s).")
            content_buffer.seek(0)
            main_content_str = content_buffer.getvalue()

        elif args.mode == "raw_source_dump":
            log.info(f"Starting raw source folder dump for '{resolved_path_str}'...")
            if not is_primary_input_directory:
                log.error("Raw source dump mode requires a directory as primary input.")
                sys.exit(1)
            files_processed_count = process_folder_raw_source(
                resolved_path_str, content_buffer,
                current_ignore_patterns, current_code_extensions, current_interesting_files,
                process_report_data
            )
            log.info(f"Finished raw source folder dump. Dumped {files_processed_count} source files.")
            content_buffer.seek(0)
            main_content_str = content_buffer.getvalue()

        else: # Standard mode
            log.info("Starting standard processing (Step 1: File Traversal & Initial Simplification)...")
            effective_skip_empty = not args.keep_empty
            effective_skip_duplicates = not args.keep_duplicates
            disable_large_literal_comp_if_pattern_comp_active = args.compress_patterns
            
            if is_primary_input_directory:
                process_folder(
                    resolved_path_str, content_buffer,
                    current_ignore_patterns, current_code_extensions, current_interesting_files,
                    effective_skip_empty, args.strip_logging, effective_skip_duplicates,
                    args.large_literal_threshold, disable_large_literal_comp_if_pattern_comp_active,
                    process_report_data
                )
            else: # Primary input is a single file
                process_single_file_standard(
                    resolved_path_str, content_buffer,
                    args.strip_logging, args.large_literal_threshold,
                    disable_large_literal_comp_if_pattern_comp_active,
                    process_report_data
                )
                files_processed_count = 1 # Processed one file

            log.info("Finished initial folder/file processing and simplification.")
            content_buffer.seek(0)
            current_processed_content = content_buffer.getvalue()
            
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
                log.info(f"Finished Step 5 (Line Minification). Minified {num_lines_minified} line instances into {len(definition_lines) if 'definition_lines' in locals() else 0} definitions.") # definition_lines is local to minify_repeated_lines
            
            if args.post_cleanup and current_processed_content.strip():
                log.info("Step 6: Applying post-processing cleanup...")
                current_processed_content, num_lines_cleaned_up, cleanup_origin_report = post_process_cleanup(current_processed_content, PLACEHOLDER_CLEANUP_PATTERN)
                log.info(f"Finished Step 6 (Post-Cleanup). Removed {num_lines_cleaned_up} lines.")
            
            main_content_str = current_processed_content
        
        file_summary_block_str = ""
        if args.mode != "raw_dump": 
            file_summary_block_str = generate_output_file_summary(process_report_data, run_mode_display)
        
        final_output_to_write = initial_header_str + file_summary_block_str + main_content_str

        log.info("Preparing to write final output...")
        if args.output:
            output_path_obj = Path(args.output).resolve()
            output_path_obj.parent.mkdir(parents=True, exist_ok=True) 
            output_handle = open(output_path_obj, 'w', encoding='utf-8')
            log.debug(f"Opening output file for writing: {output_path_obj}")
        else: 
            log.debug("Using stdout for output.")
            
        output_handle.write(final_output_to_write)
        total_bytes_written = len(final_output_to_write.encode('utf-8')) 
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
            
            # File Type Processing Report (from process_report_data) is already part of generate_output_file_summary
            # So it will be in the output file. If you want it on stderr too, you can print it here.

        elif args.mode == "raw_source_dump":
            print(f"  Dumped content of {files_processed_count} source files.", file=sys.stderr)
            if process_report_data: # process_report_data now comes from generate_output_file_summary format
                # This will be a more detailed summary now
                pass # Summary is in the output file
        else: # Raw Dump (all files)
            print(f"  Dumped content of {files_processed_count} files.", file=sys.stderr)
        
        print("-" * 40, file=sys.stderr)
        print("Content Analysis Report (on final output):", file=sys.stderr)
        report_stats = generate_content_report(final_output_to_write) # Analyze the actual final output

        if report_stats["Total Characters"] > 0 and report_stats["Breakdown"]:
            preferred_order = ['Code/Text (Total)', 'Comments', 'Placeholders & Summaries', 'Structural Elements', 'Whitespace/Empty Lines']
            sorted_breakdown_items = []
            temp_breakdown = report_stats["Breakdown"].copy()

            for key in preferred_order:
                if key in temp_breakdown:
                    sorted_breakdown_items.append((key, temp_breakdown.pop(key)))
            sorted_breakdown_items.extend(sorted(temp_breakdown.items(), key=lambda item: item[1]['chars'], reverse=True))

            for category, data in sorted_breakdown_items:
                main_percentage = data.get('percentage', 0.0)
                print(f"  - {category:<28}: {data['chars']:>10,} chars ({main_percentage:>6.2f}%)", file=sys.stderr)
                if category == 'Code/Text (Total)' and 'sub_categories' in data and data['sub_categories']:
                    for ext, sub_data in data['sub_categories'].items():
                        print(f"    - {ext:<26}: {sub_data['chars']:>10,} chars ({sub_data['percentage_of_total']:>6.2f}% of total, {sub_data['percentage_of_code_text']:>6.2f}% of Code/Text)", file=sys.stderr)
            print(f"  - {'Total Output Characters':<28}: {report_stats['Total Characters']:>10,} chars (100.00%)", file=sys.stderr)
        else:
            print("  - No content in output to analyze.", file=sys.stderr)
            print(f"  - {'Total Output Characters':<28}: {report_stats['Total Characters']:>10,} chars (100.00%)", file=sys.stderr)

        print("-" * 40, file=sys.stderr)
        if args.output and output_path_obj:
            print(f"Total bytes written to {output_path_obj}: {total_bytes_written:,}", file=sys.stderr)
        else:
            print(f"Total bytes written to stdout: {total_bytes_written:,}", file=sys.stderr)
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
        if args.output and output_handle is not sys.stdout and output_handle: 
            try:
                output_handle.close()
                log.debug(f"Closed output file: {args.output}")
            except Exception as e: # pragma: no cover
                log.error(f"Error closing output file '{args.output}': {e}")

if __name__ == "__main__":
    main()