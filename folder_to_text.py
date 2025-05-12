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
from typing import Dict, List, Optional, Tuple, Any

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(funcName)s: %(message)s', stream=sys.stderr)
log = logging.getLogger(__name__)

# --- Configuration (Used by standard mode) ---
IGNORE_PATTERNS = {
    '.git', '__pycache__', '.svn', '.hg', '.idea', '.vscode', 'node_modules',
    'build', 'dist', 'target', 'venv', '.venv', 'anaconda3', 'AppData',
    '.DS_Store', 'Thumbs.db',
    '*.bak', '*.old', '*.tmp', '*~', '*.log', '*.swp', '*.zip', '*.gz', '*.tar',
    '*.class', '*.jar', '*.exe', '*.dll', '*.so', '*.o', '*.a', '*.lib',
    '*.pyc', '*.pyo',
    '*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg', '*.ico', '*.pdf', '*.doc', '*.docx',
    '*.xls', '*.xlsx', '*.ppt', '*.pptx', '*.mp3', '*.mp4', '*.avi', '*.wav', '*.ogg', '*.opus',
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'composer.lock', 'go.sum',
    '*.min.js', '*.min.css', '*.html'
}
CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.kt', '.swift',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rb', '.php', '.sh',
    '.bash', '.zsh', '.css', '.scss', '.less', '.html', '.htm', '.xml', '.json',
    '.yaml', '.yml', '.toml', '.sql', '.md', '.rst', '.dockerfile', 'Dockerfile',
    '.r', '.pl', '.pm', '.lua', '.gradle', '.tf', '.tfvars', '.conf', '.ini', '.cfg', '.properties',
    'Makefile', 'Jenkinsfile', 'Gemfile', 'pom.xml', 'build.gradle', 'settings.gradle',
    '.ipynb', '.env.example', '.env'
}
INTERESTING_FILENAMES = {
    'README', 'LICENSE', 'CONTRIBUTING', 'CHANGELOG', 'requirements.txt', 'Pipfile', 'setup.py',
    'docker-compose.yml', 'docker-compose.yaml', 'package.json', 'go.mod',
    '.gitignore', '.dockerignore', '.editorconfig', '.gitattributes',
}
BINARY_CHECK_BYTES = 1024
BINARY_NON_PRINTABLE_THRESHOLD = 0.15

SIMPLIFICATION_PATTERNS = [
    (re.compile(r'\d{8,}'), '*NUM_LONG*'),
    (re.compile(r'\b[a-fA-F0-9]{12,}\b'), '*HEX_LONG*'),
    (re.compile(r'[a-zA-Z0-9+/=]{30,}'), '*BASE64LIKE_LONG*'),
]

POST_SIMPLIFICATION_PATTERNS = [
    (re.compile(r"""(['"])\b([a-fA-F0-9]{8}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{12})\b\1"""), r'"*UUID*"'),
    (re.compile(r"""(['"])(https?://[^/'"]+)(/[^'"\s]+/?)(\w+\.(?:py|js|html|css|png|jpg|gif|svg|json|xml|yaml|yml))\1"""), r'"\g<2>/*PATH*/\g<4>"'),
    (re.compile(r"""(['"])(https?://[^/'"]+)(/[^'"\s]+)\1"""), r'"\g<2>/*PATH*"'),
    (re.compile(r"""(['"])((?:\\.|[^\\\1])*?)\1"""),
     lambda m: '...' if len(m.group(2)) > 5 else m.group(0)),
]

SINGLE_VOICE_ID_FINDER = re.compile(
    r"""
    (                   # Start capturing group 1 (the whole match we want)
      "                   # Opening double quote
      [a-z]{2,3}          # Lang
      -
      [A-Z][a-zA-Z0-9]{1,3} # Region or Script
      (?:-(?:[A-Z]{2}|[a-zA-Z0-9]+))? # Optional Region/Variant
      -
      [A-Z][a-zA-Z0-9]+   # Name
      Neural              # Literal Neural
      "                   # Closing double quote
      ,?                  # Optional comma
    )                   # End capturing group 1
    """, re.VERBOSE
)

BLOCK_COMPRESSION_PATTERNS = {
    "VOICE_ID": re.compile(
        r"""^                 # Start of line
        \s*                   # Optional leading whitespace
        "                     # Opening double quote
        (?:[a-z]{2,3})        # Lang (non-capturing)
        -
        (?:[A-Z][a-zA-Z0-9]{1,3}) # Region or Script (non-capturing)
        (?:-(?:[A-Z]{2}|[a-zA-Z0-9]+))? # Optional Region/Variant (non-capturing)
        -
        (?:[A-Z][a-zA-Z0-9]+)   # Name (non-capturing)
        Neural                # Literal Neural
        "                     # Closing double quote
        ,?                    # Optional comma
        \s*                   # Optional trailing whitespace
        $                     # End of line
        """, re.VERBOSE
    ),
    "QUOTED_UUID": re.compile(r"""^\s*(['"])[a-fA-F0-9]{8}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{12}\1,?$"""),
}
DEFAULT_MIN_CONSECUTIVE_LINES = 2

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
    except OSError as e:
        log.warning(f"OS error checking binary status for {filepath}: {e}")
        return False
    except Exception as e:
        log.warning(f"Unexpected error checking binary status for {filepath}: {e}")
        return False

def summarize_other_files(filenames: list[str], code_extensions: set, interesting_filenames: set) -> str:
    if not filenames: return ""
    ext_counts = Counter(); explicit_files = set()
    interesting_lower = {fn.lower() for fn in interesting_filenames}; code_ext_lower = {ext.lower() for ext in code_extensions}
    for fname in filenames:
        base, ext = os.path.splitext(fname); fname_lower = fname.lower(); ext_lower = ext.lower()
        is_interesting = (fname_lower in interesting_lower or
                          base.lower() in interesting_lower or
                          fname in code_extensions or
                          base in code_extensions)
        if is_interesting: explicit_files.add(fname)
        elif ext_lower and ext_lower not in code_ext_lower: ext_counts[ext_lower] += 1
        elif not ext and base and fname_lower not in interesting_lower and base.lower() not in interesting_lower: ext_counts['<no_ext>'] += 1
    summary_parts = []
    if explicit_files: summary_parts.extend(sorted(list(explicit_files)))
    sorted_ext_counts = sorted([item for item in ext_counts.items() if item[0] not in code_ext_lower], key=lambda x: x[0])
    for ext, count in sorted_ext_counts: summary_parts.append(f"{count}x {ext}" if count > 1 else f"1x {ext}")
    if not summary_parts: return f"{len(filenames)} other file(s)"
    return ", ".join(summary_parts)

# --- Core Processing Functions ---

def simplify_source_code(
    content: str,
    strip_logging: bool,
    large_literal_threshold: int,
    disable_literal_compression: bool
) -> str:
    if not content.strip(): return ""
    # Remove block comments first
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL) # C-style block comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL) # HTML/XML comments
    # Remove multi-line string literals (often used as block comments in Python)
    content = re.sub(r'"""(?:.|\n)*?"""', '', content, flags=re.MULTILINE) # Python docstrings/multiline
    content = re.sub(r"'''(?:.|\n)*?'''", '', content, flags=re.MULTILINE) # Python docstrings/multiline

    lines = content.splitlines()
    simplified_lines = []
    in_literal_block = False
    literal_block_start_index = -1
    literal_line_count = 0
    # Pattern to identify simple literal lines (strings, numbers, booleans, None, placeholders)
    literal_line_pattern = re.compile(r"""^\s*(?:(?:r|u|f|b)?(['"])(?:(?=(\\?))\2.)*?\1|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|True|False|None|\*NUM_LONG\*|\*HEX_LONG\*|\*BASE64LIKE_LONG\*|\*UUID\*|\*INT\*|\*FLOAT\*|\*...\*)\s*[,\]\}]?\s*$""", re.VERBOSE)
    # Pattern to identify start of a list or dict assignment/declaration
    list_dict_start_pattern = re.compile(r'[:=]\s*(\[|\{)\s*$')

    for i, line in enumerate(lines):
        original_line = line # Keep original for appending if not stripped/modified
        
        # Remove line comments
        line = re.sub(r'(?<![:/])#.*$', '', line) # Python/Ruby/Shell style comments (don't remove URLs with #)
        line = re.sub(r'\s+//.*$', '', line)      # C++/Java/JS style comments
        line = re.sub(r'^\s*//.*$', '', line)     # C++/Java/JS style comments at line start
        line = re.sub(r'\s+--.*$', '', line)      # SQL/Lua style comments
        line = re.sub(r'^\s*--.*$', '', line)     # SQL/Lua style comments at line start
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
            literal_block_start_index = current_line_index
            literal_line_count = 0
            simplified_lines.append(original_line) # Append the line that starts the block
        elif in_literal_block:
            is_simple_literal_line = literal_line_pattern.match(stripped_line) is not None
            
            if is_simple_literal_line:
                literal_line_count += 1
                simplified_lines.append(original_line) # Append this literal line
            elif is_end_char: # End of the block
                simplified_lines.append(original_line) # Append the line that ends the block
                if not disable_literal_compression and literal_line_count >= large_literal_threshold and literal_block_start_index >= 0:
                    # Compress the collected literal lines
                    start_slice_idx = literal_block_start_index + 1 # Don't remove the line that opened the bracket
                    end_slice_idx = start_slice_idx + literal_line_count
                    
                    # Ensure slice indices are valid
                    if start_slice_idx <= end_slice_idx and end_slice_idx <= len(simplified_lines) -1: # -1 because the closing line is already appended
                         try:
                            # Determine indentation from the first line in the block to be compressed
                            indent_line_candidate = simplified_lines[start_slice_idx]
                            indent_level = indent_line_candidate.find(indent_line_candidate.lstrip()) if indent_line_candidate.strip() else \
                                           (simplified_lines[literal_block_start_index].find(simplified_lines[literal_block_start_index].lstrip()) + 2) # Fallback indent
                         except IndexError:
                            indent_level = 2 # Default indent if something goes wrong

                         indent = " " * (indent_level if indent_level >=0 else 2)
                         placeholder_line = f"{indent}# --- Large literal collection compressed ({literal_line_count} lines) ---"
                         
                         del simplified_lines[start_slice_idx:end_slice_idx]
                         simplified_lines.insert(start_slice_idx, placeholder_line)
                         log.debug(f"Compressed literal block, {literal_line_count} lines. Start index: {literal_block_start_index}, placeholder at {start_slice_idx}")
                    else:
                         log.warning(f"Compression slice error: start_block={literal_block_start_index}, start_slice={start_slice_idx}, end_slice={end_slice_idx}, len_simp={len(simplified_lines)}, lit_lines={literal_line_count}. Skipping.")
                
                in_literal_block = False
                literal_block_start_index = -1
                literal_line_count = 0
            else: # Non-literal line or complex structure inside the block, ends the current literal block
                log.debug(f"Complex line ended literal block: {original_line.strip()[:80]}")
                simplified_lines.append(original_line) # Add this "complex" line
                in_literal_block = False # Reset state
                literal_block_start_index = -1
                literal_line_count = 0
        elif stripped_line: # Regular non-empty line, not in a literal block
            simplified_lines.append(line)
        elif simplified_lines and simplified_lines[-1].strip(): # Add a single blank line if previous was not blank
            simplified_lines.append("") # Add the blank line (it's already stripped of comments)

    # Join lines and apply regex simplifications
    processed_content = "\n".join(simplified_lines).strip() # Strip leading/trailing newlines from the whole content
    if processed_content: # Add a single trailing newline if content exists
        processed_content += "\n" 

    for pattern, replacement in SIMPLIFICATION_PATTERNS:
        processed_content = pattern.sub(replacement, processed_content)

    # Reduce multiple blank lines to a single blank line
    processed_content = re.sub(r'\n{3,}', '\n\n', processed_content)
    return processed_content

def process_single_file_content(
    target_file_path: Path,
    buffer: io.StringIO,
    strip_logging: bool,
    large_literal_threshold: int,
    disable_literal_compression_in_simplify: bool,
    is_additional_context: bool = False 
):
    def write_to_buffer(text): buffer.write(text + "\n")
    log_prefix = "Additional Context: " if is_additional_context else ""
    log.debug(f"{log_prefix}Processing single file: {target_file_path} (Standard Mode)")
    
    # Try to get a relative path for display, fall back to absolute if not possible (e.g. different drives on Windows)
    try:
        # Check if cwd is an ancestor of target_file_path before calling relative_to
        if Path.cwd() in target_file_path.parents or Path.cwd() == target_file_path.parent:
            display_path_str = str(target_file_path.relative_to(Path.cwd()))
        else:
            display_path_str = str(target_file_path)
    except ValueError: 
        display_path_str = str(target_file_path)

    file_info_prefix = f"--- File: {display_path_str}"
    if is_additional_context: file_info_prefix = f"--- File (from additional context): {display_path_str}"

    if is_likely_binary(target_file_path):
        log.warning(f"{log_prefix}Skipping likely binary: {target_file_path}")
        write_to_buffer(file_info_prefix + " --- SKIPPED (BINARY) ---"); write_to_buffer("")
        return
    try:
        log.debug(f"{log_prefix}Reading and simplifying source file: {target_file_path}")
        with open(target_file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
        simplified_content = simplify_source_code(content, strip_logging, large_literal_threshold, disable_literal_compression_in_simplify)
        is_empty_after_simplification = not simplified_content.strip()
        write_to_buffer(file_info_prefix + " ---")
        if is_empty_after_simplification: write_to_buffer("# (File is empty or contained only comments/whitespace/logging after simplification)")
        else: write_to_buffer(simplified_content.strip()) # simplify_source_code adds trailing \n if content
        write_to_buffer(f"--- End File: {display_path_str} ---"); write_to_buffer("") 
    except OSError as e:
        log.error(f"{log_prefix}Error reading file {target_file_path}: {e}")
        write_to_buffer(file_info_prefix + " Error Reading ---"); write_to_buffer(f"### Error reading file: {type(e).__name__}: {e} ###"); write_to_buffer(f"--- End File: {display_path_str} ---"); write_to_buffer("")
    except Exception as e:
        log.error(f"{log_prefix}Error processing file {target_file_path}: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
        write_to_buffer(file_info_prefix + " Error Processing ---"); write_to_buffer(f"### Error processing file: {type(e).__name__}: {e} ###"); write_to_buffer(f"--- End File: {display_path_str} ---"); write_to_buffer("")

def process_folder_contents(
    folder_to_scan: Path, buffer: io.StringIO, ignore_set: set, code_extensions_set: set,
    interesting_filenames_set: set, skip_empty: bool, strip_logging: bool, skip_duplicates: bool, 
    large_literal_threshold: int, disable_literal_compression_in_simplify: bool, is_additional_context: bool = False
):
    global seen_content_hashes 
    def write_to_buffer(text): buffer.write(text + "\n")
    log_prefix = "Additional Context Folder: " if is_additional_context else "Primary Folder: "
    log.debug(f"Starting folder walk for: {log_prefix}{folder_to_scan} (Standard Mode)")
    if not is_additional_context: log.debug(f"Primary folder scan - Effective skip_empty: {skip_empty}, skip_duplicates: {skip_duplicates}")

    for dirpath_str, dirnames, filenames in os.walk(folder_to_scan, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} - {e}")):
        current_path = Path(dirpath_str)
        try: current_rel_path_in_scan = current_path.relative_to(folder_to_scan)
        except ValueError: log.warning(f"Could not determine relative path for {current_path} against base {folder_to_scan}. Skipping."); dirnames[:] = []; continue
        log.debug(f"{log_prefix}Processing directory: {current_rel_path_in_scan}")
        dirnames[:] = [d for d in dirnames if d not in ignore_set and not d.startswith('.') and not any(Path(d).match(p) for p in ignore_set if '*' in p or '?' in p)]
        filenames.sort(); dirnames.sort()
        source_files_to_process, other_text_filenames = [], []
        for filename in filenames:
            if filename in ignore_set or any(Path(filename).match(p) for p in ignore_set if '*' in p or '?' in p): log.debug(f"{log_prefix}Ignoring file by pattern: {current_rel_path_in_scan / filename}"); continue
            if filename.startswith('.') and filename.lower() not in interesting_filenames_set and filename not in code_extensions_set: log.debug(f"{log_prefix}Ignoring hidden file: {current_rel_path_in_scan / filename}"); continue
            file_path_abs = current_path / filename 
            if not file_path_abs.is_file(): continue
            if is_likely_binary(file_path_abs): log.debug(f"{log_prefix}Skipping likely binary file: {file_path_abs}"); continue
            base, ext = os.path.splitext(filename); ext_lower, fname_lower = ext.lower(), filename.lower()
            is_source = (ext_lower in code_extensions_set or filename in code_extensions_set or fname_lower in code_extensions_set)
            relative_file_path_in_scan = file_path_abs.relative_to(folder_to_scan)
            if is_source: source_files_to_process.append((relative_file_path_in_scan, file_path_abs))
            else: other_text_filenames.append(filename)
        if source_files_to_process:
            dir_header_display_path = str(current_rel_path_in_scan) if str(current_rel_path_in_scan) != '.' else folder_to_scan.name
            dir_header = f"\n{'=' * 10} Directory ({log_prefix.strip(': ')}): {dir_header_display_path} {'=' * 10}\n"
            if dir_header: write_to_buffer(dir_header)
            for rel_path_in_scan, full_path_abs in source_files_to_process:
                # display_path_for_marker is what appears in "--- File: display_path_for_marker ---"
                # This needs to be the path relative to the scanned root (folder_to_scan)
                # or relative to CWD if primary input was a single file.
                # For process_folder_contents, it's always relative to folder_to_scan.
                display_path_for_marker = str(rel_path_in_scan)
                if is_additional_context: # Prepend the additional context folder's name for clarity
                     display_path_for_marker = f"{folder_to_scan.name}/{rel_path_in_scan}"
                
                file_info_prefix = f"--- File: {display_path_for_marker}"
                if is_additional_context: # Adjust prefix for clarity if needed, though display_path_for_marker already has context folder name
                    file_info_prefix = f"--- File (from additional context): {display_path_for_marker}"


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
                    else: write_to_buffer(simplified_content.strip()) # simplify_source_code adds trailing \n if content
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
                dir_display_name = str(current_rel_path_in_scan) if str(current_rel_path_in_scan) != '.' else folder_to_scan.name
                summary_line = f"{indent}# Other files in '{dir_display_name}': {summary}"
                log.debug(f"{log_prefix}Adding summary for {current_rel_path_in_scan}: {summary}")
                buffer.seek(0, io.SEEK_END); current_pos = buffer.tell()
                if current_pos > 0: # Check if buffer is not empty
                    buffer.seek(max(0, current_pos - 300)); last_part = buffer.read() # Check last few lines
                    dir_header_display_path = str(current_rel_path_in_scan) if str(current_rel_path_in_scan) != '.' else folder_to_scan.name
                    # Ensure there's a blank line if the last content wasn't the directory header or another blank line
                    if not last_part.strip().endswith(f"Directory ({log_prefix.strip(': ')}): {dir_header_display_path} {'=' * 10}") and \
                       not last_part.endswith("\n\n") and \
                       not last_part.endswith("\n\r\n"): # Check for common newline sequences
                        write_to_buffer("") 
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
            # For a single file input, display_path_for_marker is just its name for raw dump
            display_path_for_marker = current_target_path_obj.name
            buffer.write(f"\n{'=' * 20} START FILE ({current_target_desc}): {display_path_for_marker} {'=' * 20}\n")
            try:
                with open(current_target_path_obj, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                if content and not content.endswith('\n'): content += '\n'
                buffer.write(content); file_count_for_target = 1
            except Exception as e: log.error(f"Error reading/writing file {display_path_for_marker} for raw dump: {e}"); buffer.write(f"### Error during raw dump of {display_path_for_marker}: {e} ###\n")
            buffer.write(f"{'=' * 20} END FILE ({current_target_desc}): {display_path_for_marker} {'=' * 20}\n")
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
                    # In raw dump, the marker includes the relative path and the filename
                    # e.g. START FILE (Primary Input/src/utils.py): utils.py ====
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
    total_replacements = 0; lines = content.splitlines(keepends=True); output_lines = []; log.debug(f"Applying {len(patterns)} post-simplification patterns...")
    pattern_counts = Counter()
    for line in lines:
        modified_line = line
        for i, (pattern, replacement) in enumerate(patterns):
            try:
                 # original_line_segment = modified_line # Not used
                 modified_line_new, count = pattern.subn(replacement, modified_line) if not callable(replacement) else pattern.subn(replacement, modified_line)
                 if count > 0:
                     total_replacements += count; pattern_counts[i] += count
                     if log.getEffectiveLevel() <= logging.DEBUG: # Simplified debug log
                         log.debug(f"  Pattern {i} ({pattern.pattern[:30]}...) matched {count} time(s).")
                     modified_line = modified_line_new
            except Exception as e: log.error(f"Error applying post-simplification pattern {i} ({pattern.pattern}) to line: {e}"); log.debug(f"Problematic line: {line.strip()[:100]}")
        output_lines.append(modified_line)
    modified_content = "".join(output_lines)
    if pattern_counts: log.debug(f"Post-simplification pattern counts: {pattern_counts}")
    log.info(f"Applied {len(patterns)} post-simplification patterns, making {total_replacements} total replacements.")
    return modified_content, total_replacements

def expand_multi_pattern_lines(content: str, finder_pattern: re.Pattern, pattern_name_for_log: str = "PATTERN") -> tuple[str, int]:
    lines = content.splitlines(keepends=False); output_lines = []; lines_expanded = 0
    log.debug(f"--- expand_multi_pattern_lines: Starting scan for '{pattern_name_for_log}' ---")
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith(("#", "---", "##", "==", "*LINE_REF_")): output_lines.append(line + "\n"); continue
        try: matches = finder_pattern.findall(line)
        except Exception as e: log.error(f"Regex error finding patterns on line {i+1}: {e}"); log.debug(f"Problematic line: {line}"); matches = []
        if len(matches) > 1:
             combined_match_len = sum(len(str(m).strip(',').strip()) for m in matches)
             stripped_len_approx = len(re.sub(r'\s+|,', '', stripped_line))
             if combined_match_len >= stripped_len_approx * 0.8: # If matches make up most of the non-whitespace line
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
        # Check if line is structural, a placeholder itself, or empty
        is_structural_or_placeholder = (stripped_line.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or
            any(re.match(pat, stripped_line) for pat in (r'^\*VOICE:.*\*$', r'^\*UUID\*$', r'^\*...\*$')) or # Check for lines that are *just* placeholders
            not stripped_line)
        if len(stripped_line) >= min_length and not is_structural_or_placeholder:
            line_counts[line] += 1
            if line not in meaningful_lines_indices: meaningful_lines_indices[line] = []
            meaningful_lines_indices[line].append(i)

    replacement_map, definition_lines, placeholder_counter = {}, [], 1
    # Sort by count (desc) then by line content (asc) for deterministic placeholder assignment
    repeated_lines = sorted([(l, c) for l, c in line_counts.items() if c >= min_repetitions], key=lambda item: (-item[1], item[0]))
    for line, count in repeated_lines:
        if line not in replacement_map: # Should always be true due to sorting and processing unique lines
            placeholder = placeholder_template.format(placeholder_counter); replacement_map[line] = placeholder
            log.debug(f"  Def {placeholder} for line ({count}x): {line.strip()[:80]}...")
            simplified_definition_content = line.rstrip() # Use rstrip to keep leading indent but remove trailing newline for definition
            for pattern, replacement in DEFINITION_SIMPLIFICATION_PATTERNS:
                 try: simplified_definition_content = pattern.sub(replacement, simplified_definition_content) if not callable(replacement) else pattern.subn(replacement, simplified_definition_content)[0]
                 except Exception as e: log.warning(f"Error def simp pattern {pattern.pattern}: {e}")
            definition_lines.append(f"{placeholder} = {simplified_definition_content}"); placeholder_counter += 1
    if not replacement_map: log.info("Line minification: No lines met criteria."); return content, 0

    new_lines = lines[:]; num_actual_replacements = 0
    for original_line, placeholder in replacement_map.items():
        for index in meaningful_lines_indices.get(original_line, []): # Should always find indices
            new_lines[index] = placeholder + ("\n" if original_line.endswith("\n") else "") # Preserve original newline ending
            num_actual_replacements += 1
    minified_content = "".join(new_lines)
    if definition_lines:
        definition_header = ["", "=" * 40, f"# Line Minification Definitions ({len(definition_lines)}):", f"# (Lines >= {min_length} chars, >= {min_repetitions}x, content simplified)", "=" * 40]
        definition_block = "\n".join(definition_header + definition_lines) + "\n\n"
        log.info(f"Line minification: Replaced {num_actual_replacements} occurrences of {len(definition_lines)} unique lines.")
        return definition_block + minified_content, num_actual_replacements
    return content, 0

def post_process_cleanup(content: str, cleanup_pattern: re.Pattern) -> tuple[str, int]:
    lines = content.splitlines(keepends=True); output_lines = []; lines_removed = 0
    log.debug(f"--- post_process_cleanup: Starting final cleanup using pattern: {cleanup_pattern.pattern} ---")
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Keep essential structural lines, definition lines, and specific summaries explicitly.
        # Also keep general comments unless they *only* consist of placeholders covered by cleanup_pattern.
        if stripped_line.startswith(("--- File:", "--- End File:", "===")) or \
           (stripped_line.startswith("*LINE_REF_") and " = " in stripped_line) or \
           stripped_line.startswith("## [Compressed Block:") or \
           stripped_line.startswith("# Line Minification Definitions") or \
           stripped_line.startswith("# (File is empty") or \
           stripped_line.startswith("# Other files in") or \
           stripped_line.startswith("# --- Large literal"):
            output_lines.append(line)
            continue

        if cleanup_pattern.match(line): # Match on full line, pattern accounts for leading/trailing whitespace
            log.debug(f"Post-cleanup removing line {i+1} (matched cleanup pattern): {line.strip()[:80]}...")
            lines_removed += 1
        else:
            output_lines.append(line)
            
    log.info(f"Post-processing cleanup removed {lines_removed} lines containing only placeholders/structure.")
    cleaned_content = "".join(output_lines); return re.sub(r'\n{3,}', '\n\n', cleaned_content), lines_removed


# --- Content Analysis Report Function ---
def generate_content_report(final_output_content: str) -> Dict[str, Any]:
    """
    Analyzes the final output content and categorizes its character composition.
    "Code/Text" is further broken down by file extension.
    Returns a dictionary with total characters and a breakdown by category.
    """
    char_counts = {
        'Structural Elements': 0,
        'Placeholders & Summaries': 0,
        'Comments': 0,
        'Code/Text': Counter(),  # Stores counts per extension, e.g., {'.py': 100, '.js': 50}
        'Whitespace/Empty Lines': 0,
    }
    total_chars = len(final_output_content)
    current_file_ext = "<metadata>"  # Default for lines before any file marker (e.g. script headers)

    if total_chars == 0:
        return {"Total Characters": 0, "Breakdown": {}}

    lines = final_output_content.splitlines() # These lines do NOT include their trailing newlines
    num_lines = len(lines)

    for idx, line_text in enumerate(lines):
        current_segment_len = len(line_text)
        if idx < num_lines - 1 or (idx == num_lines - 1 and final_output_content.endswith('\n')):
            current_segment_len += 1 # Account for the newline character

        original_line_text_for_match = line_text 
        stripped_line_text = line_text.strip()
        
        category_assigned = None

        if not stripped_line_text:
            category_assigned = 'Whitespace/Empty Lines'
        else:
            # 1. Structural Elements (most specific, e.g., headers, file markers)
            # This loop also updates current_file_ext if a file marker is hit.
            is_structural_and_file_marker = False
            for pattern in REPORT_STRUCTURAL_PATTERNS:
                if pattern.match(original_line_text_for_match):
                    category_assigned = 'Structural Elements'
                    
                    # Check if this structural line is a file marker to update current_file_ext
                    # Standard mode file marker:
                    std_match = STD_FILE_MARKER_EXTRACT_RE.match(original_line_text_for_match)
                    if std_match:
                        is_structural_and_file_marker = True
                        if "SKIPPED (BINARY)" in original_line_text_for_match:
                            current_file_ext = "<binary>"
                        else:
                            filename_in_marker = std_match.group(1)
                            _, ext_val = os.path.splitext(filename_in_marker)
                            current_file_ext = ext_val.lower() if ext_val else "<no_ext>"
                        break # Found file marker, no need to check other structural patterns

                    # Raw mode file marker:
                    raw_match = RAW_FILE_MARKER_EXTRACT_RE.match(original_line_text_for_match)
                    if raw_match:
                        is_structural_and_file_marker = True
                        # Group 2 is the actual filename, e.g., "file.py"
                        filename_in_marker = raw_match.group(2).strip()
                        _, ext_val = os.path.splitext(filename_in_marker)
                        current_file_ext = ext_val.lower() if ext_val else "<no_ext>"
                        break # Found file marker

                    # If it's structural but not a file marker, just break from structural patterns
                    if category_assigned : break 
            
            if category_assigned: # If assigned as Structural (either file marker or other)
                char_counts[category_assigned] += current_segment_len
                continue # Move to next line

            # 2. Placeholder & Summary Lines (specific summaries or full placeholder lines)
            for pattern in REPORT_PLACEHOLDER_SUMMARY_PATTERNS:
                if pattern.match(original_line_text_for_match):
                    category_assigned = 'Placeholders & Summaries'; break
            if category_assigned:
                char_counts[category_assigned] += current_segment_len
                continue
            
            # 3. Lines consisting *only* of placeholders and list/dict syntax (dynamic check)
            if PLACEHOLDER_CLEANUP_PATTERN.match(original_line_text_for_match):
                category_assigned = 'Placeholders & Summaries'
                char_counts[category_assigned] += current_segment_len
                continue

            # 4. Comments (general comments not caught by above categories)
            for pattern in REPORT_COMMENT_PATTERNS:
                if pattern.match(original_line_text_for_match):
                    category_assigned = 'Comments'; break
            if category_assigned:
                char_counts[category_assigned] += current_segment_len
                continue
            
            # 5. Default to Code/Text
            category_assigned = 'Code/Text'
        
        # Assign characters to the determined category
        if category_assigned == 'Code/Text':
            char_counts['Code/Text'][current_file_ext] += current_segment_len
        elif category_assigned: # Other main categories
             char_counts[category_assigned] += current_segment_len


    # Prepare report data structure for printing
    report_breakdown = {}
    total_code_text_chars = sum(char_counts['Code/Text'].values())

    # Process main categories (excluding Code/Text itself, which is broken down)
    for category_name, count_value in char_counts.items():
        if category_name == 'Code/Text': continue # Skip the Counter object itself

        percentage = (count_value / total_chars) * 100 if total_chars > 0 else 0
        report_breakdown[category_name] = {'chars': count_value, 'percentage': percentage}

    # Process Code/Text and its sub-categories
    if total_code_text_chars > 0:
        percentage_total_code_text = (total_code_text_chars / total_chars) * 100 if total_chars > 0 else 0
        code_text_entry = {
            'chars': total_code_text_chars,
            'percentage': percentage_total_code_text,
            'sub_categories': {}
        }
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
    elif 'Code/Text' in char_counts: # Ensure 'Code/Text (Total)' entry exists even if 0 chars
         report_breakdown['Code/Text (Total)'] = {'chars': 0, 'percentage': 0.0, 'sub_categories': {}}


    return {"Total Characters": total_chars, "Breakdown": report_breakdown}


# --- Main Function ---
def main():
    global IGNORE_PATTERNS, CODE_EXTENSIONS, INTERESTING_FILENAMES, POST_SIMPLIFICATION_PATTERNS, BLOCK_COMPRESSION_PATTERNS, SINGLE_VOICE_ID_FINDER, DEFINITION_SIMPLIFICATION_PATTERNS, PLACEHOLDER_CLEANUP_PATTERN, seen_content_hashes, DEFAULT_LARGE_LITERAL_THRESHOLD, DEFAULT_MIN_CONSECUTIVE_LINES
    global REPORT_STRUCTURAL_PATTERNS, REPORT_PLACEHOLDER_SUMMARY_PATTERNS, REPORT_COMMENT_PATTERNS
    global STD_FILE_MARKER_EXTRACT_RE, RAW_FILE_MARKER_EXTRACT_RE


    DEFAULT_MIN_LINE_LENGTH_REC = 25
    DEFAULT_MIN_REPETITIONS_REC = 2

    parser = argparse.ArgumentParser(description="Generate a text representation of a folder/file.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_path", help="Primary target folder or file.")
    parser.add_argument("--additional-context-folder", type=Path, default=None, help="Additional folder to include.")
    parser.add_argument("-o", "--output", help="Output file (stdout if not set).")
    parser.add_argument("--raw-dump", action="store_true", default=False, help="Dump content verbatim.")
    
    st_group = parser.add_argument_group('Standard Mode Options (ignored if --raw-dump)')
    st_group.add_argument("--ignore", nargs='+', default=[], help="Names/patterns to ignore.")
    st_group.add_argument("--source-ext", nargs='+', default=[], help="Source code extensions/filenames.")
    st_group.add_argument("--interesting-files", nargs='+', default=[], help="Notable filenames for summaries.")
    st_group.add_argument("--keep-empty", action="store_true", default=False, help="Keep files if empty after simplification.")
    st_group.add_argument("--keep-duplicates", action="store_true", default=False, help="Keep files if simplified content is duplicated.")
    
    st_group.add_argument("--no-strip-logging", action="store_false", dest="strip_logging", default=True, help="Disable removing logging statements.")
    st_group.add_argument("--no-apply-patterns", action="store_false", dest="apply_patterns", default=True, help="Disable detailed post-simplification (e.g., *UUID*).")
    st_group.add_argument("--minify-lines", action="store_true", dest="minify_lines", default=False, 
                        help="Enable repeated identical long line minification (*LINE_REF_*).")
    st_group.add_argument("--no-large-literal-compression", action="store_false", dest="large_literal_compression_enabled", default=True,
                        help="Disable compressing large literal collections (lists/dicts).")
    st_group.add_argument("--no-preprocess-split-lines", action="store_false", dest="preprocess_split_lines", default=True, 
                        help="Disable pre-splitting multi-pattern lines (e.g. VOICE_IDs).")
    st_group.add_argument("--no-compress-patterns", action="store_false", dest="compress_patterns", default=True, 
                        help="Disable compressing consecutive pattern lines (e.g. blocks of VOICE_IDs).")
    st_group.add_argument("--no-post-cleanup", action="store_false", dest="post_cleanup", default=True, 
                        help="Disable final removal of placeholder-only lines.")

    st_group.add_argument("--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE_LINES, help="Min lines for pattern block compression.")
    st_group.add_argument("--min-line-length", type=int, default=DEFAULT_MIN_LINE_LENGTH_REC, help="Min length for line minification (if --minify-lines).")
    st_group.add_argument("--min-repetitions", type=int, default=DEFAULT_MIN_REPETITIONS_REC, help="Min repetitions for line minification (if --minify-lines).")
    st_group.add_argument("--large-literal-threshold", type=int, default=DEFAULT_LARGE_LITERAL_THRESHOLD, help="Min lines for literal compression.")
    
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    args = parser.parse_args()

    log.setLevel(args.log_level.upper()); logging.getLogger().setLevel(args.log_level.upper())
    seen_content_hashes.clear(); total_bytes_written = 0

    primary_input_path_obj = Path(args.input_path)
    if not primary_input_path_obj.exists(): log.critical(f"Error: Primary input path not found: '{args.input_path}'"); sys.exit(1)
    if not (primary_input_path_obj.is_dir() or primary_input_path_obj.is_file()): log.critical(f"Error: Primary input path '{args.input_path}' is neither a directory nor a file."); sys.exit(1)
    
    resolved_primary_path = primary_input_path_obj.resolve(); is_primary_input_directory = resolved_primary_path.is_dir()
    resolved_additional_context_path = None
    if args.additional_context_folder:
        additional_context_path_obj = Path(args.additional_context_folder)
        if not additional_context_path_obj.is_dir(): log.warning(f"Warning: Additional context path '{args.additional_context_folder}' is not a directory. It will be ignored.")
        else:
            resolved_additional_context_path = additional_context_path_obj.resolve()
            if resolved_additional_context_path == resolved_primary_path and is_primary_input_directory:
                log.warning(f"Warning: Additional context folder is the same as the primary input folder. It will be processed once as primary."); resolved_additional_context_path = None 

    print("-" * 40, file=sys.stderr)
    print(f"Processing primary {'folder' if is_primary_input_directory else 'file'}: {resolved_primary_path}", file=sys.stderr)
    if resolved_additional_context_path: print(f"Including additional context folder: {resolved_additional_context_path}", file=sys.stderr)
    print(f"Output target: {'Stdout' if not args.output else args.output}", file=sys.stderr)
    print(f"Log Level: {args.log_level.upper()}", file=sys.stderr)
    print(f"Mode: {'RAW DUMP' if args.raw_dump else 'Standard (Compression/Simplification)'}", file=sys.stderr)

    effective_skip_empty = not args.keep_empty; effective_skip_duplicates = not args.keep_duplicates

    if not args.raw_dump:
        print(f"  Skip empty (folder processing): {effective_skip_empty} ({'ON (DEFAULT)' if effective_skip_empty else 'Disabled (--keep-empty)'})", file=sys.stderr)
        print(f"  Skip duplicates (folder processing): {effective_skip_duplicates} ({'ON (DEFAULT)' if effective_skip_duplicates else 'Disabled (--keep-duplicates)'})", file=sys.stderr)
        print(f"  Strip logging: {args.strip_logging} ({'ON (DEFAULT)' if args.strip_logging else 'Disabled (--no-strip-logging)'})", file=sys.stderr)
        print(f"  Apply detailed patterns: {args.apply_patterns} ({'ON (DEFAULT)' if args.apply_patterns else 'Disabled (--no-apply-patterns)'})", file=sys.stderr)
        print(f"  Minify identical lines: {args.minify_lines} ({'ON (--minify-lines)' if args.minify_lines else 'OFF (DEFAULT)'})", file=sys.stderr)
        if args.minify_lines: print(f"    Min line length: {args.min_line_length}, Min repetitions: {args.min_repetitions}", file=sys.stderr)
        print(f"  Large literal compression: {args.large_literal_compression_enabled} ({'ON (DEFAULT)' if args.large_literal_compression_enabled else 'Disabled (--no-large-literal-compression)'}, threshold: {args.large_literal_threshold})", file=sys.stderr)
        print(f"  Pre-process split lines: {args.preprocess_split_lines} ({'ON (DEFAULT)' if args.preprocess_split_lines else 'Disabled (--no-preprocess-split-lines)'})", file=sys.stderr)
        print(f"  Compress pattern blocks: {args.compress_patterns} ({'ON (DEFAULT)' if args.compress_patterns else 'Disabled (--no-compress-patterns)'})", file=sys.stderr)
        if args.compress_patterns: print(f"    Min consecutive lines: {args.min_consecutive}", file=sys.stderr)
        print(f"  Post-process cleanup: {args.post_cleanup} ({'ON (DEFAULT)' if args.post_cleanup else 'Disabled (--no-post-cleanup)'})", file=sys.stderr)
    else: print("  (Standard mode options are ignored in raw dump mode, except for multiple input paths)", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    output_handle = None; output_path = None; buffer = io.StringIO(); output_content_to_write = ""
    try:
        header_desc = f"Primary {'folder' if is_primary_input_directory else 'file'}: {resolved_primary_path}"
        if resolved_additional_context_path: header_desc += f" | Additional Context: {resolved_additional_context_path}"
        options_list = []
        if not args.raw_dump:
            options_list.extend([f"strip_log={args.strip_logging}", f"apply_pat={args.apply_patterns}",
                                 f"minify={args.minify_lines}({args.min_line_length}c,{args.min_repetitions}x)", 
                                 f"large_lit={args.large_literal_compression_enabled}({args.large_literal_threshold}l)",
                                 f"split_lines={args.preprocess_split_lines}", f"comp_pat={args.compress_patterns}({args.min_consecutive}l)",
                                 f"cleanup={args.post_cleanup}"])
            if is_primary_input_directory: options_list.extend([f"skip_empty_f={effective_skip_empty}", f"skip_dupl_f={effective_skip_duplicates}"])
        header_options_summary = ", ".join(options_list)
        header_lines = [f"# {'RAW DUMP' if args.raw_dump else 'Compressed Representation'} of {header_desc}",
                        f"# Generated by folder_to_text.py ({'--raw-dump' if args.raw_dump else 'Standard Mode'})"]
        if not args.raw_dump and header_options_summary: header_lines.append(f"# Options: {header_options_summary}")
        header_lines.extend(["=" * 40, ""])
        for line in header_lines: buffer.write(line + "\n")

        if args.raw_dump:
            log.info(f"Starting raw dump...")
            process_folder_raw(str(resolved_primary_path), buffer, str(resolved_additional_context_path) if resolved_additional_context_path else None)
            log.info(f"Finished raw dump.")
        else: 
            current_ignore = IGNORE_PATTERNS.copy(); current_ignore.update(args.ignore)
            current_code_ext = CODE_EXTENSIONS.copy(); current_code_ext.update(args.source_ext)
            current_interesting = INTERESTING_FILENAMES.copy(); current_interesting.update(args.interesting_files)
            disable_literal_compression_flag = not args.large_literal_compression_enabled

            log.info(f"Processing primary input: {resolved_primary_path} (Step 1a)...")
            if is_primary_input_directory:
                process_folder_contents(resolved_primary_path, buffer, current_ignore, current_code_ext, current_interesting,
                                        effective_skip_empty, args.strip_logging, effective_skip_duplicates,
                                        args.large_literal_threshold, disable_literal_compression_flag)
            else: 
                process_single_file_content(resolved_primary_path, buffer, args.strip_logging,
                                            args.large_literal_threshold, disable_literal_compression_flag)
            log.info("Finished processing primary input.")

            if resolved_additional_context_path:
                log.info(f"Processing additional context folder: {resolved_additional_context_path} (Step 1b)...")
                buffer.write(f"\n{'=' * 10} STARTING ADDITIONAL CONTEXT: {resolved_additional_context_path.name} {'=' * 10}\n\n")
                process_folder_contents(resolved_additional_context_path, buffer, current_ignore, current_code_ext, current_interesting,
                                        effective_skip_empty, args.strip_logging, effective_skip_duplicates,
                                        args.large_literal_threshold, disable_literal_compression_flag, is_additional_context=True)
                buffer.write(f"\n{'=' * 10} FINISHED ADDITIONAL CONTEXT: {resolved_additional_context_path.name} {'=' * 10}\n\n")
                log.info("Finished processing additional context folder.")
        
        num_lines_expanded, num_pattern_replacements, num_blocks_compressed, num_lines_minified, num_lines_cleaned_up = 0,0,0,0,0
        if not args.raw_dump:
            buffer.seek(0); processed_content = buffer.getvalue(); final_output_content = processed_content

            if args.preprocess_split_lines and final_output_content.strip():
                log.info("Pre-processing: Expanding multi-pattern lines (Step 2)...")
                final_output_content, num_lines_expanded = expand_multi_pattern_lines(final_output_content, SINGLE_VOICE_ID_FINDER, "VOICE_ID")
            else: log.debug("Skipping Step 2: Pre-process split lines (disabled or empty content)")

            if args.compress_patterns and final_output_content.strip():
                 log.info("Compressing consecutive pattern blocks (Step 3)...")
                 final_output_content, num_blocks_compressed = compress_pattern_blocks(final_output_content, BLOCK_COMPRESSION_PATTERNS, args.min_consecutive)
            else: log.debug("Skipping Step 3: Compress pattern blocks (disabled or empty content)")

            if args.apply_patterns and final_output_content.strip():
                log.info("Applying detailed post-simplification patterns (Step 4)...")
                final_output_content, num_pattern_replacements = apply_post_simplification_patterns(final_output_content, POST_SIMPLIFICATION_PATTERNS)
            else: log.debug("Skipping Step 4: Apply detailed patterns (disabled or empty content)")

            if args.minify_lines and final_output_content.strip(): 
                log.info("Minifying repeated identical lines (Step 5)...")
                final_output_content, num_lines_minified = minify_repeated_lines(final_output_content, args.min_line_length, args.min_repetitions)
            else: log.info(f"Skipping Step 5: Minify identical lines (minify_lines: {args.minify_lines}, content empty: {not final_output_content.strip()})")

            if args.post_cleanup and final_output_content.strip():
                 log.info("Applying post-processing cleanup (Step 6)...")
                 final_output_content, num_lines_cleaned_up = post_process_cleanup(final_output_content, PLACEHOLDER_CLEANUP_PATTERN)
            else: log.debug("Skipping Step 6: Post-process cleanup (disabled or empty content)")
            output_content_to_write = final_output_content
        else: 
            buffer.seek(0)
            output_content_to_write = buffer.getvalue()

        log.info("Writing final output...")
        if args.output:
            output_path = Path(args.output).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as output_handle: output_handle.write(output_content_to_write)
            log.debug(f"Opened and wrote to output file: {output_path}")
        else: 
            sys.stdout.write(output_content_to_write)
            output_handle = sys.stdout # For pylint check
            log.debug("Wrote to stdout for output.")
        total_bytes_written = len(output_content_to_write.encode('utf-8'))
        log.info("Finished writing output.")

        # --- Generate and Print Summary & Content Report ---
        print("-" * 40, file=sys.stderr); print("Processing complete.", file=sys.stderr)
        print(f"Mode: {'RAW DUMP' if args.raw_dump else 'Standard'}", file=sys.stderr)
        if not args.raw_dump:
             print(f"  Line expansion: {num_lines_expanded if args.preprocess_split_lines else 'SKIPPED (disabled)'}.", file=sys.stderr)
             print(f"  Pattern block compression: {num_blocks_compressed if args.compress_patterns else 'SKIPPED (disabled)'} blocks created.", file=sys.stderr)
             print(f"  Detailed pattern application: {num_pattern_replacements if args.apply_patterns else 'SKIPPED (disabled)'} replacements.", file=sys.stderr)
             if args.minify_lines: 
                 if num_lines_minified > 0: print(f"  Identical line minification: {num_lines_minified} occurrences replaced.", file=sys.stderr)
                 else: print(f"  Identical line minification: No lines met criteria for replacement (or feature enabled but no matches).", file=sys.stderr)
             else: print(f"  Identical line minification: SKIPPED (default off, use --minify-lines to enable).", file=sys.stderr)
             print(f"  Post-processing cleanup: {num_lines_cleaned_up if args.post_cleanup else 'SKIPPED (disabled)'} lines removed.", file=sys.stderr)
        
        print("-" * 40, file=sys.stderr)
        print("Content Analysis Report:", file=sys.stderr)
        report_stats = generate_content_report(output_content_to_write)
        
        if report_stats["Total Characters"] > 0 and report_stats["Breakdown"]:
            # Sort main categories for display (e.g., by character count or a defined order)
            # For simplicity, using a predefined order then sorting remaining by count
            preferred_order = ['Code/Text (Total)', 'Comments', 'Placeholders & Summaries', 'Structural Elements', 'Whitespace/Empty Lines']
            sorted_breakdown_items = []
            temp_breakdown = report_stats["Breakdown"].copy()

            for key in preferred_order:
                if key in temp_breakdown:
                    sorted_breakdown_items.append((key, temp_breakdown.pop(key)))
            # Add any remaining categories (shouldn't be any with current setup) sorted by count
            sorted_breakdown_items.extend(sorted(temp_breakdown.items(), key=lambda item: item[1]['chars'], reverse=True))

            for category, data in sorted_breakdown_items:
                main_percentage = data.get('percentage', 0.0) # For Code/Text (Total), this is its overall percentage
                print(f"  - {category:<28}: {data['chars']:>10,} chars ({main_percentage:>6.2f}%)", file=sys.stderr)
                if category == 'Code/Text (Total)' and 'sub_categories' in data and data['sub_categories']:
                    # sub_categories are already sorted by count in generate_content_report
                    for ext, sub_data in data['sub_categories'].items():
                        print(f"    - {ext:<26}: {sub_data['chars']:>10,} chars ({sub_data['percentage_of_total']:>6.2f}% of total, {sub_data['percentage_of_code_text']:>6.2f}% of Code/Text)", file=sys.stderr)
            print(f"  - {'Total Output Characters':<28}: {report_stats['Total Characters']:>10,} chars (100.00%)", file=sys.stderr)
        else:
            print("  - No content in output to analyze.", file=sys.stderr)
            print(f"  - {'Total Output Characters':<28}: {report_stats['Total Characters']:>10,} chars (100.00%)", file=sys.stderr)


        if args.output and output_path: print(f"Total bytes written to {output_path}: {total_bytes_written:,}", file=sys.stderr)
        else: print(f"Total bytes written to stdout: {total_bytes_written:,}", file=sys.stderr)
        print("-" * 40, file=sys.stderr)

    except IOError as e:
        log.critical(f"I/O Error: {e}")
        if args.output: log.critical(f"Failed operation likely involved file: {args.output}")
        sys.exit(1)
    except KeyboardInterrupt: log.warning("Processing interrupted by user (Ctrl+C)."); sys.exit(1)
    except Exception as e: log.critical(f"An unexpected error occurred: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG)); sys.exit(1)
    finally:
        if buffer: buffer.close()

if __name__ == "__main__":
    main()