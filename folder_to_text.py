#!/usr/bin/env python3

# --- folder_to_text.py ---
# Processes a folder, simplifying text files, summarizing others,
# and compresses repetitive patterns or lines using recommended settings by default,
# with a final cleanup step. Includes a raw dump mode and file type reporting.

# --- Example Usage ---

# 1. Default (Max Reduction - Recommended):
#    python folder_to_text.py /path/to/project -o max_compressed_cleaned.txt

# 2. Raw Dump (Concatenate all files verbatim):
#    python folder_to_text.py /path/to/project --raw-dump -o raw_dump.txt

# 3. Basic Simplification (Disable most compression/cleanup):
#    python folder_to_text.py /path/to/project --keep-empty --keep-duplicates --no-preprocess-split-lines --no-compress-patterns --no-minify-lines --no-post-cleanup -o basic_simple.txt

# 4. Using --apply-patterns (Disable block/line, enable apply, keep cleanup):
#    python folder_to_text.py /path/to/project --no-compress-patterns --no-minify-lines --apply-patterns -o applied_cleaned.txt

# 5. Debugging post-cleanup (Default settings + Debug log):
#    python folder_to_text.py /path/to/project --log-level DEBUG -o debug_run.txt 2> debug.log


import os
import re
import argparse
import sys
import string
from pathlib import Path
from collections import Counter
# import json # Seems unused
import logging
import hashlib
import io
# import asyncio # Seems unused
# import functools # Seems unused
# import time # Seems unused
# import datetime # Seems unused
# from enum import Enum, auto # Seems unused
# from collections import defaultdict # Seems unused
from typing import Dict, List, Optional, Tuple, Any, Union

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(funcName)s: %(message)s', stream=sys.stderr)
log = logging.getLogger(__name__)


# --- Configuration (Used by standard mode) ---
IGNORE_PATTERNS = {
    '.git', '__pycache__', '.svn', '.hg', '.idea', '.vscode', 'node_modules',
    'build', 'dist', 'target', 'venv', '.venv', 'anaconda3', 'AppData',
    '.DS_Store', 'Thumbs.db',
    'test', 'tests', '*.test.py', 'test_*.py', '*_test.py', '*.spec.py',
    '*.bak', '*.old', '*.tmp', '*~', '*.log', '*.swp', '*.zip', '*.gz', '*.tar',
    '*.class', '*.jar', '*.exe', '*.dll', '*.so', '*.o', '*.a', '*.lib',
    '*.pyc', '*.pyo',
    '*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg', '*.ico', '*.pdf', '*.doc', '*.docx',
    '*.xls', '*.xlsx', '*.ppt', '*.pptx', '*.mp3', '*.mp4', '*.avi', '*.wav', '*.ogg', '*.opus',
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'composer.lock', 'go.sum',
    '*.min.js', '*.min.css'
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
    #(re.compile(r'\b\d{2,}\.\d{1,}|\d{1,}\.\d{2,}\b'), '*FLOAT*'),
    #(re.compile(r'\b\d{3,}\b'), '*INT*'),
]

POST_SIMPLIFICATION_PATTERNS = [
    (re.compile(r"""(['"])\b([a-fA-F0-9]{8}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{4}-?[a-fA-F0-9]{12})\b\1"""), r'"*UUID*"'),
    (re.compile(r"""(['"])(https?://[^/'"]+)(/[^'"\s]+/?)(\w+\.(?:py|js|html|css|png|jpg|gif|svg|json|xml|yaml|yml))\1"""), r'"\g<2>/*PATH*/\g<4>"'),
    (re.compile(r"""(['"])(https?://[^/'"]+)(/[^'"\s]+)\1"""), r'"\g<2>/*PATH*"'),
    (re.compile(r"""(['"])((?:\\.|[^\\\1])*?)\1"""), # Corrected pattern
     lambda m: '...' if len(m.group(2)) > 10 else m.group(0)),
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
# --- Default changed ---
DEFAULT_MIN_CONSECUTIVE_LINES = 3

DEFINITION_SIMPLIFICATION_PATTERNS = [
    (re.compile(r'\b\d{2,}\.\d{1,}|\d{1,}\.\d{2,}\b'), '*FLOAT*'),
    (re.compile(r'\b\d{3,}\b'), '*INT*'),
    (re.compile(r"""(['"])((?:\\.|[^\\\1])*?)\1"""), '...'), # Use corrected pattern
    (re.compile(r'\b[a-fA-F0-9]{10,}\b'), '*HEX*'),
    (re.compile(r'[a-zA-Z0-9+/=]{20,}'), '*BASE64LIKE*'),
]

DEFAULT_LARGE_LITERAL_THRESHOLD = 10

# --- Placeholders targeted by --post-cleanup ---
ALL_PLACEHOLDERS = [
    r'\*...\*', r'\*INT\*', r'\*FLOAT\*', r'\*UUID\*',
    r'\*NUM_LONG\*', r'\*HEX_LONG\*', r'\*BASE64LIKE_LONG\*',
    r'\*HEX\*', r'\*BASE64LIKE\*', # From definition simplification
    r'\/\*PATH\*\/?' # Path placeholder (needs escaping)
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
    """.format(placeholder_group='|'.join(ALL_PLACEHOLDERS)), # Insert placeholder options
    re.VERBOSE
)


# --- Globals ---
seen_content_hashes = set() # Used by standard mode

# --- Helper Functions (Used by standard mode) ---
def is_likely_binary(filepath: Path) -> bool:
    """ Checks if a file is likely binary. Reads a small chunk. """
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
    """ Creates a compressed summary of non-source text filenames. """
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

# --- Core Processing Functions (Used by standard mode) ---

def simplify_source_code(
    content: str,
    strip_logging: bool,
    large_literal_threshold: int,
    disable_literal_compression: bool
) -> str:
    """
    Simplifies source code: removes comments, blank lines, obfuscates constants,
    compresses large literal lists/dicts (conditionally), optionally strips logging lines.
    Applies basic simplification patterns from SIMPLIFICATION_PATTERNS.
    (Used only in standard mode)
    """
    if not content.strip(): return ""

    # Stage 1: Comment Removal
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'"""(?:.|\n)*?"""', '', content, flags=re.MULTILINE)
    content = re.sub(r"'''(?:.|\n)*?'''", '', content, flags=re.MULTILINE)

    # Stage 2: Line Processing
    lines = content.splitlines()
    simplified_lines = []
    in_literal_block = False
    literal_block_start_index = -1
    literal_line_count = 0
    literal_line_pattern = re.compile(r"""^\s*(?:(?:r|u|f|b)?(['"])(?:(?=(\\?))\2.)*?\1|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|True|False|None|\*...\*|\*INT\*|\*FLOAT\*|\*NUM_LONG\*|\*HEX_LONG\*|\*BASE64LIKE_LONG\*)\s*,?\s*$""", re.VERBOSE)
    list_dict_start_pattern = re.compile(r'[:=]\s*(\[|\{)\s*$')

    for i, line in enumerate(lines):
        original_line = line
        line = re.sub(r'(?<![:/])#.*$', '', line) # Python/Shell comments
        line = re.sub(r'\s+//.*$', '', line) # C++/JS style end-of-line comments
        line = re.sub(r'^\s*//.*$', '', line) # C++/JS style whole line comments
        line = re.sub(r'\s+--.*$', '', line) # SQL style end-of-line comments
        line = re.sub(r'^\s*--.*$', '', line) # SQL style whole line comments
        stripped_line = line.strip()

        # Logging Removal (Optional)
        if strip_logging and stripped_line:
            # Simple check for common logging/print patterns
            if re.match(r'^(?:log|logger|logging|console|print|fmt\.Print|System\.out\.print|TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\b', stripped_line, re.IGNORECASE):
                 if '(' in stripped_line and ')' in stripped_line: # Basic check for function call
                    continue # Skip this line

        # Large Literal Compression (Optional, disabled if --compress-patterns active)
        is_potential_start = list_dict_start_pattern.search(line) is not None
        is_end = stripped_line.startswith(']') or stripped_line.startswith('}')
        current_line_index = len(simplified_lines) # Index where *this* line *would* go

        if is_potential_start and not in_literal_block:
            in_literal_block = True
            literal_block_start_index = current_line_index
            literal_line_count = 0
            simplified_lines.append(original_line) # Add the line that starts the block
        elif in_literal_block:
            is_simple_literal_line = literal_line_pattern.match(stripped_line) is not None
            if is_simple_literal_line:
                literal_line_count += 1
                simplified_lines.append(original_line) # Add the literal line for now
            elif is_end:
                simplified_lines.append(original_line) # Add the line that ends the block
                # Check if we should compress the block *now* that it's finished
                if not disable_literal_compression and literal_line_count >= large_literal_threshold and literal_block_start_index >= 0:
                    # Calculate slice indices relative to the *current* simplified_lines
                    start_slice_idx = literal_block_start_index + 1 # First literal line
                    end_slice_idx = start_slice_idx + literal_line_count # One past the last literal line
                    if start_slice_idx < end_slice_idx <= len(simplified_lines)-1: # Ensure indices are valid
                         # Determine indent from the line *after* the start or the line *before* the end
                         try:
                            indent_line = simplified_lines[start_slice_idx] if start_slice_idx < len(simplified_lines) else simplified_lines[literal_block_start_index]
                            indent_level = indent_line.find(indent_line.lstrip()) if indent_line.strip() else 2 # Guess indent if blank
                         except IndexError:
                            indent_level = 2 # Fallback indent

                         indent = " " * (indent_level)
                         placeholder_line = f"{indent}# --- Large literal collection compressed ({literal_line_count} lines) ---"
                         # Replace the literal lines with the placeholder
                         del simplified_lines[start_slice_idx:end_slice_idx]
                         simplified_lines.insert(start_slice_idx, placeholder_line)
                         log.debug(f"Compressed literal block, {literal_line_count} lines.")
                    else:
                         log.warning(f"Compression slice calculation error: start={start_slice_idx}, end={end_slice_idx}, len={len(simplified_lines)}. Skipping literal compression.")
                # Reset literal block state regardless of compression
                in_literal_block = False
                literal_block_start_index = -1
                literal_line_count = 0
            else:
                # A non-simple, non-end line occurred within the block - treat it as ending the potential compression
                log.debug(f"Complex line ended potential literal block compression.")
                simplified_lines.append(original_line) # Add the complex line
                in_literal_block = False
                literal_block_start_index = -1
                literal_line_count = 0
        # Regular line processing (not in a literal block)
        elif stripped_line: # Keep non-blank lines
            simplified_lines.append(line)
        elif simplified_lines and simplified_lines[-1].strip(): # Add a single blank line if the previous wasn't blank
            simplified_lines.append("")

    # Final processing after line iteration
    processed_content = "\n".join(simplified_lines).strip()
    if processed_content: processed_content += "\n" # Ensure trailing newline if content exists

    # Apply basic simplification patterns
    for pattern, replacement in SIMPLIFICATION_PATTERNS:
        processed_content = pattern.sub(replacement, processed_content)

    # Final blank line cleanup
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
    compress_patterns_enabled: bool,
    report_data: Dict[str, Counter] # <<< ADDED: Pass the report dictionary
):
    """
    Processes the folder using simplification, filtering, etc. (Standard Mode).
    Populates the report_data dictionary with file type statistics. # <<< ADDED: Docstring update
    """
    global seen_content_hashes
    target_path = Path(target_folder).resolve()
    def write_to_buffer(text): buffer.write(text + "\n")

    log.debug(f"Starting folder walk for: {target_path} (Standard Mode)")
    log.debug(f"Effective skip_empty: {skip_empty}, skip_duplicates: {skip_duplicates}")

    for dirpath, dirnames, filenames in os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} - {e}")):
        current_path = Path(dirpath)
        try:
            current_rel_path = current_path.relative_to(target_path)
        except ValueError:
            log.warning(f"Could not determine relative path for {current_path} against base {target_path}. Skipping directory.")
            dirnames[:] = [] # Don't traverse further down this path
            continue

        log.debug(f"Processing directory: {current_rel_path}")

        # Filter ignored directories *before* recursing into them
        dirnames[:] = [
            d for d in dirnames if
            d not in ignore_set and
            not d.startswith('.') and # Generic hidden folder check
            not any(Path(d).match(p) for p in ignore_set if '*' in p or '?' in p) # Glob pattern check
        ]
        filenames.sort()
        dirnames.sort() # Ensure deterministic order

        source_files_to_process = []
        other_text_filenames = []

        # Filter files in the current directory
        for filename in filenames:
            # Check basic ignores
            if filename in ignore_set or any(Path(filename).match(p) for p in ignore_set if '*' in p or '?' in p):
                log.debug(f"Ignoring file by pattern: {current_rel_path / filename}")
                continue
            # Check hidden files (unless explicitly interesting or code)
            if filename.startswith('.') and filename.lower() not in interesting_filenames_set and filename not in code_extensions_set:
                 log.debug(f"Ignoring hidden file: {current_rel_path / filename}")
                 continue

            file_path = current_path / filename
            if not file_path.is_file(): continue # Skip if not a file (e.g., broken symlink)

            # Check for binary files
            if is_likely_binary(file_path):
                log.debug(f"Skipping likely binary file: {file_path.relative_to(target_path)}")
                continue

            # Categorize file
            base, ext = os.path.splitext(filename); ext_lower = ext.lower(); fname_lower = filename.lower()
            is_source = (ext_lower in code_extensions_set or
                         filename in code_extensions_set or # Check full filename (e.g., Dockerfile)
                         fname_lower in code_extensions_set) # Check lower full filename

            relative_file_path = file_path.relative_to(target_path)
            if is_source:
                source_files_to_process.append((relative_file_path, file_path, ext_lower))
            else:
                other_text_filenames.append(filename)

        # Process source files for this directory
        if source_files_to_process:
            dir_header = f"\n{'=' * 10} Directory: {current_rel_path} {'=' * 10}\n" if str(current_rel_path) != '.' else ""
            if dir_header: write_to_buffer(dir_header)

            for rel_path, full_path, file_ext in source_files_to_process:
                file_info_prefix = f"--- File: {rel_path}"
                # <<< ADDED: Determine file identifier for reporting >>>
                file_id_for_report = file_ext if file_ext else rel_path.name

                # <<< ADDED: Initialize counter for this file type if needed >>>
                if file_id_for_report not in report_data:
                    report_data[file_id_for_report] = Counter()

                # <<< ADDED: Increment processed count >>>
                report_data[file_id_for_report]['processed'] += 1

                try:
                    log.debug(f"Processing source file: {rel_path}")
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # <<< ADDED: Get hash *before* simplification for comparison >>>
                    hash_before = hashlib.sha256(content.encode('utf-8')).hexdigest()

                    simplified_content = simplify_source_code(
                        content,
                        strip_logging,
                        large_literal_threshold,
                        disable_literal_compression=compress_patterns_enabled
                    )

                    # <<< ADDED: Get hash *after* simplification >>>
                    hash_after = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest()
                    is_simplified = hash_before != hash_after # Check if content actually changed

                    is_empty_after_simplification = not simplified_content.strip()
                    content_hash = hash_after # Use the hash of simplified content for duplication check
                    is_duplicate = False

                    # <<< ADDED: Track simplification >>>
                    if is_simplified and not is_empty_after_simplification:
                         report_data[file_id_for_report]['simplified'] += 1

                    # Check if skipping empty files
                    if skip_empty and is_empty_after_simplification:
                        log.info(f"Skipping empty file (after simplification): {rel_path}")
                        # <<< ADDED: Track skipped empty >>>
                        report_data[file_id_for_report]['skipped_empty'] += 1
                        continue # Skip to next file

                    # Check for duplicates if enabled and file is not empty
                    if skip_duplicates and not is_empty_after_simplification:
                        # content_hash = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest() # Already calculated above
                        if content_hash in seen_content_hashes:
                            is_duplicate = True
                            log.info(f"Skipping duplicate content file: {rel_path} (Hash: {content_hash[:8]}...)")
                            # <<< ADDED: Track skipped duplicate >>>
                            report_data[file_id_for_report]['skipped_duplicate'] += 1
                            continue # Skip to next file
                        else:
                             seen_content_hashes.add(content_hash)
                             log.debug(f"Adding new content hash: {content_hash[:8]}... for {rel_path}")

                    # <<< ADDED: If we reached here, the file's content was included (or would be if not empty/dup)
                    #             and potentially subject to post-processing.
                    report_data[file_id_for_report]['contributed'] += 1

                    # Write file header
                    write_to_buffer(file_info_prefix + " ---")

                    # Write content or empty message
                    if is_empty_after_simplification:
                        # Even if kept (--keep-empty), mark it as contributing 0 content size effectively
                        write_to_buffer("# (File is empty or contained only comments/whitespace/logging)")
                    else:
                        write_to_buffer(simplified_content.strip())


                    # Write file footer
                    write_to_buffer(f"--- End File: {rel_path} ---")
                    write_to_buffer("") # Add a blank line between files

                except OSError as e:
                    log.error(f"Error reading file {rel_path}: {e}")
                    write_to_buffer(file_info_prefix + " Error Reading ---")
                    write_to_buffer(f"### Error reading file: {type(e).__name__}: {e} ###")
                    write_to_buffer(f"--- End File: {rel_path} ---")
                    write_to_buffer("")
                except Exception as e:
                    log.error(f"Error processing file {rel_path}: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
                    write_to_buffer(file_info_prefix + " Error Processing ---")
                    write_to_buffer(f"### Error processing file: {type(e).__name__}: {e} ###")
                    write_to_buffer(f"--- End File: {rel_path} ---")
                    write_to_buffer("")

        # Add summary for other files in this directory
        if other_text_filenames:
            summary = summarize_other_files(other_text_filenames, code_extensions_set, interesting_filenames_set)
            if summary:
                indent_level = len(current_rel_path.parts)
                indent = "  " * indent_level if indent_level > 0 else ""
                summary_line = f"{indent}# Other files in '{current_rel_path}': {summary}"
                log.debug(f"Adding summary for {current_rel_path}: {summary}")
                # Ensure summary is added even if no source files were processed in this dir
                # Check if the buffer ends with the directory header or a blank line
                buffer.seek(0, io.SEEK_END)
                current_pos = buffer.tell()
                if current_pos > 0:
                    buffer.seek(max(0, current_pos - 200)) # Read last bit
                    last_part = buffer.read()
                    if not last_part.strip().endswith(f"Directory: {current_rel_path} {'=' * 10}") and not last_part.endswith("\n\n"):
                         write_to_buffer("") # Add separator if needed
                write_to_buffer(summary_line)
                write_to_buffer("") # Add blank line after summary

# --- NEW Raw Dump Function ---
def process_folder_raw(target_folder: str, buffer: io.StringIO):
    """
    Walks the folder and dumps every file's content verbatim into the buffer.
    (Used only in raw dump mode)
    """
    target_path = Path(target_folder).resolve()
    log.debug(f"Starting RAW folder walk for: {target_path}")
    file_count = 0

    # Sort directories and files for consistent output order
    walk_results = sorted(list(os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} - {e}"))), key=lambda x: x[0])

    for dirpath, dirnames, filenames in walk_results:
        # Sort within directory for consistency
        dirnames.sort()
        filenames.sort()

        current_path = Path(dirpath)
        try:
            current_rel_path = current_path.relative_to(target_path)
        except ValueError:
            log.warning(f"Could not determine relative path for {current_path} against base {target_path}. Skipping directory.")
            continue # Skip this directory path

        log.debug(f"Processing directory (RAW): {current_rel_path}")

        for filename in filenames:
            file_path = current_path / filename
            relative_file_path = file_path.relative_to(target_path)

            if not file_path.is_file():
                log.debug(f"Skipping non-file entry: {relative_file_path}")
                continue # Skip directories, broken symlinks etc.

            file_count += 1
            # --- File Header ---
            buffer.write(f"\n{'=' * 20} START FILE: {relative_file_path} {'=' * 20}\n")
            log.info(f"Dumping file ({file_count}): {relative_file_path}")

            try:
                # Read as text, ignoring decoding errors. Binary content will likely be mangled.
                # Consider reading as bytes if true binary handling is needed, but output becomes complex.
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                # Ensure content ends with a newline for cleaner separation
                if not content.endswith('\n'):
                    content += '\n'
                buffer.write(content)
            except OSError as e:
                log.error(f"Error reading file {relative_file_path}: {e}")
                buffer.write(f"### Error reading file: {type(e).__name__}: {e} ###\n")
            except Exception as e:
                log.error(f"Unexpected error processing file {relative_file_path}: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
                buffer.write(f"### Unexpected error processing file: {type(e).__name__}: {e} ###\n")

            # --- File Footer ---
            buffer.write(f"{'=' * 20} END FILE: {relative_file_path} {'=' * 20}\n")

    log.info(f"Finished raw dump. Processed {file_count} files.")


# --- Post-Processing Functions (Used by standard mode) ---

def apply_post_simplification_patterns(content: str, patterns: list[tuple[re.Pattern, Any]]) -> tuple[str, int]:
    """
    Applies a list of regex patterns and replacements to the content.
    Handles both string and lambda replacements.
    (Used only in standard mode)
    """
    total_replacements = 0
    modified_content = content
    lines = content.splitlines(keepends=True)
    output_lines = []

    log.debug(f"Applying {len(patterns)} post-simplification patterns...")
    pattern_counts = Counter()

    for line in lines:
        modified_line = line
        for i, (pattern, replacement) in enumerate(patterns):
            try:
                 original_line_segment = modified_line # Keep track for logging changes
                 if callable(replacement):
                     modified_line_new, count = pattern.subn(replacement, modified_line)
                 else:
                     modified_line_new, count = pattern.subn(replacement, modified_line)

                 if count > 0:
                     total_replacements += count
                     pattern_counts[i] += count
                     if log.getEffectiveLevel() <= logging.DEBUG:
                         # Find changed part for better logging (approximate)
                         diff_start = -1
                         for k in range(min(len(original_line_segment), len(modified_line_new))):
                             if original_line_segment[k] != modified_line_new[k]:
                                 diff_start = k
                                 break
                         log.debug(f"  Pattern {i} ({pattern.pattern[:30]}...) matched {count} time(s) on line: ...{original_line_segment[max(0,diff_start-10):diff_start+10]}... -> ...{modified_line_new[max(0,diff_start-10):diff_start+10]}...")
                     modified_line = modified_line_new # Update line for next pattern
            except Exception as e:
                log.error(f"Error applying post-simplification pattern {i} ({pattern.pattern}) to line: {e}")
                log.debug(f"Problematic line: {line.strip()[:100]}") # Log snippet

        output_lines.append(modified_line)

    modified_content = "".join(output_lines)

    if pattern_counts:
        log.debug("Post-simplification pattern match counts:")
        for i, count in pattern_counts.items():
            log.debug(f"  Pattern {i} ({patterns[i][0].pattern[:50]}...): {count} matches")

    log.info(f"Applied {len(patterns)} post-simplification patterns, making {total_replacements} total replacements.")
    return modified_content, total_replacements


def expand_multi_pattern_lines(
    content: str,
    finder_pattern: re.Pattern,
    pattern_name_for_log: str = "PATTERN"
    ) -> tuple[str, int]:
    """
    Scans content for lines containing multiple instances of a pattern
    and splits them into individual lines, preserving indentation.
    (Used only in standard mode)
    """
    lines = content.splitlines(keepends=False) # Don't keep ends, add them back later
    output_lines = []
    lines_expanded = 0
    log.debug(f"--- expand_multi_pattern_lines: Starting scan for '{pattern_name_for_log}' ---")

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Skip comments, headers, etc.
        if not stripped_line or stripped_line.startswith(("#", "---", "##", "==", "*LINE_REF_")):
            output_lines.append(line + "\n")
            continue

        try:
            # Use findall to get all non-overlapping matches
            matches = finder_pattern.findall(line)
        except Exception as e:
            log.error(f"Regex error finding patterns on line {i+1}: {e}")
            log.debug(f"Problematic line: {line}")
            matches = []

        # Only expand if *multiple* distinct matches are found on the line
        if len(matches) > 1:
             # Heuristic check: Ensure matches cover a significant part of the stripped line
             # This avoids splitting lines where the pattern might appear incidentally
             combined_match_len = sum(len(str(m).strip(',').strip()) for m in matches) # Use str(m) if group is captured
             stripped_len_approx = len(re.sub(r'\s+|,', '', stripped_line)) # Approx length without spaces/commas

             # Require matches to cover >80% of the significant content
             if combined_match_len >= stripped_len_approx * 0.8:
                log.info(f"Expanding line {i+1} containing {len(matches)} instances of '{pattern_name_for_log}'.")
                log.debug(f"Original line {i+1}: {line.strip()}")
                lines_expanded += 1
                indent_level = line.find(stripped_line[0]) if stripped_line else 0
                indent = " " * indent_level
                for match_item in matches:
                    # Assuming findall captures the exact string needed (Group 1 in the example)
                    match_str = str(match_item).strip() # Use str() for safety if match can be non-string
                    output_lines.append(f"{indent}{match_str}\n")
                    log.debug(f"  Expanded to: {indent}{match_str}")
                continue # Skip adding the original line

        # If not expanded, add the original line back
        output_lines.append(line + "\n")

    log.info(f"Finished line expansion pre-processing. Expanded {lines_expanded} lines containing multiple '{pattern_name_for_log}' instances.")
    return "".join(output_lines), lines_expanded


def compress_pattern_blocks(content: str, patterns_to_compress: dict[str, re.Pattern], min_consecutive: int) -> tuple[str, int]:
    """
    Scans content for consecutive lines matching predefined patterns and compresses them.
    (Used only in standard mode)
    """
    lines = content.splitlines(keepends=True)
    output_lines = []
    total_blocks_compressed = 0
    i = 0
    log.debug(f"--- compress_pattern_blocks: Starting scan (min_consecutive={min_consecutive}) ---")

    while i < len(lines):
        current_line = lines[i]
        stripped_line = current_line.strip()
        matched_pattern_name = None

        # Define lines that should *not* start or be part of a compressed block
        is_ignorable = (
            stripped_line.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or
            not stripped_line # Ignore blank lines
        )

        if not is_ignorable:
            # Check if the current stripped line matches any compression pattern
            for name, pattern in patterns_to_compress.items():
                if pattern.match(stripped_line):
                    matched_pattern_name = name
                    log.debug(f"Line {i+1} potentially starts block '{name}': {stripped_line[:60]}...")
                    break # Found a match for this line

        if matched_pattern_name:
            block_pattern_name = matched_pattern_name
            block_start_index = i
            block_lines_indices = [i] # Store indices of lines matching the *same* pattern consecutively
            j = i + 1 # Look ahead index

            # Greedily consume consecutive lines matching the *same* pattern
            while j < len(lines):
                next_line_raw = lines[j]
                next_stripped = next_line_raw.strip()

                # Check if the next line is ignorable (blank, comment, etc.)
                next_is_ignorable = (
                    next_stripped.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or
                    not next_stripped
                )
                if next_is_ignorable:
                    log.debug(f"  Block '{block_pattern_name}' interrupted at line {j+1} by ignorable line.")
                    break # Stop consuming for this block

                # Check if the next line matches the *same* pattern as the block started with
                if patterns_to_compress[block_pattern_name].match(next_stripped):
                    block_lines_indices.append(j)
                    log.debug(f"  Line {j+1} continues block '{block_pattern_name}'.")
                    j += 1
                else:
                    log.debug(f"  Block '{block_pattern_name}' ended at line {j+1} (no match or different pattern).")
                    break # Pattern doesn't match, end of this block

            # Check if enough consecutive lines were found
            block_count = len(block_lines_indices)
            if block_count >= min_consecutive:
                # Determine indentation from the first line of the block
                first_line_in_block = lines[block_start_index]
                indent = ""
                first_line_stripped = first_line_in_block.strip()
                if len(first_line_in_block) > len(first_line_stripped):
                     indent = first_line_in_block[:len(first_line_in_block) - len(first_line_stripped)]

                # Create summary line and add it
                summary_line = f"{indent}## [Compressed Block: {block_count} lines matching pattern '{block_pattern_name}'] ##\n"
                output_lines.append(summary_line)
                log.info(f"Compressed {block_count} lines (Indices {block_start_index+1}-{j}) matching '{block_pattern_name}'.")
                total_blocks_compressed += 1
                i = j # Move main index past the consumed block
            else:
                # Not enough consecutive lines, add them individually
                log.debug(f"  Block '{block_pattern_name}' starting at line {block_start_index+1} had only {block_count} lines (min={min_consecutive}). Not compressing.")
                for block_line_index in block_lines_indices:
                    output_lines.append(lines[block_line_index])
                i = j # Move main index past the checked lines
        else:
            # Line didn't match any pattern or was ignorable, add it and move to the next
            output_lines.append(current_line)
            i += 1

    log.info(f"Pattern block compression: Compressed {total_blocks_compressed} blocks of lines (min consecutive: {min_consecutive}).")
    return "".join(output_lines), total_blocks_compressed


def minify_repeated_lines(content: str, min_length: int, min_repetitions: int) -> tuple[str, int]:
    """
    Identifies repeated identical long lines, replaces them with placeholders,
    and simplifies the content within the definition block.
    (Used only in standard mode)
    """
    global DEFINITION_SIMPLIFICATION_PATTERNS

    lines = content.splitlines(keepends=True) # Keep line endings for accurate replacement
    line_counts = Counter()
    num_replaced = 0
    definitions = {}
    placeholder_template = "*LINE_REF_{}*"
    meaningful_lines_indices = {} # Map line content -> list of indices where it appears

    log.debug(f"--- minify_repeated_lines: Scanning for lines >= {min_length} chars, repeated >= {min_repetitions} times ---")

    # First pass: Count occurrences and record indices of potentially minifiable lines
    for i, line in enumerate(lines):
        # Use the raw line (including whitespace and ending) as the key for exact matches
        line_content_key = line
        stripped_line = line.strip()

        # Define lines that should NOT be minified (structure, comments, previous placeholders, short lines)
        is_structural_or_placeholder = (
            stripped_line.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or
            re.match(r'^\*VOICE:.*\*$', stripped_line) or # Example placeholder patterns
            re.match(r'^\*UUID\*$', stripped_line) or
            re.match(r'^\*...\*$', stripped_line) or
            not stripped_line # Exclude blank lines
        )

        if len(stripped_line) >= min_length and not is_structural_or_placeholder:
            line_counts[line_content_key] += 1
            if line_content_key not in meaningful_lines_indices:
                meaningful_lines_indices[line_content_key] = []
            meaningful_lines_indices[line_content_key].append(i)
            # Log potential candidates at DEBUG level if needed
            # if line_counts[line_content_key] == min_repetitions:
            #     log.debug(f"  Line '{stripped_line[:60]}...' (len={len(stripped_line)}) reached min repetitions ({min_repetitions}) at index {i}.")

    # Second pass: Identify lines meeting the repetition threshold and create definitions/replacements
    replacement_map = {} # Map original line content -> placeholder string
    definition_lines = []
    placeholder_counter = 1

    # Sort candidates by frequency (desc) and then alphabetically for deterministic placeholder assignment
    repeated_lines = sorted(
        [(line, count) for line, count in line_counts.items() if count >= min_repetitions],
        key=lambda item: (-item[1], item[0]) # Sort by count descending, then line content ascending
    )

    for line, count in repeated_lines:
        if line not in replacement_map: # Ensure we only define each unique line once
            placeholder = placeholder_template.format(placeholder_counter)
            replacement_map[line] = placeholder
            log.debug(f"  Creating definition {placeholder} for line repeated {count} times: {line.strip()[:80]}...")

            # Simplify the content *for the definition block only*
            simplified_definition_content = line.rstrip() # Use rstrip to keep leading whitespace but remove trailing newline for definition
            for pattern, replacement in DEFINITION_SIMPLIFICATION_PATTERNS:
                 try:
                     original_def_segment = simplified_definition_content
                     if callable(replacement):
                          simplified_definition_content, _ = pattern.subn(replacement, simplified_definition_content)
                     else:
                          simplified_definition_content = pattern.sub(replacement, simplified_definition_content)
                     # Optional: Log simplification details
                     # if original_def_segment != simplified_definition_content:
                     #     log.debug(f"    Simplified definition using pattern {pattern.pattern[:30]}...")
                 except Exception as e:
                      log.warning(f"Error applying definition simplification pattern {pattern.pattern}: {e}")

            definition_lines.append(f"{placeholder} = {simplified_definition_content}")
            placeholder_counter += 1

    # If no lines met the criteria, return original content
    if not replacement_map:
        log.info("Line minification enabled, but no lines met criteria.")
        return content, 0

    # Third pass: Apply the replacements to the original lines
    new_lines = lines[:] # Create a mutable copy
    replaced_indices = set() # Keep track of indices already replaced (though should be handled by map)
    num_actual_replacements = 0

    for original_line, placeholder in replacement_map.items():
        indices_to_replace = meaningful_lines_indices.get(original_line, [])
        replacements_done_for_this_line = 0
        for index in indices_to_replace:
            if index not in replaced_indices: # Should not happen if map keys are unique, but safe check
                # Replace with placeholder, preserving original newline character if present
                new_lines[index] = placeholder + ("\n" if original_line.endswith("\n") else "")
                replaced_indices.add(index)
                replacements_done_for_this_line += 1
        num_actual_replacements += replacements_done_for_this_line
        # Log detailed replacement info if needed
        # log.debug(f"  Replaced {replacements_done_for_this_line} occurrences of line with {placeholder}")


    minified_content = "".join(new_lines)

    # Prepend the definition block if definitions were created
    if definition_lines:
        definition_header = [
            "",
            "=" * 40,
            f"# Line Minification Definitions ({len(definition_lines)}):",
            f"# (Lines >= {min_length} chars repeated >= {min_repetitions} times, content simplified)",
            "=" * 40
        ]
        definition_block = "\n".join(definition_header + definition_lines) + "\n\n" # Add trailing newlines
        log.info(f"Line minification: Replaced {num_actual_replacements} occurrences of {len(definition_lines)} unique long lines. Definition content simplified.")
        return definition_block + minified_content, num_actual_replacements
    else:
        # Should not be reached if replacement_map was populated, but safe fallback
        return content, 0


# --- Post-Processing Cleanup Function (Used by standard mode) ---
def post_process_cleanup(content: str, cleanup_pattern: re.Pattern) -> tuple[str, int]:
    """
    Removes lines consisting primarily of placeholders and structural chars.
    (Used only in standard mode)
    """
    lines = content.splitlines(keepends=True)
    output_lines = []
    lines_removed = 0
    log.debug(f"--- post_process_cleanup: Starting final cleanup using pattern: {cleanup_pattern.pattern} ---")

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Always keep definition block lines, structure markers, and comments
        if (line.startswith("*LINE_REF_") or
            line.startswith("## [Compressed Block:") or
            line.startswith("--- File:") or
            line.startswith("--- End File:") or
            line.startswith("===") or
            stripped_line.startswith("#")): # Keep comments
            output_lines.append(line)
            continue

        # Check if the line matches the cleanup pattern (meaning it's mostly placeholders/structure)
        if cleanup_pattern.match(line):
            log.debug(f"Post-cleanup removing line {i+1}: {line.strip()[:80]}...")
            # Option 1: Remove the line completely (results in fewer lines)
            # No action needed here, just don't append it

            # Option 2: Replace with a marker (preserves line count somewhat, can be noisy)
            # indent = len(line) - len(line.lstrip(' '))
            # output_lines.append(" " * indent + "# [...]\n")

            lines_removed += 1
        else:
            output_lines.append(line) # Keep the line

    log.info(f"Post-processing cleanup removed {lines_removed} lines containing only placeholders/structure.")

    # Post-cleanup consolidation (optional, but recommended if removing lines)
    cleaned_content = "".join(output_lines)
    # Consolidate multiple blank lines potentially created by removal
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

    # If using markers (Option 2 above), consolidate them:
    # cleaned_content = re.sub(r'(\n\s*# \[...\])+\n', '\n# [...]\n', cleaned_content)

    return cleaned_content, lines_removed


# --- Main Function ---
def main():
    """ Main execution function """
    global IGNORE_PATTERNS, CODE_EXTENSIONS, INTERESTING_FILENAMES, POST_SIMPLIFICATION_PATTERNS, BLOCK_COMPRESSION_PATTERNS, SINGLE_VOICE_ID_FINDER, DEFINITION_SIMPLIFICATION_PATTERNS, PLACEHOLDER_CLEANUP_PATTERN, seen_content_hashes, DEFAULT_LARGE_LITERAL_THRESHOLD

    # --- Defaults aligned with "Recommended for Max Reduction" ---
    DEFAULT_MIN_LINE_LENGTH_REC = 40
    DEFAULT_MIN_REPETITIONS_REC = 2
    DEFAULT_MIN_CONSECUTIVE_REC = 3 # This was already the default

    parser = argparse.ArgumentParser(
        description=(
            "Generate a representation of a folder's text content. "
            "Default mode applies various compressions and simplifications. "
            "Use --raw-dump for a verbatim concatenation of all files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder_path", help="Path to the target folder.")
    parser.add_argument("-o", "--output", help="Output file path (optional, defaults to stdout).")

    # --- Mode Selection ---
    parser.add_argument("--raw-dump", action="store_true", default=False,
                        help="Dump all files in the folder verbatim, without any simplification, filtering, or compression. Overrides most other processing options.")

    # --- Standard Mode Options ---
    st_group = parser.add_argument_group('Standard Mode Options (ignored if --raw-dump is used)')
    st_group.add_argument("--ignore", nargs='+', default=[], help="Additional names/patterns to ignore (standard mode only).")
    st_group.add_argument("--source-ext", nargs='+', default=[], help="Additional extensions/filenames for source code (standard mode only).")
    st_group.add_argument("--interesting-files", nargs='+', default=[], help="Additional notable filenames for summaries (standard mode only).")

    # Flags to DISABLE default standard behaviors
    st_group.add_argument("--keep-empty", action="store_true", default=False,
                        help="Keep files even if empty after simplification (standard mode default: skip).")
    st_group.add_argument("--keep-duplicates", action="store_true", default=False, # Changed default to False
                        help="Keep files even if their simplified content is duplicated (standard mode default: skip).")
    st_group.add_argument("--no-preprocess-split-lines", action="store_false", dest="preprocess_split_lines", default=True,
                        help="Disable pre-processing split of multi-pattern lines (standard mode default: enabled).")
    st_group.add_argument("--no-compress-patterns", action="store_false", dest="compress_patterns", default=True,
                        help="Disable compression of consecutive pattern lines (standard mode default: enabled).")
    st_group.add_argument("--no-minify-lines", action="store_false", dest="minify_lines", default=True,
                        help="Disable repeated identical long line minification (standard mode default: enabled).")
    st_group.add_argument("--no-post-cleanup", action="store_false", dest="post_cleanup", default=True,
                        help="Disable final removal of placeholder-only lines (standard mode default: enabled).")

    # Optional standard mode flags
    st_group.add_argument("--strip-logging", action="store_true", default=False,
                        help="Attempt to remove common logging statements (standard mode only).")
    st_group.add_argument("--apply-patterns", action="store_true", default=False,
                        help="Apply detailed post-simplification patterns (UUIDs, URLs, etc.) AFTER block compression (standard mode only).")

    # Standard mode arguments with modified defaults
    st_group.add_argument("--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE_REC,
                        help="Min consecutive lines for pattern block compression (standard mode only).")
    st_group.add_argument("--min-line-length", type=int, default=DEFAULT_MIN_LINE_LENGTH_REC,
                        help="Min length for identical line minification (standard mode only).")
    st_group.add_argument("--min-repetitions", type=int, default=DEFAULT_MIN_REPETITIONS_REC,
                        help="Min repetitions for identical line minification (standard mode only).")
    st_group.add_argument("--large-literal-threshold", type=int, default=DEFAULT_LARGE_LITERAL_THRESHOLD,
                        help="Min lines in list/dict for literal compression (standard mode, skipped if pattern compression enabled).")

    # --- General Options ---
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging level.")

    args = parser.parse_args()

    # --- Setup ---
    log.setLevel(args.log_level.upper())
    logging.getLogger().setLevel(args.log_level.upper()) # Ensure root logger level is also set
    seen_content_hashes.clear() # Ensure fresh state for each run
    total_bytes_written = 0
    # <<< ADDED: Initialize report data dictionary >>>
    file_type_report_data: Dict[str, Counter] = {}


    target_folder_path = Path(args.folder_path)
    if not target_folder_path.is_dir():
        log.critical(f"Error: Folder not found or not a directory: '{args.folder_path}'")
        sys.exit(1)
    resolved_path = target_folder_path.resolve()

    # --- Print Config ---
    print("-" * 40, file=sys.stderr)
    print(f"Processing folder: {resolved_path}", file=sys.stderr)
    print(f"Output target: {'Stdout' if not args.output else args.output}", file=sys.stderr)
    print(f"Log Level: {args.log_level.upper()}", file=sys.stderr)
    print(f"Mode: {'RAW DUMP' if args.raw_dump else 'Standard (Compression/Simplification)'}", file=sys.stderr)

    if not args.raw_dump:
        # Determine effective skip values based on flags for standard mode
        effective_skip_empty = not args.keep_empty
        effective_skip_duplicates = not args.keep_duplicates # Default is now skip=True

        print(f"  Skip empty: {effective_skip_empty} ({'DEFAULT' if effective_skip_empty else 'Disabled (--keep-empty)'})", file=sys.stderr)
        print(f"  Skip duplicates: {effective_skip_duplicates} ({'DEFAULT' if effective_skip_duplicates else 'Disabled (--keep-duplicates)'})", file=sys.stderr)
        print(f"  Strip logging: {args.strip_logging} ({'Enabled' if args.strip_logging else 'Disabled (DEFAULT)'})", file=sys.stderr)
        print(f"  Pre-process split lines: {args.preprocess_split_lines} ({'DEFAULT' if args.preprocess_split_lines else 'Disabled (--no-preprocess-split-lines)'})", file=sys.stderr)
        print(f"  Compress pattern blocks: {args.compress_patterns} ({'DEFAULT' if args.compress_patterns else 'Disabled (--no-compress-patterns)'})", file=sys.stderr)
        if args.compress_patterns: print(f"    Min consecutive lines: {args.min_consecutive}", file=sys.stderr)
        print(f"  Apply detailed patterns: {args.apply_patterns} ({'Enabled' if args.apply_patterns else 'Disabled (DEFAULT)'})", file=sys.stderr)
        print(f"  Minify identical lines: {args.minify_lines} ({'DEFAULT' if args.minify_lines else 'Disabled (--no-minify-lines)'})", file=sys.stderr)
        if args.minify_lines: print(f"    Min line length: {args.min_line_length}, Min repetitions: {args.min_repetitions}", file=sys.stderr)
        print(f"  Post-process cleanup: {args.post_cleanup} ({'DEFAULT' if args.post_cleanup else 'Disabled (--no-post-cleanup)'})", file=sys.stderr)
        literal_compression_status = 'DISABLED (Pattern compression enabled)' if args.compress_patterns else f'ENABLED (threshold: {args.large_literal_threshold})'
        print(f"  Large literal compression: {literal_compression_status}", file=sys.stderr)
    else:
         print("  (Standard mode options are ignored in raw dump mode)", file=sys.stderr)

    print("-" * 40, file=sys.stderr)

    output_handle = None
    output_path = None
    buffer = io.StringIO()
    output_content_to_write = "" # Initialize

    try:
        # --- Header ---
        if args.raw_dump:
            header_lines = [
                f"# RAW DUMP of folder: {resolved_path}",
                f"# Generated by folder_to_text.py --raw-dump",
                "=" * 40, ""
            ]
        else:
            header_lines = [
                f"# Compressed Representation of: {resolved_path}",
                f"# Generated by folder_to_text.py (Standard Mode)",
                f"# Options Effective: skip_empty={effective_skip_empty}, skip_duplicates={effective_skip_duplicates}, preprocess_split={args.preprocess_split_lines}, compress_patterns={args.compress_patterns} (min={args.min_consecutive}), apply_patterns={args.apply_patterns}, minify_lines={args.minify_lines}, post_cleanup={args.post_cleanup}",
                "# (Defaults provide maximum reduction)",
                "=" * 40, ""
            ]
        for line in header_lines: buffer.write(line + "\n")

        # --- Main Processing Logic ---
        if args.raw_dump:
            # --- Raw Dump Mode ---
            log.info("Starting raw folder dump...")
            process_folder_raw(str(resolved_path), buffer)
            log.info("Finished raw folder dump.")
            buffer.seek(0)
            output_content_to_write = buffer.getvalue() # Get content directly

        else:
            # --- Standard Mode ---
            current_ignore_patterns = IGNORE_PATTERNS.copy(); current_ignore_patterns.update(args.ignore)
            current_code_extensions = CODE_EXTENSIONS.copy(); current_code_extensions.update(args.source_ext)
            current_interesting_files = INTERESTING_FILENAMES.copy(); current_interesting_files.update(args.interesting_files)

            # Step 1: Process Folder (Initial simplification, filtering)
            log.info("Starting standard folder processing (Step 1)...")
            process_folder(
                str(resolved_path), buffer, current_ignore_patterns, current_code_extensions,
                current_interesting_files,
                skip_empty=effective_skip_empty,
                strip_logging=args.strip_logging,
                skip_duplicates=effective_skip_duplicates,
                large_literal_threshold=args.large_literal_threshold,
                compress_patterns_enabled=args.compress_patterns,
                report_data=file_type_report_data # <<< ADDED: Pass the dictionary
            )
            log.info("Finished initial folder processing.")

            # Retrieve content for further steps
            buffer.seek(0)
            processed_content = buffer.getvalue()
            final_output_content = processed_content # Start with initial content
            num_lines_expanded = 0
            num_pattern_replacements = 0
            num_blocks_compressed = 0
            num_lines_minified = 0
            num_lines_cleaned_up = 0

            # Subsequent optional steps modify final_output_content

            # Step 2: Pre-process - Expand Multi-Pattern Lines
            if args.preprocess_split_lines and final_output_content.strip():
                log.info("Pre-processing: Expanding multi-pattern lines (Step 2)...")
                final_output_content, num_lines_expanded = expand_multi_pattern_lines(
                    final_output_content, SINGLE_VOICE_ID_FINDER, "VOICE_ID"
                )
                log.info("Finished expanding multi-pattern lines.")
            else:
                log.debug("Skipping Step 2: Pre-process split lines (disabled or empty content)")

            # Step 3: Compress Pattern Blocks
            if args.compress_patterns and final_output_content.strip():
                 log.info("Compressing consecutive pattern blocks (Step 3)...")
                 final_output_content, num_blocks_compressed = compress_pattern_blocks(
                     final_output_content, BLOCK_COMPRESSION_PATTERNS, args.min_consecutive
                 )
                 log.info("Finished compressing pattern blocks.")
            else:
                log.debug("Skipping Step 3: Compress pattern blocks (disabled or empty content)")

            # Step 4: Apply Post-Simplification Patterns
            if args.apply_patterns and final_output_content.strip():
                log.info("Applying detailed post-simplification patterns (Step 4)...")
                final_output_content, num_pattern_replacements = apply_post_simplification_patterns(
                    final_output_content, POST_SIMPLIFICATION_PATTERNS
                )
                log.info("Finished applying detailed post-simplification patterns.")
            else:
                log.debug("Skipping Step 4: Apply detailed patterns (disabled or empty content)")

            # Step 5: Minify Identical Lines
            if args.minify_lines and final_output_content.strip():
                log.info("Minifying repeated identical lines (Step 5)...")
                final_output_content, num_lines_minified = minify_repeated_lines(
                    final_output_content, args.min_line_length, args.min_repetitions
                )
                log.info("Finished minifying identical lines.")
            else:
                log.debug("Skipping Step 5: Minify identical lines (disabled or empty content)")

            # Step 6: Post-Process Cleanup
            if args.post_cleanup and final_output_content.strip():
                 log.info("Applying post-processing cleanup (Step 6)...")
                 final_output_content, num_lines_cleaned_up = post_process_cleanup(
                     final_output_content, PLACEHOLDER_CLEANUP_PATTERN
                 )
                 log.info("Finished post-processing cleanup.")
            else:
                 log.debug("Skipping Step 6: Post-process cleanup (disabled or empty content)")

            output_content_to_write = final_output_content # Final result after all steps


        # --- Write Final Output ---
        log.info("Writing final output...")
        if args.output:
            output_path = Path(args.output).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = open(output_path, 'w', encoding='utf-8')
            log.debug(f"Opening output file: {output_path}")
        else:
            output_handle = sys.stdout
            log.debug("Using stdout for output.")

        output_handle.write(output_content_to_write)
        total_bytes_written = len(output_content_to_write.encode('utf-8'))
        log.info("Finished writing output.")

        # --- Final Summary ---
        print("-" * 40, file=sys.stderr)
        print("Processing complete.", file=sys.stderr)
        print(f"Mode: {'RAW DUMP' if args.raw_dump else 'Standard'}", file=sys.stderr)

        if not args.raw_dump:
             # <<< ADDED: Detailed Post-Processing Summary >>>
             print("\n--- Overall Post-Processing Stage Summary ---", file=sys.stderr)
             if args.preprocess_split_lines: print(f"  Line expansion pre-processing expanded {num_lines_expanded} lines.", file=sys.stderr)
             else: print("  Line expansion pre-processing: Skipped/Disabled", file=sys.stderr)

             if args.compress_patterns: print(f"  Pattern block compression created {num_blocks_compressed} summary lines.", file=sys.stderr)
             else: print("  Pattern block compression: Skipped/Disabled", file=sys.stderr)

             if args.apply_patterns: print(f"  Detailed pattern application made {num_pattern_replacements} replacements.", file=sys.stderr)
             else: print("  Detailed pattern application: Skipped/Disabled", file=sys.stderr)

             if args.minify_lines: print(f"  Identical line minification replaced {num_lines_minified} line occurrences.", file=sys.stderr)
             else: print("  Identical line minification: Skipped/Disabled", file=sys.stderr)

             if args.post_cleanup: print(f"  Post-processing cleanup removed {num_lines_cleaned_up} lines.", file=sys.stderr)
             else: print("  Post-processing cleanup: Skipped/Disabled", file=sys.stderr)

             # <<< ADDED: File Type Processing Report >>>
             print("\n--- File Type Processing Report ---", file=sys.stderr)
             if not file_type_report_data:
                 print("  No source files were processed or met inclusion criteria.", file=sys.stderr)
             else:
                 # Sort by file type identifier for consistent output
                 sorted_file_ids = sorted(file_type_report_data.keys())
                 total_processed = 0
                 total_simplified = 0
                 total_skipped_empty = 0
                 total_skipped_duplicate = 0
                 total_contributed = 0
                 for file_id in sorted_file_ids:
                     stats = file_type_report_data[file_id]
                     processed_count = stats.get('processed', 0)
                     simplified_count = stats.get('simplified', 0)
                     skipped_empty_count = stats.get('skipped_empty', 0)
                     skipped_duplicate_count = stats.get('skipped_duplicate', 0)
                     contributed_count = stats.get('contributed', 0)

                     # Accumulate totals
                     total_processed += processed_count
                     total_simplified += simplified_count
                     total_skipped_empty += skipped_empty_count
                     total_skipped_duplicate += skipped_duplicate_count
                     total_contributed += contributed_count

                     print(f"  {file_id if file_id else '<no_ext>'}:", file=sys.stderr) # Handle empty extension case
                     print(f"    - Processed: {processed_count}", file=sys.stderr)
                     # Only show categories if they occurred for this type
                     if simplified_count > 0:
                         print(f"    - Simplified (Initial): {simplified_count}", file=sys.stderr)
                     if skipped_empty_count > 0:
                         print(f"    - Skipped (Empty): {skipped_empty_count}", file=sys.stderr)
                     if skipped_duplicate_count > 0:
                         print(f"    - Skipped (Duplicate): {skipped_duplicate_count}", file=sys.stderr)
                     if contributed_count > 0:
                          print(f"    - Contributed Content: {contributed_count}", file=sys.stderr)
                 # Print Totals
                 print("  ---", file=sys.stderr)
                 print(f"  TOTALS:", file=sys.stderr)
                 print(f"    - Processed: {total_processed}", file=sys.stderr)
                 print(f"    - Simplified (Initial): {total_simplified}", file=sys.stderr)
                 print(f"    - Skipped (Empty): {total_skipped_empty}", file=sys.stderr)
                 print(f"    - Skipped (Duplicate): {total_skipped_duplicate}", file=sys.stderr)
                 print(f"    - Contributed Content: {total_contributed}", file=sys.stderr)


        # <<< MODIFIED: Moved byte count output below reports >>>
        print("-" * 40, file=sys.stderr) # Separator before final size
        if args.output and output_path: print(f"Total bytes written to {output_path}: {total_bytes_written}", file=sys.stderr)
        else: print(f"Total bytes written to stdout: {total_bytes_written}", file=sys.stderr)
        print("-" * 40, file=sys.stderr)

    except IOError as e:
        log.critical(f"I/O Error: {e}")
        if args.output: log.critical(f"Failed operation likely involved file: {args.output}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.warning("Processing interrupted by user (Ctrl+C).")
        sys.exit(1)
    except Exception as e:
        log.critical(f"An unexpected error occurred: {e}", exc_info=(log.getEffectiveLevel() <= logging.DEBUG))
        sys.exit(1)
    finally:
        if buffer: buffer.close()
        if args.output and output_handle and output_handle is not sys.stdout:
            try:
                output_handle.close()
                log.debug(f"Closed output file: {output_path}")
            except Exception as e:
                log.error(f"Error closing output file '{args.output}': {e}")


if __name__ == "__main__":
    main()