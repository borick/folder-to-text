#!/usr/bin/env python3

# --- folder_to_text.py ---
# Processes a folder, simplifies text files, summarizes others,
# and optionally compresses repetitive patterns or lines, with a final cleanup step.

# --- Example Usage ---
# 1. Basic simplification:
#    python folder_to_text.py /path/to/project --skip-empty --skip-duplicates -o simple.txt
#
# 2. Recommended for Max Reduction (Blocks + Lines + Def Cleanup + Post Cleanup):
#    python folder_to_text.py /path/to/project --skip-empty --skip-duplicates --preprocess-split-lines --compress-patterns --min-consecutive 3 --minify-lines --min-line-length 40 --min-repetitions 2 --post-cleanup -o max_compressed_cleaned.txt
#
# 3. Using --apply-patterns with cleanup (Alternative Reduction):
#    python folder_to_text.py /path/to/project --skip-empty --skip-duplicates --apply-patterns --post-cleanup -o applied_cleaned.txt
#
# 4. Debugging post-cleanup:
#    python folder_to_text.py /path/to/project --apply-patterns --post-cleanup --log-level DEBUG -o debug_run.txt 2> debug.log

import os
import re
import argparse
import sys
import string
from pathlib import Path
from collections import Counter
import json
import logging
import hashlib
import io
import asyncio
import functools
import time
import datetime
from enum import Enum, auto
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(funcName)s: %(message)s', stream=sys.stderr)
log = logging.getLogger(__name__)


# --- Configuration ---
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
    (re.compile(r'\b\d{2,}\.\d{1,}|\d{1,}\.\d{2,}\b'), '*FLOAT*'),
    (re.compile(r'\b\d{3,}\b'), '*INT*'),
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
DEFAULT_MIN_CONSECUTIVE_LINES = 3

DEFINITION_SIMPLIFICATION_PATTERNS = [
    (re.compile(r'\b\d{2,}\.\d{1,}|\d{1,}\.\d{2,}\b'), '*FLOAT*'),
    (re.compile(r'\b\d{3,}\b'), '*INT*'),
    (re.compile(r"""(['"])((?:\\.|[^\\\1])*?)\1"""), '...'), # Use corrected pattern
    (re.compile(r'\b[a-fA-F0-9]{10,}\b'), '*HEX*'),
    (re.compile(r'[a-zA-Z0-9+/=]{20,}'), '*BASE64LIKE*'),
]

DEFAULT_LARGE_LITERAL_THRESHOLD = 10

# --- NEW: Placeholders targeted by --post-cleanup ---
# Combine all generated placeholder patterns here for the cleanup regex
ALL_PLACEHOLDERS = [
    r'\*...\*', r'\*INT\*', r'\*FLOAT\*', r'\*UUID\*',
    r'\*NUM_LONG\*', r'\*HEX_LONG\*', r'\*BASE64LIKE_LONG\*',
    r'\*HEX\*', r'\*BASE64LIKE\*', # From definition simplification
    r'\/\*PATH\*\/?' # Path placeholder (needs escaping)
    # Add any other custom placeholders you might introduce
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
# --- End NEW ---


# --- Globals ---
seen_content_hashes = set()

# --- Helper Functions ---
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

# --- Core Processing Functions ---

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
            in_literal_block = True
            literal_block_start_index = current_line_index
            literal_line_count = 0
            simplified_lines.append(original_line)
        elif in_literal_block:
            is_simple_literal_line = literal_line_pattern.match(stripped_line) is not None
            if is_simple_literal_line:
                literal_line_count += 1
                simplified_lines.append(original_line)
            elif is_end:
                simplified_lines.append(original_line)
                if not disable_literal_compression and literal_line_count >= large_literal_threshold and literal_block_start_index >= 0:
                    start_slice_idx = literal_block_start_index + 1
                    end_slice_idx = start_slice_idx + literal_line_count
                    if start_slice_idx < end_slice_idx <= len(simplified_lines)-1:
                         indent_level = line.find(stripped_line[0]) if stripped_line else (literal_block_start_index+1)*2
                         indent = " " * (indent_level + 2)
                         placeholder_line = f"{indent}# --- Large literal collection compressed ({literal_line_count} lines) ---"
                         del simplified_lines[start_slice_idx:end_slice_idx]
                         simplified_lines.insert(start_slice_idx, placeholder_line)
                         log.debug(f"Compressed literal block, {literal_line_count} lines.")
                    else:
                         log.warning(f"Compression slice calculation error: start={start_slice_idx}, end={end_slice_idx}, len={len(simplified_lines)}. Skipping literal compression.")
                in_literal_block = False; literal_block_start_index = -1; literal_line_count = 0
            else:
                log.debug(f"Complex line ended potential literal block compression.")
                simplified_lines.append(original_line)
                in_literal_block = False; literal_block_start_index = -1; literal_line_count = 0
        elif stripped_line:
            simplified_lines.append(line)
        elif simplified_lines and simplified_lines[-1].strip():
            simplified_lines.append("")

    processed_content = "\n".join(simplified_lines).strip()
    if processed_content: processed_content += "\n"
    for pattern, replacement in SIMPLIFICATION_PATTERNS:
        processed_content = pattern.sub(replacement, processed_content)
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
    compress_patterns_enabled: bool
):
    global seen_content_hashes
    target_path = Path(target_folder).resolve()
    def write_to_buffer(text): buffer.write(text + "\n")

    for dirpath, dirnames, filenames in os.walk(target_path, topdown=True, onerror=lambda e: log.warning(f"Cannot access {e.filename} - {e}")):
        current_path = Path(dirpath)
        try: current_rel_path = current_path.relative_to(target_path)
        except ValueError:
            log.warning(f"Could not determine relative path for {current_path} against base {target_path}. Skipping directory.")
            dirnames[:] = []; continue

        dirnames[:] = [ d for d in dirnames if d not in ignore_set and not d.startswith('.') and not any(Path(d).match(p) for p in ignore_set if '*' in p or '?' in p) ]
        filenames.sort(); dirnames.sort()
        source_files_to_process = []
        other_text_filenames = []

        for filename in filenames:
            if filename in ignore_set or any(Path(filename).match(p) for p in ignore_set if '*' in p or '?' in p): continue
            if filename.startswith('.') and filename.lower() not in interesting_filenames_set and filename not in code_extensions_set: continue
            file_path = current_path / filename
            if not file_path.is_file(): continue
            if is_likely_binary(file_path): log.debug(f"Skipping likely binary file: {file_path.relative_to(target_path)}"); continue
            base, ext = os.path.splitext(filename); ext_lower = ext.lower(); fname_lower = filename.lower()
            is_source = (ext_lower in code_extensions_set or filename in code_extensions_set or fname_lower in code_extensions_set)
            relative_file_path = file_path.relative_to(target_path)
            if is_source: source_files_to_process.append((relative_file_path, file_path, ext_lower))
            else: other_text_filenames.append(filename)

        if source_files_to_process:
            dir_header = f"\n{'=' * 10} Directory: {current_rel_path} {'=' * 10}\n" if str(current_rel_path) != '.' else ""
            if dir_header: write_to_buffer(dir_header)
            for rel_path, full_path, file_ext in source_files_to_process:
                file_info_prefix = f"--- File: {rel_path}"
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()

                    simplified_content = simplify_source_code(
                        content,
                        strip_logging,
                        large_literal_threshold,
                        disable_literal_compression=compress_patterns_enabled
                    )

                    is_empty_after_simplification = not simplified_content.strip()
                    content_hash = None; is_duplicate = False
                    if not is_empty_after_simplification:
                        content_hash = hashlib.sha256(simplified_content.encode('utf-8')).hexdigest()
                        if skip_duplicates:
                            if content_hash in seen_content_hashes: is_duplicate = True
                    if is_empty_after_simplification and skip_empty: log.info(f"Skipping empty file (after simplification): {rel_path}"); continue
                    if is_duplicate: log.info(f"Skipping duplicate content file: {rel_path}"); continue
                    if content_hash and not is_duplicate and skip_duplicates: seen_content_hashes.add(content_hash)
                    write_to_buffer(file_info_prefix + " ---")
                    if is_empty_after_simplification: write_to_buffer("# (File is empty or contained only comments/whitespace/logging)")
                    else: write_to_buffer(simplified_content.strip())
                    write_to_buffer(f"--- End File: {rel_path} ---"); write_to_buffer("")
                except OSError as e:
                    log.error(f"Error reading file {rel_path}: {e}")
                    write_to_buffer(file_info_prefix + " Error Reading ---"); write_to_buffer(f"### Error reading file: {type(e).__name__}: {e} ###"); write_to_buffer(f"--- End File: {rel_path} ---"); write_to_buffer("")
                except Exception as e:
                    log.error(f"Error processing file {rel_path}: {e}", exc_info=True)
                    write_to_buffer(file_info_prefix + " Error Processing ---"); write_to_buffer(f"### Error processing file: {type(e).__name__}: {e} ###"); write_to_buffer(f"--- End File: {rel_path} ---"); write_to_buffer("")

        if other_text_filenames:
            summary = summarize_other_files(other_text_filenames, code_extensions_set, interesting_filenames_set)
            if summary:
                indent_level = len(current_rel_path.parts)
                indent = "  " * indent_level if indent_level > 0 else ""
                summary_line = f"{indent}# Other files in '{current_rel_path}': {summary}"
                log.debug(f"Adding summary for {current_rel_path}: {summary}")
                buffer.seek(0, io.SEEK_END)
                if buffer.tell() > 0: write_to_buffer(summary_line); write_to_buffer("")
                else: log.debug(f"Skipping summary for {current_rel_path} as buffer is empty.")


def apply_post_simplification_patterns(content: str, patterns: list[tuple[re.Pattern, Any]]) -> tuple[str, int]:
    """
    Applies a list of regex patterns and replacements to the content.
    Handles both string and lambda replacements.
    """
    total_replacements = 0; modified_content = content
    lines = content.splitlines(keepends=True)
    output_lines = []
    for line in lines:
        modified_line = line
        for pattern, replacement in patterns:
            try:
                 if callable(replacement):
                     modified_line_new, count = pattern.subn(replacement, modified_line)
                 else:
                     modified_line_new, count = pattern.subn(replacement, modified_line)
                 if count > 0:
                     total_replacements += count
                     modified_line = modified_line_new
            except Exception as e:
                log.error(f"Error applying post-simplification pattern {pattern.pattern} to line: {e}")
        output_lines.append(modified_line)

    modified_content = "".join(output_lines)
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
    """
    lines = content.splitlines(keepends=False)
    output_lines = []
    lines_expanded = 0
    log.debug(f"--- expand_multi_pattern_lines: Starting scan for '{pattern_name_for_log}' ---")

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith(("#", "---", "##", "==", "*LINE_REF_")):
            output_lines.append(line + "\n")
            continue

        try:
            matches = finder_pattern.findall(line)
        except Exception as e:
            log.error(f"Regex error finding patterns on line {i+1}: {e}")
            matches = []

        if len(matches) > 1:
             combined_match_len = sum(len(m.strip(',').strip()) for m in matches)
             stripped_len_approx = len(re.sub(r'\s+|,', '', stripped_line))

             if combined_match_len >= stripped_len_approx * 0.8:
                log.info(f"Expanding line {i+1} containing {len(matches)} instances of '{pattern_name_for_log}'.")
                lines_expanded += 1
                indent_level = line.find(stripped_line[0]) if stripped_line else 0
                indent = " " * indent_level
                for match in matches:
                    output_lines.append(f"{indent}{match}\n")
                continue

        output_lines.append(line + "\n")

    log.info(f"Finished line expansion pre-processing. Expanded {lines_expanded} lines containing multiple '{pattern_name_for_log}' instances.")
    return "".join(output_lines), lines_expanded


def compress_pattern_blocks(content: str, patterns_to_compress: dict[str, re.Pattern], min_consecutive: int) -> tuple[str, int]:
    """
    Scans content for consecutive lines matching predefined patterns and compresses them.
    """
    lines = content.splitlines(keepends=True)
    output_lines = []
    total_blocks_compressed = 0
    i = 0
    while i < len(lines):
        current_line = lines[i]
        stripped_line = current_line.strip()
        matched_pattern_name = None

        is_ignorable = (
            stripped_line.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or
            not stripped_line
        )

        if not is_ignorable:
            for name, pattern in patterns_to_compress.items():
                if pattern.match(stripped_line):
                    matched_pattern_name = name
                    break

        if matched_pattern_name:
            block_pattern_name = matched_pattern_name
            block_start_index = i
            block_lines_indices = [i]
            j = i + 1
            while j < len(lines):
                next_line_raw = lines[j]
                next_stripped = next_line_raw.strip()
                if next_stripped and patterns_to_compress[block_pattern_name].match(next_stripped):
                    block_lines_indices.append(j)
                    j += 1
                else:
                    break

            block_count = len(block_lines_indices)
            if block_count >= min_consecutive:
                first_line_in_block = lines[block_start_index]
                indent = ""
                first_line_stripped = first_line_in_block.strip()
                if len(first_line_in_block) > len(first_line_stripped):
                     indent = first_line_in_block[:len(first_line_in_block) - len(first_line_stripped)]
                summary_line = f"{indent}## [Compressed Block: {block_count} lines matching pattern '{block_pattern_name}'] ##\n"
                output_lines.append(summary_line)
                log.info(f"Compressed {block_count} lines (Indices {block_start_index}-{j-1}) matching '{block_pattern_name}'.")
                total_blocks_compressed += 1
                i = j
            else:
                for block_line_index in block_lines_indices:
                    output_lines.append(lines[block_line_index])
                i = j
        else:
            output_lines.append(current_line)
            i += 1

    log.info(f"Pattern block compression: Compressed {total_blocks_compressed} blocks of lines (min consecutive: {min_consecutive}).")
    return "".join(output_lines), total_blocks_compressed


def minify_repeated_lines(content: str, min_length: int, min_repetitions: int) -> tuple[str, int]:
    """
    Identifies repeated identical long lines, replaces them with placeholders,
    and simplifies the content within the definition block.
    """
    global DEFINITION_SIMPLIFICATION_PATTERNS

    lines = content.splitlines(keepends=True); line_counts = Counter(); num_replaced = 0; definitions = {}; placeholder_template = "*LINE_REF_{}*"
    meaningful_lines_indices = {}
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        is_structural_or_placeholder = ( stripped_line.startswith(("--- File:", "--- End File:", "#", "===", "*LINE_REF_", "## [Compressed Block:")) or re.match(r'^\*VOICE:.*\*$', stripped_line) or re.match(r'^\*UUID\*$', stripped_line) or re.match(r'^\*...\*$', stripped_line) or not stripped_line )
        if len(stripped_line) >= min_length and not is_structural_or_placeholder:
            line_content_key = line; line_counts[line_content_key] += 1
            if line_content_key not in meaningful_lines_indices: meaningful_lines_indices[line_content_key] = []
            meaningful_lines_indices[line_content_key].append(i)

    replacement_map = {}; definition_lines = []; placeholder_counter = 1
    repeated_lines = sorted([(line, count) for line, count in line_counts.items() if count >= min_repetitions], key=lambda item: (-item[1], item[0]))

    for line, count in repeated_lines:
        if line not in replacement_map:
            placeholder = placeholder_template.format(placeholder_counter)
            replacement_map[line] = placeholder

            simplified_definition_content = line.rstrip()
            for pattern, replacement in DEFINITION_SIMPLIFICATION_PATTERNS:
                 try:
                     if callable(replacement):
                          simplified_definition_content, _ = pattern.subn(replacement, simplified_definition_content)
                     else:
                          simplified_definition_content = pattern.sub(replacement, simplified_definition_content)
                 except Exception as e:
                      log.warning(f"Error applying definition simplification pattern {pattern.pattern}: {e}")

            definition_lines.append(f"{placeholder} = {simplified_definition_content}")
            placeholder_counter += 1

    if not replacement_map: log.info("Line minification enabled, but no lines met criteria."); return content, 0
    new_lines = lines[:]; replaced_indices = set(); num_actual_replacements = 0
    for original_line, placeholder in replacement_map.items():
        indices_to_replace = meaningful_lines_indices.get(original_line, []); replacements_done_for_this_line = 0
        for index in indices_to_replace:
            if index not in replaced_indices:
                new_lines[index] = placeholder + ("\n" if original_line.endswith("\n") else "")
                replaced_indices.add(index); replacements_done_for_this_line += 1
        num_actual_replacements += replacements_done_for_this_line
    minified_content = "".join(new_lines)
    if definition_lines:
        definition_header = ["", "=" * 40, f"# Line Minification Definitions ({len(definition_lines)}):", f"# (Lines >= {min_length} chars repeated >= {min_repetitions} times, content simplified)", "=" * 40]
        definition_block = "\n".join(definition_header + definition_lines) + "\n\n"
        log.info(f"Line minification: Replaced {num_actual_replacements} occurrences of {len(definition_lines)} unique long lines. Definition content simplified.")
        return definition_block + minified_content, num_actual_replacements
    else:
        return content, 0

# --- NEW Post-Processing Cleanup Function ---
def post_process_cleanup(content: str, cleanup_pattern: re.Pattern) -> tuple[str, int]:
    """
    Removes lines consisting primarily of placeholders and structural chars.
    """
    lines = content.splitlines(keepends=True)
    output_lines = []
    lines_removed = 0
    log.debug(f"--- post_process_cleanup: Starting final cleanup ---")

    for i, line in enumerate(lines):
        # Don't clean up definition block or structural markers
        if line.startswith("*LINE_REF_") or line.startswith("## [Compressed Block:") or line.startswith("--- File:") or line.startswith("--- End File:") or line.startswith("==="):
            output_lines.append(line)
            continue

        # Check if the line matches the cleanup pattern
        if cleanup_pattern.match(line):
            log.debug(f"Post-cleanup removing line {i+1}: {line.strip()[:80]}")
            # Option 1: Remove the line completely
            # output_lines.append("") # Keep blank line to avoid collapsing too much? Or just skip? Let's skip.
            # Option 2: Replace with a marker
            indent = len(line) - len(line.lstrip(' '))
            # output_lines.append(" " * indent + "# [...]\n") # Or don't!
            lines_removed += 1
        else:
            output_lines.append(line) # Keep the line

    log.info(f"Post-processing cleanup removed {lines_removed} lines containing only placeholders/structure.")
    # Remove potentially consecutive blank lines created by removal
    cleaned_content = "".join(output_lines)
    cleaned_content = re.sub(r'(\n\s*# \[...\])+\n', '\n# [...]\n', cleaned_content) # Consolidate markers
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content) # General multi-blank line cleanup
    return cleaned_content, lines_removed
# --- End NEW Function ---


# --- Main Function ---
def main():
    """ Main execution function """
    global IGNORE_PATTERNS, CODE_EXTENSIONS, INTERESTING_FILENAMES, POST_SIMPLIFICATION_PATTERNS, BLOCK_COMPRESSION_PATTERNS, SINGLE_VOICE_ID_FINDER, DEFINITION_SIMPLIFICATION_PATTERNS, PLACEHOLDER_CLEANUP_PATTERN, seen_content_hashes, DEFAULT_LARGE_LITERAL_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Generate a compressed representation of a folder's text content, with pattern-based block compression and optional line minification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("folder_path", help="Path to the target folder.")
    parser.add_argument("-o", "--output", help="Output file path (optional, defaults to stdout).")
    parser.add_argument("--ignore", nargs='+', default=[], help="Additional names/patterns to ignore.")
    parser.add_argument("--source-ext", nargs='+', default=[], help="Additional extensions/filenames for source code.")
    parser.add_argument("--interesting-files", nargs='+', default=[], help="Additional notable filenames for summaries.")
    parser.add_argument("--skip-empty", action="store_true", help="Skip files empty after simplification.")
    parser.add_argument("--strip-logging", action="store_true", help="Attempt to remove common logging statements.")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip printing files with identical simplified content.")
    parser.add_argument("--large-literal-threshold", type=int, default=DEFAULT_LARGE_LITERAL_THRESHOLD, help="Min lines in list/dict for literal compression (skipped if --compress-patterns).")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level.")
    parser.add_argument("--preprocess-split-lines", action="store_true", help="Pre-process step: Split lines with multiple known patterns (e.g., voice IDs) into single lines.")
    parser.add_argument("--compress-patterns", action="store_true", help="Enable compression of consecutive lines matching predefined patterns (like VOICE_ID). Disables large literal list compression.")
    parser.add_argument("--min-consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE_LINES, help="Minimum number of consecutive lines matching a pattern to compress into a block.")
    parser.add_argument("--apply-patterns", action="store_true", help="Apply detailed post-simplification patterns (like UUIDs, URLs, generic strings) AFTER block compression.")
    parser.add_argument("--minify-lines", action="store_true", help="Enable repeated identical long line minification (runs AFTER other compression steps).")
    parser.add_argument("--min-line-length", type=int, default=50, help="Minimum length for identical line minification.")
    parser.add_argument("--min-repetitions", type=int, default=3, help="Minimum repetitions for identical line minification.")
    # --- NEW Argument ---
    parser.add_argument("--post-cleanup", action="store_true", help="Final step: Remove lines consisting only of placeholders and structure chars.")
    # --- End NEW ---

    args = parser.parse_args()

    # --- Setup ---
    log.setLevel(args.log_level.upper())
    logging.getLogger().setLevel(args.log_level.upper())
    current_ignore_patterns = IGNORE_PATTERNS.copy(); current_ignore_patterns.update(args.ignore)
    current_code_extensions = CODE_EXTENSIONS.copy(); current_code_extensions.update(args.source_ext)
    current_interesting_files = INTERESTING_FILENAMES.copy(); current_interesting_files.update(args.interesting_files)
    seen_content_hashes.clear()
    total_bytes_written = 0

    target_folder_path = Path(args.folder_path)
    if not target_folder_path.is_dir(): log.critical(f"Error: '{args.folder_path}' is not valid."); sys.exit(1)
    resolved_path = target_folder_path.resolve()

    # --- Print Config ---
    print("-" * 40, file=sys.stderr)
    print(f"Processing folder: {resolved_path}", file=sys.stderr)
    print(f"Output target: {'Stdout' if not args.output else args.output}", file=sys.stderr)
    print(f"Skip empty: {args.skip_empty}, Strip logging: {args.strip_logging}, Skip duplicates: {args.skip_duplicates}", file=sys.stderr)
    print(f"Pre-process split lines: {args.preprocess_split_lines}", file=sys.stderr)
    print(f"Compress pattern blocks: {args.compress_patterns}", file=sys.stderr)
    if args.compress_patterns: print(f"  Min consecutive lines: {args.min_consecutive}", file=sys.stderr)
    print(f"Apply detailed patterns: {args.apply_patterns}", file=sys.stderr)
    print(f"Minify identical lines: {args.minify_lines}", file=sys.stderr)
    if args.minify_lines: print(f"  Min line length: {args.min_line_length}, Min repetitions: {args.min_repetitions}", file=sys.stderr)
    print(f"Post-process cleanup: {args.post_cleanup}", file=sys.stderr) # New
    print(f"Large literal compression: {'DISABLED by --compress-patterns' if args.compress_patterns else f'ENABLED (threshold: {args.large_literal_threshold})'}", file=sys.stderr)
    print(f"Log Level: {args.log_level.upper()}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    output_handle = None
    output_path = None
    buffer = io.StringIO()

    try:
        # --- Header ---
        header_lines = [
            f"# Compressed Representation of: {resolved_path}",
            f"# Generated by folder_to_text.py",
            f"# Options: preprocess_split={args.preprocess_split_lines}, apply_patterns={args.apply_patterns}, compress_patterns={args.compress_patterns} (min={args.min_consecutive}), minify_lines={args.minify_lines}, post_cleanup={args.post_cleanup}", # Added post_cleanup
            "=" * 40, ""
        ]
        for line in header_lines: buffer.write(line + "\n")

        # --- Step 1: Process Folder ---
        log.info("Starting folder processing (Step 1)...")
        process_folder(
            str(resolved_path), buffer, current_ignore_patterns, current_code_extensions,
            current_interesting_files, args.skip_empty, args.strip_logging,
            args.skip_duplicates, args.large_literal_threshold, args.compress_patterns
        )
        log.info("Finished folder processing.")

        buffer.seek(0); processed_content = buffer.getvalue()
        final_output_content = processed_content
        num_lines_expanded = 0
        num_pattern_replacements = 0
        num_blocks_compressed = 0
        num_lines_minified = 0
        num_lines_cleaned_up = 0 # New counter

        # --- Step 2: Pre-process - Expand Multi-Pattern Lines ---
        if args.preprocess_split_lines and final_output_content.strip():
            log.info("Pre-processing: Expanding multi-pattern lines (Step 2)...")
            final_output_content, num_lines_expanded = expand_multi_pattern_lines(
                final_output_content, SINGLE_VOICE_ID_FINDER, "VOICE_ID"
            )
            log.info("Finished expanding multi-pattern lines.")

        # --- Step 3: Compress Pattern Blocks ---
        if args.compress_patterns and final_output_content.strip():
             log.info("Compressing consecutive pattern blocks (Step 3)...")
             final_output_content, num_blocks_compressed = compress_pattern_blocks(
                 final_output_content, BLOCK_COMPRESSION_PATTERNS, args.min_consecutive
             )
             log.info("Finished compressing pattern blocks.")

        # --- Step 4: Apply Post-Simplification Patterns ---
        if args.apply_patterns and final_output_content.strip():
            log.info("Applying post-simplification patterns (Step 4)...")
            final_output_content, num_pattern_replacements = apply_post_simplification_patterns(
                final_output_content, POST_SIMPLIFICATION_PATTERNS
            )
            log.info("Finished applying post-simplification patterns.")

        # --- Step 5: Minify Identical Lines ---
        if args.minify_lines and final_output_content.strip():
            log.info("Minifying repeated identical lines (Step 5)...")
            final_output_content, num_lines_minified = minify_repeated_lines(
                final_output_content, args.min_line_length, args.min_repetitions
            )
            log.info("Finished minifying identical lines.")

        # --- Step 6: Post-Process Cleanup --- NEW STEP ---
        if args.post_cleanup and final_output_content.strip():
             log.info("Applying post-processing cleanup (Step 6)...")
             final_output_content, num_lines_cleaned_up = post_process_cleanup(
                 final_output_content, PLACEHOLDER_CLEANUP_PATTERN
             )
             log.info("Finished post-processing cleanup.")
        # --- End NEW STEP ---

        # --- Step 7: Write Final Output --- (Was Step 6)
        log.info("Writing final output (Step 7)...")
        if args.output:
            output_path = Path(args.output).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = open(output_path, 'w', encoding='utf-8')
        else:
            output_handle = sys.stdout
        output_handle.write(final_output_content)
        total_bytes_written = len(final_output_content.encode('utf-8'))
        log.info("Finished writing output.")

        # --- Final Summary ---
        print("-" * 40, file=sys.stderr)
        print("Processing complete.", file=sys.stderr)
        if args.preprocess_split_lines: print(f"Line expansion pre-processing expanded {num_lines_expanded} lines.", file=sys.stderr)
        if args.compress_patterns: print(f"Pattern block compression created {num_blocks_compressed} summary lines.", file=sys.stderr)
        if args.apply_patterns: print(f"Detailed pattern application made {num_pattern_replacements} replacements.", file=sys.stderr)
        if args.minify_lines: print(f"Identical line minification replaced {num_lines_minified} line occurrences (definition content simplified).", file=sys.stderr)
        if args.post_cleanup: print(f"Post-processing cleanup removed/marked {num_lines_cleaned_up} lines.", file=sys.stderr) # New
        if args.output and output_path: print(f"Total bytes written to {output_path}: {total_bytes_written}", file=sys.stderr)
        else: print(f"Total bytes written to stdout: {total_bytes_written}", file=sys.stderr)

    except IOError as e: log.critical(f"Error writing to output file '{args.output}': {e}"); sys.exit(1)
    except Exception as e: log.critical(f"An unexpected error occurred: {e}", exc_info=True); sys.exit(1)
    finally:
        if buffer: buffer.close()
        if args.output and output_handle and output_handle is not sys.stdout:
            try: output_handle.close()
            except Exception as e: log.error(f"Error closing output file '{args.output}': {e}")


if __name__ == "__main__":
    main()