# Folder to Text: Advanced Text Representation Generator

This Python script (`folder_to_text.py`) processes a folder containing source code and other text files. It simplifies their content, summarizes non-source files, and applies various optional compression and cleanup techniques to generate a concise, single-text representation of the folder's relevant content. This is particularly useful for tasks like providing context to Large Language Models (LLMs).

## Key Features

*   **Recursive Folder Traversal:** Scans the specified folder and its subdirectories.

*   **Intelligent Filtering:**
    *   Ignores common unnecessary directories (`.git`, `node_modules`, `venv`, `build`, etc.).
    *   Ignores binary files and common non-text file types.
    *   Allows custom ignore patterns.

*   **Content Simplification:**
    *   Removes comments (various styles: `/* */`, `//`, `#`, `""" """`, etc.).
    *   Removes blank lines.
    *   Obfuscates common patterns like long numbers, hex strings, base64-like strings, floats, and integers using placeholders (`*NUM_LONG*`, `*HEX_LONG*`, `*FLOAT*`, etc.).

*   **Summarization:** Summarizes non-source-code text files found in each directory (e.g., `5x .txt, 2x .csv, README.md`).

*   **Optional Processing Steps:**
    *   **Literal Compression:** Compresses large list/dictionary literals in code (can be disabled).
    *   **Logging Removal:** Attempts to remove common logging/print statements.
    *   **Empty/Duplicate Skipping:** Option to exclude files that become empty after simplification or have identical content to previously processed files.
    *   **Line Expansion Pre-processing:** Splits lines containing multiple instances of certain patterns (like cloud Voice IDs) into individual lines for better downstream processing.
    *   **Pattern Block Compression:** Compresses consecutive lines matching specific, predefined patterns (e.g., lists of Voice IDs, UUIDs) into a summary line like `## [Compressed Block: 15 lines matching pattern 'VOICE_ID'] ##`.
    *   **Detailed Pattern Application:** Applies more specific regex replacements after initial simplification (e.g., replacing quoted UUIDs with `"*UUID*"`, simplifying URLs, shortening long generic strings).
    *   **Repeated Line Minification:** Identifies identical long lines repeated multiple times across the entire output, replaces them with placeholders (`*LINE_REF_1*`), and adds a definition block at the start explaining the placeholders with simplified content.
    *   **Post-Processing Cleanup:** Removes lines that *only* contain placeholders (like `*INT*`, `*UUID*`), commas, brackets, or whitespace after all other steps, reducing noise.

*   **Output:** Writes the final combined text representation to standard output or a specified file.

*   **Configurable Logging:** Allows setting the log level (DEBUG, INFO, WARNING, etc.) for detailed insights into the process.

## Usage

    python folder_to_text.py <folder_path> [options]

---

**4. Arguments (Sub-section of Usage)**

### Arguments

*   `folder_path`: (Required) The path to the directory you want to process.

*   `-o, --output <filepath>`: Path to the output file. If omitted, output goes to standard output (stdout).

*   `--ignore <pattern1> [<pattern2> ...]` : Additional file or directory names/patterns to ignore (e.g., `*.log`, `temp_folder`). Uses standard glob patterns for wildcards.

*   `--source-ext <ext1> [<ext2> ...]` : Additional file extensions or full filenames to treat as source code (e.g., `.my_lang`, `CustomBuildScript`).

*   `--interesting-files <name1> [<name2> ...]` : Additional filenames (without extension) to explicitly list in summaries even if they aren't source code (e.g., `CONFIG`, `DEPLOY_NOTES`).

*   `--skip-empty`: If set, files that are empty *after* comment/whitespace removal (and potentially logging removal) are skipped entirely.

*   `--strip-logging`: If set, attempts to remove lines matching common logging/printing patterns (e.g., `log.info(...)`, `console.log(...)`, `print(...)`). Use with caution, as it might remove intended output.

*   `--skip-duplicates`: If set, calculates a hash of the simplified content of each file. If an identical hash is encountered later, that file is skipped.

*   `--large-literal-threshold <int>`: Minimum number of lines within a list (`[...]`) or dictionary (`{...}`) declaration to trigger compression into a placeholder comment. Default: `10`. **Note:** This is automatically disabled if `--compress-patterns` is used.

*   `--log-level <LEVEL>`: Set the logging verbosity. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Default: `WARNING`. Use `DEBUG` for detailed step-by-step information, often useful for diagnosing issues. Output goes to standard error (stderr).

*   `--preprocess-split-lines`: **(Compression Step)** Before other compression, splits lines containing multiple identifiable patterns (currently configured for cloud Voice IDs like `"en-US-Standard-A Neural"`) onto separate lines, improving the effectiveness of block compression.

*   `--compress-patterns`: **(Compression Step)** Enables the compression of blocks of consecutive lines that match predefined patterns (see `BLOCK_COMPRESSION_PATTERNS` in the script, e.g., Voice IDs, quoted UUIDs). Replaces the block with a summary line. This disables the `--large-literal-threshold` feature.

*   `--min-consecutive <int>`: Minimum number of consecutive lines matching a pattern required to trigger block compression when `--compress-patterns` is enabled. Default: `3`.

*   `--apply-patterns`: **(Compression Step)** Applies more detailed regex substitutions *after* initial simplification and potential block compression (e.g., `"*UUID*"`, URL simplification, string shortening). See `POST_SIMPLIFICATION_PATTERNS`.

*   `--minify-lines`: **(Compression Step)** Enables identification and replacement of identical lines that are longer than `--min-line-length` and occur at least `--min-repetitions` times. Creates placeholders (`*LINE_REF_N*`) and a definition block at the beginning. Runs *after* other compression steps.

*   `--min-line-length <int>`: Minimum character length for a line to be considered for minification via `--minify-lines`. Default: `50`.

*   `--min-repetitions <int>`: Minimum number of times an identical line must appear to be minified via `--minify-lines`. Default: `3`.

*   `--post-cleanup`: **(Final Step)** After all other processing, removes lines that consist *only* of placeholders (like `*INT*`, `*UUID*`, `*...*`), commas, brackets/braces, and whitespace. Helps clean up artifacts from aggressive simplification/compression.

### Examples

1.  **Basic Simplification:** Process a project, skip empty/duplicate files, and output to `simple.txt`.

    python folder_to_text.py /path/to/project --skip-empty --skip-duplicates -o simple.txt


2.  **Recommended for Max Reduction:** Apply block compression, line minification, definition cleanup, and post-cleanup. Good for maximizing compression for LLM context.

    python folder_to_text.py /path/to/project \
        --skip-empty \
        --skip-duplicates \
        --preprocess-split-lines \
        --compress-patterns --min-consecutive 3 \
        --minify-lines --min-line-length 40 --min-repetitions 2 \
        --post-cleanup \
        -o max_compressed_cleaned.txt

3.  **Alternative Reduction (Apply Patterns):** Use detailed pattern application instead of block/line compression, followed by cleanup. Might be better if preserving line structure is more important than block compression.

    python folder_to_text.py /path/to/project \
        --skip-empty \
        --skip-duplicates \
        --apply-patterns \
        --post-cleanup \
        -o applied_cleaned.txt

4.  **Debugging Post-Cleanup:** Run with pattern application and post-cleanup, enabling DEBUG logging to see exactly which lines are being removed by the cleanup step. Log messages go to stderr, redirect stderr to a file (`debug.log`).

    python folder_to_text.py /path/to/project \
        --apply-patterns \
        --post-cleanup \
        --log-level DEBUG \
        -o debug_run.txt 2> debug.log

## How it Works (Processing Pipeline)

The script processes the folder in several stages:

1.  **Initialization:** Parses arguments, sets up configuration (ignore lists, extensions, etc.), and prepares the output buffer.
2.  **Folder Traversal (`process_folder`):**
    *   Recursively walks the target directory.
    *   Filters out ignored directories and files based on patterns.
    *   Checks if files are likely binary and skips them.
    *   Sorts files and directories for deterministic output.
    *   Separates files into "source" (based on `CODE_EXTENSIONS`) and "other".
3.  **Source File Processing (`simplify_source_code` within `process_folder`):**
    *   Reads the file content.
    *   Removes comments (`/* */`, `//`, `#`, etc.).
    *   Removes docstrings (`""" ""`, `''' '''`).
    *   (Optional) Removes logging lines (`--strip-logging`).
    *   Applies basic simplification patterns (`SIMPLIFICATION_PATTERNS` - numbers, hex, base64).
    *   (Optional, unless `--compress-patterns`) Compresses large list/dict literals (`--large-literal-threshold`).
    *   Removes excess blank lines.
    *   (Optional) Checks for emptiness (`--skip-empty`) or duplicates (`--skip-duplicates`) based on simplified content hash.
    *   Writes file headers (`--- File: ... ---`), simplified content, and footers to the buffer.
4.  **Other File Summarization (`summarize_other_files` within `process_folder`):**
    *   For each directory, aggregates counts of non-source file extensions and lists specified "interesting" files.
    *   Writes a summary line (e.g., `# Other files in 'src/utils': 3x .json, 1x .yaml, CONFIG`) to the buffer.
5.  **(Optional) Pre-process Split Lines (`expand_multi_pattern_lines`):** If `--preprocess-split-lines`, scans the buffered content and splits lines with multiple target patterns (e.g., Voice IDs) onto new lines.
6.  **(Optional) Compress Pattern Blocks (`compress_pattern_blocks`):** If `--compress-patterns`, scans the buffer for consecutive lines matching defined patterns and replaces them with summary markers.
7.  **(Optional) Apply Detailed Patterns (`apply_post_simplification_patterns`):** If `--apply-patterns`, applies the `POST_SIMPLIFICATION_PATTERNS` regex list to the entire buffer for more fine-grained replacements (UUIDs, URLs, etc.).
8.  **(Optional) Minify Repeated Lines (`minify_repeated_lines`):** If `--minify-lines`, scans the buffer for frequently repeated long lines, replaces them with placeholders, and prepends a definition block.
9.  **(Optional) Post-Process Cleanup (`post_process_cleanup`):** If `--post-cleanup`, scans the buffer one last time and removes lines consisting only of placeholders, commas, brackets, and whitespace.
10. **Final Output:** Writes the fully processed content from the buffer to the specified output file or stdout.

## Configuration (Internal)

The script uses several internal constants and patterns defined near the top of the file:

*   `IGNORE_PATTERNS`: Default set of directories and file patterns to ignore.
*   `CODE_EXTENSIONS`: Default set of file extensions/names treated as source code.
*   `INTERESTING_FILENAMES`: Default set of filenames prioritized in summaries.
*   `SIMPLIFICATION_PATTERNS`: Regex patterns for initial content obfuscation.
*   `POST_SIMPLIFICATION_PATTERNS`: Regex patterns for the optional `--apply-patterns` step.
*   `BLOCK_COMPRESSION_PATTERNS`: Dictionary mapping pattern names to regexes for the `--compress-patterns` step.
*   `DEFINITION_SIMPLIFICATION_PATTERNS`: Regex patterns used to simplify the content shown in the `--minify-lines` definition block.
*   `PLACEHOLDER_CLEANUP_PATTERN`: Regex used by `--post-cleanup` to identify lines for removal.

Advanced users can modify these directly within the script to customize its behavior.

## Dependencies

*   Python 3.x
*   Uses standard Python libraries only (no external packages required).

## License

[Specify Your License Here - e.g., MIT License]

*If you haven't chosen a license, MIT is a common permissive choice. You would need to add a separate `LICENSE` file containing the actual MIT license text.*

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## Disclaimer

The compression and simplification performed by this script are **lossy**. It is designed to create a condensed representation for specific purposes (like LLM context) and is **not** suitable for code archival or backup. Always keep original source files. The effectiveness of different options may vary greatly depending on the structure and content of the target folder. Experiment with different flags to find the best combination for your needs.