# folder_to_text.py

import argparse
import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
from collections import defaultdict
import ast

# --- Configuration ---
# Version 10.1: Fixed a critical bug where the DocstringRemover class was missing, causing errors on all .py files.
# - Summarization is the DEFAULT behavior. Use --full-code to disable it.
# - AST-based summarization for Python files to show structure-only.
# - New dependency: NLTK for prose stripping in documentation files.
# - New flag --strip-prose to remove filler words from .md files.
# - New concise header format: [FILE: path/to/file.ext]


def _summarize_package_json(content: str) -> str:
    """Summarizes package.json to its most essential parts."""
    try:
        data = json.loads(content)
        summary = {
            "name": data.get("name"), "version": data.get("version"), "scripts": data.get("scripts"),
        }
        return f"// Summarized package.json. Key details:\n{json.dumps(summary, indent=2)}\n// Dependencies omitted."
    except json.JSONDecodeError:
        return "// Could not parse package.json, showing raw content.\n" + content

def _summarize_requirements_txt(content: str) -> str:
    """Summarizes requirements.txt."""
    return f"// Contains {len(content.strip().splitlines())} dependencies."


# === START: BUG FIX - The missing DocstringRemover class is now included. ===
class DocstringRemover(ast.NodeTransformer):
    """AST transformer to remove docstrings from Python code."""
    def visit_FunctionDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            node.body.pop(0)
        self.generic_visit(node)
        return node
    def visit_ClassDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            node.body.pop(0)
        self.generic_visit(node)
        return node
    def visit_Module(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            node.body.pop(0)
        self.generic_visit(node)
        return node
# === END: BUG FIX ===


class CodeSummarizer(ast.NodeTransformer):
    """AST transformer to replace function/class bodies with 'pass'."""
    def visit_FunctionDef(self, node):
        node.body = [ast.Pass()]
        self.generic_visit(node)
        return node
    def visit_AsyncFunctionDef(self, node):
        node.body = [ast.Pass()]
        self.generic_visit(node)
        return node
    def visit_ClassDef(self, node):
        new_body = []
        for body_item in node.body:
            if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_item.body = [ast.Pass()]
                new_body.append(body_item)
            elif isinstance(body_item, ast.Expr) and isinstance(body_item.value, (ast.Str, ast.Constant)):
                continue
            else:
                new_body.append(body_item)
        node.body = new_body if new_body else [ast.Pass()]
        self.generic_visit(node)
        return node


CONFIG = {
    "EXCLUDE_DIRS_DEFAULT": {
        ".git", "venv", "__pycache__", "node_modules", ".pytest_cache", "build", "dist",
        "htmlcov", ".mypy_cache", "data", "ui_screenshots", "backup", "legacy", ".vscode",
    },
    "EXCLUDE_FILES_DEFAULT": {
        ".DS_Store", ".gitignore", "package-lock.json", "yarn.lock", ".env", ".env.test",
    },
    "EXCLUDE_PATTERNS_DEFAULT": [re.compile(r".*project_context_.*\.txt$"), re.compile(r"\.log$")],
    "ALWAYS_INCLUDE_FILES": {"requirements.txt", "package.json"},
    "CORE_EXTENSIONS": {".py", ".js", ".jsx", ".ts", ".tsx", ".sh", "Dockerfile", ".env.example"},
    "DOCS_EXTENSIONS": {".md"},
    "CONFIG_EXTENSIONS": {".json", ".yml", ".yaml", ".toml", ".cfg", ".ini"},
    "TEST_FILE_PATTERN": re.compile(r"([._])(test|spec)\.(js|jsx|ts|tsx)$|^(test_)|(_test)\.py$", re.IGNORECASE),
    "SUMMARIZE_FILES": {
        "package.json": _summarize_package_json,
        "requirements.txt": _summarize_requirements_txt,
    },
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

def _check_nltk_stopwords():
    try:
        from nltk.corpus import stopwords
        stopwords.words('english')
    except ImportError:
        logger.error("NLTK library not found. Please run: pip install nltk")
        sys.exit(1)
    except LookupError:
        logger.error("NLTK 'stopwords' corpus not found.")
        logger.error("Please run the following command to download it:")
        logger.error("python -m nltk.downloader stopwords")
        sys.exit(1)

def _summarize_python_with_ast(content: str, filepath: Path) -> str:
    try:
        tree = ast.parse(content)
        tree = DocstringRemover().visit(tree)
        tree = CodeSummarizer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError) as e:
        logger.warning(f"Could not summarize {filepath} with AST due to syntax error: {e}. Including full content instead.")
        return content

def _strip_prose_with_nltk(content: str) -> str:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(content)
    filtered_words = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join(filtered_words)

def process_file_content(filepath: Path, content: str, args: argparse.Namespace) -> str:
    filename = filepath.name
    if filename in CONFIG["SUMMARIZE_FILES"]:
        return CONFIG["SUMMARIZE_FILES"][filename](content)

    processed_content = content
    ext = filepath.suffix.lower()

    if not args.full_code:
        if ext == ".py":
            processed_content = _summarize_python_with_ast(processed_content, filepath)
    
    if args.strip_prose and ext in CONFIG["DOCS_EXTENSIONS"]:
        processed_content = _strip_prose_with_nltk(processed_content)

    if not args.no_minify_lines:
        lines = [line.rstrip() for line in processed_content.splitlines()]
        processed_content = "\n".join(filter(None, lines))

    final_lines = []
    for line in processed_content.splitlines():
        match = re.match(r"^(\s+)", line)
        if match:
            final_lines.append("\t" + line.lstrip())
        else:
            final_lines.append(line)
    return "\n".join(final_lines)

def analyze_output(output_content: str):
    file_pattern = re.compile(r"\[FILE: (.*?)\]\n(.*?)(?=\n\n\[FILE:|\Z)", re.DOTALL)
    analysis = {"total_chars": 0, "by_type": defaultdict(lambda: {"chars": 0, "count": 0})}
    for match in file_pattern.finditer(output_content):
        file_path_str, file_content = match.groups()
        ext = f".{file_path_str.split('.')[-1]}" if "." in file_path_str else ".other"
        content_char_count = len(file_content)
        analysis["total_chars"] += content_char_count
        analysis["by_type"][ext]["chars"] += content_char_count
        analysis["by_type"][ext]["count"] += 1
    
    print("\n" + "-" * 40, file=sys.stderr)
    print("Content Analysis Report:", file=sys.stderr)
    print(f"Total Content Characters: {analysis['total_chars']:,}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    sorted_types = sorted(analysis["by_type"].items(), key=lambda item: item[1]["chars"], reverse=True)
    for ext, data in sorted_types:
        chars, count = data["chars"], data["count"]
        percentage = (chars / analysis["total_chars"]) * 100 if analysis["total_chars"] > 0 else 0
        print(f"  - {ext:<12s}: {chars:>10,d} chars ({percentage:5.2f}%) [{count} file(s)]", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

def _should_exclude(path: Path, args: argparse.Namespace, config: dict, output_abs_path: Path, include_exts: set) -> bool:
    if path.name.lower() in config["ALWAYS_INCLUDE_FILES"]: return False
    for part in path.parts:
        if part in config["EXCLUDE_DIRS_DEFAULT"] or part in args.ignore_dir: return True
    if not args.include_tests and (any(part == "tests" for part in path.parts) or
                                  (config["TEST_FILE_PATTERN"] and config["TEST_FILE_PATTERN"].search(path.name))): return True
    if path.name in config["EXCLUDE_FILES_DEFAULT"] or path.name in args.ignore_file or path.resolve() == output_abs_path: return True
    if any(p.search(path.name) for p in config["EXCLUDE_PATTERNS_DEFAULT"]): return True
    if path.is_file():
        ext = path.suffix.lower() if path.suffix else path.name
        if ext not in include_exts: return True
    return False

def discover_files(path: Path, config: dict, args: argparse.Namespace, output_abs_path: Path, include_exts: set) -> list[Path]:
    discovered_paths = []
    if path.is_file():
        if not _should_exclude(path, args, config, output_abs_path, include_exts):
            discovered_paths.append(path)
        return discovered_paths
    for root, dirs, files in os.walk(path, topdown=True):
        current_dir = Path(root)
        dirs[:] = [d for d in dirs if not _should_exclude(current_dir / d, args, config, output_abs_path, include_exts)]
        for f in files:
            file_path = current_dir / f
            if not _should_exclude(file_path, args, config, output_abs_path, include_exts):
                discovered_paths.append(file_path)
    return discovered_paths

def main():
    parser = argparse.ArgumentParser(
        description="v10.1: An intelligent code summarizer for LLM context generation. Defaults to a highly compressed, structure-only view of your code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    path_group = parser.add_argument_group("Path Control")
    path_group.add_argument("root_path", type=Path, help="The root directory or specific file to process.")
    path_group.add_argument("-o", "--output", type=Path, default="project_context.txt", help="Output file name.")

    inclusion_group = parser.add_argument_group("Inclusion Controls (Lean by Default)")
    inclusion_group.add_argument("--full-code", action="store_true", help="Disables summarization and includes the full, unabridged content of all files.")
    inclusion_group.add_argument("--include-tests", action="store_true", help="Include test files and 'tests' directories.")
    inclusion_group.add_argument("--include-docs", action="store_true", help="Include documentation files (.md).")
    inclusion_group.add_argument("--include-config", action="store_true", help="Include common config files (.json, .yml, etc.).")
    
    exclusion_group = parser.add_argument_group("Exclusion Controls")
    exclusion_group.add_argument("--ignore-dir", action="append", default=[], help="Additional directory names to ignore.")
    exclusion_group.add_argument("--ignore-file", action="append", default=[], help="Additional file names to ignore.")

    format_group = parser.add_argument_group("Advanced Compression")
    format_group.add_argument("--strip-prose", action="store_true", help="Use NLTK to remove common English filler words from documentation files (.md). Requires 'pip install nltk'.")
    format_group.add_argument("--no-minify-lines", action="store_true", help="Preserve blank lines and original line structure (disables default line minification).")

    args = parser.parse_args()

    if args.strip_prose: _check_nltk_stopwords()
    if not args.root_path.exists():
        logger.error(f"Path '{args.root_path}' does not exist."); return

    include_extensions = set(CONFIG["CORE_EXTENSIONS"])
    if args.include_docs: include_extensions.update(CONFIG["DOCS_EXTENSIONS"])
    if args.include_config: include_extensions.update(CONFIG["CONFIG_EXTENSIONS"])
    
    output_abs_path = args.output.resolve()
    output_abs_path.parent.mkdir(parents=True, exist_ok=True)
    final_files_to_process = sorted(discover_files(args.root_path, CONFIG, args, output_abs_path, include_extensions))

    with args.output.open("w", encoding="utf-8") as f_out:
        f_out.write(f"// LLM CONTEXT FOR: {args.root_path.name}\n")
        f_out.write("// FORMATTING-RULES: 1. Content is a structural summary by default. 2. Files begin with '[FILE: path]'. 3. Indentation is a single tab.\n\n")
        for file_path in final_files_to_process:
            try:
                relative_path_str = str(file_path.relative_to(Path.cwd()))
            except ValueError:
                relative_path_str = str(file_path.resolve())
            logger.info(f"Processing: {relative_path_str}")
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if content.strip():
                    processed_content = process_file_content(file_path, content, args)
                    f_out.write(f"[FILE: {relative_path_str}]\n")
                    f_out.write(processed_content)
                    f_out.write("\n\n")
            except Exception as e:
                logger.error(f"Could not process file {file_path}: {e}")

    logger.info(f"\nProcessing complete. Output written to {args.output}")
    with args.output.open("r", encoding="utf-8") as f_in:
        analyze_output(f_in.read())

if __name__ == "__main__":
    main()