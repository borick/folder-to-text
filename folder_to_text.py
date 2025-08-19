# folder_to_text.py (v15.4 - stdout, quiet, no-header, string replacements, shrink)

import argparse
import ast
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# --- Configuration ---
CONFIG = {
    "EXCLUDE_DIRS_DEFAULT": {
        ".git",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        "build",
        "dist",
        "htmlcov",
        ".mypy_cache",
        "data",
        "ui_screenshots",
        "backup",
        "legacy",
        ".vscode",
    },
    "EXCLUDE_FILES_DEFAULT": {
        ".DS_Store",
        ".gitignore",
        "package-lock.json",
        "yarn.lock",
        ".env",
        ".env.test",
        ".contextignore",  # Exclude the ignore file itself
    },
    "EXCLUDE_PATTERNS_DEFAULT": [
        re.compile(r".*project_context_.*\.txt$"),
        re.compile(r"\.log$"),
    ],
    "ALWAYS_INCLUDE_FILES": {"requirements.txt", "package.json"},
    "CORE_EXTENSIONS": {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".sh",
        "Dockerfile",
        ".env.example",
    },
    "DOCS_EXTENSIONS": {".md"},
    "CONFIG_EXTENSIONS": {".json", ".yml", ".yaml", ".toml", ".cfg", ".ini"},
    "TEST_FILE_PATTERN": re.compile(
        r"([._])(test|spec)\.(js|jsx|ts|tsx)$|^(test_)|(_test)\.py$", re.IGNORECASE
    ),
}

# Ordered list of (pattern, replacement) for token shrinking. Order is crucial.
SHRINK_MAPPINGS: Dict[str, List[Tuple[str, str]]] = {
    "python": [
        ("__init__", "_init_"),
        ("async def", "afn"),
        ("self", "s"),
        ("True", "T"),
        ("False", "F"),
        ("None", "N"),
        ("elif", "|?"),
        ("else", "|"),
        ("if", "?"),
        ("from", "<~"),
        ("import", "~"),
        ("return", "=>"),
        ("def", "fn"),
        ("class", "cls"),
        ("await", "^"),
        ("async", "a"),
        ("and", "&"),
        ("or", "|"),
        ("not", "!"),
        ("is", "="),
        ("in", "e"),
        ("print", "p"),
    ],
    "javascript": [
        ("constructor", "_init_"),
        ("function", "fn"),
        ("export default", ">> def"),
        ("export", ">>"),
        ("default", "def"),
        ("import", "~"),
        ("from", "<~"),
        ("const", "c!"),
        ("let", "l!"),
        ("var", "v!"),
        ("return", "=>"),
        ("class", "cls"),
        ("else if", "|?"),
        ("if", "?"),
        ("else", "|"),
        ("async", "a"),
        ("await", "^"),
        ("this", "t"),
        ("true", "T"),
        ("false", "F"),
        ("null", "N"),
        ("undefined", "U"),
    ],
}
SHRINK_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx"}


# A curated list of common English stop words for the zero-dependency prose stripper.
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "can't", "cannot",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
    "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
    "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
}


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------- Summarizers ----------
def _summarize_package_json(content: str) -> str:
    """Summarizes package.json to its name, version, and scripts."""
    try:
        data = json.loads(content)
        summary = {
            "name": data.get("name"),
            "version": data.get("version"),
            "scripts": data.get("scripts"),
        }
        return (
            f"// Summarized package.json:\n{json.dumps(summary, indent=2)}\n"
            "// Dependencies omitted."
        )
    except json.JSONDecodeError:
        return "// Could not parse package.json, showing raw content.\n" + content


def _summarize_requirements_txt(content: str) -> str:
    """Provides a count of dependencies in requirements.txt."""
    return f"// Contains {len(content.strip().splitlines())} dependencies."


SUMMARIZE_FILES = {
    "package.json": _summarize_package_json,
    "requirements.txt": _summarize_requirements_txt,
}


class DocstringRemover(ast.NodeTransformer):
    """AST transformer to remove docstrings from Python code."""

    def visit_FunctionDef(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            node.body.pop(0)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            node.body.pop(0)
        self.generic_visit(node)
        return node

    def visit_Module(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            node.body.pop(0)
        self.generic_visit(node)
        return node


class CodeSummarizer(ast.NodeTransformer):
    """AST transformer to replace function/class bodies with 'pass'."""

    def visit_FunctionDef(self, node):
        node.body = [ast.Pass()]
        return node

    def visit_AsyncFunctionDef(self, node):
        node.body = [ast.Pass()]
        return node

    def visit_ClassDef(self, node):
        new_body = []
        for b in node.body:
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                b.body = [ast.Pass()]
                new_body.append(b)
        node.body = new_body or [ast.Pass()]
        return node


def _summarize_python_with_ast(content: str, filepath: Path) -> str:
    """Summarizes a Python file to show only its structure."""
    try:
        tree = ast.parse(content)
        tree = DocstringRemover().visit(tree)
        tree = CodeSummarizer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logger.warning(f"AST summarization failed for {filepath}: {e}")
        return content


# ---------- Token Shrinking (Robust Implementation) ----------
def _shrink_tokens(content: str, ext: str) -> str:
    """
    Replaces common keywords with shorter symbols using a robust multi-pass
    approach that protects string literals and comments from being modified.
    """
    mapping_key = "python" if ext == ".py" else "javascript"
    replacements = SHRINK_MAPPINGS.get(mapping_key, [])
    if not replacements:
        return content

    # Safely create the pattern for matching triple-double-quoted strings to avoid SyntaxError
    triple_double_quote_pattern = 'f?r?b?u?' + '"""' + '.*?' + '"""'

    # Use an f-string to inject the safe pattern into the main regex.
    # This avoids the parser error while keeping the regex readable.
    string_comment_pattern = re.compile(
        f"""
        (
            # Triple-quoted strings (Python)
            f?r?b?u?'''.*?''' |
            {triple_double_quote_pattern} |
            # Single/double-quoted strings (Py/JS)
            f?r?b?u?'.*?' |
            f?r?b?u?".*?" |
            # Template literals (JS)
            `.*?` |
            # Single-line comments (JS/TS)
            //.*?$ |
            # Single-line comments (Python)
            #.*?$ |
            # Multi-line comments (JS/TS)
            /\\*.*?\\*/
        )
        """,
        re.DOTALL | re.VERBOSE | re.MULTILINE,
    )

    # 1. Isolate strings and comments
    protected_parts: List[str] = []
    placeholders: List[str] = []

    def protect(match):
        placeholder = f"__PROTECTED_{len(protected_parts)}__"
        protected_parts.append(match.group(0))
        placeholders.append(placeholder)
        return placeholder

    safe_content = string_comment_pattern.sub(protect, content)

    # 2. Apply shrinking to the "safe" code
    for old, new in replacements:
        pattern = r"\b" + re.escape(old) + r"\b"
        safe_content = re.sub(pattern, new, safe_content)

    # 3. Restore the protected parts
    final_content = safe_content
    for i, placeholder in enumerate(placeholders):
        final_content = final_content.replace(placeholder, protected_parts[i], 1)

    return final_content


def generate_shrink_legend() -> str:
    """Creates a legend explaining the token shrink transformations."""
    legend = ["// TOKEN SHRINKING LEGEND (AI CONTEXT):"]
    py_map = " | ".join([f"{o}->{n}" for o, n in SHRINK_MAPPINGS["python"]])
    js_map = " | ".join([f"{o}->{n}" for o, n in SHRINK_MAPPINGS["javascript"]])
    legend.append(f"// Python: {py_map}")
    legend.append(f"// JS/TS: {js_map}")
    return "\n".join(legend) + "\n\n"


# ---------- Compression & Line Removal ----------
def compress_content(content: str, level: str) -> str:
    """Applies various levels of compression to the file content."""
    if level == "light":
        content = re.sub(r"\n\s*\n+", "\n\n", content)
    elif level == "medium":
        content = re.sub(r"\n\s*\n+", "\n\n", content)
        content = re.sub(r"[ \t]+", " ", content)
    elif level == "aggressive":
        content = re.sub(r"\n\s*\n+", "\n\n", content)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"^\s*#.*$", "", content, flags=re.MULTILINE)
    return content.strip()


def _remove_ignored_lines(content: str, line_patterns: List[str]) -> str:
    """Removes any line containing one of the specified literal string patterns."""
    if not line_patterns:
        return content
    lines = content.splitlines()
    filtered_lines = [
        line
        for line in lines
        if not any(pattern in line for pattern in line_patterns)
    ]
    return "\n".join(filtered_lines)


# ---------- Prose Stripping ----------
def _strip_prose_simplified(content: str) -> str:
    """Strips common English stop words from a string without NLTK."""
    tokens = re.findall(r"\b\w+\b", content.lower())
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(filtered_tokens)


# ---------- Processing ----------
def process_file_content(
    filepath: Path,
    content: str,
    args: argparse.Namespace,
    line_ignore_patterns: List[str],
    string_replacements: Dict[str, str],
) -> str:
    """Applies replacements, line removal, summarization, and compression."""
    # First, apply global string replacements.
    for old, new in string_replacements.items():
        content = content.replace(old, new)

    # Second, remove any globally ignored lines.
    content = _remove_ignored_lines(content, line_ignore_patterns)

    filename, ext = filepath.name, filepath.suffix.lower()

    # Apply token shrinking if enabled and file type is supported
    if args.shrink and ext in SHRINK_EXTENSIONS:
        content = _shrink_tokens(content, ext)

    if args.summarize:
        if filename in SUMMARIZE_FILES:
            content = SUMMARIZE_FILES[filename](content)
        elif ext == ".py":
            content = _summarize_python_with_ast(content, filepath)

    if args.strip_prose and ext in CONFIG["DOCS_EXTENSIONS"]:
        content = _strip_prose_simplified(content)

    if not args.no_minify_lines:
        lines = [line.rstrip() for line in content.splitlines()]
        content = "\n".join(filter(None, lines))

    comp_level = args.compress or "medium"
    content = compress_content(content, comp_level)
    return content


# ---------- Analysis ----------
def analyze_output(output_content: str):
    """Prints a report analyzing the character count of the generated output."""
    file_pattern = re.compile(r"\[FILE: (.*?)\]\n(.*?)(?=\n\n\[FILE:|\Z)", re.DOTALL)
    analysis = {
        "total_chars": 0,
        "by_type": defaultdict(lambda: {"chars": 0, "count": 0}),
    }
    for match in file_pattern.finditer(output_content):
        file_path_str, file_content = match.groups()
        ext = f".{file_path_str.split('.')[-1]}" if "." in file_path_str else ".other"
        char_count = len(file_content)
        analysis["total_chars"] += char_count
        analysis["by_type"][ext]["chars"] += char_count
        analysis["by_type"][ext]["count"] += 1
    print("\n" + "-" * 40, file=sys.stderr)
    print("Content Analysis Report:", file=sys.stderr)
    print(f"Total Characters: {analysis['total_chars']:,}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    for ext, data in sorted(
        analysis["by_type"].items(), key=lambda i: i[1]["chars"], reverse=True
    ):
        chars, count = data["chars"], data["count"]
        pct = (chars / analysis["total_chars"]) * 100 if analysis["total_chars"] else 0
        print(
            f"  - {ext:<10s}: {chars:>10,d} chars ({pct:5.2f}%) [{count} file(s)]",
            file=sys.stderr,
        )
    print("-" * 40, file=sys.stderr)


# ---------- Tree View Generation ----------
def generate_tree_view(file_paths: List[Path], root_path: Path) -> str:
    """Generates a string representation of the directory tree."""
    tree: Dict[str, Any] = {}
    for path in file_paths:
        relative_path = path.relative_to(root_path)
        parts = list(relative_path.parts)
        current_level = tree
        for part in parts:
            current_level = current_level.setdefault(part, {})

    def build_tree_string(structure: Dict, prefix: str = "") -> List[str]:
        lines = []
        entries = sorted(structure.keys())
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry}")
            if structure[entry]:
                extension = "    " if i == len(entries) - 1 else "│   "
                lines.extend(build_tree_string(structure[entry], prefix + extension))
        return lines

    tree_lines = [str(root_path.name)] + build_tree_string(tree)
    return "// DIRECTORY TREE\n" + "\n".join(tree_lines) + "\n\n"


# ---------- Discovery and Ignore Logic ----------
def _parse_context_ignore(root_path: Path) -> Dict[str, Any]:
    """
    Parses a .contextignore file for file patterns, line patterns,
    and string replacements.
    """
    ignore_file = root_path / ".contextignore"
    patterns: Dict[str, Any] = {
        "file_patterns": [],
        "line_patterns": [],
        "string_replacements": {},
    }
    if not ignore_file.is_file():
        return patterns

    current_section = "file_patterns"
    try:
        with ignore_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.lower() == "[lines-to-ignore]":
                    current_section = "line_patterns"
                    continue
                elif line.lower() == "[string-replacements]":
                    current_section = "string_replacements"
                    continue

                if current_section == "string_replacements":
                    if "::" in line:
                        old, new = line.split("::", 1)
                        patterns["string_replacements"][old.strip()] = new.strip()
                    else:
                        logger.warning(
                            f"Invalid replacement rule in .contextignore: {line}"
                        )
                else:
                    patterns[current_section].append(line)

        if not args.quiet:
            logger.info(
                f"Loaded from .contextignore: "
                f"{len(patterns['file_patterns'])} file patterns, "
                f"{len(patterns['line_patterns'])} line patterns, "
                f"{len(patterns['string_replacements'])} string replacements."
            )
    except Exception as e:
        logger.warning(f"Could not read .contextignore file: {e}")
    return patterns


def _should_exclude(
    path: Path,
    args: argparse.Namespace,
    output_abs_path: Path,
    include_exts: Set[str],
    file_ignore_patterns: List[str],
) -> bool:
    """Determines if a file or directory should be excluded from processing."""
    try:
        relative_path = path.relative_to(args.root_path.resolve())
    except ValueError:
        relative_path = path

    for pattern in file_ignore_patterns:
        if relative_path.match(pattern):
            return True

    if path.name.lower() in CONFIG["ALWAYS_INCLUDE_FILES"]:
        return False
    for part in path.parts:
        if part in CONFIG["EXCLUDE_DIRS_DEFAULT"] or part in args.ignore_dir:
            return True
    if not args.include_tests and (
        any(p == "tests" for p in path.parts)
        or CONFIG["TEST_FILE_PATTERN"].search(path.name)
    ):
        return True
    if (
        path.name in CONFIG["EXCLUDE_FILES_DEFAULT"]
        or path.name in args.ignore_file
        or (output_abs_path and path.resolve() == output_abs_path)
    ):
        return True
    if any(p.search(path.name) for p in CONFIG["EXCLUDE_PATTERNS_DEFAULT"]):
        return True
    if path.is_file():
        ext = path.suffix.lower() if path.suffix else path.name
        if ext not in include_exts:
            return True
    return False


def discover_files(
    path: Path,
    args: argparse.Namespace,
    output_abs_path: Path,
    include_exts: Set[str],
    file_ignore_patterns: List[str],
) -> List[Path]:
    """Recursively finds all files to be included in the output."""
    found = []
    if path.is_file():
        if not _should_exclude(
            path, args, output_abs_path, include_exts, file_ignore_patterns
        ):
            found.append(path)
        return found

    for root, dirs, files in os.walk(path, topdown=True):
        current_dir = Path(root)
        dirs[:] = [
            d
            for d in dirs
            if not _should_exclude(
                current_dir / d,
                args,
                output_abs_path,
                include_exts,
                file_ignore_patterns,
            )
        ]
        for f in files:
            fp = current_dir / f
            if not _should_exclude(
                fp, args, output_abs_path, include_exts, file_ignore_patterns
            ):
                found.append(fp)
    return found


# ---------- Main ----------
def main(args: argparse.Namespace):
    if not args.root_path.exists():
        logger.error(f"Path '{args.root_path}' does not exist.")
        return

    ignore_config = _parse_context_ignore(args.root_path)
    file_ignore_patterns = ignore_config["file_patterns"]
    line_ignore_patterns = ignore_config["line_patterns"]
    string_replacements = ignore_config["string_replacements"]

    include_exts = set(CONFIG["CORE_EXTENSIONS"])
    if args.include_docs:
        include_exts.update(CONFIG["DOCS_EXTENSIONS"])
    if args.include_config:
        include_exts.update(CONFIG["CONFIG_EXTENSIONS"])

    output_abs = None if args.stdout else args.output.resolve()
    if output_abs:
        output_abs.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        discover_files(
            args.root_path, args, output_abs, include_exts, file_ignore_patterns
        )
    )

    output_buffer = []

    def write_output(content):
        if args.stdout:
            print(content, end="")
        else:
            output_buffer.append(content)

    write_output(f"// LLM CONTEXT FOR: {args.root_path.name}\n")

    if args.shrink:
        write_output(generate_shrink_legend())

    if args.tree:
        tree_view = generate_tree_view(files, args.root_path)
        write_output(tree_view)

    if not args.no_file_header:
        write_output("// FORMAT: Each file starts with [FILE: path]\n\n")

    for fp in files:
        try:
            rel_path_str = str(fp.relative_to(args.root_path))
            for old, new in string_replacements.items():
                rel_path_str = rel_path_str.replace(old, new)

            if not args.quiet:
                logger.info(f"Processing: {rel_path_str}")

            text = fp.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                processed = process_file_content(
                    fp, text, args, line_ignore_patterns, string_replacements
                )
                if not args.no_file_header:
                    write_output(f"[FILE: {rel_path_str}]\n")
                write_output(f"{processed}\n\n")
        except Exception as e:
            logger.error(f"Could not process {fp}: {e}")

    if not args.stdout and output_abs:
        final_content = "".join(output_buffer)
        output_abs.write_text(final_content, encoding="utf-8")
        if not args.quiet:
            logger.info(f"\nProcessing complete. Output written to {args.output}")
        analyze_output(final_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "v15.4: Consolidate project folder into a single text file. "
            "Supports .contextignore, tree view, summarization, compression, "
            "and robust token shrinking for AI context optimization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_path", type=Path, help="Root directory or specific file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="project_context.txt",
        help="Output file name. Ignored if --stdout is used.",
    )
    parser.add_argument(
        "--stdout", action="store_true", help="Print to standard output."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress logging."
    )
    parser.add_argument(
        "--no-file-header",
        action="store_true",
        help="Omit the '[FILE: ...]' headers.",
    )
    parser.add_argument(
        "--include-tests", action="store_true", help="Include test files/dirs."
    )
    parser.add_argument(
        "--include-docs", action="store_true", help="Include docs (.md)."
    )
    parser.add_argument(
        "--include-config", action="store_true", help="Include config files."
    )
    parser.add_argument(
        "--ignore-dir", action="append", default=[], help="Dirs to ignore."
    )
    parser.add_argument(
        "--ignore-file", action="append", default=[], help="Files to ignore."
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize structure instead of full code.",
    )
    parser.add_argument(
        "--strip-prose",
        action="store_true",
        help="Strip common filler words from docs (zero-dependency).",
    )
    parser.add_argument(
        "--no-minify-lines",
        action="store_true",
        help="Preserve blank lines and spacing.",
    )
    parser.add_argument(
        "--compress",
        choices=["light", "medium", "aggressive"],
        help="Compression level (default: medium).",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Prepend a directory tree view to the output.",
    )
    parser.add_argument(
        "--shrink",
        action="store_true",
        help="Apply token shrinking to supported code files (py, js, ts).",
    )
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    main(args)