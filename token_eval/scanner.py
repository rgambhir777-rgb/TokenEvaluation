"""Static scan of a repository: LLM call sites, models, prompt text and codebase size.

Nothing in the repository is executed. Prompt token numbers are estimates of the
*static* prompt text found in the code; dynamic content (user input, retrieved
documents, chat history, loops) is only visible with ``token-eval run``.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from token_eval.counting import count_tokens, count_tokens_local
from token_eval.pricing import PricingTable, default_pricing

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build",
             ".next", ".tox", ".mypy_cache", ".pytest_cache", "site-packages", ".token_eval",
             "vendor", "target", ".idea", ".vscode"}
CODE_EXT = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".rb", ".java", ".cs",
            ".php", ".rs", ".ipynb"}
TEXT_EXT = CODE_EXT | {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css",
                       ".sh", ".sql", ".jinja", ".j2", ".prompt", ".cfg", ".ini"}
PROMPT_FILE_EXT = {".prompt", ".jinja", ".j2", ".txt", ".md", ".mustache", ".hbs"}
MAX_FILE_BYTES = 1_000_000

CALL_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("anthropic", "messages.create", re.compile(r"\.messages\.(create|stream)\s*\(")),
    ("anthropic", "langchain", re.compile(r"\bChatAnthropic\s*\(")),
    ("openai", "chat.completions", re.compile(r"\.chat\.completions\.(create|parse|stream)\s*\(")),
    ("openai", "responses", re.compile(r"\.responses\.(create|parse|stream)\s*\(")),
    ("openai", "embeddings", re.compile(r"\.embeddings\.create\s*\(")),
    ("openai", "langchain", re.compile(r"\b(Azure)?ChatOpenAI\s*\(")),
    ("generic", "litellm", re.compile(r"\blitellm\.(a?completion|embedding)\s*\(")),
    ("generic", "vercel-ai-sdk", re.compile(r"\b(generateText|streamText|generateObject)\s*\(")),
    ("google", "gemini", re.compile(r"\.generate_content\s*\(|\bgenerateContent\s*\(")),
]
# Model IDs must contain a version digit, e.g. claude-opus-5-5, gpt-4o-mini, o3-mini.
MODEL_RE = re.compile(
    r"""["'`]((?:claude|gpt|chatgpt|text-embedding|gemini|mistral|llama)-[\w.:-]*\d[\w.:-]*"""
    r"""|o[134](?:-mini|-pro|-preview)?(?:-\d{4}-\d{2}-\d{2})?)["'`]""")
NOT_MODEL_SUFFIX = re.compile(r"\.(json|js|ts|md|py|txt|ya?ml|lock|png|svg)$")
PROMPT_NAME_RE = re.compile(r"prompt|system|instruction|template|persona|message", re.I)
COMMENT_PREFIXES = ("#", "//", "*", "/*", "<!--")
MIN_PROMPT_CHARS = 200
SQL_RE = re.compile(r"\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|WITH|DROP)\s", re.I)


def looks_like_prose(text: str, min_words: int = 25) -> bool:
    """Natural-language text rather than code, CSS classes, SQL or data."""
    words = text.split()
    if len(words) < min_words or SQL_RE.match(text):
        return False
    letters = sum(c.isalpha() or c.isspace() or c in ".,;:!?'\"-()" for c in text)
    code_marks = sum(text.count(t) for t in ("{", "}", ";", "=>", "==", "()", "</"))
    return letters / len(text) >= 0.85 and code_marks / len(words) < 0.08


@dataclass
class CallSite:
    file: str
    line: int
    provider: str
    kind: str
    snippet: str


@dataclass
class PromptText:
    location: str
    tokens: int
    preview: str


@dataclass
class ScanReport:
    repo: str
    files_scanned: int
    codebase_tokens: int
    codebase_tokens_by_ext: dict[str, int]
    call_sites: list[CallSite]
    models: dict[str, int]
    count_model: str
    prompts: list[PromptText]
    static_prompt_tokens: int
    static_prompt_cost_usd: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_files(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        if p.is_file() and p.suffix.lower() in TEXT_EXT and p.stat().st_size <= MAX_FILE_BYTES:
            yield p


def _read(p: Path) -> str:
    text = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix == ".ipynb":
        try:
            nb = json.loads(text)
            return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", [])
                             if c.get("cell_type") == "code")
        except json.JSONDecodeError:
            return ""
    return text


def _python_prompts(src: str) -> list[tuple[int, str]]:
    """String literals that look like prompts: long, or assigned to prompt-ish names."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    found: dict[int, str] = {}

    def literal(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):  # f-string: keep the static parts
            return "".join(v.value for v in node.values
                           if isinstance(v, ast.Constant) and isinstance(v.value, str))
        return None

    for node in ast.walk(tree):
        targets: list[str] = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target.id], node.value
        elif isinstance(node, ast.keyword) and node.arg:
            targets, value = [node.arg], node.value
        if value is not None:
            s = literal(value)
            named = any(PROMPT_NAME_RE.search(t) for t in targets)
            if s and not SQL_RE.match(s) and ((named and len(s) >= 40) or (len(s) >= MIN_PROMPT_CHARS
                                                    and looks_like_prose(s))):
                found[value.lineno] = s
        s = literal(node) if isinstance(node, (ast.Constant, ast.JoinedStr)) else None
        if (s and len(s) >= MIN_PROMPT_CHARS and getattr(node, "lineno", 0) not in found
                and looks_like_prose(s)):
            found[node.lineno] = s
    skip = python_string_statement_lines(tree)
    return [(ln, s) for ln, s in sorted(found.items()) if ln not in skip]


def python_string_statement_lines(tree: ast.AST) -> set[int]:
    """Lines covered by docstrings and other bare string statements (not prompts)."""
    lines: set[int] = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)):
            lines.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
    return lines


def js_strings(src: str) -> list[tuple[int, str]]:
    """(line, static text) of every string/template literal, skipping comments.

    A small lexer: tracks quotes, comments and ``${...}`` nesting so that code
    between two template literals is never mistaken for a string.
    """
    out: list[tuple[int, str]] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            i = src.find("\n", i)
            i = n if i < 0 else i
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "'\"`":
            start, buf, depth = i, [], 0
            i += 1
            while i < n:
                ch = src[i]
                if ch == "\\":
                    buf.append(src[i + 1: i + 2])
                    i += 2
                    continue
                if c == "`" and depth == 0 and src.startswith("${", i):
                    depth, i = 1, i + 2
                    continue
                if depth:
                    depth += (ch == "{") - (ch == "}")
                    i += 1
                    continue
                if ch == c or (c != "`" and ch == "\n"):
                    i += 1
                    break
                buf.append(ch)
                i += 1
            out.append((src.count("\n", 0, start) + 1, "".join(buf)))
        else:
            i += 1
    return out


def _js_prompts(src: str) -> list[tuple[int, str]]:
    return [(ln, s) for ln, s in js_strings(src)
            if len(s) >= MIN_PROMPT_CHARS and looks_like_prose(s)]


def _is_prompt_file(rel: Path) -> bool:
    in_prompt_dir = any("prompt" in part.lower() for part in rel.parts[:-1])
    return rel.suffix.lower() in {".prompt", ".jinja", ".j2"} or (
        in_prompt_dir and rel.suffix.lower() in PROMPT_FILE_EXT)


def scan_repo(root: str | Path, model: str | None = None,
              pricing: PricingTable | None = None) -> ScanReport:
    root = Path(root).resolve()
    pricing = pricing or default_pricing()
    call_sites: list[CallSite] = []
    models: Counter[str] = Counter()
    raw_prompts: list[tuple[str, str]] = []
    by_ext: Counter[str] = Counter()
    files = 0

    for path in iter_files(root):
        files += 1
        rel = path.relative_to(root)
        text = _read(path)
        by_ext[path.suffix.lower()] += count_tokens_local(text, model or "gpt-4o")
        if _is_prompt_file(rel):
            raw_prompts.append((str(rel), text))
        if path.suffix.lower() not in CODE_EXT:
            continue
        skip: set[int] = set()
        if path.suffix in (".py", ".ipynb"):
            try:
                skip = python_string_statement_lines(ast.parse(text))
            except SyntaxError:
                pass
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if i in skip or stripped.startswith(COMMENT_PREFIXES):
                continue
            for provider, kind, pat in CALL_PATTERNS:
                if pat.search(line):
                    call_sites.append(CallSite(str(rel), i, provider, kind, stripped[:120]))
            models.update(m.group(1) for m in MODEL_RE.finditer(line)
                          if not NOT_MODEL_SUFFIX.search(m.group(1)))
        found = _python_prompts(text) if path.suffix in (".py", ".ipynb") else _js_prompts(text)
        raw_prompts += [(f"{rel}:{ln}", s) for ln, s in found]

    # Count prompts with the model the repo uses most (Claude counts need the right tokenizer).
    count_model = model or next((m for m, _ in models.most_common() if m in pricing.models),
                                None) or (models.most_common(1)[0][0] if models else "gpt-4o")
    prompts = [PromptText(loc, count_tokens(s, count_model), " ".join(s.split())[:100])
               for loc, s in raw_prompts]
    prompts.sort(key=lambda p: p.tokens, reverse=True)
    static = sum(p.tokens for p in prompts)

    notes = []
    if not call_sites:
        notes.append("No known LLM SDK call sites found (the repo may use a wrapper or "
                     "another provider).")
    notes.append("Prompt tokens are static estimates. Runtime content (user input, retrieved "
                 "documents, history) and the number of calls per execution are only measured "
                 "by `token-eval run`.")
    return ScanReport(
        repo=str(root),
        files_scanned=files,
        codebase_tokens=sum(by_ext.values()),
        codebase_tokens_by_ext=dict(by_ext.most_common()),
        call_sites=call_sites,
        models=dict(models.most_common()),
        count_model=count_model,
        prompts=prompts,
        static_prompt_tokens=static,
        static_prompt_cost_usd=pricing.cost(count_model, static),
        notes=notes,
    )
