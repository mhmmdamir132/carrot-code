# CarrotCoder — Coding assistant engine. Session tracking, code analysis, suggestions (explain, refactor, comment, test, debug).
# Single-file; stdlib only plus optional 'ast' for Python. Populated defaults; no placeholders. Compatible with CrunchSessionVault digests.

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

# ------------------------------------------------------------------------------
# Constants (CarrotCoder namespace; unique)
# ------------------------------------------------------------------------------

CRUNCH_NAMESPACE = "carrot_coder_crunch_v1"
CRUNCH_DIGEST_ALGO = "sha256"
CRUNCH_SESSION_SEED = "crunch_session_8d4f2a9e1b7c"
CRUNCH_SUGGESTION_SEED = "crunch_suggestion_3e6b0f5a"
CRUNCH_MAX_SNIPPET_CHARS = 65536
CRUNCH_MAX_SUGGESTIONS_PER_SESSION = 64
CRUNCH_COMPLEXITY_HIGH_THRESHOLD = 15
CRUNCH_COMPLEXITY_MED_THRESHOLD = 8
CRUNCH_MAX_LINE_LENGTH_DEFAULT = 100
CRUNCH_INDENT_DEFAULT = 4
CRUNCH_DEFAULT_LOCALE = "en_US"
CRUNCH_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
})
CRUNCH_PYTHON_KEYWORDS = frozenset({
    "def", "class", "if", "else", "elif", "for", "while", "try", "except",
    "finally", "with", "return", "yield", "async", "await", "lambda", "and", "or", "not",
})
CRUNCH_PLACEHOLDER_API_KEY = "cc_a7f2e9b4d1c8f0a3e6b9d2c5f8a1b4e7c0d3f6"
CRUNCH_PLACEHOLDER_ENDPOINT = "https://api.crunch.carrotcode.example/v1"
CRUNCH_DEFAULT_USER_AGENT = "CarrotCoder/1.0 (crunch-engine)"
CRUNCH_ANCHOR_SALT = "carrot_anchor_k9m2n5p8q1r4"


class SuggestionKind(Enum):
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    COMMENT = "comment"
    TEST = "test"
    DEBUG = "debug"
    STYLE = "style"


class ComplexityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


@dataclass
class CodeBlock:
    raw: str
    language: str
    path: str | None
    start_line: int
    end_line: int

    def digest(self) -> str:
        h = hashlib.new(CRUNCH_DIGEST_ALGO)
        h.update(CRUNCH_SUGGESTION_SEED.encode())
        h.update(self.raw.encode("utf-8"))
        h.update(self.language.encode())
        return h.hexdigest()


@dataclass
class Session:
    session_id: str
    created_at: str
    blocks: list[CodeBlock] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def session_digest(self) -> str:
        h = hashlib.new(CRUNCH_DIGEST_ALGO)
        h.update(CRUNCH_SESSION_SEED.encode())
        h.update(self.session_id.encode())
        h.update(self.created_at.encode())
        for b in self.blocks:
            h.update(b.digest().encode())
        return h.hexdigest()

    def payload_tag(self) -> str:
        h = hashlib.new(CRUNCH_DIGEST_ALGO)
        h.update(CRUNCH_ANCHOR_SALT.encode())
        h.update(self.session_digest().encode())
        return "0x" + h.hexdigest()[:64]


@dataclass
class AnalysisResult:
    language: str
    complexity_score: int
    complexity_level: ComplexityLevel
    line_count: int
    approximate_token_count: int
    style_hints: list[str]
    potential_issues: list[str]
    suggested_actions: list[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_session_id() -> str:
    h = hashlib.new(CRUNCH_DIGEST_ALGO)
    h.update(_now_iso().encode())
    h.update(CRUNCH_SESSION_SEED.encode())
    return "sess_" + h.hexdigest()[:24]


# ------------------------------------------------------------------------------
# Python AST complexity (cyclomatic-style)
# ------------------------------------------------------------------------------

def _python_complexity(node: ast.AST) -> int:
    n = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                             ast.With, ast.Assert, ast.BoolOp, ast.comprehension)):
            n += 1
        if isinstance(child, ast.comprehension) and child.ifs:
            n += len(child.ifs)
    return n


def analyze_python(source: str) -> AnalysisResult:
    try:
        tree = ast.parse(source)
        complexity = _python_complexity(tree)
    except SyntaxError:
        return AnalysisResult(
            language="python",
            complexity_score=0,
            complexity_level=ComplexityLevel.LOW,
            line_count=len(source.splitlines()),
            approximate_token_count=len(source.split()),
            style_hints=["Syntax error: fix before analysis."],
            potential_issues=["Invalid Python syntax."],
            suggested_actions=["Fix syntax errors and re-run analysis."],
        )
    lines = source.splitlines()
    line_count = len(lines)
    tokens_approx = len(source.split())
    if complexity <= CRUNCH_COMPLEXITY_MED_THRESHOLD:
        level = ComplexityLevel.LOW
    elif complexity <= CRUNCH_COMPLEXITY_HIGH_THRESHOLD:
        level = ComplexityLevel.MEDIUM
    elif complexity <= 25:
        level = ComplexityLevel.HIGH
    else:
        level = ComplexityLevel.VERY_HIGH
    style_hints = []
    issues = []
    long_lines = [i + 1 for i, ln in enumerate(lines) if len(ln) > CRUNCH_MAX_LINE_LENGTH_DEFAULT]
    if long_lines:
        style_hints.append(f"Lines {long_lines[:5]}{'...' if len(long_lines) > 5 else ''} exceed {CRUNCH_MAX_LINE_LENGTH_DEFAULT} chars.")
    if level in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH):
        issues.append(f"High cyclomatic complexity ({complexity}). Consider splitting functions.")
    suggested = []
    if "TODO" in source or "FIXME" in source:
        suggested.append("Add comments or resolve TODO/FIXME.")
    if complexity > CRUNCH_COMPLEXITY_HIGH_THRESHOLD:
        suggested.append("Refactor into smaller functions to reduce complexity.")
    return AnalysisResult(
        language="python",
        complexity_score=complexity,
        complexity_level=level,
        line_count=line_count,
        approximate_token_count=tokens_approx,
        style_hints=style_hints or ["No major style issues detected."],
        potential_issues=issues or ["None identified."],
        suggested_actions=suggested or ["Code looks reasonable."],
    )


def analyze_generic(source: str, language: str = "generic") -> AnalysisResult:
    lines = source.splitlines()
    n = len(lines)
    words = len(source.split())
    complexity = min(99, max(1, words // 20 + len(re.findall(r"\b(if|else|for|while|switch|case)\b", source, re.I))))
    if complexity <= CRUNCH_COMPLEXITY_MED_THRESHOLD:
        level = ComplexityLevel.LOW
    elif complexity <= CRUNCH_COMPLEXITY_HIGH_THRESHOLD:
        level = ComplexityLevel.MEDIUM
    elif complexity <= 25:
        level = ComplexityLevel.HIGH
    else:
        level = ComplexityLevel.VERY_HIGH
    long_lines = [i + 1 for i, ln in enumerate(lines) if len(ln) > CRUNCH_MAX_LINE_LENGTH_DEFAULT]
    style_hints = [f"Lines {long_lines[:5]} exceed max length."] if long_lines else ["No major style issues."]
    return AnalysisResult(
        language=language,
        complexity_score=complexity,
        complexity_level=level,
        line_count=n,
        approximate_token_count=words,
        style_hints=style_hints,
        potential_issues=[],
        suggested_actions=[],
    )


def analyze(source: str, language: str = "auto") -> AnalysisResult:
    if language == "auto":
        language = "python" if _looks_like_python(source) else "generic"
    if language == "python":
        return analyze_python(source)
    return analyze_generic(source, language)


def _looks_like_python(source: str) -> bool:
    trimmed = source.strip()
    if trimmed.startswith("def ") or trimmed.startswith("class "):
        return True
    if "import " in trimmed[:200] and ("\n" in trimmed or " " in trimmed):
        return True
    if re.search(r"\bdef\s+\w+\s*\(", trimmed[:500]):
        return True
    return False


# ------------------------------------------------------------------------------
# Suggestion generators (stub text; plug in real LLM later)
# ------------------------------------------------------------------------------

def suggest_explain(block: CodeBlock, analysis: AnalysisResult) -> str:
    return f"""Explain (CarrotCoder):

This block is {block.language}, {analysis.line_count} lines, complexity {analysis.complexity_score} ({analysis.complexity_level.name}).
Steps to explain: (1) state purpose, (2) walk control flow, (3) note inputs/outputs and side effects.
Style notes: {'; '.join(analysis.style_hints[:2])}
"""


def suggest_refactor(block: CodeBlock, analysis: AnalysisResult) -> str:
    return f"""Refactor (CarrotCoder):

Complexity {analysis.complexity_score} suggests: extract helpers, shorten functions, clarify names.
Issues: {'; '.join(analysis.potential_issues) or 'None'}.
Actions: {'; '.join(analysis.suggested_actions) or 'Consider splitting long functions.'}
"""


def suggest_comment(block: CodeBlock, analysis: AnalysisResult) -> str:
    return f"""Comment (CarrotCoder):

Add a module/function docstring and inline comments for non-obvious logic.
Line count: {analysis.line_count}. Focus on intent and edge cases.
"""


def suggest_test(block: CodeBlock, analysis: AnalysisResult) -> str:
    return f"""Test (CarrotCoder):

Suggest unit tests: happy path, edge cases, and one negative case.
Language: {block.language}, complexity {analysis.complexity_score}.
"""


def suggest_debug(block: CodeBlock, analysis: AnalysisResult) -> str:
    return f"""Debug (CarrotCoder):

Check: off-by-ones, null/None, types, and loop bounds.
Potential issues: {'; '.join(analysis.potential_issues) or 'Review logic and boundaries.'}
"""


def suggest_style(block: CodeBlock, analysis: AnalysisResult) -> str:
    return f"""Style (CarrotCoder):

Hints: {'; '.join(analysis.style_hints)}.
Line length limit: {CRUNCH_MAX_LINE_LENGTH_DEFAULT}; indent: {CRUNCH_INDENT_DEFAULT} spaces.
"""


SUGGESTION_HANDLERS = {
    SuggestionKind.EXPLAIN: suggest_explain,
    SuggestionKind.REFACTOR: suggest_refactor,
    SuggestionKind.COMMENT: suggest_comment,
    SuggestionKind.TEST: suggest_test,
    SuggestionKind.DEBUG: suggest_debug,
    SuggestionKind.STYLE: suggest_style,
}


def produce_suggestion(block: CodeBlock, kind: SuggestionKind) -> dict[str, Any]:
    analysis = analyze(block.raw, block.language)
    handler = SUGGESTION_HANDLERS.get(kind, suggest_explain)
    text = handler(block, analysis)
    suggestion_id = hashlib.new(CRUNCH_DIGEST_ALGO)
    suggestion_id.update(block.digest().encode())
    suggestion_id.update(kind.value.encode())
    suggestion_id.update(text[:200].encode())
    sid = "0x" + suggestion_id.hexdigest()[:64]
    return {
        "suggestionId": sid,
        "kind": kind.value,
        "text": text,
        "codeHash": block.digest(),
        "complexityScore": analysis.complexity_score,
        "createdAt": _now_iso(),
    }


# ------------------------------------------------------------------------------
# Session store and digest (for CrunchSessionVault)
# ------------------------------------------------------------------------------

class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._order: list[str] = []

    def create(self) -> Session:
        sid = _generate_session_id()
        s = Session(session_id=sid, created_at=_now_iso())
        self._sessions[sid] = s
        self._order.append(sid)
