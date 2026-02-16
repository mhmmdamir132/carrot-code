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
