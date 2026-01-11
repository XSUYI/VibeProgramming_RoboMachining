from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ParsedMatrix:
    name: str
    value: np.ndarray


MATRIX_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[(?P<body>.*?)\]\s*(?P<transpose>'?)\s*;",
    re.DOTALL,
)


def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _normalize_body(body: str) -> str:
    body = body.replace("...", " ")
    body = body.replace(",", " ")
    body = re.sub(r"\s+", " ", body.strip())
    return body


def _parse_matrix_body(body: str) -> np.ndarray:
    rows = []
    for row_text in body.split(";"):
        row_text = row_text.strip()
        if not row_text:
            continue
        tokens = row_text.split()
        row = [_safe_eval(token) for token in tokens]
        rows.append(row)
    if not rows:
        raise ValueError("No rows detected in MATLAB matrix")
    return np.array(rows, dtype=float)


def _safe_eval(expr: str) -> float:
    allowed = {"pi": math.pi}
    return float(eval(expr, {"__builtins__": {}}, allowed))


def parse_matrices(text: str) -> Dict[str, np.ndarray]:
    cleaned = _strip_comments(text)
    matrices: Dict[str, np.ndarray] = {}
    for match in MATRIX_PATTERN.finditer(cleaned):
        name = match.group("name")
        body = _normalize_body(match.group("body"))
        matrix = _parse_matrix_body(body)
        if match.group("transpose"):
            matrix = matrix.T
        matrices[name] = matrix
    return matrices


def extract_matrix(
    text: str,
    preferred_names: Iterable[str],
    expected_shape: Optional[Tuple[int, int]] = None,
) -> ParsedMatrix:
    matrices = parse_matrices(text)
    for name in preferred_names:
        if name in matrices:
            matrix = matrices[name]
            if expected_shape and matrix.shape != expected_shape:
                continue
            return ParsedMatrix(name=name, value=matrix)
    if expected_shape:
        for name, matrix in matrices.items():
            if matrix.shape == expected_shape:
                return ParsedMatrix(name=name, value=matrix)
    raise ValueError("Required MATLAB matrix not found")
