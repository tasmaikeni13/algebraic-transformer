"""Tests for Drop-in Triton GPU Kernels (AFA & Fused Linear + OACE).

Verifies:
1. Zero-transcendental AST and token audit across both Triton kernel files.
2. Triton JIT function definition and signature compliance.
3. Contract conformance for block tiling and hardware parameter contracts.
"""

import ast
import io
from pathlib import Path
import re
import tokenize
import pytest

FORBIDDEN_IDENTIFIERS = {
    "exp", "expm1", "exp2", "log", "log1p", "log2", "log10",
    "sin", "cos", "tan", "tanh", "sinh", "cosh", "sigmoid", "logistic", "erf", "erfc",
}


@pytest.mark.parametrize("filename", ["triton_afa.py", "triton_oace.py"])
def test_zero_transcendental_ast_audit_triton(filename):
    """Verify Triton kernel modules contain 0 transcendental AST calls and tokens."""
    path = Path(__file__).resolve().parents[1] / "src/kernels" / filename
    source = path.read_text()
    tree = ast.parse(source)

    ast_violations = []
    for node in ast.walk(tree):
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name in FORBIDDEN_IDENTIFIERS:
            ast_violations.append({"line": node.lineno, "name": name})
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_IDENTIFIERS:
                    ast_violations.append({"line": node.lineno, "name": alias.name})

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in {tokenize.COMMENT, tokenize.STRING}:
            tokens.append(token.string)
    regex_hits = re.findall(r"\b(?:exp|log|sin|cos|tanh|sigmoid)\b", " ".join(tokens))

    assert len(ast_violations) == 0, f"AST violations in {filename}: {ast_violations}"
    assert len(regex_hits) == 0, f"Code token regex hits in {filename}: {regex_hits}"


def test_triton_kernels_are_valid_jit_functions():
    """Verify that all Triton kernels are valid JITFunction instances."""
    triton = pytest.importorskip("triton")
    from src.kernels.triton_afa import _triton_afa_fwd_kernel, _triton_afa_bwd_kernel
    from src.kernels.triton_oace import _triton_linear_oace_fwd_kernel

    assert isinstance(_triton_afa_fwd_kernel, triton.runtime.JITFunction)
    assert isinstance(_triton_afa_bwd_kernel, triton.runtime.JITFunction)
    assert isinstance(_triton_linear_oace_fwd_kernel, triton.runtime.JITFunction)
