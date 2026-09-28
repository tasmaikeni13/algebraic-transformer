"""Inspect source and traced forward/backward graphs, including nested calls."""

import ast
from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from src.primitives import alu, avn, _alu_backward, _avn_backward

FORBIDDEN = {"exp", "expm1", "exp2", "log", "log1p", "log2", "log10", "sin", "cos",
             "tan", "tanh", "sinh", "cosh", "sigmoid", "logistic", "erf", "erfc", "pow"}


def source_audit(source):
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        if name in FORBIDDEN:
            violations.append({"line": node.lineno, "name": name})
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN:
                    violations.append({"line": node.lineno, "name": alias.name})
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                violations.append({"line": node.lineno, "name": "noninteger power"})
    return violations


def primitives_in(value):
    counts = Counter()
    if hasattr(value, "jaxpr"):
        return primitives_in(value.jaxpr)
    if hasattr(value, "eqns"):
        for eqn in value.eqns:
            counts[eqn.primitive.name] += 1
            for nested in eqn.params.values():
                counts.update(primitives_in(nested))
    elif isinstance(value, (tuple, list)):
        for item in value:
            counts.update(primitives_in(item))
    elif isinstance(value, dict):
        for item in value.values():
            counts.update(primitives_in(item))
    return counts


def audit():
    source = ROOT.joinpath("src/primitives.py").read_text()
    x = jnp.ones((2, 8), dtype=jnp.float32)
    traces = {
        "alu_forward": jax.make_jaxpr(alu)(x),
        "alu_vjp": jax.make_jaxpr(_alu_backward)(x, x),
        "avn_forward": jax.make_jaxpr(avn)(x),
        "avn_vjp": jax.make_jaxpr(lambda y, t, g: _avn_backward(1e-5, (y, t), g))(x, x[:, :1], x),
        "alu_grad": jax.make_jaxpr(jax.grad(lambda z: alu(z).sum()))(x),
        "avn_grad": jax.make_jaxpr(jax.grad(lambda z: avn(z).sum()))(x),
    }
    graphs = {name: dict(primitives_in(trace)) for name, trace in traces.items()}
    invalid = {name: sorted(set(ops) & FORBIDDEN) for name, ops in graphs.items()}
    backward_invalid = {name: sorted(set(graphs[name]) & {"rsqrt", "sqrt", "div", "pow"})
                        for name in ("alu_vjp", "avn_vjp")}
    # Remove strings/comments before the independent code-token regex audit.
    tokens = []
    import io
    import tokenize
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in {tokenize.COMMENT, tokenize.STRING}:
            tokens.append(token.string)
    regex_hits = re.findall(r"\b(?:exp|log|sin|cos|tanh|sigmoid)\b", " ".join(tokens))
    source_hits = source_audit(source)
    return {"source_violations": source_hits, "regex_hits": regex_hits, "graphs": graphs,
            "graph_violations": invalid, "backward_violations": backward_invalid,
            "passed": not (source_hits or regex_hits or any(invalid.values()) or any(backward_invalid.values()))}
