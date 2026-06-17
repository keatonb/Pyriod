# Pyriod/combinations.py
# Written by ChatGPT-5.5

from __future__ import annotations

import ast
import operator as op
from collections.abc import Mapping

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


class CombinationExpressionError(ValueError):
    """Raised when a combination-frequency expression is invalid."""


def _eval_node(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, values)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise CombinationExpressionError("Only numeric constants are allowed.")

    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key not in values:
            raise CombinationExpressionError(f"Unknown signal label: {node.id}")
        return float(values[key])

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise CombinationExpressionError(f"Operator {op_type.__name__} is not allowed.")
        return _ALLOWED_BINOPS[op_type](
            _eval_node(node.left, values),
            _eval_node(node.right, values),
        )

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise CombinationExpressionError(f"Operator {op_type.__name__} is not allowed.")
        return _ALLOWED_UNARYOPS[op_type](_eval_node(node.operand, values))

    raise CombinationExpressionError(
        f"Unsupported syntax in combination expression: {type(node).__name__}"
    )


def evaluate_combination(expr: str, values: Mapping[str, float]) -> float:
    """
    Safely evaluate expressions like 'f0+f1', '2*f0-f1', or '0.5*f2'.

    Parameters
    ----------
    expr
        Combination-frequency expression.
    values
        Mapping from signal labels, e.g. {'f0': 123.4, 'f1': 456.7}, to frequencies.

    Returns
    -------
    float
        Evaluated combination frequency.
    """
    try:
        tree = ast.parse(expr.replace(" ", "").lower(), mode="eval")
    except SyntaxError as exc:
        raise CombinationExpressionError(f"Invalid expression: {expr}") from exc

    return _eval_node(tree, {str(k).lower(): float(v) for k, v in values.items()})


def validate_combination(expr: str, known_labels: set[str]) -> bool:
    """Return True if expression is syntactically valid and uses known labels."""
    fake_values = {label.lower(): 1.0 for label in known_labels}

    try:
        value = evaluate_combination(expr, fake_values)
    except CombinationExpressionError:
        return False

    # Require at least one arithmetic operator and at least one known signal label.
    clean = expr.replace(" ", "").lower()
    has_operator = any(symbol in clean for symbol in "+-*/")
    has_label = any(label.lower() in clean for label in known_labels)

    return has_operator and has_label and value == value