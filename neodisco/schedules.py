"""Disco's schedule strings, e.g. "[12]*400+[4]*600".

Cut counts in Disco are not constants; they change as sampling proceeds. The usual
setting starts with many overview cuts and few inner cuts, so the prompt first decides
the composition, then flips to few overview and many inner cuts, so the rest of the run
spends its effort on surface detail. That flip is a large part of the look.

The strings are written as Python expressions over 1000 diffusion steps. They are parsed
here rather than eval'd, and resampled to whatever step count is actually used.
"""

import ast
import math


MAX_SOURCE_CHARS = 4096
MAX_AST_NODES = 256
MAX_EXPANDED_VALUES = 10_000


def _evaluate(node):
    if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool) and math.isfinite(float(node.value))):
        return node.value
    if isinstance(node, ast.List):
        if len(node.elts) > MAX_EXPANDED_VALUES:
            raise ValueError('schedule expands beyond the 10000-value limit')
        return [_evaluate(e) for e in node.elts]
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Mult):
            if isinstance(left, list) and isinstance(right, int) and not isinstance(right, bool):
                if right < 0 or len(left) * right > MAX_EXPANDED_VALUES:
                    raise ValueError('schedule expands beyond the 10000-value limit')
                return left * right
            if isinstance(right, list) and isinstance(left, int) and not isinstance(left, bool):
                if left < 0 or len(right) * left > MAX_EXPANDED_VALUES:
                    raise ValueError('schedule expands beyond the 10000-value limit')
                return right * left
            raise ValueError('schedule multiplication requires a list and integer')
        if isinstance(node.op, ast.Add):
            if isinstance(left, list) and isinstance(right, list):
                if len(left) + len(right) > MAX_EXPANDED_VALUES:
                    raise ValueError('schedule expands beyond the 10000-value limit')
                return left + right
            if not isinstance(left, list) and not isinstance(right, list):
                return left + right
    raise ValueError('schedule may only contain lists, numbers, * and +')


def parse_schedule(spec, steps):
    """Return a list of length `steps`, one value per sampling step."""
    steps = int(steps)
    if not 1 <= steps <= 1000:
        raise ValueError('schedule steps must be between 1 and 1000')
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return [spec] * steps
    if isinstance(spec, (list, tuple)):
        values = list(spec)
    else:
        source = str(spec)
        if len(source) > MAX_SOURCE_CHARS:
            raise ValueError('schedule expression is too long')
        tree = ast.parse(source, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise ValueError('schedule expression is too complex')
        values = _evaluate(tree.body)
        if not isinstance(values, list):
            values = [values]
    if not values:
        raise ValueError('empty schedule')
    if len(values) > MAX_EXPANDED_VALUES:
        raise ValueError('schedule expands beyond the 10000-value limit')
    n = len(values)
    return [values[min(int(i * n / steps), n - 1)] for i in range(steps)]
