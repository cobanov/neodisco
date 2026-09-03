"""Disco's schedule strings, e.g. "[12]*400+[4]*600".

Cut counts in Disco are not constants; they change as sampling proceeds. The usual
setting starts with many overview cuts and few inner cuts, so the prompt first decides
the composition, then flips to few overview and many inner cuts, so the rest of the run
spends its effort on surface detail. That flip is a large part of the look.

The strings are written as Python expressions over 1000 diffusion steps. They are parsed
here rather than eval'd, and resampled to whatever step count is actually used.
"""

import ast


def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.List):
        return [_evaluate(e) for e in node.elts]
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
    raise ValueError('schedule may only contain lists, numbers, * and +')


def parse_schedule(spec, steps):
    """Return a list of length `steps`, one value per sampling step."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return [spec] * steps
    if isinstance(spec, (list, tuple)):
        values = list(spec)
    else:
        values = _evaluate(ast.parse(str(spec), mode='eval').body)
        if not isinstance(values, list):
            values = [values]
    if not values:
        raise ValueError('empty schedule')
    n = len(values)
    return [values[min(int(i * n / steps), n - 1)] for i in range(steps)]
