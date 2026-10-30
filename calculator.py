#!/usr/bin/env python3
"""AST Scientific Calculator - a command-line calculator that never uses eval().

Evaluates arithmetic expressions WITHOUT eval(): the input is parsed with
the `ast` module and only whitelisted math nodes are executed.

Usage:
    python3 calculator.py                  # interactive REPL
    python3 calculator.py "2 * (3 + pi)"   # one-shot evaluation
    python3 calculator.py --history        # show session history (REPL only)

Supports: + - * / // % ** parentheses, unary minus, the constant `ans`
(previous result), constants pi / e / tau, and the functions sqrt, sin,
cos, tan, asin, acos, atan, log, log2, log10, ln, exp, abs, floor, ceil,
round, min, max.
"""
import ast
import math
import sys

FUNCTIONS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log2": math.log2, "log10": math.log10, "ln": math.log,
    "exp": math.exp, "abs": abs, "floor": math.floor, "ceil": math.ceil,
    "round": round, "min": min, "max": max,
}
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
# Dispatch is by operator node type, so an operator that is not in this
# table (bitwise ops, comparisons) never reaches an implementation: the
# BinOp check in evaluate() falls through to the unsupported-syntax error.
OPERATORS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


class CalculatorError(ValueError):
    """Raised for anything the calculator refuses to evaluate."""


def evaluate(node: ast.AST, ans: float):
    # The walk is recursive because the parse tree is recursive: each BinOp
    # holds its operands as subtrees. That is also why calculate() has to
    # expect RecursionError from pathological input (see below).
    if isinstance(node, ast.Expression):
        return evaluate(node.body, ans)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.left, ans),
                                        evaluate(node.right, ans))
    if isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand, ans)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise CalculatorError("unsupported unary operator")
    if isinstance(node, ast.Name):
        if node.id == "ans":
            return ans
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalculatorError(f"unknown name: {node.id}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in FUNCTIONS:
            raise CalculatorError(f"unknown function: {name}")
        if node.keywords:
            raise CalculatorError("keyword arguments are not supported")
        return FUNCTIONS[name](*[evaluate(a, ans) for a in node.args])
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def calculate(expression: str, ans: float = 0.0):
    """Evaluate one expression string and return the numeric result."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"syntax error: {exc.msg}") from exc
    except MemoryError as exc:
        # The parser itself gives up on absurdly long operator chains with
        # "Parser stack overflowed". That is an input problem, so report it
        # the same way as a syntax error instead of dying with a traceback.
        raise CalculatorError("expression is too complex to parse") from exc
    try:
        return evaluate(tree, ans)
    except RecursionError as exc:
        # "1+1+1+..." thousands of terms deep parses fine but walks thousands
        # of frames deep here. Refuse it as too complex, do not crash.
        raise CalculatorError("expression is nested too deeply") from exc
    except (ArithmeticError, ValueError, OverflowError) as exc:
        raise CalculatorError(str(exc)) from exc


def format_result(value: float):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


def repl():
    print("AST Scientific Calculator - type 'help' for functions, 'quit' to exit.")
    ans, history = 0.0, []
    while True:
        try:
            line = input("calc> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            return
        if line.lower() == "help":
            print("functions:", ", ".join(sorted(FUNCTIONS)))
            print("constants:", ", ".join(sorted(CONSTANTS)), "+ ans")
            continue
        if line.lower() == "history":
            for expr, result in history:
                print(f"  {expr} = {format_result(result)}")
            continue
        try:
            ans = calculate(line, ans)
        except CalculatorError as exc:
            print(f"error: {exc}")
            continue
        history.append((line, ans))
        print(f"= {format_result(ans)}")


def main(argv):
    if len(argv) > 1:
        expression = " ".join(argv[1:])
        try:
            print(format_result(calculate(expression)))
        except CalculatorError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
