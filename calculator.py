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

# I list every function I allow by name. If a name is not in this table
# I refuse it, because I never want a typed expression to reach a
# function I did not choose on purpose.
# I map both log and ln to math.log. People expect both names for the
# natural log, and I would rather accept both than teach that in an
# error message. I use the built in abs, round, min, and max here, not
# the math versions, because min and max need to take several arguments.
FUNCTIONS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log2": math.log2, "log10": math.log10, "ln": math.log,
    "exp": math.exp, "abs": abs, "floor": math.floor, "ceil": math.ceil,
    "round": round, "min": min, "max": max,
}
# I keep ans out of this table. It changes after every result, so I pass
# it through evaluate as a value instead of freezing it in a dict.
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
# I dispatch by operator node type, so an operator that is not in this
# table never reaches an implementation. Bitwise work and comparisons
# fall through in evaluate to my unsupported syntax error.
OPERATORS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


class CalculatorError(ValueError):
    """Raised for anything the calculator refuses to evaluate."""


def evaluate(node: ast.AST, ans: float):
    # I walk the tree with recursion because the tree itself is built
    # that way. Each BinOp holds its operands as smaller trees. That
    # choice is also why calculate has to expect RecursionError from
    # input that nests far deeper than any real expression should.
    if isinstance(node, ast.Expression):
        return evaluate(node.body, ans)
    if isinstance(node, ast.Constant):
        # I only accept numbers. A string or a bytes literal in an
        # arithmetic tool is input I do not trust and do not need.
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"unsupported constant: {node.value!r}")
    # I check the operator against my table before I touch the operands.
    # That order matters: I refuse a bad operator without walking into
    # subtrees I may not want to evaluate at all.
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.left, ans),
                                        evaluate(node.right, ans))
    if isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand, ans)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        # I land here for things like not and invert. I do not support
        # them, so I say so instead of guessing what the user meant.
        raise CalculatorError("unsupported unary operator")
    if isinstance(node, ast.Name):
        # I check ans before the fixed constants because ans is the one
        # name whose value belongs to this session, not to math.
        if node.id == "ans":
            return ans
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalculatorError(f"unknown name: {node.id}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in FUNCTIONS:
            raise CalculatorError(f"unknown function: {name}")
        # I reject keyword arguments outright. None of the functions I
        # allow need them, and accepting them would widen the surface
        # for no gain in a calculator.
        if node.keywords:
            raise CalculatorError("keyword arguments are not supported")
        # I evaluate the arguments in order and pass them as plain
        # positional values. A starred argument has no handler in this
        # function, so it falls to the error at the bottom.
        return FUNCTIONS[name](*[evaluate(a, ans) for a in node.args])
    # Anything I did not name above lands here: attribute access,
    # subscripts, comparisons, and every other node type. I refuse it
    # by name so the user can see what part I rejected.
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def calculate(expression: str, ans: float = 0.0):
    """Evaluate one expression string and return the numeric result."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"syntax error: {exc.msg}") from exc
    except MemoryError as exc:
        # The parser itself gives up on absurdly long operator chains
        # with a parser stack overflow. I treat that as an input
        # problem and report it like a syntax error, because the person
        # who typed the chain needs a shorter expression, not a crash.
        raise CalculatorError("expression is too complex to parse") from exc
    try:
        return evaluate(tree, ans)
    except RecursionError as exc:
        # A chain with thousands of terms can parse fine and still walk
        # thousands of frames deep in my evaluator. I refuse it as too
        # deep instead of letting the interpreter kill the process.
        raise CalculatorError("expression is nested too deeply") from exc
    except (ArithmeticError, ValueError, OverflowError) as exc:
        # I fold the math failures, like division by zero and a domain
        # error from sqrt of a negative, into my one error type. That
        # gives the REPL and the one shot mode a single thing to catch.
        raise CalculatorError(str(exc)) from exc


def format_result(value: float):
    # I print a whole float as an int. Seeing 4 instead of 4.0 is what
    # I want from a calculator for the common case.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    # I cap the other values at 12 significant digits. That hides the
    # long binary tail on results like 0.1 + 0.2 without hiding real
    # precision a user would care about.
    return f"{value:.12g}"


def repl():
    print("AST Scientific Calculator - type 'help' for functions, 'quit' to exit.")
    # I start ans at zero so a first expression that uses ans gets a
    # plain number instead of an error about an unset value.
    ans, history = 0.0, []
    while True:
        try:
            line = input("calc> ").strip()
        except (EOFError, KeyboardInterrupt):
            # I treat end of input and Ctrl-C as a clean exit. A piped
            # session that runs out of lines should end the same way a
            # person typing quit does, with no traceback.
            print()
            return
        if not line:
            continue
        # I check the session words before I try to calculate anything.
        # That order means quit, help, and history can never be taken
        # as names inside an expression while I am in the REPL.
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
            # I leave ans and history alone when a line fails. One bad
            # expression should not erase the last good result.
            print(f"error: {exc}")
            continue
        # I only record lines that worked. History is a record of
        # results, not a log of everything that was typed.
        history.append((line, ans))
        print(f"= {format_result(ans)}")


def main(argv):
    if len(argv) > 1:
        # I join the arguments back into one expression so a caller can
        # leave the quotes off and still pass something like 2 + 2 as
        # separate shell words.
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
