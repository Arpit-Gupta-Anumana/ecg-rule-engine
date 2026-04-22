"""Minimal, safe expression language for derived feature terms in the Rule DSL.

Supported grammar:

    expr    := term (('+' | '-') term)*
    term    := factor (('*' | '/') factor)*
    factor  := NUMBER | IDENT | func '(' arglist ')' | '(' expr ')' | '-' factor | 'abs' '(' expr ')'
    func    := 'max' | 'min' | 'abs' | 'sum' | 'avg'
    arglist := expr (',' expr)*

- IDENTs must be known feature names (validated at DSL load time against the registry).
- No attribute access, no arbitrary calls, no builtins.
- Evaluation takes a `Mapping[str, float]` and returns a `float`.

This is deliberately much smaller than Python's expression grammar so every rule
remains auditable. The full AST is retained so traces can pretty-print the
expression with substituted values.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

from ..features.registry import is_known_feature

ALLOWED_FUNCS: frozenset[str] = frozenset({"max", "min", "abs", "sum", "avg"})


class ExprError(ValueError):
    """Raised when an expression is malformed or references unknown features."""


# --- AST nodes ---------------------------------------------------------------

@dataclass(frozen=True)
class Num:
    value: float

    def eval(self, env: Mapping[str, float]) -> float:
        return self.value

    def features(self) -> set[str]:
        return set()

    def pretty(self, env: Mapping[str, float]) -> str:
        return _fmt(self.value)


@dataclass(frozen=True)
class Var:
    name: str

    def eval(self, env: Mapping[str, float]) -> float:
        if self.name not in env:
            raise ExprError(f"Feature '{self.name}' missing from input feature vector")
        v = env[self.name]
        if v is None:
            raise ExprError(f"Feature '{self.name}' is None")
        return float(v)

    def features(self) -> set[str]:
        return {self.name}

    def pretty(self, env: Mapping[str, float]) -> str:
        if self.name in env and env[self.name] is not None:
            return f"{self.name}={_fmt(env[self.name])}"
        return self.name


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Node"
    right: "Node"

    def eval(self, env: Mapping[str, float]) -> float:
        a = self.left.eval(env)
        b = self.right.eval(env)
        if self.op == "+": return a + b
        if self.op == "-": return a - b
        if self.op == "*": return a * b
        if self.op == "/":
            if b == 0:
                raise ExprError("Division by zero")
            return a / b
        raise ExprError(f"Unknown operator: {self.op}")

    def features(self) -> set[str]:
        return self.left.features() | self.right.features()

    def pretty(self, env: Mapping[str, float]) -> str:
        return f"({self.left.pretty(env)} {self.op} {self.right.pretty(env)})"


@dataclass(frozen=True)
class Neg:
    node: "Node"

    def eval(self, env: Mapping[str, float]) -> float:
        return -self.node.eval(env)

    def features(self) -> set[str]:
        return self.node.features()

    def pretty(self, env: Mapping[str, float]) -> str:
        return f"-{self.node.pretty(env)}"


@dataclass(frozen=True)
class Call:
    func: str
    args: tuple["Node", ...]

    def eval(self, env: Mapping[str, float]) -> float:
        vals = [a.eval(env) for a in self.args]
        if self.func == "max": return max(vals)
        if self.func == "min": return min(vals)
        if self.func == "abs":
            if len(vals) != 1:
                raise ExprError("abs() takes exactly one argument")
            return abs(vals[0])
        if self.func == "sum": return sum(vals)
        if self.func == "avg":
            if not vals:
                raise ExprError("avg() requires at least one argument")
            return sum(vals) / len(vals)
        raise ExprError(f"Unknown function: {self.func}")

    def features(self) -> set[str]:
        s: set[str] = set()
        for a in self.args:
            s |= a.features()
        return s

    def pretty(self, env: Mapping[str, float]) -> str:
        inner = ", ".join(a.pretty(env) for a in self.args)
        return f"{self.func}({inner})"


Node = Union[Num, Var, BinOp, Neg, Call]


# --- Tokenizer ---------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    \s* (?:
        (?P<NUMBER>\d+(?:\.\d+)?)
      | (?P<IDENT>[A-Za-z_][A-Za-z_0-9]*)
      | (?P<OP>[+\-*/(),])
    )
    """,
    re.VERBOSE,
)


def _tokenize(s: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m or m.end() == pos:
            if s[pos].isspace():
                pos += 1
                continue
            raise ExprError(f"Unexpected character '{s[pos]}' at position {pos} in: {s!r}")
        kind = m.lastgroup or "?"
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    tokens.append(("EOF", ""))
    return tokens


# --- Parser (recursive descent) ---------------------------------------------

class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> tuple[str, str]:
        return self.toks[self.i]

    def _eat(self, kind: str, value: str | None = None) -> tuple[str, str]:
        k, v = self._peek()
        if k != kind or (value is not None and v != value):
            raise ExprError(f"Expected {kind}{'=' + value if value else ''}, got {k}={v!r}")
        self.i += 1
        return k, v

    def parse(self) -> Node:
        node = self._expr()
        if self._peek()[0] != "EOF":
            raise ExprError(f"Trailing tokens: {self.toks[self.i:]}")
        return node

    def _expr(self) -> Node:
        node = self._term()
        while self._peek() == ("OP", "+") or self._peek() == ("OP", "-"):
            _, op = self._eat("OP")
            rhs = self._term()
            node = BinOp(op, node, rhs)
        return node

    def _term(self) -> Node:
        node = self._factor()
        while self._peek() == ("OP", "*") or self._peek() == ("OP", "/"):
            _, op = self._eat("OP")
            rhs = self._factor()
            node = BinOp(op, node, rhs)
        return node

    def _factor(self) -> Node:
        k, v = self._peek()
        if k == "OP" and v == "-":
            self._eat("OP")
            return Neg(self._factor())
        if k == "OP" and v == "(":
            self._eat("OP")
            inner = self._expr()
            self._eat("OP", ")")
            return inner
        if k == "NUMBER":
            self._eat("NUMBER")
            return Num(float(v))
        if k == "IDENT":
            self._eat("IDENT")
            if self._peek() == ("OP", "("):
                self._eat("OP")
                args: list[Node] = []
                if self._peek() != ("OP", ")"):
                    args.append(self._expr())
                    while self._peek() == ("OP", ","):
                        self._eat("OP")
                        args.append(self._expr())
                self._eat("OP", ")")
                if v not in ALLOWED_FUNCS:
                    raise ExprError(f"Unknown function '{v}'. Allowed: {sorted(ALLOWED_FUNCS)}")
                return Call(v, tuple(args))
            return Var(v)
        raise ExprError(f"Unexpected token {k}={v!r}")


# --- Public API --------------------------------------------------------------

def parse_expr(s: str) -> Node:
    """Parse a feature expression into an AST. Raises ExprError on failure."""
    if not s or not s.strip():
        raise ExprError("Empty expression")
    return _Parser(_tokenize(s)).parse()


def validate_features(node: Node) -> list[str]:
    """Return a list of feature names in `node` that are NOT in the registry."""
    return sorted(f for f in node.features() if not is_known_feature(f))


def _fmt(x: float) -> str:
    if x == int(x):
        return str(int(x))
    return f"{x:.3f}".rstrip("0").rstrip(".")


__all__ = [
    "Node", "Num", "Var", "BinOp", "Neg", "Call",
    "parse_expr", "validate_features", "ExprError", "ALLOWED_FUNCS",
]
