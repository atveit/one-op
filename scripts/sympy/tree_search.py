"""Explicit-witness search for depth-bounded EML trees.

For each target primitive (``add``, ``mul``, ``neg``, ``recip``, ``sqrt``),
enumerate binary trees of depth <= d over leaves ``{1, x, y}`` (or ``{1, x}``
for unary targets) where each internal node is ``eml(l, r) = exp(l) -
log(r)``. Simplify each resulting SymPy expression and test equality against
the target.

Search is bounded per primitive by (1) a time budget of 3 minutes and
(2) a tree-enumeration ceiling of 1,000,000 trees.

Hits are written to ``results/sympy/witnesses/<op>_tree.txt`` (ready for
translation to Lean), and a human summary is appended to
``results/sympy/tree_search.md``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import sympy as sp


# ---------------------------------------------------------------------------
# Tree representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leaf:
    name: str
    value: Any  # SymPy expression or Integer

    def to_str(self) -> str:
        return self.name

    def depth(self) -> int:
        return 0


@dataclass(frozen=True)
class Node:
    left: Any
    right: Any

    def to_str(self) -> str:
        return f"eml({self.left.to_str()}, {self.right.to_str()})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())


def tree_to_sympy(t):
    if isinstance(t, Leaf):
        return t.value
    return sp.exp(tree_to_sympy(t.left)) - sp.log(tree_to_sympy(t.right))


# ---------------------------------------------------------------------------
# Tree enumeration (depth-bounded, deduplicated by string form)
# ---------------------------------------------------------------------------

def enumerate_trees(leaves: list[Leaf], max_depth: int) -> Iterator[Any]:
    """Yield all distinct binary trees with the given leaves at depth <= d.

    Yields in increasing depth order. Depth-0 trees are just the leaves.
    """
    # cache: depth -> list of trees of exactly that depth
    by_depth: dict[int, list[Any]] = {0: list(leaves)}
    for t in leaves:
        yield t
    for d in range(1, max_depth + 1):
        new_level: list[Any] = []
        # trees at depth d have max(depth(L), depth(R)) = d-1
        # so at least one of L, R has depth d-1; the other has depth < d.
        for dl in range(0, d):
            for dr in range(0, d):
                if max(dl, dr) != d - 1:
                    continue
                for l in by_depth[dl]:
                    for r in by_depth[dr]:
                        new_level.append(Node(l, r))
                        yield new_level[-1]
        by_depth[d] = new_level


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

x_sym, y_sym = sp.symbols("x y", positive=True, real=True)


def _check(expr_target, candidate):
    """Return True iff ``candidate`` simplifies to ``expr_target``."""
    try:
        diff = sp.simplify(candidate - expr_target)
        if diff == 0:
            return True
        # Try stronger rewriting.
        diff2 = sp.simplify(sp.expand_log(sp.expand(diff), force=True))
        if diff2 == 0:
            return True
        return False
    except Exception:
        return False


TARGETS = {
    # op_name -> (sympy target, leaves, max_depth)
    "add": (x_sym + y_sym,
            [Leaf("1", sp.Integer(1)), Leaf("x", x_sym), Leaf("y", y_sym)], 7),
    "mul": (x_sym * y_sym,
            [Leaf("1", sp.Integer(1)), Leaf("x", x_sym), Leaf("y", y_sym)], 5),
    "neg": (-x_sym,
            [Leaf("1", sp.Integer(1)), Leaf("x", x_sym)], 5),
    "recip": (1 / x_sym,
             [Leaf("1", sp.Integer(1)), Leaf("x", x_sym)], 5),
    "sqrt": (sp.sqrt(x_sym),
             [Leaf("1", sp.Integer(1)), Leaf("x", x_sym)], 5),
}


# ---------------------------------------------------------------------------
# Search runner
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    op: str
    hit: bool
    tree_str: str | None
    tree_depth: int | None
    expr: Any | None
    trees_tried: int
    elapsed: float
    reason: str | None


def search_one(op: str, target, leaves: list[Leaf], max_depth: int,
               time_budget_s: float = 180.0,
               tree_budget: int = 1_000_000) -> SearchResult:
    start = time.time()
    tried = 0
    for t in enumerate_trees(leaves, max_depth):
        tried += 1
        if tried % 5000 == 0:
            if time.time() - start > time_budget_s:
                return SearchResult(op, False, None, None, None, tried,
                                    time.time() - start,
                                    "time budget exhausted")
            if tried > tree_budget:
                return SearchResult(op, False, None, None, None, tried,
                                    time.time() - start,
                                    "tree budget exhausted")
        expr = tree_to_sympy(t)
        if _check(target, expr):
            return SearchResult(op, True, t.to_str(), t.depth(), expr, tried,
                                time.time() - start, None)
    return SearchResult(op, False, None, None, None, tried,
                        time.time() - start,
                        f"exhausted depth <= {max_depth}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("results/sympy")
    witness_dir = out_dir / "witnesses"
    witness_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Explicit-witness search for EML depth theorems",
        "",
        "SymPy enumerates binary trees of depth <= d over the leaves ``{1, x, y}``,",
        "builds the nested ``exp(_) - log(_)`` expression, and checks equality to the",
        "target after ``sympy.simplify``. Time budget per primitive: 3 minutes.",
        "Tree budget: 1,000,000.",
        "",
        "| Op | Target depth | Found? | Witness depth | Trees tried | Elapsed (s) |",
        "|----|-------------|--------|---------------|-------------|-------------|",
    ]

    for op, (target, leaves, max_depth) in TARGETS.items():
        print(f"[tree_search] searching {op!r} at depth <= {max_depth} ...",
              flush=True)
        res = search_one(op, target, leaves, max_depth)
        hit_str = "yes" if res.hit else f"no ({res.reason})"
        wd_str = str(res.tree_depth) if res.tree_depth is not None else "-"
        lines.append(
            f"| {op} | {max_depth} | {hit_str} | {wd_str} | {res.trees_tried} | {res.elapsed:.1f} |"
        )
        if res.hit:
            out = witness_dir / f"{op}_tree.txt"
            out.write_text(
                f"# Explicit EML witness for {op}\n"
                f"# Target: {target}\n"
                f"# Depth: {res.tree_depth}\n"
                f"# Tree string (read-only):\n"
                f"{res.tree_str}\n"
                f"# SymPy expression (pre-simplify):\n"
                f"{res.expr}\n"
                f"# Simplified:\n"
                f"{sp.simplify(res.expr)}\n"
            )
            print(f"  HIT at depth {res.tree_depth}: {res.tree_str}",
                  flush=True)
        else:
            print(f"  miss ({res.reason})", flush=True)

    lines.append("")
    lines.append("## Witness details")
    lines.append("")
    for op in TARGETS:
        path = witness_dir / f"{op}_tree.txt"
        if path.exists():
            lines.append(f"### {op}")
            lines.append("```")
            lines.append(path.read_text().rstrip())
            lines.append("```")
            lines.append("")
        else:
            lines.append(f"### {op}")
            lines.append(
                "No witness found within depth bound and time budget."
            )
            lines.append("")

    (out_dir / "tree_search.md").write_text("\n".join(lines) + "\n")
    print(f"[tree_search] wrote {out_dir / 'tree_search.md'}", flush=True)


if __name__ == "__main__":
    main()
