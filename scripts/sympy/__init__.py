"""SymPy complement layer to the Lean proofs.

Provides:
- ``primitives``: symbolic mirrors of ``eml_nn/arith.py``.
- ``tree_search``: explicit-witness search for ``add_tree_exists`` / ``mul_tree_exists``.
- ``latex_table``: auto-renders ``paper/table1_constructions.tex`` from ``ComponentCoverage.md``.
- ``derivative_supplement``: closed-form derivatives (appendix material).

SymPy is a **symbolic sanity check**, not load-bearing proof. See ``SymPyPlan.md``.
"""
