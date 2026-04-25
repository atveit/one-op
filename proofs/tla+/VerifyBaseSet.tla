---- MODULE VerifyBaseSet ----
(***************************************************************************
  TLA+ specification of Odrzywolek's VerifyBaseSet search procedure
  (arXiv:2603.21852, 2026, Section 2).

  State machine:
    S_0 = {"1", "eml"}             (verified primitives)
    C_0 = Primitives \ S_0         (to-be-computed)
  At each step, pick some p in C_i, "verify" it (find an expression using
  only S_i), and move p from C to S.  Terminates when C is empty.

  The model is abstracted: we assume that for any p in pending, an
  expression using only verified primitives is eventually discovered.
  This matches the paper's claim that the base set is closed under the
  search.
 ***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS Primitives   \* finite set of symbolic primitives to verify

VARIABLES verified, pending, step
vars == <<verified, pending, step>>

TypeOk ==
  /\ verified \subseteq (Primitives \cup {"1", "eml"})
  /\ pending  \subseteq Primitives
  /\ step \in Nat

Init ==
  /\ verified = {"1", "eml"}
  /\ pending  = Primitives \ verified
  /\ step = 0

Verify(p) ==
  /\ p \in pending
  /\ verified' = verified \cup {p}
  /\ pending'  = pending \ {p}
  /\ step'     = step + 1

Done ==
  /\ pending = {}
  /\ UNCHANGED vars

Next == (\E p \in pending : Verify(p)) \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(\E p \in pending : Verify(p))

(*************************** Safety invariants ***************************)

\* verified and pending are disjoint at every reachable state.
Disjoint == verified \cap pending = {}

\* Conservation: the union of verified and pending covers Primitives \cup {"1","eml"}.
Conservation ==
  verified \cup pending = Primitives \cup {"1", "eml"}

(*************************** Liveness ************************************)

\* The procedure terminates: pending eventually empties.
Termination == <>(pending = {})

====
