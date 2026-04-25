---- MODULE PagedAttention ----
EXTENDS Naturals, Sequences

(*
  TLA+ specification of a simplified PagedAttention KV Cache Manager.
  TurboQuant compresses KV blocks, and the memory manager asynchronously
  allocates and evicts them across concurrent generation requests.
*)

CONSTANTS 
  MaxBlocks,     \* Maximum number of physical blocks in VRAM
  NumRequests    \* Number of concurrent requests

VARIABLES 
  free_blocks,   \* Set of currently available block IDs
  allocations,   \* Mapping from Request ID to a sequence of allocated block IDs
  state          \* State of each request (e.g., "generating", "done")

vars == <<free_blocks, allocations, state>>

Init ==
  /\ free_blocks = 1..MaxBlocks
  /\ allocations = [r \in 1..NumRequests |-> <<>>]
  /\ state = [r \in 1..NumRequests |-> "generating"]

(* A request allocates a new block if one is free *)
Allocate(r) ==
  /\ state[r] = "generating"
  /\ free_blocks /= {}
  /\ \E b \in free_blocks :
      /\ free_blocks' = free_blocks \ {b}
      /\ allocations' = [allocations EXCEPT ![r] = Append(allocations[r], b)]
      /\ state' = state

(* A request finishes and frees all its blocks *)
Finish(r) ==
  /\ state[r] = "generating"
  /\ state' = [state EXCEPT ![r] = "done"]
  /\ free_blocks' = free_blocks \cup {b \in 1..MaxBlocks : \E i \in 1..Len(allocations[r]) : allocations[r][i] = b}
  /\ allocations' = [allocations EXCEPT ![r] = <<>>]

Next ==
  \E r \in 1..NumRequests : Allocate(r) \/ Finish(r)

(* Invariants *)

(* No block is allocated to two different requests simultaneously (unless explicitly shared, simplified here) *)
NoDoubleAllocation ==
  \A r1, r2 \in 1..NumRequests :
    r1 /= r2 =>
      \A i \in 1..Len(allocations[r1]), j \in 1..Len(allocations[r2]) :
        allocations[r1][i] /= allocations[r2][j]

(* The total number of allocated blocks plus free blocks equals MaxBlocks *)
ConservationOfBlocks ==
  (Cardinality(free_blocks) + 
   (LET SumAlloc[r \in 0..NumRequests] == 
          IF r = 0 THEN 0 ELSE Len(allocations[r]) + SumAlloc[r-1]
    IN SumAlloc[NumRequests])) = MaxBlocks

TypeOK ==
  /\ free_blocks \subseteq 1..MaxBlocks
  /\ \A r \in 1..NumRequests : state[r] \in {"generating", "done"}

====