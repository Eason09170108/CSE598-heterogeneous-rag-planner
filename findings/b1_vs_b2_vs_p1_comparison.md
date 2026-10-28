# B1 vs B2 vs P1: First Planner Result

**Context:** P1 is the first *planner* (decides store_subset per-question)
run via `scripts/planner.py` (`decide_stores_p1`), on the same n=20,
seed=42 sample used for B1/B2, so all three are directly row-comparable.
Raw results: `findings/p1_results/p1_n20_seed42.csv`.

P1's rule: always query V+G; add R only if the question contains a
drug-state keyword (solid/liquid/gas). On this sample it picked V+G for
19/20 questions and V+R+G for 1/20 (q_id=3811, which asks about
"solid-state drugs").

## Results

| Metric | B1 (V+R+G, fixed) | B2 (V only, fixed) | P1 (rule-based planner) |
|---|---|---|---|
| Retrieval hit rate | 50% (10/20) | 30% (6/20) | **50% (10/20)** |
| Answer-contains-gold-name (proxy) | 30% (6/20) | 25% (5/20) | **30% (6/20)** |
| Avg total latency | 30.05s | 20.80s | **23.70s** |
| Avg input tokens | 6,504 | 3,233 | **6,286** |
| Avg output tokens | 678.5 | 517.0 | 680.6 |
| Total input tokens (20 q) | 130,081 | 64,667 | 125,711 |

## Interpretation

**P1 matches B1's accuracy exactly** (retrieval hit and proxy rate
identical to B1 on this sample) while cutting latency by **21.1%**
(23.70s vs. 30.05s) — a real win, achieved purely by skipping R on the
19/20 questions where the state-keyword rule decided it wasn't needed,
with no accuracy cost on this sample (R only ever changes the answer
when it's actually relevant, and the rule correctly judged it wasn't
for 19 of these 20 questions).

The input-token saving is smaller (3.4%) than the latency saving (21.1%)
— R's own context is small (a short drug-table listing), so dropping it
barely shrinks the prompt, but R's *query* (a Postgres lookup) still
costs wall-clock time that V+G's embedding/BFS retrieval doesn't, so
most of the latency win is from retrieval-side time, not LLM time. (This
is a secondary effect — see `b1_vs_b2_comparison.md`'s point that >99%
of total latency is LLM time, not retrieval. The 21% drop here is a
smaller, retrieval-adjacent effect, not a token-driven LLM speedup.)

**P1 does not land between B1 and B2's accuracy — it matches B1's
exactly, at lower cost than B1 but higher cost than B2.** This is the
intended story for a cost-aware planner: pay B2-like savings only where
they're actually free (R is skippable almost every time in this sample),
rather than B2's blanket savings (which also drops G, costing 20pp of
retrieval hit rate as shown in `b1_vs_b2_comparison.md`).

## Caveats

1. **n=20 means P1's R-trigger case (1/20) is a single data point** —
   the 19/20 "always V+G" pattern reflects this sample's low rate of
   state-related questions, not necessarily the full dataset's rate.
   A larger run would validate whether R is triggered this rarely in
   general.
2. **P1 never drops G, and never drops V.** Its only lever is whether to
   add R. This is a known limitation (see `scripts/planner.py`'s
   module docstring) — it cannot discover that some questions don't
   need G either (which would be needed to approach B2-level savings on
   the subset of questions where V alone suffices), nor can it ever go
   below V+G in cost. A more flexible planner (P2 zero-shot-LLM-router,
   P3 cost-based) is needed to explore that larger decision space.
3. The accuracy numbers inherit the same caveats as
   `accuracy_metric_validation.md` (the substring proxy is validated as
   ~95% agreeing with an LLM judge but overstates true accuracy by
   ~5pp) and `retrieval_depth_investigation.md` (the ~61.5% miss-rate
   ceiling applies to all three of B1/B2/P1 equally, since they share
   the same executor).

## Implications for the project

1. **This is the core "cost-aware planning works" result for Phase 2.**
   A simple, interpretable rule beats a fixed-subset baseline (B1) on
   cost with zero accuracy loss on this sample, which is exactly the
   project's central claim (RQ1/RQ2).
2. **Next comparison point: B3 (zero-shot LLM router).** B3 asks the
   LLM itself which stores to query, as a non-rule-based point of
   reference for P1 — does an LLM router do better than this hand-written
   rule, and at what extra cost (an additional LLM call per question)?
3. P2/P3/P4 are the natural next planners once B3 exists, per the
   project's original staged design.
