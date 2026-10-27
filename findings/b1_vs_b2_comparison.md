# B1 (query-all) vs B2 (vector-only) Baseline Comparison

**Context:** First real experimental comparison, run via `scripts/baselines.py`
on `qwen38-27b` (ASU Voyager). Both runs use the same 20-question sample
(`seed=42`, no `split` filter — the STaRK-Prime `split` field appears
unset/`None` for all sampled questions, not yet investigated further).
Raw results: `findings/b1_results/b1_n20_seed42.csv` and
`findings/b1_results/b2_n20_seed42.csv`.

## Results

| Metric | B1 (V+R+G) | B2 (V only) | Change |
|---|---|---|---|
| Retrieval hit rate | 50% (10/20) | 30% (6/20) | **-20pp (-40% relative)** |
| Answer-contains-gold-name (proxy) | 30% (6/20) | 25% (5/20) | -5pp (-17% relative) |
| Avg total latency | 30.05s | 20.80s | **-30.8%** |
| Avg retrieval latency | 0.093s | 0.027s | -70.5% (both negligible vs. LLM time) |
| Avg LLM latency | 29.96s | 20.77s | -30.7% |
| Avg input tokens | 6,504 | 3,233 | **-50.3%** |
| Avg output tokens | 678.5 | 517.0 | -23.8% |
| Total input tokens (20 q) | 130,081 | 64,667 | -65,414 tokens |
| Total output tokens (20 q) | 13,570 | 10,340 | -3,230 tokens |

## Interpretation

Querying V alone instead of V+R+G cuts latency by ~31% and input tokens by
~50% — a substantial cost reduction, consistent with R/G's `context_text`
contributing most of the extra prompt length (B1's average input is 2x
B2's). This is expected: dropping two of three stores roughly halves the
context assembled per question.

The accuracy cost is real but smaller in relative terms on the proxy
metric (-17%) than on raw retrieval hit rate (-40%). Two caveats on reading
this gap:
1. `answer_contains_gold_name` is a crude substring proxy (see
   `scripts/baselines.py` docstring) — it both under- and over-counts
   relative to true correctness, so the 30%→25% drop should not be quoted
   as "only a 5-point accuracy cost" without that caveat attached.
2. **Latency dominates total cost, not retrieval.** Both B1 and B2 spend
   >99% of their time in the LLM call (29.96s / 20.77s out of ~30s / ~21s
   totals) — retrieval itself takes under 0.1s regardless of store subset.
   This means a planner's main lever for "cost savings" is *token count*
   (shorter context → fewer input tokens → somewhat faster LLM response),
   not retrieval time, which was already cheap in absolute terms.

## Implications for the project

1. **This is directly usable as a Phase 2 experimental result.** It
   empirically supports RQ1/RQ2: store selection has a measurable,
   non-trivial cost/accuracy tradeoff (31% latency, 50% input-token
   reduction vs. a real but smaller accuracy cost), which is exactly the
   tradeoff space a cost-aware planner should navigate.
2. **A rule-based planner (P1)'s job is now concretely scoped**: pick V-only
   when the question doesn't need R/G, V+R+G (or some other subset) when it
   does, aiming to land close to B1's accuracy at closer to B2's cost.
3. **The 61.5% retrieval-ceiling finding (see
   `retrieval_depth_investigation.md`) still applies to B1 here** — even
   the "best case" V+R+G baseline only hits 50% retrieval on this n=20
   sample (consistent with the earlier n=200 estimate's ~38.5% hit rate;
   small-sample variance expected at n=20).
4. Next baseline to add: **B3 (zero-shot LLM router)** — ask the LLM itself
   which stores to query before retrieving, as a non-rule-based point of
   comparison for P1.
