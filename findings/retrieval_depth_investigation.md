# Retrieval Depth Investigation: Why Single-Hop Anchoring Fails

**Context:** Running `scripts/labeling.py` (execution-derived labeling) on a
200-question sample surfaced a major problem *before* any planner work could
proceed: the retrieval strategy itself, not store selection, was the
bottleneck. This note documents the investigation, including an attempted
fix that made things worse — a negative result, included because it's
informative about why the problem is hard.

## Method

`label_dataset(qa, stores, n_samples=200, seed=42)` was run after each
retrieval change below. The key metric is the fraction of questions where
**no store subset (including V+R+G combined) retrieves the gold answer
node** — i.e. retrieval itself fails, independent of which stores a planner
would pick.

## Results

| Retrieval strategy | "No subset hits" rate (/200) | Notes |
|---|---|---|
| V top-5 → single top-1 anchor → G 1-hop | 146 (73.0%) | Original Stage 2 baseline |
| V top-10 → top-3 anchors → G 2-hop BFS, node cap 50 | 127 (63.5%) | Cap hit almost immediately after hop 1, limiting effective depth |
| Same, node cap raised to 200 | 123 (61.5%) | Best result so far; marginal gain over cap=50, confirming the cap wasn't the main bottleneck |
| Multi-symptom-intersection (phrase-split V anchors, G 1-hop, rank by cross-phrase vote count) | 141 (70.5%) | **Worse than the simple BFS baseline above — reverted.** |

## Diagnosis (why top-3-anchor BFS still misses ~61%)

Manual inspection of a miss case: the question "Could you assist in
identifying a hereditary neurological disease that causes night blindness,
severe vision impairment from early childhood, and learning challenges?"
(gold answer: **infantile Refsum disease**) — V's whole-question top-10
candidates were all diseases/phenotypes semantically matching "night
blindness" (e.g. "Incomplete congenital stationary night blindness",
similarity 0.639). None were the gold answer. The embedding anchors on
whichever single symptom phrase dominates the question's overall semantic
content and ignores the others ("learning challenges", "early-childhood
onset") entirely.

## Why the multi-symptom-intersection fix made it worse, not better

The hypothesis was: split the question into per-symptom phrases, find a V
anchor per phrase, expand each 1-hop via G, and rank candidates by how many
*distinct* phrases' anchors reach them — a disease satisfying multiple
symptoms should outrank one satisfying only one.

This regressed performance (70.5% vs. 61.5% miss rate) for two likely
reasons, not yet disambiguated:

1. **Intersection is an AND over a possibly-incomplete graph.** If even one
   symptom's disease–phenotype edge is missing from STaRK-Prime's KG (a real
   possibility — knowledge graphs are rarely complete), that phrase
   contributes zero votes for the gold node, and the vote-count ranking
   never recovers it — whereas the simpler BFS, with no such AND
   requirement, was more forgiving of partial graph coverage.
2. **Phrase quality.** The regex-based splitter (`_split_into_phrases`) can
   produce low-quality fragments (e.g. leading clauses like "Could you
   assist in identifying a hereditary neurological disease that causes")
   that are not real symptom descriptions, diluting the per-phrase V search
   with an irrelevant "phrase."

Both are testable in isolation (e.g., rank by *max* vote instead of
requiring agreement, or hand-filter phrase quality on a sample) but were not
pursued further given the Nov 21 deadline — this is flagged as future work
below.

## Decision

Reverted to the 2-hop BFS / top-3-anchor / node-cap-200 strategy (61.5% miss
rate) as the current default in `scripts/executor.py`. This is the best
result obtained so far and is what Stage 2's executor and labeling currently
use.

## Implications for the project

1. **This is itself a legitimate research finding, not just an engineering
   blocker.** It directly supports the project's premise that naive
   single-pass retrieval is insufficient for heterogeneous, multi-hop KG
   question answering — motivating the need for a more deliberate query
   execution strategy, which is squarely in scope for "Technical
   Challenges" (Phase 2 §3.4 / Phase 3 §3.3) and as an "unsuccessful
   approach" (Phase 3 §3.1 explicitly asks for these).
2. **The 61.5% miss rate is a ceiling on any planner's achievable accuracy**
   under the current executor, regardless of which store subset a planner
   picks — this needs to be reported honestly alongside planner results,
   not hidden, and should be framed as "retrieval recall" separate from
   "store-selection accuracy."
3. **Future work candidates** (not yet implemented): (a) rank
   intersection candidates by max-vote rather than requiring multi-phrase
   agreement, relaxing the strict AND; (b) use an LLM (or simpler NER) to
   extract clean symptom/entity phrases instead of regex splitting on
   punctuation/conjunctions; (c) increase BFS hops further now that the
   node cap is less of a bottleneck, trading latency for recall — this
   latency/recall tradeoff is itself relevant to the project's cost-aware
   framing and worth an explicit experiment.
