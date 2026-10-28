# CSE598 Heterogeneous RAG Planner

**Cost-Aware Query Planning for Heterogeneous RAG over Vector, Relational, and Graph Stores**

CSE 598 (Data-Intensive Systems for Machine Learning) group project. We
build a RAG pipeline over the [STaRK-Prime](https://github.com/snap-stanford/stark)
benchmark (biomedical entities/relations) backed by three heterogeneous
stores -- a vector store (V), a relational store (R), and a knowledge
graph (G) -- and study how to decide, per question, which subset of
stores to query, trading off cost (latency, tokens) against answer
accuracy.

## Project structure

```
scripts/
  setup_stark_env.py   # Colab setup: installs stark-qa, builds all three stores
  stores.py            # StoreManager: query_v / query_r / query_g wrappers
  executor.py          # Stage 2: given a question + store_subset, retrieve + generate an answer
  labeling.py          # Execution-derived labeling: brute-force search over all 7 store subsets
  baselines.py         # B1 (query-all) and B2 (vector-only) fixed-subset baselines
  b3_llm_router.py     # B3: zero-shot LLM router baseline (no hand-written rule)
  planner.py           # P1: rule-based planner
  planner_p3.py        # P3: cost-based planner (explicit, tunable cost/benefit threshold)
  llm_judge.py                     # LLM-as-judge accuracy scoring (same-backend)
  build_gemini_judge_prompt.py     # Batches answers into one prompt for an external judge model
  apply_external_judge_results.py # Merges an external judge's verdicts back in
  build_manual_validation_sheet.py # Builds a sheet for human correctness judging
  score_manual_validation.py       # Compares the free proxy metric against a reference judgement

findings/              # Write-ups and raw result CSVs for every experiment (see below)
```

## Setup (Google Colab)

1. Run `scripts/setup_stark_env.py` -- installs `stark-qa` and its
   dependencies, downloads the STaRK-Prime knowledge base (`skb`) and
   QA set (`qa`), and builds the PostgreSQL `drug` table (R).
2. Build the FAISS vector index (V) and the NetworkX graph (G) from
   `skb` (see `stores.py` for the expected shape: `index`,
   `all_node_ids`, `model`, `G`).
3. Set `LLM_API_KEY` (and optionally `LLM_BASE_URL` / `LLM_MODEL`) --
   defaults target ASU Voyager's free OpenAI-compatible gateway
   (`qwen38-27b`). See `executor.py`'s docstring for details.
4. Clone this repo inside Colab and add it to `sys.path`, then import
   from `scripts.*` as shown in each module's own docstring.

## Current results (n=20, seed=42, same sample across all runs)

| | B1 (V+R+G) | B2 (V only) | P1 (rule) | B3 (LLM router) | P3 (α=5e-5) | P3 (α=2e-4) |
|---|---|---|---|---|---|---|
| Retrieval hit rate | 50% | 30% | 50% | 50% | 50% | 30% |
| Proxy accuracy | 30% | 25% | 30% | 25% | 30% | 30% |
| Avg total latency | 30.05s | 20.80s | 23.70s | 31.30s | 20.60s | **16.97s** |
| Avg input tokens | 6,504 | 3,233 | 6,286 | 6,513 | 6,286 | 3,245 |

**Takeaway:** P3's tunable cost/benefit threshold (`alpha`) reproduces
both B1's accuracy and (close to) B2's cost by moving a single
parameter -- one mechanism spans the range that B1/B2 define as fixed
endpoints, while B3 (asking an LLM zero-shot) actually costs *more*
than B1 without matching its accuracy. See `findings/` for the full
write-up behind every number above, including known caveats (proxy
metric validation, a retrieval-recall ceiling shared by all methods,
and a parametric-knowledge confound found while running P3).

## Findings index

| File | What it covers |
|---|---|
| `v_validation.md`, `r_scope_decision.md`, `r1_validation.md` | Early validation of V's retrieval quality and R's scope (drug physical-state only) |
| `retrieval_depth_investigation.md` | Why single-hop/single-anchor retrieval misses ~73% of answers, the fix (2-hop BFS, top-3 anchors) that got it to ~61.5%, and a reverted attempt that made it worse |
| `accuracy_metric_validation.md` | Validates the free substring-match proxy against an external LLM judge (95% agreement, proxy overstates accuracy by ~5pp) |
| `b1_vs_b2_comparison.md` | First baseline comparison: query-all vs. vector-only |
| `b1_vs_b2_vs_p1_comparison.md` | P1 (rule-based) matches B1's accuracy at lower cost |
| `b1_b2_p1_b3_comparison.md` | B3 (zero-shot LLM router) degenerates to a constant decision; costs more than B1, undercuts P1 |
| `p3_cost_accuracy_frontier.md` | P3 (cost-based, tunable) traces a frontier spanning B1<->B2; documents a proxy-accuracy anomaly worth flagging as a limitation |

## Status

Baselines B1/B2/B3 and planners P1/P3 are built and compared on a
shared n=20 sample. A learned planner (P4) and larger-scale runs are
possible next steps, time permitting, ahead of the Phase 2 (slides)
and Phase 3 (report) deadlines.
