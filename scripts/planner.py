"""
planner.py

Stage 4: Planners.

P1 (rule-based): a simple, hand-written decision rule for which stores
to query per question, in contrast to B1/B2's *fixed* store_subset. This
is the first "planner" in the project's sense -- it decides the subset
per-question instead of using the same one for every question -- and is
intentionally simple (a few keyword/heuristic rules) so it's cheap to
reason about and serves as a lower bound that smarter planners (P2
zero-shot-LLM-router, P3 cost-based, P4 learned) should beat.

Rule (see findings/r1_validation.md and findings/b1_vs_b2_comparison.md
for the evidence behind each part):
  - V is always queried (it's cheap and is the anchor step G depends on).
  - G is always queried, because findings/r1_validation.md found ~88-92%
    of STaRK-Prime questions are relation-driven and need the graph --
    skipping G by default would be wrong most of the time.
  - R is added only if the question contains a drug-state keyword
    (solid/liquid/gas), the same keyword check executor.py's _retrieve_r
    already uses internally for its own state_filter -- R is scoped to
    drug.state only (findings/r_scope_decision.md), so it's only useful
    when that keyword is actually present.

This means P1 only ever picks one of two subsets: ("V","G") or
("V","R","G") -- it never picks V-alone or R/G-without-V, which is a
real limitation (see "Limitations" below), but it is a legitimate first
step up from B1/B2's fixed subsets and the natural one to measure first.

Usage (inside Colab, same client/stores setup as scripts/baselines.py):

    from scripts.stores import StoreManager
    from scripts.planner import run_planner

    stores = StoreManager(skb=skb, G=G, index=index, all_node_ids=all_node_ids, model=model)
    p1_df = run_planner(qa, stores, decide_stores_p1, n_samples=20, seed=42)
    p1_df.to_csv("/content/p1_results.csv", index=False)
"""

import random

import pandas as pd

from scripts.baselines import answer_contains_gold_name
from scripts.executor import STATE_KEYWORDS, run_executor

ALL_STORES_ORDER = ("V", "R", "G")  # canonical ordering for display/joining


def decide_stores_p1(question):
    """
    P1's rule: V and G always; add R only if a drug-state keyword
    (solid/liquid/gas) appears in the question. Returns a tuple ordered
    per ALL_STORES_ORDER for consistent display (e.g. "V+G" or "V+R+G").
    """
    chosen = {"V", "G"}
    question_lower = question.lower()
    if any(kw in question_lower for kw in STATE_KEYWORDS):
        chosen.add("R")
    return tuple(s for s in ALL_STORES_ORDER if s in chosen)


def run_planner(qa, stores, decide_stores_fn, n_samples=50, seed=42, split=None,
                 client=None, llm_model=None, verbose=True):
    """
    Like scripts.baselines.run_baseline, but the store_subset is decided
    per-question by `decide_stores_fn(question) -> tuple`, instead of
    being fixed for the whole run. Uses the SAME random sample (same
    seed) as run_baseline, so a planner's results are directly comparable
    row-for-row against B1/B2 on identical questions.

    Returns a pandas DataFrame with the same columns as run_baseline's
    output (store_subset now varies per row), so
    scripts.baselines.summarize_baseline works unchanged on the result.
    """
    rng = random.Random(seed)
    indices = list(range(len(qa)))
    if split is not None:
        indices = [i for i in indices if qa[i][3] == split]
    sample_indices = rng.sample(indices, min(n_samples, len(indices)))

    rows = []
    for n, i in enumerate(sample_indices):
        question, q_id, answer_ids, q_split = qa[i]
        gold_names = [stores.skb.node_info[aid]["name"] for aid in answer_ids]

        store_subset = decide_stores_fn(question)

        kwargs = {}
        if client is not None:
            kwargs["client"] = client
        if llm_model is not None:
            kwargs["llm_model"] = llm_model

        result = run_executor(question, store_subset, stores, **kwargs)

        retrieval_hit = bool(set(answer_ids) & set(result.retrieved_node_ids))
        proxy_hit = answer_contains_gold_name(result.answer, gold_names)

        rows.append({
            "q_id": q_id,
            "question": question,
            "split": q_split,
            "gold_answer_ids": "|".join(str(a) for a in answer_ids),
            "gold_names": "|".join(gold_names),
            "store_subset": "+".join(store_subset),
            "answer": result.answer,
            "retrieval_hit": retrieval_hit,
            "answer_contains_gold_name": proxy_hit,
            "retrieval_latency_sec": round(result.retrieval_latency_sec, 4),
            "llm_latency_sec": round(result.llm_latency_sec, 4),
            "total_latency_sec": round(result.total_latency_sec, 4),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        })

        if verbose:
            print(f"[{n+1}/{len(sample_indices)}] q_id={q_id} "
                  f"store_subset={'+'.join(store_subset)} "
                  f"retrieval_hit={retrieval_hit} proxy_hit={proxy_hit} "
                  f"latency={result.total_latency_sec:.1f}s")

    return pd.DataFrame(rows)
