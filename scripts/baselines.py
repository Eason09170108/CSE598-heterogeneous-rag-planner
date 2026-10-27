"""
baselines.py

Stage 3: Baselines.

B1 (query-all): always query V+R+G regardless of the question, then
generate an answer. This is the "naive" baseline every planner (P1-P4)
needs to beat on cost (latency/tokens) without losing much accuracy -- or,
if it can't beat it on accuracy either, that's itself a result worth
reporting honestly.

B2 (vector-only): always query V alone. Cheaper than B1, but expected to
do worse on relation-driven questions (see findings/v_validation.md and
findings/r1_validation.md, which found ~88-92% of questions are relation-
driven and need G).

Both reuse scripts/executor.run_executor, so cost accounting (retrieval
latency, LLM latency, input/output tokens) is directly comparable to
whatever a planner does later -- same code path, just a fixed store_subset
instead of a decided one.

Answer correctness: STaRK-Prime answers are open-ended text, so there is
no automatic exact-match judge here. `answer_contains_gold_name` is a
CRUDE proxy -- case-insensitive substring check of the gold node's name
in the generated answer -- included only because it's free (no extra LLM
call) and gives a rough signal while iterating. It will under-count
correct answers that paraphrase the gold name and over-count answers that
happen to mention the right name without actually answering the question.
A real accuracy metric (e.g. LLM-as-judge, or human-labeled sample) is
listed as future work -- do not report this proxy alone as "accuracy" in
the final report without flagging its limitation.

Usage (inside Colab):

    from scripts.stores import StoreManager
    from scripts.baselines import run_baseline

    stores = StoreManager(skb=skb, G=G, index=index, all_node_ids=all_node_ids, model=model)
    b1_df = run_baseline(qa, stores, store_subset=("V", "R", "G"), n_samples=50, seed=42)
    b1_df.to_csv("/content/b1_results.csv", index=False)

    b2_df = run_baseline(qa, stores, store_subset=("V",), n_samples=50, seed=42)
    b2_df.to_csv("/content/b2_results.csv", index=False)
"""

import random
import time

import pandas as pd

from scripts.executor import run_executor


def answer_contains_gold_name(answer_text, gold_names):
    """
    Crude correctness proxy: True if any gold answer node's name appears
    (case-insensitive substring) in the generated answer text. See module
    docstring for this metric's known limitations.
    """
    answer_lower = (answer_text or "").lower()
    return any(name.lower() in answer_lower for name in gold_names if name)


def run_baseline(qa, stores, store_subset, n_samples=50, seed=42, split=None,
                  client=None, llm_model=None, verbose=True):
    """
    Run a fixed store_subset against a random sample of questions and
    collect latency/token/correctness-proxy results.

    store_subset: tuple, e.g. ("V","R","G") for B1, ("V",) for B2.
    n_samples: how many questions to sample (keep small -- each question
        costs one real LLM call; 50 questions took roughly 15-30 min at
        ~20-30s/question in testing).
    split: optionally restrict sampling to one qa split (e.g. "test").

    Returns a pandas DataFrame, one row per question, with columns:
      q_id, question, gold_answer_ids, gold_names, store_subset,
      answer, retrieval_hit (gold node id was among retrieved ids),
      answer_contains_gold_name (crude proxy, see module docstring),
      retrieval_latency_sec, llm_latency_sec, total_latency_sec,
      input_tokens, output_tokens
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
                  f"retrieval_hit={retrieval_hit} proxy_hit={proxy_hit} "
                  f"latency={result.total_latency_sec:.1f}s")

    return pd.DataFrame(rows)


def summarize_baseline(df):
    """Quick aggregate stats for a run_baseline() result DataFrame."""
    return {
        "n": len(df),
        "retrieval_hit_rate": df["retrieval_hit"].mean(),
        "answer_contains_gold_name_rate": df["answer_contains_gold_name"].mean(),
        "avg_total_latency_sec": df["total_latency_sec"].mean(),
        "avg_retrieval_latency_sec": df["retrieval_latency_sec"].mean(),
        "avg_llm_latency_sec": df["llm_latency_sec"].mean(),
        "avg_input_tokens": df["input_tokens"].mean(),
        "avg_output_tokens": df["output_tokens"].mean(),
        "total_input_tokens": df["input_tokens"].sum(),
        "total_output_tokens": df["output_tokens"].sum(),
    }
