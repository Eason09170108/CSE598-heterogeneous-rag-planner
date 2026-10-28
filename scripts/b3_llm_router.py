"""
b3_llm_router.py

Stage 3 (continued): B3, the zero-shot LLM router baseline.

Unlike P1 (a hand-written rule), B3 asks the LLM itself which of V/R/G
to query, *before* retrieval, via a separate short routing call. This is
the natural "can an LLM just do this without any hand-written rule?"
comparison point for P1 -- same per-question decision-making goal, but
zero domain-specific logic, at the cost of one extra LLM call per
question.

IMPORTANT cost accounting: the router call is itself a real cost (extra
latency + extra input/output tokens) that a rule-based planner like P1
doesn't pay. This module tracks that separately (router_latency_sec,
router_input_tokens, router_output_tokens) AND folds it into the row's
total_latency_sec/input_tokens/output_tokens, so B3's reported "total
cost" is the honest end-to-end cost (routing + retrieval + generation),
not just the generation step -- otherwise B3 would look artificially
cheap next to P1's free (near-instant) rule evaluation.

Usage (inside Colab, same client/stores setup as scripts/baselines.py):

    from scripts.stores import StoreManager
    from scripts.b3_llm_router import run_b3

    stores = StoreManager(skb=skb, G=G, index=index, all_node_ids=all_node_ids, model=model)
    b3_df = run_b3(qa, stores, n_samples=20, seed=42)
    b3_df.to_csv("/content/b3_results.csv", index=False)
"""

import json
import os
import random
import time

import pandas as pd

from scripts.baselines import answer_contains_gold_name
from scripts.executor import LLM_BASE_URL, LLM_MODEL, run_executor

ALL_STORES_ORDER = ("V", "R", "G")

ROUTER_PROMPT_TEMPLATE = """You are deciding which data stores to query to answer a question, before any retrieval happens. There are three stores:

V (vector store): semantic search over disease/drug/gene/phenotype entity descriptions. Good for finding a specific named entity by what it is or does.
R (relational store): a structured table of drugs, filterable by physical state (solid/liquid/gas). Only useful if the question is actually about a drug's physical state.
G (knowledge graph): multi-hop relations between entities (disease-gene, drug-target, phenotype-disease, etc). Needed whenever the answer depends on how entities relate to each other, not just what a single entity is.

Question: {question}

Pick the minimal set of stores needed to answer this question. Respond with ONLY a JSON object, no other text:
{{"stores": ["V", "G"]}}
(the array can contain any non-empty subset of "V", "R", "G")"""


def _parse_router_output(raw_text):
    """Best-effort JSON parse of the router's store choice. Falls back to
    V+G (the safest, most common default per findings/r1_validation.md)
    if parsing fails, so a malformed router response doesn't crash the run."""
    text = (raw_text or "").strip()
    candidates = [text]
    if text.startswith("```"):
        stripped = text.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        candidates.append(stripped.strip())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            stores = obj.get("stores", [])
            chosen = {s.upper() for s in stores if s.upper() in ALL_STORES_ORDER}
            if chosen:
                return tuple(s for s in ALL_STORES_ORDER if s in chosen)
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue

    return ("V", "G")  # fallback default, not silently empty


def decide_stores_llm_router(question, client, llm_model=LLM_MODEL, max_tokens=50):
    """
    One routing LLM call: returns (store_subset, router_latency_sec,
    router_input_tokens, router_output_tokens).
    """
    prompt = ROUTER_PROMPT_TEMPLATE.format(question=question)

    t0 = time.time()
    response = client.chat.completions.create(
        model=llm_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    router_latency = time.time() - t0

    raw = response.choices[0].message.content if response.choices else ""
    store_subset = _parse_router_output(raw)

    return (
        store_subset,
        router_latency,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def run_b3(qa, stores, n_samples=50, seed=42, split=None, client=None,
            llm_model=None, verbose=True):
    """
    Like scripts.planner.run_planner, but the store_subset is decided by
    an LLM routing call instead of a hand-written rule. Uses the SAME
    random sample (same seed) as run_baseline/run_planner, so results are
    directly row-comparable to B1/B2/P1 on identical questions.

    Returns a DataFrame with the same columns as run_baseline's output,
    PLUS router_latency_sec, router_input_tokens, router_output_tokens --
    total_latency_sec/input_tokens/output_tokens already include the
    router's cost (see module docstring).
    """
    import openai

    if client is None:
        client = openai.OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=LLM_BASE_URL,
        )
    model = llm_model or LLM_MODEL

    rng = random.Random(seed)
    indices = list(range(len(qa)))
    if split is not None:
        indices = [i for i in indices if qa[i][3] == split]
    sample_indices = rng.sample(indices, min(n_samples, len(indices)))

    rows = []
    for n, i in enumerate(sample_indices):
        question, q_id, answer_ids, q_split = qa[i]
        gold_names = [stores.skb.node_info[aid]["name"] for aid in answer_ids]

        store_subset, router_latency, router_in_tok, router_out_tok = (
            decide_stores_llm_router(question, client, llm_model=model)
        )

        result = run_executor(question, store_subset, stores, client=client,
                                llm_model=model)

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
            "router_latency_sec": round(router_latency, 4),
            "router_input_tokens": router_in_tok,
            "router_output_tokens": router_out_tok,
            "retrieval_latency_sec": round(result.retrieval_latency_sec, 4),
            "llm_latency_sec": round(result.llm_latency_sec, 4),
            # Honest end-to-end cost: routing + retrieval + generation.
            "total_latency_sec": round(router_latency + result.total_latency_sec, 4),
            "input_tokens": result.input_tokens + router_in_tok,
            "output_tokens": result.output_tokens + router_out_tok,
        })

        if verbose:
            print(f"[{n+1}/{len(sample_indices)}] q_id={q_id} "
                  f"store_subset={'+'.join(store_subset)} "
                  f"retrieval_hit={retrieval_hit} proxy_hit={proxy_hit} "
                  f"router_latency={router_latency:.1f}s "
                  f"total_latency={rows[-1]['total_latency_sec']:.1f}s")

    return pd.DataFrame(rows)
