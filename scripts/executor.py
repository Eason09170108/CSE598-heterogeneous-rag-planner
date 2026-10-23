"""
executor.py

Stage 2: Executor.

Given a question and a *store subset* (which of V, R, G to use — decided
either by a planner later, or exhaustively by the labeling script in this
stage), this module:
  1. retrieves context from each store in the subset,
  2. assembles that context into a prompt,
  3. calls an LLM to generate the final answer,
  4. instruments wall-clock latency per step and token usage.

LLM backend: uses the OpenAI python SDK against an OpenAI-compatible
endpoint, defaulting to ASU Voyager (https://openai.rc.asu.edu/v1, free
for ASU accounts, no Claude models available there as of Oct 2026 — the
project uses a strong open model served through it instead, see LLM_MODEL
below). Set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL env vars to point at a
different OpenAI-compatible backend (including the real OpenAI API).

Retrieval strategy per store (simplified for Stage 2 — a real planner /
query-decomposition step is future work, not this module's job):
  - V: always embed the raw question text and retrieve top-k candidate
    nodes. This also serves as the "entity anchor" step for G.
  - G: starting from the top-ranked V candidate (the anchor entity),
    collect its 1-hop neighbors as additional structured context.
  - R: scoped to `drug.state` only (see findings/r_scope_decision.md).
    A tiny keyword match against {solid, liquid, gas} in the question
    decides the state_filter; if none of those words appear, R is
    queried with no filter (returns a small sample instead of nothing).

This executor is intentionally "dumb" about routing — it executes
whatever store_subset it's told to use. Deciding the subset is the
planner's job (P1/P2/P3/P4, not yet built) or, in this stage, the
labeling script's brute-force search over all subsets.
"""

import os
import time
from dataclasses import dataclass, field

import openai

STATE_KEYWORDS = ("solid", "liquid", "gas")

# Defaults target ASU Voyager's free OpenAI-compatible gateway.
# qwen38-27b: "Recommended" tag on Voyager, highest LiveBench score (81.7)
# among non-testing models as of Oct 2026. Override via env vars to use a
# different Voyager model or a different OpenAI-compatible provider
# entirely (e.g. real OpenAI: LLM_BASE_URL unset, LLM_API_KEY=<openai key>,
# LLM_MODEL=gpt-4o-mini).
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openai.rc.asu.edu/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen38-27b")


@dataclass
class ExecutionResult:
    question: str
    store_subset: tuple
    answer: str
    context_text: str
    retrieved_node_ids: list = field(default_factory=list)
    retrieval_latency_sec: float = 0.0
    llm_latency_sec: float = 0.0
    total_latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


def _retrieve_v(stores, question, k=10):
    """Semantic search over node descriptions. Returns (context_lines, node_ids)."""
    hits = stores.query_v(question, k=k)
    lines = []
    node_ids = []
    for node_id, score in hits:
        info = stores.skb.node_info[node_id]
        lines.append(f"[V] {info['name']} ({info['type']}): "
                      f"{info.get('details', {})} (similarity={score:.3f})")
        node_ids.append(node_id)
    return lines, node_ids


def _retrieve_g(stores, anchor_node_ids, hops=2, per_node_k=10, total_k=200):
    """
    Multi-hop BFS over G starting from multiple anchor nodes (the top-N V
    matches). Each hop expands every frontier node by up to per_node_k
    neighbors (via stores.query_g), stopping at `hops` levels or once
    total_k nodes have been collected, whichever comes first.

    Single-hop-from-one-anchor was found (via labeling.py on a 200-question
    sample) to recover the gold answer in only ~27% of cases even combined
    with V and R — this is the fix for that: see findings/ for the before/
    after comparison once re-run.

    Returns (context_lines, node_ids).
    """
    if not anchor_node_ids:
        return [], []

    visited = set(anchor_node_ids)
    frontier = list(anchor_node_ids)
    lines = []
    node_ids = []

    for hop in range(1, hops + 1):
        next_frontier = []
        for node_id in frontier:
            hits = stores.query_g(node_id, relation=None, k=per_node_k)
            for neighbor_id, name, ntype, relation in hits:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                lines.append(f"[G hop={hop}] --[{relation}]--> {name} ({ntype})")
                node_ids.append(neighbor_id)
                next_frontier.append(neighbor_id)
                if len(node_ids) >= total_k:
                    return lines, node_ids
        frontier = next_frontier
        if not frontier:
            break

    return lines, node_ids


def _retrieve_r(stores, question, k=10):
    """Keyword-matched state filter over the drug table. Returns (context_lines, node_ids)."""
    question_lower = question.lower()
    state_filter = next((s for s in STATE_KEYWORDS if s in question_lower), None)
    hits = stores.query_r(state_filter=state_filter, k=k)
    lines = [f"[R] drug id={drug_id}, name={name}, state={state_filter or 'any'}"
              for drug_id, name in hits]
    node_ids = [drug_id for drug_id, _ in hits]
    return lines, node_ids


def retrieve_context(question, store_subset, stores):
    """
    Run retrieval ONLY (no LLM call) against the given store_subset.
    Returns (context_text, retrieved_node_ids, retrieval_latency_sec).

    Separated from run_executor so the labeling script can brute-force all
    2^|{V,R,G}|-1 subsets per question without paying for an LLM call on
    every one of them — labeling only needs to know whether the gold
    answer node is among the retrieved node ids, not a generated answer.
    """
    t0 = time.time()
    context_lines = []
    retrieved_node_ids = []

    # V always runs first if requested, since G's anchors depend on it.
    # Top-3 V matches (not just top-1) are used as anchors for G, since a
    # single top-1 anchor was found to miss the correct entity too often.
    anchor_node_ids = []
    if "V" in store_subset:
        v_lines, v_ids = _retrieve_v(stores, question)
        context_lines.extend(v_lines)
        retrieved_node_ids.extend(v_ids)
        anchor_node_ids = v_ids[:3]

    if "G" in store_subset:
        # If V wasn't run, G has no anchors and is skipped (logged as empty).
        g_lines, g_ids = _retrieve_g(stores, anchor_node_ids)
        context_lines.extend(g_lines)
        retrieved_node_ids.extend(g_ids)

    if "R" in store_subset:
        r_lines, r_ids = _retrieve_r(stores, question)
        context_lines.extend(r_lines)
        retrieved_node_ids.extend(r_ids)

    retrieval_latency = time.time() - t0
    context_text = "\n".join(context_lines) if context_lines else "(no retrieved context)"
    return context_text, retrieved_node_ids, retrieval_latency


def run_executor(question, store_subset, stores, client=None, llm_model=LLM_MODEL,
                  max_tokens=1024):
    """
    Execute a question against the given store_subset (e.g. ("V",), ("V","G"),
    ("V","R","G")) and return an ExecutionResult with the answer plus
    instrumentation (latency, token counts).

    store_subset: tuple of any of "V", "R", "G".
    stores: a scripts.stores.StoreManager instance.
    client: an openai.OpenAI() client; created from LLM_API_KEY/LLM_BASE_URL
            env vars if not provided.
    max_tokens: completion token budget. qwen38-27b (the default model) is a
        reasoning model that emits a `reasoning` field before `content` —
        both count against max_tokens, so this must be large enough to leave
        room for the final answer after reasoning, or `content` comes back
        empty/None (observed at 256 on non-trivial prompts; 1024 is a safer
        default, raise further if answers still truncate).
    """
    if client is None:
        client = openai.OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=LLM_BASE_URL,
        )

    context_text, retrieved_node_ids, retrieval_latency = retrieve_context(
        question, store_subset, stores
    )
    prompt = (
        "Answer the question using ONLY the retrieved context below. "
        "If the answer is not contained in the context, say you don't know.\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer concisely:"
    )

    t1 = time.time()
    response = client.chat.completions.create(
        model=llm_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    llm_latency = time.time() - t1

    answer = response.choices[0].message.content if response.choices else ""
    answer = (answer or "").strip()

    return ExecutionResult(
        question=question,
        store_subset=tuple(store_subset),
        answer=answer,
        context_text=context_text,
        retrieved_node_ids=retrieved_node_ids,
        retrieval_latency_sec=retrieval_latency,
        llm_latency_sec=llm_latency,
        total_latency_sec=retrieval_latency + llm_latency,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )
