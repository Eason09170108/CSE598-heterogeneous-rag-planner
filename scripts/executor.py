"""
executor.py

Stage 2: Executor.

Given a question and a *store subset* (which of V, R, G to use — decided
either by a planner later, or exhaustively by the labeling script in this
stage), this module:
  1. retrieves context from each store in the subset,
  2. assembles that context into a prompt,
  3. calls an LLM (Anthropic Claude) to generate the final answer,
  4. instruments wall-clock latency per step and token usage.

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
import re
import time
from dataclasses import dataclass, field

import anthropic

STATE_KEYWORDS = ("solid", "liquid", "gas")

LLM_MODEL = "claude-sonnet-4-5"  # change here if the team standardizes on another model

# Set ANTHROPIC_BASE_URL to point at an alternate Anthropic-SDK-compatible
# endpoint (e.g. ASU Voyager: https://openai.rc.asu.edu, no trailing /v1).
# Left unset, the anthropic SDK talks to Anthropic's own API as usual.
# If using Voyager (or any non-Anthropic backend), also set LLM_MODEL above
# to that backend's actual model name — it will NOT be a real Claude model.
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL")


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


def _retrieve_v(stores, question, k=5):
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


def _retrieve_g(stores, anchor_node_id, k=10):
    """1-hop neighbors of the anchor node. Returns (context_lines, node_ids)."""
    if anchor_node_id is None:
        return [], []
    hits = stores.query_g(anchor_node_id, relation=None, k=k)
    lines = []
    node_ids = []
    for neighbor_id, name, ntype, relation in hits:
        lines.append(f"[G] --[{relation}]--> {name} ({ntype})")
        node_ids.append(neighbor_id)
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

    # V always runs first if requested, since G's anchor depends on it.
    anchor_node_id = None
    if "V" in store_subset:
        v_lines, v_ids = _retrieve_v(stores, question)
        context_lines.extend(v_lines)
        retrieved_node_ids.extend(v_ids)
        anchor_node_id = v_ids[0] if v_ids else None

    if "G" in store_subset:
        # If V wasn't run, G has no anchor and is skipped (logged as empty).
        g_lines, g_ids = _retrieve_g(stores, anchor_node_id)
        context_lines.extend(g_lines)
        retrieved_node_ids.extend(g_ids)

    if "R" in store_subset:
        r_lines, r_ids = _retrieve_r(stores, question)
        context_lines.extend(r_lines)
        retrieved_node_ids.extend(r_ids)

    retrieval_latency = time.time() - t0
    context_text = "\n".join(context_lines) if context_lines else "(no retrieved context)"
    return context_text, retrieved_node_ids, retrieval_latency


def run_executor(question, store_subset, stores, client=None, llm_model=LLM_MODEL):
    """
    Execute a question against the given store_subset (e.g. ("V",), ("V","G"),
    ("V","R","G")) and return an ExecutionResult with the answer plus
    instrumentation (latency, token counts).

    store_subset: tuple of any of "V", "R", "G".
    stores: a scripts.stores.StoreManager instance.
    client: an anthropic.Anthropic() client; created from ANTHROPIC_API_KEY
            env var if not provided.
    """
    if client is None:
        client_kwargs = {"api_key": os.environ["ANTHROPIC_API_KEY"]}
        if ANTHROPIC_BASE_URL:
            client_kwargs["base_url"] = ANTHROPIC_BASE_URL
        client = anthropic.Anthropic(**client_kwargs)

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
    response = client.messages.create(
        model=llm_model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    llm_latency = time.time() - t1

    answer = response.content[0].text if response.content else ""

    return ExecutionResult(
        question=question,
        store_subset=tuple(store_subset),
        answer=answer,
        context_text=context_text,
        retrieved_node_ids=retrieved_node_ids,
        retrieval_latency_sec=retrieval_latency,
        llm_latency_sec=llm_latency,
        total_latency_sec=retrieval_latency + llm_latency,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
