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
  - V alone: embed the whole question and retrieve top-k candidate nodes.
  - V+G together: multi-symptom-intersection — split the question into
    phrases, find V anchors per phrase, expand 1-hop via G, and rank
    candidates by how many distinct phrases' anchors reach them. See
    _retrieve_g_intersection's docstring for why (plain single-anchor BFS
    was found, via labeling.py, to miss ~60% of questions even using all
    three stores — V's whole-question embedding anchors on only the
    single strongest-matching symptom and ignores the rest).
  - G alone: no anchor without V, contributes nothing (by design).
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
from collections import Counter
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


def _split_into_phrases(question):
    """
    Heuristic splitter: breaks a question into candidate symptom/claim
    sub-phrases on commas and conjunctions, so each can be embedded and
    matched against V separately instead of the whole question at once.
    Fragments shorter than 3 words are dropped (usually leftover
    connectives, e.g. "and", "while"). Falls back to the whole question
    if splitting produces nothing usable.
    """
    raw_parts = re.split(r",|\band\b|\bor\b|\bbut\b|\bwhile\b", question, flags=re.IGNORECASE)
    phrases = [p.strip() for p in raw_parts if len(p.strip().split()) >= 3]
    return phrases if phrases else [question]


def _retrieve_g_intersection(stores, question, per_phrase_v_k=5, per_anchor_g_k=15, top_k=20):
    """
    Multi-symptom-intersection retrieval, used whenever both V and G are
    requested together (replaces plain top-3-anchor BFS for that case).

    Motivation (found via labeling.py on a 200-question sample): a single
    V search over the *whole* question tends to anchor on whichever one
    symptom/phrase has the strongest textual match, then BFS from there —
    missing diseases whose correct identification requires combining
    several symptoms together (e.g. "infantile Refsum disease": V's
    whole-question top-10 were all diseases matching "night blindness"
    alone; none were the gold answer, because "learning challenges" and
    "severe early vision impairment" were never used as separate signals).
    See findings/ for this case in detail.

    Method: split the question into phrases, run V separately per phrase
    to get phrase-level anchors, expand each anchor 1-hop via G, and rank
    candidate nodes by how many DISTINCT phrases' anchor sets reach them
    (a node reachable from anchors found under multiple different symptom
    phrases outranks one reachable from only one).

    Returns (context_lines, node_ids, v_anchor_ids) — v_anchor_ids are the
    phrase-level V anchors themselves, included so callers can credit V's
    own contribution without a separate whole-question V call.
    """
    phrases = _split_into_phrases(question)

    phrase_neighbor_sets = []
    v_anchor_ids = []
    for phrase in phrases:
        hits = stores.query_v(phrase, k=per_phrase_v_k)
        phrase_anchor_ids = [node_id for node_id, _ in hits]
        v_anchor_ids.extend(phrase_anchor_ids)

        neighbors_this_phrase = set()
        for anchor_id in phrase_anchor_ids:
            for neighbor_id, _name, _ntype, _relation in stores.query_g(
                anchor_id, relation=None, k=per_anchor_g_k
            ):
                neighbors_this_phrase.add(neighbor_id)
        phrase_neighbor_sets.append(neighbors_this_phrase)

    vote_counts = Counter()
    for neighbor_set in phrase_neighbor_sets:
        for node_id in neighbor_set:
            vote_counts[node_id] += 1

    lines = []
    node_ids = []
    for node_id, votes in vote_counts.most_common(top_k):
        info = stores.skb.node_info[node_id]
        lines.append(f"[G intersection] {info['name']} ({info['type']}) "
                      f"matched {votes}/{len(phrases)} symptom phrases")
        node_ids.append(node_id)

    return lines, node_ids, v_anchor_ids


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

    has_v = "V" in store_subset
    has_g = "G" in store_subset

    if has_v and has_g:
        # Combined: multi-symptom-intersection (phrase-split V anchors ->
        # G 1-hop voting). See _retrieve_g_intersection's docstring.
        g_lines, g_ids, v_anchor_ids = _retrieve_g_intersection(stores, question)
        context_lines.extend(g_lines)
        retrieved_node_ids.extend(g_ids)
        retrieved_node_ids.extend(v_anchor_ids)
    elif has_v:
        # V alone: plain whole-question semantic search, no G to anchor into.
        v_lines, v_ids = _retrieve_v(stores, question)
        context_lines.extend(v_lines)
        retrieved_node_ids.extend(v_ids)
    elif has_g:
        # G alone has no anchor without V, so there is nothing to traverse
        # from. Intentionally contributes nothing (consistent with G's
        # near-zero standalone hit rate observed via labeling.py).
        pass

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
