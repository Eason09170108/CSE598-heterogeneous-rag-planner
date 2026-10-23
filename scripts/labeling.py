"""
labeling.py

Stage 2: Execution-derived labeling.

For each question, brute-force search over all non-empty subsets of
{V, R, G} (7 combinations). For each subset, run retrieval only (no LLM
call — see executor.retrieve_context) and check whether the gold answer
node id(s) appear among the retrieved node ids. Among the subsets that
"hit" (contain at least one gold answer id), label the question with the
cheapest one — cost approximated by number of stores queried, tie-broken
by measured retrieval latency. This becomes the "required store set"
ground-truth label used to train/evaluate the planner (P1-P4) and to
compute the naive query-all baseline's wasted cost.

This directly answers the methodology concern raised in instructor
feedback (Sep 26, Prof. Jia Zou): labels are derived from actual
retrieval behavior, not assumed from answer-node type. The validation
step that feedback also requested — sampling these automatic labels and
having team members independently check agreement / error rate — is a
separate, later step (see findings/label_validation.md once written);
this script only produces the candidate labels.

Usage (inside Colab, after running setup_stark_env.py):

    from scripts.stores import StoreManager
    from scripts.labeling import label_dataset

    stores = StoreManager(skb=skb, G=G, index=index,
                           all_node_ids=all_node_ids, model=model)
    labels_df = label_dataset(qa, stores, n_samples=200, seed=42)
    labels_df.to_csv("findings/execution_derived_labels.csv", index=False)
"""

import itertools
import random

import pandas as pd

from scripts.executor import retrieve_context

ALL_STORES = ("V", "R", "G")


def all_nonempty_subsets(stores=ALL_STORES):
    """All 7 non-empty subsets of {V, R, G}, smallest (cheapest) first."""
    subsets = []
    for r in range(1, len(stores) + 1):
        subsets.extend(itertools.combinations(stores, r))
    # Sort by subset size (proxy for cost: fewer stores = cheaper), then
    # alphabetically for determinism among same-size subsets.
    subsets.sort(key=lambda s: (len(s), s))
    return subsets


def label_question(question_text, gold_answer_ids, stores, subsets=None):
    """
    Brute-force all store subsets for one question. Returns a dict with:
      - required_stores: cheapest subset whose retrieval contains a gold
        answer id, or None if no subset (including all three) succeeds
      - per_subset_hit: {subset_str: bool} whether each subset's retrieval
        contained a gold answer id
      - per_subset_latency: {subset_str: float} retrieval latency per subset
    """
    if subsets is None:
        subsets = all_nonempty_subsets()

    gold_set = set(gold_answer_ids)
    per_subset_hit = {}
    per_subset_latency = {}
    required_stores = None

    for subset in subsets:
        _, retrieved_ids, latency = retrieve_context(question_text, subset, stores)
        hit = bool(gold_set & set(retrieved_ids))
        per_subset_hit["+".join(subset)] = hit
        per_subset_latency["+".join(subset)] = latency
        if hit and required_stores is None:
            # subsets is sorted cheapest-first, so the first hit is the label.
            required_stores = subset

    return {
        "required_stores": "+".join(required_stores) if required_stores else None,
        "per_subset_hit": per_subset_hit,
        "per_subset_latency": per_subset_latency,
    }


def label_dataset(qa, stores, n_samples=200, seed=42, split=None):
    """
    Label a random sample of n_samples questions from the STaRK-Prime qa
    object. If split is given (e.g. "test"), only sample from that split.

    Returns a pandas DataFrame with one row per question:
      q_id, question, gold_answer_ids, required_stores,
      hit_V, hit_R, hit_G, hit_V+R, hit_V+G, hit_R+G, hit_V+R+G,
      latency_V, latency_R, latency_G, ... (one per subset)
    """
    rng = random.Random(seed)
    indices = list(range(len(qa)))
    if split is not None:
        indices = [i for i in indices if qa[i][3] == split]
    sample_indices = rng.sample(indices, min(n_samples, len(indices)))

    rows = []
    for i in sample_indices:
        question, q_id, answer_ids, q_split = qa[i]
        result = label_question(question, answer_ids, stores)
        row = {
            "q_id": q_id,
            "question": question,
            "split": q_split,
            "gold_answer_ids": "|".join(str(a) for a in answer_ids),
            "required_stores": result["required_stores"],
        }
        for subset_str, hit in result["per_subset_hit"].items():
            row[f"hit_{subset_str}"] = hit
        for subset_str, latency in result["per_subset_latency"].items():
            row[f"latency_{subset_str}"] = round(latency, 4)
        rows.append(row)

    return pd.DataFrame(rows)
