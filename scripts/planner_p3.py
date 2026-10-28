"""
planner_p3.py

Stage 4 (continued): P3, the cost-based planner.

P1 decides store_subset with a fixed if/else rule. B3 asks an LLM to
decide with no explicit cost model at all (and was found to degenerate
to a constant answer, see findings/b1_b2_p1_b3_comparison.md). P3 is
different from both: it makes the cost/accuracy tradeoff an EXPLICIT,
TUNABLE quantity, rather than a hard-coded yes/no or an opaque LLM
judgment -- this is what makes it a *cost-aware* planner in the sense
the project's title claims, and it's the first planner here that can
trace out a whole family of plans (a cost-accuracy frontier) by varying
one parameter, instead of producing a single fixed policy.

Decision rule (utility-threshold form): for each optional store s (G,
R -- V is always included, since it's cheap and both G and R's
retrieval strategies in executor.py depend on V's anchors), include s
only if its estimated marginal BENEFIT (in retrieval-hit-rate units)
is at least `alpha` times its estimated marginal COST (in input-token
units):

    benefit(s) >= alpha * cost(s)

`alpha` is the knob: it's the "price" the planner is willing to pay,
in hit-rate points, per input token of added context. alpha=0 means
"always include anything with non-negative benefit" (closest to B1).
Increasing alpha makes the planner progressively more cost-averse,
dropping stores whose benefit doesn't justify their cost -- sweeping
alpha traces out the cost-accuracy frontier (see `sweep_alpha_offline`
below).

benefit(s) and cost(s) default to EMPIRICAL estimates measured directly
from the already-collected B1/B2/P1 runs (not guessed):
  - cost("G")  ~= 3052 input tokens  (P1's ~6286 avg vs. B2's ~3233 avg,
    i.e. the token cost of adding G on top of V)
  - benefit("G") ~= +0.20 hit-rate   (P1/B1's 50% hit rate vs. B2's 30%)
  - cost("R")  ~= 219 input tokens   (B1's ~6504 avg vs. P1's ~6286 avg,
    i.e. R's marginal context is small, consistent with
    findings/b1_vs_b2_comparison.md's point that R's own context_text
    is just a short drug-table listing)
  - benefit("R") ~= 0.0 hit-rate     (P1, which includes R only 1/20
    times, matched B1's hit rate exactly on this sample -- R changed
    the outcome on 0/20 questions here)

See "Limitations" in the module docstring below for why benefit("R")
~= 0 is almost certainly an underestimate for the *subset* of questions
R is actually relevant to (state-related questions were rare in this
n=20 sample) -- the same keyword signal P1 uses is kept here as a
`state_keyword_boost` added to R's benefit estimate specifically for
state-related questions, rather than applied as a blanket keyword gate.

Usage (inside Colab, same client/stores setup as scripts/baselines.py):

    from scripts.planner_p3 import make_decide_stores_p3
    from scripts.planner import run_planner

    decide_stores_p3 = make_decide_stores_p3(alpha=0.00005)
    p3_df = run_planner(qa, stores, decide_stores_p3, n_samples=20, seed=42)
    p3_df.to_csv("/content/p3_results.csv", index=False)

Limitations:
  - The cost/benefit priors are point estimates from a single n=20
    sample -- especially benefit("R"), which is based on only 1/20
    questions actually triggering R. A larger run (B1/B2/P1 re-run at
    higher n) would sharpen these estimates; this module's priors can
    be overridden via make_decide_stores_p3's arguments once that's
    done, without changing the decision logic itself.
  - G's benefit is a single pooled estimate (+0.20 on average); it
    likely varies a lot by question type (findings/r1_validation.md
    found most questions are relation-driven, so G's true benefit is
    probably bimodal -- very high for relation-driven questions, ~0 for
    the rest). A per-question benefit *estimate* (vs. this flat prior)
    is exactly what distinguishes a cost-based planner (P3) from a
    learned planner (P4, not yet built) -- P4 would learn to predict
    per-question benefit instead of using one dataset-wide number.
"""

from scripts.executor import STATE_KEYWORDS

ALL_STORES_ORDER = ("V", "R", "G")

# Empirical priors -- see module docstring for how these were measured.
DEFAULT_STORE_COST_TOKENS = {"G": 3052.0, "R": 219.0}
DEFAULT_STORE_BENEFIT_HITRATE = {"G": 0.20, "R": 0.0}


def make_decide_stores_p3(alpha=0.00005, state_keyword_boost=0.30,
                            cost_tokens=None, benefit_hitrate=None):
    """
    Returns a decide_stores_fn(question) -> tuple, parameterized by
    `alpha` (see module docstring), suitable for passing directly to
    scripts.planner.run_planner (same interface as decide_stores_p1).

    alpha: utility-price of one input token, in hit-rate-point units.
        Try sweeping this (e.g. 0, 0.00003, 0.00005, 0.0001, 0.0005) to
        see the decision flip at different cost-sensitivities --
        G's benefit/cost ratio is 0.20/3052 ~= 0.0000655, so alpha
        below that keeps G, alpha above it drops G (collapsing P3
        toward B2-like behavior). R's benefit/cost ratio is much lower
        (near 0 without the keyword boost), so R drops out at almost
        any alpha > 0 UNLESS the state-keyword boost fires.
    state_keyword_boost: added to R's benefit estimate specifically for
        questions containing a solid/liquid/gas keyword, since the
        flat dataset-wide benefit("R") ~= 0 estimate understates R's
        usefulness on exactly the narrow subset of questions it's
        designed for (see "Limitations" above).
    cost_tokens, benefit_hitrate: override the empirical priors (dicts
        with "G"/"R" keys) if better estimates become available (e.g.
        from a larger B1/B2/P1 re-run).
    """
    cost = dict(DEFAULT_STORE_COST_TOKENS)
    if cost_tokens:
        cost.update(cost_tokens)
    benefit = dict(DEFAULT_STORE_BENEFIT_HITRATE)
    if benefit_hitrate:
        benefit.update(benefit_hitrate)

    def decide_stores_p3(question):
        chosen = {"V"}

        if benefit["G"] >= alpha * cost["G"]:
            chosen.add("G")

        r_benefit = benefit["R"]
        question_lower = question.lower()
        if any(kw in question_lower for kw in STATE_KEYWORDS):
            r_benefit += state_keyword_boost
        if r_benefit >= alpha * cost["R"]:
            chosen.add("R")

        return tuple(s for s in ALL_STORES_ORDER if s in chosen)

    return decide_stores_p3


def sweep_alpha_offline(questions, alphas, state_keyword_boost=0.30,
                          cost_tokens=None, benefit_hitrate=None):
    """
    No LLM calls, no retrieval -- just shows which store_subset P3 would
    pick for each (alpha, question) pair, and the resulting *predicted*
    average cost/benefit (using the same priors the decision uses, not
    a real measurement) at each alpha. Useful for sanity-checking the
    alpha knob and for a quick illustrative cost-accuracy-frontier
    figure before spending real LLM calls on a couple of chosen alphas.

    Returns a dict: {alpha: {"subsets": [...], "avg_predicted_cost":
    float, "avg_predicted_benefit": float}}

    IMPORTANT: this is NOT a substitute for actually running
    scripts.planner.run_planner(qa, stores, decide_stores_p3, ...) --
    it only reflects the priors fed in, not real executor measurements.
    """
    cost = dict(DEFAULT_STORE_COST_TOKENS)
    if cost_tokens:
        cost.update(cost_tokens)
    benefit = dict(DEFAULT_STORE_BENEFIT_HITRATE)
    if benefit_hitrate:
        benefit.update(benefit_hitrate)

    results = {}
    for alpha in alphas:
        decide_fn = make_decide_stores_p3(
            alpha=alpha, state_keyword_boost=state_keyword_boost,
            cost_tokens=cost, benefit_hitrate=benefit,
        )
        subsets = [decide_fn(q) for q in questions]

        predicted_costs = []
        predicted_benefits = []
        for subset in subsets:
            c = sum(cost[s] for s in subset if s in cost)
            b = sum(benefit[s] for s in subset if s in benefit)
            predicted_costs.append(c)
            predicted_benefits.append(b)

        results[alpha] = {
            "subsets": subsets,
            "avg_predicted_cost": sum(predicted_costs) / len(predicted_costs),
            "avg_predicted_benefit": sum(predicted_benefits) / len(predicted_benefits),
        }

    return results
