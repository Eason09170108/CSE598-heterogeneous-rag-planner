# P3: Tracing a Cost-Accuracy Frontier

**Context:** P3 (`scripts/planner_p3.py`) makes the cost/accuracy
tradeoff an explicit, tunable parameter (`alpha`), unlike P1's fixed
rule or B3's opaque LLM judgment. Two alpha values were run on the same
n=20, seed=42 sample as all prior baselines/planners, chosen from an
offline sweep (no LLM calls) that showed the decision transition
points: `alpha=0.00005` (keeps G for 19/20, matches P1's pattern almost
exactly) and `alpha=0.0002` (drops G for all 20, V-only except the
1/20 question where the state-keyword boost keeps R). Raw results:
`findings/p3_results/p3_alpha0.00005_n20_seed42.csv` and
`findings/p3_results/p3_alpha0.0002_n20_seed42.csv`.

## Results: the full comparison table

| | B1 (V+R+G) | B2 (V only) | P1 (rule) | B3 (LLM router) | P3 (α=5e-5) | P3 (α=2e-4) |
|---|---|---|---|---|---|---|
| Retrieval hit rate | 50% | 30% | 50% | 50% | **50%** | **30%** |
| Proxy accuracy | 30% | 25% | 30% | 25% | **30%** | **30%** |
| Avg total latency | 30.05s | 20.80s | 23.70s | 31.30s | **20.60s** | **16.97s** |
| Avg input tokens | 6,504 | 3,233 | 6,286 | 6,513 | 6,286 | 3,245 |

## Interpretation

1. **P3 traces out exactly the frontier it's designed to trace.** Low
   alpha (5e-5) reproduces P1's decisions almost exactly (same store
   subsets on 19/20 questions) and lands at the same accuracy/cost
   point. High alpha (2e-4) collapses toward B2's decisions (V-only on
   19/20) and lands at essentially the same accuracy/cost point as B2
   (30%/30% proxy vs. B2's 25%, 16.97s vs. B2's 20.80s, 3245 vs. 3233
   tokens). **One planner, one parameter, reproduces both ends of the
   B1↔B2 spectrum** -- this is the concrete demonstration of
   "cost-aware planning" the project's title claims: a single
   mechanism that can be dialed toward either end of the tradeoff,
   rather than needing a separate hand-built policy for each point on
   the curve.
2. **A genuinely informative anomaly: proxy accuracy did NOT degrade
   from α=5e-5 to α=2e-4, even though retrieval hit rate dropped 20pp
   (50%→30%).** Two questions (q_id=2286, 1424) were marked proxy-
   correct at α=2e-4 despite a retrieval MISS (the gold node was never
   retrieved) -- the LLM's answer happened to name the right entity
   anyway. The same pattern appears in B2 (q_id=1424 again). This means
   the free-text generator sometimes answers correctly from its own
   pretrained knowledge, independent of whether retrieval actually
   found the right node -- a real caveat for interpreting "accuracy"
   gains from retrieval on this benchmark with a capable base LLM: not
   every "correct" answer is actually grounded in what was retrieved.
   This is worth flagging explicitly in the report rather than treating
   proxy accuracy as a clean measure of retrieval quality.
3. **Despite that anomaly, the ranking holds in the expected direction
   once latency/token cost is weighed in**: α=2e-4 gets equal proxy
   accuracy to α=5e-5 at roughly half the token cost and latency -- on
   this sample, that specific alpha setting looks like a free lunch,
   but caveat #2 above means this is likely partly a small-sample
   artifact (2 "lucky" answers) rather than a reliable finding that G
   is worthless. A larger sample would be needed to tell whether this
   holds up or whether α=5e-5's retrieval-hit advantage (50% vs. 30%)
   would show up as a real accuracy gap on more questions.

## Caveats

- Same shared caveats as all prior notes: n=20, the proxy's known ~5pp
  optimism bias relative to an LLM judge (`accuracy_metric_validation.md`),
  and the 61.5%-miss retrieval ceiling (`retrieval_depth_investigation.md`).
- The cost/benefit priors driving alpha's decisions (G's benefit
  ≈+0.20 hit-rate, cost ≈3052 tokens; R's benefit ≈0 baseline + a
  keyword boost) are themselves estimated from this same small B1/B2/P1
  sample -- see `scripts/planner_p3.py`'s "Limitations" section. A
  larger labeling run (`scripts/labeling.py` at higher n) would sharpen
  these and could change exactly where the alpha transition points
  fall, though the *mechanism* (an explicit, sweepable utility
  threshold) doesn't depend on getting the priors exactly right.
- Only 2 alpha values were run with real LLM calls; a finer sweep (e.g.
  5-7 alpha values) would make a cleaner Pareto-frontier plot for
  slides, budget permitting.

## Implications for the project

1. **This is the single best figure for "cost-aware planning" in Phase
   2's slides**: plot proxy-accuracy vs. avg-latency (or avg-input-
   tokens) for all six rows above -- B1/B2 as fixed endpoints, P1/B3 as
   fixed-policy comparisons, and the two P3 points showing a single
   mechanism spanning the same range B1/B2 define. A third, intermediate
   alpha (if time allows) would make the "frontier" visual even clearer.
2. The parametric-knowledge-vs-retrieval confound (point 2 above) is
   good material for Phase 3's "Analysis, Findings & Research Insights"
   section -- it's a genuine, non-obvious limitation of using
   answer-text proxies (even an LLM-judged one, which would likely make
   the same "mistake" of crediting a lucky guess) to evaluate retrieval
   quality specifically.
3. Natural next step per the project's staged design: P4 (learned
   planner) would replace P3's flat, dataset-wide benefit priors with
   a per-question benefit *prediction* -- directly motivated by this
   note's point that G's true benefit is likely bimodal (very high for
   relation-driven questions, ~0 for others) rather than the single
   pooled number P3 uses.
