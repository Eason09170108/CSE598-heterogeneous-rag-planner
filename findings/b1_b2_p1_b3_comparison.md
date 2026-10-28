# B1 vs B2 vs P1 vs B3: Does a Zero-Shot LLM Router Beat a Hand-Written Rule?

**Context:** B3 (`scripts/b3_llm_router.py`) asks the LLM itself, via one
extra routing call per question, which of V/R/G to query -- the
no-domain-knowledge comparison point for P1's hand-written rule. Same
n=20, seed=42 sample as B1/B2/P1. Raw results:
`findings/b3_results/b3_n20_seed42.csv`.

## Results

| Metric | B1 (V+R+G, fixed) | B2 (V only, fixed) | P1 (rule-based) | B3 (LLM router) |
|---|---|---|---|---|
| Retrieval hit rate | 50% (10/20) | 30% (6/20) | 50% (10/20) | 50% (10/20) |
| Answer-contains-gold-name (proxy) | 30% (6/20) | 25% (5/20) | 30% (6/20) | **25% (5/20)** |
| Avg total latency | 30.05s | 20.80s | **23.70s** | **31.30s** |
| Avg input tokens | 6,504 | 3,233 | 6,286 | 6,513 |
| Avg router/decision overhead | — (free) | — (free) | ~0s (string match) | **+1.88s/question** |

**B3 picked V+G for all 20/20 questions — it never once selected R**,
including for q_id=3811 ("What are the current investigational
**solid-state** drugs...?"), the one question in this sample where P1's
keyword rule correctly added R. The router degenerated to a constant
output regardless of question content.

## Interpretation

1. **P1 strictly dominates B3 here: lower cost AND equal-or-better
   accuracy.** P1 matches B1's accuracy at 23.70s; B3 is *worse* than
   B1 on the proxy metric (25% vs 30%) while being the *slowest* of all
   four (31.30s) — slower even than B1, which does strictly more
   retrieval work (V+R+G vs. B3's V+G). The extra ~1.88s/question router
   call is pure overhead here: it never changed the outcome (always
   V+G) and never saved anything downstream, since V+G's own retrieval
   and context size are nearly identical to V+R+G's in this pipeline
   (R's marginal context, and thus marginal token/latency cost, is
   small — consistent with `b1_vs_b2_comparison.md`'s earlier finding).
2. **Why the router degenerated to a constant answer is itself worth
   reporting.** The routing prompt (see `ROUTER_PROMPT_TEMPLATE` in
   `scripts/b3_llm_router.py`) describes R's narrow scope (drug
   physical-state filtering only) in general terms, but apparently
   doesn't give the model enough signal to connect a question's
   wording (e.g. "solid-state drugs") to R's specific utility —
   whereas P1's rule is a direct keyword match built from that same
   domain knowledge. This suggests zero-shot LLM routing needs either
   few-shot examples or a more explicit decision procedure in the
   prompt to beat a cheap hand-written rule, at least for a narrow,
   easily-missed store like R.
3. **The 20pp retrieval-hit gap is still entirely about G, not R.**
   B1/P1/B3 all hit 50% (all three query G on effectively every
   question); only B2 (no G) drops to 30%. This reconfirms
   `findings/r1_validation.md`'s finding that G is the store that
   matters most for this dataset's relation-driven questions — R's
   presence/absence barely moves accuracy at all in this sample (B1 vs.
   P1's near-identical subsets confirm R changed the outcome on 0/20
   questions here).

## Caveats

- n=20, same caveats as prior notes: small sample, the proxy's ~5pp
  optimism bias (`accuracy_metric_validation.md`), and the shared
  61.5%-miss-rate retrieval ceiling (`retrieval_depth_investigation.md`).
- This is one zero-shot prompt design for B3, not a tuned one. A
  differently-worded router prompt (e.g. with explicit examples of
  state-related questions, or asked to reason step-by-step before
  deciding) might route more accurately -- the negative result here is
  about *this* zero-shot formulation, not a claim that LLM routing is
  inherently worse than rules in general.

## Implications for the project

1. **Direct evidence for the project's central claim**, now from a
   second angle: not just "rules beat naive fixed baselines" (P1 vs.
   B1/B2) but "a cheap hand-written rule beats an LLM asked to do the
   same job zero-shot" (P1 vs. B3) -- motivating the staged planner
   design (P1 → P3 cost-based → P4 learned) rather than assuming a
   bigger/smarter model alone solves routing.
2. For the report, this is a good "unsuccessful approach" write-up
   (Phase 3 §3.1 explicitly asks for these): B3 was a reasonable thing
   to try, it cost more than the baseline it was meant to improve on,
   and the reason why is diagnosable and informative.
3. Next steps per the original staged design: P3 (cost-based planner)
   is now the natural next experiment, since P1/B3 together establish
   that routing needs either domain-specific rules or an explicit
   cost-accuracy objective -- not just "ask the LLM."
