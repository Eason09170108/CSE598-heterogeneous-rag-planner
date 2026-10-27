# Accuracy Metric Validation: Proxy vs. External LLM Judge

**Context:** `scripts/baselines.py` uses a free `answer_contains_gold_name`
substring proxy (case-insensitive check of the gold entity's name in the
generated answer) to measure correctness while iterating, explicitly
flagged in its own docstring as a crude metric that should not be reported
as "accuracy" without validation. This note is that validation.

## Method

All 40 existing B1/B2 answers (`findings/b1_results/*.csv`, no new
generation calls) were judged a second way: a single batch prompt
(`scripts/build_gemini_judge_prompt.py` → `findings/gemini_judge_prompt.txt`)
was pasted into **Gemini** — a different model from the one that generated
the answers (`qwen38-27b` via ASU Voyager) — to avoid a model grading its
own outputs. Gemini's JSON verdicts were merged back in
(`scripts/apply_external_judge_results.py`) and compared against the
proxy (`scripts/score_manual_validation.py`).

## Results

| Metric | Automated proxy | Gemini judge |
|---|---|---|
| B1 (V+R+G) accuracy | 30% (6/20) | **25% (5/20)** |
| B2 (V only) accuracy | 25% (5/20) | **20% (4/20)** |
| Overall accuracy | 27.5% (11/40) | 22.5% (9/40) |

**Agreement rate: 95.0% (38/40 rows).** 2 disagreements, both the same
question (q_id=1424, both the B1 and B2 run of it): the proxy marked it
correct (the gold name string appeared somewhere in the answer) but Gemini
judged it incorrect (the answer did not actually commit to that entity as
the answer — it appeared only as one of several hedged guesses). 0 cases
of the reverse (proxy under-counting a correct answer).

## Interpretation

1. **The proxy is directionally trustworthy but systematically
   optimistic.** It never missed a genuinely correct answer in this
   sample (0 false negatives), but it over-counts by exactly the
   "mentioned-but-not-committed" failure mode illustrated above (2 false
   positives, both on the same underlying question). On this sample that
   is a **~5 percentage point absolute inflation** (27.5% → 22.5%), i.e.
   proxy accuracy overstates true accuracy by roughly 20% in relative
   terms.
2. **The B1-vs-B2 ranking is unaffected.** Under both metrics B1 beats B2
   by the same 5pp gap (30/25 proxy, 25/20 judge) — the relative
   cost/accuracy tradeoff reported in `b1_vs_b2_comparison.md` holds
   regardless of which correctness metric is used. This is the key
   takeaway for the report: **comparisons between baselines/planners are
   robust to this metric choice; absolute accuracy numbers are not** and
   should always be reported with the metric named, not as "accuracy"
   unqualified.
3. **External LLM-judge is a reasonable metric going forward** for
   remaining baselines (B3) and planners (P1-P4): it is cheap (one short
   judge call per answer, or one batched prompt for many), and the 95%
   agreement with the free proxy on this sample means the proxy can still
   be used for fast iteration, with periodic judge spot-checks (as done
   here) rather than a judge call on every run.

## Caveats

- n=40 is small; the 2 disagreements came from a single question, so the
  "~5pp inflation" estimate itself has wide uncertainty and should not be
  treated as a precise correction factor.
- Gemini itself is not ground truth — this validates the proxy against a
  *better* reference, not a perfect one. A small human-adjudicated sample
  (e.g. 10 questions) would be the next tier of rigor if time allows, and
  `scripts/score_manual_validation.py` already supports comparing a
  `manual_correct` column against `llm_judge_correct` for exactly this,
  automatically reporting their agreement if both are filled in.

## Implications for the project

This directly addresses the methodology concern (raised earlier re:
Prof. Jia Zou's known sensitivity to unvalidated automatically-derived
labels/metrics) for the *answer-correctness* side of evaluation.
Separately, the *retrieval-hit* metric (`retrieval_hit` column) is exact
and objective (checks whether the gold node id literally appears among
retrieved ids) and needs no such validation — see
`retrieval_depth_investigation.md` for that metric's own (different)
caveats, which are about retrieval coverage, not judgment reliability.
