"""
build_manual_validation_sheet.py

Stage 3.5: Accuracy metric validation.

Builds a flat, human-readable sheet combining the B1 and B2 baseline
results so a person can manually judge, question by question, whether
the LLM's generated answer is actually correct -- independent of the
automated `answer_contains_gold_name` proxy used in scripts/baselines.py.

This does NOT make any new LLM calls. It only reformats data that
already exists in findings/b1_results/*.csv, so it costs nothing to run.

Output: findings/accuracy_metric_validation_sheet.csv with an empty
`manual_correct` column (to be filled in by hand with TRUE/FALSE) and
the existing automated proxy result side by side, so agreement can be
computed afterwards.

Usage:
    python3 scripts/build_manual_validation_sheet.py
"""

import pandas as pd

B1_PATH = "findings/b1_results/b1_n20_seed42.csv"
B2_PATH = "findings/b1_results/b2_n20_seed42.csv"
OUT_PATH = "findings/accuracy_metric_validation_sheet.csv"

COLUMNS = [
    "baseline",
    "q_id",
    "question",
    "gold_names",
    "store_subset",
    "answer",
    "answer_contains_gold_name",  # automated proxy, for later comparison
    "manual_correct",             # <-- fill this column by hand: TRUE / FALSE
    "manual_notes",                # optional: why, if the judgement is non-obvious
]


def main():
    b1 = pd.read_csv(B1_PATH)
    b1.insert(0, "baseline", "B1")
    b2 = pd.read_csv(B2_PATH)
    b2.insert(0, "baseline", "B2")

    combined = pd.concat([b1, b2], ignore_index=True)
    combined["manual_correct"] = ""
    combined["manual_notes"] = ""

    out = combined[COLUMNS]
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print("Fill in the 'manual_correct' column (TRUE/FALSE) by hand, "
          "then run scripts/score_manual_validation.py to compute agreement "
          "with the automated proxy.")


if __name__ == "__main__":
    main()
