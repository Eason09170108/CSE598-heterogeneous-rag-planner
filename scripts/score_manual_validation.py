"""
score_manual_validation.py

Stage 3.5: Accuracy metric validation (scoring step).

Reads findings/accuracy_metric_validation_sheet.csv and compares the
free `answer_contains_gold_name` substring proxy against a better
reference judgement, computing:
  - reference-judged accuracy, overall and per baseline (B1 / B2)
  - the automated proxy's accuracy, for comparison
  - the agreement rate between the automated proxy and the reference
  - a breakdown of disagreements (false positives / false negatives of
    the proxy, relative to the reference)

The "reference" column is picked automatically, in priority order:
  1. manual_correct, if a person filled it in by hand for a row
  2. llm_judge_correct, if scripts/llm_judge.py was run (see that
     module's docstring -- an LLM judge is itself imperfect and should
     be spot-checked against a few manually-judged rows; if both
     columns are present for a row, this script also reports judge-vs-
     human agreement so that check is visible, not just assumed).

This directly answers the methodology concern raised about validating
automatically-derived labels/metrics: it reports how much the free
proxy over- or under-counts correctness relative to the reference,
rather than just asserting the proxy number as "accuracy."

Usage:
    python3 scripts/score_manual_validation.py
"""

import pandas as pd

SHEET_PATH = "findings/accuracy_metric_validation_sheet.csv"


def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
    )


def filled_mask(series):
    """True where a cell is a real, non-empty value (handles NaN, which
    str()-ifies to the literal string "nan" and would otherwise look
    "filled" to a naive non-empty-string check)."""
    return series.notna() & (series.astype(str).str.strip().str.lower() != "nan") \
        & (series.astype(str).str.strip() != "")


def main():
    df = pd.read_csv(SHEET_PATH)

    manual_filled = "manual_correct" in df.columns and filled_mask(df["manual_correct"])
    has_manual = bool(manual_filled.any()) if "manual_correct" in df.columns else False
    has_judge = "llm_judge_correct" in df.columns and df["llm_judge_correct"].notna().any()

    if has_manual and has_judge:
        both = manual_filled & df["llm_judge_correct"].notna()
        if both.any():
            sub = df[both].copy()
            manual_b = to_bool(sub["manual_correct"])
            judge_b = to_bool(sub["llm_judge_correct"].astype(str))
            print("=== LLM judge vs. human agreement "
                  f"(n={len(sub)} rows with both) ===")
            print(f"Agreement: {(manual_b == judge_b).mean():.1%}\n")

    reference_col = "manual_correct" if has_manual else "llm_judge_correct"
    print(f"(Using '{reference_col}' as the reference judgement column.)\n")

    missing = df[reference_col].isna() | (df[reference_col].astype(str).str.strip() == "")
    if missing.any():
        print(f"WARNING: {missing.sum()} row(s) have no {reference_col} value yet -- "
              f"excluding them from scoring.")
        df = df[~missing].copy()

    df["manual_correct_bool"] = to_bool(df[reference_col].astype(str))
    df["proxy_bool"] = to_bool(df["answer_contains_gold_name"].astype(str))

    df["agree"] = df["manual_correct_bool"] == df["proxy_bool"]
    # Proxy said correct, human disagreed (proxy over-counts).
    df["proxy_false_positive"] = df["proxy_bool"] & ~df["manual_correct_bool"]
    # Proxy said wrong, human says correct (proxy under-counts).
    df["proxy_false_negative"] = ~df["proxy_bool"] & df["manual_correct_bool"]

    print("=== Overall ===")
    print(f"n = {len(df)}")
    print(f"Human-judged accuracy:  {df['manual_correct_bool'].mean():.1%}")
    print(f"Automated proxy rate:   {df['proxy_bool'].mean():.1%}")
    print(f"Agreement rate:         {df['agree'].mean():.1%}")
    print(f"Proxy false positives:  {df['proxy_false_positive'].sum()} "
          f"({df['proxy_false_positive'].mean():.1%} of all rows)")
    print(f"Proxy false negatives:  {df['proxy_false_negative'].sum()} "
          f"({df['proxy_false_negative'].mean():.1%} of all rows)")

    print("\n=== By baseline ===")
    for name, g in df.groupby("baseline"):
        print(f"\n{name} (n={len(g)}):")
        print(f"  Human-judged accuracy: {g['manual_correct_bool'].mean():.1%}")
        print(f"  Automated proxy rate:  {g['proxy_bool'].mean():.1%}")
        print(f"  Agreement rate:        {g['agree'].mean():.1%}")

    disagreements = df[~df["agree"]]
    if len(disagreements):
        print(f"\n=== {len(disagreements)} disagreement row(s) (q_id, baseline, "
              f"proxy, manual) ===")
        for _, row in disagreements.iterrows():
            print(f"  q_id={row['q_id']} baseline={row['baseline']} "
                  f"proxy={row['proxy_bool']} manual={row['manual_correct_bool']}")


if __name__ == "__main__":
    main()
