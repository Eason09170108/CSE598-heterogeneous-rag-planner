"""
score_manual_validation.py

Stage 3.5: Accuracy metric validation (scoring step).

Reads findings/accuracy_metric_validation_sheet.csv after its
`manual_correct` column has been filled in by hand (TRUE/FALSE per row),
and computes:
  - human-judged accuracy, overall and per baseline (B1 / B2)
  - the automated proxy's accuracy, for comparison
  - the agreement rate between the automated proxy and manual judgement
  - a breakdown of disagreements (false positives / false negatives of
    the proxy, relative to the human judgement)

This directly answers the methodology concern raised about validating
automatically-derived labels/metrics: it reports how much the free
`answer_contains_gold_name` proxy over- or under-counts correctness,
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


def main():
    df = pd.read_csv(SHEET_PATH)

    missing = df["manual_correct"].isna() | (df["manual_correct"].astype(str).str.strip() == "")
    if missing.any():
        print(f"WARNING: {missing.sum()} row(s) have no manual_correct value yet -- "
              f"excluding them from scoring.")
        df = df[~missing].copy()

    df["manual_correct_bool"] = to_bool(df["manual_correct"])
    df["proxy_bool"] = to_bool(df["answer_contains_gold_name"])

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
