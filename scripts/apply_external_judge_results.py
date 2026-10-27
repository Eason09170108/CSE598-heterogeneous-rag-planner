"""
apply_external_judge_results.py

Stage 3.5: Accuracy metric validation, external-LLM-judge version
(merge step).

Takes the JSON array reply from an external model (e.g. Gemini, pasted
after running scripts/build_gemini_judge_prompt.py) and merges it back
into findings/accuracy_metric_validation_sheet.csv as two new columns:
  llm_judge_correct (bool), llm_judge_reason (str)

These are the same column names scripts/score_manual_validation.py
already looks for, so no change is needed there.

Usage:
    python3 scripts/apply_external_judge_results.py path/to/gemini_reply.json

Or, if you have the reply as a Python string already, just paste it
into the JUDGE_REPLY variable below and run with no arguments.
"""

import json
import sys

import pandas as pd

SHEET_PATH = "findings/accuracy_metric_validation_sheet.csv"

# If running without a command-line argument, paste the raw JSON array
# text Gemini returned between the triple quotes below.
JUDGE_REPLY = """
"""


def load_reply(argv):
    if len(argv) > 1:
        with open(argv[1]) as f:
            return f.read()
    if JUDGE_REPLY.strip():
        return JUDGE_REPLY
    raise SystemExit(
        "No judge reply provided. Pass a file path as an argument, or "
        "paste the JSON array into the JUDGE_REPLY variable in this script."
    )


def parse_reply(raw_text):
    text = raw_text.strip()
    # Strip markdown code fences if the model wrapped its JSON in them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def main():
    raw = load_reply(sys.argv)
    verdicts = parse_reply(raw)

    df = pd.read_csv(SHEET_PATH)
    if "llm_judge_correct" not in df.columns:
        df["llm_judge_correct"] = pd.NA
    if "llm_judge_reason" not in df.columns:
        df["llm_judge_reason"] = ""

    by_index = {int(v["index"]): v for v in verdicts}
    missing = [i for i in range(len(df)) if i not in by_index]
    if missing:
        print(f"WARNING: no verdict found for row index(es) {missing} -- "
              f"leaving those unchanged.")

    for i, row in df.iterrows():
        if i in by_index:
            verdict = str(by_index[i].get("verdict", "")).strip().upper()
            df.at[i, "llm_judge_correct"] = (verdict == "CORRECT")
            df.at[i, "llm_judge_reason"] = by_index[i].get("reason", "")

    df.to_csv(SHEET_PATH, index=False)
    print(f"Merged {len(by_index)} verdict(s) into {SHEET_PATH}")
    print("Run scripts/score_manual_validation.py next to see the comparison.")


if __name__ == "__main__":
    main()
