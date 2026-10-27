"""
build_gemini_judge_prompt.py

Stage 3.5: Accuracy metric validation, external-LLM-judge version.

Packs all rows of findings/accuracy_metric_validation_sheet.csv into a
single prompt that batch-judges every question/answer pair at once, so
it can be pasted into a chat UI for a different model (e.g. Gemini)
rather than called row-by-row through an API. Using a different model
as judge than the one that generated the answers (qwen38-27b) avoids
the model grading its own outputs.

Writes the prompt to findings/gemini_judge_prompt.txt. Paste that
file's full contents into Gemini (or any other chat model), then save
its reply (expected to be a JSON array) and run
scripts/apply_external_judge_results.py to merge it back into the
validation sheet.

Usage:
    python3 scripts/build_gemini_judge_prompt.py
"""

import json

import pandas as pd

SHEET_PATH = "findings/accuracy_metric_validation_sheet.csv"
OUT_PATH = "findings/gemini_judge_prompt.txt"

INSTRUCTIONS = """You are grading {n} generated answers against their ground-truth answers. For EACH item below, judge whether the generated answer correctly identifies the ground-truth entity.

Rules:
- Wording does not need to match exactly -- synonyms, paraphrases, or partial names that unambiguously refer to the same entity count as CORRECT.
- An answer that mentions the right entity only as one of several uncommitted guesses, or that is about a different entity, is INCORRECT.
- An answer that says it cannot determine the answer is INCORRECT.

Respond with ONLY a single JSON array, no other text, no markdown code fences. One object per item, in the same order, in this exact shape:
[{{"index": 0, "verdict": "CORRECT", "reason": "one short sentence"}}, {{"index": 1, "verdict": "INCORRECT", "reason": "one short sentence"}}, ...]

Here are the {n} items:

"""


def main():
    df = pd.read_csv(SHEET_PATH)

    items_text = []
    for i, row in df.iterrows():
        items_text.append(
            f"--- Item {i} ---\n"
            f"Question: {row['question']}\n"
            f"Ground-truth answer: {row['gold_names']}\n"
            f"Generated answer to grade: {row['answer']}\n"
        )

    prompt = INSTRUCTIONS.format(n=len(df)) + "\n".join(items_text)

    with open(OUT_PATH, "w") as f:
        f.write(prompt)

    print(f"Wrote prompt for {len(df)} items to {OUT_PATH}")
    print("Paste its full contents into Gemini (or another model), then save "
          "the reply and run scripts/apply_external_judge_results.py.")


if __name__ == "__main__":
    main()
