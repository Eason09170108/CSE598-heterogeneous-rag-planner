"""
llm_judge.py

Stage 3.5: Accuracy metric validation, LLM-as-judge version.

Replaces/supplements manual human judging with an LLM judge call per
answer: given the question, gold answer name(s), and the generated
answer, the judge LLM returns CORRECT/INCORRECT plus a one-sentence
reason. This is cheap (short prompt, short output) relative to the
original generation calls, and more robust than the free
`answer_contains_gold_name` substring proxy to paraphrases.

IMPORTANT: an LLM judge is itself an imperfect metric and should be
spot-checked against human judgement on a small sample (e.g. 10
questions) to report an agreement rate -- don't just report the judge's
verdicts as ground truth without that check. See
scripts/score_manual_validation.py for the comparison step once you
have a few manually-judged rows to compare against.

Usage (inside Colab, same client setup as scripts/baselines.py):

    import os
    import openai
    from scripts.llm_judge import run_llm_judge

    client = openai.OpenAI(api_key=os.environ["LLM_API_KEY"],
                            base_url="https://openai.rc.asu.edu/v1")
    judged_df = run_llm_judge(
        "findings/accuracy_metric_validation_sheet.csv",
        client=client,
        llm_model="qwen38-27b",
    )
    judged_df.to_csv("findings/accuracy_metric_validation_sheet.csv", index=False)
"""

import json
import os
import time

import pandas as pd

JUDGE_PROMPT_TEMPLATE = """You are grading whether a generated answer correctly answers a question, given the ground-truth answer entity name(s).

Question: {question}

Ground-truth answer (the correct entity): {gold_names}

Generated answer to grade: {answer}

Judge whether the generated answer correctly identifies the ground-truth entity. The wording does not need to match exactly -- synonyms, paraphrases, or partial names that unambiguously refer to the same entity should be judged CORRECT. An answer that mentions the right entity but gives it as one of several guesses without committing to it, or an answer that is about a different entity, should be judged INCORRECT. An answer that says it cannot determine the answer should be judged INCORRECT.

Respond with ONLY a JSON object, no other text:
{{"verdict": "CORRECT" or "INCORRECT", "reason": "one short sentence"}}"""


def _parse_judge_output(raw_text):
    """
    Best-effort JSON parse of the judge's response. Falls back to a
    naive substring check on the raw text if JSON parsing fails (e.g.
    the model wrapped the JSON in markdown fences or added stray text).
    """
    text = (raw_text or "").strip()
    try:
        obj = json.loads(text)
        verdict = str(obj.get("verdict", "")).strip().upper()
        reason = str(obj.get("reason", "")).strip()
        return verdict == "CORRECT", reason
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: strip common markdown code-fence wrapping and retry once.
    stripped = text.strip("`").replace("json\n", "").strip()
    try:
        obj = json.loads(stripped)
        verdict = str(obj.get("verdict", "")).strip().upper()
        reason = str(obj.get("reason", "")).strip()
        return verdict == "CORRECT", reason
    except (json.JSONDecodeError, AttributeError):
        pass

    # Last resort: keyword search, flagged in the reason so it's visible
    # which rows need manual double-checking.
    upper = text.upper()
    is_correct = "CORRECT" in upper and "INCORRECT" not in upper
    return is_correct, f"[UNPARSED JUDGE OUTPUT, keyword fallback] {text[:200]}"


def run_llm_judge(sheet_path, client, llm_model="qwen38-27b", max_tokens=150,
                   verbose=True):
    """
    Reads the validation sheet CSV (as built by
    build_manual_validation_sheet.py), runs one judge LLM call per row,
    and returns a DataFrame with two new columns:
      llm_judge_correct (bool), llm_judge_reason (str)

    Does not call run_executor / regenerate any answers -- only judges
    the existing `answer` column.
    """
    df = pd.read_csv(sheet_path)

    judge_correct = []
    judge_reason = []

    for n, row in df.iterrows():
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=row["question"],
            gold_names=row["gold_names"],
            answer=row["answer"],
        )

        start = time.time()
        response = client.chat.completions.create(
            model=llm_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.time() - start

        raw = response.choices[0].message.content if response.choices else ""
        is_correct, reason = _parse_judge_output(raw)

        judge_correct.append(is_correct)
        judge_reason.append(reason)

        if verbose:
            print(f"[{n+1}/{len(df)}] q_id={row['q_id']} baseline={row['baseline']} "
                  f"judge={is_correct} proxy={row['answer_contains_gold_name']} "
                  f"({elapsed:.1f}s)")

    df["llm_judge_correct"] = judge_correct
    df["llm_judge_reason"] = judge_reason
    return df
