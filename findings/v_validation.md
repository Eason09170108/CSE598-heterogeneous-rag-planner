# V-Store Validation: Semantic Search Alone Is Not Sufficient

**Context:** See `r1_validation.md` (R1) for the companion validation of the
relational store. This note documents a parallel validation check on the
vector store (V): whether free-text semantic search against node
descriptions, used on its own, reliably retrieves the correct answer.

## Method

For a disease-identification question whose gold answer was a specific
disease node (**sebaceous adenoma**, node id `95886`), we queried V directly
with the question's symptom/description text — i.e. running semantic search
over node descriptions *without* first narrowing the candidate set via a
graph traversal (G) from a known anchor entity (e.g. an affected gene,
phenotype, or anatomical structure named in the question).

## Result

The gold answer (sebaceous adenoma, node `95886`) did **not** appear in the
top-5 results. The top-ranked candidates were diseases whose text
descriptions are superficially similar (plausible-sounding matches on
symptom language) but are not the correct answer.

> TODO (Eason): paste the exact query text and the actual top-5
> `(node_id, name, score)` results from the Colab run here, so this finding
> has the full raw evidence attached, not just the summary conclusion.

## Interpretation

This is **not** a defect in the embedding model or the V store — it is
expected behavior and directly motivates the project's core premise. Disease
names and symptom descriptions in STaRK-Prime are often textually similar
across many candidate diseases, so free-text similarity alone is a weak
signal for this question type. The structured relation path (e.g.
gene/phenotype → `associated_with` → disease, via G) disambiguates where V
alone cannot.

## Implications for the project

1. **Supports RQ1/RQ2 directly.** If V alone were sufficient, there would be
   no need for a heterogeneous, cost-aware planner — a vector-only RAG
   baseline would suffice. This finding is evidence for why the planner's
   store-selection decision matters, and is useful as a motivating example
   in the Phase 2 presentation and Phase 3 report.
2. **Informs executor design.** `scripts/executor.py` uses V's top-1 match
   only as an *anchor* for G traversal, not as a standalone answer source —
   this finding is the empirical justification for that design choice.
3. **Candidate baseline (B2).** This also defines a natural "vector-only"
   baseline (query V alone, no G/R) to report against in Phase 2/3: we
   expect it to show measurably lower accuracy on relation-driven questions,
   which is itself a result worth presenting, not just a sanity check.
