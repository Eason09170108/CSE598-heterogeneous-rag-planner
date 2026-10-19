# Decision: Scope of the Relational Store (R)

**Date:** Oct 3
**Context:** See `r1_validation.md` for the full analysis.

## Decision

R will be built to store only `drug.state` (dosage form: oral / solid /
injection / etc.) initially, rather than the full drug/gene-protein
attribute schema (`molecular_weight`, `half_life`, `atc_1`–`atc_4`, `tpsa`,
`clogp`, `genomic_pos.*`, etc.).

## Rationale

- Manual review of 50 sampled questions found R was needed in ~6–8% of
  cases, almost entirely driven by dosage-form conditions (e.g. "solid-state
  medications", "oral medications").
- No sampled question used the other structured fields, despite their
  presence in the schema.
- Building the full schema would cost implementation time disproportionate
  to its observed usage, time better spent on G (needed in ~88–92% of
  questions) and V.

## Design note

The PostgreSQL schema is kept extensible (e.g. an `attributes JSONB` column
in addition to `state`), so additional fields can be added later without a
schema rewrite if evaluation shows they are needed.
