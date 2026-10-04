# R1 Risk Validation: Does the Relational Store (R) Matter on STaRK-Prime?

## Method

We sampled 50 questions from STaRK-Prime whose gold answer nodes have type
`drug` or `gene/protein` (the two node types in the knowledge base that carry
genuinely structured attributes, e.g. `drug.molecular_weight`, `drug.state`,
`drug.atc_1`–`atc_4`, `gene/protein.genomic_pos.*`). For each question, a team
member read the question text and judged which store(s) its retrieval logic
actually requires:

- **relation** — the question's core logic is a relationship between entities
  (e.g. "drug X targets gene Y", "drug X treats disease Y") — maps to the
  graph store (G).
- **attribute** — the question filters on a structured property of an entity
  (e.g. dosage form, numeric range) — maps to the relational store (R).
- **text** — the question's condition is a free-text / semantic description
  not captured by structured fields — maps to the vector store (V).
- **mixed** — more than one of the above.

## Results

### Answer-node type distribution (28,706 answer nodes across all splits)

| Node type | Count | % |
|---|---|---|
| drug | 8,913 | 31.0% |
| gene/protein | 6,393 | 22.3% |
| disease | 6,185 | 21.5% |
| anatomy | 1,815 | 6.3% |
| effect/phenotype | 1,678 | 5.8% |
| cellular_component | 1,579 | 5.5% |
| pathway | 1,149 | 4.0% |
| biological_process | 957 | 3.3% |
| exposure | 37 | 0.1% |

### Manual read of 50 sampled questions (drug / gene-protein answers only)

- ~88–92% of questions are driven purely by **relations** between entities
  (e.g. drug–targets–gene, drug–treats–disease, gene–interacts–gene).
- ~6–8% of questions reference a genuinely structured **attribute**, almost
  entirely `drug.state` (oral / solid-phase / solid-state medications).
- A small remainder reference **free-text / semantic** qualifiers (e.g.
  "effective", "mechanism not completely understood") that are not captured
  by any structured field, and these typically co-occur with a relation
  condition rather than standing alone.
- No sampled question used numeric fields such as `molecular_weight`,
  `half_life`, `tpsa`, `clogp`, or `genomic_pos.*`, despite these fields
  existing in the schema.

## Interpretation

Raw answer-type distribution (drug + gene/protein = 53.3% of answers)
substantially **overstates** how often the relational store is actually
needed. Most questions whose answer happens to be a drug or gene/protein are
resolved via graph traversal (G), not attribute filtering (R). The dataset's
three stores are not equally important:

- **G (graph)** is the backbone — required by nearly every question.
- **V (vector)** is a frequent secondary signal for semantic/descriptive
  conditions, usually combined with a relation condition.
- **R (relational)** is a minority-case filter, relevant mainly for dosage
  form (`drug.state`) and used in roughly one in ten to one in fifteen
  questions in this sample.

## Implications for the project

1. **R's build scope can be narrowed.** Rather than importing the full drug /
   gene-protein attribute schema into PostgreSQL, we can prioritize the
   fields with observed usage (starting with `drug.state`) and expand only if
   later analysis shows a need.
2. **This does not weaken RQ1.** The planner's task — deciding which stores
   to query — remains meaningful precisely because store relevance is
   unevenly distributed. A naive query-all baseline pays the cost of
   querying R on the ~90%+ of questions that do not need it; a cost-aware
   planner that learns to skip R on those questions has a clear efficiency
   argument to demonstrate.
3. **Evaluation design should report R's precision/recall separately**,
   not only as part of an aggregate store-selection score, since the overall
   average will be dominated by G and will dilute the (smaller but still
   meaningful) signal on R.
4. **This sample is a starting point, not the final label-validation step.**
   Section 6's execution-derived labeling (deriving required-store labels
   from which store subset's retrieval contains the gold answer) still needs
   its own validation once implemented: comparing its automatic labels
   against human judgment on a sample, and reporting agreement / error rate,
   per the Sep 26 instructor feedback from Prof. Jia Zou. That is a separate,
   later step from this document.

## Data

Raw sample with questions, answer types, and keyword flags:
[`findings/r1_sample_50.csv`](./r1_sample_50.csv)

Decision (Oct 3): R will be built with only the drug.state field initially, given that 6-8% of the R1 sample needed R and nearly all of that was driven by dosage-form conditions. Schema will be left extensible (e.g. a JSON attributes column) in case later evaluation shows other fields are needed.
