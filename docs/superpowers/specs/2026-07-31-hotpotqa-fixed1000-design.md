# HotpotQA Fixed-1000 Reproducible Evaluation Design

## Status and Motivation

The historical HotpotQA table is retained as a legacy aggregate only. Its three
per-question shards and selected question IDs are no longer present, so the
reported weighted 1000-question result cannot be reproduced or used as the
source of a new paper table.

The replacement evaluation set will be generated from the official HotpotQA
`distractor` validation parquet. The recovered source has:

- 7,405 rows;
- SHA256 `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`;
- 5,918 bridge questions and 1,487 comparison questions, all marked hard;
- 18,005 annotated supporting facts.

The source audit found one invalid official annotation:
`5ae61bfd5542992663a4f261` refers to sentence 902 of a five-sentence
paragraph. This row is excluded before sampling and recorded in the manifest.
No other supporting-fact title or sentence index is missing.

## Sampling Contract

The fixed set contains 1,000 questions and uses seed 42.

1. Load the official parquet in its original row order.
2. Require a non-empty, unique question ID and the expected HotpotQA fields.
3. Exclude only rows whose gold supporting facts cannot resolve to a source
   context sentence. Record every exclusion with its question ID, fact, and
   reason.
4. Form strata from `(type, level)`. For this validation release, the effective
   strata are `(bridge, hard)` and `(comparison, hard)`.
5. Allocate quotas proportionally with Hamilton largest-remainder allocation.
   Ties are resolved by lexicographic stratum name. The expected quotas are
   799 bridge and 201 comparison questions.
6. Within each stratum, rank rows by
   `SHA256("hotpotqa-fixed1000-v1\\0seed=42\\0" + question_id)` and take the
   lowest-ranked rows up to the quota.
7. Write selected rows in official source order, not hash order.

This hash-ranking algorithm avoids dependence on Python's random sampling
implementation and produces identical IDs across machines.

## Artifacts and Provenance

The data preparation command writes:

- the selected raw HotpotQA rows;
- the unified BookRAG/EviBridge dataset;
- a dataset YAML pointing to the unified file and a dedicated working
  directory;
- `tree.pkl` and `tree.json` for every selected question;
- a fixed-sample manifest.

The manifest contains:

- schema and sampling algorithm versions;
- source path, row count, size, and SHA256;
- seed, requested sample size, eligible count, and exclusion records;
- source, eligible, and selected counts for every stratum;
- the complete ordered list of selected question IDs;
- SHA256 hashes for selected raw and unified JSON files;
- supporting-fact audit totals and mapping coverage.

Evaluation and inference must fail before starting if the source hash, selected
ID order, artifact hash, document count, or supporting-fact mapping coverage
does not match the manifest.

## Supporting-Fact Mapping

Every sentence node and paragraph EvidenceBlock preserves:

- `hotpot_title` exactly as it appears in the source context;
- zero-based `hotpot_sent_id`;
- `source=hotpotqa_sentence`.

Gold title/sentence mapping follows this protocol:

1. Match the cleaned title exactly and then match `sent_id`.
2. Only if exact matching fails, use normalized-title matching when the
   normalized title maps to exactly one source paragraph.
3. Treat ambiguous normalized matches as an error; never silently select the
   last duplicate title.
4. Reject missing and out-of-range sentence IDs during preflight.

This exact-first rule fixes case-collision examples such as `Popular Science`
versus `Popular science`, `Bully boy` versus `Bully Boy`, and `Monkey gland`
versus `Monkey Gland`.

Predicted supporting facts are exported from the same exact metadata. Strict
Supporting Fact EM and Joint EM are restored to the evaluation output and must
not be omitted from the new main table.

## Error Handling

- A source hash or schema mismatch aborts preparation.
- Duplicate question IDs abort preparation.
- Unresolvable source annotations are written to the manifest and excluded
  before allocation; no other filtering is allowed.
- An exact-title collision or ambiguous normalized fallback aborts mapping.
- Missing tree/index/output files abort evaluation rather than reducing the
  denominator.
- Historical aggregate metrics remain labeled non-reproducible and are never
  merged with the fixed-1000 results.

## Verification

Automated tests cover:

- deterministic proportional quota allocation and stable ID selection;
- manifest hashes and full ordered selected-ID coverage;
- exact-title precedence over normalized duplicate titles;
- rejection and audit of out-of-range supporting facts;
- 100% gold fact-to-node and fact-to-EvidenceBlock coverage on generated data;
- strict Supporting Fact EM and Joint EM restoration.

The first execution checkpoint is a 20-question smoke run drawn from the fixed
manifest. After it passes structural validation, the same manifest is used for
the 1,000-question EviBridge and baseline runs.
