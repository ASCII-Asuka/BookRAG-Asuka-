# Qasper Coverage-Aware Supporting Evidence Pruning Design

## Status

Approved for implementation design on 2026-08-01. This document covers the
small-set optimization phase only. It does not authorize a full Qasper-1005
baseline run until the acceptance gates below pass.

## Context

The completed EviBridge Qasper-1005 run produced:

| Slice | Questions | Answer F1 | Evidence P | Evidence R | Evidence F1 |
|---|---:|---:|---:|---:|---:|
| Full 1005 | 1005 | 0.422615 | 0.451773 | 0.576496 | 0.474138 |
| Original optimization slice | 764 | 0.423522 | 0.446604 | 0.566835 | 0.467596 |
| New extension slice | 241 | 0.419740 | 0.468158 | 0.607123 | 0.494876 |

The run covered 281/281 papers and 1005/1005 questions with zero inference
errors, zero illegal final citations, and zero verifier-state contradictions.
Average cost was 2383.82 tokens and 11.76 seconds per question.

The current 764-question fair baselines include:

| Method | Answer F1 | Evidence F1 |
|---|---:|---:|
| Full-document | 0.443955 | 0.496775 |
| LongRAG | 0.436791 | 0.438242 |
| EviBridge support controller | 0.423522 | 0.467596 |

EviBridge therefore exceeds LongRAG on evidence but remains below
Full-document on Answer F1 and Evidence F1. The weakest EviBridge demand types
are comparison and multi-hop. Controller-triggered questions have relatively
high Evidence Recall but low Precision, indicating that evidence added after
the draft answer is often noisy.

This optimization targets that precision loss before changing PPR, the answer
prompt, the answer model, or long-context fallback behavior.

## Goals

1. Improve final Evidence Precision and Evidence F1 without materially reducing
   Evidence Recall or Answer F1.
2. Isolate the effect of the supporting-evidence controller.
3. Add deterministic retrieval Recall@5, Recall@10, and Recall@20 diagnostics.
4. Avoid additional LLM calls and avoid increasing token cost.
5. Validate on a locked small set before touching the 664-question holdout.
6. Run all Qasper-1005 baselines only after the optimized method passes the
   small-set and 764-question acceptance gates.

## Non-Goals

- Do not tune typed PPR, seed recall, graph-edge weights, or the selector that
  produces the ten answer-context blocks.
- Do not modify the answer prompt, LLM provider, or model name.
- Do not add short-answer extraction or a full-document fallback.
- Do not run ablations in this phase.
- Do not expand other baselines to 1005 questions until EviBridge is accepted.
- Do not use gold evidence, per-question metric outcomes, or error inspection to
  choose the optimization sample.

## Data Isolation

### Source population

The optimization population is the existing 210-paper, 764-question slice used
by the fair baseline table. The completed 1005-question run supplies frozen
draft answers and retrieval diagnostics for replay, but the additional 241
questions are excluded from optimization.

### Tuning set

Create and commit a deterministic seed-42 manifest containing 100 questions.
Sampling may use only answer type and demand intent. It must not use current
scores, gold evidence IDs, controller trigger status, or observed failure type.

Target answer-type quotas are:

| Answer type | Questions |
|---|---:|
| Extractive | 45 |
| Abstractive | 25 |
| Boolean | 15 |
| Unanswerable | 15 |

The same manifest must contain at least 30 comparison questions and eight
multi-hop questions. Sampling and tie-breaking must be deterministic. The
manifest records question ID, document ID, answer type, demand intent, source
dataset hash, and selection seed.

### Holdout

The remaining 664 questions form a locked holdout. They are evaluated once only
after replay100 and fresh100 pass. No threshold or rule may be changed in
response to per-question holdout outcomes. The 241 newly added questions remain
outside this optimization cycle.

## Existing Retrieval Budgets

The current configured retrieval layers use:

- BM25 seed recall: 20
- Dense seed recall: 20, but vector recall is disabled in the accepted run
- typed PPR candidates: 30
- answer context: 10
- weak-support candidate pool: 20
- final supporting evidence: dynamic rather than a retrieval Recall@k budget

Recall@10 therefore corresponds to the answer-context budget. Recall@20
corresponds to the weak-support candidate budget.

## Proposed Architecture

### Module boundary

Add a focused support-pruning module under `Core/rag/` and keep orchestration in
`evibridge_rag.py`. The module accepts plain payloads and has no provider or
index ownership, which makes selection deterministic and unit-testable.

Inputs:

- question text;
- frozen draft answer;
- demand intent and up to two demand subqueries;
- legal model citation IDs, treated as anchors;
- up to 20 answer-bearing candidate payloads;
- answer-conditioned reranker scores;
- configured evidence-type whitelist and evidence-count cap.

Outputs:

- ordered final supporting evidence IDs;
- candidate-level selection diagnostics;
- selected additions and rejected candidates;
- a deterministic stopping reason.

Existing compatibility fields remain unchanged:

- `retrieved_node_ids`
- `retrieved_block_ids`
- `supporting_block_ids`
- `answer_context_block_ids`

### Eligibility

Only configured answer-bearing evidence types may be submitted. Entity, title,
summary, patch, and other bridge-only auxiliary nodes may provide connectivity
but cannot consume a final supporting-evidence slot. Invalid and duplicate IDs
are removed and recorded.

Canonical unanswerable answers always export an empty evidence list.

### Anchors

Every legal model citation inside the selected answer context is retained in
its original order. Anchors are never removed by relevance or redundancy
pruning in this phase. If anchors already meet or exceed the dynamic cap, no
candidate is added.

This preserves the approved weak-only policy: the controller may prune noisy
automatic additions but does not second-guess otherwise legal model citations.

### Candidate relevance

The existing answer-conditioned reranker receives:

```text
Question: <question>
Draft answer: <answer>
Evidence intent: <intent>
```

Scores are normalized within each question. When finite scores have a non-zero
range, use min-max normalization. When all scores tie, use deterministic rank
normalization. Non-finite scores are invalid.

The first implementation exposes one relevance parameter:
`support_min_normalized_relevance`. The top candidate remains eligible when no
legal anchor exists so that answerable weak-citation questions have a safe,
deterministic fallback.

### Coverage

Build lexical facets from:

1. the question;
2. the draft answer;
3. each demand subquery.

Tokenization and stop-word handling must be deterministic and dependency-free.
For each candidate, compute the additional fraction of each facet covered
beyond the current anchor/selection coverage. Comparison and multi-hop demands
prefer candidates that improve distinct subquery facets or, when subqueries are
absent, distinct question/answer facets.

Candidate utility is a fixed weighted combination of normalized reranker
relevance and marginal facet coverage. The weights are fixed in code for this
experiment; only the minimum relevance threshold is tuned.

### Redundancy

Before accepting a candidate, compute lexical overlap against every retained
anchor and previously accepted addition. Use the overlap coefficient so that a
short paragraph contained by a longer paragraph is recognized as redundant.

The second and final tuned parameter is
`support_redundancy_overlap_threshold`. Candidates at or above the threshold
are rejected unless they are already legal anchors.

### Dynamic budget and stopping

There is no minimum evidence count and no fixed-count padding.

- fact, boolean, table/figure, aggregation, and global-summary: at most two;
- comparison and multi-hop: at most three.

Legal anchors may exceed these caps. Greedy addition stops when the cap is
reached, no remaining candidate passes the relevance threshold, no remaining
candidate adds coverage, or all remaining candidates are redundant.

### Answer isolation

This variant does not expand the ten-block answer context and sets conditional
answer regeneration off. The draft answer is therefore unchanged in replay,
and fresh inference performs no second answer-generation call because of
support expansion. This cleanly attributes metric changes to evidence export.

The variant uses a distinct method suffix and never overwrites the accepted
`evibridge_support_controller` outputs.

## Replay Design

The completed run persists the 20 candidate IDs and enough index information to
reconstruct candidate text, but it does not persist all answer-conditioned
reranker scores or their final order.

Therefore replay has two phases:

1. Make one answer-conditioned reranker call for each controller-triggered
   replay question and save every candidate score.
2. Reuse that immutable cache for both the current fixed-padding controller and
   the proposed pruning controller.

The cache fingerprint includes question text, draft answer, ordered candidate
IDs, candidate-text hashes, reranker model name, reranker endpoint identity
without credentials, and relevant configuration. A missing or mismatched cache
fails the experiment instead of silently reranking.

After cache construction, threshold searches are model-free. They may evaluate
only a small fixed grid over the two approved thresholds. Both controllers must
consume exactly the same cached scores.

## Failure Handling

Production behavior and experiment-integrity behavior are intentionally
different:

- At runtime, reranker failure retains legal anchors. If no anchor exists, use
  the highest-ranked eligible block from the original selected set. If none
  exists, export empty evidence and record `no_eligible_candidate`.
- In replay evaluation, missing payloads, incomplete score caches, fingerprint
  mismatches, or question-manifest mismatches fail the run.
- Invalid requested citations are ignored and recorded.
- Duplicate IDs are deduplicated while preserving first occurrence.
- NaN and infinite scores are rejected and recorded.
- Every exit path records a stopping reason and fallback status.

No failure path may emit an ID outside the eligible candidate/anchor whitelist.

## Diagnostics

Extend `support_controller` without removing existing keys. Record:

- policy version and threshold values;
- candidate raw score, normalized relevance, and rank;
- per-facet current and marginal coverage;
- maximum overlap and the conflicting selected block ID;
- accepted/rejected status and reason;
- anchor IDs, added IDs, and final ordered IDs;
- stopping reason;
- cache fingerprint and cache-hit state in replay;
- whether runtime fallback was used.

`answer_regeneration` remains present and must report that regeneration was not
required for this variant.

## Metrics

### Official evidence metrics

For each Qasper gold annotation, compute Evidence Precision, Recall, and F1.
Choose the annotation with the highest Evidence F1 and report P/R/F1 from that
same annotation. Resolve F1 ties by dataset order. Local Evidence F1 must match
the official evaluator within `1e-6`.

### Retrieval Recall@k

Report Recall@5, Recall@10, and Recall@20 before support pruning.

For multiple annotations, choose one annotation using maximum Recall@20, then
Recall@10, then Recall@5, then dataset order. Report all three k values against
that same annotation. This keeps the diagnostic deterministic and prevents a
different annotation from being selected independently at each k.

Report two labeled views when applicable:

- base candidate Recall@k over all questions;
- answer-conditioned support-rerank Recall@k over triggered questions.

Also report final evidence-count distribution, controller trigger rate,
stopping-reason distribution, token cost, latency, and paired bootstrap 95%
confidence intervals.

## Acceptance Gates

### Replay100

Compared with the current controller on the same frozen questions and cached
reranker scores:

- Evidence F1 improves by at least 0.020;
- Evidence Precision improves by at least 0.030;
- Evidence Recall drops by no more than 0.015;
- Answer F1 is identical;
- zero illegal final citations;
- zero verifier-state contradictions;
- no additional LLM calls;
- token cost per question does not increase.

### Fresh100

Run the complete EviBridge pipeline on the same manifest only after replay100
passes:

- Evidence F1 improves by at least 0.015 over the frozen current-controller
  reference for those questions;
- Answer F1 drops by no more than 0.005;
- all integrity and cost gates from replay100 remain satisfied.

### Holdout664 and full764

Evaluate the frozen configuration once on holdout664, then combine it with the
100-question optimization slice to produce the fair 764-question table. Report
paired bootstrap 95% confidence intervals for metric levels and deltas. The
Evidence F1 delta lower bound must be greater than zero before accepting the
method for the full baseline campaign.

If a gate fails, do not run Qasper-1005 baselines and do not tune against
per-question holdout outcomes.

## Test Strategy

Use test-driven development. Required unit coverage includes:

1. legal anchors are retained in order;
2. invalid and duplicate citations are filtered deterministically;
3. evidence is not padded to a minimum count;
4. fact and comparison caps are enforced;
5. comparison and multi-hop candidates can cover distinct facets;
6. redundant candidates are rejected by overlap coefficient;
7. low-relevance and zero-marginal-coverage candidates stop selection;
8. bridge-only auxiliary blocks cannot be exported;
9. canonical unanswerable produces empty evidence;
10. tied, NaN, infinite, empty, and failed-reranker cases follow the specified
    fallback behavior;
11. diagnostics contain a reason for every candidate and exit path;
12. cache fingerprints reject any input or configuration mismatch;
13. Recall@5/@10/@20 uses one deterministic annotation;
14. mock call counters prove no answer regeneration and no additional LLM call;
15. legacy output fields and current support-controller tests remain compatible.

Add a small integration fixture that replays several frozen questions through
both controllers with one shared score cache. Before reporting success, run the
focused tests, the complete EviBridge/Qasper test group, and the repository test
suite that is available in the current environment.

## Rollout

1. Implement and verify manifest generation, score caching, metrics, and the
   pure pruning module.
2. Run replay100 and select thresholds only from the approved two-parameter
   grid.
3. Freeze thresholds and run fresh100.
4. If accepted, run holdout664 once and publish the fair 764 comparison.
5. Freeze code commit, configs, prompts, provider/model names, manifests, and
   evaluator version.
6. Only then run all agreed Qasper baselines over the complete 281-paper,
   1005-question dataset and produce a fair main table.

All code changes are made directly on `dev`; no worktree is created. Runtime
artifacts remain under ignored `runs/` paths and are not committed.
