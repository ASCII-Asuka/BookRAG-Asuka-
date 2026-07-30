# ReAct-RAG Baseline Design

## Objective

Add a training-free ReAct baseline to BookRAG-Asuka and evaluate it on:

- Qasper validation full: 281 papers and 1,005 questions.
- HotpotQA validation 1,000-question sample: first200, second200, and random600.

The implementation must preserve the ReAct paper's interleaved
`Thought -> Action -> Observation` interaction. It must not collapse ReAct
into query rewriting, an IRCoT alias, or a generic framework agent.

## Fidelity Boundary

The reference implementation is the official ICLR 2023 ReAct repository at
commit `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`.

The following behavior is preserved:

- Six few-shot ReAct trajectories.
- Three actions: `Search[query]`, `Lookup[keyword]`, and `Finish[answer]`.
- Alternating Thought, Action, and Observation records.
- At most seven interaction steps.
- A second LLM call to obtain an Action when the first response cannot be
  parsed into a Thought and Action.
- Empty `Finish[]` after the step limit, rather than a separate answer
  generator that could hide an unsuccessful trajectory.

The only intentional environment adaptation is replacing live Wikipedia with
the fixed local corpus already used by the other baselines. This prevents
corpus leakage and makes the comparison with EviBridge-RAG fair.

## Alternatives Considered

### Native Repository Integration

Implement a focused ReAct environment inside the existing `vanilla` strategy
and reuse the repository's BM25 indexes, LLM provider, result schema, and
evaluators.

This is the selected approach. It preserves the algorithm while avoiding old
Gym and OpenAI Completion API dependencies.

### Vendor the Official Gym Environment

Copy the official environment and wrappers unchanged, then replace their
Wikipedia network calls. This retains the historical file structure but adds
obsolete dependencies and still requires substantial environment changes.

### Generic Agent Framework

Use a LangChain or similar ReAct agent. This is rejected because its prompt,
parser, retries, and stopping behavior would not match the paper.

## Architecture

### Configuration

Extend `VanillaConfig` with `retrieval_method: react` and these controls:

- `react_max_steps: 7`
- `react_search_topk: 1`
- `react_page_observation_units: 5`
- `react_prompt_file`

`react_search_topk` controls how many candidate pages BM25 considers before the
environment selects the best page. It does not change the observation size.

### Local Page Environment

Create a focused ReAct environment module rather than adding page-state logic
to `VanillaRAG`.

The environment groups BM25 documents into pages:

- HotpotQA page key: `hotpot_title`, then `section_id` or `title_path`.
- Qasper page key: `section_id`, then `section` or `title_path`.
- Last-resort page key: the evidence block's own stable ID.

`Search[query]` works as follows:

1. Try case-insensitive exact matching against page keys.
2. Try normalized containment matching against page keys.
3. Fall back to BM25 paragraph retrieval and select the first result's page.
4. Set the selected page as the active page.
5. Return the first five evidence units in document order.

`Lookup[keyword]` works as follows:

1. Require an active page.
2. Split that page's evidence units into sentences while retaining source IDs.
3. Build the ordered list of sentences containing the keyword,
   case-insensitively.
4. Return the next match on repeated calls with the same keyword.
5. Reset the lookup cursor when the keyword changes.

`Finish[answer]` stores the exact text inside the brackets and terminates the
episode.

Invalid actions produce an explicit observation and consume a step, matching
the original environment's behavior.

### Prompting

The system instruction mirrors the official prompt and explains only the
three available actions.

HotpotQA uses the official six `webthink_simple6` training trajectories.

Qasper uses six fixed trajectories from Qasper train. Their questions and
evidence must not overlap Qasper validation. They use the same action syntax
and demonstrate:

- Direct section search.
- Section search followed by keyword lookup.
- Cross-section evidence gathering.
- Boolean synthesis.
- Extractive answers.
- Short abstractive answers.

The prompt for one query contains the instruction, six demonstrations, the
question, and the accumulated trajectory.

### Agent Loop

For step `i` from 1 through 7:

1. Ask the LLM for `Thought i` and `Action i`.
2. Parse the first complete action line.
3. If parsing fails, retain the first thought line and make one repair call for
   `Action i`.
4. Execute the action in the local environment.
5. Append the exact Thought, Action, and Observation to the trajectory.
6. Stop only when `Finish[...]` succeeds.

If step 7 does not finish, execute `Finish[]`. This produces a completed
prediction record with an empty answer, which official evaluation scores as
incorrect rather than treating it as a missing prediction.

### Evidence Semantics

Generation context and official supporting evidence are both derived only
from observations visible to the ReAct agent.

- A Search observation records the first five units of the selected page.
- A Lookup observation records the matched source unit.
- Repeated units are deduplicated by stable block ID.
- Supporting evidence is ordered by first observation.
- Qasper exports at most four text evidence units through the existing dynamic
  evidence path.
- HotpotQA exports every observed sentence mapping needed by the existing
  supporting-fact evaluator.

No post-hoc reranker or answer-aware evidence selector is added.

## Output Contract

The method name is `react`, producing `eval_<dataset>_react`.

`result.json` keeps the existing compatibility fields and adds:

- `answer_short`
- `answer_rationale`
- `retrieved_block_ids`
- `supporting_block_ids`

`retrieval_res.json` contains:

- `strategy: react`
- `ranked_results`
- `supporting_evidence`
- `react_steps`
- `react_num_calls`
- `react_num_bad_calls`
- `react_search_count`
- `react_lookup_count`
- `react_termination_reason`

`evidence_chain.json` contains the ordered trajectory and observed evidence
IDs needed to reproduce the run.

## Failure Handling

- Missing BM25 index: fail before processing the document.
- Empty corpus: produce `Finish[]` with a recorded `empty_corpus` reason.
- Malformed Thought/Action response: perform the official-style Action repair
  call once.
- Malformed repair response: execute it as an invalid action, record the
  observation, and continue.
- Unknown page or empty search: return a deterministic "Could not find"
  observation.
- Lookup without an active page: return "No active page."
- LLM or network failure: use the provider's existing retry policy; persistent
  failure is recorded by the normal document-level experiment recovery flow.

## Experiment Procedure

### Qasper

Run four inference shards over `qasper_validation_full.yaml`, merge through the
existing directory layout, validate 1,005/1,005 predictions, and run the
official Qasper exporter with dynamic text evidence.

Report:

- Answer F1
- Evidence F1

### HotpotQA

Run:

- `hotpotqa_validation_200.yaml`
- `hotpotqa_validation_200_split2.yaml`
- `hotpotqa_validation_random600_exclude400_bridge_diag_seed42.yaml`

Evaluate each split with the existing official evaluator and compute the
1,000-question weighted result:

`first200 * 0.2 + second200 * 0.2 + random600 * 0.6`.

Report:

- Answer EM
- Answer F1
- Supporting Fact F1
- Joint F1

### Diagnostics

For each dataset also report:

- Mean interaction steps.
- Mean LLM calls.
- Mean Search and Lookup actions.
- Invalid-action rate.
- Step-limit termination rate.
- Token cost and elapsed time.

## Testing

Unit tests cover:

- Configuration acceptance and defaults.
- Exact, containment, and BM25 page search.
- Lookup cursor progression and reset.
- Finish parsing, including empty answers.
- Seven-step termination.
- Thought/Action repair calls.
- Evidence deduplication and source mapping.
- Qasper section and HotpotQA title grouping.

Integration tests run a deterministic fake LLM trajectory through Search,
Lookup, and Finish, then verify all three output files.

Regression tests cover existing BM25, IRCoT, Qasper official export, and
HotpotQA evaluation behavior.

Before formal evaluation, a small Qasper and HotpotQA smoke run must pass the
experiment result validator with no missing predictions.

