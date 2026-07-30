# EviBridge Citation ID Protocol Design

## Problem

EviBridge currently renders each evidence item with both a presentation ordinal
and a persistent block ID:

```text
[1] block_id=37 ...
```

The answer model frequently copies the presentation ordinal into
`supporting_block_ids`. Those values are invalid whenever the ordinal differs
from the selected evidence's real `block_id`. In the Qasper seed-42 canary this
produced a 45.1% invalid citation rate.

## Decision

Render one unambiguous identifier per evidence item:

```text
[block_id=37] ...
```

The answer instructions must tell the model to copy the exact integer shown in
`block_id=...` into `supporting_block_ids`. The citation validator remains
strict: IDs outside the selected evidence set are ignored and recorded.

## Rejected Alternatives

- Do not map ordinal citations such as `[3]` to the third evidence item. A real
  block with ID `3` can exist, so ordinal fallback would introduce collisions.
- Do not introduce a dynamic structured-output enum in this change. It would
  couple the fix to provider-specific structured-output behavior and is larger
  than necessary.

## Compatibility

No persisted output fields change. `retrieved_node_ids`, `retrieved_block_ids`,
`supporting_block_ids`, citation diagnostics, and evidence-chain payloads keep
their current schemas.

## Verification

1. A unit test must fail against the old prompt because it contains `[1]`.
2. The updated prompt must contain `[block_id=<id>]`, contain no ordinal label,
   and explicitly require copying exact block IDs.
3. Existing citation-whitelist tests and the full unit-test suite must pass.
4. Re-run the fixed one-question canary, then a deterministic 10-question
   subset. Compare invalid citation rate, Answer F1, Evidence F1, token cost,
   and latency with the pre-fix run.
5. Run the 50-question canary only if the smaller run confirms that the prompt
   change reduces invalid citations without materially reducing answer quality.
