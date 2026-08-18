# Official-control-flow ReAct reproduction

This adapter reproduces the control loop in the official ReAct HotpotQA
notebook at commit `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`.
The unavailable original `text-davinci-002` model is replaced by the unified
`Qwen/Qwen3-VL-8B-Instruct` service used by the other baselines. The official
seven-step loop, stop sequences, 100-token step limit, temperature zero, and
single Action-repair call are retained.

The environment is intentionally closed. `Search` and `Lookup` can access only
the Qasper paper or the ten HotpotQA distractor passages supplied for the
current question. The adapter never accesses live Wikipedia and never reads
gold answers or gold evidence to select, repair, or fall back to evidence.

## Inputs and outputs

- Prepared Qasper: `runs/hipporag2/full_prepared/qasper`
- Prepared HotpotQA: `runs/hipporag2/full_prepared/hotpotqa`
- Public parameters: `config/react_official_repro.yaml`
- Private credentials: existing ignored YAML files under `runs/_configs/`
- Official source: ignored `runs/third_party/ReAct`
- Raw runs and status: ignored `runs/react_official_repro/`
- Evaluator outputs: `eval_<dataset>_react_official_repro` under each locked
  dataset working directory

No credential or private endpoint is stored in the adapter, public config,
logs, source lock, or paper source.

## Commands

Verify the source lock:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Scripts/baselines/react_official/run_full_experiments.ps1 -Stage source-smoke
```

Run a bounded two-dataset smoke:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Scripts/baselines/react_official/run_full_experiments.ps1 -Stage smoke
```

Run or resume the full experiment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File Scripts/baselines/react_official/run_full_experiments.ps1 -Stage full
```

Each completed query is reused only when the corpus hash, public config hash,
prompt hash, official source revision, and model name match. A changed identity
creates a different run directory. Old `eval_*_react` results are never
overwritten.

