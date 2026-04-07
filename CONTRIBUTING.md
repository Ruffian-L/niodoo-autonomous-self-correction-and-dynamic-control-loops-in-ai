# Contributing

Thank you for taking the time to inspect or extend Niodoo.

This repository is an experimental local assistant runtime and research snapshot. Contributions are welcome, but the most helpful changes are the ones that improve clarity, reproducibility, and the current core runtime path.

## Start Here

Before opening a change, read:

- `README.md`
- `docs/REPO_MAP.md`
- `docs/TERMINOLOGY.md`
- `SECURITY.md`

If you are working on the core runtime, inspect these files first:

- `src/main.rs`
- `src/physics/naked_llama.rs`
- `scripts/chat_raw.py`
- `generate_universe_from_gguf.py`

## Contribution Priorities

High-value contributions for this public snapshot:

- reproducibility fixes
- documentation and code comments that clarify the current runtime
- runtime stability improvements that preserve the existing control-loop behavior
- performance improvements with clear before/after measurements
- tests or evaluation harnesses around self-correction behavior

Lower-priority or riskier changes:

- large refactors across exploratory modules
- renaming runtime control tags such as `[REQUEST: FOCUS]` or `[REQUEST: SPIKE]`
- changing prompt templates or steering defaults without documenting behavioral impact
- committing large local assets, weights, logs, or personal runtime data

## Terminology Guidance

Use technical language in public-facing docs, comments, and pull requests.

Prefer terms like:

- `activation perturbation` instead of metaphor-heavy phrasing
- `steering vector` or `counter-steering vector` instead of poetic labels when discussing implementation
- `stochastic perturbation` instead of `wobble` in technical summaries
- `introspection` or `telemetry` instead of anthropomorphic descriptions unless the specific phrasing is part of the runtime prompt or emitted control surface

Exceptions:

- keep operational tags exactly as implemented when they are part of runtime behavior
- keep compatibility-sensitive CLI flags and serialized field names stable unless the change is deliberate and documented

## Build Expectations

For changes that affect the runtime path, validate at minimum:

```bash
cargo build --release --bin niodoo --offline
python3 scripts/chat_raw.py --max-steps 128
```

If local assets are unavailable, say so explicitly in the PR notes.

## Scope And Assets

Do not commit:

- GGUF model files
- `.safetensors` universe files
- telemetry logs
- private conversations
- credentials or tokens

The `vendor/` directory is present intentionally for offline Cargo builds.

## Pull Request Notes

A useful PR description should include:

- what changed
- why it changed
- whether behavior changed
- how it was validated
- whether the change targets the core runtime or an exploratory subsystem
