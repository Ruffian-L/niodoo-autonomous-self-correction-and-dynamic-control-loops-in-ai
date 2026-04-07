# Repo Map

This repository contains both the core Niodoo runtime path and a larger set of exploratory modules.

If you are here to understand or try the current local assistant runtime, start with the core path below.

## Core Path

These files are the shortest path to understanding the current working runtime:

- `README.md`: project framing, quick start, recommended model, examples
- `SECURITY.md`: security and asset-handling policy
- `CONTRIBUTING.md`: contribution scope and validation expectations
- `docs/TERMINOLOGY.md`: technical terminology guidance for public-facing writing
- `src/main.rs`: runtime loop, prompt template, steering state, telemetry, request parsing
- `src/physics/naked_llama.rs`: activation perturbation during the forward pass
- `scripts/chat_raw.py`: simple local multi-turn CLI
- `scripts/INSTALL.sh`: local build and run helper
- `generate_universe_from_gguf.py`: universe generation from the model-aligned token space
- `universe_top60000_token_map.json`: token map for the checked-in local runtime setup

## Recommended First Run

From the repo root:

```bash
cargo build --release --bin niodoo --offline
python3 scripts/chat_raw.py --max-steps 512
```

## What Is Experimental Or Exploratory

A large part of `src/` reflects ongoing experimentation, partially recovered work, or future-facing subsystems.

That includes areas such as:

- memory and topology experiments
- retrieval / RAG components
- splat and visualization work
- broader organism / regulation abstractions
- GPU kernels and rendering paths not required for the basic local chat path
- older server or daemon-adjacent modules that are not the primary intended interface

Those files are being published for openness and continuity, but they should not all be read as equally central to the current working runtime.

## Practical Reading Order

If you want to inspect the project efficiently:

1. Read `README.md`
2. Read `docs/REPO_MAP.md`
3. Read `src/main.rs`
4. Read `src/physics/naked_llama.rs`
5. Run `scripts/chat_raw.py`
6. Only then branch into the other `src/` modules

## Asset Notes

The public repository does not include the large runtime assets needed to run the full local system.

Keep these local:

- GGUF model files
- `.safetensors` universe files
- telemetry logs
- other generated artifacts

## Why `vendor/` Is Present

The Rust dependency tree is vendored intentionally so the project can build offline.

That makes the repo larger and can create false positives in secret scanners, but it keeps the public snapshot reproducible.
