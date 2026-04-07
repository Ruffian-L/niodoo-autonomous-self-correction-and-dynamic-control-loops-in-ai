# Launch Post Draft

## Short Version

I just open-sourced Niodoo.

Niodoo is an experimental local assistant runtime for GGUF models that steers generation during inference using activation perturbations and verbal control loops like `[REQUEST: FOCUS]` and `[REQUEST: SPIKE]`.

The goal was never to build a benchmark toy. I wanted a more helpful local assistant that could notice drift, revise itself while generating, and eventually carry memory forward in a way that supports a user's longer trajectory.

This is still active experimental work, not production code, but I wanted to share the snapshot openly.

Repo:
https://github.com/Ruffian-L/niodoo-autonomous-self-correction-and-dynamic-control-loops-in-ai

## Longer Version

I just published Niodoo, a local assistant runtime I have been rebuilding and recovering over the last few months.

The core idea is not “make Llama slightly better on a benchmark.”

The goal is a more helpful local assistant: one that can traverse its own generation, notice drift, revise itself mid-stream, and eventually support continuity and memory instead of acting like every prompt begins from zero.

In the current snapshot, Niodoo uses a hybrid runtime:

- local GGUF inference
- activation perturbations during decoding
- physics-inspired steering forces
- verbal self-control tags like `[REQUEST: FOCUS]`, `[REQUEST: SPIKE]`, and internal monitor traces
- visible telemetry and a simple local chat wrapper

This repository is public because I want the work to be inspectable and usable by the open-source community, even though it is still clearly experimental.

Important caveats:

- behavior depends heavily on the model and quantization
- tuning is still active
- some modules in the repo are exploratory or future-facing rather than core to the current runtime
- this is not production-hardened software yet

If you want the shortest path through the repo, start with `README.md`, `docs/REPO_MAP.md`, `src/main.rs`, `src/physics/naked_llama.rs`, and `scripts/chat_raw.py`.

Repo:
https://github.com/Ruffian-L/niodoo-autonomous-self-correction-and-dynamic-control-loops-in-ai
