# Terminology Notes

This repository contains a mix of technical terminology and legacy metaphor-heavy naming from earlier experiments and recovery work.

For public-facing explanations, paper drafts, and code comments, prefer the technical descriptions below.

## Preferred Technical Terms

- `activation perturbation`: modification of intermediate activations during decoding
- `steering vector`: an auxiliary vector used to bias the trajectory of generation
- `counter-steering vector`: a vector used to reduce or repel undesirable trajectories
- `stochastic perturbation`: controlled noise injected during decoding
- `repulsion targets`: tokens or regions that the runtime explicitly pushes away from
- `ramp schedule`: delayed application of steering at the start of decoding
- `telemetry` or `introspection`: runtime-visible measurements of steering state
- `request tags`: emitted verbal control tags such as `[REQUEST: FOCUS]`
- `runtime parameter update`: the state change caused by a parsed request tag

## Legacy Terms You May Still See In Code

These are retained in some places for compatibility or because they are part of the recovered codebase:

- `ghost_vector`: legacy name for an auxiliary steering vector
- `anti_ghost_vector`: legacy name for a counter-steering vector
- `wobble`: informal name for stochastic perturbation
- `black holes`: informal name for strongly repulsive token targets
- `Cognitive Mirror`, `Internal Monitor`, `Autonomic Override`: runtime prompt language used by the current system

## Important Distinction

The runtime control tags themselves are not just decorative terminology. They are part of the live control surface of the current system.

That means public documentation may describe them technically, but the emitted forms such as `[REQUEST: FOCUS]` and `[REQUEST: SPIKE]` should remain unchanged unless a compatibility-breaking change is intentional.

## Writing Guidance

If you are writing about Niodoo in a paper-friendly style:

- describe the system as a hybrid inference-time steering runtime
- distinguish prompt-level control semantics from actual activation perturbation
- avoid anthropomorphic claims unless you are explicitly discussing generated self-modeling behavior
- separate exploratory modules from the current working runtime path
