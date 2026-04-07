# Security Policy

## Scope

This repository is an experimental local assistant runtime and research snapshot.

It includes source code, documentation, vendored Rust dependencies for offline builds, and small repo-safe assets.

It does not intentionally include:

- model weights
- local `.safetensors` universe files
- local telemetry logs
- personal runtime data
- API keys, tokens, or credentials

## Reporting

If you believe you have found a real security issue, secret exposure, or accidental inclusion of sensitive material, please open a private security advisory or contact the maintainer directly before posting public details.

When reporting, include:

- the file path or component involved
- reproduction steps
- impact assessment
- whether the issue affects only local use or could affect downstream users

## Current Status

This project is experimental and not production-hardened.

Examples of known limitations:

- behavior varies by model and quantization
- control-loop tuning is still under active iteration
- interfaces may change without notice
- code paths not needed for the core local runtime may remain rough or incomplete

## Dependency Vendoring

The `vendor/` directory is present intentionally so the Rust project can build offline.

Secret scanners often flag vendored dependency files, checksums, GUID-like constants, or test fixtures as possible secrets. In this repository, those hits should be treated as scanner noise unless independently verified.

A follow-up TruffleHog pass on authored repo content, excluding `vendor/`, `target/`, local model assets, and the universe tensor, returned:

- `verified_secrets: 0`
- `unverified_secrets: 0`

## Asset Handling

Large local runtime assets should remain local and out of git unless there is a specific reason to publish them and their licensing allows it.

That includes:

- GGUF model files
- `.safetensors` universe files
- generated local artifacts
- logs or telemetry containing private conversations
