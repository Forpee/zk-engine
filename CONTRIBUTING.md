# Contributing

## Development setup

Install the pinned toolchain (`rustup` does this automatically from
`rust-toolchain.toml`) plus `cargo-nextest` and `taplo-cli`.

## Before opening a PR

```bash
cargo fmt --all
taplo fmt
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p jolt-verifier -p jolt-kernels -p jolt-prover --all-targets \
  --features jolt-prover/zk,jolt-prover/profiling,jolt-prover/allocative -- -D warnings
python3 scripts/check_style_invariants.py --base origin/main
cargo nextest run --workspace --cargo-quiet
cargo nextest run -p jolt-prover --features zk --cargo-quiet -E 'binary(wasm_e2e)'
```

`.github/workflows/rust.yml` runs the same steps.

## Code style

The invariants are in [`CLAUDE.md`](CLAUDE.md) (Development Guidelines); the
machine-checked subset is enforced by `scripts/check_style_invariants.py`.
Design documents for larger changes live in [`specs/`](specs/).

## Commit messages

Conventional Commits: `feat`, `fix`, `chore`, `docs`, `refactor`, `perf`,
`test`, `build`, `ci`, `style`, `revert`, `spec`, optionally scoped
(`feat(frontend): ...`).
