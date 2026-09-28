# Lean verification

The Lean toolchain is pinned in `lean-toolchain`; Mathlib is pinned in
`lake-manifest.json`. Build the imported modules with:

```bash
cd formal
lake exe cache get
lake build
```

[`PROOF_COVERAGE.md`](PROOF_COVERAGE.md) lists the principal theorems and
their assumptions. The certificates concern real-algebra identities. They do
not establish floating-point correctness, model quality, or hardware speed.
