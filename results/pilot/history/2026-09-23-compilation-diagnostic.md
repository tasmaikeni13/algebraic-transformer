# Compilation diagnostic, 23 September 2026

Commit `6af306ae88aa38590e14abcd0647f673d708ac5e` reached the standard
model compilation on four TPU hosts. All hosts passed the CPU tests. The
maintained Mosaic/Pallas attention call failed inside a mesh-wide SPMD `jit`:

```text
NotImplementedError: Mosaic kernels cannot be automatically partitioned. Please wrap the call in a shard_map.
```

The job stopped before its first optimizer update, so it contains no training
measurement. The training step uses an explicit `shard_map` boundary for this
kernel; the standard-model step has a regression test for that path.
