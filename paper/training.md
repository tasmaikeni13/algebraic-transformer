# Training and sweep protocol

## Matched configurations

The standard model uses RMSNorm, SwiGLU, RoPE, softmax attention, and cross-entropy. The pure algebraic model uses learned RMS gains, QK normalization, Cayley rotations, ALU-GLU, learned-sink algebraic attention, and an algebraic likelihood head. Both tie the input embedding and output projection.

| Scale | Layers | Width | Heads | Head width | Feed-forward width | Budget |
|---|---:|---:|---:|---:|---:|---:|
| 350M | 24 | 1,024 | 16 | 64 | 2,688 | 7B tokens |
| 1B | 32 | 1,536 | 24 | 64 | 4,224 | 20B tokens |

The configurations in `configs/comparison_350m.json` and `configs/comparison_1b.json` are authoritative. They specify one training seed (42), data seed 42, context 2,048, global batch 512, AdamW β₁=0.9/β₂=0.95, clipping at 1, 2% warmup, cosine decay to 10% of peak learning rate, and fused attention/head execution. The selected learning rate and weight decay override their planning defaults.

On 16 TPU chips, the per-device micro-batches are 2 and 1, with 16 and 32 accumulation iterations per optimizer step. Parameters, moments, and averaged gradients are replicated. Rematerialization limits activation memory. Parameter counts include the tied embedding once; QK gains and sinks add 3,456 parameters at 350M and 4,864 at 1B.

Token budgets round up to full steps, yielding 6,676 and 19,074 steps. Data sampling uses deterministic contiguous windows keyed by the data seed and step. It does not promise that every corpus token is presented exactly once. Validation uses contiguous held-out windows shared across models.

## Budgeted pilots

`configs/sweep_350m.json` and `configs/sweep_1b.json` define four candidates per architecture. Three explore learning rate; the fourth lowers weight decay at the central rate. The pilot budgets are 100M and 200M tokens per candidate. Eight runs require approximately 805M and 1.602B token presentations after rounding.

Pilots use one seed, the full model dimensions, the same fused paths as the main run, and approximately 1M held-out tokens. They disable intermediate evaluation, keep one optimizer checkpoint, and omit the final weight export. They are designed to screen settings cheaply; they do not establish the optimum at 7B/20B tokens or quantify training-seed variance.

`hparam_sweep.py --stage select` checks all eight jobs against their requested configuration, seed, corpus paths, token count, step count, and finite results. It writes independently selected learning rates and weight decays plus result hashes. Missing or mismatched candidates prevent selection. The main comparison checks that this selection matches its frozen configuration before training.

## Workers and files

Run from worker zero of a four-host TPU v4 slice. Install `requirements-tpu.txt` and the package on each host. The Python executable, repository, and corpus files must have identical absolute paths. Configure SSH aliases before running `setup_workers.sh`.

Data preparation pins the FineWeb-Edu source revision and discovers the parquet files rather than assuming a fixed shard count. Use `sample/10BT` for the 7B corpus and `sample/100BT` for the 20B corpus. GPT-2 IDs are stored as uint16 arrays. Validation and training contain separate documents; partial arrays lack a completion marker and are never accepted as a finished corpus.

Each full run retains two optimizer checkpoints. A checkpoint includes parameters, moments, step, sampler identity, recipe, and elapsed time. Resume rejects a changed seed, corpus path, schedule, optimizer recipe, model configuration, or total step count. Final model weights use `params.npz` with a matching `meta.json`, `result.json`, and training log. The launcher stops on worker failure. Optimizer checkpoints are retained for deliberate cleanup by the author.

## Commands

After preparing corpus arrays and benchmark data, run the short sweep, select, and launch the full comparison:

```bash
python scripts/hparam_sweep.py --scale 350m --stage run
python scripts/hparam_sweep.py --scale 350m --stage select
python scripts/run_comparison.py --scale 350m --run

python scripts/hparam_sweep.py --scale 1b --stage run
python scripts/hparam_sweep.py --scale 1b --stage select
python scripts/run_comparison.py --scale 1b --run
```

`--hosts`, `--data-dir`, `--benchmarks`, `--recall-benchmarks`, `--eval-chip`, and `--eval-batch` allow explicit locations and evaluation batch sizes. The default evaluation batch is one, suitable for both scales. Both models finish training before the public and recall suites are launched on one local TPU chip.

Evaluation can also be run independently:

```bash
bash scripts/tpu_python.sh 0 scripts/evaluate_main.py \
  --config configs/comparison_350m.json --hparams results/tuning_350m/selected.json \
  --runs-root runs/comparison_350m --out-dir results/comparison_350m \
  --benchmarks ~/data/evals --recall-benchmarks ~/data/recall_350m --batch 1 --resume

bash scripts/tpu_python.sh 0 scripts/evaluate_main.py \
  --config configs/comparison_1b.json --hparams results/tuning_1b/selected.json \
  --runs-root runs/comparison_1b --out-dir results/comparison_1b \
  --benchmarks ~/data/evals --recall-benchmarks ~/data/recall_1b --batch 1 --resume
```

Adding `--dry-run` writes evaluation plans without weights or an accelerator. Scripts without an explicit execution flag only print plans. Full runs have not been executed as part of repository preparation; the author will launch them. Smoke reports verify actual model shapes and fused execution using synthetic tokens.
