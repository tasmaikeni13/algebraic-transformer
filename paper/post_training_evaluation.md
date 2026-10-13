# Post-training evaluation protocol

The completed 125M / 2.5B-token comparison evaluates standard, algebraic, and hybrid models with seeds 42, 43, 44. The implemented 350M / 7B-token and 1B / 20B-token comparisons evaluate standard and pure algebraic with one paired seed (42), after their budgeted hyperparameter sweeps. The same public prompts, recall matrix, tokenizer, and scoring rules apply at every scale.

The frozen plans are [125M](../configs/post_training_eval.json), [350M](../configs/comparison_350m.json), and [1B](../configs/comparison_1b.json). The 125M [published scores](../results/main/comparison.md) are measured; full 350M and 1B quality runs remain for the author to launch. [Training and sweep commands](training.md) describe the larger comparisons.

## What each benchmark measures

| Public task | Capability | Evaluated split |
|---|---|---|
| [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) | Grade-school science knowledge and relatively straightforward reasoning | ARC-Easy / test |
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | Harder science questions requiring knowledge and reasoning | ARC-Challenge / test |
| [HellaSwag](https://rowanzellers.com/hellaswag/) | Everyday commonsense; choosing the plausible continuation of a situation or activity | validation |
| [PIQA](https://arxiv.org/abs/1911.11641) | Physical commonsense; choosing an action that accomplishes a goal | validation |
| [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa) | Applying elementary science knowledge and commonsense | main / test |
| [WinoGrande](https://huggingface.co/datasets/allenai/winogrande) | Resolving references using sentence context and commonsense | winogrande_xl / validation |
| [BoolQ](https://arxiv.org/abs/1905.10044) | Reading comprehension; answering yes/no questions about a supplied passage | validation |
| [LAMBADA](https://arxiv.org/abs/1606.06031) | Predicting the final word using information from the broader passage | test |

The labeled validation splits are used where public test labels are
unavailable. OpenBookQA uses the question and choices, without the supplied
supporting fact from the `additional` configuration. Held-out FineWeb-Edu
NLL and perplexity are also saved by the existing trainer; they measure
general next-token prediction under each model's normalized distribution.

Multiple-choice accuracy uses the **mean continuation-token log-probability**,
including the completed sentence suffix for WinoGrande. BoolQ compares
` yes` and ` no` after the passage and question. LAMBADA requires an exact
greedy match of the final word's token prefix. Matching every teacher-forced
target argmax is equivalent to that free greedy prefix matching, by causality.
These are project protocol scores; external harnesses may use different
prompts, preprocessing, or length normalization.

Public task contexts may be left-truncated to the trained context, with
the number of truncated choice/target pairs reported for each task.
Recall prompts use a stricter protocol described below.

## Recall suite

| Task | Capability |
|---|---|
| `needle_passkey` | Retrieve one random six-digit code hidden in neutral prose |
| `associative_recall` | Select the code for a requested key among many unrelated key–value records |
| `multi_needle` | Retrieve three codes at different positions and return them in the requested order |
| `latest_value` | Retrieve the last of three updates to one key, ignoring obsolete values |

These are deterministic **project-specific** retrieval tasks, not official
RULER or other external benchmark scores. Related task patterns are used
in [NVIDIA's RULER suite](https://github.com/NVIDIA/RULER).

### Frozen matrix and controls

- Total context budgets: **256, 512, 1,024, 2,048 GPT-2 tokens**.
- Needle positions: **0%, 25%, 50%, 75%, 100%** of the distractor region.
- **32 trials** per task, budget, and position; dataset seed **20261012**.
- Each prompt includes **one fixed answer-format demonstration**. The public
  task suite is zero-shot; this recall suite is one-shot for format only.
- Reserve **16 tokens for free greedy generation**, temperature zero, for
  every model. The longest prompt is **2,032 tokens**, leaving all evidence
  and answer space within the trained 2,048-token context.
- At each matrix cell, an **absent-evidence control** uses the same query,
  target and prompt length, replacing all target facts with distractor tokens.
  Matching this unsupported target is a guess, not successful retrieval.
- **32 short-context controls per task** provide the complete facts directly
  next to the question. These help identify answer-format or basic task
  difficulties before interpreting a decline over longer contexts.

There are **640 evidence-present matrix examples**, **640 absent-evidence
controls**, and **32 short controls** per task: **5,248 examples overall**.
Keys, codes, choices and distractors are identical across architectures
and training seeds. Codes are random and excluded from distractors and
worked examples. Trials reuse their keys/codes across lengths and positions
to make those comparisons paired. They are not independent pooled trials.

For multiple needles, the other two positions are offset by one-third and
two-thirds of the distractor region, wrapping around. For latest-value recall,
the old updates appear at one-third and two-thirds of the requested latest
update position; ties retain chronological order. Stored needle spans and
actual token depths make these placements inspectable.

Only distractor material is cut to reach an exact token budget. Headers,
complete target facts and queries are retained. Authoritative raw token IDs
are saved alongside decoded text. The evaluator rejects any prompt or
recognition continuation that exceeds trained context, checks every present
needle span, and never extends context or truncates evidence.

### Recall metrics and interpretation

Primary: **exact retrieval** from free autoregressive generation, accepting
the leading requested six-digit codes in the correct order. Trailing prose
is allowed. Extra/wrong codes before the answer make the result incorrect.
The decoder never uses gold answers to stop or choose tokens.

Diagnostics: partial needle recall, answer-format validity, exact greedy
target-prefix match, target mean log-probability, and **four-choice recognition
accuracy** (25% random-choice baseline, balanced gold positions across the
32 trials). Latest-value choices include an
obsolete value as a distractor. Recognition and generation are reported
separately. Absent-evidence results are labeled **unsupported target hits**.

Per-cell JSON includes Wilson 95% intervals across the independent trials in
that cell. Comparison tables report **mean ± sample SD across training seeds**
and **paired differences from standard** in percentage points. They do not
treat the same trial across all lengths/positions, or multiple training
seeds on one example, as independent observations. Three seeds support a
descriptive comparison; they do not establish a universal recall guarantee.

The measured 125M results show low free retrieval for all architectures. Recognition, short controls, absent controls, and the entire length/position matrix are necessary to interpret those scores. Toy attention retrieval experiments do not establish a language model's post-training recall performance.

For the 350M and 1B comparisons, tables report the individual score and paired difference for seed 42. Sample SD across training seeds is unavailable and remains null; no additional training seeds are implied. Per-cell Wilson intervals describe trials within a cell and remain available.

## Commands

Prepare datasets once, before evaluating any model:

```bash
python scripts/prepare_eval_benchmarks.py --output-dir ~/data/evals
python scripts/prepare_recall_benchmarks.py --output-dir ~/data/recall
python scripts/evaluate_main.py --dry-run
```

After completing the nine 125M runs, evaluate on a single TPU
chip on `w0`, where the trainer saves the final weights:

```bash
bash scripts/tpu_python.sh 0 scripts/evaluate_main.py --resume
```

With another environment or dataset location:

```bash
python scripts/evaluate_main.py --runs-root runs/main \
  --benchmarks /path/to/evals --recall-benchmarks /path/to/recall \
  --out-dir results/main --resume
```

`--dry-run` writes the 18 evaluation commands without requiring checkpoints,
loading JAX, or starting training. The real run requires every selected final
`params.npz`, matching `meta.json`, finite `result.json`, and a completed
training loss history. It rejects limited preparation files. `--resume`
reuses an evaluation only when weights, metadata, source, dataset hashes,
protocol, batch size and prediction hash all match. Each scoring subprocess
releases accelerator memory before the next suite. On CPU/GPU the scorer
uses the dense attention implementation; comparisons require matching
backends, dtype and batch size across architectures/seeds.

For one model, or development-only subsets:

```bash
bash scripts/tpu_python.sh 0 scripts/evaluate_zero_shot.py runs/main/hybrid_s42 \
  --benchmarks ~/data/evals --out results/main/hybrid_s42_zero_shot.json
bash scripts/tpu_python.sh 0 scripts/evaluate_recall.py runs/main/hybrid_s42 \
  --benchmarks ~/data/recall --out results/main/hybrid_s42_recall.json
# --tasks needle_passkey --limit 8 is a development subset, not a main score.
```

Outputs in `results/main/`:

- Per-model/suite JSON and per-example prediction JSONL, with checkpoint
  hashes, dataset hashes, source fingerprint, execution settings and limits.
- `comparison.md`, `comparison.csv`, `comparison.json`: all task scores and
  matched-seed differences from standard. Markdown/JSON also include held-out
  NLL/perplexity and training time/throughput copied from the trainer.
- `recall_cells.csv`: the full length/position matrix for every recall task.
- `needle_recall.png` and `needle_recall.pdf`: shared-scale needle heatmaps
  for standard, algebraic and hybrid, generated from actual scores only.
- `evaluation_plan.json`: commands and completion status. Evaluation starts
  no training jobs and requires no instruction tuning.

## Larger comparisons

Use the same public dataset directory and prepare recall with the corresponding frozen configuration. The recall examples are identical across the larger scales because their recall specifications agree; each directory records its configuration hash.

```bash
python scripts/prepare_recall_benchmarks.py --config configs/comparison_350m.json --output-dir ~/data/recall_350m
python scripts/prepare_recall_benchmarks.py --config configs/comparison_1b.json --output-dir ~/data/recall_1b
python scripts/evaluate_main.py --config configs/comparison_350m.json --dry-run --out-dir /tmp/plan_350m
python scripts/evaluate_main.py --config configs/comparison_1b.json --dry-run --out-dir /tmp/plan_1b
```

`run_comparison.py --scale <350m|1b> --run` performs the full evaluation after both training runs, using the selected hyperparameters. Results go to `results/comparison_<scale>/`. The evaluator checks scale-specific dimensions, token budget, seed, trained context, fused implementation, selected recipe, and completion before scoring. It rejects short smoke runs and limited datasets. Every example and output is matched between the two architectures.
