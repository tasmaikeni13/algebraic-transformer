#!/usr/bin/env python3
"""Phase 1 ablation job lists for scripts/run_queue.py (proxy model, one chip).

Common protocol: ~33M-parameter proxy (d=384, 8 layers, 6 heads of 64),
context 1024, batch 64 sequences (65,536 tokens), 3,000 AdamW steps
(196.6M tokens), 2% warmup, b1 0.9, b2 0.95, clip 1, held-out evaluation on
2M FineWeb-Edu-1B validation tokens. Seeds fix initialization and data order,
so runs with equal seeds see identical batches (paired comparisons).

  python analysis/phase1_jobs.py batch1 > /tmp/jobs.json
"""

import json
import sys

from src.sota.algebraic import kernel_sink_logit

BASE = ["--size", "proxy", "--seq-len", "1024", "--batch", "64", "--micro", "16",
        "--steps", "3000", "--warmup", "0.02", "--b2", "0.95", "--eval-every", "500",
        "--eval-tokens", "2e6"]

# The first study's algebraic components, as config overrides on the standard model.
V1_COMPONENTS = {
    "norm": ["norm=avn"],
    "ffn": ["ffn=alu_glu", "alu_beta=1.0"],
    "pos": ["pos=cayley_orig"],
    "attn": ["attn=asinh", "attn_n=8", "attn_c=1.0", "sink=const",
             f"sink_init={kernel_sink_logit(1.0, 8, 1.0)}"],
    "head": ["head=oace", "eps_vocab=100.0", "gamma=2.0"],
}
STANDARD_COMPONENTS = {
    "norm": ["norm=rms"],
    "ffn": ["ffn=swiglu"],
    "pos": ["pos=rope"],
    "attn": ["attn=softmax", "sink=none"],
    "head": ["head=ce"],
}
V1_RECIPE = ["--schedule", "ards", "--wd", "0.01"]
STD_RECIPE = ["--schedule", "cosine", "--min-lr-ratio", "0.1", "--wd", "0.1"]


def job(name, preset, sets=(), extra=(), lr=2e-3, seed=0):
    args = ["--preset", preset, *BASE, "--lr", str(lr), "--seed", str(seed)]
    for s in sets:
        args += ["--set", s]
    args += list(extra)
    return {"name": name, "args": args}


def batch1():
    jobs = []
    for lr in (1e-3, 2e-3, 4e-3):
        jobs.append(job(f"std_lr{lr:g}", "standard", extra=STD_RECIPE, lr=lr))
        jobs.append(job(f"v1_lr{lr:g}", "algebraic_v1", extra=V1_RECIPE, lr=lr))
    jobs.append(job("v1_stdrecipe", "algebraic_v1", extra=STD_RECIPE))
    jobs.append(job("std_v1recipe", "standard", extra=V1_RECIPE))
    for comp, sets in V1_COMPONENTS.items():
        jobs.append(job(f"std+v1{comp}", "standard", sets=sets, extra=STD_RECIPE))
    jobs.append(job("v2prov", "algebraic", extra=STD_RECIPE))
    jobs.append(job("std_lr0.002_s1", "standard", extra=STD_RECIPE, seed=1))
    return jobs


def batch2():
    jobs = []
    # reverse ablation: the first-study model with one costly component made standard
    for comp in ("head", "norm", "attn"):
        jobs.append(job(f"v1-{comp}", "algebraic_v1", sets=STANDARD_COMPONENTS[comp], extra=STD_RECIPE))
    # second-generation exploration around the provisional algebraic config
    v2 = {
        "v2_attn16x16": ["attn_n=16", "attn_c=16.0"],
        "v2_attnsoftmax": ["attn=softmax"],
        "v2_headce": ["head=ce"],
        "v2_head8x2": ["head_n=8", "head_c=2.0"],
        "v2_head32x32": ["head_n=32", "head_c=32.0"],
        "v2_cayleyorig": ["pos=cayley_orig"],
        "v2_qknorm": ["qk_norm=true"],
        "v2_alu0.5": ["alu_beta=0.5"],
    }
    for name, sets in v2.items():
        jobs.append(job(name, "algebraic", sets=sets, extra=STD_RECIPE))
    jobs.append(job("v2prov_s1", "algebraic", extra=STD_RECIPE, seed=1))
    jobs.append(job("std+sink", "standard", sets=["sink=learned"], extra=STD_RECIPE))
    jobs.append(job("std_lr0.002_s2", "standard", extra=STD_RECIPE, seed=2))
    jobs.append(job("v2prov_s2", "algebraic", extra=STD_RECIPE, seed=2))
    return jobs


def batch3():
    """Combine the batch-2 winners; push the head link; attribution controls."""
    flash = ["attn_impl=flash"]
    v2b = flash + ["head_n=8", "head_c=2.0", "pos=cayley_orig", "alu_beta=0.5"]
    jobs = [
        job("v2b", "algebraic", sets=v2b, extra=STD_RECIPE),
        job("v2b_s1", "algebraic", sets=v2b, extra=STD_RECIPE, seed=1),
        job("v2b_attnsoftmax", "algebraic", sets=v2b + ["attn=softmax"], extra=STD_RECIPE),
        job("v2b_head8x1", "algebraic", sets=v2b + ["head_c=1.0"], extra=STD_RECIPE),
        job("v2b_head4x1", "algebraic", sets=v2b + ["head_n=4", "head_c=1.0"], extra=STD_RECIPE),
        job("v2b_head16x4", "algebraic", sets=v2b + ["head_n=16", "head_c=4.0"], extra=STD_RECIPE),
        job("v2b_head4x0.5", "algebraic", sets=v2b + ["head_n=4", "head_c=0.5"], extra=STD_RECIPE),
        job("v2b_attn8x2", "algebraic", sets=v2b + ["attn_c=2.0"], extra=STD_RECIPE),
        job("v2b_attn4x1", "algebraic", sets=v2b + ["attn_n=4"], extra=STD_RECIPE),
        job("v2b_qknorm", "algebraic", sets=v2b + ["qk_norm=true"], extra=STD_RECIPE),
        job("v2b_lr0.003", "algebraic", sets=v2b, extra=STD_RECIPE, lr=3e-3),
        job("std+head8x2", "standard", sets=flash + ["head=alg_ce", "head_n=8", "head_c=2.0"], extra=STD_RECIPE),
        job("std_lr0.003", "standard", sets=flash, extra=STD_RECIPE, lr=3e-3),
        job("v2b_qk", "algebraic", sets=v2b + ["qk_norm=true"], extra=STD_RECIPE, seed=1),
        job("v2b_qk_attnsoftmax", "algebraic", sets=v2b + ["qk_norm=true", "attn=softmax"], extra=STD_RECIPE),
        job("std+qknorm", "standard", sets=flash + ["qk_norm=true"], extra=STD_RECIPE),
    ]
    return jobs


def batch4():
    """Seeds for the leaders, attention kernel under QK-norm, honest controls."""
    flash = ["attn_impl=flash"]
    v2c = flash + ["head_n=8", "head_c=2.0", "pos=cayley_orig", "alu_beta=0.5", "qk_norm=true"]
    jobs = [job(f"v2c_s{s}", "algebraic", sets=v2c, extra=STD_RECIPE, seed=s) for s in (2, 3)]
    jobs += [job(f"hyb_s{s}", "algebraic", sets=v2c + ["attn=softmax"], extra=STD_RECIPE, seed=s) for s in (1, 2)]
    jobs += [job(f"std_flash_s{s}", "standard", sets=flash, extra=STD_RECIPE, seed=s) for s in (0, 3)]
    for name, extra_sets in {"attn8x2": ["attn_c=2.0"], "attn16x4": ["attn_n=16", "attn_c=4.0"],
                             "attn16x8": ["attn_n=16", "attn_c=8.0"], "attn4x1": ["attn_n=4"],
                             "nosink": ["sink=none"], "geo": ["pos=cayley_geo"],
                             "alu1": ["alu_beta=1.0"]}.items():
        jobs.append(job(f"v2c_{name}", "algebraic", sets=v2c + extra_sets, extra=STD_RECIPE))
    jobs.append(job("std+qk+head", "standard", sets=flash + ["qk_norm=true", "head=alg_ce", "head_n=8",
                                                             "head_c=2.0"], extra=STD_RECIPE))
    jobs.append(job("std+qk+head+sink+pos", "standard", sets=flash + [
        "qk_norm=true", "head=alg_ce", "head_n=8", "head_c=2.0", "sink=learned", "pos=cayley_orig"],
        extra=STD_RECIPE))
    jobs.append(job("v2c_s0_save", "algebraic", sets=v2c, extra=STD_RECIPE + ["--save-params"]))
    return jobs


def batch5():
    """Close the backbone gap: combine the best attention kernel and grid."""
    flash = ["attn_impl=flash"]
    v2d = flash + ["head_n=8", "head_c=2.0", "pos=cayley_geo", "alu_beta=0.5", "qk_norm=true",
                   "attn_n=16", "attn_c=8.0"]
    jobs = [job(f"v2d_s{s}", "algebraic", sets=v2d, extra=STD_RECIPE, seed=s) for s in range(4)]
    variants = {"attn32x16": ["attn_n=32", "attn_c=16.0"], "attn16x16": ["attn_c=16.0"],
                "attn16x4": ["attn_c=4.0"], "alu0.6": ["alu_beta=0.6"], "alu0.75": ["alu_beta=0.75"],
                "rot0.5": ["rope_fraction=0.5"], "head8x3": ["head_c=3.0"], "head8x1.5": ["head_c=1.5"],
                "nosink": ["sink=none"]}
    for name, extra_sets in variants.items():
        jobs.append(job(f"v2d_{name}", "algebraic", sets=v2d + extra_sets, extra=STD_RECIPE))
    jobs.append(job("v2d_lr0.0015", "algebraic", sets=v2d, extra=STD_RECIPE, lr=1.5e-3))
    ctrl = flash + ["qk_norm=true", "head=alg_ce", "head_n=8", "head_c=2.0"]
    jobs += [job(f"std+qk+head_s{s}", "standard", sets=ctrl, extra=STD_RECIPE, seed=s) for s in (1, 2)]
    return jobs


if __name__ == "__main__":
    print(json.dumps(globals()[sys.argv[1]](), indent=1))
