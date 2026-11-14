#!/usr/bin/env python3
"""Unified CLI runner and dispatcher for the Algebraic Transformer analysis suite.

Modules:
  figures       Publication plotting and figure generation
  kernels       TPU Pallas FlashAttention and Fused Head probes and benchmarks
  certificates  Formal real-arithmetic certificates and numerical bounds
  profiling     Component and step performance profiling scripts
  experimental  Exploratory attention link probes
  scaling       Scaling jobs and power-law summaries

Usage:
  python analysis/run_analysis.py list
  python analysis/run_analysis.py <module> <script> [args...]
  python analysis/run_analysis.py <script> [args...]
"""

import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"

MODULES = {
    "figures": {
        "desc": "Publication plotting and figure generation",
        "scripts": [
            "plot_figures.py",
            "plot_paper_figures.py",
            "summarize_runs.py",
        ],
    },
    "kernels": {
        "desc": "TPU Pallas FlashAttention and Fused Head probes/benchmarks",
        "scripts": [
            "attention_performance.py",
            "attention_resident_probe.py",
            "attention_head_resident_probe.py",
            "attention_streamed_probe.py",
            "attention_backward_schedule_probe.py",
            "attention_checkpoint_probe.py",
            "attention_segmented_probe.py",
            "attention_statistics.py",
            "attention_head_statistics.py",
            "attention_followup_statistics.py",
            "head_gradient_cache_probe.py",
            "qk_rms_kernel.py",
            "qk_normalization_probe.py",
            "qk_model_validation.py",
        ],
    },
    "certificates": {
        "desc": "Formal real-arithmetic certificates and numerical bounds",
        "scripts": [
            "activation_analysis.py",
            "attention_numerics.py",
            "attention_polynomial.py",
            "attention_interpolation_certificate.py",
            "attention_newton_certificate.py",
            "attention_anchored_certificate.py",
            "attention_rational_radical.py",
            "attention_rational_radical_probe.py",
        ],
    },
    "profiling": {
        "desc": "Component and step performance profiling scripts",
        "scripts": [
            "model_component_performance.py",
            "model_step_performance.py",
            "model_profile.py",
        ],
    },
    "experimental": {
        "desc": "Exploratory attention link probes",
        "scripts": [
            "attention_link_analysis.py",
            "attention_link_probe.py",
            "attention_link_statistics.py",
            "attention_recall_probe.py",
            "attention_radical_probe.py",
            "attention_interpolation_probe.py",
            "attention_projective_probe.py",
            "attention_affine_guard_probe.py",
        ],
    },
    "scaling": {
        "desc": "Scaling jobs and power-law summaries",
        "scripts": [
            "scaling_jobs.py",
            "scaling_summary.py",
            "verify_head_eval.py",
        ],
    },
}

SCRIPT_TO_MODULE = {}
for mod_name, mod_info in MODULES.items():
    for script_name in mod_info["scripts"]:
        SCRIPT_TO_MODULE[script_name] = mod_name
        if script_name.endswith(".py"):
            SCRIPT_TO_MODULE[script_name[:-3]] = mod_name


def get_docstring(script_path: Path) -> str:
    """Extract first line of docstring without importing."""
    if not script_path.exists():
        return ""
    try:
        tree = ast.parse(script_path.read_text())
        doc = ast.get_docstring(tree)
        if doc:
            for line in doc.strip().splitlines():
                line = line.strip()
                if line:
                    return line
    except Exception:
        pass
    return ""


def print_list():
    """Print complete listing of modules and scripts."""
    print("=" * 80)
    print("Algebraic Transformer Analysis Suite — Module Index")
    print("=" * 80)
    for mod_name, info in MODULES.items():
        print(f"\n[{mod_name.upper()}] — {info['desc']}")
        print("-" * 80)
        for s in info["scripts"]:
            spath = ANALYSIS_DIR / mod_name / s
            doc = get_docstring(spath)
            print(f"  {s:<36} {doc}")
    print("\n" + "=" * 80)
    print("Run with: python analysis/run_analysis.py <module> <script> [args...]")
    print("Example:  python analysis/run_analysis.py figures plot_figures --help")
    print("=" * 80)


def print_module_help(mod_name: str):
    """Print help for a specific module."""
    info = MODULES[mod_name]
    print(f"Module: {mod_name}")
    print(f"Description: {info['desc']}\n")
    print("Scripts in this module:")
    for s in info["scripts"]:
        spath = ANALYSIS_DIR / mod_name / s
        doc = get_docstring(spath)
        print(f"  {s:<36} {doc}")
    print(f"\nRun with: python analysis/run_analysis.py {mod_name} <script> [args...]")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        print("\nAvailable modules:")
        for name, info in MODULES.items():
            print(f"  {name:<14} {info['desc']} ({len(info['scripts'])} scripts)")
        print("\nUse 'python analysis/run_analysis.py list' for full script index.")
        sys.exit(0)

    if args[0] in ("list", "--list", "-l"):
        print_list()
        sys.exit(0)

    first = args[0]

    # Case 1: First argument is a module name
    if first in MODULES:
        mod_name = first
        sub_args = args[1:]
        if not sub_args or sub_args[0] in ("-h", "--help"):
            print_module_help(mod_name)
            sys.exit(0)
        script_arg = sub_args[0]
        script_name = script_arg if script_arg.endswith(".py") else f"{script_arg}.py"
        script_path = ANALYSIS_DIR / mod_name / script_name
        extra_args = sub_args[1:]

    # Case 2: First argument is directly a script name
    elif first in SCRIPT_TO_MODULE:
        mod_name = SCRIPT_TO_MODULE[first]
        script_name = first if first.endswith(".py") else f"{first}.py"
        script_path = ANALYSIS_DIR / mod_name / script_name
        extra_args = args[1:]

    else:
        print(f"Error: Unknown module or script '{first}'.", file=sys.stderr)
        print("Run 'python analysis/run_analysis.py list' to see all available scripts.", file=sys.stderr)
        sys.exit(1)

    if not script_path.exists():
        print(f"Error: Script '{script_path}' not found.", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(script_path), *extra_args]
    result = subprocess.run(cmd, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
