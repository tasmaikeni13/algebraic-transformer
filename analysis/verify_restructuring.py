#!/usr/bin/env python3
"""Comprehensive verification suite for the analysis restructuring."""

import importlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EXPECTED_MODULES = {
    "figures": ["plot_figures.py", "plot_paper_figures.py", "summarize_runs.py"],
    "kernels": [
        "attention_performance.py", "attention_resident_probe.py", "attention_head_resident_probe.py",
        "attention_streamed_probe.py", "attention_backward_schedule_probe.py", "attention_checkpoint_probe.py",
        "attention_segmented_probe.py", "attention_statistics.py", "attention_head_statistics.py",
        "attention_followup_statistics.py", "head_gradient_cache_probe.py", "qk_rms_kernel.py",
        "qk_normalization_probe.py", "qk_model_validation.py"
    ],
    "certificates": [
        "activation_analysis.py", "attention_numerics.py", "attention_polynomial.py",
        "attention_interpolation_certificate.py", "attention_newton_certificate.py",
        "attention_anchored_certificate.py", "attention_rational_radical.py",
        "attention_rational_radical_probe.py"
    ],
    "profiling": ["model_component_performance.py", "model_step_performance.py", "model_profile.py"],
    "experimental": [
        "attention_link_analysis.py", "attention_link_probe.py", "attention_link_statistics.py",
        "attention_recall_probe.py", "attention_radical_probe.py", "attention_interpolation_probe.py",
        "attention_projective_probe.py", "attention_affine_guard_probe.py"
    ],
    "scaling": ["scaling_jobs.py", "scaling_summary.py", "verify_head_eval.py"]
}


def test_directory_structure():
    """Verify all 6 subdirectories exist and hold exact expected scripts."""
    print("Testing directory structure...")
    total_scripts = 0
    for mod, expected_files in EXPECTED_MODULES.items():
        mod_dir = ANALYSIS / mod
        assert mod_dir.is_dir(), f"Missing directory: {mod_dir}"
        assert (mod_dir / "__init__.py").is_file(), f"Missing __init__.py in {mod_dir}"
        
        actual_files = sorted([p.name for p in mod_dir.glob("*.py") if p.name != "__init__.py"])
        assert actual_files == sorted(expected_files), f"Mismatch in {mod}: {actual_files} != {expected_files}"
        total_scripts += len(actual_files)
    
    assert total_scripts == 39, f"Expected 39 scripts across modules, got {total_scripts}"
    print(f"  [PASS] All 6 subdirectories verified ({total_scripts} modular scripts).")


def test_symlinks():
    """Verify all 39 symlinks at analysis/ root point to valid files."""
    print("Testing backward-compatible symlinks...")
    symlink_count = 0
    for mod, expected_files in EXPECTED_MODULES.items():
        for s in expected_files:
            symlink_path = ANALYSIS / s
            assert symlink_path.is_symlink(), f"Expected symlink at {symlink_path}"
            assert symlink_path.is_file(), f"Symlink points to non-existent target: {symlink_path}"
            target = symlink_path.resolve()
            expected_target = (ANALYSIS / mod / s).resolve()
            assert target == expected_target, f"Symlink target mismatch: {target} != {expected_target}"
            symlink_count += 1
    
    assert symlink_count == 39, f"Expected 39 symlinks, got {symlink_count}"
    print(f"  [PASS] All 39 root symlinks resolve correctly.")


def test_python_syntax():
    """Verify that every script compiles cleanly without syntax errors."""
    print("Testing syntax compilation for all scripts...")
    for mod, expected_files in EXPECTED_MODULES.items():
        for s in expected_files:
            script_path = ANALYSIS / mod / s
            source = script_path.read_text()
            try:
                compile(source, str(script_path), "exec")
            except Exception as e:
                raise AssertionError(f"Syntax error compiling {script_path}: {e}")
    print("  [PASS] All 38 scripts compile with zero syntax errors.")


def test_import_compatibility():
    """Verify that modular and legacy imports work cleanly."""
    print("Testing modular and legacy imports...")
    
    # 1. Modular import
    import analysis.kernels.attention_performance as kp
    assert hasattr(kp, "ratio_ci")
    
    # 2. Legacy import via symlink / dynamic getattr
    import analysis.attention_performance as lp
    assert hasattr(lp, "ratio_ci")
    
    # 3. Direct function import from modular
    from analysis.kernels.attention_performance import ratio_ci as rci_mod
    
    # 4. Direct function import from legacy
    from analysis.attention_performance import ratio_ci as rci_leg
    
    # Verify functional behavior is identical
    res_mod = rci_mod([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    res_leg = rci_leg([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert res_mod == res_leg, f"Results differ: {res_mod} != {res_leg}"
    
    # 5. Certificates import
    from analysis.certificates.activation_analysis import alu as alu_mod
    from analysis.activation_analysis import alu as alu_leg
    x_test = 1.5
    assert alu_mod(x_test, 1.0) == alu_leg(x_test, 1.0)
    
    # 6. Figures import
    import analysis.figures.plot_figures as pf
    import analysis.plot_figures as pf_legacy
    assert hasattr(pf, "save")
    assert hasattr(pf_legacy, "save")
    
    print("  [PASS] Modular and legacy imports function identically.")


def test_cli_dispatcher():
    """Verify run_analysis.py commands."""
    print("Testing run_analysis.py CLI...")
    cli = str(ANALYSIS / "run_analysis.py")
    
    # Test 1: list
    res = subprocess.run([sys.executable, cli, "list"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Algebraic Transformer Analysis Suite" in res.stdout
    assert "plot_figures.py" in res.stdout
    assert "attention_performance.py" in res.stdout
    
    # Test 2: module help
    res = subprocess.run([sys.executable, cli, "figures", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "plot_figures.py" in res.stdout
    
    # Test 3: module subcommand execution
    res = subprocess.run([sys.executable, cli, "figures", "plot_figures", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--results-root" in res.stdout
    
    # Test 4: direct shortcut execution
    res = subprocess.run([sys.executable, cli, "plot_figures", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--results-root" in res.stdout
    
    print("  [PASS] run_analysis.py CLI fully functional.")


def main():
    print("=" * 60)
    print("Running Analysis Restructuring Verification Suite")
    print("=" * 60)
    test_directory_structure()
    test_symlinks()
    test_python_syntax()
    test_import_compatibility()
    test_cli_dispatcher()
    print("=" * 60)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (5/5)")
    print("=" * 60)


if __name__ == "__main__":
    main()
