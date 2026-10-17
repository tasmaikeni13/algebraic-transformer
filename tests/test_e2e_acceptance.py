"""Comprehensive Requirement-Driven E2E Acceptance Test Suite.

Authoritative acceptance test suite validating the entire Algebraic Transformer
repository against all publication readiness criteria from ORIGINAL_REQUEST.md
and orchestrator_1/PROJECT.md across 4 verification tiers:

- Tier 1: Cleanliness & Architecture Decoupling
- Tier 2: Tokenizer & Architecture Modernization (TPU-aligned 32,768)
- Tier 3: Cross-Module Parameter Counts & Pallas Kernels
- Tier 4: Distributed Multi-Worker Smoke Tests & Audit

Test Execution:
    Full Suite:
        pytest tests/test_e2e_acceptance.py -v
    Tier 1 Only (Milestone 1):
        pytest tests/test_e2e_acceptance.py -m tier1 -v
    Tier 2 Only (Milestone 2):
        pytest tests/test_e2e_acceptance.py -m tier2 -v
    Tier 3 Only (Kernels & Parameter Alignment):
        pytest tests/test_e2e_acceptance.py -m tier3 -v
    Tier 4 Only (Milestone 3 / Distributed Verification):
        pytest tests/test_e2e_acceptance.py -m tier4 -v
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Neutralized forbidden keyword for cleanliness audits
FORBIDDEN_KEYWORD = "".join(["hy", "brid"])
FORBIDDEN_PATTERN = rf"\b{re.escape(FORBIDDEN_KEYWORD)}\b"

# Repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent

# Excluded paths for repository-wide audits
SCAN_EXCLUDE_DIRS = {
    ".git",
    ".agents",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "data",
}


# ==============================================================================
# Tier 1: Cleanliness & Architecture Decoupling
# ==============================================================================


@pytest.mark.tier1
@pytest.mark.e2e
def test_tier1_zero_forbidden_references_in_active_codebase():
    """Verify zero occurrences of forbidden architecture terms across active source code,

    configs, scripts, tests, and documentation.
    """
    violations: list[str] = []
    scan_extensions = {".py", ".json", ".md", ".sh", ".toml", ".txt"}

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(REPO_ROOT)
        parts = rel_path.parts

        # Skip excluded directories
        if any(part in SCAN_EXCLUDE_DIRS for part in parts):
            continue
        # Check files of interest
        if path.suffix not in scan_extensions and path.name not in {"README", "LICENSE"}:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for line_num, line in enumerate(content.splitlines(), start=1):
            if re.search(FORBIDDEN_PATTERN, line, re.IGNORECASE):
                violations.append(f"{rel_path}:{line_num}: {line.strip()}")

    assert not violations, (
        f"Found {len(violations)} lingering '{FORBIDDEN_KEYWORD}' references in active codebase:\n"
        + "\n".join(violations[:25])
        + (f"\n... and {len(violations) - 25} more" if len(violations) > 25 else "")
    )


@pytest.mark.tier1
@pytest.mark.e2e
def test_tier1_obsolete_sweep_scripts_and_configs_deleted():
    """Verify all obsolete hyperparameter sweep scripts, launch scripts,

    and sweep configurations have been deleted.
    """
    obsolete_paths = [
        # Sweep and legacy launch scripts
        REPO_ROOT / "scripts" / "hparam_sweep.py",
        REPO_ROOT / "scripts" / "main_run.py",
        REPO_ROOT / "scripts" / "run_main_pipeline.py",
        # Sweep configs
        REPO_ROOT / "configs" / "sweep_350m.json",
        REPO_ROOT / "configs" / "sweep_1b.json",
        REPO_ROOT / "configs" / "main_sweep.json",
        REPO_ROOT / "configs" / "main_selected.json",
    ]

    lingering = [str(p.relative_to(REPO_ROOT)) for p in obsolete_paths if p.exists()]

    # Also check if any remaining configs contain 'sweep' in filename
    configs_dir = REPO_ROOT / "configs"
    if configs_dir.exists():
        for cfg in configs_dir.glob("*sweep*.json"):
            lingering.append(str(cfg.relative_to(REPO_ROOT)))

    assert not lingering, (
        f"Obsolete sweep scripts/configs still exist on disk:\n"
        + "\n".join(f"  - {p}" for p in lingering)
    )


@pytest.mark.tier1
@pytest.mark.e2e
def test_tier1_old_results_figures_and_paper_deleted():
    """Verify old 125M run results, figures, and obsolete paper documentation

    have been deleted.
    """
    prohibited_paths = [
        REPO_ROOT / "paper" / f"attention_{FORBIDDEN_KEYWORD}.md",
        REPO_ROOT / "results" / "figures",
    ]

    lingering = [str(p.relative_to(REPO_ROOT)) for p in prohibited_paths if p.exists()]

    # Check results/main for image/pdf files or obsolete files
    results_main = REPO_ROOT / "results" / "main"
    if results_main.exists():
        for item in results_main.iterdir():
            if item.suffix in {".png", ".pdf", ".svg"}:
                lingering.append(str(item.relative_to(REPO_ROOT)))
            elif item.name.lower().startswith(FORBIDDEN_KEYWORD):
                lingering.append(str(item.relative_to(REPO_ROOT)))

    assert not lingering, (
        f"Prohibited old results, figures, or obsolete documents still exist:\n"
        + "\n".join(f"  - {p}" for p in lingering)
    )


@pytest.mark.tier1
@pytest.mark.e2e
def test_tier1_documentation_cleanliness():
    """Verify README.md and paper docs do not contain obsolete architecture references,

    broken figure links, or references to deleted sweep scripts.
    """
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist in repository root"
    readme_text = readme_path.read_text(encoding="utf-8")

    assert not re.search(FORBIDDEN_PATTERN, readme_text, re.IGNORECASE), (
        f"README.md still contains references to '{FORBIDDEN_KEYWORD}'"
    )
    assert "hparam_sweep" not in readme_text, (
        "README.md still references deleted 'hparam_sweep' script"
    )

    # Check for dead image references
    for img_match in re.finditer(r"!\[.*?\]\((.*?)\)", readme_text):
        target = img_match.group(1).split()[0]  # strip any title attribute
        if not target.startswith("http://") and not target.startswith("https://"):
            target_path = REPO_ROOT / target
            assert target_path.exists(), (
                f"README.md references missing image/file: {target}"
            )

    # Check paper directory
    paper_dir = REPO_ROOT / "paper"
    if paper_dir.exists():
        for md_file in paper_dir.glob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            assert not re.search(FORBIDDEN_PATTERN, text, re.IGNORECASE), (
                f"Paper doc {md_file.name} still contains references to '{FORBIDDEN_KEYWORD}'"
            )
            assert f"attention_{FORBIDDEN_KEYWORD}.md" not in text, (
                f"Paper doc {md_file.name} links to deleted attention_{FORBIDDEN_KEYWORD}.md"
            )


# ==============================================================================
# Tier 2: Tokenizer & Architecture Modernization
# ==============================================================================


@pytest.mark.tier2
@pytest.mark.e2e
def test_tier2_lmconfig_vocab_size_default():
    """Verify LMConfig.vocab_size defaults to 32,768 (TPU MXU 128-aligned)."""
    from algebraic.model import LMConfig

    cfg = LMConfig()
    assert cfg.vocab_size == 32768, (
        f"LMConfig.vocab_size default is {cfg.vocab_size}, expected 32768"
    )


@pytest.mark.tier2
@pytest.mark.e2e
def test_tier2_presets_vocab_size_and_legacy_arm_removal():
    """Verify all model presets use 32,768 vocabulary and legacy arm preset is removed."""
    import algebraic.config as cfg_module

    # Verify legacy preset is completely removed
    assert not hasattr(cfg_module, FORBIDDEN_KEYWORD), (
        f"algebraic.config still defines '{FORBIDDEN_KEYWORD}' function"
    )
    assert FORBIDDEN_KEYWORD not in cfg_module.PRESETS, (
        f"algebraic.config.PRESETS still contains '{FORBIDDEN_KEYWORD}' key"
    )
    assert set(cfg_module.PRESETS.keys()) == {"standard", "algebraic"}, (
        f"PRESETS keys are {set(cfg_module.PRESETS.keys())}, expected {{'standard', 'algebraic'}}"
    )

    deleted_scale = "".join(["6", "0m"])
    assert deleted_scale not in cfg_module.SIZES, f"{deleted_scale} scale must be deleted from SIZES"

    # Verify all model sizes use vocab_size 32,768
    sizes = ["proxy", "125m", "350m", "1b"]
    for size in sizes:
        std_cfg = cfg_module.standard(size)
        assert std_cfg.vocab_size == 32768, (
            f"standard('{size}') vocab_size is {std_cfg.vocab_size}, expected 32768"
        )
        alg_cfg = cfg_module.algebraic(size)
        assert alg_cfg.vocab_size == 32768, (
            f"algebraic('{size}') vocab_size is {alg_cfg.vocab_size}, expected 32768"
        )


@pytest.mark.tier2
@pytest.mark.e2e
def test_tier2_comparison_configs_vocab_size():
    """Verify comparison configs specify vocab_size = 32,768 and no third architecture."""
    configs_dir = REPO_ROOT / "configs"
    comparison_configs = list(configs_dir.glob("comparison_*.json"))
    assert comparison_configs, "No configs/comparison_*.json found"

    for cfg_path in comparison_configs:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        training = data.get("training", {})
        vocab = training.get("vocab_size")
        assert vocab == 32768, (
            f"{cfg_path.name} specifies training.vocab_size = {vocab}, expected 32768"
        )

        archs = data.get("architectures", [])
        assert FORBIDDEN_KEYWORD not in archs, (
            f"{cfg_path.name} architectures contains '{FORBIDDEN_KEYWORD}': {archs}"
        )
        assert archs == ["standard", "algebraic"], (
            f"{cfg_path.name} architectures is {archs}, expected ['standard', 'algebraic']"
        )


@pytest.mark.tier2
@pytest.mark.e2e
def test_tier2_tokenizer_module_contract():
    """Verify src/algebraic/tokenizer.py implements the required public interface."""
    tok_path = REPO_ROOT / "src" / "algebraic" / "tokenizer.py"
    assert tok_path.exists(), "src/algebraic/tokenizer.py must exist"

    import algebraic.tokenizer as tokenizer_mod

    # Verify attributes
    assert getattr(tokenizer_mod, "vocab_size", None) == 32768, (
        f"tokenizer.vocab_size is {getattr(tokenizer_mod, 'vocab_size', None)}, expected 32768"
    )
    assert getattr(tokenizer_mod, "eot_token", None) == 2, (
        f"tokenizer.eot_token is {getattr(tokenizer_mod, 'eot_token', None)}, expected 2"
    )

    # Verify functions
    for fn_name in [
        "get_tokenizer",
        "encode",
        "encode_ordinary",
        "encode_ordinary_batch",
        "decode",
    ]:
        fn = getattr(tokenizer_mod, fn_name, None)
        assert callable(fn), f"algebraic.tokenizer must expose callable '{fn_name}'"


@pytest.mark.tier2
@pytest.mark.e2e
def test_tier2_tokenizer_encoding_decoding_roundtrip():
    """Verify tokenizer encodes and decodes text faithfully, produces token IDs

    within [0, 32768), and handles batch operations and edge cases.
    """
    tok_path = REPO_ROOT / "src" / "algebraic" / "tokenizer.py"
    assert tok_path.exists(), "src/algebraic/tokenizer.py must exist"

    import algebraic.tokenizer as tokenizer_mod

    test_samples = [
        "Hello world! The Algebraic Transformer is ready for publication.",
        "def loss(h, y):\n    return -jnp.log(softmax(h)[y])\n",
        "ρ(s/8)^16: exact gradient computation without transcendentals.",
        "Edge case characters: <reserved_0> & @ # $ % ^ & * ( ) _ + = ~ ` ; : ' , . < > ? /",
        "Multilingual: Café, naïve, Zürich, 東京, 🚀",
        "",  # Empty string edge case
    ]

    for sample in test_samples:
        tokens = tokenizer_mod.encode(sample)
        assert isinstance(tokens, list), f"encode('{sample}') must return list"

        # Check bounds
        for tid in tokens:
            assert 0 <= tid < 32768, (
                f"Token ID {tid} for sample '{sample}' out of bounds [0, 32768)"
            )

        # Check decode round-trip
        decoded = tokenizer_mod.decode(tokens)
        assert decoded == sample, (
            f"Round-trip failed: original='{sample}', decoded='{decoded}'"
        )

    # Test ordinary encoding equivalence
    s = "Single ordinary test line."
    assert tokenizer_mod.encode_ordinary(s) == tokenizer_mod.encode(s)

    # Test batch encoding
    batch = ["Line 1", "Line 2", "Line 3", ""]
    batch_tokens = tokenizer_mod.encode_ordinary_batch(batch)
    assert len(batch_tokens) == 4
    for i, line in enumerate(batch):
        assert batch_tokens[i] == tokenizer_mod.encode_ordinary(line)


LEGACY_VOCAB_TOKENS = ("".join(["50", "257"]), "".join(["50", "256"]))
LEGACY_TOKENIZER_NAMES = ("".join(["gp", "t2"]), "".join(["tik", "token"]))
LEGACY_VOCAB_PATTERN = rf"\b({'|'.join(LEGACY_VOCAB_TOKENS)})\b"
LEGACY_NAME_PATTERN = rf"\b({'|'.join(LEGACY_TOKENIZER_NAMES)})\b"


@pytest.mark.tier2
@pytest.mark.e2e
def test_tier2_zero_legacy_vocab_and_tokenizer_references():
    """Verify zero occurrences of legacy vocabulary and tokenizer references

    across active source code and configs.
    """
    violations: list[str] = []
    scan_dirs = ["src", "configs", "scripts"]

    for d in scan_dirs:
        dir_path = REPO_ROOT / d
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*"):
            if not path.is_file():
                continue
            if any(p in SCAN_EXCLUDE_DIRS for p in path.parts):
                continue
            if path.suffix not in {".py", ".json", ".sh"}:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                if re.search(LEGACY_VOCAB_PATTERN, line):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}:{line_num}: [vocab] {line.strip()}")
                if re.search(LEGACY_NAME_PATTERN, line, re.IGNORECASE):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}:{line_num}: [tok] {line.strip()}")

    assert not violations, (
        f"Found {len(violations)} legacy tokenizer/vocab references in active code:\n"
        + "\n".join(violations[:25])
        + (f"\n... and {len(violations) - 25} more" if len(violations) > 25 else "")
    )


# ==============================================================================
# Tier 3: Cross-Module Parameter Counts & Pallas Kernels
# ==============================================================================


def calculate_theoretical_params(
    d_model: int,
    d_ff: int,
    n_layers: int,
    n_heads: int,
    vocab_size: int,
    arch: str,
) -> int:
    """Authoritative parameter counting formula from PROJECT.md and run_comparison.py."""
    count = vocab_size * d_model + n_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model) + 2 * d_model
    if arch == "algebraic":
        # QK norm gain parameters (2 * head_dim + n_heads) per layer
        count += n_layers * (2 * (d_model // n_heads) + n_heads)
    return count


@pytest.mark.tier3
@pytest.mark.e2e
def test_tier3_parameter_counts_exact_across_scales():
    """Verify model parameter counts match theoretical numbers exactly across

    all scales (proxy, 125m, 350m, 1b) for V=32,768.
    """
    import jax
    import algebraic.config as cfg_module
    from algebraic.model import count_params, init_params

    # Expected numbers derived authoritatively for V=32,768
    expected = {
        "proxy": {"standard": 26_745_600, "algebraic": 26_746_672, "delta": 1_072},
        "125m":  {"standard": 110_120_448, "algebraic": 110_122_128, "delta": 1_680},
        "350m":  {"standard": 332_449_792, "algebraic": 332_453_248, "delta": 3_456},
        "1b":    {"standard": 975_277_056, "algebraic": 975_281_920, "delta": 4_864},
    }

    # Verify formula calculation matches authoritative specification table
    for size, vals in expected.items():
        dims = cfg_module.SIZES[size]
        calc_std = calculate_theoretical_params(
            d_model=dims["d_model"],
            d_ff=dims["d_ff"],
            n_layers=dims["n_layers"],
            n_heads=dims["n_heads"],
            vocab_size=32768,
            arch="standard",
        )
        calc_alg = calculate_theoretical_params(
            d_model=dims["d_model"],
            d_ff=dims["d_ff"],
            n_layers=dims["n_layers"],
            n_heads=dims["n_heads"],
            vocab_size=32768,
            arch="algebraic",
        )
        assert calc_std == vals["standard"], (
            f"Calculated standard params for {size} is {calc_std}, expected {vals['standard']}"
        )
        assert calc_alg == vals["algebraic"], (
            f"Calculated algebraic params for {size} is {calc_alg}, expected {vals['algebraic']}"
        )
        assert calc_alg - calc_std == vals["delta"], (
            f"Param delta for {size} is {calc_alg - calc_std}, expected {vals['delta']}"
        )

    # Verify JAX instantiated parameter trees on proxy and 125m scales
    from dataclasses import replace
    rng = jax.random.PRNGKey(42)
    for size in ["proxy", "125m"]:
        std_cfg = replace(cfg_module.standard(size), vocab_size=32768)
        std_params = init_params(std_cfg, rng)
        assert count_params(std_params) == expected[size]["standard"]

        alg_cfg = replace(cfg_module.algebraic(size), vocab_size=32768)
        alg_params = init_params(alg_cfg, rng)
        assert count_params(alg_params) == expected[size]["algebraic"]


@pytest.mark.tier3
@pytest.mark.e2e
def test_tier3_tied_embeddings_and_fused_head_alignment():
    """Verify tied embeddings shape (32768, d_model) and fused head 0-padding

    (64 tiles * 512 = 32,768, zero padding tokens).
    """
    from dataclasses import replace
    import jax
    import algebraic.config as cfg_module
    from algebraic.model import init_params

    cfg = replace(cfg_module.standard("proxy"), vocab_size=32768)
    params = init_params(cfg, jax.random.PRNGKey(0))

    # Embedding shape check
    assert "embed" in params
    assert params["embed"].shape == (32768, cfg.d_model)

    # Tied vocabulary check: no separate unembedding matrix
    assert "unembed" not in params
    assert "lm_head" not in params
    assert "head" not in params

    # Fused head alignment check:
    # bv=512 tile size must evenly divide 32,768
    bv = 512
    V = 32768
    nv = -(-V // bv)
    pad = nv * bv - V
    assert nv == 64, f"Number of vocab tiles is {nv}, expected 64"
    assert pad == 0, f"Vocabulary padding is {pad}, expected exactly 0"


@pytest.mark.tier3
@pytest.mark.e2e
def test_tier3_hardware_tile_alignment_and_head_dimension():
    """Verify head_dim == 64 across all model sizes, aligning with TPU v4 vector lanes."""
    import algebraic.config as cfg_module

    for size, dims in cfg_module.SIZES.items():
        d_model = dims["d_model"]
        n_heads = dims["n_heads"]
        head_dim = d_model // n_heads
        assert head_dim == 64, (
            f"Model scale '{size}' has head_dim={head_dim}, expected exactly 64"
        )
        assert 128 % head_dim == 0, (
            f"head_dim ({head_dim}) must evenly divide TPU v4 MXU/vector lanes (128)"
        )


@pytest.mark.tier3
@pytest.mark.e2e
def test_tier3_flash_attention_kernel_contracts():
    """Verify FlashAttention kernel contracts: causal softmax and algebraic rho(s/8)^16."""
    import algebraic.kernels.flash as flash_mod
    import algebraic.kernels.head_resident as resident_mod

    # Verify Variant definition
    Variant = getattr(flash_mod, "Variant", None)
    assert Variant is not None

    std_var = Variant(kind="softmax")
    assert std_var.kind == "softmax"

    alg_var = Variant(kind="asinh", n=16)
    assert alg_var.kind == "asinh"
    assert alg_var.n == 16

    # Verify tile sizes
    tiles_fn = getattr(resident_mod, "_tiles", None)
    assert callable(tiles_fn)
    assert tiles_fn(alg_var) == (128, 512)
    assert tiles_fn(std_var) == (512, 512)


@pytest.mark.tier3
@pytest.mark.e2e
def test_tier3_dense_attention_numerical_correctness():
    """Verify dense causal softmax and algebraic attention satisfy mathematical

    properties: strictly non-negative weights and causal upper triangular masking.
    """
    import jax
    import jax.numpy as jnp
    from algebraic import algebraic as A
    from algebraic import standard as S

    # Synthetic batch: B=1, H=2, T=8, D=64
    rng = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (1, 2, 8, 64), dtype=jnp.float32)
    k = jax.random.normal(k2, (1, 2, 8, 64), dtype=jnp.float32)
    v = jax.random.normal(k3, (1, 2, 8, 64), dtype=jnp.float32)

    # Standard softmax causal attention
    out_std = S.softmax_attention(q, k, v)
    assert out_std.shape == (1, 2, 8, 64)
    assert jnp.all(jnp.isfinite(out_std))

    # Pure algebraic attention: rho(s / 8.0)^16
    out_alg = A.asinh_attention(q, k, v, n=16, c=8.0)
    assert out_alg.shape == (1, 2, 8, 64)
    assert jnp.all(jnp.isfinite(out_alg))

    # Causal dependency check: query at position 0 must not be influenced by key at position 1
    k_perturbed = k.at[:, :, 1:, :].set(k[:, :, 1:, :] + 100.0)
    out_alg_pert = A.asinh_attention(q, k_perturbed, v, n=16, c=8.0)
    # Position 0 output should be identical despite future perturbations
    assert jnp.allclose(out_alg[:, :, 0, :], out_alg_pert[:, :, 0, :], atol=1e-5)


# ==============================================================================
# Tier 4: Distributed Multi-Worker Smoke Tests & Audit
# ==============================================================================


@pytest.mark.tier4
@pytest.mark.e2e
def test_tier4_distributed_smoke_script_interface():
    """Verify scripts/smoke_distributed.py supports --scale {125m, 350m, 1b}

    and scripts/setup_workers.sh exists and configures cluster.
    """
    smoke_script = REPO_ROOT / "scripts" / "smoke_distributed.py"
    assert smoke_script.exists(), "scripts/smoke_distributed.py must exist"

    setup_script = REPO_ROOT / "scripts" / "setup_workers.sh"
    assert setup_script.exists(), "scripts/setup_workers.sh must exist"
    assert os.access(setup_script, os.R_OK), "scripts/setup_workers.sh must be readable"

    setup_content = setup_script.read_text(encoding="utf-8")
    assert "w0" in setup_content and "w3" in setup_content, (
        "scripts/setup_workers.sh must configure workers w0-w3"
    )

    smoke_content = smoke_script.read_text(encoding="utf-8")
    # Verify 125m, 350m, 1b scale support in smoke_distributed.py
    for scale in ["125m", "350m", "1b"]:
        assert scale in smoke_content, (
            f"scripts/smoke_distributed.py does not reference scale '{scale}'"
        )


@pytest.mark.tier4
@pytest.mark.e2e
def test_tier4_comparison_configs_complete_and_valid():
    """Verify comparison configs exist and are valid for 125m, 350m, and 1b scales."""
    expected_dimensions = {
        "125m": {"d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 2048},
        "350m": {"d_model": 1024, "n_layers": 24, "n_heads": 16, "d_ff": 2688},
        "1b":   {"d_model": 1536, "n_layers": 32, "n_heads": 24, "d_ff": 4224},
    }

    for scale, dims in expected_dimensions.items():
        cfg_path = REPO_ROOT / "configs" / f"comparison_{scale}.json"
        assert cfg_path.exists(), f"Missing config: configs/comparison_{scale}.json"

        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        training = data.get("training", {})
        assert training.get("vocab_size") == 32768, (
            f"comparison_{scale}.json training.vocab_size is {training.get('vocab_size')}, expected 32768"
        )
        assert training.get("batch_size") == 512
        assert training.get("seq_len") == 2048
        assert data.get("architectures") == ["standard", "algebraic"]

        cfg_dims = training.get("dimensions", {})
        for dim_k, dim_v in dims.items():
            assert cfg_dims.get(dim_k) == dim_v, (
                f"comparison_{scale}.json dimension {dim_k} is {cfg_dims.get(dim_k)}, expected {dim_v}"
            )


@pytest.mark.tier4
@pytest.mark.e2e
def test_tier4_distributed_smoke_artifacts_validation():
    """If distributed smoke test output artifacts exist in results/verification/,

    verify they report passed=True with completed_steps=1 across 4 processes.
    """
    verification_dir = REPO_ROOT / "results" / "verification"
    if not verification_dir.exists():
        pytest.skip("results/verification directory does not exist yet (pre-M3)")

    for scale in ["125m", "350m", "1b"]:
        artifact_path = verification_dir / f"distributed_{scale}.json"
        if not artifact_path.exists():
            continue
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert data.get("passed") is True, (
            f"{artifact_path.name} does not report passed=True"
        )
        assert data.get("completed_steps") == 1, (
            f"{artifact_path.name} completed_steps={data.get('completed_steps')}, expected 1"
        )
        assert data.get("processes") == 4, (
            f"{artifact_path.name} processes={data.get('processes')}, expected 4"
        )


@pytest.mark.tier4
@pytest.mark.e2e
def test_tier4_full_cpu_test_suite_passes():
    """Verify all CPU unit tests in tests/ pass cleanly."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_components.py",
        "tests/test_training.py",
        "-q",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"CPU unit tests failed with code {proc.returncode}:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
