"""Hardware-fused kernels for algebraic intelligence."""

from src.kernels.pallas_afa import (
    afa_kernel,
    pallas_afa_forward,
    tiled_afa_forward,
    tiled_afa_backward,
    exact_afa_reference,
    exact_afa_with_denominator,
    pallas_afa,
    distributed_ring_afa,
    sharded_pallas_afa,
    algebraic_flash_attention,
)
from src.kernels.pallas_oace import (
    fused_linear_oace_forward,
    fused_linear_oace_backward,
    fused_linear_oace,
)
from src.kernels.linear_afa import (
    OCTIC_POLYNOMIAL_COEFFICIENTS,
    LinearAFAState,
    compute_algebraic_feature_map,
    linear_afa_init_state,
    linear_afa_step,
    linear_afa_parallel_scan,
)
from src.kernels.pallas_flash_attention import (
    tiled_flash_attention_forward,
    tiled_flash_attention_backward,
    standard_flash_attention,
)
from src.kernels.fused_cross_entropy import (
    standard_fused_linear_ce_forward,
    standard_fused_linear_ce_backward,
    standard_fused_cross_entropy,
)

# Triton is an optional CUDA dependency.  TPU and CPU installations must still
# be able to import the portable JAX/Pallas kernels through this package.
try:
    from src.kernels.triton_afa import (
        _triton_afa_fwd_kernel,
        _triton_afa_bwd_kernel,
        triton_algebraic_flash_attention,
    )
    from src.kernels.triton_oace import (
        _triton_linear_oace_fwd_kernel,
        triton_fused_linear_oace,
    )
    from src.kernels.triton_flash_attention import (
        _triton_std_flash_attn_fwd_kernel,
        triton_standard_flash_attention,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _TRITON_EXPORTS: list[str] = []
else:
    _TRITON_EXPORTS = [
        "_triton_afa_fwd_kernel",
        "_triton_afa_bwd_kernel",
        "triton_algebraic_flash_attention",
        "_triton_linear_oace_fwd_kernel",
        "triton_fused_linear_oace",
        "_triton_std_flash_attn_fwd_kernel",
        "triton_standard_flash_attention",
    ]

__all__ = [
    "afa_kernel",
    "pallas_afa_forward",
    "tiled_afa_forward",
    "tiled_afa_backward",
    "exact_afa_reference",
    "exact_afa_with_denominator",
    "pallas_afa",
    "distributed_ring_afa",
    "sharded_pallas_afa",
    "algebraic_flash_attention",
    "fused_linear_oace_forward",
    "fused_linear_oace_backward",
    "fused_linear_oace",
    "OCTIC_POLYNOMIAL_COEFFICIENTS",
    "LinearAFAState",
    "compute_algebraic_feature_map",
    "linear_afa_init_state",
    "linear_afa_step",
    "linear_afa_parallel_scan",
    "tiled_flash_attention_forward",
    "tiled_flash_attention_backward",
    "standard_flash_attention",
    "standard_fused_linear_ce_forward",
    "standard_fused_linear_ce_backward",
    "standard_fused_cross_entropy",
] + _TRITON_EXPORTS
