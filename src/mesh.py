"""SPMD distributed sharding topology for Google Cloud TPU v4 Pod slice.

Defines the physical 3D Torus mesh across the 16 TPU v4 chips (Cloud TPU v4-32)
with axes ('data', 'fsdp', 'model') and NamedSharding specifications.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_tpu_mesh(
    devices: Optional[Sequence[jax.Device]] = None,
    mesh_shape: Tuple[int, int, int] = (2, 2, 4),
    axis_names: Tuple[str, str, str] = ("data", "fsdp", "model"),
) -> Mesh:
    """Creates a 3D Torus JAX distributed mesh for Cloud TPU v4-32.

    Args:
        devices: Sequence of JAX devices. If None, uses jax.devices().
        mesh_shape: Shape of the 3D mesh (data, fsdp, model). Defaults to (2, 2, 4)
            matching the 16 TPU v4 chips of a v4-32 slice.
        axis_names: Logical axis names for sharding. Defaults to ('data', 'fsdp', 'model').

    Returns:
        A jax.sharding.Mesh over the physical accelerator devices.
    """
    if devices is None:
        devices = jax.devices()

    num_devices = len(devices)
    total_mesh_size = mesh_shape[0] * mesh_shape[1] * mesh_shape[2]

    if num_devices == total_mesh_size:
        actual_shape = mesh_shape
    elif num_devices >= 16 and total_mesh_size == 16:
        actual_shape = mesh_shape
        devices = devices[:16]
    else:
        # Graceful fallback for single-device or partial test environments (e.g. CPU)
        if num_devices == 1:
            actual_shape = (1, 1, 1)
        elif num_devices % 4 == 0:
            actual_shape = (num_devices // 4, 2, 2)
        elif num_devices % 2 == 0:
            actual_shape = (num_devices // 2, 2, 1)
        else:
            actual_shape = (num_devices, 1, 1)

    device_mesh = mesh_utils.create_device_mesh(
        actual_shape, devices, allow_split_physical_axes=True
    )
    return Mesh(device_mesh, axis_names)


def get_sharding(
    mesh: Mesh,
    spec: P,
) -> NamedSharding:
    """Returns a NamedSharding for a given partition specification on the mesh."""
    return NamedSharding(mesh, spec)


class ModelSharding:
    """Container holding canonical NamedSharding specs for distributed training."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        # Replicated weights across all devices
        self.replicated = NamedSharding(mesh, P())
        # Data-parallel activation sharding across data axis
        self.data_parallel = NamedSharding(mesh, P("data"))
        # FSDP data-parallel sharding across (data, fsdp)
        self.fsdp_data = NamedSharding(mesh, P(("data", "fsdp")))
        # Full 16-way data-parallel sharding
        self.full_data_parallel = NamedSharding(mesh, P(("data", "fsdp", "model")))
        # Sequence parallel sharding
        self.seq_parallel = NamedSharding(mesh, P(None, "model"))


def compile_data_parallel_step(
    step_fn: Callable[..., Any],
    mesh: Mesh,
    data_spec: P,
) -> Callable[..., Any]:
    """Compile a replicated-state training step inside an explicit shard map.

    Mosaic/Pallas kernels cannot be automatically partitioned by a surrounding
    SPMD ``jit``.  The shard-map boundary gives each device its local batch while
    keeping parameters, optimizer state, and scalar metrics replicated.  The
    step function must average its gradients over ``mesh.axis_names`` before it
    applies optimizer updates.
    """
    from jax.experimental.shard_map import shard_map

    replicated_spec = P()
    replicated = NamedSharding(mesh, replicated_spec)
    data = NamedSharding(mesh, data_spec)
    mapped_step = shard_map(
        step_fn,
        mesh=mesh,
        in_specs=(replicated_spec, replicated_spec, data_spec, data_spec),
        out_specs=(replicated_spec, replicated_spec, replicated_spec),
        # JAX requires this checker to be disabled when the mapped function
        # contains a Pallas kernel. Replication is established explicitly by
        # the gradient pmean in each training-step factory.
        check_rep=False,
    )
    return jax.jit(
        mapped_step,
        in_shardings=(replicated, replicated, data, data),
        out_shardings=(replicated, replicated, replicated),
    )
