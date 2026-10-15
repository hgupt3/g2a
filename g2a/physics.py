"""Simulator allocation helpers that do not require importing Isaac Sim."""

import torch

from .math import quat_apply


def root_link_velocity(
    root_com_velocity: torch.Tensor,
    root_link_quaternion: torch.Tensor,
    center_of_mass_in_link: torch.Tensor,
) -> torch.Tensor:
    """Recover world-frame link velocity from the current public COM state.

    A reset updates the SDK's COM velocity immediately but can leave its
    separate link-velocity cache at the previous episode's value until the next
    physics step. Computing the rigid-body offset relation avoids that cache.
    Angular velocity is identical at the COM and link origin.
    """
    offset_world = quat_apply(root_link_quaternion, center_of_mass_in_link)
    angular = root_com_velocity[..., 3:]
    linear = root_com_velocity[..., :3] - torch.linalg.cross(angular, offset_world, dim=-1)
    return torch.cat((linear, angular), dim=-1)


def gpu_rigid_patch_capacity(num_envs: int, patches_per_env: int, existing_capacity: int) -> int:
    """Allocate enough contact patches without reducing the SDK's existing budget.

    PhysX drops contacts when this allocation overflows. Scaling the buffer with
    the number of complete hand/tool scenes preserves their intended dynamics;
    the allocation itself does not alter forces, materials or solver settings.
    """
    for name, value in (
        ("num_envs", num_envs),
        ("patches_per_env", patches_per_env),
        ("existing_capacity", existing_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return max(existing_capacity, num_envs * patches_per_env)
