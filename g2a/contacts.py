"""Interpret native tool-filtered contact points without importing the simulator."""

import math

import torch

from .math import quat_apply, quat_conjugate


def empty_opposition_geometry(count: int, reference: torch.Tensor) -> dict:
    return {
        "opposition_contact_centers_object_m": reference.new_zeros((count, 4, 3)),
        "opposition_contact_center_forces_n": reference.new_zeros((count, 4)),
        "opposition_contact_point_counts": torch.zeros(
            (count, 4), dtype=torch.int64, device=reference.device
        ),
        "opposition_contact_center_valid": torch.zeros(
            (count, 4), dtype=torch.bool, device=reference.device
        ),
        "opposition_contact_geometry_complete": torch.zeros(
            count, dtype=torch.bool, device=reference.device
        ),
    }


def opposition_contact_geometry(
    contact_data,
    force_matrix: torch.Tensor,
    object_pose_world: torch.Tensor,
    body_names,
    body_groups,
    minimum_contact_force_n: float,
) -> tuple[dict, dict]:
    """Aggregate actual SDK points, retaining their body and semantic group IDs.

    ``get_contact_data(physics_dt)`` returns signed normal forces, world points,
    world normals, separations, counts and starts. Only referenced buffer slots
    are initialized. Normal-force magnitudes weight each group centroid. The
    native aggregate must equal the detailed signed force/normal sum within
    floating-point summation error. No tactile point is inferred from link FK.
    """
    if (
        not isinstance(contact_data, (tuple, list))
        or len(contact_data) != 6
        or not all(isinstance(value, torch.Tensor) for value in contact_data)
        or not isinstance(force_matrix, torch.Tensor)
        or not isinstance(object_pose_world, torch.Tensor)
    ):
        raise RuntimeError("Native contact data must contain six tensor buffers")
    forces, points, normals, separations, counts, starts = contact_data
    count, bodies = len(object_pose_world), len(body_names)
    capacity = len(forces)
    floating = (forces, points, normals, separations, force_matrix, object_pose_world)
    if (
        capacity < 1
        or forces.shape != (capacity, 1)
        or points.shape != (capacity, 3)
        or normals.shape != (capacity, 3)
        or separations.shape != (capacity, 1)
        or counts.shape != (count, bodies)
        or starts.shape != counts.shape
        or force_matrix.shape != (count, bodies, 3)
        or object_pose_world.shape != (count, 7)
        or any(not value.is_floating_point() or value.dtype != forces.dtype for value in floating)
        or any(value.device != forces.device for value in (*floating, counts, starts))
        or counts.dtype not in (torch.uint32, torch.int32, torch.int64)
        or starts.dtype not in (torch.uint32, torch.int32, torch.int64)
    ):
        raise RuntimeError("Native detailed contact buffer shapes, dtypes or devices do not match")
    if (
        len(body_names) != len(set(body_names))
        or len(body_groups) != 4
        or any(not group for group in body_groups)
        or len({name for group in body_groups for name in group}) != sum(map(len, body_groups))
        or any(name not in body_names for group in body_groups for name in group)
        or not math.isfinite(minimum_contact_force_n)
        or minimum_contact_force_n <= 0
    ):
        raise ValueError("Contact geometry requires four distinct declared native body groups")

    result = empty_opposition_geometry(count, object_pose_world)
    counts, starts = counts.to(torch.int64), starts.to(torch.int64)
    reported = counts.clamp_min(0).sum(-1)
    # The SDK exposes no overflow flag. A full buffer is conservatively
    # incomplete even when it might happen to fit exactly.
    saturated = bool(reported.sum() >= capacity)
    result["opposition_contact_geometry_reported_points"] = reported
    result["opposition_contact_geometry_capacity"] = torch.full_like(reported, capacity)
    result["opposition_contact_geometry_capacity_saturated"] = torch.full(
        (count,), saturated, dtype=torch.bool, device=forces.device
    )
    force_consistent = torch.zeros(count, dtype=torch.bool, device=forces.device)
    result["opposition_contact_geometry_force_consistent"] = force_consistent
    raw = {}
    if saturated:
        return result, raw

    active = counts > 0
    range_valid = (counts >= 0) & (~active | ((starts >= 0) & (starts + counts <= capacity)))
    safe_counts = counts.masked_fill(~range_valid, 0).flatten()
    pairs = torch.repeat_interleave(torch.arange(count * bodies, device=forces.device), safe_counts)
    block_starts = safe_counts.cumsum(0) - safe_counts
    offsets = torch.arange(len(pairs), device=forces.device) - torch.repeat_interleave(
        block_starts, safe_counts
    )
    indices = starts.flatten()[pairs] + offsets
    env_ids, body_ids = pairs // bodies, pairs % bodies
    bad_pair = ~range_valid.flatten()
    if len(indices):
        ordered, order = indices.sort()
        duplicate = ordered[1:] == ordered[:-1]
        bad_pair[pairs[order[1:][duplicate]]] = True
        bad_pair[pairs[order[:-1][duplicate]]] = True

    f, p, normal, separation = (value[indices] for value in (forces, points, normals, separations))
    eps = torch.finfo(forces.dtype).eps
    finite = torch.cat((f, p, normal, separation), -1).isfinite().all(-1)
    normal_length = normal.norm(dim=-1)
    finite &= (f[:, 0] == 0) | ((normal_length - 1).abs() <= 128 * eps)
    bad_pair[pairs[~finite]] = True
    pose_valid = object_pose_world.isfinite().all(-1) & (
        (object_pose_world[:, 3:].norm(dim=-1) - 1).abs() <= 128 * eps
    )
    quaternion = quat_conjugate(object_pose_world[env_ids, 3:])
    local_points = quat_apply(quaternion, p - object_pose_world[env_ids, :3])
    local_normals = quat_apply(quaternion, normal)
    weights = torch.nan_to_num(f[:, 0].abs()).masked_fill(~finite, 0)
    weighted = torch.nan_to_num(local_points) * weights[:, None]
    pair_forces = forces.new_zeros((count * bodies, 3))
    pair_forces.index_add_(0, pairs, torch.nan_to_num(f * normal))
    pair_weights = forces.new_zeros(count * bodies)
    pair_weights.index_add_(0, pairs, weights)
    error = (pair_forces.view(count, bodies, 3) - force_matrix).norm(dim=-1)
    roundoff = (
        8 * eps * (counts + 2) * (pair_weights.view(count, bodies) + force_matrix.norm(dim=-1))
    )
    force_consistent[:] = force_matrix.isfinite().all(dim=(-1, -2)) & (error <= roundoff).all(-1)
    complete = ~bad_pair.view(count, bodies).any(-1) & pose_valid & force_consistent

    group_for_body = torch.full((bodies,), -1, dtype=torch.int64, device=forces.device)
    for group_id, names in enumerate(body_groups):
        group_for_body[[body_names.index(name) for name in names]] = group_id
    group_ids = group_for_body[body_ids]
    selected = group_ids >= 0
    destinations = env_ids[selected] * 4 + group_ids[selected]
    group_forces = forces.new_zeros(count * 4)
    group_points = forces.new_zeros((count * 4, 3))
    group_counts = torch.zeros(count * 4, dtype=torch.int64, device=forces.device)
    group_forces.index_add_(0, destinations, weights[selected])
    group_points.index_add_(0, destinations, weighted[selected])
    group_counts.index_add_(0, destinations, torch.ones_like(destinations))
    group_forces = group_forces.view(count, 4)
    group_counts = group_counts.view(count, 4)
    valid = complete[:, None] & (group_counts > 0) & (group_forces >= minimum_contact_force_n)
    centers = (
        group_points.view(count, 4, 3)
        / group_forces.clamp_min(torch.finfo(forces.dtype).tiny)[:, :, None]
    )
    result.update(
        opposition_contact_centers_object_m=centers.masked_fill(~valid[:, :, None], 0),
        opposition_contact_center_forces_n=group_forces.masked_fill(~complete[:, None], 0),
        opposition_contact_point_counts=group_counts.masked_fill(~complete[:, None], 0),
        opposition_contact_center_valid=valid,
        opposition_contact_geometry_complete=complete,
    )
    raw.update(
        env_ids=env_ids.clone(),
        body_ids=body_ids.clone(),
        group_ids=group_ids,
        points_object_m=local_points,
        normals_object=local_normals,
        signed_normal_forces_n=f[:, 0].clone(),
        separations_m=separation[:, 0].clone(),
        finite=finite & pose_valid[env_ids],
    )
    return result, raw
