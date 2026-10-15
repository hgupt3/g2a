"""Batched, scalar-first quaternion and pose operations, independent of Isaac Sim."""

import torch


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        (
            aw * bw - (av * bv).sum(-1, keepdim=True),
            aw * bv + bw * av + torch.cross(av, bv, dim=-1),
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    t = 2 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def rotvec_to_quat(v: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle * 0.5), v * (0.5 * torch.sinc(angle / (2 * torch.pi)))), -1)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    length = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, q[..., :1].clamp_min(0))
    scale = torch.where(length > 1e-7, angle / length.clamp_min(1e-12), 2.0)
    return q[..., 1:] * scale


def quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(quat_to_rotvec(quat_mul(a, quat_conjugate(b))), dim=-1)


def rotation_6d(q: torch.Tensor) -> torch.Tensor:
    """First two rotation-matrix columns, avoiding quaternion sign discontinuities."""
    x_axis = torch.zeros_like(q[..., 1:])
    y_axis = torch.zeros_like(x_axis)
    x_axis[..., 0] = 1
    y_axis[..., 1] = 1
    return torch.cat((quat_apply(q, x_axis), quat_apply(q, y_axis)), -1)


def compose_pose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (a[..., :3] + quat_apply(a[..., 3:], b[..., :3]), quat_mul(a[..., 3:], b[..., 3:])), -1
    )


def relative_pose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inverse = quat_conjugate(a[..., 3:])
    return torch.cat(
        (quat_apply(inverse, b[..., :3] - a[..., :3]), quat_mul(inverse, b[..., 3:])), -1
    )


def pose_features(pose: torch.Tensor) -> torch.Tensor:
    return torch.cat((pose[..., :3], rotation_6d(pose[..., 3:])), -1)


def clamp_norm(v: torch.Tensor, maximum: float) -> torch.Tensor:
    if maximum < 0:
        raise ValueError("Norm limit cannot be negative")
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return v * (maximum / norm.clamp_min(max(maximum, 1e-12)))
