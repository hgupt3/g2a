"""Accumulate native collision observations without losing repeated trials."""

import math

import torch

from .artifacts import COLLISION_AUDIT_PHASES


def collision_margin_cost(
    tool_depth: torch.Tensor,
    self_depth: torch.Tensor,
    tool_budget_m: float,
    self_budget_m: float,
    soft_fraction: float,
) -> torch.Tensor:
    """Quadratic soft-margin cost from already validated native step peaks.

    A channel costs zero below the soft fraction and one at its acceptance
    budget. Values beyond that budget remain uncapped. This does not resample
    geometry, alter feasibility, or replace the episode's sticky audit.
    """
    if (
        not isinstance(tool_depth, torch.Tensor)
        or not isinstance(self_depth, torch.Tensor)
        or tool_depth.ndim != 1
        or self_depth.shape != tool_depth.shape
        or not tool_depth.is_floating_point()
        or not self_depth.is_floating_point()
    ):
        raise ValueError("Collision margin cost requires per-row floating-point native peaks")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
        for value in (tool_budget_m, self_budget_m)
    ) or (
        isinstance(soft_fraction, bool)
        or not isinstance(soft_fraction, (float, int))
        or not math.isfinite(soft_fraction)
        or not 0 <= soft_fraction < 1
    ):
        raise ValueError(
            "Collision margin cost requires positive budgets and a soft fraction in [0, 1)"
        )
    return (
        (tool_depth / tool_budget_m - soft_fraction).clamp_min(0) / (1 - soft_fraction)
    ).square() + (
        (self_depth / self_budget_m - soft_fraction).clamp_min(0) / (1 - soft_fraction)
    ).square()


class CollisionAudit:
    def __init__(self, count: int, device: str):
        self.tool = torch.zeros(count, device=device)
        self.self_overlap = torch.zeros_like(self.tool)
        self.samples = torch.zeros(count, dtype=torch.int64, device=device)
        self.phases = torch.zeros(
            (count, len(COLLISION_AUDIT_PHASES)), dtype=torch.int64, device=device
        )
        self.phase_tool = torch.zeros(self.phases.shape, device=device)
        self.phase_self = torch.zeros_like(self.phase_tool)
        self.feasible = torch.ones(count, dtype=torch.bool, device=device)
        # Allocate and update these only for an opted-in RL reward. Episode
        # certificates and default audit work remain independent of shaping.
        self.control_tool = None
        self.control_self_overlap = None

    def enable_control_step_tracking(self) -> None:
        if self.control_tool is None:
            self.control_tool = torch.zeros_like(self.tool)
            self.control_self_overlap = torch.zeros_like(self.self_overlap)

    def begin_control_step(self) -> None:
        """Clear just the current control interval, preserving episode evidence."""
        if self.control_tool is not None:
            self.control_tool.zero_()
            self.control_self_overlap.zero_()

    def update(self, metrics: dict, indices: torch.Tensor, phase: str) -> None:
        if phase not in COLLISION_AUDIT_PHASES:
            raise ValueError(f"Unknown collision-audit phase: {phase}")
        count = len(indices)
        tool = metrics["tool_hand_penetration_m"]
        self_depth = metrics["nonadjacent_self_penetration_m"]
        safe = metrics["feasible"]
        if (
            indices.ndim != 1
            or indices.dtype != torch.int64
            or tool.shape != (count,)
            or self_depth.shape != (count,)
            or safe.shape != (count,)
            or safe.dtype != torch.bool
        ):
            raise ValueError("Collision audit requires finite nonnegative per-row observations")
        if count == 0:
            return
        # Combine device-side checks before the one host decision. Separate
        # Python ``or`` predicates synchronize the GPU for every condition at
        # every substep, despite examining the same small observation batch.
        invalid = (
            ((indices < 0) | (indices >= len(self.samples))).any()
            | (~torch.isfinite(tool) | (tool < 0)).any()
            | (~torch.isfinite(self_depth) | (self_depth < 0)).any()
        )
        if invalid:
            raise ValueError("Collision audit requires finite nonnegative per-row observations")
        self.tool.scatter_reduce_(0, indices, tool, reduce="amax", include_self=True)
        self.self_overlap.scatter_reduce_(0, indices, self_depth, reduce="amax", include_self=True)
        if self.control_tool is not None:
            self.control_tool.scatter_reduce_(0, indices, tool, reduce="amax", include_self=True)
            self.control_self_overlap.scatter_reduce_(
                0, indices, self_depth, reduce="amax", include_self=True
            )
        ones = torch.ones_like(indices)
        self.samples.scatter_add_(0, indices, ones)
        phase_id = COLLISION_AUDIT_PHASES.index(phase)
        self.phases[:, phase_id].scatter_add_(0, indices, ones)
        self.phase_tool[:, phase_id].scatter_reduce_(
            0, indices, tool, reduce="amax", include_self=True
        )
        self.phase_self[:, phase_id].scatter_reduce_(
            0, indices, self_depth, reduce="amax", include_self=True
        )
        violations = torch.zeros_like(self.samples)
        violations.scatter_add_(0, indices, (~safe).long())
        self.feasible &= violations == 0

    def reset(self, indices: torch.Tensor) -> None:
        for value in (
            self.tool,
            self.self_overlap,
            self.samples,
            self.phases,
            self.phase_tool,
            self.phase_self,
        ):
            value[indices] = 0
        self.feasible[indices] = True
        if self.control_tool is not None:
            self.control_tool[indices] = 0
            self.control_self_overlap[indices] = 0

    def tensors(self) -> dict:
        return {
            "collision_tool_hand_max_penetration_m": self.tool.clone(),
            "collision_nonadjacent_self_max_penetration_m": self.self_overlap.clone(),
            "collision_audit_samples": self.samples.clone(),
            "collision_phase_samples": self.phases.clone(),
            "collision_tool_hand_phase_max_penetration_m": self.phase_tool.clone(),
            "collision_nonadjacent_self_phase_max_penetration_m": self.phase_self.clone(),
            "collision_feasible": self.feasible.clone(),
        }
