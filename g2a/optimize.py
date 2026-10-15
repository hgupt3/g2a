"""Finite area grasp search with independent wrench tests and task verification."""

import math
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from .artifacts import (
    COLLISION_AUDIT_METHOD,
    COLLISION_AUDIT_PHASES,
    COLLISION_AUDIT_VERSION,
    COLLISION_SAMPLING_PROTOCOL,
    GRASP_SCHEMA_VERSION,
    atomic_torch_save,
    candidate_state_hashes,
    file_hash,
    load_grasp,
    load_warm_proposals,
    validate_collision_audit,
    write_json,
)
from .config import CollisionSpec, SearchSpec
from .control import JOINT_NAMES
from .stress import PARAMETER_FIELDS, SEARCH_DIM, WRENCH_DIRECTIONS, in_hand_errors


class CrossEntropySearch:
    def __init__(
        self,
        dimensions: int,
        population: int,
        device: str,
        seed: int,
        spec: SearchSpec | None = None,
    ):
        if population < 4:
            raise ValueError("CEM requires at least four candidates")
        if dimensions < 1:
            raise ValueError("CEM requires at least one search dimension")
        spec = SearchSpec() if spec is None else spec
        self.population = population
        self.spec = spec
        self.mean = torch.zeros(dimensions, device=device)
        self.std = torch.full_like(self.mean, spec.initial_std)
        self.best = self.mean.clone()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self._uniform = True

    def sample(self) -> torch.Tensor:
        shape = (self.population, len(self.mean))
        if self._uniform:
            candidates = (
                2 * torch.rand(shape, device=self.mean.device, generator=self.generator) - 1
            )
            self._uniform = False
        else:
            candidates = self.mean + self.std * torch.randn(
                shape, device=self.mean.device, generator=self.generator
            )
        candidates.clamp_(-1, 1)
        candidates[0] = self.best
        candidates[1] = self.mean
        return candidates

    def update(self, candidates: torch.Tensor, scores: torch.Tensor) -> None:
        if candidates.shape != (self.population, len(self.mean)) or scores.shape != (
            self.population,
        ):
            raise ValueError("CEM update dimensions do not match its population")
        if not torch.isfinite(scores).all() or not torch.isfinite(candidates).all():
            raise ValueError("Optimizer received non-finite candidates or scores")
        # A completely failed population carries no evidence for narrowing the
        # region. Resample uniformly instead of converging on arbitrary ties.
        if scores.max() - scores.min() <= torch.finfo(scores.dtype).eps:
            self._uniform = True
            return
        elite_count = max(2, math.ceil(self.population * self.spec.elite_fraction))
        elite_count = min(elite_count, self.population)
        ids = scores.topk(elite_count).indices
        elites = candidates[ids]
        self.mean.lerp_(elites.mean(dim=0), self.spec.smoothing)
        self.std.lerp_(
            elites.std(dim=0, unbiased=False).clamp_min(self.spec.min_std), self.spec.smoothing
        )
        self.best = candidates[ids[0]].clone()


class IslandCrossEntropySearch:
    """Independent CEM distributions for a union of disconnected area patches.

    The area decoder packs patch selection and local x into parameter zero.
    Each island instead optimizes a *local* x in [-1, 1] and only encodes the
    patch interval when returning proposals. Even elite poses from opposite
    approaches never contribute to the same mean or standard deviation.
    """

    def __init__(
        self,
        dimensions: int,
        population: int,
        patch_count: int,
        device: str,
        seed: int,
        spec: SearchSpec | None = None,
    ):
        self.spec = SearchSpec() if spec is None else spec
        if dimensions < 1 or patch_count < 1:
            raise ValueError("Island CEM requires positive dimension and patch counts")
        if self.spec.minimum_candidates_per_patch < 4:
            raise ValueError("Island CEM requires at least four candidates per patch")
        if not 0 < self.spec.uniform_exploration_fraction <= 0.5:
            raise ValueError("Island CEM uniform exploration fraction must be in (0, 0.5]")
        required = patch_count * self.spec.minimum_candidates_per_patch
        if population < required:
            raise ValueError(f"Island CEM requires at least {required} candidates")
        self.population = population
        self.patch_count = patch_count
        self.means = torch.zeros((patch_count, dimensions), device=device)
        self.stds = torch.full_like(self.means, self.spec.initial_std)
        self.best_local = self.means.clone()
        self.best_scores = torch.full((patch_count,), -torch.inf, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self._uniform = torch.ones(patch_count, dtype=torch.bool, device=device)
        # Balanced, deterministic allocation keeps good modes alive and keeps
        # each generation's simulator cost independent of its preceding scores.
        counts = torch.full((patch_count,), population // patch_count, device=device)
        counts[: population % patch_count] += 1
        self.counts = counts.long()
        self.patch_ids = torch.repeat_interleave(
            torch.arange(patch_count, device=device), self.counts
        )
        self.starts = self.counts.cumsum(0) - self.counts
        self.slots = torch.arange(population, device=device) - self.starts[self.patch_ids]
        self.exploration_counts = (
            (self.counts * self.spec.uniform_exploration_fraction).ceil().long().clamp_min(1)
        )
        self._warm_candidates = {}
        self.warm_patch_counts = [0] * patch_count

    def decode(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover exactly the patch selector used by GraspArea.decode_wrist."""
        if (
            parameters.ndim != 2
            or parameters.shape[1] != self.means.shape[1]
            or not parameters.is_floating_point()
            or not torch.isfinite(parameters).all()
            or (parameters.abs() > 1).any()
        ):
            raise ValueError("Island CEM parameters must be finite normalized candidate rows")
        coordinate = (parameters[:, 0] + 1) * 0.5 * self.patch_count
        patch_ids = coordinate.floor().long().clamp_max(self.patch_count - 1)
        local = parameters.clone()
        local[:, 0] = 2 * (coordinate - patch_ids) - 1
        return patch_ids, local

    def encode(self, local: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        parameters = local.clone().clamp_(-1, 1)
        # At a patch's right edge, floating-point rounding could select the next
        # patch instead. Keep both endpoints a few representable values inside
        # the interval; the margin is far below any geometric search tolerance.
        margin = 4 * torch.finfo(parameters.dtype).eps * self.patch_count
        local_x = parameters[:, 0].clamp(-1 + margin, 1 - margin)
        parameters[:, 0] = 2 * (patch_ids + (local_x + 1) * 0.5) / self.patch_count - 1
        return parameters

    @property
    def best_parameters(self) -> torch.Tensor:
        return self.encode(
            self.best_local, torch.arange(self.patch_count, device=self.means.device)
        )

    @torch.no_grad()
    def warm_start(self, parameters: torch.Tensor, scores: torch.Tensor | None = None) -> None:
        """Replay warm proposals in their own islands, without inheriting scores.

        Recorded SG only prioritizes which proposals to reclose first. Every
        other island continues exploring, and the new simulator must measure
        every proposal before it can become a best-scoring grasp in this run.
        """
        patch_ids, local = self.decode(parameters)
        if scores is not None and (
            scores.shape != (len(parameters),) or not torch.isfinite(scores).all()
        ):
            raise ValueError("Warm-start scores must be finite per-candidate values")
        for patch in range(self.patch_count):
            ids = (patch_ids == patch).nonzero(as_tuple=False).flatten()
            if len(ids) == 0:
                continue
            if scores is not None:
                ids = ids[torch.argsort(scores[ids], descending=True, stable=True)]
            proposals = local[ids].to(self.means)
            self.means[patch] = proposals[0]
            self.best_local[patch] = proposals[0]
            self.stds[patch].fill_(self.spec.warm_std)
            self._uniform[patch] = False
            self._warm_candidates[patch] = proposals.clone()
            self.warm_patch_counts[patch] = len(proposals)

    @torch.no_grad()
    def sample(self) -> torch.Tensor:
        shape = (self.population, self.means.shape[1])
        uniform = 2 * torch.rand(shape, device=self.means.device, generator=self.generator) - 1
        gaussian = self.means[self.patch_ids] + self.stds[self.patch_ids] * torch.randn(
            shape, device=self.means.device, generator=self.generator
        )
        exploring = self.slots >= (self.counts - self.exploration_counts)[self.patch_ids]
        uniform_rows = self._uniform[self.patch_ids] | exploring
        local = torch.where(uniform_rows[:, None], uniform, gaussian).clamp_(-1, 1)
        local[self.starts] = self.best_local
        # Before there is any evidence, the zero local pose gets one slot and
        # all remaining slots explore. Do not waste a second slot duplicating it.
        distinct = (self.means - self.best_local).abs().amax(-1) > torch.finfo(local.dtype).eps
        mean_rows = self.starts[distinct & ~self._uniform] + 1
        local[mean_rows] = self.means[distinct & ~self._uniform]
        for patch, proposals in self._warm_candidates.items():
            count = min(len(proposals), int(self.counts[patch] - self.exploration_counts[patch]))
            start = int(self.starts[patch])
            local[start : start + count] = proposals[:count]
        self._warm_candidates.clear()
        return self.encode(local, self.patch_ids)

    @torch.no_grad()
    def update(self, candidates: torch.Tensor, scores: torch.Tensor) -> None:
        if candidates.shape != (self.population, self.means.shape[1]) or scores.shape != (
            self.population,
        ):
            raise ValueError("Island CEM update dimensions do not match its population")
        if not torch.isfinite(scores).all():
            raise ValueError("Optimizer received non-finite scores")
        patch_ids, local = self.decode(candidates)
        if not torch.equal(torch.bincount(patch_ids, minlength=self.patch_count), self.counts):
            raise ValueError("Island CEM proposals must preserve the per-patch allocation")
        for patch in range(self.patch_count):
            ids = (patch_ids == patch).nonzero(as_tuple=False).flatten()
            patch_scores = scores[ids]
            value, best_id = patch_scores.max(0)
            if value > self.best_scores[patch]:
                self.best_scores[patch] = value
                self.best_local[patch] = local[ids[best_id]]
            # Failed or tied islands restart broadly, while successful islands
            # keep their own evidence and never drag them into a different box.
            if patch_scores.max() - patch_scores.min() <= torch.finfo(scores.dtype).eps:
                self._uniform[patch] = True
                continue
            elite_count = min(len(ids), max(2, math.ceil(len(ids) * self.spec.elite_fraction)))
            elites = local[ids[patch_scores.topk(elite_count).indices]]
            self.means[patch].lerp_(elites.mean(0), self.spec.smoothing)
            self.stds[patch].lerp_(
                elites.std(0, unbiased=False).clamp_min(self.spec.min_std), self.spec.smoothing
            )
            self._uniform[patch] = False

    def metadata(self) -> dict:
        return {
            "algorithm": "patch_island_cem_v1",
            "patch_count": self.patch_count,
            "candidates_per_patch": self.counts.cpu().tolist(),
            "uniform_candidates_per_patch": self.exploration_counts.cpu().tolist(),
            "minimum_candidates_per_patch": self.spec.minimum_candidates_per_patch,
            "uniform_exploration_fraction": self.spec.uniform_exploration_fraction,
            "warm_candidates_per_patch": self.warm_patch_counts,
            "best_scores_per_patch": [
                value if math.isfinite(value) else None for value in self.best_scores.cpu().tolist()
            ],
            "local_means": self.means.cpu().tolist(),
            "local_stds": self.stds.cpu().tolist(),
            "coordinates": "independent normalized local wrist/closing parameters; selector intervals encoded only when proposing candidates",
        }


@torch.no_grad()
def first_episodes(
    env, policy=None, max_steps: int | None = None, *, deadline: float | None = None
) -> tuple[dict, dict]:
    """Collect exactly the first episode from every environment, including failures."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Evaluation wall-time limit reached")
    obs, _ = env.reset()
    seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    metrics = {name: torch.zeros_like(value) for name, value in env.last_metrics.items()}
    grasps = {name: torch.zeros_like(value) for name, value in env.last_grasp.items()}
    zero = torch.zeros((env.num_envs, env.cfg.control.action_dim), device=env.device)
    limit = env.max_episode_length + 2 if max_steps is None else max_steps

    def collection_error(error, completed_steps):
        # Zero-filled unseen rows are placeholders, never completed episodes.
        # Preserve exact terminal packets so bounded checks remain diagnosable.
        error.verification_diagnostics = {
            "completed_mask": seen.clone(),
            "episode_metrics": {name: value.clone() for name, value in metrics.items()},
            "first_episode_grasps": {name: value.clone() for name, value in grasps.items()},
            "control_steps": completed_steps,
            "unseen_rows_are_placeholders": True,
        }
        capture = getattr(env, "verification_diagnostic_state", None)
        if capture is not None:
            try:
                error.verification_diagnostics["current_native_state"] = capture()
            except Exception as capture_error:
                error.verification_diagnostics["native_state_capture_error"] = (
                    f"{type(capture_error).__name__}: {capture_error}"
                )
        return error

    for step in range(limit):
        if deadline is not None and time.perf_counter() >= deadline:
            raise collection_error(TimeoutError("Evaluation wall-time limit reached"), step)
        actions = zero if policy is None else policy(obs["policy"])
        obs, reward, terminated, truncated, _ = env.step(actions)
        if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
            raise collection_error(
                RuntimeError("Simulation produced a non-finite observation or reward"), step + 1
            )
        new = (terminated | truncated) & ~seen
        for name in metrics:
            metrics[name][new] = env.last_metrics[name][new]
        for name in grasps:
            grasps[name][new] = env.last_grasp[name][new]
        seen |= new
        if seen.all():
            break
    if not seen.all():
        raise collection_error(
            RuntimeError(
                f"Evaluation step limit reached with {int((~seen).sum())} unfinished episodes"
            ),
            limit,
        )
    return metrics, grasps


def _take(bundle: dict, indices: torch.Tensor) -> dict:
    return {name: value[indices].clone() for name, value in bundle.items()}


COLLISION_TENSOR_FIELDS = (
    "collision_tool_hand_max_penetration_m",
    "collision_nonadjacent_self_max_penetration_m",
    "collision_audit_samples",
    "collision_feasible",
    "collision_phase_samples",
)
COLLISION_PHASE_DEPTH_FIELDS = (
    "collision_tool_hand_phase_max_penetration_m",
    "collision_nonadjacent_self_phase_max_penetration_m",
)
SEARCH_COLLISION_PHASE_IDS = tuple(
    index for index, phase in enumerate(COLLISION_AUDIT_PHASES) if phase != "nominal_trajectory"
)
NOMINAL_COLLISION_PHASE_ID = COLLISION_AUDIT_PHASES.index("nominal_trajectory")


def collision_gate(bundle: dict, collision, *, required_phases=None) -> torch.Tensor:
    """Fail closed on missing native evidence; never infer safety from SG/error.

    Search requires the first four phases. A verified selected state requires
    fresh full-trajectory evidence as well. Tolerances are geometric acceptance
    budgets, independent of contact offsets and of the optimizer's objective.
    """
    count = len(bundle["scores"]) if "scores" in bundle else len(bundle["settle_valid"])
    for name in COLLISION_TENSOR_FIELDS:
        value = bundle.get(name)
        shape = (
            (count, len(COLLISION_AUDIT_PHASES)) if name == "collision_phase_samples" else (count,)
        )
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(f"Missing or invalid per-candidate native collision evidence: {name}")
        if name == "collision_feasible":
            if value.dtype != torch.bool:
                raise ValueError("Native collision feasibility must be boolean")
        elif name in ("collision_phase_samples", "collision_audit_samples"):
            if value.dtype != torch.int64 or (value < 0).any():
                raise ValueError("Native collision sample counts must be nonnegative int64")
        elif not value.is_floating_point() or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError("Native collision penetration must be finite and nonnegative")
    samples = bundle["collision_phase_samples"]
    if not torch.equal(samples.sum(-1), bundle["collision_audit_samples"]):
        raise ValueError("Native collision phase counts do not match the total sample count")
    required = list(
        range(len(COLLISION_AUDIT_PHASES)) if required_phases is None else required_phases
    )
    if not required or any(index < 0 or index >= len(COLLISION_AUDIT_PHASES) for index in required):
        raise ValueError("Collision gate requires valid native audit phases")
    return (
        bundle["collision_feasible"]
        & (samples[:, required] > 0).all(-1)
        & (bundle["collision_tool_hand_max_penetration_m"] <= collision.max_tool_penetration_m)
        & (
            bundle["collision_nonadjacent_self_max_penetration_m"]
            <= collision.max_self_penetration_m
        )
    )


def collision_search_fitness(grasps: dict, scores: torch.Tensor, collision) -> torch.Tensor:
    """Retain measured SG; use penetration magnitude to steer invalid proposals.

    Every audited feasible candidate ranks above every infeasible candidate,
    including an invalid state with perfect SG. A completely invalid island can
    still learn which motions reduce overlap instead of receiving arbitrary ties.
    The numerical denominator does not relax a zero-penetration acceptance gate.
    """
    feasible = collision_gate(
        {**grasps, "scores": scores}, collision, required_phases=SEARCH_COLLISION_PHASE_IDS
    )
    if scores.ndim != 1 or not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError("Collision steering requires finite per-candidate stability scores")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("Stable-time fractions must lie in [0, 1]")
    penalty = torch.zeros_like(scores)
    epsilon = torch.finfo(scores.dtype).eps
    for field, allowed in (
        ("collision_tool_hand_max_penetration_m", collision.max_tool_penetration_m),
        ("collision_nonadjacent_self_max_penetration_m", collision.max_self_penetration_m),
    ):
        excess = (grasps[field] - allowed).clamp_min(0)
        penalty += torch.log1p(excess / max(allowed, epsilon))
    return torch.where(feasible, scores, -1 - penalty)


def release_retention_fitness(grasps, fitness, wrench):
    """Steer unsuccessful closure toward the declared free-retention limits.

    Zero wrench scores otherwise tie every slipping candidate. Relative native
    object/palm motion supplies a dimensionless penalty using the same position
    and angle tolerances as the existing settle gate. Raw SG is unchanged, and
    measured overlap or functional penalties can only decrease further.
    """
    count = len(fitness)
    names = ("object_pose", "palm_pose", "closure_reference_object_in_palm")
    for name in names:
        value = grasps.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (count, 7)
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Missing or invalid native free-retention pose: {name}")
    settled = grasps.get("settle_valid")
    if (
        not isinstance(settled, torch.Tensor)
        or settled.shape != (count,)
        or settled.dtype != torch.bool
    ):
        raise ValueError("Free-retention steering requires a boolean native settle gate")
    position, angle = in_hand_errors(*(grasps[name] for name in names))
    feasible = (
        settled & (position <= wrench.position_tolerance_m) & (angle <= wrench.angle_tolerance_rad)
    )
    penalty = torch.log1p((position / wrench.position_tolerance_m - 1).clamp_min(0))
    penalty += torch.log1p((angle / wrench.angle_tolerance_rad - 1).clamp_min(0))
    grasps["release_position_error_m"] = position
    grasps["release_angle_error_rad"] = angle
    grasps["release_retention_feasible"] = feasible
    rejected = torch.minimum(fitness, -torch.ones_like(fitness)) - penalty
    return torch.where(feasible, fitness, rejected), feasible


def opposition_grip_gate(grasps, spec):
    """Require freshly observed simultaneous distal opposition during settling."""
    names = (
        "opposition_contact_samples",
        "opposition_audit_samples",
        "opposition_contact_fraction",
        "opposition_group_contact_fraction",
    )
    if any(name not in grasps for name in names):
        raise ValueError("Missing native opposition contact audit")
    contact, samples, fraction, groups = [grasps[name] for name in names]
    count = len(grasps["hand_in_object"])
    if (
        any(not isinstance(value, torch.Tensor) for value in (contact, samples, fraction, groups))
        or contact.shape != (count,)
        or samples.shape != (count,)
        or contact.dtype != torch.int64
        or samples.dtype != torch.int64
        or fraction.shape != (count,)
        or groups.shape != (count, 4)
        or not fraction.is_floating_point()
        or not groups.is_floating_point()
        or not torch.isfinite(fraction).all()
        or not torch.isfinite(groups).all()
        or (contact < 0).any()
        or (samples < contact).any()
        or (fraction < 0).any()
        or (fraction > 1).any()
        or (groups < 0).any()
        or (groups > 1).any()
    ):
        raise ValueError("Invalid native opposition contact audit")
    measured_fraction = contact.to(fraction) / samples.clamp_min(1)
    if not torch.allclose(fraction, measured_fraction, atol=1e-6, rtol=0):
        raise ValueError("Opposition fraction contradicts its native sample counts")
    if (fraction > groups[:, 0] + 1e-6).any() or (
        spec["minimum_finger_groups"] * fraction > groups[:, 1:].sum(-1) + 1e-6
    ).any():
        raise ValueError("Simultaneous opposition contradicts distal group coverage")
    fingers = groups[:, 1:].topk(spec["minimum_finger_groups"], dim=-1).values.mean(-1)
    penalty = (1 - groups[:, 0]) + (1 - fingers) + (1 - fraction)
    return (samples > 0) & (fraction >= spec["minimum_contact_fraction"]), penalty


def native_opposition_centers(grasps, spec):
    """Validate bounded, complete native contact-point geometry separately from forces."""
    names = (
        "opposition_contact_centers_object_m",
        "opposition_contact_center_forces_n",
        "opposition_contact_point_counts",
        "opposition_contact_center_valid",
        "opposition_contact_geometry_complete",
    )
    if any(name not in grasps for name in names):
        raise ValueError("Missing native opposition contact-point geometry")
    centers, forces, counts, valid, complete = [grasps[name] for name in names]
    count = len(grasps["hand_in_object"])
    if (
        any(
            not isinstance(value, torch.Tensor)
            for value in (centers, forces, counts, valid, complete)
        )
        or centers.shape != (count, 4, 3)
        or forces.shape != (count, 4)
        or counts.shape != (count, 4)
        or valid.shape != (count, 4)
        or complete.shape != (count,)
        or not centers.is_floating_point()
        or not forces.is_floating_point()
        or counts.dtype != torch.int64
        or valid.dtype != torch.bool
        or complete.dtype != torch.bool
        or not torch.isfinite(centers).all()
        or not torch.isfinite(forces).all()
        or (forces < 0).any()
        or (counts < 0).any()
    ):
        raise ValueError("Invalid native opposition contact-point geometry")
    expected_valid = (counts > 0) & (forces >= spec["minimum_contact_force_n"]) & complete[:, None]
    if (
        not torch.equal(valid, expected_valid)
        or ((counts == 0) & (forces > 0) & complete[:, None]).any()
    ):
        raise ValueError(
            "Native contact-center validity contradicts force/count/completeness evidence"
        )
    return centers, valid


def functional_grip_metrics(area, grasps, patch_ids):
    """Use true final contact points for opposed grips, fixed pivots for proposals otherwise."""
    if area.functional_frame.opposition is None:
        return area.functional_metrics(grasps["hand_in_object"], patch_ids)
    centers, valid = native_opposition_centers(grasps, area.functional_frame.opposition)
    return area.functional_metrics(
        grasps["hand_in_object"],
        patch_ids,
        contact_centers_object_m=centers,
        contact_center_valid=valid,
    )


def functional_grip_fitness(area, grasps, patch_ids, fitness):
    """Enforce a declared tool-use frame on freshly captured canonical poses.

    SG cannot reward an upside-down functional grip or a grip outside the
    declared handle. The unmodified stability scores remain separate evidence.
    Areas without an explicit functional convention keep their existing behavior.
    """
    if not hasattr(area, "functional_metrics") or area.functional_frame is None:
        return fitness, torch.ones_like(fitness, dtype=torch.bool)
    metrics = functional_grip_metrics(area, grasps, patch_ids)
    for name, value in metrics.items():
        grasps[f"functional_grip_{name}"] = value
    frame = area.functional_frame
    penalty = (
        (metrics["axis_angle_rad"] / frame.max_axis_deviation_rad - 1).clamp_min(0)
        + (metrics["palm_angle_rad"] / frame.max_palm_deviation_rad - 1).clamp_min(0)
        + metrics["segment_excess_m"] / frame.max_transverse_distance_m
        + (metrics["transverse_distance_m"] / frame.max_transverse_distance_m - 1).clamp_min(0)
    )
    if "handle_placement_excess_m" in metrics:
        penalty += metrics["handle_placement_excess_m"] / frame.max_transverse_distance_m
    if "hand_pocket_excess_m" in metrics:
        penalty += metrics["hand_pocket_excess_m"] / frame.max_transverse_distance_m
    if "contact_group_segment_excess_m" in metrics:
        group_excess = metrics["contact_group_segment_excess_m"].masked_fill(
            ~grasps["opposition_contact_center_valid"], 0
        )
        penalty += group_excess.amax(-1) / frame.max_transverse_distance_m
    if frame.opposition is not None:
        opposed, contact_penalty = opposition_grip_gate(grasps, frame.opposition)
        metrics["feasible"] &= opposed
        grasps["functional_grip_opposition_feasible"] = opposed
        penalty += contact_penalty
        grasps["functional_grip_feasible"] = metrics["feasible"]
    rejected = torch.minimum(fitness, -torch.ones_like(fitness)) - penalty
    return torch.where(metrics["feasible"], fitness, rejected), metrics["feasible"]


def merge_verification_collisions(bundle: dict, metrics: dict, collision) -> dict:
    """Merge new trajectory samples without overwriting the construction audit."""
    fresh = {name: metrics[name] for name in COLLISION_TENSOR_FIELDS if name in metrics}
    fresh["scores"] = bundle["scores"]
    # The new scene must supply its own nominal trajectory evidence. It cannot
    # satisfy that phase merely by inheriting a former nominal verification.
    fresh_safe = collision_gate(fresh, collision, required_phases=(NOMINAL_COLLISION_PHASE_ID,))
    collision_gate(bundle, collision, required_phases=SEARCH_COLLISION_PHASE_IDS)
    phase_depths = {}
    for name in COLLISION_PHASE_DEPTH_FIELDS:
        if name not in bundle and name not in metrics:
            continue
        old, new = bundle.get(name), metrics.get(name)
        expected = (len(bundle["scores"]), len(COLLISION_AUDIT_PHASES))
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != expected
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
            or (value < 0).any()
            for value in (old, new)
        ):
            raise ValueError(f"Invalid native per-phase collision penetration: {name}")
        phase_depths[name] = torch.maximum(old, new)
    # Validate every supplied array before changing the original evidence.
    for name in COLLISION_TENSOR_FIELDS:
        if name == "collision_feasible":
            bundle[name] = bundle[name] & fresh_safe
        elif name in ("collision_phase_samples", "collision_audit_samples"):
            bundle[name] = bundle[name] + fresh[name]
        else:
            bundle[name] = torch.maximum(bundle[name], fresh[name])
    bundle.update(phase_depths)
    return {
        name: value
        for name, value in metrics.items()
        if name not in COLLISION_TENSOR_FIELDS + COLLISION_PHASE_DEPTH_FIELDS
    }


def rank_verified_grasps(
    bundle: dict, search, evaluation, collision: CollisionSpec | None = None
) -> tuple[torch.Tensor, dict]:
    """Prefer task accuracy among grasps with comparable measured robustness.

    Reliability remains lexicographic: success first, then completion. Only the
    best reliability tier defines the relative stability cutoff. A nominally
    perfect trajectory cannot displace a robust grip by having negligible SG.
    Raw SG is never blended into or replaced by the tracking score.
    """
    scores = bundle["scores"]
    fields = {
        name: bundle[name]
        for name in (
            "metric_success",
            "metric_completion",
            "metric_position_rmse",
            "metric_angle_mean",
        )
    }
    if scores.ndim != 1 or len(scores) == 0 or not torch.isfinite(scores).all():
        raise ValueError("Grasp ranking requires a nonempty finite stability vector")
    for name, value in fields.items():
        if value.shape != scores.shape or not torch.isfinite(value).all():
            raise ValueError(f"Grasp ranking requires finite per-candidate {name}")
    success = fields["metric_success"]
    completion = fields["metric_completion"]
    feasible = collision_gate(bundle, CollisionSpec() if collision is None else collision)
    bundle["collision_feasible"] = feasible
    if not feasible.any():
        raise RuntimeError("No collision-feasible grasp completed native trajectory verification")
    functional = bundle.get("functional_grip_feasible", torch.ones_like(feasible))
    if (
        not isinstance(functional, torch.Tensor)
        or functional.shape != feasible.shape
        or functional.dtype != torch.bool
    ):
        raise ValueError("Functional grasp feasibility must be a per-candidate boolean tensor")
    admissible = feasible & functional
    if not admissible.any():
        raise RuntimeError("No functional grasp completed native trajectory verification")
    executable = (
        admissible
        & (scores > 0)
        & (success == 1)
        & (completion >= 1 - torch.finfo(completion.dtype).eps)
    )
    if not executable.any():
        raise RuntimeError(
            "No collision-feasible grasp completed the trajectory with strict success"
        )
    best_success = success[executable].max()
    best_completion = completion[executable].max()
    best_tier = executable & (success == best_success) & (completion == best_completion)
    best_tier_stability = scores[best_tier].max()
    cutoff = search.minimum_relative_stability * best_tier_stability
    eligible = best_tier & (scores >= cutoff)
    tracking_cost = (fields["metric_position_rmse"] / evaluation.position_tolerance_m).square()
    tracking_cost += (fields["metric_angle_mean"] / evaluation.angle_tolerance_rad).square()
    if not torch.isfinite(tracking_cost).all():
        raise ValueError("Normalized grasp tracking cost must be finite")
    # Stable sorts apply the most important criterion last. Eligible grips win
    # within the best reliability tier; their tracking cost wins before SG.
    order = torch.arange(len(scores), device=scores.device)
    for values, descending in (
        (scores, True),
        (tracking_cost, False),
        (eligible.int(), True),
        (completion, True),
        (success, True),
        (admissible.int(), True),
    ):
        order = order[torch.argsort(values[order], descending=descending, stable=True)]
    bundle["selection_best_tier"] = best_tier
    bundle["selection_eligible"] = eligible
    bundle["selection_tracking_cost"] = tracking_cost
    metadata = {
        "minimum_relative_stability": search.minimum_relative_stability,
        "best_success_tier": float(best_success),
        "best_completion_tier": float(best_completion),
        "best_tier_stability": float(best_tier_stability),
        "stability_cutoff": float(cutoff),
        "eligible_grasps": int(eligible.sum()),
        "collision_feasible_grasps": int(feasible.sum()),
        "functional_collision_feasible_grasps": int(admissible.sum()),
        "strictly_executable_grasps": int(executable.sum()),
        "selected_tracking_cost": float(tracking_cost[order[0]]),
        "tracking_cost_definition": "(COM_position_RMSE/position_tolerance)^2 + (mean_angle/angle_tolerance)^2",
    }
    return order, metadata


def _artifact(env, args, best: dict, iteration: int, population: int, verified: bool) -> dict:
    artifact = {
        "schema_version": GRASP_SCHEMA_VERSION,
        "task": env.task.name,
        "task_fingerprint": env.task.fingerprint(),
        "joint_names": list(JOINT_NAMES),
        "seed": args.seed,
        "iterations": iteration,
        "population": population,
        "physical_environments": env.num_envs,
        "control_dt": env.step_dt,
        "hand_asset_sha256": file_hash(Path(env.cfg.hand_asset)),
        "grasp_fingerprint": env.cfg.experiment.grasp_fingerprint(),
        "physics_fingerprint": env.cfg.experiment.physics_fingerprint(),
        "discovery_sha256": env.area.fingerprint(),
        "experiment": env.cfg.experiment.to_dict(),
        "search_parameter_fields": [list(field) for field in PARAMETER_FIELDS],
        "score_definition": "mean stable-time fraction across 12 independent +/- Cartesian force/torque ramps",
        "trajectory_verified": verified,
        "collision_verified": verified,
        "initialization": "fixture closing, then freely settling under gravity and contact",
        "selection": "native collision feasibility, strict success/full completion, relative stability eligibility, normalized COM/angle tracking cost, SG tie-break"
        if verified
        else "wrench stability",
        "search_sampling": env.search_sampling,
        "physics": env.physics_metadata,
        **{name: value.cpu() for name, value in best.items()},
    }
    if getattr(env.area, "functional_frame", None) is not None:
        metrics = functional_grip_metrics(env.area, best, best["area_patch_ids"])
        if verified and not metrics["feasible"].all():
            raise ValueError(
                "A verified artifact cannot violate its declared functional grip frame"
            )
        if env.area.functional_frame.opposition is not None:
            opposed, _ = opposition_grip_gate(best, env.area.functional_frame.opposition)
            if verified and not opposed.all():
                raise ValueError("A verified functional grip requires native distal opposition")
        artifact["functional_grip_verified"] = verified
        artifact["functional_grip_convention"] = deepcopy(env.area.calibration["functional_grip"])
        contact_metadata = getattr(env, "opposition_contact_metadata", None)
        if env.area.functional_frame.opposition is not None:
            if verified and (not isinstance(contact_metadata, dict) or not contact_metadata):
                raise ValueError(
                    "A verified opposed grasp requires native contact-point provenance"
                )
            if contact_metadata is not None:
                artifact["opposition_contact_metadata"] = deepcopy(contact_metadata)
    audit = deepcopy(env.collision_audit_metadata)
    if (
        audit.get("schema_version") != COLLISION_AUDIT_VERSION
        or audit.get("method") != COLLISION_AUDIT_METHOD
    ):
        raise ValueError("Native collision audit metadata has an unsupported version or method")
    if audit.get("phases") != list(COLLISION_AUDIT_PHASES):
        raise ValueError("Native collision audit metadata has an incompatible phase order")
    if verified and audit.get("sampling_protocol") != COLLISION_SAMPLING_PROTOCOL:
        raise ValueError(
            "Native collision audit did not sample every required fixture/substep state"
        )
    audit["physics_fingerprint"] = artifact["physics_fingerprint"]
    audit["candidate_state_sha256"] = candidate_state_hashes(artifact)
    artifact["collision_audit"] = audit
    proposals = getattr(env, "warm_proposal_provenance", None)
    if proposals is not None:
        artifact["warm_proposal_provenance"] = {
            **deepcopy(proposals),
            "fresh_reclosure_executed": bool(getattr(env, "fresh_reclosure_executed", False)),
            "reclosed_under_grasp_fingerprint": artifact["grasp_fingerprint"],
            "reclosed_under_physics_fingerprint": artifact["physics_fingerprint"],
            "source_states_scores_and_collision_proof_inherited": False,
        }
    if verified:
        if not collision_gate(artifact, env.cfg.experiment.collision).all():
            raise ValueError("A verified artifact cannot contain a collision-infeasible nominee")
        validate_collision_audit(artifact, collision_spec=env.cfg.experiment.collision)
    return artifact


def _save_population_diagnostics(
    context, args, population, iteration, candidates, grasps, fractions, scores, fitness, patch_ids
):
    """Preserve rejected native states without presenting them as usable grasps."""
    diagnostic = _artifact(
        context,
        args,
        {
            **grasps,
            "scores": scores,
            "parameters": candidates,
            "area_patch_ids": patch_ids,
            "wrench_stable_fractions": fractions,
            "collision_steering_fitness": fitness,
            "stress_sampled": grasps["collision_phase_samples"][
                :, COLLISION_AUDIT_PHASES.index("wrench")
            ]
            > 0,
        },
        iteration,
        population,
        verified=False,
    )
    diagnostic.update(
        artifact_kind="unverified_native_candidate_population",
        diagnostic_only=True,
        score_interpretation="Returned wrench fractions are diagnostic; untested candidates retain zero placeholders and stress_sampled=false. Collision fitness is separate, and these states are not selected grasps.",
        initial_pose_coordinates="Captured native poses with environment-grid translations removed; no authored FK substitution.",
    )
    path = Path(args.output).with_name(
        f"{Path(args.output).stem}.iteration-{iteration}.candidates.pt"
    )
    atomic_torch_save(diagnostic, path)
    return str(path)


def save_verification_diagnostics(
    grasp: dict, metrics: dict | None, output, *, native_metadata=None, error=None
) -> str:
    """Save fresh verification evidence before selection can reject every row.

    Source states and their construction audit remain separate from new nominal
    measurements. A timeout packet preserves only completed terminal rows and
    labels its unseen buffers, rather than silently treating them as successes.
    This archive can never qualify as a verified grasp.
    """

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu(item) for item in value)
        return deepcopy(value)

    diagnostic = cpu(grasp)
    diagnostic.update(
        artifact_kind="unverified_native_nominal_verification",
        diagnostic_only=True,
        trajectory_verified=False,
        collision_verified=False,
        verification_collection_completed=metrics is not None,
        verification_metrics={} if metrics is None else cpu(metrics),
        verification_native_metadata=cpu(native_metadata),
        verification_interpretation="Fresh nominal measurements remain separate from original construction/wrench evidence. This diagnostic archive has not passed selection and cannot be used as a verified grasp.",
    )
    if error is not None:
        diagnostic["verification_error"] = f"{type(error).__name__}: {error}"
        diagnostic["partial_verification"] = cpu(getattr(error, "verification_diagnostics", {}))
    path = Path(output).with_name(f"{Path(output).stem}.verification.pt")
    atomic_torch_save(diagnostic, path)
    return str(path)


def run_optimization(env, args) -> dict:
    # Compact verification closes the native search world. Preserve its plain
    # provenance now; DirectRLEnv.num_envs/device depend on deleted native data.
    context = SimpleNamespace(
        cfg=env.cfg,
        task=env.task,
        area=env.area,
        physics_metadata=env.physics_metadata,
        num_envs=env.num_envs,
        step_dt=env.step_dt,
        device=env.device,
        collision_audit_metadata=deepcopy(env.collision_audit_metadata),
        opposition_contact_metadata=deepcopy(getattr(env, "opposition_contact_metadata", None)),
        warm_proposal_provenance=None,
        fresh_reclosure_executed=False,
    )
    spec = env.cfg.experiment.search
    patch_count = len(env.area.patches)
    minimum_population = patch_count * spec.minimum_candidates_per_patch
    if patch_count < 1 or env.num_envs < minimum_population:
        raise ValueError(
            f"Area search requires at least {minimum_population} physical environments "
            f"for {patch_count} patches"
        )
    population = max(
        minimum_population, env.num_envs // env.cfg.experiment.wrench.directions_per_batch
    )
    search = IslandCrossEntropySearch(
        SEARCH_DIM, population, patch_count, env.device, args.seed, spec
    )
    best = None
    warm_start, warm_proposals = (
        getattr(args, "warm_start", None),
        getattr(args, "warm_proposals", None),
    )
    if warm_start and warm_proposals:
        raise ValueError(
            "Strict warm-start and parameter-only warm-proposals are mutually exclusive"
        )
    if warm_proposals:
        proposals = load_warm_proposals(
            warm_proposals,
            env.task,
            hand_asset=Path(env.cfg.hand_asset),
            control_dt=env.step_dt,
            discovery_sha256=env.area.fingerprint(),
            device=env.device,
        )
        # This dedicated loader returns no canonical states or inherited SG.
        # Proposal ordering is a prior; each state and audit is rebuilt here.
        search.warm_start(proposals["parameters"])
        context.warm_proposal_provenance = deepcopy(proposals["provenance"])
    elif warm_start:
        previous = load_grasp(
            warm_start,
            env.task,
            env.device,
            hand_asset=Path(env.cfg.hand_asset),
            control_dt=env.step_dt,
            grasp_fingerprint=env.cfg.experiment.grasp_fingerprint(),
            discovery_sha256=env.area.fingerprint(),
        )
        if previous.get("parameters", torch.empty(0)).shape[1:] != (SEARCH_DIM,):
            raise ValueError("Warm-start artifact has no compatible search parameters")
        # Reclose and retest the warm proposal under this run; do not inherit
        # unknown canonical state or optimistic scores from another simulator.
        search.warm_start(previous["parameters"], previous["scores"])
    context.search_sampling = search.metadata()
    print(
        f"Area CEM: {patch_count} islands, {population} candidates "
        f"({context.search_sampling['candidates_per_patch']} per patch), "
        f"{math.ceil(population * WRENCH_DIRECTIONS / env.num_envs)} wrench waves per generation",
        flush=True,
    )
    history = []
    started = time.perf_counter()
    deadline = started + spec.max_wall_time_s
    stop_reason = "iteration_limit"
    completed = 0
    feasible_patch_best = torch.full((patch_count,), -torch.inf, device=env.device)
    for iteration in range(args.iterations):
        try:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Grasp search wall-time limit reached")
            candidates = search.sample()
            grasps = env.generate_grasps(candidates, deadline=deadline)
            context.fresh_reclosure_executed = True
            # Native opposition is already observed during free settling. Do
            # not spend wrench waves on a merely palm/MCP-clamped candidate.
            functional_grip_fitness(
                context.area, grasps, search.patch_ids, candidates.new_zeros(population)
            )
            fractions = env.stress_grasps(grasps, deadline=deadline)
            if (
                fractions.shape != (population, WRENCH_DIRECTIONS)
                or not torch.isfinite(fractions).all()
            ):
                raise RuntimeError("Wrench evaluation returned invalid stability scores")
            scores = fractions.mean(-1)
            fitness = collision_search_fitness(grasps, scores, context.cfg.experiment.collision)
            fitness, retention_valid = release_retention_fitness(
                grasps, fitness, context.cfg.experiment.wrench
            )
            fitness, functional_valid = functional_grip_fitness(
                context.area, grasps, search.patch_ids, fitness
            )
            search.update(candidates, fitness)
            context.search_sampling = search.metadata()
            collision_valid = collision_gate(
                {**grasps, "scores": scores},
                context.cfg.experiment.collision,
                required_phases=SEARCH_COLLISION_PHASE_IDS,
            )
            valid = retention_valid & collision_valid & functional_valid & (scores > 0)
            for patch in range(patch_count):
                rows = valid & (search.patch_ids == patch)
                if rows.any():
                    feasible_patch_best[patch] = torch.maximum(
                        feasible_patch_best[patch], scores[rows].max()
                    )
            context.search_sampling["best_steering_fitness_per_patch"] = (
                context.search_sampling.pop("best_scores_per_patch")
            )
            context.search_sampling["best_scores_per_patch"] = [
                score if math.isfinite(score) else None
                for score in feasible_patch_best.cpu().tolist()
            ]
            context.search_sampling["score_semantics"] = (
                "raw SG only for collision-feasible freely settled candidates; separate CEM steering fitness penalizes overlap, release slip normalized by declared wrench tolerances, and functional/contact violations"
            )
            diagnostic_path = _save_population_diagnostics(
                context,
                args,
                population,
                iteration + 1,
                candidates,
                grasps,
                fractions,
                scores,
                fitness,
                search.patch_ids,
            )
            if valid.any():
                current = _take(
                    {
                        "scores": scores,
                        "parameters": candidates,
                        "area_patch_ids": search.decode(candidates)[0],
                        "wrench_stable_fractions": fractions,
                        **grasps,
                    },
                    valid.nonzero(as_tuple=False).flatten(),
                )
                if best is not None:
                    current = {
                        name: torch.cat((best[name], value)) for name, value in current.items()
                    }
                count = min(
                    max(args.keep, spec.trajectory_shortlist), env.num_envs, len(current["scores"])
                )
                order = current["scores"].topk(count).indices
                best = _take(current, order)
            completed = iteration + 1
            row = {
                "iteration": completed,
                "best_score": float(best["scores"].max()) if best is not None else 0.0,
                "raw_best_score": float(scores.max()),
                "collision_feasible_grasps": int(collision_valid.sum()),
                "functional_grip_feasible_grasps": int(functional_valid.sum()),
                "minimum_tool_hand_penetration_m": float(
                    grasps["collision_tool_hand_max_penetration_m"].min()
                ),
                "minimum_nonadjacent_self_penetration_m": float(
                    grasps["collision_nonadjacent_self_max_penetration_m"].min()
                ),
                "best_steering_fitness": float(fitness.max()),
                "candidate_diagnostics": diagnostic_path,
                "settled_grasps": int(grasps["settle_valid"].sum()),
                "usable_grasps": int(valid.sum()),
                "candidate_population": population,
                "settled_per_patch": torch.bincount(
                    search.patch_ids[grasps["settle_valid"]], minlength=patch_count
                )
                .cpu()
                .tolist(),
                "usable_per_patch": torch.bincount(search.patch_ids[valid], minlength=patch_count)
                .cpu()
                .tolist(),
                "collision_feasible_per_patch": torch.bincount(
                    search.patch_ids[collision_valid], minlength=patch_count
                )
                .cpu()
                .tolist(),
                "best_scores_per_patch": context.search_sampling["best_scores_per_patch"],
                "elapsed_s": time.perf_counter() - started,
            }
            history.append(row)
            if best is not None:
                atomic_torch_save(
                    _artifact(context, args, best, completed, population, verified=False),
                    args.output,
                )
            write_json(
                {"task": env.task.name, "history": history, "stop_reason": "running"},
                Path(args.output).with_suffix(".json"),
            )
            print(
                f"CEM {completed}/{args.iterations}: SG={row['best_score']:.3f}, "
                f"settled={row['settled_grasps']}/{population}, usable={row['usable_grasps']}, "
                f"elapsed={row['elapsed_s']:.1f}s",
                flush=True,
            )
        except TimeoutError:
            stop_reason = "wall_time_limit"
            break
    if best is None:
        write_json(
            {
                "task": env.task.name,
                "history": history,
                "stop_reason": stop_reason,
                "usable_grasps": 0,
                "trajectory_verified": False,
                "collision_verified": False,
                "artifact_written_in_this_run": False,
            },
            Path(args.output).with_suffix(".json"),
        )
        raise RuntimeError(
            "No collision-feasible freely settled grasp survived wrench testing within the finite search budget. "
            "Inspect the grasp area and increase the candidate population or bounded budget."
        )
    # A second explicitly bounded phase verifies the shortlist on the complete
    # tool trajectory. No per-candidate Python loop or partial-trajectory score.
    verification_started = time.perf_counter()
    verification_deadline = (
        verification_started + env.cfg.experiment.evaluation.verification_max_wall_time_s
    )
    compact_verifier = getattr(env, "verify_grasps_in_small_scene", None)
    verification_count = len(best["scores"]) if compact_verifier is not None else context.num_envs
    verification_diagnostic = None
    verification_metrics = None
    try:
        verification_metrics = (
            compact_verifier(best, deadline=verification_deadline)
            if compact_verifier is not None
            else env.verify_grasps(best, deadline=verification_deadline)
        )
        verification_diagnostic = save_verification_diagnostics(
            _artifact(context, args, best, completed, population, verified=False),
            verification_metrics,
            args.output,
            native_metadata=context.collision_audit_metadata,
        )
        metrics = merge_verification_collisions(
            best, verification_metrics, context.cfg.experiment.collision
        )
        best["collision_feasible"] = collision_gate(best, context.cfg.experiment.collision)
        best.update({f"metric_{name}": value for name, value in metrics.items()})
        order, selection = rank_verified_grasps(
            best,
            context.cfg.experiment.search,
            context.cfg.experiment.evaluation,
            context.cfg.experiment.collision,
        )
    except (RuntimeError, ValueError, TimeoutError) as error:
        if verification_diagnostic is None:
            verification_diagnostic = save_verification_diagnostics(
                _artifact(context, args, best, completed, population, verified=False),
                verification_metrics,
                args.output,
                native_metadata=context.collision_audit_metadata,
                error=error,
            )
        write_json(
            {
                "task": context.task.name,
                "history": history,
                "stop_reason": "verification_failed",
                "verification_error": str(error),
                "iterations_completed": completed,
                "candidate_population": population,
                "trajectory_verified": False,
                "collision_verified": False,
                "partial_grasp_artifact_preserved": str(args.output),
                "verification_diagnostic_artifact": verification_diagnostic,
                "elapsed_s": time.perf_counter() - started,
            },
            Path(args.output).with_suffix(".json"),
        )
        raise
    safe_order = order[best["selection_eligible"][order]]
    best = _take(best, safe_order[: args.keep])
    artifact = _artifact(context, args, best, completed, population, verified=True)
    artifact["verification_environments"] = verification_count
    artifact["verification_elapsed_s"] = time.perf_counter() - verification_started
    artifact["selection_metadata"] = selection
    atomic_torch_save(artifact, args.output)
    summary = {
        "task": context.task.name,
        "history": history,
        "stop_reason": stop_reason,
        "iterations_completed": completed,
        "candidate_population": population,
        "physical_environments": context.num_envs,
        "verification_environments": verification_count,
        "verification_elapsed_s": time.perf_counter() - verification_started,
        "verification_diagnostic_artifact": verification_diagnostic,
        "elapsed_s": time.perf_counter() - started,
        "best_stability_score": float(best["scores"][0]),
        "best_completion": float(best["metric_completion"][0]),
        "best_success": bool(best["metric_success"][0]),
        "best_position_rmse_m": float(best["metric_position_rmse"][0]),
        "best_angle_mean_rad": float(best["metric_angle_mean"][0]),
        "wrench_stable_fractions": best["wrench_stable_fractions"][0].cpu().tolist(),
        "trajectory_verified": True,
        "collision_verified": True,
        "collision_audit": artifact["collision_audit"],
        "best_tool_hand_penetration_m": float(best["collision_tool_hand_max_penetration_m"][0]),
        "best_nonadjacent_self_penetration_m": float(
            best["collision_nonadjacent_self_max_penetration_m"][0]
        ),
        "search_sampling": context.search_sampling,
        "selection_metadata": selection,
    }
    if "warm_proposal_provenance" in artifact:
        summary["warm_proposal_provenance"] = artifact["warm_proposal_provenance"]
    write_json(summary, Path(args.output).with_suffix(".json"))
    print(
        f"Selected SG={summary['best_stability_score']:.3f}, "
        f"trajectory completion={summary['best_completion']:.1%}, "
        f"strict success={summary['best_success']}",
        flush=True,
    )
    return summary
