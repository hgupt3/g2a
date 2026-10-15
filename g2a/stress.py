"""Batched grasp parameters and wrench physics, usable without starting Isaac Sim.

The search includes wrist pose, finger spread/opposition, one closing speed per
joint group, and an independent stopping torque for each of the 12 flexion
joints. Stability is the mean survival fraction of 12 independent Cartesian
force/torque ramps, measured relative to the actual palm rather than the goal.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull

from .math import (
    clamp_norm,
    quat_angle,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_to_rotvec,
    relative_pose,
    rotvec_to_quat,
)

SPREAD_JOINTS = (0, 4, 8)
THUMB_OPPOSITION_JOINT = 12
CLOSING_JOINTS = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
CLOSING_GROUPS = (0, 1, 2) * 4
SEARCH_DIM = 6 + 4 + 3 + len(CLOSING_JOINTS)
WRENCH_DIRECTIONS = 12
PARAMETER_FIELDS = (
    ("wrist_region", 6),
    ("finger_spread", 3),
    ("thumb_opposition", 1),
    ("group_closing_speed", 3),
    ("joint_stopping_torque", 12),
)


class ExternalWrenchWriter:
    """Refresh Isaac Lab's cached link frame before writing a world wrench.

    A permanent composer otherwise keeps its first link pose after the object
    rotates. Calling it with ``is_global=True`` does not invalidate that cache.
    Every physics substep must rewrite the current task or stress wrench.
    """

    def __init__(self, asset):
        self.composer = getattr(asset, "permanent_wrench_composer", None)
        self.setter = (
            self.composer.set_forces_and_torques
            if self.composer is not None
            else asset.set_external_force_and_torque
        )

    def __call__(self, forces, torques, *, env_ids=None, is_global=False):
        if is_global and self.composer is not None:
            # A subset reset preserves other environments' buffers while still
            # invalidating the shared pose cache for the next conversion.
            self.composer.reset(env_ids=env_ids)
        self.setter(forces, torques, env_ids=env_ids, is_global=is_global)


def sample_physics_multipliers(
    count: int,
    interval: tuple[float, float],
    device,
    *,
    paired: bool = False,
    buckets: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Independent episode physics, or identical first/second evaluation halves."""
    if count < 1 or (paired and count % 2):
        raise ValueError("Paired physics draws require a positive even environment count")
    draws = count // 2 if paired else count
    if buckets is None:
        samples = interval[0] + (interval[1] - interval[0]) * torch.rand(
            draws, device=device, generator=generator
        )
    else:
        samples = buckets[torch.randint(len(buckets), (draws,), device=device, generator=generator)]
    return samples.repeat(2) if paired else samples


@dataclass
class ClosingParameters:
    initial_joints: torch.Tensor
    joint_speeds: torch.Tensor
    joint_torque_thresholds: torch.Tensor


def _uniform_range(parameters: torch.Tensor, interval: tuple[float, float]) -> torch.Tensor:
    low, high = interval
    return low + (parameters + 1) * (high - low) / 2


def decode_closing(parameters: torch.Tensor, search, lower, upper) -> ClosingParameters:
    """Convert bounded search coordinates to physical LEAP closing commands."""
    if parameters.ndim != 2 or parameters.shape[1] != SEARCH_DIM:
        raise ValueError(f"Grasp search parameters must have shape (N, {SEARCH_DIM})")
    if not torch.isfinite(parameters).all() or (parameters.abs() > 1).any():
        raise ValueError("Grasp search parameters must be finite and in [-1, 1]")
    initial = parameters.new_zeros((len(parameters), 16))
    initial[:, SPREAD_JOINTS] = _uniform_range(parameters[:, 6:9], search.spread_range_rad)
    initial[:, THUMB_OPPOSITION_JOINT] = _uniform_range(
        parameters[:, 9], search.thumb_opposition_range_rad
    )
    initial[:, 14] = search.thumb_initial_intermediate_rad
    speeds = parameters.new_zeros(initial.shape)
    speed_groups = _uniform_range(parameters[:, 10:13], search.closing_speed_range_rad_s)
    speeds[:, CLOSING_JOINTS] = speed_groups[:, CLOSING_GROUPS]
    thresholds = parameters.new_full(initial.shape, torch.inf)
    thresholds[:, CLOSING_JOINTS] = _uniform_range(
        parameters[:, 13:25], search.closing_torque_range_nm
    )
    return ClosingParameters(initial.clamp(lower, upper), speeds, thresholds)


class ClosingTorqueStopper:
    """Latch sustained motor load in the commanded closing direction.

    Implicit actuator torque is a PD estimate, including velocity damping. A
    negative estimate while a positive-speed finger closes is braking effort,
    not evidence that contact is resisting closure. Persistence rejects single
    positive load spikes too. The optional tool_contact mode additionally
    requires independently measured tool touch and pauses command advance
    during confirmation. Other modes use motor feedback alone.
    """

    def __init__(self, shape: tuple[int, int], device):
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("Closing torque state requires positive environment/joint dimensions")
        self.stopped = torch.zeros(shape, dtype=torch.bool, device=device)
        self.contact_time_s = torch.zeros(shape, device=device)
        self.pause = torch.zeros_like(self.stopped)
        self.newly_stopped = torch.zeros_like(self.stopped)

    @torch.no_grad()
    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.stopped[ids] = False
        self.contact_time_s[ids] = 0
        self.pause[ids] = False
        self.newly_stopped[ids] = False

    @torch.no_grad()
    def update(
        self,
        motor_torque: torch.Tensor,
        thresholds: torch.Tensor,
        closing_speeds: torch.Tensor,
        active: torch.Tensor,
        control_dt_s: float,
        *,
        mode: str,
        hold_time_s: float,
        tool_contact: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mode not in ("closing_direction", "absolute", "tool_contact"):
            raise ValueError(
                "Closing torque mode must be closing_direction, absolute, or tool_contact"
            )
        if not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("Closing torque sample duration must be finite and positive")
        if not math.isfinite(hold_time_s) or hold_time_s < 0:
            raise ValueError("Closing torque hold duration must be finite and nonnegative")
        if (
            any(
                value.shape != self.stopped.shape
                for value in (motor_torque, thresholds, closing_speeds)
            )
            or active.shape != self.stopped.shape[:1]
            or active.dtype != torch.bool
        ):
            raise ValueError("Closing torque feedback must match environment/joint dimensions")
        if mode == "tool_contact" and (
            not isinstance(tool_contact, torch.Tensor)
            or tool_contact.shape != self.stopped.shape
            or tool_contact.dtype != torch.bool
            or tool_contact.device != self.stopped.device
        ):
            raise ValueError("Tool-contact stopping requires boolean per-joint tool touch evidence")
        directed = (
            motor_torque.abs() if mode == "absolute" else motor_torque * closing_speeds.sign()
        )
        loaded = (
            active[:, None]
            & (closing_speeds != 0)
            & torch.isfinite(motor_torque)
            & (directed >= thresholds)
        )
        if mode == "tool_contact":
            loaded &= tool_contact
        # Pause throughout genuine loaded contact confirmation. Continuing the
        # target ramp during a 50ms hold at 2rad/s would add 0.1rad wind-up.
        # Motor-only modes continue advancing their targets during the hold.
        self.pause.copy_(loaded if mode == "tool_contact" else torch.zeros_like(loaded))
        self.contact_time_s.copy_(torch.where(loaded, self.contact_time_s + control_dt_s, 0))
        tolerance = torch.finfo(self.contact_time_s.dtype).eps * max(hold_time_s, control_dt_s)
        self.newly_stopped.copy_(
            ~self.stopped & loaded & (self.contact_time_s >= hold_time_s - tolerance)
        )
        self.stopped |= self.newly_stopped
        return self.stopped


def joint_descendant_mask(joint_names, body_names, joint_links, *, device="cpu") -> torch.Tensor:
    """Map each authored joint to its child and descendant native rigid links.

    Contact on the palm or another finger cannot certify a closing joint. The
    map uses the asset's body0/body1 tree, rather than task labels or guessed
    finger/link indices. Requested output rows preserve the policy joint order.
    """
    names, bodies = tuple(joint_names), tuple(body_names)
    if (
        not names
        or not bodies
        or len(set(names)) != len(names)
        or len(set(bodies)) != len(bodies)
        or any(not isinstance(name, str) or not name for name in names + bodies)
        or not isinstance(joint_links, dict)
        or any(name not in joint_links for name in names)
    ):
        raise ValueError("Joint descendant mapping requires unique named joints and native bodies")
    children = {body: [] for body in bodies}
    parent_of = {}
    for joint, links in joint_links.items():
        if (
            not isinstance(joint, str)
            or not isinstance(links, (tuple, list))
            or len(links) != 2
            or any(body not in children for body in links)
            or links[0] == links[1]
        ):
            raise ValueError("Authored joint endpoints must name distinct native rigid bodies")
        parent, child = links
        if child in parent_of:
            raise ValueError("Authored joint graph must be a tree with one parent per rigid body")
        parent_of[child] = parent
        children[parent].append(child)

    completed, active = {}, set()

    def descendants(body):
        if body in active:
            raise ValueError("Authored joint graph contains a cycle")
        if body in completed:
            return completed[body]
        active.add(body)
        found = {body}
        for child in children[body]:
            found |= descendants(child)
        active.remove(body)
        completed[body] = found
        return found

    for body in bodies:
        descendants(body)
    return torch.tensor(
        [[body in completed[joint_links[name][1]] for body in bodies] for name in names],
        dtype=torch.bool,
        device=device,
    )


def joint_descendants_from_usd(hand_asset, joint_names, body_names, *, device="cpu"):
    """Read the source hand's authored joint tree after SDK launch or with usd-core."""
    try:
        from pxr import Usd, UsdPhysics
    except ImportError as error:
        raise RuntimeError(
            "USD joint discovery requires pxr; call after Isaac Sim launch or install usd-core"
        ) from error
    path = Path(hand_asset)
    if not path.is_file():
        raise ValueError(f"Hand USD does not exist: {path}")
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError("Hand USD has no valid default primitive")
    links = {}
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not parent or not child:
            # A world attachment is not a hand-internal articulated joint.
            if prim.GetName() in joint_names:
                raise ValueError("A requested closing joint is attached to the world")
            continue
        if len(parent) != 1 or len(child) != 1 or prim.GetName() in links:
            raise ValueError("Hand joint endpoints/names are not unique")
        links[prim.GetName()] = (
            stage.GetPrimAtPath(parent[0]).GetName(),
            stage.GetPrimAtPath(child[0]).GetName(),
        )
    return joint_descendant_mask(joint_names, body_names, links, device=device)


def tool_contact_mask(forces_w, descendant_mask, minimum_force_n) -> torch.Tensor:
    """Confirm tool-only contact from a public native force matrix, in newtons.

    ``forces_w[N,B,3]`` must already be filtered exclusively to the tool and
    ordered like native hand.body_names. No self/environment or motor feedback
    can be substituted. A strictly positive force is required even when the
    configured reporting-noise threshold is zero. The caller supplies physics
    dt when reading RigidContactView.get_contact_force_matrix.
    """
    if (
        not isinstance(forces_w, torch.Tensor)
        or forces_w.ndim != 3
        or forces_w.shape[-1] != 3
        or not forces_w.is_floating_point()
        or not torch.isfinite(forces_w).all()
        or not isinstance(descendant_mask, torch.Tensor)
        or descendant_mask.ndim != 2
        or descendant_mask.dtype != torch.bool
        or descendant_mask.shape[1] != forces_w.shape[1]
        or descendant_mask.device != forces_w.device
        or isinstance(minimum_force_n, bool)
        or not isinstance(minimum_force_n, (int, float))
        or not math.isfinite(minimum_force_n)
        or minimum_force_n < 0
    ):
        raise ValueError(
            "Tool touch needs finite tool-only body forces, descendant map and noise threshold"
        )
    touched = torch.linalg.vector_norm(forces_w, dim=-1) > minimum_force_n
    return (touched[:, None, :] & descendant_mask[None, :, :]).any(-1)


def bounded_contact_stop_targets(
    targets, measured_q, newly_stopped, thresholds, closing_speeds, stiffness, lower, upper
) -> torch.Tensor:
    """Remove excess target wind-up only on newly confirmed tool-contact stops.

    The retained closing displacement is at most threshold/kp, bounding the
    quasi-static position-drive contribution. Damping is not mislabeled as
    external contact torque. This changes commands only: measured joints and
    collision geometry are never projected or teleported to appear clear.
    """
    if (
        not isinstance(targets, torch.Tensor)
        or targets.ndim != 2
        or not targets.is_floating_point()
        or any(
            not isinstance(value, torch.Tensor) or value.shape != targets.shape
            for value in (measured_q, newly_stopped, thresholds, closing_speeds)
        )
        or any(not value.is_floating_point() for value in (measured_q, thresholds, closing_speeds))
        or newly_stopped.dtype != torch.bool
        or any(
            value.device != targets.device
            for value in (measured_q, newly_stopped, thresholds, closing_speeds)
        )
        or not torch.isfinite(targets).all()
        or not torch.isfinite(measured_q).all()
        or not torch.isfinite(closing_speeds).all()
        or isinstance(stiffness, bool)
        or not isinstance(stiffness, (int, float))
        or not math.isfinite(stiffness)
        or stiffness <= 0
    ):
        raise ValueError(
            "Contact stopping needs matching finite joint state/commands and positive stiffness"
        )
    if (
        not isinstance(lower, torch.Tensor)
        or not isinstance(upper, torch.Tensor)
        or lower.shape != (targets.shape[1],)
        or upper.shape != lower.shape
        or lower.device != targets.device
        or upper.device != targets.device
        or not torch.isfinite(lower).all()
        or not torch.isfinite(upper).all()
        or (lower > upper).any()
        or (
            newly_stopped
            & (~torch.isfinite(thresholds) | (thresholds <= 0) | (closing_speeds == 0))
        ).any()
    ):
        raise ValueError("Confirmed stops need finite positive torque/rate and valid joint limits")
    direction = closing_speeds.sign()
    available = (direction * (targets - measured_q)).clamp_min(0)
    retained = torch.minimum(available, thresholds / stiffness)
    bounded = (measured_q + direction * retained).clamp(lower, upper)
    return torch.where(newly_stopped, bounded, targets)


def wrench_directions(device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Rows are +Fx, -Fx, +Fy, -Fy, +Fz, -Fz, then the six torque tests."""
    axes = torch.eye(6, device=device, dtype=dtype)
    return torch.stack((axes, -axes), dim=1).reshape(WRENCH_DIRECTIONS, 6)


def ramp_wrenches(direction_ids: torch.Tensor, elapsed_s: float, spec) -> torch.Tensor:
    fraction = min(max(elapsed_s / spec.duration_s, 0.0), 1.0) ** spec.ramp_power
    limits = torch.tensor(
        [spec.max_force_n] * 3 + [spec.max_torque_nm] * 3,
        device=direction_ids.device,
    )
    return wrench_directions(direction_ids.device)[direction_ids] * limits * fraction


def in_hand_errors(
    object_pose: torch.Tensor, palm_pose: torch.Tensor, reference_in_palm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slip stays zero when the object and actual palm move together."""
    current = relative_pose(palm_pose, object_pose)
    return (
        (current[:, :3] - reference_in_palm[:, :3]).norm(dim=-1),
        quat_angle(current[:, 3:], reference_in_palm[:, 3:]),
    )


def stable_mask(
    object_pose: torch.Tensor, palm_pose: torch.Tensor, reference_in_palm: torch.Tensor, spec
) -> torch.Tensor:
    position, angle = in_hand_errors(object_pose, palm_pose, reference_in_palm)
    finite = torch.isfinite(object_pose).all(-1) & torch.isfinite(palm_pose).all(-1)
    return finite & (position <= spec.position_tolerance_m) & (angle <= spec.angle_tolerance_rad)


def point_load_wrench(
    object_pose: torch.Tensor,
    com_pose: torch.Tensor,
    com_velocity: torch.Tensor,
    point_in_object: torch.Tensor,
    application_force_in_object: torch.Tensor,
    resistance: float,
    max_resistance: float,
    multiplier: torch.Tensor,
    enabled: torch.Tensor,
    force_frame: str = "object",
) -> torch.Tensor:
    """Return world-frame COM wrench for a load applied at a tool contact point.

    The point velocity includes angular motion. Applying a force away from the
    COM adds its physical moment; applying only the force at the COM loses it.
    """
    point_world = object_pose[:, :3] + quat_apply(object_pose[:, 3:], point_in_object)
    lever = point_world - com_pose[:, :3]
    point_velocity = com_velocity[:, :3] + torch.cross(com_velocity[:, 3:], lever, dim=-1)
    resistance_force = clamp_norm(-resistance * point_velocity, max_resistance)
    if force_frame == "object":
        application_force = quat_apply(object_pose[:, 3:], application_force_in_object)
    elif force_frame == "world":
        application_force = application_force_in_object
    else:
        raise ValueError("Application force frame must be object or world")
    force = application_force + resistance_force
    force *= (multiplier * enabled)[:, None]
    moment = torch.cross(lever, force, dim=-1)
    return torch.cat((force, moment), -1)


def palm_servo_velocity(
    previous_target: torch.Tensor,
    next_target: torch.Tensor,
    measured_pose: torch.Tensor,
    control_dt: float,
    substep: int,
    decimation: int,
    position_gain: float,
    orientation_gain: float,
    linear_speed_limit: float,
    angular_speed_limit: float,
) -> torch.Tensor:
    """Interpolate the arm reference with feedforward and actual-pose feedback.

    Corrections compare the measured palm with the reference at the *current*
    physics substep; feedforward carries it to the next one. Repeatedly chasing
    the endpoint with error/control_dt would under-travel when decimation > 1.
    """
    if control_dt <= 0 or decimation < 1 or not 0 <= substep < decimation:
        raise ValueError("Palm servo needs a positive timestep and a valid substep")
    position_delta = next_target[:, :3] - previous_target[:, :3]
    rotation_delta = quat_to_rotvec(
        quat_mul(next_target[:, 3:], quat_conjugate(previous_target[:, 3:]))
    )
    fraction = substep / decimation
    position_reference = previous_target[:, :3] + fraction * position_delta
    orientation_reference = quat_mul(
        rotvec_to_quat(fraction * rotation_delta), previous_target[:, 3:]
    )
    position_error = position_reference - measured_pose[:, :3]
    orientation_error = quat_to_rotvec(
        quat_mul(orientation_reference, quat_conjugate(measured_pose[:, 3:]))
    )
    return torch.cat(
        (
            clamp_norm(
                position_delta / control_dt + position_gain * position_error, linear_speed_limit
            ),
            clamp_norm(
                rotation_delta / control_dt + orientation_gain * orientation_error,
                angular_speed_limit,
            ),
        ),
        -1,
    )


def validate_inertias(mass: torch.Tensor, inertia: torch.Tensor) -> None:
    """Reject missing or physically impossible inertia before physics testing."""
    if not torch.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("Object masses must be finite and positive")
    matrix = inertia.reshape(-1, 3, 3)
    if not torch.isfinite(matrix).all():
        raise ValueError("Object inertia contains non-finite values")
    if not torch.allclose(matrix, matrix.transpose(-1, -2), atol=1e-7, rtol=1e-5):
        raise ValueError("Object inertia must be symmetric")
    moments = torch.linalg.eigvalsh(matrix)
    if (moments <= 0).any():
        raise ValueError("Object inertia must be positive definite")
    # A rigid body's principal moments obey the triangle inequality.
    if (moments[:, 2] > moments[:, 0] + moments[:, 1] + 1e-7).any():
        raise ValueError("Object inertia violates the principal-moment triangle inequality")


def mesh_mass_properties(
    vertices: np.ndarray, triangles: np.ndarray, mass: float
) -> tuple[np.ndarray, np.ndarray, str]:
    """Uniform-density COM and inertia from a closed mesh in rigid-body coordinates.

    RGB-D scans often have open or inconsistently wound surfaces. For those, use
    their convex solid hull, and report that approximation in artifact metadata.
    This also matches a convex collision representation more closely than an
    arbitrary inertia diagonal. Neither method introduces hand-set dimensions.
    """
    points = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("Mesh inertia needs at least four 3D vertices")
    if not np.isfinite(points).all() or not math.isfinite(mass) or mass <= 0:
        raise ValueError("Mesh vertices and positive mass must be finite")
    if faces.ndim != 2 or faces.shape[1] != 3 or (faces < 0).any() or (faces >= len(points)).any():
        raise ValueError("Mesh inertia needs valid triangle indices")
    method = "closed_mesh_uniform_density"
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered = np.sort(edges, axis=-1)
    _, ids, counts = np.unique(ordered, axis=0, return_inverse=True, return_counts=True)
    orientations = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    winding = np.bincount(ids, weights=orientations)
    closed = len(faces) > 0 and np.all(counts == 2) and np.all(winding == 0)
    if not closed:
        hull = ConvexHull(points)
        faces = hull.simplices.copy()
        tri = points[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
        faces[reverse] = faces[reverse][:, [0, 2, 1]]
        method = "convex_hull_uniform_density"
    # Recenter before tetrahedral integration to avoid cancellation on meshes
    # translated far from the link origin.
    shift = points.mean(axis=0)
    tri = points[faces] - shift
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6
    total_volume = volume.sum()
    if abs(total_volume) <= np.finfo(np.float64).eps:
        raise ValueError("Mesh has no solid volume for inertia computation")
    sums = tri.sum(axis=1)
    center_local = np.einsum("i,ij->j", volume, sums) / (4 * total_volume)
    second = np.einsum("nki,nkj->nij", tri, tri) + np.einsum("ni,nj->nij", sums, sums)
    covariance = np.einsum("n,nij->ij", volume, second) / (20 * total_volume)
    covariance -= np.outer(center_local, center_local)
    inertia = mass * (np.trace(covariance) * np.eye(3) - covariance)
    validate_inertias(torch.tensor([mass]), torch.as_tensor(inertia).reshape(1, 9))
    return center_local + shift, inertia, method


class StabilityAccumulator:
    """Each direction can fail only once; subsequent recovery cannot erase slip."""

    def __init__(self, count: int, duration_s: float, control_dt: float, device="cpu"):
        self.steps = math.ceil(duration_s / control_dt)
        self.duration_s = duration_s
        self.control_dt = control_dt
        self.alive = torch.ones(count, device=device, dtype=torch.bool)
        self.stable_time = torch.zeros(count, device=device)
        self.observations = 0

    def update(self, stable: torch.Tensor) -> None:
        if stable.shape != self.alive.shape or stable.dtype != torch.bool:
            raise ValueError("Stability updates need one Boolean per environment")
        if self.observations >= self.steps:
            raise ValueError("Stability duration has already been completed")
        dt = min(self.control_dt, self.duration_s - self.observations * self.control_dt)
        self.alive &= stable
        self.stable_time += self.alive * dt
        self.observations += 1

    @property
    def fractions(self) -> torch.Tensor:
        return (self.stable_time / self.duration_s).clamp(0, 1)
