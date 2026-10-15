"""The physical action and observation contract shared by simulation and inference.

The default policy adapts finger targets around a discovered grasp. The arm
replays the nominal palm trajectory. Wrist residuals are an explicit ablation.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .math import (
    clamp_norm,
    compose_pose,
    pose_features,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_to_rotvec,
    relative_pose,
    rotvec_to_quat,
)

# Preserve the physical articulation order (lexicographic order puts a_10 before
# a_2). Position and effort limits come from the loaded physical hand asset.
JOINT_NAMES = tuple(f"a_{i}" for i in range(16))
JOINT_COUNT = len(JOINT_NAMES)
VECTOR_DIM = 3
ROTATION_FEATURE_DIM = 6
POSE_FEATURE_DIM = VECTOR_DIM + ROTATION_FEATURE_DIM
TWIST_DIM = 2 * VECTOR_DIM
WRENCH_DIM = 2 * VECTOR_DIM


@dataclass(frozen=True)
class ControlSpec:
    action_mode: str = "finger_residual"
    wrist_translation: float = 0.025  # maximum offset norm, meters in nominal palm frame
    wrist_rotation: float = 0.2  # maximum rotation vector norm, radians
    joint_residual: float = 0.1  # radians around the optimized joint targets
    joint_speed: float = 2.0  # maximum commanded target change, rad/s
    action_filter: float = 0.3  # interpolation coefficient at the control timestep
    wrist_linear_speed: float = 2.0
    wrist_angular_speed: float = 10.0

    @property
    def wrist_residual(self) -> bool:
        return self.action_mode == "wrist_finger_residual"

    @property
    def action_dim(self) -> int:
        return JOINT_COUNT + (TWIST_DIM if self.wrist_residual else 0)

    @property
    def finger_action_start(self) -> int:
        return TWIST_DIM if self.wrist_residual else 0

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.action_mode not in ("finger_residual", "wrist_finger_residual"):
            raise ValueError("action_mode must be finger_residual or wrist_finger_residual")
        for name, value in asdict(self).items():
            if name != "action_mode" and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"Control scale {name} must be finite and positive")
        if self.action_filter > 1:
            raise ValueError("action_filter must be in (0, 1]")


@dataclass(frozen=True)
class ObservationSpec:
    """Physical divisors and anticipation horizon, saved with each policy.

    Torques are estimated applied actuator torques. Their utilization is a load
    proxy, not a tactile measurement. Wrenches are the known or estimated external
    task wrench in world coordinates, excluding gravity and unobserved contacts.
    Physics context must use nominal/estimated, not privileged hidden parameters.
    """

    position_scale_m: float = 0.5
    relative_position_scale_m: float = 0.05
    joint_velocity_scale_rad_s: float = 12.0
    joint_torque_scale_nm: float = 1.5
    linear_velocity_scale_m_s: float = 1.0
    angular_velocity_scale_rad_s: float = 10.0
    force_scale_n: float = 10.0
    torque_scale_nm: float = 1.0
    mass_scale_kg: float = 1.0
    friction_scale: float = 1.0
    load_multiplier_scale: float = 1.0
    future_horizon_s: float = 0.1
    clip: float = 10.0
    tracking_features: bool = False
    tracking_position_scale_m: float = 0.01
    tracking_angle_scale_rad: float = 0.1

    def to_dict(self) -> dict:
        result = asdict(self)
        if not self.tracking_features:
            # Keep the base observation contract independent of optional
            # tracking features and their normalization scales.
            for name in (
                "tracking_features",
                "tracking_position_scale_m",
                "tracking_angle_scale_rad",
            ):
                result.pop(name)
        return result

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "tracking_features":
                if not isinstance(value, bool):
                    raise ValueError("tracking_features must be Boolean")
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Observation scale {name} must be finite and positive")


DEFAULT_CONTROL = ControlSpec()
DEFAULT_OBSERVATION = ObservationSpec()


def observation_fields(
    spec: ObservationSpec = DEFAULT_OBSERVATION, control: ControlSpec = DEFAULT_CONTROL
) -> tuple[tuple[str, int], ...]:
    """Ordered fields; widths are derived from the physical contract."""
    fields = (
        ("joint_position", JOINT_COUNT),
        ("joint_velocity", JOINT_COUNT),
        ("joint_target", JOINT_COUNT),
        ("previous_action", control.action_dim),
        ("joint_torque", JOINT_COUNT),
        ("actuator_load_proxy", JOINT_COUNT),
        ("object_pose", POSE_FEATURE_DIM),
        ("object_velocity", TWIST_DIM),
        ("palm_pose", POSE_FEATURE_DIM),
        ("palm_velocity", TWIST_DIM),
        ("object_to_goal", POSE_FEATURE_DIM),
        ("goal_velocity", TWIST_DIM),
        ("object_to_future_goal", POSE_FEATURE_DIM),
        ("current_task_wrench", WRENCH_DIM),
        ("expected_task_wrench", WRENCH_DIM),
        ("mass_friction_load", VECTOR_DIM),
        ("phase_and_setup", VECTOR_DIM),
    )
    if spec.tracking_features:
        fields += (
            ("tracking_com_error_palm", VECTOR_DIM),
            ("tracking_orientation_error_palm", VECTOR_DIM),
            ("measured_grip_wrist_offset", TWIST_DIM),
        )
    return fields


def observation_dim(
    spec: ObservationSpec = DEFAULT_OBSERVATION, control: ControlSpec = DEFAULT_CONTROL
) -> int:
    return sum(size for _, size in observation_fields(spec, control))


# Aliases describe the default contract; explicit wrist policies derive theirs.
ACTION_DIM = DEFAULT_CONTROL.action_dim
OBSERVATION_FIELDS = observation_fields()
OBSERVATION_DIM = observation_dim()


def joint_features(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2 * (q - lower) / (upper - lower).clamp_min(torch.finfo(q.dtype).eps) - 1


def tracking_features(
    object_pose: torch.Tensor,
    palm_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    nominal_grasp: torch.Tensor,
    center_of_mass_in_body: torch.Tensor,
    spec: ObservationSpec,
    control: ControlSpec,
) -> torch.Tensor:
    """Measured error in the frame of the policy's absolute wrist offsets.

    COM tracking matches the reward. The grasp offset separately exposes the
    absolute correction needed for the acquired contact configuration, rather
    than requiring the network to infer it from three world-frame poses.
    Neither field contains hidden mass/friction/load randomization.
    """
    nominal_grasp = torch.broadcast_to(nominal_grasp, goal_pose.shape)
    center_of_mass_in_body = torch.broadcast_to(center_of_mass_in_body, object_pose[:, :3].shape)
    nominal_palm = compose_pose(goal_pose, nominal_grasp)
    palm_inverse = quat_conjugate(nominal_palm[:, 3:])
    actual_com = object_pose[:, :3] + quat_apply(object_pose[:, 3:], center_of_mass_in_body)
    goal_com = goal_pose[:, :3] + quat_apply(goal_pose[:, 3:], center_of_mass_in_body)
    com_error = quat_apply(palm_inverse, goal_com - actual_com)
    orientation_error = quat_to_rotvec(
        quat_mul(goal_pose[:, 3:], quat_conjugate(object_pose[:, 3:]))
    )
    orientation_error = quat_apply(palm_inverse, orientation_error)
    grip_delta = relative_pose(nominal_grasp, relative_pose(object_pose, palm_pose))
    return torch.cat(
        (
            com_error / spec.tracking_position_scale_m,
            orientation_error / spec.tracking_angle_scale_rad,
            grip_delta[:, :3] / control.wrist_translation,
            quat_to_rotvec(grip_delta[:, 3:]) / control.wrist_rotation,
        ),
        -1,
    )


def tracking_score(error: torch.Tensor, scale: float, kernel: str) -> torch.Tensor:
    """Positive tracking credit with a configurable physical error scale."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Tracking scale must be finite and positive")
    normalized_squared = (error / scale).square()
    if kernel == "inverse_quadratic":
        return 1 / (1 + normalized_squared)
    if kernel == "gaussian":
        return torch.exp(-normalized_squared)
    raise ValueError("Tracking kernel must be inverse_quadratic or gaussian")


def action_targets(
    action: torch.Tensor,
    nominal_palm: torch.Tensor,
    grasp_targets: torch.Tensor,
    previous_targets: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    spec: ControlSpec,
    control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded absolute residuals: holding an action never integrates target drift.

    Action smoothing happens in the environment. This function bounds finger
    target rate. Wrist velocity limits are applied by the arm servo and its
    offset is bounded in the nominal palm's local frame.
    """
    if action.ndim != 2 or action.shape[-1] != spec.action_dim:
        raise ValueError(f"Expected {spec.action_dim} actions for {spec.action_mode}")
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    action = action.clamp(-1, 1)
    if spec.wrist_residual:
        offset = torch.cat(
            (
                clamp_norm(action[:, :VECTOR_DIM] * spec.wrist_translation, spec.wrist_translation),
                rotvec_to_quat(
                    clamp_norm(
                        action[:, VECTOR_DIM:TWIST_DIM] * spec.wrist_rotation, spec.wrist_rotation
                    )
                ),
            ),
            -1,
        )
        palm = compose_pose(nominal_palm, offset)
    else:
        palm = nominal_palm
    target = (grasp_targets + spec.joint_residual * action[:, spec.finger_action_start :]).clamp(
        lower, upper
    )
    max_delta = spec.joint_speed * control_dt
    joints = previous_targets + (target - previous_targets).clamp(-max_delta, max_delta)
    return palm, joints.clamp(lower, upper)


def observation(
    q: torch.Tensor,
    qd: torch.Tensor,
    targets: torch.Tensor,
    previous_action: torch.Tensor,
    object_pose: torch.Tensor,
    object_velocity: torch.Tensor,
    palm_pose: torch.Tensor,
    palm_velocity: torch.Tensor,
    goal_pose: torch.Tensor,
    goal_velocity: torch.Tensor,
    future_goal_pose: torch.Tensor,
    phase: torch.Tensor,
    setup: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    joint_torque: torch.Tensor,
    joint_effort_limits: torch.Tensor,
    current_wrench: torch.Tensor,
    expected_wrench: torch.Tensor,
    mass: torch.Tensor,
    friction: torch.Tensor,
    load_multiplier: torch.Tensor,
    spec: ObservationSpec = DEFAULT_OBSERVATION,
    control: ControlSpec = DEFAULT_CONTROL,
    nominal_grasp: torch.Tensor | None = None,
    center_of_mass_in_body: torch.Tensor | None = None,
) -> torch.Tensor:
    if previous_action.shape[-1] != control.action_dim:
        raise ValueError(f"Expected {control.action_dim} previous-action features")

    def pose(p: torch.Tensor, scale: float) -> torch.Tensor:
        features = pose_features(p)
        return torch.cat((features[:, :VECTOR_DIM] / scale, features[:, VECTOR_DIM:]), -1)

    def twist(v: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                v[:, :VECTOR_DIM] / spec.linear_velocity_scale_m_s,
                v[:, VECTOR_DIM:] / spec.angular_velocity_scale_rad_s,
            ),
            -1,
        )

    def wrench(w: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (w[:, :VECTOR_DIM] / spec.force_scale_n, w[:, VECTOR_DIM:] / spec.torque_scale_nm), -1
        )

    features = (
        joint_features(q, lower, upper),
        qd / spec.joint_velocity_scale_rad_s,
        joint_features(targets, lower, upper),
        previous_action,
        joint_torque / spec.joint_torque_scale_nm,
        joint_torque.abs() / joint_effort_limits.clamp_min(torch.finfo(q.dtype).eps),
        pose(object_pose, spec.position_scale_m),
        twist(object_velocity),
        pose(palm_pose, spec.position_scale_m),
        twist(palm_velocity),
        pose(relative_pose(object_pose, goal_pose), spec.relative_position_scale_m),
        twist(goal_velocity),
        pose(relative_pose(object_pose, future_goal_pose), spec.relative_position_scale_m),
        wrench(current_wrench),
        wrench(expected_wrench),
        torch.stack(
            (
                mass / spec.mass_scale_kg,
                friction / spec.friction_scale,
                load_multiplier / spec.load_multiplier_scale,
            ),
            -1,
        ),
        torch.stack(
            (torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase), setup.float()),
            -1,
        ),
    )
    if spec.tracking_features:
        if nominal_grasp is None or center_of_mass_in_body is None:
            raise ValueError("Tracking observations require the nominal grasp and known body COM")
        features += (
            tracking_features(
                object_pose,
                palm_pose,
                goal_pose,
                nominal_grasp,
                center_of_mass_in_body,
                spec,
                control,
            ),
        )
    return torch.cat(features, -1)
