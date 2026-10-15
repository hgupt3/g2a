"""Validated experiment parameters shared by discovery, optimization and PPO.

Defaults describe the supported LEAP simulator. User overrides live in JSON,
are copied into every artifact, and never require editing environment code.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .control import ControlSpec, ObservationSpec


@dataclass(frozen=True)
class SearchSpec:
    num_envs: int = 4096
    iterations: int = 8
    keep: int = 8
    setup_time_s: float = 3.0
    settle_time_s: float = 0.3
    spread_range_rad: tuple[float, float] = (-0.5, 0.5)
    thumb_opposition_range_rad: tuple[float, float] = (0.4, 2.0)
    thumb_initial_intermediate_rad: float = 0.0
    closing_speed_range_rad_s: tuple[float, float] = (0.3, 1.0)
    closing_torque_range_nm: tuple[float, float] = (0.05, 0.15)
    closing_torque_mode: str = "closing_direction"
    closing_contact_hold_s: float = 0.0
    closing_minimum_tool_contact_force_n: float = 0.0
    initial_std: float = 0.6
    warm_std: float = 0.2
    min_std: float = 0.05
    elite_fraction: float = 0.1
    smoothing: float = 0.7
    minimum_candidates_per_patch: int = 4
    uniform_exploration_fraction: float = 0.25
    trajectory_shortlist: int = 8
    minimum_relative_stability: float = 0.9
    max_wall_time_s: float = 400.0


@dataclass(frozen=True)
class WrenchSpec:
    duration_s: float = 2.0
    max_force_n: float = 50.0
    max_torque_nm: float = 3.0
    position_tolerance_m: float = 0.015
    angle_tolerance_rad: float = math.radians(5)
    ramp_power: float = 1.0
    directions_per_batch: int = 12
    restore_settle_time_s: float = 0.35
    clearance_m: float = 1.0


@dataclass(frozen=True)
class PhysicsSpec:
    fixture_mode: str = "fixed_joint"
    physics_dt_s: float = 1 / 120
    control_decimation: int = 2
    env_spacing_m: float = 2.0
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    stiffness_nm_rad: float = 4.0
    damping_nms_rad: float = 0.15
    joint_effort_limit_nm: float = 1.5
    joint_velocity_limit_rad_s: float = 12.0
    joint_position_tolerance_rad: float = 0.02
    contact_offset_m: float = 0.002
    max_depenetration_velocity: float = 1.0
    solver_position_iterations: int = 64
    solver_velocity_iterations: int = 8
    enable_external_forces_every_iteration: bool = True
    friction_static: float = 1.0
    friction_dynamic: float = 1.0
    restitution: float = 0.0
    floor_size_m: float = 1000.0
    floor_z_m: float = -0.01
    floor_thickness_m: float = 0.02
    rl_setup_time_s: float = 0.35
    wrist_position_gain_s_inv: float = 30.0
    wrist_orientation_gain_s_inv: float = 30.0


@dataclass(frozen=True)
class RandomizationSpec:
    enabled: bool = True
    friction_buckets: int = 64
    mass_range: tuple[float, float] = (0.7, 1.3)
    friction_range: tuple[float, float] = (0.7, 1.3)
    load_range: tuple[float, float] = (0.7, 1.3)


@dataclass(frozen=True)
class ResourceSpec:
    """Simulator allocation capacity; these settings do not change dynamics."""

    gpu_rigid_patches_per_env: int = 256
    tool_contact_points_per_filter: int = 32


@dataclass(frozen=True)
class CollisionSpec:
    """Geometric acceptance budgets, measured on the actual cooked shapes.

    Defaults use a one-millimeter profile. An explicit
    experiment can allow small contact overlap and use a larger rejection
    limit for severe penetration. Contact generation offsets are independent
    of this geometric acceptance decision. Both tolerances may be zero.
    Chunking changes only audit memory use and does not relax either gate.
    """

    max_tool_penetration_m: float = 0.001
    max_self_penetration_m: float = 0.001
    pair_chunk_size: int = 4096


@dataclass(frozen=True)
class RewardSpec:
    tracking_kernel: str = "inverse_quadratic"
    position_scale_m: float = 0.01
    angle_scale_rad: float = 0.1
    position_weight: float = 1.5
    angle_weight: float = 1.0
    progress_reward: float = 0.25
    action_rate_penalty: float = 0.01
    effort_penalty: float = 0.001
    action_penalty: float = 0.0005
    failure_penalty: float = 5.0
    completion_bonus: float = 5.0
    # Optional dense feedback from the peak native overlap in the current
    # control interval. This does not replace the hard collision termination.
    collision_margin_penalty: float = 0.0
    collision_soft_budget_fraction: float = 0.5


@dataclass(frozen=True)
class EvaluationSpec:
    position_tolerance_m: float = 0.03
    angle_tolerance_rad: float = 0.35
    failure_position_m: float = 0.15
    failure_angle_rad: float = math.pi / 2
    num_envs: int = 32
    batches: int = 2
    seed: int = 42
    max_wall_time_s: float = 300.0
    verification_max_wall_time_s: float = 180.0


@dataclass(frozen=True)
class TrainingSpec:
    num_envs: int = 4096
    epochs: int = 1200
    horizon: int = 32
    random_start: bool | None = None
    max_wall_time_s: float = 1200.0
    final_validation_reserve_s: float | None = None
    checkpoint_every_epochs: int | None = None
    validation_every_epochs: int = 100
    validation_batches: int = 2
    validation_num_envs: int | None = 64
    validation_parallel_batches: bool | None = None
    validation_seed: int = 1000
    validation_max_wall_time_s: float = 180.0
    learning_rate: float = 0.0001
    learning_rate_schedule: str = "fixed"
    learning_rate_min: float = 1e-6
    learning_rate_max: float = 0.0003
    actor_critic_separate: bool = True
    zero_residual_regularization: float = 0.001
    critic_loss_coefficient: float = 1.0
    initial_log_std: float = -1.6
    initial_wrist_log_std: float | None = None
    initial_finger_log_std: float | None = None
    reset_exploration_on_resume: bool = False
    minimum_log_std: float = -3.0
    maximum_log_std: float = 0.0
    entropy_coefficient: float = 0.0
    mini_epochs: int = 4
    network_units: tuple[int, ...] = (256, 128)
    minibatch_size: int = 16384
    early_stopping_patience: int = 4
    minimum_epochs: int = 100
    selection_tracking_mode: str = "componentwise"
    selection_min_tracking_gain: float = 0.05
    selection_position_relative_tolerance: float = 0.0
    selection_angle_relative_tolerance: float = 0.0
    selection_success_tolerance: float = 0.0
    selection_completion_tolerance: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    kl_threshold: float = 0.008
    gradient_clip: float = 1.0
    clip_ratio: float = 0.15


SECTIONS = {
    "control": ControlSpec,
    "observation": ObservationSpec,
    "search": SearchSpec,
    "wrench": WrenchSpec,
    "physics": PhysicsSpec,
    "randomization": RandomizationSpec,
    "resources": ResourceSpec,
    "collision": CollisionSpec,
    "reward": RewardSpec,
    "evaluation": EvaluationSpec,
    "training": TrainingSpec,
}


@dataclass(frozen=True)
class ExperimentSpec:
    control: ControlSpec = field(default_factory=ControlSpec)
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    search: SearchSpec = field(default_factory=SearchSpec)
    wrench: WrenchSpec = field(default_factory=WrenchSpec)
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    randomization: RandomizationSpec = field(default_factory=RandomizationSpec)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    collision: CollisionSpec = field(default_factory=CollisionSpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)

    @property
    def control_dt(self) -> float:
        return self.physics.physics_dt_s * self.physics.control_decimation

    def physics_fingerprint(self) -> str:
        """Fixed-grasp physics and initialization, excluding closure-only knobs."""
        physics = asdict(self.physics)
        # This tolerance checks saved joint states; it does not drive the motors.
        physics.pop("joint_position_tolerance_rad")
        payload = {
            "physics": physics,
            # Both search and RL use this duration after releasing the fixture.
            "initialization": {"free_settle_time_s": float(self.search.settle_time_s)},
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def grasp_fingerprint(self) -> str:
        search = asdict(self.search)
        for key in (
            "num_envs",
            "iterations",
            "keep",
            "initial_std",
            "warm_std",
            "min_std",
            "elite_fraction",
            "smoothing",
            "minimum_candidates_per_patch",
            "uniform_exploration_fraction",
            "trajectory_shortlist",
            "minimum_relative_stability",
            "max_wall_time_s",
        ):
            search.pop(key)
        physics = asdict(self.physics)
        physics.pop(
            "joint_position_tolerance_rad"
        )  # validates artifacts, not construction dynamics
        collision = asdict(self.collision)
        collision.pop("pair_chunk_size")  # changes audit allocation, not acceptance
        payload = {
            "physics": physics,
            "closure": search,
            "wrench": asdict(self.wrench),
            "collision": collision,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ExperimentSpec":
        if path is None:
            result = cls()
        else:
            with Path(path).expanduser().open() as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise ValueError("Experiment configuration must be a JSON object")
            unknown = set(data) - set(SECTIONS)
            if unknown:
                raise ValueError(f"Unknown experiment sections: {sorted(unknown)}")
            sections = {}
            for name, values in data.items():
                if not isinstance(values, dict):
                    raise ValueError(f"{name} must be a JSON object")
                defaults = SECTIONS[name]()
                allowed = {item.name for item in fields(defaults)}
                extra = set(values) - allowed
                if extra:
                    raise ValueError(f"Unknown {name} settings: {sorted(extra)}")
                converted = {}
                for key, value in values.items():
                    if isinstance(getattr(defaults, key), tuple):
                        if not isinstance(value, (list, tuple)):
                            raise ValueError(f"{name}.{key} must be an array")
                        value = tuple(value)
                    converted[key] = value
                sections[name] = SECTIONS[name](**converted)
            result = cls(**sections)
        result.validate()
        return result

    def to_dict(self) -> dict:
        # JSON types, independent of whether parameters were loaded or constructed.
        payload = asdict(self)
        # An absent reporting override keeps saved configuration hashes stable.
        if (
            self.resources.tool_contact_points_per_filter
            == ResourceSpec().tool_contact_points_per_filter
        ):
            payload["resources"].pop("tool_contact_points_per_filter")
        # Omit unset overrides so loading and resaving a profile is stable.
        for name in (
            "random_start",
            "initial_wrist_log_std",
            "initial_finger_log_std",
            "final_validation_reserve_s",
            "checkpoint_every_epochs",
            "validation_parallel_batches",
        ):
            if payload["training"][name] is None:
                payload["training"].pop(name)
        if self.training.selection_tracking_mode == "componentwise":
            payload["training"].pop("selection_tracking_mode")
        # An inactive default collision cost preserves earlier experiment
        # hashes. Explicit nondefault settings remain round-trippable.
        if (
            self.reward.collision_margin_penalty == 0
            and self.reward.collision_soft_budget_fraction == 0.5
        ):
            payload["reward"].pop("collision_margin_penalty")
            payload["reward"].pop("collision_soft_budget_fraction")
        return json.loads(json.dumps(payload, allow_nan=False))

    def validate(self) -> None:
        for section in SECTIONS:
            defaults = SECTIONS[section]()
            values = asdict(getattr(self, section))
            for name, value in values.items():
                expected = getattr(defaults, name)
                if section == "training" and name in (
                    "validation_num_envs",
                    "checkpoint_every_epochs",
                ):
                    valid = value is None or (
                        isinstance(value, int) and not isinstance(value, bool)
                    )
                elif section == "training" and name in (
                    "random_start",
                    "validation_parallel_batches",
                ):
                    valid = value is None or isinstance(value, bool)
                elif section == "training" and name in (
                    "initial_wrist_log_std",
                    "initial_finger_log_std",
                    "final_validation_reserve_s",
                ):
                    valid = value is None or (
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                    )
                elif isinstance(expected, bool):
                    valid = isinstance(value, bool)
                elif isinstance(expected, str):
                    valid = isinstance(value, str)
                elif isinstance(expected, tuple):
                    integer_items = all(isinstance(item, int) for item in expected)
                    valid = isinstance(value, tuple) and all(
                        not isinstance(item, bool)
                        and isinstance(item, int if integer_items else (float, int))
                        for item in value
                    )
                elif isinstance(expected, int):
                    valid = isinstance(value, int) and not isinstance(value, bool)
                else:
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                if not valid:
                    raise ValueError(f"{section}.{name} has an invalid type")
                numbers = value if isinstance(value, tuple) else (value,)
                for number in numbers:
                    if isinstance(number, (int, float)) and not math.isfinite(number):
                        raise ValueError(f"{section}.{name} must be finite")
                if name.endswith("_range") or "_range_" in name:
                    if len(value) != 2 or value[0] > value[1]:
                        raise ValueError(f"{section}.{name} must be an ordered pair")
        self.control.validate()
        self.observation.validate()
        if self.physics.fixture_mode != "fixed_joint":
            raise ValueError("physics.fixture_mode must be fixed_joint")
        positive = {
            "search": (
                "num_envs",
                "iterations",
                "keep",
                "setup_time_s",
                "initial_std",
                "warm_std",
                "min_std",
                "max_wall_time_s",
                "trajectory_shortlist",
                "minimum_candidates_per_patch",
            ),
            "wrench": (
                "duration_s",
                "max_force_n",
                "max_torque_nm",
                "position_tolerance_m",
                "angle_tolerance_rad",
                "ramp_power",
            ),
            "physics": (
                "physics_dt_s",
                "control_decimation",
                "env_spacing_m",
                "stiffness_nm_rad",
                "joint_effort_limit_nm",
                "joint_velocity_limit_rad_s",
                "contact_offset_m",
                "max_depenetration_velocity",
                "solver_position_iterations",
                "solver_velocity_iterations",
                "floor_size_m",
                "floor_thickness_m",
                "rl_setup_time_s",
                "wrist_position_gain_s_inv",
                "wrist_orientation_gain_s_inv",
            ),
            "reward": ("position_scale_m", "angle_scale_rad", "position_weight", "angle_weight"),
            "evaluation": (
                "position_tolerance_m",
                "angle_tolerance_rad",
                "failure_position_m",
                "failure_angle_rad",
                "num_envs",
                "batches",
                "max_wall_time_s",
                "verification_max_wall_time_s",
            ),
            "training": (
                "num_envs",
                "epochs",
                "horizon",
                "max_wall_time_s",
                "validation_every_epochs",
                "validation_batches",
                "validation_max_wall_time_s",
                "learning_rate",
                "learning_rate_min",
                "learning_rate_max",
                "critic_loss_coefficient",
                "mini_epochs",
                "minibatch_size",
                "early_stopping_patience",
                "minimum_epochs",
                "kl_threshold",
                "gradient_clip",
                "clip_ratio",
            ),
            "randomization": ("friction_buckets",),
            "resources": ("gpu_rigid_patches_per_env", "tool_contact_points_per_filter"),
            "collision": ("pair_chunk_size",),
        }
        for section, names in positive.items():
            for name in names:
                value = getattr(getattr(self, section), name)
                if isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{section}.{name} must be positive")
        if self.training.validation_num_envs is not None and self.training.validation_num_envs <= 0:
            raise ValueError("training.validation_num_envs must be positive or null")
        if (
            self.training.validation_parallel_batches is True
            and self.training.validation_num_envs is None
        ):
            raise ValueError("training.validation_parallel_batches requires validation_num_envs")
        cadence = self.training.checkpoint_every_epochs
        if cadence is not None and cadence <= 0:
            raise ValueError("training.checkpoint_every_epochs must be positive or null")
        reserve = self.training.final_validation_reserve_s
        if reserve is not None and not 0 < reserve < self.training.max_wall_time_s:
            raise ValueError(
                "training.final_validation_reserve_s must be positive and less than max_wall_time_s"
            )
        for section, names in {
            "search": (
                "settle_time_s",
                "closing_contact_hold_s",
                "closing_minimum_tool_contact_force_n",
            ),
            "wrench": ("restore_settle_time_s", "clearance_m"),
            "physics": (
                "damping_nms_rad",
                "joint_position_tolerance_rad",
                "friction_static",
                "friction_dynamic",
                "restitution",
            ),
            "reward": (
                "progress_reward",
                "action_rate_penalty",
                "effort_penalty",
                "action_penalty",
                "failure_penalty",
                "completion_bonus",
                "collision_margin_penalty",
            ),
            "training": (
                "entropy_coefficient",
                "zero_residual_regularization",
                "selection_min_tracking_gain",
                "selection_position_relative_tolerance",
                "selection_angle_relative_tolerance",
                "selection_success_tolerance",
                "selection_completion_tolerance",
            ),
            "collision": ("max_tool_penetration_m", "max_self_penetration_m"),
        }.items():
            for name in names:
                if getattr(getattr(self, section), name) < 0:
                    raise ValueError(f"{section}.{name} cannot be negative")
        if self.reward.tracking_kernel not in ("gaussian", "inverse_quadratic"):
            raise ValueError("reward.tracking_kernel must be gaussian or inverse_quadratic")
        if self.training.selection_tracking_mode not in ("componentwise", "combined"):
            raise ValueError("training.selection_tracking_mode must be componentwise or combined")
        if not 0 <= self.reward.collision_soft_budget_fraction < 1:
            raise ValueError("reward.collision_soft_budget_fraction must be in [0, 1)")
        if self.reward.collision_margin_penalty > 0 and (
            self.collision.max_tool_penetration_m <= 0 or self.collision.max_self_penetration_m <= 0
        ):
            raise ValueError("reward.collision_margin_penalty requires positive collision budgets")
        if (
            not self.training.minimum_log_std
            <= self.training.initial_log_std
            <= self.training.maximum_log_std
        ):
            raise ValueError(
                "training exploration requires minimum_log_std <= initial_log_std <= maximum_log_std"
            )
        for name in ("initial_wrist_log_std", "initial_finger_log_std"):
            value = getattr(self.training, name)
            if value is not None and not (
                self.training.minimum_log_std <= value <= self.training.maximum_log_std
            ):
                raise ValueError(
                    f"training.{name} must lie between minimum_log_std and maximum_log_std"
                )
        for name in (
            "selection_min_tracking_gain",
            "selection_position_relative_tolerance",
            "selection_angle_relative_tolerance",
        ):
            if getattr(self.training, name) >= 1:
                raise ValueError(f"training.{name} must be less than one")
        if not 0 < self.search.elite_fraction <= 1 or not 0 < self.search.smoothing <= 1:
            raise ValueError("CEM elite_fraction and smoothing must be in (0, 1]")
        if self.search.minimum_candidates_per_patch < 4:
            raise ValueError("search.minimum_candidates_per_patch must be at least four")
        if self.search.num_envs < 4:
            raise ValueError("search.num_envs must be at least four")
        if not 0 < self.search.uniform_exploration_fraction <= 0.5:
            raise ValueError("search.uniform_exploration_fraction must be in (0, 0.5]")
        if self.search.closing_torque_mode not in ("closing_direction", "absolute", "tool_contact"):
            raise ValueError(
                "search.closing_torque_mode must be closing_direction, absolute or tool_contact"
            )
        if not 0 < self.search.minimum_relative_stability <= 1:
            raise ValueError("search.minimum_relative_stability must be in (0, 1]")
        if not 1 <= self.wrench.directions_per_batch <= 12:
            raise ValueError("wrench.directions_per_batch must be in [1, 12]")
        if len(self.physics.gravity_m_s2) != 3:
            raise ValueError("physics.gravity_m_s2 must be a three-vector")
        if (
            self.search.closing_torque_range_nm[0] <= 0
            or self.search.closing_speed_range_rad_s[0] <= 0
        ):
            raise ValueError("Closing speeds and torque thresholds must be positive")
        if self.search.closing_torque_range_nm[1] > self.physics.joint_effort_limit_nm:
            raise ValueError("Closing torque threshold exceeds the actuator effort limit")
        if self.physics.friction_dynamic > self.physics.friction_static:
            raise ValueError("Dynamic friction cannot exceed static friction")
        if self.physics.restitution > 1:
            raise ValueError("Restitution must be in [0, 1]")
        for name in ("mass_range", "friction_range", "load_range"):
            if getattr(self.randomization, name)[0] <= 0:
                raise ValueError(f"randomization.{name} must be positive")
        if not self.training.network_units or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in self.training.network_units
        ):
            raise ValueError("training.network_units must contain positive integers")
        for name in ("gamma", "gae_lambda", "clip_ratio"):
            if not 0 < getattr(self.training, name) <= 1:
                raise ValueError(f"training.{name} must be in (0, 1]")
        if self.training.learning_rate_schedule not in ("adaptive", "fixed"):
            raise ValueError("training.learning_rate_schedule must be adaptive or fixed")
        if (
            not self.training.learning_rate_min
            <= self.training.learning_rate
            <= self.training.learning_rate_max
        ):
            raise ValueError(
                "training learning_rate must lie between learning_rate_min and learning_rate_max"
            )
        for section, name in (("evaluation", "seed"), ("training", "validation_seed")):
            if not 0 <= getattr(getattr(self, section), name) < 2**32:
                raise ValueError(f"{section}.{name} must lie in [0, 2**32)")
