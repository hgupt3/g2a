"""Shared physical environment for area grasp discovery and residual PPO.

Import only after launching Isaac Sim. A virtual fixture supports the object
while fingers close; it is then released for a separate, freely settling phase.
The arm carries the palm through a velocity servo. Tool loads act at their
functional contact point, including its velocity and force-induced moment.
"""

import gc
import math
import time
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from gymnasium import spaces
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation

from .artifacts import load_grasp
from .collision_audit import CollisionAudit, collision_margin_cost
from .config import ExperimentSpec
from .contacts import empty_opposition_geometry, opposition_contact_geometry
from .control import (
    ACTION_DIM,
    JOINT_NAMES,
    OBSERVATION_DIM,
    ControlSpec,
    action_targets,
    observation,
    observation_dim,
    tracking_score,
)
from .math import compose_pose, quat_angle, quat_apply, relative_pose
from .physics import gpu_rigid_patch_capacity, root_link_velocity
from .stress import (
    SEARCH_DIM,
    WRENCH_DIRECTIONS,
    ClosingTorqueStopper,
    ExternalWrenchWriter,
    StabilityAccumulator,
    bounded_contact_stop_targets,
    decode_closing,
    joint_descendants_from_usd,
    mesh_mass_properties,
    palm_servo_velocity,
    point_load_wrench,
    ramp_wrenches,
    sample_physics_multipliers,
    stable_mask,
    tool_contact_mask,
    validate_inertias,
)
from .tasks import RESOURCE_ROOT, TaskSpec
from .trajectory import Trajectory


@configclass
class G2AEnvCfg(DirectRLEnvCfg):
    task: str = "hammer"
    mode: str = "rl"
    grasp_path: str | None = None
    grasp_index: int = 0
    random_start: bool = False
    randomize_forces: bool = False  # enables mass, friction AND load randomization
    paired_evaluation: bool = False
    parallel_evaluation_seeds: tuple[int, ...] = ()
    experiment: ExperimentSpec = ExperimentSpec()
    control: ControlSpec = ControlSpec()
    hand_asset: str = str(RESOURCE_ROOT / "assets/leap_hand/leap.usd")
    decimation = ExperimentSpec().physics.control_decimation
    episode_length_s = 45.0  # replaced by the actual recording duration before construction
    action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,))
    observation_space = OBSERVATION_DIM
    state_space = 0
    is_finite_horizon = True
    sim: SimulationCfg = SimulationCfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, replicate_physics=True)
    robot_cfg: ArticulationCfg | None = None
    object_cfg: RigidObjectCfg | None = None


def _check_deadline(deadline: float | None, operation: str) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError(f"{operation} wall-time limit reached")


def _author_tool_contact_reporting(stage, path: str) -> dict:
    """Add force reporting without SDK activation's sleep-threshold override."""
    from pxr import PhysxSchema, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Tool contact reporting requires its source rigid-body primitive")
    sleep = prim.GetAttribute("physxRigidBody:sleepThreshold")
    before = (sleep.Get(), sleep.HasAuthoredValueOpinion()) if sleep else (None, False)
    report = (
        PhysxSchema.PhysxContactReportAPI.Get(stage, path)
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
        else PhysxSchema.PhysxContactReportAPI.Apply(prim)
    )
    report.CreateThresholdAttr().Set(0.0)
    sleep = prim.GetAttribute("physxRigidBody:sleepThreshold")
    after = (sleep.Get(), sleep.HasAuthoredValueOpinion()) if sleep else (None, False)
    if after != before:
        raise RuntimeError("Contact reporting unexpectedly changed the authored sleep settings")
    return {
        "method": "report_only_PhysxContactReportAPI_before_clone",
        "force_report_threshold_n": 0.0,
        "sleep_threshold_value": before[0],
        "sleep_threshold_authored": bool(before[1]),
        "sleep_settings_preserved": True,
    }


class G2AEnv(DirectRLEnv):
    cfg: G2AEnvCfg

    def __init__(self, cfg: G2AEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.mode not in ("search", "rl"):
            raise ValueError("mode must be search or rl")
        if cfg.paired_evaluation and (cfg.mode != "rl" or cfg.scene.num_envs % 2):
            raise ValueError(
                "Paired evaluation requires RL mode and an even physical environment count"
            )
        self._parallel_evaluation_plan = None
        self.parallel_evaluation_initial_physics = None
        if cfg.parallel_evaluation_seeds:
            from .evaluation_batches import PairedBatchPlan

            batches = len(cfg.parallel_evaluation_seeds)
            if not cfg.paired_evaluation or cfg.random_start or cfg.scene.num_envs % (2 * batches):
                raise ValueError(
                    "Parallel seed batches require paired full trajectories and divisible physical groups"
                )
            self._parallel_evaluation_plan = PairedBatchPlan(
                cfg.parallel_evaluation_seeds, cfg.scene.num_envs // (2 * batches)
            )
        if cfg.mode == "search" and cfg.randomize_forces:
            raise ValueError(
                "Grasp search requires nominal physics so each wrench direction restores identical conditions; "
                "enable mass/friction/load randomization during RL training or evaluation"
            )
        if cfg.mode == "rl" and cfg.grasp_path is None:
            raise ValueError("Run g2a optimize first and provide --grasp")
        cfg.experiment.validate()
        cfg.control = cfg.experiment.control
        physics = cfg.experiment.physics
        cfg.decimation = physics.control_decimation
        cfg.sim.dt = physics.physics_dt_s
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.gravity = physics.gravity_m_s2
        cfg.sim.physx.enable_external_forces_every_iteration = (
            physics.enable_external_forces_every_iteration
        )
        patches_per_env = cfg.experiment.resources.gpu_rigid_patches_per_env
        cfg.sim.physx.gpu_max_rigid_patch_count = gpu_rigid_patch_capacity(
            cfg.scene.num_envs, patches_per_env, cfg.sim.physx.gpu_max_rigid_patch_count
        )
        print(
            f"PhysX contact allocation: {cfg.sim.physx.gpu_max_rigid_patch_count:,} rigid "
            f"patches for {cfg.scene.num_envs:,} environments "
            f"({patches_per_env:,} patches/environment budget)",
            flush=True,
        )
        cfg.scene.env_spacing = physics.env_spacing_m
        cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=physics.friction_static,
            dynamic_friction=physics.friction_dynamic,
            restitution=physics.restitution,
        )
        cfg.action_space = spaces.Box(-1.0, 1.0, shape=(cfg.control.action_dim,))
        cfg.observation_space = observation_dim(cfg.experiment.observation, cfg.control)
        self.task = TaskSpec.load(cfg.task)
        self.trajectory = Trajectory.load(self.task, cfg.experiment.control_dt, cfg.sim.device)
        from .regions import GraspArea

        self.area = GraspArea.load(
            self.task.area_path,
            object_asset=self.task.object_asset,
            body_offset=self.task.body_offset,
            hand_asset=Path(cfg.hand_asset),
        )
        functional = self.area.functional_frame
        self._opposition_spec = (
            functional.opposition if cfg.mode == "search" and functional is not None else None
        )
        self._opposition_env_ids = None
        self._needs_tool_contact_reporting = cfg.mode == "search" and (
            cfg.experiment.search.closing_torque_mode == "tool_contact"
            or self._opposition_spec is not None
        )
        self.grasp = (
            load_grasp(
                cfg.grasp_path,
                self.task,
                cfg.sim.device,
                hand_asset=Path(cfg.hand_asset),
                control_dt=cfg.experiment.control_dt,
                **(
                    {"grasp_fingerprint": cfg.experiment.grasp_fingerprint()}
                    if cfg.mode == "search"
                    else {"physics_fingerprint": cfg.experiment.physics_fingerprint()}
                ),
                discovery_sha256=self.area.fingerprint(),
                require_verified=cfg.mode == "rl",
                grasp_index=cfg.grasp_index,
                collision_spec=cfg.experiment.collision,
            )
            if cfg.grasp_path
            else None
        )
        if self.grasp is not None and not 0 <= cfg.grasp_index < len(self.grasp["scores"]):
            raise ValueError("grasp_index is outside the grasp artifact")
        if not Path(cfg.hand_asset).is_file():
            raise FileNotFoundError(f"Hand asset does not exist: {cfg.hand_asset}")
        fixture_s = (
            cfg.experiment.search.setup_time_s if cfg.mode == "search" else physics.rl_setup_time_s
        )
        cfg.episode_length_s = (
            self.trajectory.duration
            + fixture_s
            + cfg.experiment.search.settle_time_s
            + 2 * cfg.experiment.control_dt
        )
        rigid_settings = dict(
            max_depenetration_velocity=physics.max_depenetration_velocity,
            solver_position_iteration_count=physics.solver_position_iterations,
            solver_velocity_iteration_count=physics.solver_velocity_iterations,
        )
        cfg.robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/hand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=cfg.hand_asset,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, **rigid_settings
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=False,
                    enabled_self_collisions=True,
                    solver_position_iteration_count=physics.solver_position_iterations,
                    solver_velocity_iteration_count=physics.solver_velocity_iterations,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.task.initial_position, joint_pos={"a_.*": 0.0}
            ),
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=list(JOINT_NAMES),
                    stiffness=physics.stiffness_nm_rad,
                    damping=physics.damping_nms_rad,
                    effort_limit_sim=physics.joint_effort_limit_nm,
                    velocity_limit_sim=physics.joint_velocity_limit_rad_s,
                )
            },
        )
        cfg.object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(self.task.object_asset),
                # SDK activation also disables sleeping. Reporting is authored
                # directly on the source rigid body before scene cloning.
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, **rigid_settings
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.task.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=physics.contact_offset_m, rest_offset=0.0
                ),
            ),
        )
        self._verification_grasps = None
        self._search_hold = False
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids, names = self.hand.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != JOINT_NAMES or self.hand.num_joints != len(JOINT_NAMES):
            raise ValueError(f"LEAP asset must contain exactly {JOINT_NAMES}; got {names}")
        self.lower = self.hand.data.joint_pos_limits[0, self.joint_ids, 0]
        self.upper = self.hand.data.joint_pos_limits[0, self.joint_ids, 1]
        self.joint_effort_limits = self.hand.data.joint_effort_limits[:, self.joint_ids]
        if self.grasp is not None:
            targets = self.grasp["joint_targets"]
            positions = self.grasp["joint_positions"]
            tolerance = physics.joint_position_tolerance_rad
            numeric_tolerance = torch.finfo(targets.dtype).eps
            if (
                (targets < self.lower - numeric_tolerance)
                | (targets > self.upper + numeric_tolerance)
            ).any():
                raise ValueError(
                    "Saved grasp joint targets exceed this hand's physical joint limits"
                )
            if ((positions < self.lower - tolerance) | (positions > self.upper + tolerance)).any():
                raise ValueError(
                    "Saved physical grasp positions exceed the configured joint solver tolerance"
                )
        self.fixture_steps = max(1, math.ceil(fixture_s / self.step_dt))
        self.settle_steps = math.ceil(cfg.experiment.search.settle_time_s / self.step_dt)
        self.setup_steps = self.fixture_steps + self.settle_steps
        self.num_frames = len(self.trajectory.poses)
        self.start_frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.goal_frame = self.start_frame.clone()
        self.actions = torch.zeros((self.num_envs, cfg.control.action_dim), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.invalid_actions = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.joint_targets = torch.zeros((self.num_envs, len(JOINT_NAMES)), device=self.device)
        self.grasp_targets = torch.zeros_like(self.joint_targets)
        self.grasp_positions = torch.zeros_like(self.joint_targets)
        self.closing_speeds = torch.zeros_like(self.joint_targets)
        self.closing_torques = torch.full_like(self.joint_targets, torch.inf)
        self.closing_stop = ClosingTorqueStopper(self.joint_targets.shape, self.device)
        self.stopped_joints = self.closing_stop.stopped
        self.hand_in_object = torch.zeros((self.num_envs, 7), device=self.device)
        self.hand_in_object[:, 3] = 1
        self.palm_target = torch.zeros_like(self.hand_in_object)
        self.nominal_palm = torch.zeros_like(self.hand_in_object)
        self.previous_palm_target = torch.zeros_like(self.hand_in_object)
        self._servo_substep = 0
        self.zero_velocity = torch.zeros((self.num_envs, 6), device=self.device)
        self.applied_wrench = self.zero_velocity.clone()
        self._stress_wrench_world = None
        self.candidates = torch.zeros((self.num_envs, SEARCH_DIM), device=self.device)
        self.setup = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.setup_ids = self.hand._ALL_INDICES
        self.body_offset = torch.tensor(self.task.body_offset, device=self.device).expand(
            self.num_envs, -1
        )
        self.force_point = torch.tensor(self.task.force_point, device=self.device).expand(
            self.num_envs, -1
        )
        self.task_force = torch.tensor(
            self.task.force, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self.force_multiplier = torch.ones(self.num_envs, device=self.device)
        self.mass_multiplier = torch.ones_like(self.force_multiplier)
        self.friction_multiplier = torch.ones_like(self.force_multiplier)
        randomization = cfg.experiment.randomization
        self._friction_buckets = torch.linspace(
            *randomization.friction_range, randomization.friction_buckets, device=self.device
        )
        # Randomized physics is hidden; policy context uses the nominal measured
        # mass/material/load model and the observed motor response.
        self.nominal_mass = torch.full_like(self.force_multiplier, self.task.mass)
        self.nominal_friction = torch.full_like(self.force_multiplier, physics.friction_dynamic)
        self.nominal_load_multiplier = torch.ones_like(self.force_multiplier)
        self._base_mass = self.object.root_physx_view.get_masses().cpu().clone()
        self._base_inertia = self.object.root_physx_view.get_inertias().cpu().clone()
        validate_inertias(self._base_mass, self._base_inertia)
        self._validate_physics_properties()
        self._base_object_material = (
            self.object.root_physx_view.get_material_properties().cpu().clone()
        )
        self._base_hand_material = self.hand.root_physx_view.get_material_properties().cpu().clone()
        for material in (self._base_object_material, self._base_hand_material):
            material[..., 0] = physics.friction_static
            material[..., 1] = physics.friction_dynamic
            material[..., 2] = physics.restitution
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "return",
                "position_squared",
                "position_distance",
                "angle",
                "tracking",
                "steps",
            )
        }
        self.last_metrics = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "score",
                "position_rmse",
                "position_mean",
                "angle_mean",
                "completion",
                "success",
                "failed",
                "return",
                "steps",
                "collision_tool_hand_max_penetration_m",
                "collision_nonadjacent_self_max_penetration_m",
                "collision_violation",
                "collision_audit_samples",
            )
        }
        self.last_grasp = {
            "hand_in_object": self.hand_in_object.clone(),
            "joint_positions": self.grasp_positions.clone(),
            "joint_targets": self.grasp_targets.clone(),
        }
        self.started = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.set_wrench = ExternalWrenchWriter(self.object)
        self._randomize_physics(self.hand._ALL_INDICES, enabled=False)
        from .collision import (
            CollisionGeometry,
            CollisionModel,
            ToolConvexGeometry,
            validate_native_shape_settings,
        )

        hand_geometry = CollisionGeometry.from_usd(
            Path(cfg.hand_asset), body_names=self.hand.body_names
        )
        tool_geometry = ToolConvexGeometry.from_stage(
            sim_utils.get_current_stage(),
            "/World/envs/env_0/object",
            asset_path=self.task.object_asset,
        )
        self.collision_model = CollisionModel(
            hand_geometry,
            tool_geometry,
            device=self.device,
            max_tool_penetration_m=cfg.experiment.collision.max_tool_penetration_m,
            max_self_penetration_m=cfg.experiment.collision.max_self_penetration_m,
            pair_chunk_size=cfg.experiment.collision.pair_chunk_size,
        )
        self.collision_audit_metadata = self.collision_model.metadata()
        self.collision_audit_metadata["sampling_protocol"] = (
            "native_before_fixture_integration_and_after_every_physics_substep_v1"
        )
        self.collision_audit_metadata["native_shape_settings"] = validate_native_shape_settings(
            self.collision_model, self.hand, self.object
        )
        self._runtime_collision_audit = CollisionAudit(self.num_envs, self.device)
        if cfg.mode == "rl" and cfg.experiment.reward.collision_margin_penalty > 0:
            self._runtime_collision_audit.enable_control_step_tracking()
        if not hasattr(self, "_collision_pending_reset"):
            self._collision_pending_reset = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self._collision_reset_pending = False
        self._fixture_active = torch.full(
            (self.num_envs,), self.fixture is not None, device=self.device, dtype=torch.bool
        )
        self.collision_audit_metadata["fixture"] = self.fixture.metadata()
        self._active_collision_audit = None
        self._active_collision_env_ids = None
        self._active_collision_source_ids = None
        self._active_collision_phase = None
        self._first_collision_finished = None
        self._tool_contact_view = None
        self._tool_contact_fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if cfg.mode == "search" and cfg.experiment.search.closing_torque_mode == "tool_contact":
            self._joint_descendants = joint_descendants_from_usd(
                cfg.hand_asset, JOINT_NAMES, self.hand.body_names, device=self.device
            )
        if self._needs_tool_contact_reporting:
            tool_paths = [f"{path}/object" for path in self.scene.env_prim_paths]
            hand_paths = [
                [f"{path}/hand/{body}" for body in self.hand.body_names]
                for path in self.scene.env_prim_paths
            ]
            points_per_filter = cfg.experiment.resources.tool_contact_points_per_filter
            capacity = (
                self.num_envs * len(self.hand.body_names) * points_per_filter
                if self._opposition_spec is not None
                else 0
            )
            self._tool_contact_view = self.sim.physics_sim_view.create_rigid_contact_view(
                tool_paths, filter_patterns=hand_paths, max_contact_data_count=capacity
            )
            if (
                self._tool_contact_view.sensor_count != self.num_envs
                or self._tool_contact_view.filter_count != len(self.hand.body_names)
            ):
                raise RuntimeError("Native tool-filtered contact channels do not match hand bodies")
            if self._tool_contact_view.max_contact_data_count != capacity:
                raise RuntimeError(
                    "Native detailed contact capacity differs from requested storage"
                )
        if self._opposition_spec is not None:
            self.opposition_contact_metadata = {
                "method": "physx_rigid_contact_view_get_contact_data_v1",
                "point_frame": "actual object rigid-body actor/link frame, not COM",
                "sampling": "final fresh positive-dt physics substep of free settling",
                "centroid_weight": "absolute signed normal force in newtons",
                "body_names": list(self.hand.body_names),
                "body_groups": self._opposition_spec["body_groups"],
                "capacity": capacity,
                "storage_points_per_filter": points_per_filter,
                "completeness": "fresh; unsaturated, valid nonoverlapping native ranges; detailed signed normal-vector sums match filtered aggregate within summation roundoff",
            }
            self.final_opposition_contact_data = {}
            self._opposition_contact_samples = torch.zeros(
                self.num_envs, dtype=torch.int64, device=self.device
            )
            self._opposition_audit_samples = torch.zeros_like(self._opposition_contact_samples)
            self._opposition_group_samples = torch.zeros(
                (self.num_envs, 4), dtype=torch.int64, device=self.device
            )
            self._opposition_last_step = torch.full_like(
                self._opposition_contact_samples, self._sim_step_counter
            )
        self._scene_update_without_collision = self.scene.update
        self.scene.update = self._update_scene_and_audit

    def _update_scene_and_audit(self, dt: float) -> None:
        self._scene_update_without_collision(dt)
        if dt > 0:
            if self._tool_contact_view is not None:
                self._tool_contact_fresh[:] = True
            self._sample_collision()
            if getattr(self, "_opposition_env_ids", None) is not None:
                self._sample_opposition_contacts()

    def _reset_opposition_contacts(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self, "_opposition_contact_samples"):
            return
        self._opposition_contact_samples[env_ids] = 0
        self._opposition_audit_samples[env_ids] = 0
        self._opposition_group_samples[env_ids] = 0
        self._opposition_last_step[env_ids] = self._sim_step_counter

    def _sample_opposition_contacts(self) -> None:
        """Count tool-only distal contacts once per actual free-settle substep."""
        ids = self._opposition_env_ids
        fresh = self._tool_contact_fresh[ids] & (
            self._opposition_last_step[ids] != self._sim_step_counter
        )
        self._opposition_last_step[ids] = self._sim_step_counter
        if self._active_collision_phase != "free_settle":
            return
        fresh &= ~self._fixture_active[ids]
        sampled = ids[fresh]
        if not sampled.numel():
            return
        forces = self._tool_contact_view.get_contact_force_matrix(self.physics_dt)
        if not isinstance(forces, torch.Tensor) or forces.shape != (
            self.num_envs,
            len(self.hand.body_names),
            3,
        ):
            raise RuntimeError("Native tool-filtered contact matrix must have shape [N,B,3]")
        qualifies, groups = self.area.opposition_contacts(forces, self.hand.body_names)
        self._opposition_contact_samples[sampled] += qualifies[sampled].to(torch.int64)
        self._opposition_audit_samples[sampled] += 1
        self._opposition_group_samples[sampled] += groups[sampled].to(torch.int64)

    def _opposition_contact_tensors(self, count: int) -> dict:
        if self._opposition_spec is None:
            return {}
        audited = self._opposition_audit_samples[:count]
        denominator = audited.clamp_min(1).to(torch.float32)
        return {
            "opposition_contact_samples": self._opposition_contact_samples[:count].clone(),
            "opposition_audit_samples": audited.clone(),
            "opposition_contact_fraction": self._opposition_contact_samples[:count] / denominator,
            "opposition_group_contact_fraction": self._opposition_group_samples[:count]
            / denominator[:, None],
        }

    def _final_opposition_contact_geometry(self, count: int, object_pose_world) -> dict:
        if self._opposition_spec is None:
            return {}
        fresh = (
            self._tool_contact_fresh
            & (self._opposition_last_step == self._sim_step_counter)
            & (self._opposition_audit_samples > 0)
            & ~self._fixture_active
        )
        if self._active_collision_phase != "free_settle":
            fresh[:] = False
        result = empty_opposition_geometry(self.num_envs, object_pose_world)
        self.final_opposition_contact_data = {}
        if fresh.any():
            result, raw = opposition_contact_geometry(
                self._tool_contact_view.get_contact_data(self.physics_dt),
                self._tool_contact_view.get_contact_force_matrix(self.physics_dt),
                object_pose_world,
                self.hand.body_names,
                self._opposition_spec["body_groups"],
                self._opposition_spec["minimum_contact_force_n"],
            )
            if raw:
                selected = (raw["env_ids"] < count) & fresh[raw["env_ids"]]
                self.final_opposition_contact_data = {
                    name: value[selected].clone() for name, value in raw.items()
                }
        # A later scene refresh or a reset cannot make old points current.
        for name in (
            "opposition_contact_centers_object_m",
            "opposition_contact_center_forces_n",
            "opposition_contact_point_counts",
            "opposition_contact_center_valid",
            "opposition_contact_geometry_complete",
        ):
            result[name][~fresh] = 0
        result["opposition_contact_geometry_sample_step"] = fresh.to(torch.int64) * (
            self._sim_step_counter
        )
        return {name: value[:count].clone() for name, value in result.items()}

    def collision_metrics(self, env_ids=None) -> dict:
        """Measure actual native shapes, including immediately after joint resets."""
        from .collision import read_native_poses

        hand, obj = read_native_poses(
            self.hand, self.object, simulation_view=self.sim.physics_sim_view
        )
        if env_ids is not None:
            hand, obj = hand[env_ids], obj[env_ids]
        return self.collision_model.metrics(hand, obj)

    def _sample_collision(self, env_ids=None) -> None:
        """Read each required native state once, then share it across auditors.

        A restricted sample is used for newly reset rows and fixture
        rewrites. Every positive-dt scene update still samples all live rows.
        """
        audit = self._active_collision_audit
        active_ids = source_ids = None
        if audit is not None:
            active_ids, source_ids = (
                self._active_collision_env_ids,
                self._active_collision_source_ids,
            )
            if self._first_collision_finished is not None:
                live = ~self._first_collision_finished[active_ids]
                active_ids, source_ids = active_ids[live], source_ids[live]
            if env_ids is not None:
                requested = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                requested[env_ids] = True
                selected = requested[active_ids]
                active_ids, source_ids = active_ids[selected], source_ids[selected]
        runtime = getattr(self, "_runtime_collision_audit", None) if self.cfg.mode == "rl" else None
        measured_ids = (
            (self.hand._ALL_INDICES if env_ids is None else env_ids)
            if runtime is not None
            else active_ids
        )
        if measured_ids is not None and len(measured_ids):
            metrics = self.collision_metrics(measured_ids)
            if runtime is not None:
                runtime.update(metrics, measured_ids, "nominal_trajectory")
            if audit is not None and len(active_ids):
                if runtime is None:
                    active_metrics = metrics
                else:
                    if env_ids is None:
                        positions = active_ids
                    else:
                        positions_by_env = torch.empty(
                            self.num_envs, device=self.device, dtype=torch.long
                        )
                        positions_by_env[measured_ids] = torch.arange(
                            len(measured_ids), device=self.device
                        )
                        positions = positions_by_env[active_ids]
                    active_metrics = {name: value[positions] for name, value in metrics.items()}
                audit.update(active_metrics, source_ids, self._active_collision_phase)
        if env_ids is None:
            self._collision_pending_reset[:] = False
            self._collision_reset_pending = False
        else:
            self._collision_pending_reset[env_ids] = False

    def _mark_collision_reset(self, env_ids: torch.Tensor) -> None:
        """Remember reset placements until an actual pre-integration sample."""
        # Some SDK reset paths can run before collision-model initialization.
        if not hasattr(self, "_collision_pending_reset"):
            self._collision_pending_reset = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        self._collision_pending_reset[env_ids] = True
        self._collision_reset_pending = True

    def _sample_collision_before_step(self, rewritten_ids=None) -> None:
        """Observe discontinuous placements, without resampling unchanged support."""
        pending = self._collision_reset_pending
        if pending:
            if rewritten_ids is not None:
                self._collision_pending_reset[rewritten_ids] = True
            ids = self._collision_pending_reset.nonzero(as_tuple=False).flatten()
        else:
            ids = rewritten_ids
        if ids is not None and len(ids):
            self._sample_collision(ids)
        self._collision_reset_pending = False

    def _validate_physics_properties(self) -> None:
        """Check authored mesh properties actually reached every PhysX clone."""
        com = self.object.data.body_com_pos_b[:, 0]
        expected_com = com.new_tensor(self.physics_metadata["center_of_mass_in_body_m"])
        expected_mass = torch.full_like(self._base_mass, self.task.mass)
        expected_inertia = torch.tensor(
            self.physics_metadata["inertia_in_body_kg_m2"], dtype=self._base_inertia.dtype
        )
        expected_moments = torch.linalg.eigvalsh(expected_inertia)
        measured_moments = torch.linalg.eigvalsh(self._base_inertia.reshape(-1, 3, 3))
        if not torch.allclose(self._base_mass, expected_mass, rtol=1e-5, atol=1e-7):
            raise RuntimeError("PhysX object mass does not match the task's measured mass")
        if not torch.allclose(com, expected_com.expand_as(com), rtol=1e-5, atol=1e-6):
            raise RuntimeError("PhysX COM does not match the mesh-derived center")
        if not torch.allclose(
            measured_moments, expected_moments.expand_as(measured_moments), rtol=1e-4, atol=1e-8
        ):
            raise RuntimeError("PhysX inertia does not match the mesh-derived principal moments")
        self.physics_metadata.update(
            {
                "physx_mass_kg": float(self._base_mass[0].flatten()[0]),
                "physx_com_in_body_m": com[0].cpu().tolist(),
                "physx_com_orientation_in_body_wxyz": self.object.data.body_com_quat_b[0, 0]
                .cpu()
                .tolist(),
                "physx_inertia_matrix": self._base_inertia[0].reshape(3, 3).tolist(),
                "physx_mesh_properties_verified": True,
            }
        )

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._author_object_mass_properties()
        self._tool_contact_reporting = None
        if self._needs_tool_contact_reporting:
            self._tool_contact_reporting = _author_tool_contact_reporting(
                sim_utils.get_current_stage(), "/World/envs/env_0/object"
            )
        physics = self.cfg.experiment.physics
        from .fixture import author_fixture

        author_fixture(sim_utils.get_current_stage())
        floor = sim_utils.CuboidCfg(
            size=(physics.floor_size_m, physics.floor_size_m, physics.floor_thickness_m),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        )
        floor.func("/World/ground", floor, translation=(0.0, 0.0, physics.floor_z_m))
        for contact in self.task.contact_objects:
            proxy = sim_utils.CuboidCfg(
                size=contact.size,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=contact.friction, dynamic_friction=contact.friction
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
            )
            xyzw = Rotation.from_euler("xyz", contact.euler).as_quat()
            proxy.func(
                f"/World/envs/env_0/contact_{contact.name}",
                proxy,
                translation=contact.position,
                orientation=tuple(xyzw[[3, 0, 1, 2]]),
            )
        self.scene.clone_environments(copy_from_source=False)
        from .fixture import TemporaryFixture

        self.fixture = TemporaryFixture(sim_utils.get_current_stage(), self.scene.env_prim_paths)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light.func("/World/light", light)

    def _author_object_mass_properties(self) -> None:
        """Replace inherited inertia with a measured-mass mesh-derived solid."""
        from pxr import Gf, Usd, UsdPhysics

        from .regions import read_usd_mesh

        points, triangles = read_usd_mesh(self.task.object_asset)
        center, inertia, method = mesh_mass_properties(points, triangles, self.task.mass)
        principal_moments, axes = np.linalg.eigh(inertia)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        xyzw = Rotation.from_matrix(axes).as_quat()
        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/object")
        bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise ValueError(f"Tool asset must contain one rigid body; found {len(bodies)}")
        mass_api = UsdPhysics.MassAPI.Apply(bodies[0])
        mass_api.CreateMassAttr().Set(self.task.mass)
        mass_api.CreateDensityAttr().Set(0.0)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center.tolist()))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*principal_moments.tolist()))
        mass_api.CreatePrincipalAxesAttr().Set(
            Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*xyzw[:3].tolist()))
        )
        self.physics_metadata = {
            "mass_kg": self.task.mass,
            "center_of_mass_in_body_m": center.tolist(),
            "inertia_in_body_kg_m2": inertia.tolist(),
            "inertia_method": method,
            "inertia_density": "uniform",
            "contact_load_point_in_body_m": list(self.task.force_point),
            "contact_proxy_names": [proxy.name for proxy in self.task.contact_objects],
            "domain_randomization": self.cfg.experiment.to_dict()["randomization"],
            "external_wrench_frame": "world wrench recomposed from the current link pose each physics substep",
            "wrench_composer_pose_cache": "refreshed before every global wrench write",
        }

    def _poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj = self.object.data.root_link_state_w[:, :7].clone()
        palm = self.hand.data.root_link_state_w[:, :7].clone()
        obj[:, :3] -= self.scene.env_origins
        palm[:, :3] -= self.scene.env_origins
        return obj, palm

    @staticmethod
    def _root_link_velocity(asset) -> torch.Tensor:
        """Read current public COM state without the SDK's stale reset cache."""
        return root_link_velocity(
            asset.data.root_com_vel_w,
            asset.data.root_link_quat_w,
            asset.data.body_com_pos_b[:, 0],
        )

    def _task_wrench(self, pose, com_pose, com_velocity, frames, multiplier) -> torch.Tensor:
        time_s = frames * self.step_dt
        enabled = (
            (~self.setup)
            & (time_s >= self.task.force_interval[0])
            & (time_s <= self.task.force_interval[1])
        )
        return point_load_wrench(
            pose,
            com_pose,
            com_velocity,
            self.force_point,
            self.task_force,
            self.task.resistance,
            self.task.max_resistance,
            multiplier,
            enabled,
            force_frame=self.task.force_frame,
        )

    def _known_wrenches(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = self._task_wrench(
            self.object.data.root_link_state_w[:, :7],
            self.object.data.root_com_state_w[:, :7],
            self.object.data.root_com_vel_w,
            self.goal_frame,
            self.nominal_load_multiplier,
        )
        lookahead = math.ceil(self.cfg.experiment.observation.future_horizon_s / self.step_dt)
        future_frame = (self.goal_frame + lookahead).clamp_max(self.num_frames - 1)
        goal = self.trajectory.poses[future_frame].clone()
        goal[:, :3] += self.scene.env_origins
        com_local = torch.cat(
            (self.object.data.body_com_pos_b[:, 0], self.object.data.body_com_quat_b[:, 0]), -1
        )
        goal_com = compose_pose(goal, com_local)
        goal_velocity = self.trajectory.velocities[future_frame].clone()
        lever = goal_com[:, :3] - goal[:, :3]
        goal_velocity[:, :3] += torch.cross(goal_velocity[:, 3:], lever, dim=-1)
        expected = self._task_wrench(
            goal, goal_com, goal_velocity, future_frame, self.nominal_load_multiplier
        )
        return current, expected

    def _capture_grasp(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        obj, palm = self._poses()
        self.hand_in_object[env_ids] = relative_pose(obj[env_ids], palm[env_ids])
        self.grasp_positions[env_ids] = self.hand.data.joint_pos[env_ids][:, self.joint_ids]
        self.grasp_targets[env_ids] = self.joint_targets[env_ids]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        collision = getattr(self, "_runtime_collision_audit", None)
        if collision is not None and getattr(collision, "control_tool", None) is not None:
            collision.begin_control_step()
        self.previous_palm_target.copy_(self.palm_target)
        self._servo_substep = 0
        self.setup = self.episode_length_buf < self.setup_steps
        fixture = self.episode_length_buf < self.fixture_steps
        self.setup_ids = fixture.nonzero(as_tuple=False).flatten()
        if self.fixture is not None:
            released = (self._fixture_active & ~fixture).nonzero(as_tuple=False).flatten()
            if released.numel():
                self.fixture.set_enabled(released, False)
                self._fixture_active[released] = False
        moving = (self.episode_length_buf - self.setup_steps + 1).clamp_min(0)
        self.goal_frame = (self.start_frame + moving).clamp_max(self.num_frames - 1)
        if self._search_hold:
            self.goal_frame.copy_(self.start_frame)
        goal = self.trajectory.poses[self.goal_frame]
        self.previous_actions.copy_(self.actions)
        self.invalid_actions = ~torch.isfinite(actions).all(dim=-1)
        self.actions.lerp_(torch.nan_to_num(actions).clamp(-1, 1), self.cfg.control.action_filter)
        self.actions[self.setup] = 0
        search_closing = self.cfg.mode == "search" and self._verification_grasps is None
        if search_closing:
            search = self.cfg.experiment.search
            touch = None
            if search.closing_torque_mode == "tool_contact":
                forces = self._tool_contact_view.get_contact_force_matrix(self.physics_dt)
                touch = tool_contact_mask(
                    forces, self._joint_descendants, search.closing_minimum_tool_contact_force_n
                )
                touch &= self._tool_contact_fresh[:, None]
            self.closing_stop.update(
                self.hand.data.applied_torque[:, self.joint_ids],
                self.closing_torques,
                self.closing_speeds,
                fixture,
                self.step_dt,
                mode=search.closing_torque_mode,
                hold_time_s=search.closing_contact_hold_s,
                tool_contact=touch,
            )
            if search.closing_torque_mode == "tool_contact":
                self.joint_targets.copy_(
                    bounded_contact_stop_targets(
                        self.joint_targets,
                        self.hand.data.joint_pos[:, self.joint_ids],
                        self.closing_stop.newly_stopped,
                        self.closing_torques,
                        self.closing_speeds,
                        self.cfg.experiment.physics.stiffness_nm_rad,
                        self.lower,
                        self.upper,
                    )
                )
            close = fixture[:, None] & ~self.stopped_joints & ~self.closing_stop.pause
            self.joint_targets.add_(close * self.closing_speeds * self.step_dt)
            self.joint_targets.clamp_(self.lower, self.upper)
            capture = (
                (self.episode_length_buf == self.setup_steps).nonzero(as_tuple=False).flatten()
            )
            if capture.numel():
                self._capture_grasp(capture)
        self.nominal_palm = compose_pose(goal, self.hand_in_object)
        if search_closing:
            self.palm_target.copy_(self.nominal_palm)
        else:
            self.palm_target, self.joint_targets = action_targets(
                self.actions,
                self.nominal_palm,
                self.grasp_targets,
                self.joint_targets,
                self.lower,
                self.upper,
                self.cfg.control,
                self.step_dt,
            )

    def _apply_action(self) -> None:
        # Reset rows must be sampled before their first integration step.
        self._sample_collision_before_step()
        _, palm = self._poses()
        physics = self.cfg.experiment.physics
        velocity = palm_servo_velocity(
            self.previous_palm_target,
            self.palm_target,
            palm,
            self.step_dt,
            self._servo_substep,
            self.cfg.decimation,
            physics.wrist_position_gain_s_inv,
            physics.wrist_orientation_gain_s_inv,
            self.cfg.control.wrist_linear_speed,
            self.cfg.control.wrist_angular_speed,
        )
        self._servo_substep = (self._servo_substep + 1) % self.cfg.decimation
        self.hand.write_root_link_velocity_to_sim(velocity)
        self.hand.set_joint_position_target(self.joint_targets, joint_ids=self.joint_ids)
        # The composer stores body-local vectors. Recompose world loads every
        # physics substep so they stay world-fixed as the object rotates, and so
        # local point drag responds to the latest angular/linear velocity.
        self.applied_wrench = (
            self._stress_wrench_world
            if self._stress_wrench_world is not None
            else self._task_wrench(
                self.object.data.root_link_state_w[:, :7],
                self.object.data.root_com_state_w[:, :7],
                self.object.data.root_com_vel_w,
                self.goal_frame,
                self.force_multiplier,
            )
        )
        self.set_wrench(
            self.applied_wrench[:, None, :3], self.applied_wrench[:, None, 3:], is_global=True
        )

    def _get_observations(self) -> dict:
        obj, palm = self._poses()
        goal = self.trajectory.poses[self.goal_frame]
        lookahead = math.ceil(self.cfg.experiment.observation.future_horizon_s / self.step_dt)
        future = self.trajectory.poses[(self.goal_frame + lookahead).clamp_max(self.num_frames - 1)]
        current_wrench, expected_wrench = self._known_wrenches()
        obs = observation(
            self.hand.data.joint_pos[:, self.joint_ids],
            self.hand.data.joint_vel[:, self.joint_ids],
            self.joint_targets,
            self.actions,
            obj,
            self._root_link_velocity(self.object),
            palm,
            self._root_link_velocity(self.hand),
            goal,
            self.trajectory.velocities[self.goal_frame],
            future,
            self.goal_frame.float() / (self.num_frames - 1),
            self.setup,
            self.lower,
            self.upper,
            joint_torque=self.hand.data.applied_torque[:, self.joint_ids],
            joint_effort_limits=self.joint_effort_limits,
            current_wrench=current_wrench,
            expected_wrench=expected_wrench,
            mass=self.nominal_mass,
            friction=self.nominal_friction,
            load_multiplier=self.nominal_load_multiplier,
            spec=self.cfg.experiment.observation,
            control=self.cfg.control,
            nominal_grasp=self.hand_in_object,
            center_of_mass_in_body=self.object.data.body_com_pos_b[:, 0],
        )
        return {
            "policy": obs.clamp(
                -self.cfg.experiment.observation.clip, self.cfg.experiment.observation.clip
            )
        }

    def _errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj, _ = self._poses()
        goal = self.trajectory.poses[self.goal_frame]
        center_in_body = self.object.data.body_com_pos_b[:, 0]
        current_center = obj[:, :3] + quat_apply(obj[:, 3:], center_in_body)
        goal_center = goal[:, :3] + quat_apply(goal[:, 3:], center_in_body)
        return (current_center - goal_center).norm(dim=-1), quat_angle(obj[:, 3:], goal[:, 3:])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.position_error, self.angle_error = self._errors()
        finite = torch.isfinite(self.object.data.root_link_state_w).all(dim=-1)
        finite &= torch.isfinite(self.hand.data.root_link_state_w).all(dim=-1)
        finite &= torch.isfinite(self.hand.data.joint_pos).all(dim=-1)
        finite &= torch.isfinite(self.hand.data.joint_vel).all(dim=-1)
        spec = self.cfg.experiment.evaluation
        failed = (
            self.invalid_actions
            | (~finite)
            | (
                ~self._runtime_collision_audit.feasible
                if self.cfg.mode == "rl"
                else torch.zeros_like(self.invalid_actions)
            )
            | (
                (~self.setup)
                & (
                    (self.position_error > spec.failure_position_m)
                    | (self.angle_error > spec.failure_angle_rad)
                )
            )
        )
        finished = (self.goal_frame >= self.num_frames - 1) | (
            self.episode_length_buf >= self.max_episode_length
        )
        if self._active_collision_audit is not None:
            env_ids = self._active_collision_env_ids
            rows = self._active_collision_source_ids
            if self._first_collision_finished is not None:
                live = ~self._first_collision_finished[env_ids]
                env_ids, rows = env_ids[live], rows[live]
            failed[env_ids] |= ~self._active_collision_audit.feasible[rows]
        return failed, finished & ~failed

    def _get_rewards(self) -> torch.Tensor:
        spec = self.cfg.experiment.reward
        kernel = getattr(spec, "tracking_kernel", "gaussian")
        tracking = spec.position_weight * tracking_score(
            self.position_error, spec.position_scale_m, kernel
        )
        tracking += spec.angle_weight * tracking_score(
            self.angle_error, spec.angle_scale_rad, kernel
        )
        action_rate = (self.actions - self.previous_actions).square().mean(dim=-1)
        effort = (
            (self.hand.data.applied_torque[:, self.joint_ids] / self.joint_effort_limits)
            .square()
            .mean(dim=-1)
        )
        reward = tracking + spec.progress_reward - spec.action_rate_penalty * action_rate
        reward -= spec.effort_penalty * effort + spec.action_penalty * self.actions.square().mean(
            dim=-1
        )
        if self.cfg.mode == "rl" and spec.collision_margin_penalty > 0:
            collision = self._runtime_collision_audit
            budgets = self.cfg.experiment.collision
            reward -= spec.collision_margin_penalty * collision_margin_cost(
                collision.control_tool,
                collision.control_self_overlap,
                budgets.max_tool_penetration_m,
                budgets.max_self_penetration_m,
                spec.collision_soft_budget_fraction,
            )
        reward = torch.where(self.setup, 0, reward)
        reward -= spec.failure_penalty * self.reset_terminated.float()
        reward += spec.completion_bonus * self.reset_time_outs.float()
        reward = torch.nan_to_num(
            reward,
            nan=-spec.failure_penalty,
            posinf=-spec.failure_penalty,
            neginf=-spec.failure_penalty,
        )
        active = (~self.setup).float()
        self.episode_sums["return"] += reward
        self.episode_sums["position_squared"] += (
            torch.nan_to_num(self.position_error.square(), nan=1) * active
        )
        self.episode_sums["position_distance"] += (
            torch.nan_to_num(self.position_error, nan=1) * active
        )
        self.episode_sums["angle"] += torch.nan_to_num(self.angle_error, nan=math.pi) * active
        self.episode_sums["tracking"] += torch.nan_to_num(tracking, nan=0) * active
        self.episode_sums["steps"] += active
        return reward

    def _randomize_physics(self, env_ids, enabled: bool | None = None) -> None:
        spec = self.cfg.experiment.randomization
        if enabled is None:
            enabled = self.cfg.randomize_forces and spec.enabled
        ids = env_ids.cpu()
        paired = self.cfg.paired_evaluation and len(env_ids) == self.num_envs
        plan = self._parallel_evaluation_plan
        draws = (
            plan.initial_physics(spec, self._friction_buckets, self.device, enabled=enabled)
            if plan is not None and len(env_ids) == self.num_envs
            else None
        )
        for name, destination, interval in (
            ("mass", self.mass_multiplier, spec.mass_range),
            ("load", self.force_multiplier, spec.load_range),
        ):
            destination[env_ids] = (
                draws[name][env_ids]
                if draws is not None
                else sample_physics_multipliers(len(env_ids), interval, self.device, paired=paired)
                if enabled
                else 1.0
            )
        # PhysX has a finite material registry. Reuse a fixed, bounded bank
        # instead of creating a new continuous material at every episode reset.
        self.friction_multiplier[env_ids] = (
            draws["friction"][env_ids]
            if draws is not None
            else sample_physics_multipliers(
                len(env_ids),
                spec.friction_range,
                self.device,
                paired=paired,
                buckets=self._friction_buckets,
            )
            if enabled
            else 1.0
        )
        masses = self._base_mass.clone()
        inertias = self._base_inertia.clone()
        mass_scale = self.mass_multiplier[env_ids].cpu()
        masses[ids] = self._base_mass[ids] * mass_scale[:, None]
        inertias[ids] = self._base_inertia[ids] * mass_scale[:, None]
        self.object.root_physx_view.set_masses(masses, ids)
        self.object.root_physx_view.set_inertias(inertias, ids)
        friction_scale = self.friction_multiplier[env_ids].cpu()
        for asset, base in (
            (self.object, self._base_object_material),
            (self.hand, self._base_hand_material),
        ):
            materials = base.clone()
            materials[ids, :, :2] = base[ids, :, :2] * friction_scale[:, None, None]
            asset.root_physx_view.set_material_properties(materials, ids)
        if draws is not None:
            self.parallel_evaluation_initial_physics = plan.verify_native_pairs(
                {
                    "object_mass": self.object.root_physx_view.get_masses(),
                    "object_inertia": self.object.root_physx_view.get_inertias(),
                    "hand_mass": self.hand.root_physx_view.get_masses(),
                    "hand_inertia": self.hand.root_physx_view.get_inertias(),
                    "object_material": self.object.root_physx_view.get_material_properties(),
                    "hand_material": self.hand.root_physx_view.get_material_properties(),
                    **draws,
                },
                origins=self.scene.env_origins,
            )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        terminal = env_ids[self.started[env_ids]]
        if terminal.numel():
            raw_steps = self.episode_sums["steps"][terminal]
            steps = raw_steps.clamp_min(1)
            total = (self.num_frames - 1 - self.start_frame[terminal]).clamp_min(1)
            failed = self.reset_terminated[terminal]
            rmse = (self.episode_sums["position_squared"][terminal] / steps).sqrt()
            angle = self.episode_sums["angle"][terminal] / steps
            completion = (raw_steps / total).clamp_max(1)
            spec = self.cfg.experiment.evaluation
            metrics = {
                "score": self.episode_sums["tracking"][terminal] / total
                + completion
                - failed.float(),
                "position_rmse": rmse,
                "position_mean": self.episode_sums["position_distance"][terminal] / steps,
                "angle_mean": angle,
                "completion": completion,
                "success": (
                    ~failed
                    & (rmse < spec.position_tolerance_m)
                    & (angle < spec.angle_tolerance_rad)
                ).float(),
                "failed": failed.float(),
                "return": self.episode_sums["return"][terminal],
                "steps": raw_steps,
            }
            collision = self._runtime_collision_audit
            if self.cfg.mode == "rl":
                metrics.update(
                    collision_tool_hand_max_penetration_m=collision.tool[terminal],
                    collision_nonadjacent_self_max_penetration_m=collision.self_overlap[terminal],
                    collision_violation=(~collision.feasible[terminal]).float(),
                    collision_audit_samples=collision.samples[terminal].float(),
                )
            else:
                tool = torch.zeros(self.num_envs, device=self.device)
                self_depth = torch.zeros_like(tool)
                violations = torch.zeros_like(tool)
                samples = torch.zeros_like(tool)
                if self._active_collision_audit is not None:
                    ids, rows = self._active_collision_env_ids, self._active_collision_source_ids
                    audit = self._active_collision_audit
                    tool[ids] = audit.tool[rows]
                    self_depth[ids] = audit.self_overlap[rows]
                    violations[ids] = (~audit.feasible[rows]).float()
                    samples[ids] = audit.samples[rows].float()
                metrics.update(
                    collision_tool_hand_max_penetration_m=tool[terminal],
                    collision_nonadjacent_self_max_penetration_m=self_depth[terminal],
                    collision_violation=violations[terminal],
                    collision_audit_samples=samples[terminal],
                )
            if self._first_collision_finished is not None:
                self._first_collision_finished[terminal] = True
            for name, value in metrics.items():
                self.last_metrics[name][terminal] = value
            for name in self.last_grasp:
                attribute = {
                    "joint_positions": "grasp_positions",
                    "joint_targets": "grasp_targets",
                }.get(name, name)
                self.last_grasp[name][terminal] = getattr(self, attribute)[terminal]
            self.extras["log"] = {
                f"Episode/{name}": value.mean() for name, value in metrics.items()
            }
        super()._reset_idx(env_ids)
        if hasattr(self, "_runtime_collision_audit"):
            self._runtime_collision_audit.reset(env_ids)
            self._tool_contact_fresh[env_ids] = False
        if hasattr(self, "_opposition_contact_samples"):
            self._reset_opposition_contacts(env_ids)
        self.started[env_ids] = True
        for values in self.episode_sums.values():
            values[env_ids] = 0
        self.actions[env_ids] = 0
        self.invalid_actions[env_ids] = False
        self.previous_actions[env_ids] = 0
        self.closing_stop.reset(env_ids)
        self.start_frame[env_ids] = 0
        if self.cfg.random_start:
            self.start_frame[env_ids] = torch.randint(
                self.num_frames - 1, (len(env_ids),), device=self.device
            )
        self.goal_frame[env_ids] = self.start_frame[env_ids]
        self.setup[env_ids] = True
        self._randomize_physics(env_ids)
        if self.cfg.mode == "search" and self._verification_grasps is None:
            params = self.candidates[env_ids]
            self.hand_in_object[env_ids] = self.area.decode_wrist(params[:, :6])
            closing = decode_closing(params, self.cfg.experiment.search, self.lower, self.upper)
            self.grasp_positions[env_ids] = closing.initial_joints
            self.grasp_targets[env_ids] = closing.initial_joints
            self.closing_speeds[env_ids] = closing.joint_speeds
            self.closing_torques[env_ids] = closing.joint_torque_thresholds
        else:
            source = (
                self._verification_grasps if self._verification_grasps is not None else self.grasp
            )
            indices = env_ids if self._verification_grasps is not None else self.cfg.grasp_index
            for name, attribute in (
                ("hand_in_object", "hand_in_object"),
                ("joint_positions", "grasp_positions"),
                ("joint_targets", "grasp_targets"),
            ):
                getattr(self, attribute)[env_ids] = source[name][indices]
        self.joint_targets[env_ids] = self.grasp_targets[env_ids]
        obj = self.trajectory.poses[self.start_frame[env_ids]].clone()
        palm = compose_pose(obj, self.hand_in_object[env_ids])
        self.nominal_palm[env_ids] = palm
        self.palm_target[env_ids] = palm
        self.previous_palm_target[env_ids] = palm
        obj[:, :3] += self.scene.env_origins[env_ids]
        palm[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_link_pose_to_sim(obj, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(self.zero_velocity[env_ids], env_ids=env_ids)
        if self.fixture is not None:
            self.fixture.write_target_poses(obj, env_ids=env_ids)
            self.fixture.set_enabled(env_ids, True)
            self._fixture_active[env_ids] = True
        self.set_wrench(
            self.zero_velocity[env_ids, None, :3],
            self.zero_velocity[env_ids, None, :3],
            env_ids=env_ids,
            is_global=True,
        )
        self.hand.write_root_link_pose_to_sim(palm, env_ids=env_ids)
        self.hand.write_root_velocity_to_sim(self.zero_velocity[env_ids], env_ids=env_ids)
        self.hand.write_joint_state_to_sim(
            self.grasp_positions[env_ids],
            torch.zeros_like(self.grasp_positions[env_ids]),
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        self.hand.set_joint_position_target(
            self.joint_targets[env_ids], joint_ids=self.joint_ids, env_ids=env_ids
        )
        self._mark_collision_reset(env_ids)

    def set_candidates(self, candidates: torch.Tensor) -> None:
        if self.cfg.mode != "search" or candidates.shape != self.candidates.shape:
            raise ValueError(f"Search candidates must have shape {tuple(self.candidates.shape)}")
        if not torch.isfinite(candidates).all() or (candidates.abs() > 1).any():
            raise ValueError("Search candidates must be finite and in [-1, 1]")
        self.candidates.copy_(candidates)
        self.started[:] = False

    def _manual_physics_step(self) -> None:
        """Search-only stepping: automatic episode resets cannot contaminate a test."""
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self.sim.has_gui() and self._sim_step_counter % self.cfg.sim.render_interval == 0:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

    @torch.no_grad()
    def generate_grasps(self, candidates: torch.Tensor, *, deadline: float | None = None) -> dict:
        if self.cfg.mode != "search" or not 1 <= len(candidates) <= self.num_envs:
            raise ValueError(
                "Grasp generation requires a nonempty search population within num_envs"
            )
        count = len(candidates)
        mapping = torch.arange(self.num_envs, device=self.device) % count
        self.set_candidates(candidates[mapping])
        self._verification_grasps = None
        self._stress_wrench_world = None
        self._search_hold = True
        self.fixture_steps = max(
            1, math.ceil(self.cfg.experiment.search.setup_time_s / self.step_dt)
        )
        self.setup_steps = self.fixture_steps + self.settle_steps
        zero = torch.zeros_like(self.actions)
        audit = CollisionAudit(count, self.device)
        ids = torch.arange(count, device=self.device)
        self._active_collision_audit = audit
        self._active_collision_env_ids = ids
        self._active_collision_source_ids = ids
        self._active_collision_phase = "initial"
        if self._opposition_spec is not None:
            self._reset_opposition_contacts(ids)
            self._opposition_env_ids = ids
        from .collision import read_native_poses

        reference_before_release = None
        try:
            self.reset()
            initial_links, initial_object = read_native_poses(
                self.hand, self.object, simulation_view=self.sim.physics_sim_view
            )
            initial_links = initial_links[:count].clone()
            initial_object = initial_object[:count].clone()
            initial_links[..., :3] -= self.scene.env_origins[:count, None]
            initial_object[:, :3] -= self.scene.env_origins[:count]
            initial_joints = self.hand.data.joint_pos[:count, self.joint_ids].clone()
            initial_targets = self.joint_targets[:count].clone()
            self._sample_collision()
            for step in range(self.setup_steps):
                _check_deadline(deadline, "Grasp generation")
                if step == self.fixture_steps:
                    obj, palm = self._poses()
                    reference_before_release = relative_pose(palm[:count], obj[:count]).clone()
                self._active_collision_phase = (
                    "closure" if step < self.fixture_steps else "free_settle"
                )
                self._pre_physics_step(zero)
                self._manual_physics_step()
                self.episode_length_buf += 1
        finally:
            self._search_hold = False
            self._active_collision_audit = None
            self._opposition_env_ids = None
            if self.fixture is not None:
                self.fixture.release_all()
                self._fixture_active[:] = False
        self._capture_grasp()
        obj, palm = self._poses()
        settled_links, settled_object = read_native_poses(
            self.hand, self.object, simulation_view=self.sim.physics_sim_view
        )
        contact_geometry = self._final_opposition_contact_geometry(count, settled_object)
        settled_links = settled_links[:count].clone()
        settled_object = settled_object[:count].clone()
        settled_links[..., :3] -= self.scene.env_origins[:count, None]
        settled_object[:, :3] -= self.scene.env_origins[:count]
        if reference_before_release is None:
            reference_before_release = relative_pose(palm[:count], obj[:count]).clone()
        valid = stable_mask(
            obj[:count], palm[:count], reference_before_release, self.cfg.experiment.wrench
        )
        valid &= torch.isfinite(self.hand.data.joint_pos[:count]).all(-1)
        measured_q = self.hand.data.joint_pos[:count][:, self.joint_ids]
        tolerance = self.cfg.experiment.physics.joint_position_tolerance_rad
        valid &= (
            (measured_q >= self.lower - tolerance) & (measured_q <= self.upper + tolerance)
        ).all(-1)
        valid &= audit.feasible
        grasp = {
            "hand_in_object": self.hand_in_object[:count].clone(),
            "joint_positions": self.grasp_positions[:count].clone(),
            "joint_targets": self.grasp_targets[:count].clone(),
            "object_pose": obj[:count].clone(),
            "palm_pose": palm[:count].clone(),
            "object_velocity": self.object.data.root_com_vel_w[:count].clone(),
            "palm_velocity": self.hand.data.root_link_vel_w[:count].clone(),
            "joint_velocity": self.hand.data.joint_vel[:count][:, self.joint_ids].clone(),
            "settle_valid": valid,
            "initial_hand_link_poses": initial_links,
            "initial_object_pose": initial_object,
            "initial_joint_positions": initial_joints,
            "initial_joint_targets": initial_targets,
            "closure_reference_object_in_palm": reference_before_release,
            "settled_hand_link_poses": settled_links,
            "settled_object_pose": settled_object,
            **audit.tensors(),
            **self._opposition_contact_tensors(count),
            **contact_geometry,
        }
        for name, value in grasp.items():
            if name != "settle_valid" and not torch.isfinite(value).all():
                raise RuntimeError(
                    f"Grasp generation produced non-finite {name}; inspect physics settings"
                )
        return grasp

    def _restore_stress_grasps(self, grasps: dict, source_ids: torch.Tensor) -> torch.Tensor:
        """Restore the canonical grasp, including velocities, in each physics slot."""
        if self.fixture is not None:
            self.fixture.release_all()
            self._fixture_active[:] = False
        obj = grasps["object_pose"][source_ids].clone()
        palm = grasps["palm_pose"][source_ids].clone()
        obj[:, 2] += self.cfg.experiment.wrench.clearance_m
        palm[:, 2] += self.cfg.experiment.wrench.clearance_m
        self.palm_target.copy_(palm)
        self.nominal_palm.copy_(palm)
        self.previous_palm_target.copy_(palm)
        self._servo_substep = 0
        reference = relative_pose(palm, obj)
        obj[:, :3] += self.scene.env_origins
        palm[:, :3] += self.scene.env_origins
        self.hand.reset()
        self.object.reset()
        self.object.write_root_link_pose_to_sim(obj)
        self.object.write_root_com_velocity_to_sim(grasps["object_velocity"][source_ids])
        self.hand.write_root_link_pose_to_sim(palm)
        self.hand.write_root_link_velocity_to_sim(grasps["palm_velocity"][source_ids])
        self.hand.write_joint_state_to_sim(
            grasps["joint_positions"][source_ids],
            grasps["joint_velocity"][source_ids],
            joint_ids=self.joint_ids,
        )
        self.joint_targets.copy_(grasps["joint_targets"][source_ids])
        self.setup[:] = False
        self.setup_ids = self.hand._ALL_INDICES[:0]
        self._stress_wrench_world = self.zero_velocity
        self.set_wrench(
            self.zero_velocity[:, None, :3], self.zero_velocity[:, None, :3], is_global=True
        )
        self.scene.write_data_to_sim()
        self.scene.update(0.0)
        return reference

    @torch.no_grad()
    def stress_grasps(self, grasps: dict, *, deadline: float | None = None) -> torch.Tensor:
        count = len(grasps["settle_valid"])
        audit = CollisionAudit(count, self.device)
        for name, attribute in (
            ("collision_tool_hand_max_penetration_m", "tool"),
            ("collision_nonadjacent_self_max_penetration_m", "self_overlap"),
            ("collision_audit_samples", "samples"),
            ("collision_phase_samples", "phases"),
            ("collision_tool_hand_phase_max_penetration_m", "phase_tool"),
            ("collision_nonadjacent_self_phase_max_penetration_m", "phase_self"),
            ("collision_feasible", "feasible"),
        ):
            getattr(audit, attribute).copy_(grasps[name])
        output = torch.zeros((count, WRENCH_DIRECTIONS), device=self.device)
        eligible = grasps["settle_valid"] & audit.feasible
        if "functional_grip_feasible" in grasps:
            functional = grasps["functional_grip_feasible"]
            if (
                not isinstance(functional, torch.Tensor)
                or functional.dtype != torch.bool
                or functional.shape != (count,)
                or functional.device != eligible.device
            ):
                raise ValueError("functional_grip_feasible must be one boolean per candidate")
            eligible &= functional
        eligible = eligible.nonzero(as_tuple=False).flatten()
        tests = (
            eligible[:, None] * WRENCH_DIRECTIONS
            + torch.arange(WRENCH_DIRECTIONS, device=self.device)
        ).flatten()
        try:
            return self._stress_grasp_batches(grasps, tests, output, audit, deadline=deadline)
        finally:
            self.set_wrench(
                self.zero_velocity[:, None, :3], self.zero_velocity[:, None, :3], is_global=True
            )
            self._stress_wrench_world = None
            self._active_collision_audit = None
            grasps.update(audit.tensors())

    def _stress_grasp_batches(self, grasps, tests, output, audit, *, deadline):
        spec = self.cfg.experiment.wrench
        for offset in range(0, len(tests), self.num_envs):
            _check_deadline(deadline, "Wrench evaluation")
            batch = tests[offset : offset + self.num_envs]
            active_count = len(batch)
            padded = batch[torch.arange(self.num_envs, device=self.device) % active_count]
            source_ids = padded // WRENCH_DIRECTIONS
            direction_ids = padded % WRENCH_DIRECTIONS
            reference = self._restore_stress_grasps(grasps, source_ids)
            alive = grasps["settle_valid"][source_ids].clone()
            self._active_collision_audit = audit
            self._active_collision_phase = "wrench"
            self._active_collision_env_ids = torch.arange(active_count, device=self.device)
            self._active_collision_source_ids = source_ids[:active_count]
            self._sample_collision()
            for _ in range(math.ceil(spec.restore_settle_time_s / self.step_dt)):
                _check_deadline(deadline, "Wrench restoration")
                self._manual_physics_step()
                obj, palm = self._poses()
                alive &= stable_mask(obj, palm, reference, spec)
                alive &= audit.feasible[source_ids]
                active = alive[:active_count].nonzero(as_tuple=False).flatten()
                self._active_collision_env_ids = active
                self._active_collision_source_ids = source_ids[active]
                if not active.numel():
                    break
            accumulator = StabilityAccumulator(
                self.num_envs, spec.duration_s, self.step_dt, self.device
            )
            accumulator.alive &= alive
            for step in range(accumulator.steps):
                _check_deadline(deadline, "Wrench evaluation")
                elapsed = min((step + 1) * self.step_dt, spec.duration_s)
                wrench = ramp_wrenches(direction_ids, elapsed, spec)
                wrench *= accumulator.alive[:, None]
                self._stress_wrench_world = wrench
                self._manual_physics_step()
                obj, palm = self._poses()
                stable = stable_mask(obj, palm, reference, spec)
                stable &= torch.isfinite(self.hand.data.joint_pos).all(-1)
                stable &= audit.feasible[source_ids]
                accumulator.update(stable)
                active = accumulator.alive[:active_count].nonzero(as_tuple=False).flatten()
                self._active_collision_env_ids = active
                self._active_collision_source_ids = source_ids[active]
                if not accumulator.alive[:active_count].any():
                    break
            output[batch // WRENCH_DIRECTIONS, batch % WRENCH_DIRECTIONS] = accumulator.fractions[
                :active_count
            ]
        return output

    @torch.no_grad()
    def verify_grasps(self, grasps: dict, *, deadline: float | None = None) -> dict:
        from .optimize import first_episodes

        count = len(grasps["hand_in_object"])
        if not 1 <= count <= self.num_envs:
            raise ValueError(
                "Trajectory shortlist must fit within the physical environment population"
            )
        mapping = torch.arange(self.num_envs, device=self.device) % count
        self._stress_wrench_world = None
        self._verification_grasps = {
            name: grasps[name][mapping]
            for name in ("hand_in_object", "joint_positions", "joint_targets")
        }
        original_random_start, original_randomization = (
            self.cfg.random_start,
            self.cfg.randomize_forces,
        )
        original_fixture, original_setup = self.fixture_steps, self.setup_steps
        self.cfg.random_start = False
        self.cfg.randomize_forces = False
        self.fixture_steps = max(
            1, math.ceil(self.cfg.experiment.physics.rl_setup_time_s / self.step_dt)
        )
        self.setup_steps = self.fixture_steps + self.settle_steps
        self.started[:] = False
        audit = CollisionAudit(count, self.device)
        ids = torch.arange(count, device=self.device)
        self._active_collision_audit = audit
        self._active_collision_env_ids = ids
        self._active_collision_source_ids = ids
        self._active_collision_phase = "nominal_trajectory"
        self._first_collision_finished = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        try:
            metrics, _ = first_episodes(self, deadline=deadline)
            return {
                **{name: value[:count].clone() for name, value in metrics.items()},
                **audit.tensors(),
            }
        except TimeoutError as error:
            diagnostic = getattr(error, "verification_diagnostics", {})
            diagnostic["collision_audit"] = audit.tensors()
            error.verification_diagnostics = diagnostic
            raise
        finally:
            self._verification_grasps = None
            self._active_collision_audit = None
            self._first_collision_finished = None
            if self.fixture is not None:
                self.fixture.release_all()
                self._fixture_active[:] = False
            self.cfg.random_start, self.cfg.randomize_forces = (
                original_random_start,
                original_randomization,
            )
            self.fixture_steps, self.setup_steps = original_fixture, original_setup

    def verification_diagnostic_state(self) -> dict:
        """Actual current poses at a bounded verifier's deadline, not an outcome."""
        from .collision import read_native_poses

        links, obj = read_native_poses(
            self.hand, self.object, simulation_view=self.sim.physics_sim_view
        )
        links[..., :3] -= self.scene.env_origins[:, None]
        obj[:, :3] -= self.scene.env_origins
        position, angle = self._errors()
        return {
            "frame": "environment",
            "hand_link_poses": links,
            "object_pose": obj,
            "joint_positions": self.hand.data.joint_pos[:, self.joint_ids].clone(),
            "joint_targets": self.joint_targets.clone(),
            "position_error_m": position.clone(),
            "angle_error_rad": angle.clone(),
            "goal_frame": self.goal_frame.clone(),
            "episode_length": self.episode_length_buf.clone(),
            "first_episode_finished": (
                self._first_collision_finished.clone()
                if self._first_collision_finished is not None
                else None
            ),
        }

    def _close_for_scene_rebuild(self) -> None:
        """Release the search world while keeping the already-launched Kit app."""
        if self._is_closed:
            return
        # Isaac Lab's own multi-environment tests disable this subscription when
        # stopping a world inside a persistent AppLauncher. Otherwise STOP can
        # request application shutdown before the replacement scene is built.
        self.sim._app_control_on_stop_handle = None
        self.sim.stop()
        self.close()
        # These additional references belong to our subclass, so deleting scene
        # alone does not release the assets and their PhysX/timeline callbacks.
        del self.set_wrench
        del self.hand
        del self.object
        del self.fixture
        del self._tool_contact_view
        del self.sim
        gc.collect()

    def close(self) -> None:
        if hasattr(self, "_scene_update_without_collision"):
            self.scene.update = self._scene_update_without_collision
            del self._scene_update_without_collision
        super().close()

    @torch.no_grad()
    def verify_grasps_in_small_scene(self, grasps: dict, *, deadline: float | None = None) -> dict:
        """Consume a large search world and verify each shortlist grasp once.

        The former implementation repeated eight candidates across hundreds of
        articulations for the entire recording. A fresh shortlist-sized scene
        keeps identical physics and removes that redundant solver workload.
        """
        count = len(grasps["hand_in_object"])
        if not 1 <= count <= self.num_envs:
            raise ValueError("Trajectory shortlist must fit within the search population")
        if count == self.num_envs:
            return self.verify_grasps(grasps, deadline=deadline)
        _check_deadline(deadline, "Shortlist scene rebuild")
        cfg = self.cfg.copy()
        cfg.scene.num_envs = count
        cfg.random_start = False
        cfg.randomize_forces = False
        cfg.paired_evaluation = False
        render_mode = self.render_mode
        independent_grasps = {
            name: tensor.detach().cpu().clone() for name, tensor in grasps.items()
        }
        print(f"Rebuilding verification scene: {self.num_envs} -> {count} environments", flush=True)
        self._close_for_scene_rebuild()
        sim_utils.create_new_stage()
        _check_deadline(deadline, "Shortlist scene rebuild")
        verifier = type(self)(cfg, render_mode=render_mode)
        try:
            transferred = {
                name: tensor.to(verifier.device) for name, tensor in independent_grasps.items()
            }
            return verifier.verify_grasps(transferred, deadline=deadline)
        finally:
            verifier.close()
