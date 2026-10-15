"""Versioned, portable artifacts with explicit task and policy compatibility checks."""

import hashlib
import io
import json
import math
import os
from dataclasses import fields
from pathlib import Path

import torch

from .config import CollisionSpec, ExperimentSpec, PhysicsSpec, SearchSpec
from .control import JOINT_COUNT, JOINT_NAMES, ControlSpec, ObservationSpec, observation_fields
from .tasks import TaskSpec

# Policy contracts bind the observation/action layout; grasp artifacts bind
# physical states to their collision and full-trajectory checks.
SCHEMA_VERSION = 2
GRASP_SCHEMA_VERSION = 3
COLLISION_AUDIT_VERSION = 1
COLLISION_AUDIT_METHOD = "native_box_cooked_convex_sat"
COLLISION_SAMPLING_PROTOCOL = "native_before_fixture_integration_and_after_every_physics_substep_v1"
COLLISION_AUDIT_PHASES = ("initial", "closure", "free_settle", "wrench", "nominal_trajectory")


def config_hash(value: dict) -> str:
    """Stable content hash, independent of JSON formatting and mapping order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def saved_physics_fingerprint(grasp: dict) -> str:
    """Derive physical compatibility from the recorded simulation settings."""
    experiment = grasp.get("experiment")
    physics = experiment.get("physics") if isinstance(experiment, dict) else None
    required = {item.name for item in fields(PhysicsSpec)} - {"joint_position_tolerance_rad"}
    if not isinstance(physics, dict) or not required.issubset(physics):
        raise ValueError("Grasp is missing saved physical simulation settings; regenerate it")
    if physics["fixture_mode"] != "fixed_joint":
        raise ValueError(
            "Grasp physical simulation settings require fixture_mode=fixed_joint; regenerate it"
        )
    search = experiment.get("search")
    if not isinstance(search, dict) or "settle_time_s" not in search:
        raise ValueError(
            "Grasp is missing its recorded initial free-settling duration; regenerate it"
        )
    settle = search["settle_time_s"]
    if (
        isinstance(settle, bool)
        or not isinstance(settle, (int, float))
        or not math.isfinite(settle)
        or settle < 0
    ):
        raise ValueError("Grasp initial free-settling duration must be finite and nonnegative")
    try:
        actual = ExperimentSpec(
            physics=PhysicsSpec(**physics), search=SearchSpec(settle_time_s=settle)
        ).physics_fingerprint()
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Grasp contains incompatible saved physical simulation settings: {error}"
        ) from error
    declared = grasp.get("physics_fingerprint")
    if declared is not None and declared != actual:
        raise ValueError("Grasp physics fingerprint does not match its recorded physical settings")
    return actual


def candidate_state_hashes(grasp: dict) -> list[str]:
    """Bind collision evidence to the exact saved candidate states/provenance.

    Scores do not change a grasp. Wrist poses, joint positions, and targets do;
    their previous collision evidence cannot be reused. Dtype and shape are
    included, without depending on torch archive serialization or a simulator.
    """
    tensors = []
    count = None
    for name, width in (
        ("hand_in_object", 7),
        ("joint_positions", JOINT_COUNT),
        ("joint_targets", JOINT_COUNT),
    ):
        value = grasp.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.ndim != 2
            or value.shape[1] != width
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid grasp field {name} for collision audit binding")
        count = len(value) if count is None else count
        if len(value) != count:
            raise ValueError("Grasp candidate fields disagree on collision audit population")
        tensors.append((name, value.detach().cpu().contiguous()))
    binding = config_hash(
        {
            "schema_version": COLLISION_AUDIT_VERSION,
            "task_fingerprint": grasp.get("task_fingerprint"),
            "joint_names": list(grasp.get("joint_names", ())),
            "hand_asset_sha256": grasp.get("hand_asset_sha256"),
            "physics_fingerprint": saved_physics_fingerprint(grasp),
        }
    ).encode()
    result = []
    for index in range(count):
        digest = hashlib.sha256(binding)
        for name, tensor in tensors:
            row = tensor[index]
            digest.update(
                config_hash(
                    {"name": name, "dtype": str(row.dtype), "shape": list(row.shape)}
                ).encode()
            )
            digest.update(row.view(torch.uint8).numpy().tobytes())
        result.append(digest.hexdigest())
    return result


def _collision_vector(grasp, name, count, dtype, width=None):
    value = grasp.get(name)
    shape = (count,) if width is None else (count, width)
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or (not value.is_floating_point() if dtype == "float" else value.dtype != dtype)
        or not torch.isfinite(value).all()
        or (value < 0).any()
    ):
        raise ValueError(f"Invalid native collision audit field {name}")
    return value


def validate_collision_audit(
    grasp: dict, grasp_index: int = 0, collision_spec: CollisionSpec | None = None
) -> None:
    """Validate archive evidence and enforce physical feasibility of one row.

    Rejected rows are useful diagnostic evidence and are never dropped or
    reordered here. Positive samples from every native phase are required for
    the selected row; trajectory metrics alone do not certify collision safety.
    """
    count = len(grasp["scores"])
    audit = grasp.get("collision_audit")
    if (
        not isinstance(audit, dict)
        or type(audit.get("schema_version")) is not int
        or audit["schema_version"] != COLLISION_AUDIT_VERSION
    ):
        raise ValueError("Grasp lacks a versioned native collision audit; reoptimize and verify it")
    if audit.get("method") != COLLISION_AUDIT_METHOD:
        raise ValueError("Grasp collision audit is not the supported native cooked-geometry method")
    if audit.get("sampling_protocol") != COLLISION_SAMPLING_PROTOCOL:
        raise ValueError(
            "Grasp collision audit lacks the current native fixture/substep sampling protocol; "
            "reclose and retest it rather than relabeling past construction evidence"
        )
    if audit.get("native_pose_source") != "PhysX body_link_state_w/root_link_state_w":
        raise ValueError("Grasp collision audit does not record supported native body poses")
    geometry = audit.get("geometry_sha256")
    if (
        not isinstance(geometry, str)
        or len(geometry) != 64
        or any(character not in "0123456789abcdef" for character in geometry)
    ):
        raise ValueError("Grasp collision audit lacks a cooked-geometry SHA256")
    hand_asset = grasp.get("hand_asset_sha256")
    if (
        not isinstance(hand_asset, str)
        or len(hand_asset) != 64
        or any(character not in "0123456789abcdef" for character in hand_asset)
    ):
        raise ValueError("Grasp collision audit is missing its actual hand asset SHA256")
    exclusions = audit.get("self_collision_exclusions")
    if not isinstance(exclusions, (list, tuple)) or any(
        not isinstance(pair, (list, tuple))
        or len(pair) != 2
        or not all(isinstance(name, str) and name for name in pair)
        or pair[0] == pair[1]
        for pair in exclusions
    ):
        raise ValueError("Grasp collision audit must record source self-collision exclusions")
    if (
        not isinstance(audit.get("phases"), (list, tuple))
        or tuple(audit["phases"]) != COLLISION_AUDIT_PHASES
    ):
        raise ValueError("Grasp collision audit does not cover all five required native phases")
    if audit.get("physics_fingerprint") != saved_physics_fingerprint(grasp):
        raise ValueError("Grasp collision audit does not match its recorded physical settings")
    if not isinstance(audit.get("candidate_state_sha256"), list) or audit[
        "candidate_state_sha256"
    ] != candidate_state_hashes(grasp):
        raise ValueError(
            "Grasp collision audit does not match the exact saved candidate states/order"
        )
    experiment = grasp.get("experiment")
    recorded = experiment.get("collision") if isinstance(experiment, dict) else None
    if not isinstance(recorded, dict) or not {
        "max_tool_penetration_m",
        "max_self_penetration_m",
    }.issubset(recorded):
        raise ValueError("Grasp is missing recorded native collision acceptance budgets")
    try:
        recorded_spec = CollisionSpec(**recorded)
        physics_defaults = PhysicsSpec()
        recorded_physics = PhysicsSpec(
            **{
                name: tuple(value)
                if isinstance(getattr(physics_defaults, name), tuple)
                and isinstance(value, (list, tuple))
                else value
                for name, value in experiment["physics"].items()
            }
        )
        ExperimentSpec(physics=recorded_physics, collision=recorded_spec).validate()
        if collision_spec is not None:
            if not isinstance(collision_spec, CollisionSpec):
                raise TypeError("expected CollisionSpec")
            ExperimentSpec(physics=recorded_physics, collision=collision_spec).validate()
    except (TypeError, ValueError) as error:
        raise ValueError("Grasp collision acceptance budgets are invalid") from error
    for name in ("max_tool_penetration_m", "max_self_penetration_m"):
        declared = audit.get(name)
        if (
            isinstance(declared, bool)
            or not isinstance(declared, (int, float))
            or not math.isfinite(declared)
            or declared != getattr(recorded_spec, name)
        ):
            raise ValueError(f"Grasp collision audit budget {name} disagrees with saved settings")
    tool = _collision_vector(grasp, "collision_tool_hand_max_penetration_m", count, "float")
    hand = _collision_vector(grasp, "collision_nonadjacent_self_max_penetration_m", count, "float")
    samples = _collision_vector(grasp, "collision_audit_samples", count, torch.int64)
    phases = _collision_vector(
        grasp, "collision_phase_samples", count, torch.int64, len(COLLISION_AUDIT_PHASES)
    )
    for name, overall in (
        ("collision_tool_hand_phase_max_penetration_m", tool),
        ("collision_nonadjacent_self_phase_max_penetration_m", hand),
    ):
        if name in grasp:
            phase_depths = _collision_vector(
                grasp, name, count, "float", len(COLLISION_AUDIT_PHASES)
            )
            if (phase_depths.max(-1).values > overall).any():
                raise ValueError(
                    f"Native collision maximum understates recorded phase depth: {name}"
                )
    feasible = _collision_vector(grasp, "collision_feasible", count, torch.bool)
    if not torch.equal(samples, phases.sum(dim=-1)):
        raise ValueError("Grasp native collision sample totals disagree with phase coverage")
    recorded_safe = (tool <= recorded_spec.max_tool_penetration_m) & (
        hand <= recorded_spec.max_self_penetration_m
    )
    if (feasible & ~recorded_safe).any():
        raise ValueError("Grasp collision_feasible flags contradict measured native penetration")
    if not bool(feasible[grasp_index]):
        raise ValueError("Selected grasp failed the native collision feasibility audit")
    if not bool((phases[grasp_index] > 0).all()):
        raise ValueError("Selected grasp has incomplete native collision phase coverage")
    active_spec = collision_spec or recorded_spec
    if bool(tool[grasp_index] > active_spec.max_tool_penetration_m) or bool(
        hand[grasp_index] > active_spec.max_self_penetration_m
    ):
        raise ValueError("Selected grasp exceeds the current native collision penetration budgets")


def load_grasp(
    path: str | Path,
    task: TaskSpec,
    device: str = "cpu",
    hand_asset: Path | None = None,
    control_dt: float | None = None,
    grasp_fingerprint: str | None = None,
    discovery_sha256: str | None = None,
    require_verified: bool = False,
    physics_fingerprint: str | None = None,
    grasp_index: int = 0,
    collision_spec: CollisionSpec | None = None,
) -> dict:
    """Load compatible candidates; optionally require finished task verification.

    Search warm starts bind construction settings because they re-close the
    fingers. Fixed grasps bind physical settings independently of search/closure
    choices. Training/evaluation require explicit full-trajectory and native
    collision verification for the selected row; rejected rows remain intact.
    Failed candidate outcomes remain preserved for diagnostics, but cannot be
    selected through require_verified=True.
    """
    if not isinstance(require_verified, bool):
        raise ValueError("require_verified must be a boolean")
    if isinstance(grasp_index, bool) or not isinstance(grasp_index, int) or grasp_index < 0:
        raise ValueError("grasp_index must be a nonnegative integer")
    grasp = torch.load(path, map_location=device, weights_only=True)
    schema = grasp.get("schema_version") if isinstance(grasp, dict) else None
    if type(schema) is not int or schema != GRASP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported grasp artifact schema {schema}; regenerate schema "
            f"{GRASP_SCHEMA_VERSION} grasps with g2a optimize"
        )
    if grasp.get("task_fingerprint") != task.fingerprint():
        raise ValueError("Grasp belongs to different task settings or data; regenerate it")
    if tuple(grasp.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("Grasp joint order does not match the LEAP hand")
    if hand_asset is not None and grasp.get("hand_asset_sha256") != file_hash(hand_asset):
        raise ValueError("Grasp was generated for a different hand asset; regenerate it")
    if control_dt is not None and grasp.get("control_dt") != control_dt:
        raise ValueError("Grasp control timestep does not match the environment")
    if grasp_fingerprint is not None and grasp.get("grasp_fingerprint") != grasp_fingerprint:
        raise ValueError("Grasp construction configuration does not match; regenerate it")
    if physics_fingerprint is not None and saved_physics_fingerprint(grasp) != physics_fingerprint:
        raise ValueError("Grasp physical settings do not match the environment; regenerate it")
    if discovery_sha256 is not None and grasp.get("discovery_sha256") != discovery_sha256:
        raise ValueError("Grasp area discovery data does not match; regenerate it")
    scores = grasp.get("scores")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError("Grasp scores must be a floating-point vector")
    count = len(scores)
    for name, width in (
        ("hand_in_object", 7),
        ("joint_positions", JOINT_COUNT),
        ("joint_targets", JOINT_COUNT),
    ):
        tensor = grasp.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or not tensor.is_floating_point()
            or tensor.shape != (count, width)
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Invalid grasp field {name}")
    if count == 0 or not torch.isfinite(grasp["scores"]).all():
        raise ValueError("Grasp contains no finite candidates")
    if grasp_index >= count:
        raise ValueError("grasp_index is outside the grasp artifact")
    if require_verified:
        if grasp.get("trajectory_verified") is not True:
            raise ValueError(
                "Grasp has not completed full-trajectory verification; "
                "resume g2a optimize before training or evaluation"
            )
        for name in ("metric_completion", "metric_success"):
            tensor = grasp.get(name)
            if (
                not isinstance(tensor, torch.Tensor)
                or not tensor.is_floating_point()
                or tensor.shape != (count,)
                or not torch.isfinite(tensor).all()
                or (tensor < 0).any()
                or (tensor > 1).any()
            ):
                raise ValueError(
                    f"Invalid grasp verification metric {name}; "
                    "expected one finite value in [0, 1] per candidate"
                )
        validate_collision_audit(grasp, grasp_index, collision_spec)
        completion = grasp["metric_completion"]
        if bool(grasp["metric_success"][grasp_index] != 1) or bool(
            completion[grasp_index] < 1 - torch.finfo(completion.dtype).eps
        ):
            raise ValueError(
                "Selected grasp did not complete its nominal trajectory with strict success"
            )
    expected_norm = torch.ones_like(grasp["hand_in_object"][:, 0])
    if not torch.allclose(grasp["hand_in_object"][:, 3:].norm(dim=-1), expected_norm, atol=1e-4):
        raise ValueError("Grasp orientations must be unit quaternions")
    return grasp


def load_warm_proposals(
    path: str | Path,
    task: TaskSpec,
    *,
    hand_asset: Path,
    control_dt: float,
    discovery_sha256: str,
    device: str = "cpu",
) -> dict:
    """Load only normalized search proposals, never saved physical evidence.

    A deliberate solver or closure change needs fresh reconstruction rather
    than a fixed-grasp compatibility exception. Task/asset/area/timing and the
    parameter decoder layout still match exactly. Historical SG only orders
    the proposals; no score, wrist, joint state or audit is returned for reuse.
    """
    from .stress import PARAMETER_FIELDS, SEARCH_DIM

    if (
        isinstance(control_dt, bool)
        or not isinstance(control_dt, (int, float))
        or not math.isfinite(control_dt)
        or control_dt <= 0
    ):
        raise ValueError("Warm proposal control timestep must be finite and positive")
    if (
        not isinstance(discovery_sha256, str)
        or len(discovery_sha256) != 64
        or any(character not in "0123456789abcdef" for character in discovery_sha256)
    ):
        raise ValueError("Warm proposals require the exact current area-discovery SHA256")
    source_path = Path(path)
    # Hash the exact bytes being parsed even if an atomic producer later
    # replaces the source path. Deserialize old states on CPU, not the GPU.
    encoded = source_path.read_bytes()
    source = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    schema = source.get("schema_version") if isinstance(source, dict) else None
    if type(schema) is not int or schema != GRASP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported warm-proposal artifact schema {schema}; expected explicit "
            f"schema-{GRASP_SCHEMA_VERSION} parameter provenance"
        )
    task_fingerprint = task.fingerprint()
    if source.get("task_fingerprint") != task_fingerprint:
        raise ValueError(
            "Warm proposals belong to different task settings or object/trajectory data"
        )
    if "task" in source and source["task"] != task.name:
        raise ValueError("Warm proposal task name disagrees with its task fingerprint")
    if tuple(source.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("Warm proposal joint order does not match the LEAP hand")
    hand_sha256 = file_hash(Path(hand_asset))
    if source.get("hand_asset_sha256") != hand_sha256:
        raise ValueError("Warm proposals were generated for a different hand asset")
    recorded_dt = source.get("control_dt")
    if (
        isinstance(recorded_dt, bool)
        or not isinstance(recorded_dt, (int, float))
        or not math.isfinite(recorded_dt)
        or recorded_dt != control_dt
    ):
        raise ValueError("Warm proposal control timestep does not match the environment")
    if source.get("discovery_sha256") != discovery_sha256:
        raise ValueError("Warm proposal area discovery data does not match the current decoder")
    layout = source.get("search_parameter_fields")
    if (
        not isinstance(layout, (list, tuple))
        or any(
            not isinstance(field, (list, tuple))
            or len(field) != 2
            or not isinstance(field[0], str)
            or type(field[1]) is not int
            for field in layout
        )
        or tuple(tuple(field) for field in layout) != PARAMETER_FIELDS
    ):
        raise ValueError("Warm proposal search_parameter_fields do not match PARAMETER_FIELDS")
    parameters = source.get("parameters")
    if (
        not isinstance(parameters, torch.Tensor)
        or not parameters.is_floating_point()
        or parameters.ndim != 2
        or parameters.shape[1] != SEARCH_DIM
        or len(parameters) == 0
        or not torch.isfinite(parameters).all()
        or (parameters.abs() > 1).any()
    ):
        raise ValueError(
            f"Warm proposal parameters must be nonempty finite normalized rows in [-1, 1] "
            f"with shape (N, {SEARCH_DIM})"
        )
    scores = source.get("scores")
    if (
        not isinstance(scores, torch.Tensor)
        or not scores.is_floating_point()
        or scores.shape != (len(parameters),)
        or not torch.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("Warm proposal ordering requires finite source SG values in [0, 1]")
    experiment = source.get("experiment")
    if not isinstance(experiment, dict) or not experiment:
        raise ValueError("Warm proposals are missing their recorded source experiment")
    try:
        source_configuration_sha256 = config_hash(experiment)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Warm proposal source experiment must be finite JSON configuration"
        ) from error
    order = torch.argsort(scores, descending=True, stable=True)
    proposals = parameters[order].detach().clone().to(device)
    provenance = {
        "mode": "parameter_only_fresh_reconstruction",
        "source_path": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_schema_version": schema,
        "source_experiment": experiment,
        "source_experiment_sha256": source_configuration_sha256,
        "source_grasp_fingerprint": source.get("grasp_fingerprint"),
        "source_physics_fingerprint": source.get("physics_fingerprint"),
        "source_row_order": order.tolist(),
        "ordering": "descending recorded SG, stable ties; proposal priority only",
        "task_fingerprint": task_fingerprint,
        "object_asset_sha256": file_hash(task.object_asset),
        "hand_asset_sha256": hand_sha256,
        "discovery_sha256": discovery_sha256,
        "control_dt": control_dt,
        "search_parameter_fields": [list(field) for field in PARAMETER_FIELDS],
        "normalized_range": [-1, 1],
        "proposal_count": len(proposals),
        "requires_fresh_reclosure": True,
        "inherited_physical_states": False,
        "inherited_collision_audit": False,
        "inherited_fitness": False,
    }
    return {"parameters": proposals, "provenance": provenance}


def policy_contract(
    task: TaskSpec,
    control: ControlSpec,
    control_dt: float,
    observation_spec: ObservationSpec | None = None,
) -> dict:
    observation_spec = observation_spec or ObservationSpec()
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("Policy control timestep must be finite and positive")
    control.validate()
    observation_spec.validate()
    fields = observation_fields(observation_spec, control)
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task.name,
        "task_fingerprint": task.fingerprint(),
        "joint_names": list(JOINT_NAMES),
        "action_dim": control.action_dim,
        "observation_dim": sum(width for _, width in fields),
        "observation_fields": [list(field) for field in fields],
        "control": control.to_dict(),
        "observation": observation_spec.to_dict(),
        "sensor_sources": {
            "object_pose_velocity": "measured in the environment frame",
            "palm_pose_velocity": "measured in the environment frame",
            "joint_torque": "estimated applied actuator torque, Nm",
            "actuator_load_proxy": "absolute actuator torque / physical effort limit; not tactile contact",
            "current_task_wrench": "known or estimated task force/torque in world frame; excludes contact and gravity",
            "expected_task_wrench": "nominal task load model at the anticipation horizon, world frame",
            "mass_friction_load": "nominal or estimated context; hidden randomization is not observed",
        },
        "control_dt": control_dt,
    }


def validate_policy_contract(actual: dict, expected: dict) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Policy contract mismatch for {key}; train a matching checkpoint")


def write_json(value: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
