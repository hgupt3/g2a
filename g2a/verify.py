"""Full-trajectory verification of saved optimized grasps."""

import hashlib
import json
import os
import time
from copy import deepcopy

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
    validate_collision_audit,
    write_json,
)


def _source_summary_history(source, output) -> tuple[dict, dict | None]:
    """Keep the exact prior sidecar before an in-place successful replay replaces it."""
    if not source.is_file():
        return {}, None
    encoded = source.read_bytes()
    previous = json.loads(encoded)
    if not isinstance(previous, dict):
        raise ValueError("The source grasp summary must contain a JSON object")
    digest = hashlib.sha256(encoded).hexdigest()
    snapshot = output.with_name(f"{output.stem}.source-summary-{digest}.json")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if snapshot.read_bytes() != encoded:
            raise RuntimeError("The preserved source summary does not match its SHA256")
    else:
        temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
        try:
            temporary.write_bytes(encoded)
            os.replace(temporary, snapshot)
        finally:
            temporary.unlink(missing_ok=True)
    return previous, {
        "original_path": str(source),
        "snapshot_path": str(snapshot),
        "sha256": digest,
        "prior_stop_reason": previous.get("stop_reason"),
        "prior_trajectory_verified": previous.get("trajectory_verified"),
        "prior_collision_verified": previous.get("collision_verified"),
        "summary": previous,
    }


@torch.no_grad()
def reverify(env, args, grasp: dict) -> dict:
    """Finish or repeat full trajectory verification without repeating grasp search."""
    from .optimize import (
        SEARCH_COLLISION_PHASE_IDS,
        collision_gate,
        functional_grip_fitness,
        merge_verification_collisions,
        rank_verified_grasps,
        save_verification_diagnostics,
    )

    started = time.perf_counter()
    limit = args.max_wall_time or env.cfg.experiment.evaluation.verification_max_wall_time_s
    count = len(grasp["scores"])
    source_hash = file_hash(args.grasp)
    audit = grasp.get("collision_audit")
    if (
        grasp.get("schema_version") != GRASP_SCHEMA_VERSION
        or not isinstance(audit, dict)
        or audit.get("schema_version") != COLLISION_AUDIT_VERSION
        or audit.get("method") != COLLISION_AUDIT_METHOD
        or audit.get("phases") != list(COLLISION_AUDIT_PHASES)
        or audit.get("sampling_protocol") != COLLISION_SAMPLING_PROTOCOL
    ):
        raise ValueError(
            "Grasp lacks a native construction/wrench collision audit; reconstruct it with "
            "g2a optimize. Replaying a legacy trajectory cannot certify past closure or stress."
        )
    if audit.get("candidate_state_sha256") != candidate_state_hashes(grasp):
        raise ValueError("Collision audit is not bound to the saved grasp states; reconstruct them")
    observed_metadata = env.collision_audit_metadata
    if (
        observed_metadata.get("schema_version") != COLLISION_AUDIT_VERSION
        or observed_metadata.get("method") != COLLISION_AUDIT_METHOD
        or observed_metadata.get("sampling_protocol") != COLLISION_SAMPLING_PROTOCOL
        or observed_metadata.get("native_pose_source") != audit.get("native_pose_source")
        or observed_metadata.get("geometry_sha256") != audit.get("geometry_sha256")
        or observed_metadata.get("self_collision_exclusions")
        != audit.get("self_collision_exclusions")
    ):
        raise ValueError("Collision geometry changed; reconstruct and stress-test the grasp")
    bundle = {
        name: value.to(env.device)
        for name, value in grasp.items()
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == count
    }
    construction_safe = collision_gate(
        bundle, env.cfg.experiment.collision, required_phases=SEARCH_COLLISION_PHASE_IDS
    )
    area = getattr(env, "area", None)
    if area is not None and area.functional_frame is not None:
        # A nominal replay cannot supply missing grasp-construction contact geometry.
        # Evaluate the actual construction certificate against the active area,
        # keeping raw SG and collision evidence independent of this condition.
        if area.functional_frame.opposition is not None and not grasp.get(
            "opposition_contact_metadata"
        ):
            raise ValueError("Opposed grasps require native construction contact-point provenance")
        _, functional = functional_grip_fitness(
            area, bundle, bundle["area_patch_ids"], bundle["scores"]
        )
        construction_safe &= functional
    if not construction_safe.any():
        raise RuntimeError(
            "No saved grasp has complete collision-feasible construction/wrench evidence; reconstruct it"
        )
    try:
        metrics = env.verify_grasps(bundle, deadline=started + limit)
    except (RuntimeError, ValueError, TimeoutError) as error:
        save_verification_diagnostics(
            grasp,
            None,
            args.output,
            native_metadata=observed_metadata,
            error=error,
        )
        raise
    verification_diagnostic = save_verification_diagnostics(
        grasp, metrics, args.output, native_metadata=observed_metadata
    )
    metrics = merge_verification_collisions(bundle, metrics, env.cfg.experiment.collision)
    bundle["collision_feasible"] = collision_gate(bundle, env.cfg.experiment.collision)
    bundle.update({f"metric_{name}": value for name, value in metrics.items()})
    order, selection = rank_verified_grasps(
        bundle,
        env.cfg.experiment.search,
        env.cfg.experiment.evaluation,
        env.cfg.experiment.collision,
    )
    order = order[bundle["selection_eligible"][order]]
    result = {**grasp, **{name: value[order].cpu() for name, value in bundle.items()}}
    result.update(
        trajectory_verified=True,
        physics=env.physics_metadata,
        # The saved experiment describes construction, including its recorded
        # closing rule. Replaying fixed states does not construct a new grasp.
        verification_experiment=env.cfg.experiment.to_dict(),
        selection="success/completion tier, relative stability eligibility, normalized COM/angle tracking cost, SG tie-break",
        selection_details=selection,
        verification_environments=count,
        verification_method="one fresh physical environment per saved grasp",
        verification_source_sha256=source_hash,
        collision_verified=True,
    )
    if area is not None and area.functional_frame is not None:
        result["functional_grip_verified"] = True
        result["functional_grip_convention"] = deepcopy(area.calibration["functional_grip"])
    # Retain the original construction budgets. New maxima must satisfy those
    # recorded limits as well as the active verification limits; replay cannot
    # loosen old evidence or rewrite the closing experiment that produced it.
    fresh_metadata = deepcopy(audit)
    fresh_metadata["physics_fingerprint"] = result["physics_fingerprint"]
    fresh_metadata["candidate_state_sha256"] = candidate_state_hashes(result)
    result["collision_audit"] = fresh_metadata
    validate_collision_audit(result, collision_spec=env.cfg.experiment.collision)
    previous, source_history = _source_summary_history(args.grasp.with_suffix(".json"), args.output)
    # Search history still describes construction. Past verification status,
    # errors and elapsed time belong only to the immutable source snapshot.
    summary = {
        name: deepcopy(previous[name])
        for name in (
            "history",
            "iterations_completed",
            "candidate_population",
            "physical_environments",
            "search_elapsed_s",
            "search_sampling",
            "warm_proposal_provenance",
        )
        if name in previous
    }
    elapsed = time.perf_counter() - started
    summary.update(
        task=env.task.name,
        operation="reverification",
        stop_reason="reverification_completed",
        elapsed_s=elapsed,
        source_summary_history=source_history,
        trajectory_verified=True,
        best_stability_score=float(result["scores"][0]),
        best_completion=float(result["metric_completion"][0]),
        best_success=bool(result["metric_success"][0]),
        best_position_rmse_m=float(result["metric_position_rmse"][0]),
        best_angle_mean_rad=float(result["metric_angle_mean"][0]),
        wrench_stable_fractions=result["wrench_stable_fractions"][0].tolist(),
        selection_details=selection,
        reverification_elapsed_s=elapsed,
        verification_environments=count,
        verification_method=result["verification_method"],
        verification_source_sha256=source_hash,
        verification_experiment=result["verification_experiment"],
        verification_diagnostic_artifact=verification_diagnostic,
        collision_verified=True,
        collision_audit=result["collision_audit"],
        best_tool_hand_penetration_m=float(result["collision_tool_hand_max_penetration_m"][0]),
        best_nonadjacent_self_penetration_m=float(
            result["collision_nonadjacent_self_max_penetration_m"][0]
        ),
    )
    atomic_torch_save(result, args.output)
    write_json(summary, args.output.with_suffix(".json"))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
