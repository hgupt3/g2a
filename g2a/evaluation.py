"""Failure-inclusive tracking and explicitly conditioned comparison metrics.

The paper reports mean Euclidean distance over completed, nondropping episodes.
That is different from mean per-episode RMSE over every first episode. Preserve
both definitions, and compare matched completed pairs on the same subset.
"""

import math
from collections.abc import Mapping

import torch

_REQUIRED = (
    "score",
    "success",
    "failed",
    "completion",
    "position_rmse",
    "angle_mean",
    "return",
)
_COMPLETION_ROUNDING_TOLERANCE = 1e-6
COLLISION_FIELDS = (
    "collision_violation",
    "collision_tool_hand_max_penetration_m",
    "collision_nonadjacent_self_max_penetration_m",
)
COLLISION_SAMPLE_FIELD = "collision_audit_samples"


def _validate(metrics: Mapping[str, torch.Tensor]) -> int:
    missing = set(_REQUIRED) - metrics.keys()
    if missing:
        raise ValueError(f"Evaluation is missing episode metrics: {sorted(missing)}")
    score = metrics["score"]
    if not isinstance(score, torch.Tensor) or score.ndim != 1:
        raise ValueError("Episode metric score must be a one-dimensional tensor")
    count = len(score)
    if count == 0:
        raise ValueError("Evaluation contains no episodes")
    for name, value in metrics.items():
        if not isinstance(value, torch.Tensor) or value.shape != (count,):
            raise ValueError(f"Episode metric {name} must have shape ({count},)")
        if not torch.isfinite(value).all():
            raise ValueError("Evaluation contains non-finite episode metrics")
    if (
        "collision_violation" in metrics
        and not (
            (metrics["collision_violation"] == 0) | (metrics["collision_violation"] == 1)
        ).all()
    ):
        raise ValueError("Episode collision_violation flags must be zero or one")
    for name in COLLISION_FIELDS[1:]:
        if name in metrics and (metrics[name] < 0).any():
            raise ValueError(f"Episode metric {name} must be nonnegative penetration depth")
    if COLLISION_SAMPLE_FIELD in metrics:
        samples = metrics[COLLISION_SAMPLE_FIELD]
        if (
            samples.dtype == torch.bool
            or (samples < 0).any()
            or samples.is_floating_point()
            and (samples != samples.floor()).any()
        ):
            raise ValueError("Episode collision_audit_samples must be nonnegative integer counts")
    return count


def _mean(value: torch.Tensor) -> float:
    # Preserve the original floating precision; boolean/integer outcome flags
    # need conversion because their tensor mean operation is undefined.
    return float((value if value.is_floating_point() else value.float()).mean())


def completed_survivors(metrics: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Exclude failures and incomplete episodes, regardless of strict success.

    This changes reporting subsets only. Simulator failure/tracking thresholds
    and the original strict success flags remain authoritative and untouched.
    """
    _validate(metrics)
    complete = (metrics["failed"] == 0) & (
        metrics["completion"] >= 1 - _COMPLETION_ROUNDING_TOLERANCE
    )
    if "collision_violation" in metrics:
        complete &= metrics["collision_violation"] == 0
    return complete


def _measured_collisions(metrics):
    if not all(name in metrics for name in (*COLLISION_FIELDS, COLLISION_SAMPLE_FIELD)):
        return torch.zeros_like(metrics["failed"], dtype=torch.bool)
    return metrics[COLLISION_SAMPLE_FIELD] > 0


def _collisions(metrics, mask):
    """Keep unmeasured collisions unknown; never infer them from success."""
    count = int(mask.sum())
    measured = _measured_collisions(metrics) & mask
    measured_count = int(measured.sum())
    complete = count > 0 and measured_count == count
    samples = metrics.get(COLLISION_SAMPLE_FIELD)
    return {
        "collision_metrics_complete": complete,
        "collision_measured_episodes": measured_count,
        "collision_unknown_episodes": count - measured_count,
        "collision_audit_samples_min": float(samples[mask].min())
        if count and samples is not None
        else None,
        "collision_violation_rate": (
            _mean(metrics["collision_violation"][mask]) if complete else None
        ),
        **{
            name: float(metrics[name][mask].max()) if complete else None
            for name in COLLISION_FIELDS[1:]
        },
    }


def _tracking(metrics: Mapping[str, torch.Tensor], mask: torch.Tensor) -> dict:
    count = int(mask.sum())

    def mean(field):
        if count == 0 or field not in metrics:
            return None
        return _mean(metrics[field][mask])

    distance = mean("position_mean")
    angle = mean("angle_mean")
    return {
        "episodes": count,
        "position_mean_m": distance,
        "position_mean_cm": None if distance is None else 100 * distance,
        "position_rmse_m": mean("position_rmse"),
        "angle_mean_rad": angle,
        "angle_mean_deg": None if angle is None else math.degrees(angle),
        **_collisions(metrics, mask),
    }


def summarize(metrics: Mapping[str, torch.Tensor]) -> dict:
    """Keep original all-episode metrics and add a paper-like survivor subset.

    Mean distance must come from a native per-step distance accumulator; RMSE
    cannot recover it. Older metric packets report null instead of inventing a
    distance. Every mean weights first episodes equally, including failures in
    the original aggregate and only complete survivors in the conditioned one.
    """
    count = _validate(metrics)
    all_episodes = torch.ones_like(metrics["failed"], dtype=torch.bool)
    tracking = _tracking(metrics, all_episodes)
    survivors = _tracking(metrics, completed_survivors(metrics))
    survivors["survival_rate"] = survivors["episodes"] / count
    return {
        "episodes": count,
        "success_rate": _mean(metrics["success"]),
        "failure_rate": _mean(metrics["failed"]),
        "completion_mean": _mean(metrics["completion"]),
        "position_rmse_m": tracking["position_rmse_m"],
        "angle_mean_rad": tracking["angle_mean_rad"],
        "return_mean": _mean(metrics["return"]),
        "position_mean_m": tracking["position_mean_m"],
        "paper_like_survivors": survivors,
        **_collisions(metrics, all_episodes),
    }


def paired_summary(
    policy_metrics: Mapping[str, torch.Tensor], baseline_metrics: Mapping[str, torch.Tensor]
) -> dict:
    """Compare corresponding mirrored rows where both controllers complete.

    Callers must align rows by initial physics draw and batch seed, including
    counterbalanced groups. This helper never pairs independently sampled runs
    or conditions one controller on a different set of survivors.
    """
    count = _validate(policy_metrics)
    if _validate(baseline_metrics) != count:
        raise ValueError("Paired episode metrics must have identical row counts")
    policy_complete = completed_survivors(policy_metrics)
    baseline_complete = completed_survivors(baseline_metrics)
    if policy_complete.device != baseline_complete.device:
        raise ValueError("Paired episode metrics must be on the same device")
    shared = policy_complete & baseline_complete
    policy = _tracking(policy_metrics, shared)
    baseline = _tracking(baseline_metrics, shared)

    def reduction(field):
        reference, candidate = baseline[field], policy[field]
        if reference is None or candidate is None or reference <= 0:
            return None
        return (reference - candidate) / reference

    return {
        "pairs": int(shared.sum()),
        "candidate_pairs": count,
        "subset": "corresponding first episodes completed without failure or any recorded collision violation by both controllers",
        "collision_metrics_complete": bool(_measured_collisions(policy_metrics).all())
        and bool(_measured_collisions(baseline_metrics).all()),
        "policy": policy,
        "baseline": baseline,
        "position_mean_reduction_fraction": reduction("position_mean_m"),
        "position_rmse_reduction_fraction": reduction("position_rmse_m"),
        "angle_mean_reduction_fraction": reduction("angle_mean_rad"),
    }


def episode_rows(metrics: Mapping[str, torch.Tensor]) -> list[dict]:
    """Serialize every first episode, including failures, with explicit units."""
    count = _validate(metrics)
    measured = _measured_collisions(metrics)
    rows = []
    for index in range(count):
        row = {
            "episode_index": index,
            "score": float(metrics["score"][index]),
            "success": float(metrics["success"][index]),
            "failed": float(metrics["failed"][index]),
            "completion": float(metrics["completion"][index]),
            "position_rmse_m": float(metrics["position_rmse"][index]),
            "position_mean_m": (
                float(metrics["position_mean"][index]) if "position_mean" in metrics else None
            ),
            "angle_mean_rad": float(metrics["angle_mean"][index]),
            "return": float(metrics["return"][index]),
        }
        if "steps" in metrics:
            row["steps"] = float(metrics["steps"][index])
        row["collision_measured"] = bool(measured[index])
        row[COLLISION_SAMPLE_FIELD] = (
            float(metrics[COLLISION_SAMPLE_FIELD][index])
            if COLLISION_SAMPLE_FIELD in metrics
            else None
        )
        for name in COLLISION_FIELDS:
            row[name] = float(metrics[name][index]) if measured[index] else None
        rows.append(row)
    return rows
