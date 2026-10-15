"""Finite PPO, complete-trajectory model selection, and portable inference.

Training return from randomly started episodes does not decide release quality.
Fresh paired full trajectories compare each candidate with its optimized grasp.
If training fails to improve it, the selected controller is explicitly a zero
residual fallback and the learned candidate is retained separately for inspection.
"""

import copy
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .artifacts import config_hash, file_hash, policy_contract, validate_policy_contract, write_json
from .config import CollisionSpec, RewardSpec, TrainingSpec
from .control import DEFAULT_OBSERVATION, TWIST_DIM, VECTOR_DIM, ControlSpec, observation_fields
from .evaluation import episode_rows, paired_summary
from .evaluation import summarize as summarize_episode_metrics
from .optimize import first_episodes


def environment_contract(env) -> dict:
    experiment = env.cfg.experiment
    contract = policy_contract(env.task, env.cfg.control, env.step_dt, experiment.observation)
    contract["hand_asset_sha256"] = file_hash(Path(env.cfg.hand_asset))
    digest = hashlib.sha256()
    for field in ("hand_in_object", "joint_positions", "joint_targets"):
        digest.update(env.grasp[field][env.cfg.grasp_index].detach().cpu().numpy().tobytes())
    contract["grasp_sha256"] = digest.hexdigest()
    contract["grasp_fingerprint"] = env.grasp["grasp_fingerprint"]
    contract["discovery_sha256"] = env.grasp["discovery_sha256"]
    contract["physics_dt"] = env.physics_dt
    # Search budgets, PPO learning rates and evaluation batch counts do not
    # change the physical inference contract. Actuator/solver changes do.
    sections = experiment.to_dict()
    contract["physics"] = sections["physics"]
    contract["physics_sha256"] = config_hash(sections["physics"])
    return contract


def training_spec(args, defaults: TrainingSpec) -> TrainingSpec:
    """CLI overrides are optional; a configuration file retains its own values."""
    names = {
        "iterations": "epochs",
        "horizon": "horizon",
        "max_wall_time": "max_wall_time_s",
        "validation_every": "validation_every_epochs",
        "validation_batches": "validation_batches",
    }
    overrides = {
        target: getattr(args, source)
        for source, target in names.items()
        if getattr(args, source, None) is not None
    }
    return replace(defaults, **overrides)


def ppo_config(
    num_envs: int,
    horizon: int,
    iterations: int,
    device: str,
    seed: int,
    training: TrainingSpec | None = None,
) -> dict:
    import yaml

    training = training or TrainingSpec()
    if min(num_envs, horizon, iterations) < 1:
        raise ValueError("PPO environment, horizon and epoch counts must be positive")
    if num_envs * horizon < 2:
        raise ValueError("PPO advantage normalization requires at least two samples")
    with Path(__file__).with_name("ppo.yaml").open() as stream:
        result = yaml.safe_load(stream)
    params = result["params"]
    config = params["config"]
    params["seed"] = seed
    params["network"]["mlp"]["units"] = list(training.network_units)
    params["network"]["separate"] = training.actor_critic_separate
    params["network"]["space"]["continuous"]["sigma_init"]["val"] = training.initial_log_std
    config.update(
        num_actors=num_envs,
        horizon_length=horizon,
        max_epochs=iterations,
        device=device,
        device_name=device,
        learning_rate=training.learning_rate,
        lr_schedule="adaptive" if training.learning_rate_schedule == "adaptive" else None,
        critic_coef=training.critic_loss_coefficient,
        bound_loss_type="regularisation",
        bounds_loss_coef=training.zero_residual_regularization,
        entropy_coef=training.entropy_coefficient,
        mini_epochs=training.mini_epochs,
        gamma=training.gamma,
        tau=training.gae_lambda,
        kl_threshold=training.kl_threshold,
        grad_norm=training.gradient_clip,
        e_clip=training.clip_ratio,
        save_frequency=0,
        # The library's random-start return checkpoint is not a selected policy.
        save_best_after=iterations + 1,
    )
    batch = num_envs * horizon
    # PPO needs an exact divisor, including for short CPU/GPU runs.
    minibatch = min(training.minibatch_size, batch)
    while batch % minibatch:
        minibatch -= 1
    config["minibatch_size"] = minibatch
    return result


class DeterministicPolicy(torch.nn.Module):
    """The supported feedforward actor, with its exact trained normalizer.

    Calling the full rl-games model in evaluation mode still samples an unused
    Gaussian action and evaluates the critic. Selecting only the actor modules
    avoids those computations and leaves the random generator unchanged.
    """

    def __init__(self, model, clip_obs: float = 10.0):
        super().__init__()
        network = getattr(model, "a2c_network", None)
        if (
            network is None
            or not getattr(network, "is_continuous", False)
            or getattr(network, "has_rnn", False)
            or getattr(network, "has_cnn", False)
        ):
            raise ValueError("Portable G2A policies require a continuous feedforward MLP actor")
        self.normalizer = model.running_mean_std if model.normalize_input else torch.nn.Identity()
        self.preprocessor = network.actor_cnn
        self.encoder = network.actor_mlp
        self.mean_head = network.mu
        self.mean_activation = network.mu_act
        self.clip_obs = clip_obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Match rl-games norm_obs, including its stopped input gradient, rather
        # than approximating running statistics or adding a second normalizer.
        with torch.no_grad():
            normalized = self.normalizer(obs.clamp(-self.clip_obs, self.clip_obs))
        features = self.preprocessor(normalized).contiguous().flatten(1)
        mean = self.mean_activation(self.mean_head(self.encoder(features)))
        return mean.clamp(-1, 1)


class ZeroResidualPolicy(torch.nn.Module):
    """A named optimized-grasp fallback; it must never be called learned PPO."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.new_zeros((obs.shape[0], self.action_dim))


def summarize(metrics: dict) -> dict:
    return summarize_episode_metrics(metrics)


def tracking_cost(metrics: dict, reward) -> float:
    return (
        reward.position_weight * metrics["position_rmse_m"] / reward.position_scale_m
        + reward.angle_weight * metrics["angle_mean_rad"] / reward.angle_scale_rad
    ) / (reward.position_weight + reward.angle_weight)


def candidate_rank(metrics: dict, reward) -> tuple[float, ...]:
    """Rank eligible, nonregressing controllers by tracking first."""
    return (
        -tracking_cost(metrics, reward),
        metrics["success_rate"],
        -metrics["failure_rate"],
        metrics["completion_mean"],
    )


def collision_metrics_available(metrics: dict) -> bool:
    """Require measured native collision outcomes in addition to task success."""
    if metrics.get("collision_metrics_complete") is not True:
        return False
    episodes = metrics.get("episodes")
    measured = metrics.get("collision_measured_episodes")
    unknown = metrics.get("collision_unknown_episodes")
    minimum_samples = metrics.get("collision_audit_samples_min")
    if (
        type(episodes) is not int
        or episodes < 1
        or type(measured) is not int
        or measured != episodes
        or type(unknown) is not int
        or unknown != 0
        or isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, (int, float))
        or not math.isfinite(minimum_samples)
        or minimum_samples <= 0
        or minimum_samples != math.floor(minimum_samples)
    ):
        return False
    for name in (
        "collision_violation_rate",
        "collision_tool_hand_max_penetration_m",
        "collision_nonadjacent_self_max_penetration_m",
    ):
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or name == "collision_violation_rate"
            and value > 1
        ):
            return False
    return True


def collision_safe(metrics: dict, collision_spec: CollisionSpec | None = None) -> bool:
    """Require zero observed violations and acceptable native peak depths."""
    if not collision_metrics_available(metrics) or metrics["collision_violation_rate"] != 0:
        return False
    collision_spec = collision_spec or CollisionSpec()
    for name, bound in (
        ("collision_tool_hand_max_penetration_m", collision_spec.max_tool_penetration_m),
        ("collision_nonadjacent_self_max_penetration_m", collision_spec.max_self_penetration_m),
    ):
        # Native PhysX/SAT depths use float32. Serialize its threshold with the
        # same precision so the exact declared boundary cannot become unsafe
        # solely through promotion to a Python float during JSON reporting.
        limit = float(torch.tensor(bound, dtype=torch.float32))
        if metrics[name] > limit:
            return False
    return True


def does_not_regress(
    candidate: dict,
    baseline: dict,
    training: TrainingSpec,
    collision_spec: CollisionSpec | None = None,
    *,
    reward: RewardSpec | None = None,
) -> bool:
    """Preserve reliability and the explicitly chosen tracking criterion."""
    if not collision_safe(candidate, collision_spec) or not collision_metrics_available(baseline):
        return False
    for metrics in (candidate, baseline):
        for key in ("position_rmse_m", "angle_mean_rad"):
            value = metrics[key]
            if not math.isfinite(value) or value < 0:
                return False
    tolerance = training.selection_success_tolerance
    if candidate["success_rate"] < baseline["success_rate"] - tolerance:
        return False
    if candidate["failure_rate"] > baseline["failure_rate"] + tolerance:
        return False
    completion_tolerance = training.selection_completion_tolerance
    if candidate["completion_mean"] < baseline["completion_mean"] - completion_tolerance:
        return False
    if training.selection_tracking_mode == "combined":
        if reward is None:
            raise ValueError(
                "Combined tracking selection requires the experiment's reward scales and weights"
            )
        return tracking_cost(candidate, reward) <= tracking_cost(baseline, reward)
    for key, tolerance in (
        ("position_rmse_m", training.selection_position_relative_tolerance),
        ("angle_mean_rad", training.selection_angle_relative_tolerance),
    ):
        if candidate[key] > baseline[key] * (1 + tolerance):
            return False
    return True


def improves_baseline(
    candidate: dict,
    baseline: dict,
    training: TrainingSpec,
    reward,
    collision_spec: CollisionSpec | None = None,
) -> bool:
    """Require a material tracking gain; success cannot substitute for accuracy."""
    if not does_not_regress(candidate, baseline, training, collision_spec, reward=reward):
        return False
    reference = tracking_cost(baseline, reward)
    current = tracking_cost(candidate, reward)
    return reference > 0 and current < reference * (1 - training.selection_min_tracking_gain)


@contextmanager
def preserve_rng(cuda_device: str | torch.device | None = None):
    """Validation seeding must not alter subsequent exploration draws."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if (
        cuda_device is not None
        and torch.device(cuda_device).type == "cuda"
        and torch.cuda.is_initialized()
    ):
        # Reading every GPU's RNG would create contexts on otherwise unused
        # devices. This experiment only draws CUDA randomness on its own device.
        cuda_state = torch.cuda.get_rng_state(cuda_device)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_device)


def full_validation(env, policy, seeds: list[int], deadline: float) -> dict:
    """Fresh complete trajectories, even when PPO uses random episode starts."""
    random_start = env.cfg.random_start
    env.cfg.random_start = False
    try:
        with preserve_rng(getattr(env, "device", None)):
            batches = []
            for seed in seeds:
                env.seed(seed)
                batches.append(first_episodes(env, policy, deadline=deadline)[0])
        metrics = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        return summarize(metrics)
    finally:
        env.cfg.random_start = random_start


def _cleanup_status(log_path: Path | None, message: str) -> None:
    if log_path is not None:
        with log_path.open("a") as log:
            log.write(f"\n[G2A validation cleanup] {message}\n")


def _stop_validation_process(process, log_path: Path | None = None) -> bool:
    """Reap only the process group created for this validation."""
    if process.poll() is not None:
        return True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # Native CUDA shutdown can remain in the kernel after SIGKILL.
            # Keep the parent deadline bounded and preserve its normal timeout
            # handling; the already-killed child is the only process we reap.
            _cleanup_status(log_path, f"owned PID {process.pid}: SIGKILL sent; OS reap pending")

            def reap():
                try:
                    returncode = process.wait()
                    _cleanup_status(
                        log_path, f"owned PID {process.pid}: reaped, status {returncode}"
                    )
                except Exception as error:
                    _cleanup_status(log_path, f"owned PID {process.pid}: reap error {error}")

            threading.Thread(
                target=reap, daemon=True, name=f"g2a-validation-reaper-{process.pid}"
            ).start()
            return False
    _cleanup_status(log_path, f"owned PID {process.pid}: reaped, status {process.returncode}")
    return True


def validation_error_details(log_path: Path) -> str:
    """Include the first native cause as well as a bounded final traceback."""
    contents = log_path.read_text(errors="replace")
    lines = contents.splitlines()
    allocator = [
        i
        for i, line in enumerate(lines)
        if "cudamemoryallocator" in line.lower()
        or "out of memory" in line.lower()
        or "not enough gpu memory" in line.lower()
        or "alloc" in line.lower()
        and any(word in line.lower() for word in ("error", "fail"))
    ]
    errors = [
        i
        for i, line in enumerate(lines)
        if "[Error]" in line or "PhysX error:" in line or "Traceback" in line
    ]
    first = allocator[0] if allocator else errors[0] if errors else None
    context = "\n".join(lines[first : first + 12])[:1200] if first is not None else ""
    return f"First allocator/SDK error:\n{context}\nEnd of child log:\n{contents[-2800:]}"


def run_validation_process(command: list[str], log_path: Path, deadline: float) -> None:
    """Run a cold simulator with a deadline that includes its startup."""
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("Validation deadline reached before subprocess startup")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=max(0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired as error:
            reaped = _stop_validation_process(process, log_path)
            status = (
                "reaped" if reaped else f"owned PID {process.pid} OS reap pending after SIGKILL"
            )
            raise TimeoutError(
                f"Validation subprocess exceeded its deadline ({status}); see {log_path}"
            ) from error
        except BaseException:
            _stop_validation_process(process, log_path)
            raise
    if returncode:
        raise RuntimeError(
            f"Validation subprocess exited with status {returncode}; see {log_path}\n"
            f"{validation_error_details(log_path)}"
        )


def _validate_parallel_validation_report(report: dict, plan, log_path: Path) -> None:
    """Require native pairing proof and every correctly attributed first episode."""
    message = f"Parallel validation report is incomplete or invalid; see {log_path}"
    plan.validate_native_proof(report.get("initial_physics_pairing"))
    groups = json.loads(json.dumps(plan.groups()))
    if report.get("physical_groups") != groups:
        raise ValueError(message)
    batches = report.get("batch_results")
    if not isinstance(batches, list) or len(batches) != len(groups):
        raise ValueError(message)

    def finite(value):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    def check_summary(rows, metrics):
        if not isinstance(metrics, dict) or metrics.get("episodes") != len(rows):
            raise ValueError(message)
        if not collision_metrics_available(metrics):
            raise ValueError(message)
        fields = {
            "success_rate": ("success", "mean"),
            "failure_rate": ("failed", "mean"),
            "completion_mean": ("completion", "mean"),
            "position_rmse_m": ("position_rmse_m", "mean"),
            "angle_mean_rad": ("angle_mean_rad", "mean"),
            "collision_violation_rate": ("collision_violation", "mean"),
            "collision_audit_samples_min": ("collision_audit_samples", "min"),
            "collision_tool_hand_max_penetration_m": (
                "collision_tool_hand_max_penetration_m",
                "max",
            ),
            "collision_nonadjacent_self_max_penetration_m": (
                "collision_nonadjacent_self_max_penetration_m",
                "max",
            ),
        }
        for name, (field, reduction) in fields.items():
            values = [row[field] for row in rows]
            expected = (
                sum(values) / len(values)
                if reduction == "mean"
                else min(values)
                if reduction == "min"
                else max(values)
            )
            actual = metrics.get(name)
            if not finite(actual) or not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-8):
                raise ValueError(message)

    episodes = report.get("episode_metrics")
    if not isinstance(episodes, dict):
        raise ValueError(message)
    count = plan.per_controller * len(groups)
    for label in ("policy", "baseline"):
        rows = episodes.get(label)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(message)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(message)
            batch_index, pair_index = divmod(index, plan.per_controller)
            group = groups[batch_index]
            expected_ids = {
                "episode_index": index,
                "batch_index": batch_index,
                "pair_index": pair_index,
                "seed": group["seed"],
                "physical_env_id": group[label][pair_index],
            }
            if any(
                not isinstance(row.get(name), int)
                or isinstance(row[name], bool)
                or row[name] != value
                for name, value in expected_ids.items()
            ):
                raise ValueError(message)
            for name in (
                "score",
                "success",
                "failed",
                "completion",
                "position_rmse_m",
                "angle_mean_rad",
                "return",
                "collision_violation",
                "collision_audit_samples",
                "collision_tool_hand_max_penetration_m",
                "collision_nonadjacent_self_max_penetration_m",
            ):
                if not finite(row.get(name)):
                    raise ValueError(message)
            if (
                row.get("collision_measured") is not True
                or any(
                    row[name] not in (0, 1) for name in ("success", "failed", "collision_violation")
                )
                or not 0 <= row["completion"] <= 1
                or any(
                    row[name] < 0
                    for name in (
                        "position_rmse_m",
                        "angle_mean_rad",
                        "collision_tool_hand_max_penetration_m",
                        "collision_nonadjacent_self_max_penetration_m",
                    )
                )
                or row["collision_audit_samples"] <= 0
                or int(row["collision_audit_samples"]) != row["collision_audit_samples"]
            ):
                raise ValueError(message)
        check_summary(rows, report.get(label))
        for batch_index, group in enumerate(groups):
            batch = batches[batch_index]
            if not isinstance(batch, dict) or any(
                batch.get(name) != group[name] for name in ("batch_index", "seed")
            ):
                raise ValueError(message)
            if batch.get("controller_groups") != {
                name: group[name] for name in ("policy", "baseline")
            }:
                raise ValueError(message)
            begin = batch_index * plan.per_controller
            check_summary(rows[begin : begin + plan.per_controller], batch.get(label))


def subprocess_validation(
    env, policy, args, contract: dict, training: TrainingSpec, directory: Path, deadline: float
) -> dict:
    """Complete paired validation in a small scene while the PPO scene is paused.

    The cold child uses the same task, grasp, assets, observation/control and
    physical configuration. It cannot step or reseed the parent simulator.
    """
    directory.mkdir(parents=True, exist_ok=True)
    policy_path = directory / "policy.pt"
    report_path = directory / "evaluation.json"
    log_path = directory / "native.log"
    experiment_path = directory / "experiment.json"
    experiment = env.cfg.experiment.to_dict()
    write_json(experiment, experiment_path)
    report_path.unlink(missing_ok=True)
    export_policy(
        policy if policy is not None else ZeroResidualPolicy(contract["action_dim"]),
        policy_path,
        env,
        contract,
        controller_kind=getattr(policy, "controller_kind", "learned_ppo")
        if policy is not None
        else "optimized_grasp_fallback",
        validation_completed=False,
        improved_over_optimized_grasp=False,
    )
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("Validation deadline reached during policy export")
    task_source = getattr(args, "task", None) or getattr(env.cfg, "task", env.task.name)
    grasp_path = getattr(env.cfg, "grasp_path", None) or getattr(args, "grasp", None)
    if grasp_path is None:
        raise ValueError("Subprocess validation requires the source grasp artifact")
    command = [
        sys.executable,
        "-u",
        "-m",
        "g2a",
        "evaluate",
        "--task",
        str(task_source),
        "--config",
        str(experiment_path.resolve()),
        "--grasp",
        str(Path(grasp_path).resolve()),
        "--grasp-index",
        str(env.cfg.grasp_index),
        "--hand-asset",
        str(Path(env.cfg.hand_asset).resolve()),
        "--checkpoint",
        str(policy_path.resolve()),
        "--output",
        str(report_path.resolve()),
        "--baseline",
        "--disturbances",
        "--seed",
        str(training.validation_seed),
        "--batches",
        str(training.validation_batches),
        "--num-envs",
        str(training.validation_num_envs),
        "--device",
        str(env.device),
        "--max-wall-time",
        str(remaining),
    ]
    parallel_plan = None
    if training.validation_parallel_batches is True:
        from .evaluation_batches import PARALLEL_MATCHING_METHOD, PairedBatchPlan

        parallel_plan = PairedBatchPlan(
            tuple(training.validation_seed + i for i in range(training.validation_batches)),
            training.validation_num_envs,
        )
        command.append("--parallel-batches")
    run_validation_process(command, log_path, deadline)
    if not report_path.is_file():
        raise RuntimeError(f"Validation subprocess did not write its report; see {log_path}")
    report = json.loads(report_path.read_text())
    expected = {
        "task": env.task.name,
        "batch_seeds": [training.validation_seed + i for i in range(training.validation_batches)],
        "physical_num_envs": parallel_plan.physical_count
        if parallel_plan is not None
        else 2 * training.validation_num_envs,
        "num_envs_per_controller": training.validation_num_envs,
        "matching_method": PARALLEL_MATCHING_METHOD
        if parallel_plan is not None
        else "simultaneous_counterbalanced_mirrored_initial_physics",
        "disturbances": True,
        "policy_sha256": file_hash(policy_path),
        "experiment_sha256": config_hash(experiment),
    }
    if parallel_plan is not None:
        expected["parallel_seed_batches"] = True
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(f"Validation report provenance mismatch: {name}; see {log_path}")
    if parallel_plan is not None:
        _validate_parallel_validation_report(report, parallel_plan, log_path)
    episodes = training.validation_num_envs * training.validation_batches
    for label in ("policy", "baseline"):
        metrics = report.get(label, {})
        if metrics.get("episodes") != episodes:
            raise ValueError(f"Validation report is incomplete for {label}; see {log_path}")
        for name in (
            "position_rmse_m",
            "angle_mean_rad",
            "success_rate",
            "failure_rate",
            "completion_mean",
            "collision_violation_rate",
            "collision_audit_samples_min",
            "collision_tool_hand_max_penetration_m",
            "collision_nonadjacent_self_max_penetration_m",
        ):
            value = metrics.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name.endswith("rate") or name == "completion_mean")
                and value > 1
            ):
                raise ValueError(f"Validation report has invalid {label}.{name}; see {log_path}")
        if not collision_metrics_available(metrics):
            raise ValueError(
                f"Validation report lacks complete native collision evidence for {label}; see {log_path}"
            )
    report["validation_provenance"] = {
        **expected,
        "report": str(report_path),
        "log": str(log_path),
        "parent_scene_num_envs": env.num_envs,
        "startup_in_parent_deadline": True,
    }
    if parallel_plan is not None:
        report["validation_provenance"].update(
            initial_physics_pairing=report["initial_physics_pairing"],
            physical_groups=report["physical_groups"],
        )
    return report


def reset_training_rollout(agent) -> None:
    """Validation stepped the simulator; discard every stale rollout accumulator."""
    agent.obs = agent.env_reset()
    agent.dones.fill_(1)
    for name in ("current_rewards", "current_shaped_rewards", "current_lengths"):
        getattr(agent, name).zero_()
    if agent.is_rnn:
        for state in agent.rnn_states:
            state.zero_()


def parameter_delta(model, initial: list[torch.Tensor]) -> float:
    return (
        sum(
            (parameter.detach() - original).square().sum()
            for parameter, original in zip(model.parameters(), initial, strict=True)
        )
        .sqrt()
        .item()
    )


def check_model_finite(model) -> None:
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("PPO produced non-finite parameters or normalization buffers")


def diagonal_gaussian_kl(
    p0_mu: torch.Tensor,
    p0_sigma: torch.Tensor,
    p1_mu: torch.Tensor,
    p1_sigma: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Exact KL(p0 || p1) for independent Gaussian action components.

    A fixed epsilon added to variances biases KL negative for small exploration
    scales. Identical policies must instead have zero divergence, preventing an
    adaptive optimizer from increasing its learning rate on a negative KL.
    """
    tiny = torch.finfo(p0_sigma.dtype).tiny
    sigma0, sigma1 = p0_sigma.clamp_min(tiny), p1_sigma.clamp_min(tiny)
    ratio = sigma0 / sigma1
    displacement = (p1_mu - p0_mu) / sigma1
    per_action = torch.log(sigma1) - torch.log(sigma0)
    per_action += 0.5 * (ratio.square() + displacement.square() - 1)
    # Roundoff can be slightly negative near equality; true KL cannot be.
    kl = per_action.sum(dim=-1).clamp_min(0)
    return kl.mean() if reduce else kl


@contextmanager
def exact_ppo_kl(torch_ext):
    """Scope our numerical fix to this training call; shared SDK files stay intact."""
    previous = torch_ext.policy_kl
    torch_ext.policy_kl = diagonal_gaussian_kl
    try:
        yield
    finally:
        torch_ext.policy_kl = previous


def configure_learning_rate(agent, training: TrainingSpec) -> None:
    if training.learning_rate_schedule == "adaptive":
        agent.scheduler.min_lr = training.learning_rate_min
        agent.scheduler.max_lr = training.learning_rate_max
    # Restoring an optimizer can also restore a rate above the requested cap.
    # Apply the requested rate before the first gradient update, not afterward.
    rate = min(max(training.learning_rate, training.learning_rate_min), training.learning_rate_max)
    agent.last_lr = rate
    agent.update_lr(rate)


@torch.no_grad()
def configure_exploration(
    model, training: TrainingSpec, control: ControlSpec | None = None, *, reset: bool = False
) -> dict:
    """Initialize requested action groups or preserve learned std on normal resume."""
    log_std = model.a2c_network.sigma
    if not isinstance(log_std, torch.Tensor):
        raise ValueError("G2A PPO requires an input-independent log-standard-deviation tensor")
    if control is not None and log_std.shape != (control.action_dim,):
        raise ValueError("PPO exploration tensor does not match the configured action space")
    group_override = (
        training.initial_wrist_log_std is not None or training.initial_finger_log_std is not None
    )
    if reset and group_override and control is None:
        raise ValueError("Grouped PPO exploration requires the configured ControlSpec")
    original = log_std.detach().clone()
    if reset:
        log_std.fill_(training.initial_log_std)
        if control is not None:
            finger_start = control.finger_action_start
            if finger_start and training.initial_wrist_log_std is not None:
                log_std[:finger_start].fill_(training.initial_wrist_log_std)
            if training.initial_finger_log_std is not None:
                log_std[finger_start:].fill_(training.initial_finger_log_std)
    log_std.clamp_(training.minimum_log_std, training.maximum_log_std)
    std = log_std.exp()
    result = {
        "reset": reset,
        "clamped": bool(not torch.equal(original, log_std)),
        "minimum_log_std": training.minimum_log_std,
        "maximum_log_std": training.maximum_log_std,
        "mean_std": float(std.mean()),
        "minimum_std": float(std.min()),
        "maximum_std": float(std.max()),
        "per_action_std": std.cpu().tolist(),
    }
    if control is not None:
        finger_start = control.finger_action_start
        result.update(
            group_initialization_applied=bool(reset and group_override),
            wrist_mean_std=float(std[:finger_start].mean()) if finger_start else None,
            finger_mean_std=float(std[finger_start:].mean()),
        )
    return result


def learning_parameter_groups(model) -> dict[str, list[torch.Tensor]]:
    actor = list(DeterministicPolicy(model).parameters())
    sigma = model.a2c_network.sigma
    excluded = {id(parameter) for parameter in actor} | {id(sigma)}
    critic = [parameter for parameter in model.parameters() if id(parameter) not in excluded]
    return {"actor": actor, "critic": critic}


def learning_diagnostics(
    model, snapshots: dict, training: TrainingSpec, control: ControlSpec | None = None
) -> dict:
    groups = learning_parameter_groups(model)
    deltas = {
        name: float(
            sum(
                (parameter.detach() - initial).square().sum()
                for parameter, initial in zip(parameters, snapshots[name], strict=True)
            ).sqrt()
        )
        if parameters
        else 0.0
        for name, parameters in groups.items()
    }
    return {
        "actor_parameter_delta_l2": deltas["actor"],
        "critic_parameter_delta_l2": deltas["critic"],
        "actor_updated": deltas["actor"] > 0,
        "exploration": configure_exploration(model, training, control),
    }


class ActionDiagnostics:
    """Accumulate actual active validation means without per-step CPU syncs."""

    def __init__(self, policy, control, observation_spec=None):
        self.policy, self.control = policy, control
        self.count = self.square_sum = self.saturation_sum = None
        self.wrist_square_sum = None
        offset = 0
        for name, width in observation_fields(observation_spec or DEFAULT_OBSERVATION, control):
            if name == "phase_and_setup":
                self.setup_index = offset + width - 1
                break
            offset += width
        else:
            raise ValueError("Action diagnostics require the declared phase_and_setup field")

    @torch.no_grad()
    def __call__(self, observations):
        actions = self.policy(observations)
        active = (observations[:, self.setup_index] == 0).to(actions.dtype)
        if self.count is None:
            self.count = actions.new_zeros(())
            self.square_sum = actions.new_zeros(actions.shape[1])
            self.saturation_sum = actions.new_zeros(())
            self.wrist_square_sum = actions.new_zeros(2)
        self.count += active.sum()
        self.square_sum += (actions.square() * active[:, None]).sum(dim=0)
        self.saturation_sum += ((actions.abs() >= 1).to(actions.dtype) * active[:, None]).sum()
        if self.control.wrist_residual:
            for index, section in enumerate((slice(0, VECTOR_DIM), slice(VECTOR_DIM, TWIST_DIM))):
                squared_norm = actions[:, section].square().sum(dim=-1).clamp_max(1)
                self.wrist_square_sum[index] += (squared_norm * active).sum()
        return actions

    def summary(self) -> dict:
        if self.count is None or float(self.count) == 0:
            return {"active_observations": 0}
        rms = (self.square_sum / self.count).sqrt()
        fingers = rms[self.control.finger_action_start :]
        result = {
            "active_observations": int(self.count),
            "scope": "active validation observations, including automatic resets",
            "mean_action_rms": float(rms.square().mean().sqrt()),
            "per_action_rms": rms.cpu().tolist(),
            "requested_finger_offset_rms_rad": float(fingers.square().mean().sqrt())
            * self.control.joint_residual,
            "saturation_fraction": float(self.saturation_sum / (self.count * len(rms))),
        }
        if self.control.wrist_residual:
            wrist_rms = (self.wrist_square_sum / self.count).sqrt()
            result["requested_wrist_translation_rms_m"] = (
                float(wrist_rms[0]) * self.control.wrist_translation
            )
            result["requested_wrist_rotation_rms_rad"] = (
                float(wrist_rms[1]) * self.control.wrist_rotation
            )
        return result


def restore_checkpoint(agent, path: str | Path, device: str | torch.device) -> None:
    """Restore full PPO state on the requested device, preserving archive bytes.

    The pinned SDK loader otherwise restores each tensor on its original GPU
    index, which can be unavailable on another machine. Match its compiled
    model-key normalization, then retain its full optimizer/statistics/epoch/
    environment-state restoration through the SDK's own setter.
    """
    print(f"Restoring PPO checkpoint on {device}: {path}", flush=True)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" in checkpoint:
        checkpoint["model"] = {
            key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()
        }
    agent.set_full_state_weights(checkpoint)


def network_contract(cfg: dict) -> dict:
    params = cfg["params"]
    network, config = params["network"], params["config"]
    continuous = network["space"]["continuous"]
    return {
        "model": params["model"]["name"],
        "network": network["name"],
        "separate": network.get("separate", False),
        "mlp": {name: network["mlp"][name] for name in ("units", "activation", "d2rl")},
        "continuous": {
            name: continuous[name] for name in ("mu_activation", "sigma_activation", "fixed_sigma")
        },
        "normalize_input": config["normalize_input"],
        "normalize_value": config["normalize_value"],
    }


def export_policy(policy, path: Path, env, contract: dict, **selection) -> None:
    """Freeze normalization into the portable model and verify actual serialization."""
    policy = copy.deepcopy(policy).eval().cpu()
    width = contract["observation_dim"]
    with torch.inference_mode(), preserve_rng():
        example = torch.zeros((2, width))
        traced = torch.jit.freeze(torch.jit.trace(policy, example))
        torch.jit.save(traced, str(path))
        restored = torch.jit.load(str(path)).eval()
        for batch in (1, 7):
            sample = torch.randn((batch, width))
            expected, actual = policy(sample), restored(sample)
            if actual.shape != (batch, contract["action_dim"]) or not torch.isfinite(actual).all():
                raise RuntimeError("Exported policy has an invalid action shape or values")
            torch.testing.assert_close(expected, actual)
    experiment = env.cfg.experiment.to_dict()
    task_settings = asdict(env.task) if is_dataclass(env.task) else vars(env.task).copy()
    for name in ("object_asset", "trajectory", "area_path"):
        task_settings.pop(name, None)
    effort = getattr(env, "joint_effort_limits", None)
    if effort is None:
        effort = torch.full_like(env.lower, env.cfg.experiment.physics.joint_effort_limit_nm)
    elif effort.ndim == 2:
        effort = effort[0]
    write_json(
        {
            "contract": contract,
            "sha256": file_hash(path),
            "experiment": experiment,
            "experiment_sha256": config_hash(experiment),
            "task_settings": task_settings,
            "nominal_context": {
                "mass_kg": task_settings.get("mass"),
                "friction": env.cfg.experiment.physics.friction_dynamic,
                "load_multiplier": 1.0,
            },
            "object_physics": getattr(env, "physics_metadata", {}),
            "inference": {
                "computation": "actor only"
                if isinstance(policy, DeterministicPolicy)
                else "constant zero residual",
                "samples_actions": False,
                "evaluates_critic": False,
            },
            "joint_limits": {
                "lower": env.lower.cpu().tolist(),
                "upper": env.upper.cpu().tolist(),
                "effort_nm": effort.cpu().tolist(),
            },
            "grasp": {
                name: env.grasp[name][env.cfg.grasp_index].cpu().tolist()
                for name in ("hand_in_object", "joint_positions", "joint_targets")
            },
            "grasp_index": env.cfg.grasp_index,
            **selection,
        },
        path.with_suffix(".json"),
    )


def save_unvalidated_checkpoint(
    agent, path: Path, *, stage: str, contract: dict, experiment: dict
) -> None:
    """Preserve full optimizer/model state before infrastructure can fail.

    This is a resumable work checkpoint, with no validation or improvement
    claim. Replacing completed temporary files keeps earlier work intact if a
    write itself fails.
    """
    pending = path.with_name(f".{path.stem}.pending")
    agent.save(str(pending))
    os.replace(Path(str(pending) + ".pth"), path)
    metadata_path = path.with_suffix(".json")
    pending_metadata = metadata_path.with_name(f".{metadata_path.name}.pending")
    write_json(
        {
            "epoch": agent.epoch_num,
            "stage": stage,
            "controller_kind": "unvalidated_ppo_checkpoint",
            "validation_completed": False,
            "improved_over_optimized_grasp": False,
            "sha256": file_hash(path),
            "contract_sha256": config_hash(contract),
            "experiment_sha256": config_hash(experiment),
            "validation_num_envs_per_controller": experiment["training"]["validation_num_envs"],
        },
        pending_metadata,
    )
    os.replace(pending_metadata, metadata_path)


class _TrainingStopped(Exception):
    pass


class CompletedEpisodeStatistics:
    """Count each actual terminal row once, independent of reset batch size.

    Training samples come from changing stochastic policies and are not frozen
    policy validation. SDK scalar episode logs can repeat between resets and
    give small reset batches the same weight as large ones; use done indices
    and the saved pre-reset row measurements instead.
    """

    def __init__(self):
        self.count = 0
        self.total_count = 0
        self.sums = {}
        self.total_sums = {}

    def add(self, metrics: dict, done_indices: torch.Tensor) -> None:
        indices = done_indices.reshape(-1)
        if not len(indices):
            return
        if indices.dtype != torch.int64:
            raise ValueError("Completed episode indices must be int64")
        if self.sums and self.sums.keys() != metrics.keys():
            raise ValueError("Completed episode metric fields changed during training")
        for name, values in metrics.items():
            if values.ndim != 1:
                raise ValueError("Completed episode metrics must have one value per environment")
            observed_sum = values[indices.to(values.device)].detach().to(torch.float64).sum()
            if name not in self.sums:
                self.sums[name] = torch.zeros_like(observed_sum)
                self.total_sums[name] = torch.zeros_like(observed_sum)
            self.sums[name] += observed_sum
            self.total_sums[name] += observed_sum
        self.count += len(indices)
        self.total_count += len(indices)

    @staticmethod
    def _summary(count: int, sums: dict) -> dict:
        if not count:
            return {"completed_episodes": 0}
        values = torch.stack(list(sums.values())).cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Completed episode statistics contain nonfinite measurements")
        return {
            "completed_episodes": count,
            **{name: value / count for name, value in zip(sums, values, strict=True)},
        }

    def drain(self) -> dict:
        result = self._summary(self.count, self.sums)
        self.count = 0
        for value in self.sums.values():
            value.zero_()
        return result

    def total_summary(self) -> dict:
        return self._summary(self.total_count, self.total_sums)


class TrainingBudget:
    """Keep complete final validation inside a finite overall stage budget."""

    def __init__(self, training: TrainingSpec, started: float):
        self.deadline = started + training.max_wall_time_s
        self.final_reserve_s = training.final_validation_reserve_s or 0.0
        self.update_deadline = self.deadline - self.final_reserve_s
        self.validation_cap_s = training.validation_max_wall_time_s

    def can_start_periodic_validation(self, now: float) -> bool:
        # Legacy runs retain their original periodic behavior. An opted-in
        # run must keep its final allowance after the entire periodic cap.
        return self.final_reserve_s == 0 or (
            now + self.validation_cap_s + self.final_reserve_s <= self.deadline
        )


def train(env, args) -> None:
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from rl_games.algos_torch import torch_ext
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = environment_contract(env)
    training = training_spec(args, env.cfg.experiment.training)
    # Validate the resolved CLI overrides with the same configuration rules.
    env.cfg.experiment = replace(env.cfg.experiment, training=training)
    env.cfg.experiment.validate()
    if args.checkpoint:
        previous = json.loads(
            args.checkpoint.resolve().parent.joinpath("contract.json").read_text()
        )
        validate_policy_contract(previous, contract)
    cfg = ppo_config(
        env.num_envs, training.horizon, training.epochs, env.device, args.seed, training
    )
    if args.checkpoint:
        source_agent = args.checkpoint.resolve().parent / "agent.json"
        if source_agent.is_file() and network_contract(
            json.loads(source_agent.read_text())
        ) != network_contract(cfg):
            raise ValueError(
                "PPO checkpoint network/normalization differs from the requested configuration; "
                "resume with its original architecture or train a new policy"
            )
    write_json(env.cfg.experiment.to_dict(), output / "experiment.json")
    write_json(contract, output / "contract.json")
    cfg["params"]["config"].update(
        name=f"g2a_{env.task.name}",
        train_dir=str(output.parent),
        full_experiment_name=output.name,
    )
    clip_obs = env.cfg.experiment.observation.clip
    wrapped = RlGamesVecEnvWrapper(env, env.device, clip_obs=clip_obs, clip_actions=1.0)
    vecenv.register(
        "G2A", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw)
    )
    env_configurations.register("g2a", {"vecenv_type": "G2A", "env_creator": lambda **kw: wrapped})

    completed_training_episodes = CompletedEpisodeStatistics()

    class ValidationObserver(IsaacAlgoObserver):
        def process_infos(self, infos, done_indices):
            # Remove stale, batch-averaged packets from the SDK episode logger.
            # Keep all unrelated direct diagnostics and score handling.
            super().process_infos(
                {key: value for key, value in infos.items() if key != "episode"}, done_indices
            )
            completed_training_episodes.add(env.last_metrics, done_indices)

        def after_print_stats(self, frame, epoch_num, total_time):
            nonlocal periodic_checks_deferred
            super().after_print_stats(frame, epoch_num, total_time)
            episode_window = completed_training_episodes.drain()
            if episode_window["completed_episodes"]:
                for name, value in episode_window.items():
                    self.writer.add_scalar(f"TrainingEpisode/{name}", value, epoch_num)
            check_model_finite(self.algo.model)
            configure_exploration(self.algo.model, training, env.cfg.experiment.control)
            if (
                training.checkpoint_every_epochs is not None
                and (epoch_num - start_epoch) % training.checkpoint_every_epochs == 0
            ):
                # Preserve completed optimizer updates independently of the
                # slower full-trajectory qualification schedule. This remains
                # an explicitly unvalidated recovery checkpoint.
                save_unvalidated_checkpoint(
                    self.algo,
                    output / "latest_checkpoint.pth",
                    stage="periodic_completed_epoch",
                    contract=contract,
                    experiment=env.cfg.experiment.to_dict(),
                )
            if time.perf_counter() >= budget.update_deadline:
                raise _TrainingStopped(
                    "final_validation_reserve" if budget.final_reserve_s else "wall_time_limit"
                )
            if (epoch_num - start_epoch) % training.validation_every_epochs == 0:
                if budget.can_start_periodic_validation(time.perf_counter()):
                    validate_candidate(self.algo)
                else:
                    periodic_checks_deferred += 1
            if time.perf_counter() >= budget.update_deadline:
                raise _TrainingStopped(
                    "final_validation_reserve" if budget.final_reserve_s else "wall_time_limit"
                )
            if (
                epoch_num - start_epoch >= training.minimum_epochs
                and stale_validations >= training.early_stopping_patience
            ):
                raise _TrainingStopped("validation_plateau")

    runner = Runner(ValidationObserver())
    runner.load(cfg)
    runner.reset()
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    if args.checkpoint:
        restore_checkpoint(agent, args.checkpoint.resolve(), env.device)
    check_model_finite(agent.model)
    initial_exploration = configure_exploration(
        agent.model,
        training,
        env.cfg.experiment.control,
        reset=bool(not args.checkpoint or training.reset_exploration_on_resume),
    )
    configure_learning_rate(agent, training)
    start_epoch = agent.epoch_num
    incoming_kind = "learned_ppo"

    def inference_policy():
        policy = DeterministicPolicy(agent.model, clip_obs).eval()
        policy.controller_kind = incoming_kind
        return policy

    agent.max_epochs = start_epoch + training.epochs
    agent.save_best_after = agent.max_epochs + 1
    cfg["params"]["config"]["max_epochs"] = agent.max_epochs
    cfg["params"]["config"]["save_best_after"] = agent.save_best_after
    write_json(cfg, output / "agent.json")
    initial_parameters = [parameter.detach().clone() for parameter in agent.model.parameters()]
    learning_snapshots = {
        name: [parameter.detach().clone() for parameter in parameters]
        for name, parameters in learning_parameter_groups(agent.model).items()
    }
    started = time.perf_counter()
    budget = TrainingBudget(training, started)
    deadline = budget.deadline
    validation_elapsed_s = 0.0
    periodic_checks_deferred = 0
    seeds = [training.validation_seed + index for index in range(training.validation_batches)]
    reward = env.cfg.experiment.reward
    collision = env.cfg.experiment.collision
    external_validation = training.validation_num_envs is not None

    def collect_validation(policy, label: str):
        nonlocal validation_elapsed_s
        validation_started = time.perf_counter()
        validation_deadline = min(
            deadline, time.perf_counter() + training.validation_max_wall_time_s
        )
        try:
            if external_validation:
                paired = subprocess_validation(
                    env,
                    policy,
                    args,
                    contract,
                    training,
                    output / "validation" / label,
                    validation_deadline,
                )
                return paired["baseline" if policy is None else "policy"], paired
            return full_validation(env, policy, seeds, validation_deadline), None
        finally:
            validation_elapsed_s += time.perf_counter() - validation_started

    save_unvalidated_checkpoint(
        agent,
        output / "latest_checkpoint.pth",
        stage="before_baseline",
        contract=contract,
        experiment=env.cfg.experiment.to_dict(),
    )
    shared_resume_validation = bool(
        args.checkpoint and external_validation and budget.final_reserve_s
    )
    shared_initialization = shared_initialization_report = None
    if shared_resume_validation:
        # This scene already measures both the incoming actor and its zero
        # reference under mirrored, counterbalanced initial conditions. Keep
        # its actual zero rows as the original reference, saving an otherwise
        # duplicate cold full-trajectory replay.
        was_training = agent.model.training
        try:
            shared_initialization, shared_initialization_report = collect_validation(
                inference_policy(), "initialization"
            )
        finally:
            agent.model.train(was_training)
        baseline = shared_initialization_report["baseline"]
        baseline_report = shared_initialization_report
    else:
        baseline, baseline_report = collect_validation(None, "baseline")
    history = []
    best_learned = None
    best_improving = None
    stale_validations = 0
    initialization = baseline
    initialization_report = baseline_report
    initialization_qualified = False
    initialization_actor_trained = False
    selection_reference = baseline
    if args.checkpoint:
        # A resumed model may already be better than the zero controller. Keep
        # it unless this run produces a better validated model.
        if shared_resume_validation:
            initialization = shared_initialization
            initialization_report = shared_initialization_report
        else:
            was_training = agent.model.training
            try:
                initialization, initialization_report = collect_validation(
                    inference_policy(), "initialization"
                )
            finally:
                agent.model.train(was_training)
        # A work checkpoint can be saved before the first optimizer update.
        # Its zero actor may score differently from another zero controller
        # because native clones differ. Such variation cannot qualify an
        # untrained initialization as an improved learned policy. A positive
        # saved epoch and a nonzero mean head are both necessary; a trained
        # zero actor also remains the optimized-grasp controller in behavior.
        initialization_actor_trained = start_epoch > 0 and any(
            bool(torch.count_nonzero(parameter.detach()))
            for parameter in agent.model.a2c_network.mu.parameters()
        )
        improved = initialization_actor_trained and improves_baseline(
            initialization, baseline, training, reward, collision
        )
        if initialization_report is not None:
            improved = improved and improves_baseline(
                initialization, initialization_report["baseline"], training, reward, collision
            )
        initialization_qualified = improved
        if initialization_qualified:
            selection_reference = initialization
        row = {
            "epoch": start_epoch,
            "stage": "resumed_initialization",
            "controller_kind": incoming_kind,
            "completed": True,
            "metrics": initialization,
            "tracking_cost": tracking_cost(initialization, reward),
            "improves_optimized_grasp": improved,
            "improves_initialization": False,
            "collision_validation_passed": collision_safe(initialization, collision),
            "learning": learning_diagnostics(
                agent.model, learning_snapshots, training, env.cfg.experiment.control
            ),
            "paired_baseline": initialization_report["baseline"]
            if initialization_report is not None
            else None,
            "validation_provenance": initialization_report["validation_provenance"]
            if initialization_report is not None
            else None,
            "state": {
                key: value.detach().cpu().clone() for key, value in agent.model.state_dict().items()
            },
        }
        if initialization_actor_trained:
            best_learned = row
            agent.save(str(output / "learned_checkpoint"))
        if improved:
            best_improving = row
            agent.save(str(output / "selected_checkpoint"))

    def validate_candidate(algo) -> None:
        nonlocal best_learned, best_improving, stale_validations
        check_model_finite(algo.model)
        save_unvalidated_checkpoint(
            algo,
            output / "latest_checkpoint.pth",
            stage="before_candidate_validation",
            contract=contract,
            experiment=env.cfg.experiment.to_dict(),
        )
        was_training = algo.model.training
        policy = DeterministicPolicy(algo.model, clip_obs).eval()
        if not external_validation:
            policy = ActionDiagnostics(
                policy, env.cfg.experiment.control, env.cfg.experiment.observation
            )
        try:
            candidate, paired_report = collect_validation(policy, f"epoch-{algo.epoch_num}")
        except TimeoutError as error:
            history.append({"epoch": algo.epoch_num, "error": str(error), "completed": False})
            raise _TrainingStopped("validation_wall_time_limit") from error
        finally:
            algo.model.train(was_training)
            if not external_validation:
                reset_training_rollout(algo)
        learning = learning_diagnostics(
            algo.model, learning_snapshots, training, env.cfg.experiment.control
        )
        improved = (
            improves_baseline(candidate, baseline, training, reward, collision)
            and does_not_regress(candidate, selection_reference, training, collision, reward=reward)
            and learning["actor_updated"]
        )
        if paired_report is not None:
            improved = improved and improves_baseline(
                candidate, paired_report["baseline"], training, reward, collision
            )
        row = {
            "epoch": algo.epoch_num,
            "stage": "ppo_validation",
            "controller_kind": "learned_ppo",
            "elapsed_s": time.perf_counter() - started,
            "completed": True,
            "metrics": candidate,
            "tracking_cost": tracking_cost(candidate, reward),
            "improves_optimized_grasp": improved,
            "collision_validation_passed": collision_safe(candidate, collision),
            "learning": learning,
            "actions": paired_report.get("actions", {})
            if paired_report is not None
            else policy.summary(),
            "paired_baseline": paired_report["baseline"] if paired_report is not None else None,
            "validation_provenance": paired_report["validation_provenance"]
            if paired_report is not None
            else None,
            "common_completed_pairs": paired_report.get("common_completed_pairs")
            if paired_report is not None
            else None,
            "improves_initialization": improves_baseline(
                candidate, initialization, training, reward, collision
            ),
        }
        history.append(row)
        snapshot = None

        def learned_rank(validation):
            metrics = validation["metrics"]
            safe = does_not_regress(
                metrics, baseline, training, collision, reward=reward
            ) and does_not_regress(metrics, selection_reference, training, collision, reward=reward)
            if validation.get("paired_baseline") is not None:
                safe = safe and does_not_regress(
                    metrics, validation["paired_baseline"], training, collision, reward=reward
                )
            return (safe, *candidate_rank(metrics, reward))

        if best_learned is None or learned_rank(row) > learned_rank(best_learned):
            snapshot = {
                key: value.detach().cpu().clone() for key, value in algo.model.state_dict().items()
            }
            best_learned = {**row, "state": snapshot}
            algo.save(str(output / "learned_checkpoint"))
        if improved and (
            best_improving is None
            or candidate_rank(candidate, reward) > candidate_rank(best_improving["metrics"], reward)
        ):
            if snapshot is None:
                snapshot = {
                    key: value.detach().cpu().clone()
                    for key, value in algo.model.state_dict().items()
                }
            best_improving = {**row, "state": snapshot}
            algo.save(str(output / "selected_checkpoint"))
            stale_validations = 0
        else:
            stale_validations += 1
        write_json(
            {
                "baseline": baseline,
                "initialization": initialization,
                "batch_seeds": seeds,
                "history": history,
            },
            output / "validation.json",
        )
        print(
            f"Full validation epoch {algo.epoch_num}: success={candidate['success_rate']:.1%}, "
            f"completion={candidate['completion_mean']:.1%}, "
            f"tracking={row['tracking_cost']:.4f}, improves grasp={improved}, "
            f"actor delta={learning['actor_parameter_delta_l2']:.5g}, "
            f"std={learning['exploration']['mean_std']:.4g}",
            flush=True,
        )

    stop_reason = "epoch_limit"
    try:
        with exact_ppo_kl(torch_ext):
            agent.train()
    except _TrainingStopped as error:
        stop_reason = str(error)
    # Stopping updates for the reserve must still test the final committed
    # weights. Do not retry an already attempted/timed-out check at this epoch.
    if (not history or history[-1]["epoch"] != agent.epoch_num) and time.perf_counter() < deadline:
        try:
            validate_candidate(agent)
        except _TrainingStopped as error:
            stop_reason = str(error)
    check_model_finite(agent.model)
    delta = parameter_delta(agent.model, initial_parameters)
    final_learning = learning_diagnostics(
        agent.model, learning_snapshots, training, env.cfg.experiment.control
    )
    if delta == 0:
        raise RuntimeError("PPO completed without updating policy parameters")
    agent.save(str(output / "checkpoint"))
    validation_details = {
        "initialization_qualified": initialization_qualified,
        "initialization_actor_trained": initialization_actor_trained,
        "selection_reference_kind": "qualified_resumed_initialization"
        if initialization_qualified
        else "optimized_grasp_baseline",
        "validation_method": "subprocess_paired_small_scene"
        if external_validation
        else "in_process_full_scene",
        "validation_num_envs_per_controller": training.validation_num_envs
        if external_validation
        else env.num_envs,
        "collision_validation_required": True,
        "selection_criteria": {
            "tracking_mode": training.selection_tracking_mode,
            "minimum_relative_tracking_gain": training.selection_min_tracking_gain,
            "position_weight": reward.position_weight,
            "angle_weight": reward.angle_weight,
            "position_scale_m": reward.position_scale_m,
            "angle_scale_rad": reward.angle_scale_rad,
            "success_tolerance": training.selection_success_tolerance,
            "completion_tolerance": training.selection_completion_tolerance,
        },
        "collision_acceptance_budgets_m": {
            "tool_hand": collision.max_tool_penetration_m,
            "nonadjacent_self": collision.max_self_penetration_m,
        },
    }
    if best_learned is not None:
        agent.model.load_state_dict(best_learned["state"])
        export_policy(
            DeterministicPolicy(agent.model, clip_obs),
            output / "learned_policy.pt",
            env,
            contract,
            controller_kind=best_learned.get("controller_kind", "learned_ppo"),
            selected_epoch=best_learned["epoch"],
            validation=best_learned["metrics"],
            validation_completed=True,
            collision_validation_passed=collision_safe(best_learned["metrics"], collision),
            improved_over_optimized_grasp=best_learned["improves_optimized_grasp"],
            improved_over_initialization=best_learned["improves_initialization"],
            validation_provenance=best_learned.get("validation_provenance"),
            **validation_details,
        )
    else:
        # A tight cap can end before the first candidate validation. Keep the
        # actual learned model without selecting or endorsing it.
        export_policy(
            DeterministicPolicy(agent.model, clip_obs),
            output / "learned_policy.pt",
            env,
            contract,
            controller_kind="learned_ppo",
            selected_epoch=agent.epoch_num,
            validation=None,
            validation_completed=False,
            collision_validation_passed=False,
            improved_over_optimized_grasp=False,
            improved_over_initialization=False,
            **validation_details,
        )
    if best_improving is None:
        selected = ZeroResidualPolicy(contract["action_dim"])
        selected_kind = "optimized_grasp_fallback"
        selected_epoch = None
        selected_validation = baseline
    else:
        agent.model.load_state_dict(best_improving["state"])
        selected = DeterministicPolicy(agent.model, clip_obs)
        selected_kind = best_improving.get("controller_kind", "learned_ppo")
        selected_epoch = best_improving["epoch"]
        selected_validation = best_improving["metrics"]
    export_policy(
        selected,
        output / "policy.pt",
        env,
        contract,
        controller_kind=selected_kind,
        selected_epoch=selected_epoch,
        validation=selected_validation,
        validation_completed=True,
        improved_over_optimized_grasp=best_improving is not None,
        improved_over_initialization=improves_baseline(
            selected_validation, initialization, training, reward, collision
        ),
        collision_validation_passed=collision_safe(selected_validation, collision),
        validation_provenance=best_improving.get("validation_provenance")
        if best_improving is not None
        else baseline_report["validation_provenance"]
        if baseline_report is not None
        else None,
        **validation_details,
    )
    report = {
        "task": env.task.name,
        "epochs": agent.epoch_num,
        "epochs_this_run": agent.epoch_num - start_epoch,
        "initialization_controller_kind": incoming_kind if args.checkpoint else None,
        "elapsed_s": time.perf_counter() - started,
        "time_budget": {
            "max_wall_time_s": training.max_wall_time_s,
            "final_validation_reserve_s": training.final_validation_reserve_s,
            "update_cutoff_elapsed_s": budget.update_deadline - started,
            "validation_elapsed_s": validation_elapsed_s,
            "periodic_checks_deferred": periodic_checks_deferred,
            "shared_resumed_initialization_and_original_baseline": shared_resume_validation,
        },
        "parameter_delta_l2": delta,
        "learning": final_learning,
        "initial_exploration": initial_exploration,
        "stop_reason": stop_reason,
        "training_episode_statistics": {
            "scope": "all actual terminal episodes across changing stochastic training policies",
            "weighting": "one contribution per completed environment episode; no stale log packets",
            "frozen_policy_validation": False,
            **completed_training_episodes.total_summary(),
        },
        "training": replace(env.cfg.experiment, training=training).to_dict()["training"],
        "ppo_stability": {
            "gaussian_kl": "exact diagonal Gaussian KL without biased variance epsilon",
            "learning_rate_schedule": training.learning_rate_schedule,
            "learning_rate_min": training.learning_rate_min,
            "learning_rate_max": training.learning_rate_max,
            "actor_critic_separate": training.actor_critic_separate,
            "zero_residual_regularization": training.zero_residual_regularization,
            "zero_residual_penalty": "coefficient * sum(deterministic_action_mean ** 2)",
        },
        "validation_batch_seeds": seeds,
        "validation_method": "subprocess_paired_small_scene"
        if external_validation
        else "in_process_full_scene",
        "validation_num_envs_per_controller": training.validation_num_envs
        if external_validation
        else env.num_envs,
        "baseline_validation_provenance": baseline_report["validation_provenance"]
        if baseline_report is not None
        else None,
        "initialization_validation_provenance": initialization_report["validation_provenance"]
        if initialization_report is not None
        else None,
        "baseline_validation": baseline,
        "initialization_validation": initialization,
        "initialization_qualified": initialization_qualified,
        "selection_reference_kind": "qualified_resumed_initialization"
        if initialization_qualified
        else "optimized_grasp_baseline",
        "selection_reference_validation": selection_reference,
        "selected_validation": selected_validation,
        "collision_validation_passed": collision_safe(selected_validation, collision),
        "validation_history": history,
        "controller_kind": selected_kind,
        "selected_epoch": selected_epoch,
        "improved_over_optimized_grasp": best_improving is not None,
        "improved_over_initialization": improves_baseline(
            selected_validation, initialization, training, reward, collision
        ),
        "policy": "policy.pt",
        "learned_policy": "learned_policy.pt",
        "learned_validation_completed": best_learned is not None,
        "checkpoint": "checkpoint.pth",
        "latest_checkpoint": "latest_checkpoint.pth",
    }
    write_json(report, output / "training.json")
    print(f"Saved {selected_kind} controller and PPO training artifacts to {output}", flush=True)


def load_policy(path: Path, env):
    metadata = json.loads(path.with_suffix(".json").read_text())
    contract = environment_contract(env)
    validate_policy_contract(metadata["contract"], contract)
    if metadata["sha256"] != file_hash(path):
        raise ValueError("Policy file does not match its metadata checksum")
    policy = torch.jit.load(str(path), map_location=env.device).eval()
    with torch.inference_mode():
        probe = policy(torch.zeros((1, contract["observation_dim"]), device=env.device))
    if (
        probe.shape != (1, contract["action_dim"])
        or not torch.isfinite(probe).all()
        or (probe.abs() > 1).any()
    ):
        raise ValueError("Policy output does not satisfy its bounded action contract")
    return policy


def _parallel_paired_evaluation(env, policy, plan, results, kwargs):
    """Collect all seed/controller groups once; retain every first episode."""
    env.seed(results["seed"])
    controller = plan.controller(policy, env.cfg.control.action_dim, env.device)
    metrics = first_episodes(env, controller, **kwargs)[0]
    proof = getattr(env, "parallel_evaluation_initial_physics", None)
    plan.validate_native_proof(proof)
    results["initial_physics_pairing"] = proof
    groups = {"policy": [], "baseline": []}
    for index, (group, packet) in enumerate(
        zip(plan.groups(), plan.split_metrics(metrics), strict=True)
    ):
        row = results["batch_results"][index]
        row["controller_groups"] = {label: list(group[label]) for label in groups}
        for label in groups:
            groups[label].append(packet[label])
            row[label] = summarize(packet[label])
        row["common_completed_pairs"] = paired_summary(packet["policy"], packet["baseline"])
    merged = {
        label: {name: torch.cat([batch[name] for batch in batches]) for name in batches[0]}
        for label, batches in groups.items()
    }
    results["episode_metrics"] = {}
    for label, packet in merged.items():
        results[label] = summarize(packet)
        rows = episode_rows(packet)
        for index, row in enumerate(rows):
            batch_index, pair_index = divmod(index, plan.per_controller)
            group = plan.groups()[batch_index]
            row.update(
                batch_index=batch_index,
                seed=group["seed"],
                pair_index=pair_index,
                physical_env_id=group[label][pair_index],
            )
        results["episode_metrics"][label] = rows
    results["common_completed_pairs"] = paired_summary(merged["policy"], merged["baseline"])


def evaluate(env, args) -> dict:
    started = time.perf_counter()
    try:
        results = _evaluate(env, args)
    except (TimeoutError, RuntimeError) as error:
        diagnostics = getattr(error, "verification_diagnostics", None)
        if not isinstance(error, TimeoutError) and not isinstance(diagnostics, dict):
            raise
        seen = diagnostics.get("completed_mask") if isinstance(diagnostics, dict) else None
        measured = isinstance(seen, torch.Tensor) and seen.dtype == torch.bool
        write_json(
            {
                "completed": False,
                "task": env.task.name,
                "error": str(error),
                "error_type": type(error).__name__,
                "elapsed_s": time.perf_counter() - started,
                "first_episode_scope": "interrupted_collection",
                "first_episode_completed": int(seen.sum()) if measured else None,
                "first_episode_total": seen.numel() if measured else None,
            },
            args.output,
        )
        raise
    results["completed"] = True
    write_json(results, args.output)
    print(json.dumps(results, indent=2), flush=True)
    return results


def _evaluate(env, args) -> dict:
    started = time.perf_counter()
    experiment = getattr(env.cfg, "experiment", None) if hasattr(env, "cfg") else None
    count = getattr(args, "batches", None)
    if count is None:
        count = experiment.evaluation.batches if experiment is not None else 1
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = experiment.evaluation.seed if experiment is not None else 42
    policy = load_policy(args.checkpoint.resolve(), env) if args.checkpoint else None
    action_diagnostics = None
    if (
        policy is not None
        and experiment is not None
        and hasattr(experiment, "control")
        and hasattr(experiment, "observation")
    ):
        action_diagnostics = ActionDiagnostics(policy, experiment.control, experiment.observation)
        policy = action_diagnostics
    paired = bool(
        policy is not None
        and args.baseline
        and getattr(getattr(env, "cfg", None), "paired_evaluation", False)
    )
    physical_count = env.num_envs
    if paired and (physical_count < 2 or physical_count % 2):
        raise ValueError("Paired evaluation requires an even physical environment count")
    parallel = bool(getattr(args, "parallel_batches", False))
    plan = getattr(env, "_parallel_evaluation_plan", None)
    if parallel != (plan is not None) or parallel and not paired:
        raise ValueError(
            "Parallel batches require a matching explicitly planned paired evaluation scene"
        )
    per_controller = (
        plan.per_controller if parallel else physical_count // 2 if paired else physical_count
    )
    batch_seeds = [seed + index for index in range(count)]
    if parallel and (tuple(batch_seeds) != plan.seeds or physical_count != plan.physical_count):
        raise ValueError(
            "Parallel scene groups do not match the requested seeds and episode counts"
        )
    results = {
        "task": env.task.name,
        "seed": seed,
        "batch_seeds": batch_seeds,
        "disturbances": bool(getattr(args, "disturbances", False)),
        "physical_num_envs": physical_count,
        "num_envs_per_controller": per_controller,
        "matching_method": (
            "simultaneous_counterbalanced_mirrored_initial_physics"
            if paired
            else "sequential_reseeded_initial_physics"
            if policy is not None and args.baseline
            else "single_controller_seeded_initial_physics"
        ),
        "batch_results": [
            {"batch_index": index, "seed": seed} for index, seed in enumerate(batch_seeds)
        ],
    }
    if args.checkpoint:
        sidecar = args.checkpoint.resolve().with_suffix(".json")
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text())
            results["controller_kind"] = metadata.get("controller_kind", "learned_ppo")
            results["policy_sha256"] = metadata.get("sha256")
    if parallel:
        from .evaluation_batches import PARALLEL_MATCHING_METHOD

        results.update(
            matching_method=PARALLEL_MATCHING_METHOD,
            parallel_seed_batches=True,
            physical_groups=plan.groups(),
            scene_layout_note="Concurrent batches use a larger physical scene; these are fresh paired measurements, not bitwise-equivalent to the sequential scene.",
        )
    if experiment is not None:
        results["experiment_sha256"] = config_hash(experiment.to_dict())
    limit = getattr(args, "max_wall_time", None)
    if limit is None and experiment is not None:
        limit = experiment.evaluation.max_wall_time_s
    deadline = time.perf_counter() + limit if limit is not None else None
    kwargs = {"deadline": deadline} if deadline is not None else {}

    def merge(batches):
        return {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}

    if parallel:
        _parallel_paired_evaluation(env, policy, plan, results, kwargs)
        results["elapsed_s"] = time.perf_counter() - started
        if action_diagnostics is not None:
            results["actions"] = action_diagnostics.summary()
        return results
    if paired:
        # The environment mirrors the initial mass/friction/load draw into the
        # second half on each full reset. Later failure resets are independent;
        # first_episodes preserves only the initially matched episode outcomes.
        halves = (slice(0, per_controller), slice(per_controller, physical_count))
        groups = {"policy": [], "baseline": []}
        for index, seed in enumerate(batch_seeds):
            # Absolute scene placement and GPU contact arithmetic can affect
            # clones. Alternate assignments without changing matched draws.
            policy_first = index % 2 == 0
            policy_section = halves[0 if policy_first else 1]
            baseline_section = halves[1 if policy_first else 0]

            def paired_controller(obs, section=policy_section, first=policy_first):
                actions = policy(obs[section])
                if actions.shape != (per_controller, env.cfg.control.action_dim):
                    raise ValueError(
                        "Policy action shape does not match the paired controller group"
                    )
                zero = torch.zeros_like(actions)
                return torch.cat((actions, zero) if first else (zero, actions), dim=0)

            env.seed(seed)
            metrics = first_episodes(env, paired_controller, **kwargs)[0]
            if any(value.shape != (physical_count,) for value in metrics.values()):
                raise ValueError("Paired episode metrics must cover every physical environment")
            results["batch_results"][index]["controller_groups"] = {
                "policy": "first_half" if policy_first else "second_half",
                "baseline": "second_half" if policy_first else "first_half",
            }
            for label, section in (("policy", policy_section), ("baseline", baseline_section)):
                batch = {key: value[section].clone() for key, value in metrics.items()}
                groups[label].append(batch)
                results["batch_results"][index][label] = summarize(batch)
            results["batch_results"][index]["common_completed_pairs"] = paired_summary(
                groups["policy"][-1], groups["baseline"][-1]
            )
        for label, batches in groups.items():
            results[label] = summarize(merge(batches))
        merged_policy, merged_baseline = merge(groups["policy"]), merge(groups["baseline"])
        results["common_completed_pairs"] = paired_summary(merged_policy, merged_baseline)
        results["episode_metrics"] = {
            "policy": episode_rows(merged_policy),
            "baseline": episode_rows(merged_baseline),
        }
        results["elapsed_s"] = time.perf_counter() - started
        if action_diagnostics is not None:
            results["actions"] = action_diagnostics.summary()
        return results
    for label, controller in (("policy", policy), ("baseline", None)):
        if label == "policy" and policy is None:
            continue
        if label == "baseline" and not args.baseline:
            continue
        # Failed episodes may reset while others finish. Reseed each paired
        # batch so extra failure resets never change the next batch's loads.
        batches = []
        for index, seed in enumerate(batch_seeds):
            env.seed(seed)
            batch = first_episodes(env, controller, **kwargs)[0]
            batches.append(batch)
            results["batch_results"][index][label] = summarize(batch)
        results[label] = summarize(merge(batches))
        results.setdefault("episode_metrics", {})[label] = episode_rows(merge(batches))
    results["elapsed_s"] = time.perf_counter() - started
    if action_diagnostics is not None:
        results["actions"] = action_diagnostics.summary()
    return results
