"""Commands for preparing tasks, optimizing grasps, and learning residual policies."""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from .device import isolate_gpu, launch_isolated_app
from .tasks import RELEASE_TASKS, RESOURCE_ROOT, TaskSpec

HAND_ASSET = RESOURCE_ROOT / "assets/leap_hand/leap.usd"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="g2a", description="Area grasp optimization and pose-conditioned residual PPO"
    )
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Check tasks, grasp areas and trajectories without Isaac Sim"
    )
    validate.add_argument(
        "--task", action="append", help="Task name or JSON manifest; repeat for several"
    )
    validate.add_argument("--config", type=Path, help="Experiment JSON overrides")
    convert = commands.add_parser(
        "import-trajectory", help="Pack numbered pose matrices into one NPZ"
    )
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path, help="Output NPZ file path")
    convert.add_argument("--fps", type=positive_float, default=30.0)
    mesh = commands.add_parser(
        "convert-mesh", help="Convert an OBJ/STL/FBX into a portable rigid-body USD"
    )
    mesh.add_argument("source", type=Path)
    mesh.add_argument("output", type=Path, help="Output USD file path")
    mesh.add_argument("--mass", type=positive_float, required=True, help="Kilograms")
    mesh.add_argument("--scale", type=positive_float, default=1.0, help="Mesh units to meters")
    mesh.add_argument("--device", default="cuda:0")
    mesh.set_defaults(gui=False)

    discover = commands.add_parser(
        "discover-area", help="Infer grasp-area boxes from mesh and hand geometry"
    )
    discover.add_argument("--task", required=True)
    discover.add_argument(
        "--output", type=Path, help="Output JSON file path; defaults to the task's area_path"
    )
    discover.add_argument("--hand-asset", type=Path, default=HAND_ASSET)
    discover.add_argument("--settings", type=Path, help="DiscoveryConfig JSON overrides")
    discover.add_argument("--device", default="cuda:0")
    discover.set_defaults(gui=False)

    from .visualize import add_arguments

    add_arguments(commands.add_parser("visualize", help="View a grasp or its full trajectory"))

    for command in ("optimize", "verify-grasps", "train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--task", help="Task name or task JSON manifest (default: hammer)")
        sub.add_argument("--config", type=Path, help="Experiment JSON overrides")
        sub.add_argument("--num-envs", type=positive_int)
        sub.add_argument("--hand-asset", type=Path, default=HAND_ASSET)
        sub.add_argument(
            "--device",
            default="cuda:0",
            help="GPU ordinal in CUDA_VISIBLE_DEVICES, or host GPU index when unmasked",
        )
        sub.add_argument("--seed", type=int)
        sub.add_argument("--gui", action="store_true")
        sub.add_argument(
            "--max-wall-time",
            type=positive_float,
            help="Finite stage budget in seconds, excluding simulator startup",
        )
        if command == "optimize":
            sub.add_argument("--iterations", type=positive_int)
            sub.add_argument("--keep", type=positive_int)
            warm = sub.add_mutually_exclusive_group()
            warm.add_argument(
                "--warm-start", type=Path, help="Reclose and retest a matching grasp artifact"
            )
            warm.add_argument(
                "--warm-proposals",
                type=Path,
                help="Use only matching normalized search parameters; rebuild all states and audits under current settings",
            )
            sub.add_argument(
                "--output",
                type=Path,
                help="Grasp artifact file path; defaults to runs/TASK/grasps.pt",
            )
        elif command == "verify-grasps":
            sub.add_argument("--grasp", type=Path, required=True)
            sub.add_argument(
                "--output",
                type=Path,
                help="Grasp artifact file path; defaults to the input artifact",
            )
        elif command in ("train", "evaluate"):
            sub.add_argument("--grasp", type=Path, required=command == "train")
            sub.add_argument("--grasp-index", type=int)
            sub.add_argument(
                "--checkpoint", type=Path, help="Resume PPO .pth (train) or exported .pt (evaluate)"
            )
            sub.add_argument(
                "--output",
                type=Path,
                help=(
                    "Training directory path; defaults to runs/TASK/ppo"
                    if command == "train"
                    else "Evaluation JSON file path; defaults to runs/TASK/evaluation.json"
                ),
            )
            if command == "train":
                sub.add_argument("--iterations", type=positive_int)
                sub.add_argument("--horizon", type=positive_int)
                sub.add_argument("--validation-every", type=positive_int)
                sub.add_argument("--validation-batches", type=positive_int)
            else:
                sub.add_argument(
                    "--model",
                    type=Path,
                    help="Model directory containing grasps.pt, policy.pt and experiment.json",
                )
                sub.add_argument(
                    "--baseline",
                    action="store_true",
                    help="Compare the optimized grasp with zero residuals",
                )
                sub.add_argument("--batches", type=positive_int)
                sub.add_argument(
                    "--parallel-batches",
                    action="store_true",
                    help="Evaluate independent counterbalanced seed batches concurrently; requires --checkpoint and --baseline",
                )
                sub.add_argument(
                    "--disturbances",
                    action="store_true",
                    help="Randomize mass, friction and task loads using the configuration",
                )
    return root


def _area_output(args, task) -> Path:
    output = args.output or task.area_path
    if output is None or output.suffix != ".json":
        raise ValueError("Provide --output ending in .json or set the task's area_path")
    return output


def _validate_output_path(output: Path, *, directory: bool = False) -> None:
    """Reject incompatible existing destinations/parents before expensive work."""
    kind = "directory" if directory else "file"
    if output.exists() or output.is_symlink():
        matches = output.is_dir() if directory else output.is_file()
        if not matches:
            raise ValueError(f"Output must be a {kind} path: {output}")
    for parent in output.parents:
        if parent.exists() or parent.is_symlink():
            if not parent.is_dir():
                raise ValueError(f"Output parent must be a directory: {parent}")
            break


def _stage_output(args, task) -> Path | None:
    defaults = {
        "optimize": Path(f"runs/{task.name}/grasps.pt"),
        "verify-grasps": getattr(args, "grasp", None),
        "train": Path(f"runs/{task.name}/ppo"),
        "evaluate": Path(f"runs/{task.name}/evaluation.json"),
    }
    return getattr(args, "output", None) or defaults.get(args.command)


def main(argv=None) -> None:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.command == "visualize":
        from .visualize import run

        run(args)
        return
    if args.command == "evaluate":
        if args.model is not None:
            try:
                metadata = json.loads((args.model / "policy.json").read_text())
                model_task = metadata["contract"]["task"]
                if not isinstance(model_task, str) or not model_task:
                    raise ValueError("policy.json is missing the task name")
                args.task = args.task or model_task
                args.config = args.config or args.model / "experiment.json"
                args.grasp = args.grasp or args.model / "grasps.pt"
                args.checkpoint = args.checkpoint or args.model / "policy.pt"
                if args.grasp_index is None:
                    args.grasp_index = metadata.get("grasp_index", 0)
            except (OSError, ValueError, KeyError, TypeError) as error:
                argument_parser.error(f"Cannot load model directory {args.model}: {error}")
        if args.grasp is None:
            argument_parser.error("evaluate needs --model or --grasp")
    if hasattr(args, "grasp_index") and args.grasp_index is None:
        args.grasp_index = 0
    if args.command not in ("validate", "import-trajectory", "convert-mesh", "discover-area"):
        args.task = args.task or "hammer"
    if args.command == "optimize" and args.num_envs is not None and args.num_envs < 4:
        argument_parser.error("Optimization needs at least four environments")
    if args.command == "evaluate" and args.checkpoint is None and not args.baseline:
        argument_parser.error("evaluate needs --checkpoint or --baseline")
    if (
        args.command == "evaluate"
        and args.parallel_batches
        and not (args.checkpoint and args.baseline)
    ):
        argument_parser.error("--parallel-batches requires --checkpoint and --baseline")
    grasp_index = getattr(args, "grasp_index", 0)
    if type(grasp_index) is not int or grasp_index < 0:
        argument_parser.error("--grasp-index must be a nonnegative integer")
    if getattr(args, "seed", None) is not None and not 0 <= args.seed < 2**32:
        argument_parser.error("--seed must lie in [0, 2**32)")
    if getattr(args, "output", None) is not None:
        try:
            _validate_output_path(args.output, directory=args.command == "train")
        except ValueError as error:
            argument_parser.error(str(error))

    gpu_assignment = None
    if args.command not in ("validate", "import-trajectory"):
        # Run before regions/artifacts can import Torch. A CUDA
        # device argument alone does not isolate SDK startup enumeration.
        try:
            gpu_assignment = isolate_gpu(args.device)
        except (ValueError, RuntimeError) as error:
            argument_parser.error(str(error))
        args.requested_device = args.device
        args.device = gpu_assignment.torch_device
        args.gpu_assignment = gpu_assignment.to_dict()

    if args.command == "import-trajectory":
        import numpy as np

        from .trajectory import read_transforms

        if args.output.suffix != ".npz":
            raise ValueError("Output must have the .npz extension")
        raw = read_transforms(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, transforms=raw, source_fps=np.array(args.fps))
        print(f"Packed {len(raw)} frames at {args.fps:g} Hz into {args.output}")
        return
    if args.command == "convert-mesh":
        from .assets import validate_conversion

        validate_conversion(args.source, args.output, args.mass, args.scale)
    elif args.command == "discover-area":
        task = TaskSpec.load(args.task)
        args.output = _area_output(args, task)
        try:
            _validate_output_path(args.output)
        except ValueError as error:
            argument_parser.error(str(error))

        from .regions import DiscoveryConfig

        if not args.hand_asset.is_file():
            raise FileNotFoundError(args.hand_asset)
        discovery = (
            DiscoveryConfig(**json.loads(args.settings.read_text()))
            if args.settings
            else DiscoveryConfig()
        )
        discovery.validate()
    else:
        from .config import ExperimentSpec
        from .regions import GraspArea
        from .trajectory import Trajectory

        experiment = ExperimentSpec.load(args.config)
        if args.command == "validate":
            for name in args.task or RELEASE_TASKS:
                task = TaskSpec.load(name)
                if task.area_path is None:
                    raise ValueError(
                        "Task must define area_path; provide grasp-area boxes or run discover-area"
                    )
                area = GraspArea.load(
                    task.area_path,
                    object_asset=task.object_asset,
                    body_offset=task.body_offset,
                    hand_asset=HAND_ASSET,
                )
                trajectory = Trajectory.load(task, experiment.control_dt)
                print(
                    json.dumps(
                        {
                            "task": task.name,
                            "source_frames": trajectory.source_frames,
                            "control_frames": len(trajectory.poses),
                            "duration_s": trajectory.duration,
                            "task_fingerprint": task.fingerprint(),
                            "area_boxes": len(area.patches),
                            "area_source": area.source["kind"],
                            "area_fingerprint": area.fingerprint(),
                        }
                    )
                )
            return
        task = TaskSpec.load(args.task)
        if (output := _stage_output(args, task)) is not None:
            args.output = output
            try:
                _validate_output_path(args.output, directory=args.command == "train")
            except ValueError as error:
                argument_parser.error(str(error))
        if task.area_path is None:
            raise ValueError(
                "Task must define area_path; provide grasp-area boxes or run discover-area"
            )
        area = GraspArea.load(
            task.area_path,
            object_asset=task.object_asset,
            body_offset=task.body_offset,
            hand_asset=args.hand_asset,
        )
        if args.num_envs is None:
            args.num_envs = (
                experiment.evaluation.num_envs
                if args.command == "evaluate"
                else experiment.training.num_envs
                if args.command == "train"
                else experiment.search.num_envs
            )
        if args.command == "optimize":
            minimum = len(area.patches) * experiment.search.minimum_candidates_per_patch
            if args.num_envs < minimum:
                argument_parser.error(
                    f"This area needs at least {minimum} physical search environments "
                    f"({experiment.search.minimum_candidates_per_patch} per box)"
                )
        args.seed = experiment.evaluation.seed if args.seed is None else args.seed
        if args.command == "optimize":
            args.iterations = args.iterations or experiment.search.iterations
            args.keep = args.keep or experiment.search.keep
            experiment = replace(
                experiment,
                search=replace(
                    experiment.search,
                    iterations=args.iterations,
                    keep=args.keep,
                    max_wall_time_s=args.max_wall_time or experiment.search.max_wall_time_s,
                ),
            )
        elif args.command == "evaluate":
            args.batches = args.batches or experiment.evaluation.batches
        experiment.validate()
        if getattr(args, "warm_proposals", None) is not None:
            from .artifacts import load_warm_proposals

            try:
                load_warm_proposals(
                    args.warm_proposals,
                    task,
                    hand_asset=args.hand_asset,
                    control_dt=experiment.control_dt,
                    discovery_sha256=area.fingerprint(),
                )
            except (ValueError, FileNotFoundError) as error:
                argument_parser.error(str(error))
        grasp_path = getattr(args, "grasp", None) or getattr(args, "warm_start", None)
        if grasp_path is not None:
            from .artifacts import load_grasp

            try:
                grasp = load_grasp(
                    grasp_path,
                    task,
                    hand_asset=args.hand_asset,
                    control_dt=experiment.control_dt,
                    grasp_fingerprint=(
                        experiment.grasp_fingerprint() if args.command == "optimize" else None
                    ),
                    physics_fingerprint=experiment.physics_fingerprint(),
                    discovery_sha256=area.fingerprint(),
                    require_verified=args.command in ("train", "evaluate"),
                    grasp_index=getattr(args, "grasp_index", 0),
                    collision_spec=experiment.collision,
                )
            except (ValueError, FileNotFoundError) as error:
                argument_parser.error(str(error))
            if args.command == "verify-grasps":
                args.num_envs = len(grasp["scores"])
        checkpoint = getattr(args, "checkpoint", None)
        if checkpoint is not None:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
            metadata = (
                checkpoint.parent / "contract.json"
                if args.command == "train"
                else checkpoint.with_suffix(".json")
            )
            if not metadata.is_file():
                raise FileNotFoundError(f"Checkpoint metadata does not exist: {metadata}")
            from .artifacts import SCHEMA_VERSION

            contents = json.loads(metadata.read_text())
            if contents.get("contract", contents).get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported policy schema; regenerate with the current grasp and policy contract"
                )

    from isaaclab.app import AppLauncher

    launcher = launch_isolated_app(
        gpu_assignment,
        launcher_class=AppLauncher,
        headless=not args.gui,
        livestream=0,
        multi_gpu=False,
        fast_shutdown=True,
        # Each process uses one simulation GPU. The pinned GPU foundation
        # plugin's peer benchmarks otherwise allocate across visible devices
        # during startup, competing with other training/validation processes.
        kit_args=(
            "--/validate/p2p/enabled=false "
            "--/validate/p2p/memoryCheck/enabled=false "
            "--/validate/iommu/enabled=false"
        ),
    )
    env = None
    exit_code = 0
    try:
        print(json.dumps({"gpu_assignment": args.gpu_assignment}), flush=True)
        output = getattr(args, "output", None)
        if output is not None:
            provenance = (
                output / "gpu_assignment.json"
                if args.command == "train"
                else output.with_suffix(".gpu.json")
            )
            provenance.parent.mkdir(parents=True, exist_ok=True)
            provenance.write_text(json.dumps(args.gpu_assignment, indent=2) + "\n")
        if args.command == "convert-mesh":
            from .assets import convert_mesh

            convert_mesh(args.source, args.output, args.mass, args.scale)
            return
        if args.command == "discover-area":
            from .regions import discover_area

            discovered = discover_area(
                task, args.output, hand_asset=args.hand_asset, config=discovery
            )
            print(
                json.dumps(
                    {
                        "area": str(args.output),
                        "boxes": len(discovered.patches),
                        "source": discovered.source,
                    }
                )
            )
            return
        from .env import G2AEnv, G2AEnvCfg

        cfg = G2AEnvCfg()
        cfg.task = args.task
        cfg.seed = args.seed
        cfg.experiment = experiment
        cfg.hand_asset = str(args.hand_asset.resolve())
        cfg.sim.device = args.device
        cfg.scene.num_envs = args.num_envs
        cfg.paired_evaluation = (
            args.command == "evaluate" and args.checkpoint is not None and args.baseline
        )
        if cfg.paired_evaluation:
            cfg.scene.num_envs *= 2
            if args.parallel_batches:
                cfg.parallel_evaluation_seeds = tuple(
                    args.seed + index for index in range(args.batches)
                )
                cfg.scene.num_envs *= args.batches
        cfg.mode = "search" if args.command in ("optimize", "verify-grasps") else "rl"
        if cfg.mode == "rl":
            cfg.grasp_path = str(args.grasp.resolve())
            cfg.grasp_index = args.grasp_index
        cfg.random_start = args.command == "train" and experiment.training.random_start is not False
        cfg.randomize_forces = args.command == "train" or getattr(args, "disturbances", False)
        env = G2AEnv(cfg)
        if args.command == "optimize":
            from .optimize import run_optimization

            run_optimization(env, args)
        elif args.command == "verify-grasps":
            from .verify import reverify

            reverify(env, args, grasp)
        elif args.command == "train":
            from .rl import train

            train(env, args)
        elif args.command == "evaluate":
            from .rl import evaluate

            evaluate(env, args)
    except BaseException:
        import traceback

        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        try:
            if env is not None:
                env.close()
        except BaseException:
            import traceback

            exit_code = 1
            traceback.print_exc()
            raise
        finally:
            import omni.kit.app

            # Fast shutdown avoids a native teardown crash. Kit otherwise exits 0 on errors.
            omni.kit.app.get_app().post_quit(exit_code)
            launcher.app.close()
