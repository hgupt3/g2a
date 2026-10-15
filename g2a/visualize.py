"""Native grasp views and trajectory playback, using saved grasps and exported policies.

Simulator imports happen only in a fresh, GPU-isolated worker. Importing this
module or displaying command help does not import Torch or initialize CUDA.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PEER_SETTINGS = (
    "--/validate/p2p/enabled=false "
    "--/validate/p2p/memoryCheck/enabled=false "
    "--/validate/iommu/enabled=false"
)
MAX_FORCE_ARROW_M = 0.25


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def add_arguments(parser):
    parser.add_argument(
        "--task", help="Task name or custom task JSON; inferred from model metadata"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Model directory, such as pretrained/hammer, or a training output directory",
    )
    parser.add_argument(
        "--config", type=Path, help="Experiment JSON; defaults to the model's experiment.json"
    )
    parser.add_argument(
        "--grasp", type=Path, help="Saved grasps.pt; defaults to the model's grasps.pt"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Exported policy.pt; defaults to the model's policy.pt"
    )
    parser.add_argument(
        "--grasp-index", type=int, help="Grasp row; defaults to policy metadata or zero"
    )
    parser.add_argument(
        "--hand-asset", type=Path, help="Custom LEAP hand USD; defaults to the packaged hand"
    )
    parser.add_argument("--mode", choices=("grasp", "trajectory"), default="trajectory")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device; physical index when CUDA_VISIBLE_DEVICES is unset",
    )
    parser.add_argument(
        "--output", type=Path, help="New or empty output directory; omit to open the native GUI"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Show the native GUI while saving images/video"
    )
    parser.add_argument(
        "--hold-seconds", type=positive, default=10, help="GUI viewing time for a settled grasp"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Sample the configured mass, friction and task-load ranges",
    )
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=900)
    parser.add_argument(
        "--hide-forces", action="store_true", help="Hide applied task-force arrows and labels"
    )
    parser.add_argument(
        "--force-scale",
        type=positive,
        default=0.02,
        help="Force arrow length in metres per newton (default: 0.02; arrows capped at 0.25 m)",
    )
    parser.add_argument(
        "--fps", type=positive, default=15, help="Video rate; must divide the control frequency"
    )
    parser.add_argument(
        "--max-wall-time",
        type=positive,
        default=900,
        help="Finite seconds after environment construction",
    )
    parser.add_argument(
        "--startup-timeout",
        type=positive,
        default=180,
        help="Additional seconds for simulator startup and cleanup",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        help="Optional short trajectory probe; reported as incomplete",
    )
    parser.add_argument("--ffmpeg", type=Path, help="Optional FFmpeg executable")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def parser():
    return add_arguments(argparse.ArgumentParser(description=__doc__))


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, document):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def find_ffmpeg(requested=None):
    candidate = str(requested) if requested else shutil.which("ffmpeg")
    if not candidate:
        try:
            import imageio_ffmpeg

            candidate = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as error:
            raise ValueError(
                "Install grasp-to-act[visual] or provide --ffmpeg to save video"
            ) from error
    path = Path(candidate).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"FFmpeg is not executable: {path}")
    return path


def resolve_inputs(args):
    """Resolve the task, configuration, grasp and policy for playback."""
    from .config import ExperimentSpec
    from .tasks import RESOURCE_ROOT, TaskSpec

    if args.width % 2 or args.height % 2:
        raise ValueError("Image width and height must be even")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must lie in [0, 2**32)")
    if args.grasp_index is not None and args.grasp_index < 0:
        raise ValueError("--grasp-index must be nonnegative")
    model = args.model.expanduser().resolve() if args.model else None
    if model is not None:
        if not model.is_dir():
            raise ValueError(f"Model directory does not exist: {model}")
        if not (model / "policy.pt").is_file() and (model / "ppo/policy.pt").is_file():
            model = model / "ppo"
    checkpoint = args.checkpoint
    if checkpoint is None and model is not None and (model / "policy.pt").is_file():
        checkpoint = model / "policy.pt"
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        if checkpoint.suffix != ".pt":
            raise ValueError(
                "Visualization uses exported policy.pt; checkpoint.pth is for resuming training"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metadata_path = checkpoint.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("sha256") != file_hash(checkpoint):
            raise ValueError("Policy file does not match its metadata checksum")
        policy_task = metadata.get("contract", {}).get("task")
        if not isinstance(policy_task, str) or not policy_task:
            raise ValueError("Policy metadata is missing its task contract")
        recorded_index = metadata.get("grasp_index", 0)
        if type(recorded_index) is not int or recorded_index < 0:
            raise ValueError("Policy grasp_index must be a nonnegative integer")
        if args.grasp_index is None:
            args.grasp_index = recorded_index
        elif args.mode == "trajectory" and args.grasp_index != recorded_index:
            raise ValueError("Requested grasp index differs from the policy's saved grasp index")
    else:
        metadata, policy_task = None, None
    if args.task is None:
        args.task = policy_task
    if args.task is None:
        raise ValueError("Provide --task when no policy metadata is available")
    task = TaskSpec.load(args.task)
    if policy_task is not None and task.name != policy_task:
        raise ValueError("Requested task differs from the policy's saved task")
    if Path(args.task).suffix == ".json":
        args.task = str(Path(args.task).expanduser().resolve())
    bases = list(dict.fromkeys(p for p in (model, checkpoint.parent if checkpoint else None) if p))

    def first_file(paths):
        return next((path for path in paths if path.is_file()), None)

    config = args.config or first_file(base / "experiment.json" for base in bases)
    grasp = args.grasp or first_file(
        path
        for base in bases
        for path in (
            base / "grasps.pt",
            base.parent / "grasps.pt",
            base.parent / "source_inputs/grasps.pt",
        )
    )
    if grasp is None:
        raise ValueError("Provide --grasp grasps.pt, or a --model directory containing it")
    args.grasp = Path(grasp).expanduser().resolve()
    if not args.grasp.is_file():
        raise FileNotFoundError(args.grasp)
    args.config = Path(config).expanduser().resolve() if config else None
    experiment = ExperimentSpec.load(args.config)
    args.grasp_index = args.grasp_index if args.grasp_index is not None else 0
    args.checkpoint = checkpoint
    args.hand_asset = (
        (args.hand_asset or RESOURCE_ROOT / "assets/leap_hand/leap.usd").expanduser().resolve()
    )
    if not args.hand_asset.is_file():
        raise FileNotFoundError(args.hand_asset)
    stride = 1 / (args.fps * experiment.control_dt)
    if stride < 1 or not math.isclose(stride, round(stride), abs_tol=1e-6):
        raise ValueError("--fps must divide the experiment's control frequency")
    args.capture_stride = round(stride)
    args.gui = args.gui or args.output is None
    # Fixed camera intrinsics keep auto framing predictable; no simulator defaults.
    args.focal_length, args.horizontal_aperture = 24.0, 20.955
    args.warmup_renders = 8
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        if args.output.exists() and not args.output.is_dir():
            raise ValueError("--output must be a directory")
        if not args._worker and args.output.is_dir() and any(args.output.iterdir()):
            raise ValueError("Output directory must be empty; use a new destination")
        try:
            from PIL import Image  # noqa: F401
        except ImportError as error:
            raise ValueError("Install grasp-to-act[visual] to save images or video") from error
        if args.mode == "trajectory":
            args.ffmpeg = find_ffmpeg(args.ffmpeg)
    paths = [args.grasp, args.hand_asset, *([args.config] if args.config else [])]
    if checkpoint is not None:
        paths.extend((checkpoint, checkpoint.with_suffix(".json")))
    return experiment, metadata, {str(path): file_hash(path) for path in paths}


def look_at_rotation(offset):
    """OpenGL camera basis: -Z looks towards target, +Y is the local up axis."""
    forward = -np.asarray(offset, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), -forward))


def palm_camera_pose(palm, offset, target):
    rotation = Rotation.from_quat(np.asarray(palm)[[4, 5, 6, 3]]).as_matrix()
    focal = np.asarray(palm)[:3] + rotation @ np.asarray(target)
    eye = focal + rotation @ np.asarray(offset)
    xyzw = Rotation.from_matrix(rotation @ look_at_rotation(offset)).as_quat()
    return eye, xyzw[[3, 0, 1, 2]], focal


def trajectory_camera_direction(palm, obj, poses):
    """Choose one overview direction from the palm-facing side of the motion."""
    palm_rotation = Rotation.from_quat(np.asarray(palm)[[4, 5, 6, 3]])
    object_rotation = Rotation.from_quat(np.asarray(obj)[[4, 5, 6, 3]])
    local_direction = object_rotation.inv().apply(palm_rotation.apply((1, -0.15, 0.12)))
    directions = Rotation.from_quat(np.asarray(poses)[:, [4, 5, 6, 3]]).apply(local_direction)
    direction = directions.mean(axis=0)
    if np.linalg.norm(direction) < 1e-8:
        direction = directions[len(directions) // 2]
    return direction


def fixed_camera_pose(points, args, direction=(1, -0.35, 0.18)):
    # Reserve extra room for tracking deviations and changing finger poses.
    center, offset = fit_view(points, direction, args, fill=0.72)
    xyzw = Rotation.from_matrix(look_at_rotation(offset)).as_quat()
    return center + offset, xyzw[[3, 0, 1, 2]], center


class CpuRgbCamera:
    """Public CPU annotator; only the camera pose is authored in USD."""

    def __init__(self, name, args):
        import isaaclab.sim as sim_utils
        import omni.replicator.core as rep
        from pxr import UsdGeom

        path = f"/World/ReviewCamera_{name}"
        cfg = sim_utils.PinholeCameraCfg(
            focal_length=args.focal_length,
            horizontal_aperture=args.horizontal_aperture,
            clipping_range=(0.01, 100.0),
        )
        cfg.func(path, cfg)
        prim = sim_utils.get_current_stage().GetPrimAtPath(path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        self.transform = xform.AddTransformOp()
        self.product = rep.create.render_product(path, (args.width, args.height))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annotator.attach(self.product.path)

    def set_pose(self, eye, quaternion):
        from pxr import Gf

        quaternion = np.asarray(quaternion, dtype=float)
        matrix = Gf.Matrix4d().SetRotate(
            Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:])))
        )
        matrix.SetTranslateOnly(Gf.Vec3d(*map(float, eye)))
        self.transform.Set(matrix)

    def rgb(self):
        return np.asarray(self.annotator.get_data())[..., :3].copy()


def make_review_env(base, camera, args):
    """Create render products before the base's first sim.reset, without sensors."""

    class ReviewEnv(base):
        def _setup_scene(self):
            super()._setup_scene()
            self.review_cameras = {}
            self.review_terminal_hook = None
            self.review_capture_error = None
            self.review_force_point_world = None
            for name in ("follow", "wide") if args.output is not None else ():
                self.review_cameras[name] = camera(name, args)

        def _apply_action(self):
            if args.mode == "trajectory" and not args.hide_forces:
                from .math import quat_apply

                pose = self.object.data.root_link_state_w[:, :7]
                self.review_force_point_world = pose[:, :3] + quat_apply(
                    pose[:, 3:], self.force_point
                )
            super()._apply_action()

        def _reset_idx(self, env_ids):
            # DirectRLEnv calls this after computing the final reward/done, but
            # before writing the next episode's poses. Keep that final frame.
            if self.review_terminal_hook is not None and hasattr(self, "started"):
                terminal = bool((self.reset_terminated | self.reset_time_outs).any())
                if terminal:
                    try:
                        self.review_terminal_hook(self)
                    except Exception as error:
                        # Complete the original metrics/reset path even if a
                        # camera fails. Report capture failure after collecting.
                        self.review_capture_error = error
            super()._reset_idx(env_ids)

    return ReviewEnv


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("Finite rendering stage deadline reached")


def pose_error(actual, expected):
    """Translation/angular difference for antipodal-equivalent scalar-first poses."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != (7,)
        or expected.shape != (7,)
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
        or np.linalg.norm(actual[3:]) == 0
        or np.linalg.norm(expected[3:]) == 0
    ):
        raise RuntimeError("Invalid native/renderer pose in geometry parity check")
    displacement = float(np.linalg.norm(actual[:3] - expected[:3]))
    a = actual[3:] / np.linalg.norm(actual[3:])
    b = expected[3:] / np.linalg.norm(expected[3:])
    angle = float(2 * np.arccos(np.clip(abs(a @ b), 0, 1)))
    return displacement, angle


def native_render_parity(env):
    """Reject stale Fabric renderer transforms; never overwrite body transforms."""
    import isaaclab.sim as sim_utils
    import usdrt

    stage = usdrt.Usd.Stage.Attach(sim_utils.get_current_stage_id())
    hand = env.hand.root_physx_view.get_link_transforms()[0].detach().cpu().numpy()
    tool = env.object.root_physx_view.get_transforms()[0].detach().cpu().numpy()
    paths = list(env.hand.root_physx_view.link_paths[0]) + [
        env.object.root_physx_view.prim_paths[0]
    ]
    errors = []
    for path, pose in zip(paths, list(hand) + [tool], strict=True):
        xform = usdrt.Rt.Xformable(stage.GetPrimAtPath(path))
        position_attr, rotation_attr = xform.GetWorldPositionAttr(), xform.GetWorldOrientationAttr()
        if position_attr.IsValid() and rotation_attr.IsValid():
            position, quaternion = position_attr.Get(), rotation_attr.Get()
        else:
            matrix_attr = xform.GetFabricHierarchyWorldMatrixAttr()
            if not matrix_attr.IsValid():
                raise RuntimeError(f"No native renderer world transform for {path}")
            matrix = matrix_attr.Get()
            position, quaternion = matrix.ExtractTranslation(), matrix.ExtractRotationQuat()
        rendered = [*position, quaternion.GetReal(), *quaternion.GetImaginary()]
        native = np.concatenate((pose[:3], pose[[6, 3, 4, 5]]))  # Tensor API xyzw -> wxyz.
        position, angle = pose_error(rendered, native)
        errors.append({"path": str(path), "position_error_m": position, "angle_error_rad": angle})
    worst_position = max(item["position_error_m"] for item in errors)
    worst_angle = max(item["angle_error_rad"] for item in errors)
    if worst_position > 1e-4 or worst_angle > 1e-3:
        raise RuntimeError(
            f"Stale Fabric geometry: position={worst_position:g}m, angle={worst_angle:g}rad; {errors}"
        )
    return {
        "body_count": len(errors),
        "maximum_position_error_m": worst_position,
        "maximum_angle_error_rad": worst_angle,
    }


def snapshot(env):
    goal = env.trajectory.poses[env.goal_frame[0]].detach().cpu().tolist()
    return {
        "control_step": int(env.episode_length_buf[0]),
        "physics_step": int(env._sim_step_counter),
        "simulation_time_s": float(env.episode_length_buf[0]) * env.step_dt,
        "task_time_s": int(env.goal_frame[0]) * env.step_dt,
        "object_pose_world": env.object.data.root_link_state_w[0, :7].detach().cpu().tolist(),
        "palm_pose_world": env.hand.data.root_link_state_w[0, :7].detach().cpu().tolist(),
        "goal_pose_environment": goal,
        "joint_positions_rad": env.hand.data.joint_pos[0, env.joint_ids].detach().cpu().tolist(),
        "joint_targets_rad": env.joint_targets[0].detach().cpu().tolist(),
        "position_error_m": float(env._errors()[0][0]),
        "angle_error_rad": float(env._errors()[1][0]),
    }


def force_arrow_geometry(point, force, scale):
    """Map a world force to a bounded display arrow, preserving its direction."""
    point, force = np.asarray(point, dtype=float), np.asarray(force, dtype=float)
    if (
        point.shape != (3,)
        or force.shape != (3,)
        or not np.isfinite(point).all()
        or not np.isfinite(force).all()
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Force display needs finite world vectors and a positive scale")
    magnitude = float(np.linalg.norm(force))
    length = min(magnitude * scale, MAX_FORCE_ARROW_M)
    endpoint = point + force * (length / magnitude) if magnitude > 0 else point.copy()
    return endpoint, magnitude, length


def applied_force_sample(env, scale):
    """Read the last physics-substep load, including resistance and randomization."""
    from .math import quat_apply

    point = env.review_force_point_world
    if point is None:
        pose = env.object.data.root_link_state_w[:, :7]
        point = pose[:, :3] + quat_apply(pose[:, 3:], env.force_point)
    point = point[0].detach().cpu().numpy()
    force = env.applied_wrench[0, :3].detach().cpu().numpy()
    endpoint, magnitude, length = force_arrow_geometry(point, force, scale)
    return {
        "physics_step": int(env._sim_step_counter),
        "application_point_world_m": point.tolist(),
        "force_world_n": force.tolist(),
        "magnitude_n": magnitude,
        "arrow_endpoint_world_m": endpoint.tolist(),
        "arrow_length_m": length,
    }


class ForceArrow:
    """Non-colliding USD geometry, shared by native GUI and saved camera views."""

    def __init__(self):
        import isaaclab.sim as sim_utils
        from pxr import Gf, UsdGeom

        stage = sim_utils.get_current_stage()
        path = "/World/AppliedTaskForce"
        self.root = UsdGeom.Xform.Define(stage, path)
        self.position = self.root.AddTranslateOp()
        self.orientation = self.root.AddOrientOp()
        self.shaft = UsdGeom.Cylinder.Define(stage, path + "/shaft")
        self.shaft.CreateRadiusAttr(0.002)
        self.head = UsdGeom.Cone.Define(stage, path + "/head")
        self.head.CreateRadiusAttr(0.007)
        self.origin = UsdGeom.Sphere.Define(stage, path + "/application_point")
        self.origin.CreateRadiusAttr(0.004)
        self.shaft_position = self.shaft.AddTranslateOp()
        self.head_position = self.head.AddTranslateOp()
        for shape in (self.shaft, self.head, self.origin):
            shape.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.25, 0.025)])
        self.root.MakeInvisible()

    def update(self, sample):
        from pxr import Gf

        length = sample["arrow_length_m"]
        if length < 1e-6:
            self.root.MakeInvisible()
            return
        self.root.MakeVisible()
        force = np.asarray(sample["force_world_n"])
        xyzw = Rotation.from_matrix(look_at_rotation(force)).as_quat()
        self.position.Set(Gf.Vec3d(*sample["application_point_world_m"]))
        self.orientation.Set(Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*map(float, xyzw[:3]))))
        head = min(0.02, length * 0.3)
        self.shaft.GetHeightAttr().Set(length - head)
        self.head.GetHeightAttr().Set(head)
        self.shaft_position.Set(Gf.Vec3d(0, 0, (length - head) / 2))
        self.head_position.Set(Gf.Vec3d(0, 0, length - head / 2))


def label_force(image, sample, scale):
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default(size=max(12, round(image.width / 64)))
    force = sample["force_world_n"]
    text = (
        f"Applied task load: {sample['magnitude_n']:.2f} N\n"
        f"World XYZ: [{force[0]:+.2f}, {force[1]:+.2f}, {force[2]:+.2f}] N\n"
        f"Arrow: {scale * 100:g} cm/N (max {MAX_FORCE_ARROW_M * 100:g} cm)"
    )
    draw = ImageDraw.Draw(image)
    bounds = draw.multiline_textbbox((16, 14), text, font=font, spacing=4)
    draw.rounded_rectangle((8, 8, bounds[2] + 8, bounds[3] + 8), radius=6, fill=(25, 28, 31))
    draw.multiline_text((16, 14), text, font=font, spacing=4, fill=(255, 177, 91))


def fit_view(points, direction, args, *, fill=0.82):
    """Fit all requested geometry inside a perspective camera, with 9% margins."""
    points = np.asarray(points, dtype=float)
    direction = np.asarray(direction, dtype=float).copy()
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Camera framing needs finite 3-D geometry")
    if (
        direction.shape != (3,)
        or not np.isfinite(direction).all()
        or np.linalg.norm(direction) == 0
    ):
        raise ValueError("Camera direction must be finite and nonzero")
    direction /= np.linalg.norm(direction)
    rotation = look_at_rotation(direction)
    center = (points.min(0) + points.max(0)) / 2
    relative = (points - center) @ rotation
    tan_h = args.horizontal_aperture / (2 * args.focal_length)
    tan_v = tan_h * args.height / args.width
    distance = max(
        np.max(relative[:, 2] + np.abs(relative[:, 0]) / (tan_h * fill)),
        np.max(relative[:, 2] + np.abs(relative[:, 1]) / (tan_v * fill)),
        np.max(relative[:, 2]) + 0.05,
    )
    projected = relative[:, :2] / ((distance - relative[:, 2:3]) * [tan_h, tan_v])
    if not np.isfinite(projected).all() or np.abs(projected).max() > fill + 1e-9:
        raise RuntimeError("Camera framing does not contain the requested geometry")
    return center, direction * distance


def geometry_in_palm(env, state):
    """Use live native poses and the actual authored hand boxes/tool hulls."""
    links = env.hand.root_physx_view.get_link_transforms()[0].detach().cpu().numpy()
    obj = env.object.root_physx_view.get_transforms()[0].detach().cpu().numpy()
    palm = np.asarray(state["palm_pose_world"])
    rotation = Rotation.from_quat(palm[[4, 5, 6, 3]]).as_matrix()
    signs = np.asarray(list(product((-1, 1), repeat=3)))
    names = list(env.collision_model.body_names)
    hand_points = []
    for box in env.collision_model.hand.boxes:
        pose = links[names.index(box.body)]
        corners = (signs * box.half_extents) @ box.axes.T + box.center
        hand_points.append(corners @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3])
    hand = np.concatenate(hand_points)
    tool_local = np.concatenate([hull.vertices for hull in env.collision_model.tool.hulls])
    tool = tool_local @ Rotation.from_quat(obj[3:]).as_matrix().T + obj[:3]
    return (hand - palm[:3]) @ rotation, (tool - palm[:3]) @ rotation


def trajectory_points(env, force_margin=0):
    """Fit the whole reference path and the complete initial hand/tool assembly."""
    state = snapshot(env)
    hand, tool = geometry_in_palm(env, state)
    palm = np.asarray(state["palm_pose_world"])
    palm_rotation = Rotation.from_quat(palm[[4, 5, 6, 3]])
    obj = np.asarray(state["object_pose_world"])
    object_rotation = Rotation.from_quat(obj[[4, 5, 6, 3]])
    world = palm_rotation.apply(np.concatenate((hand, tool))) + palm[:3]
    local = object_rotation.inv().apply(world - obj[:3])
    # Local bounding corners cover every hand/tool point at every path rotation.
    bounds = np.asarray(list(product(*zip(local.min(0), local.max(0), strict=True))))
    poses = env.trajectory.poses.detach().cpu().numpy()
    rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    points = np.concatenate([poses[:, :3] + rotations.apply(corner) for corner in bounds])
    if force_margin:
        centers = poses[:, :3] + rotations.apply(np.asarray(env.task.force_point))
        corners = np.asarray(
            list(
                product(
                    *zip(centers.min(0) - force_margin, centers.max(0) + force_margin, strict=True)
                )
            )
        )
        points = np.concatenate((points, corners))
    return points


class Recorder:
    def __init__(self, env, args, deadline, directory, wide_pose):
        self.env, self.args, self.deadline = env, args, deadline
        self.directory, self.wide_pose = directory, wide_pose
        self.frames, self.terminal_captured = 0, False
        self.force_arrow = (
            ForceArrow() if args.mode == "trajectory" and not args.hide_forces else None
        )
        self.force_log = (
            (directory / "forces.jsonl").open("w") if self.force_arrow and args.output else None
        )
        self.max_position_error, self.max_angle_error, self.body_checks = 0.0, 0.0, 0
        self.temporary = tempfile.TemporaryDirectory(prefix="g2a-frames-") if args.output else None
        if self.temporary:
            for view in ("follow", "wide"):
                (Path(self.temporary.name) / view).mkdir()

    @staticmethod
    def set_pose(camera, pose):
        camera.set_pose(pose[0], pose[1])

    def render(self, repeats=1):
        for _ in range(repeats):
            check_deadline(self.deadline)
            self.env.sim.render()  # Rendering is paused; never integrate physics here.
        parity = native_render_parity(self.env)
        self.max_position_error = max(self.max_position_error, parity["maximum_position_error_m"])
        self.max_angle_error = max(self.max_angle_error, parity["maximum_angle_error_rad"])
        self.body_checks += parity["body_count"]
        if not self.args.output:
            return {}
        from PIL import Image

        result = {}
        for name, camera in self.env.review_cameras.items():
            frame = camera.rgb()
            if frame.shape != (self.args.height, self.args.width, 3) or frame.dtype != np.uint8:
                raise RuntimeError(
                    f"Invalid {name} camera RGB shape/type: {frame.shape}/{frame.dtype}"
                )
            if float(frame.std()) < 1:
                raise RuntimeError(f"The {name} camera returned a blank image")
            result[name] = Image.fromarray(frame)
        return result

    def capture(self, terminal=False, warmup=False):
        force = applied_force_sample(self.env, self.args.force_scale) if self.force_arrow else None
        if force is not None:
            self.force_arrow.update(force)
        if self.args.output:
            state = snapshot(self.env)
            hand, tool = geometry_in_palm(self.env, state)
            points = np.concatenate((hand, tool))
            if force is not None and force["magnitude_n"] > 0:
                palm = np.asarray(state["palm_pose_world"])
                rotation = Rotation.from_quat(palm[[4, 5, 6, 3]])
                ends = np.asarray(
                    [force["application_point_world_m"], force["arrow_endpoint_world_m"]]
                )
                points = np.concatenate((points, rotation.inv().apply(ends - palm[:3])))
            target, offset = fit_view(points, (1, 0, 0), self.args)
            self.set_pose(
                self.env.review_cameras["follow"],
                palm_camera_pose(state["palm_pose_world"], offset, target),
            )
            self.set_pose(self.env.review_cameras["wide"], self.wide_pose)
        images = self.render(self.args.warmup_renders if warmup else 1)
        for name, image in images.items():
            if force is not None:
                label_force(image, force, self.args.force_scale)
            image.save(
                Path(self.temporary.name) / name / f"{self.frames:06d}.png", compress_level=1
            )
        if self.force_log is not None:
            self.force_log.write(
                json.dumps({"frame": self.frames, **force}, allow_nan=False) + "\n"
            )
        self.frames += 1
        self.terminal_captured |= terminal

    def grasp_views(self):
        state = snapshot(self.env)
        hand, tool = geometry_in_palm(self.env, state)
        before = int(self.env._sim_step_counter)
        for label, points, direction in (
            ("front", np.concatenate((hand, tool)), (1, 0, 0)),
            ("contacts", hand, (1, 0, 0)),
            ("oblique", np.concatenate((hand, tool)), (1, -0.30, 0.12)),
        ):
            target, offset = fit_view(points, direction, self.args)
            pose = palm_camera_pose(state["palm_pose_world"], offset, target)
            if self.args.gui and label == "front":
                import isaaclab.sim as sim_utils

                sim_utils.set_camera_view(pose[0], pose[2])
            for camera in self.env.review_cameras.values():
                self.set_pose(camera, pose)
            if self.args.output:
                self.render(self.args.warmup_renders)["follow"].save(
                    self.directory / f"{label}.png"
                )
                self.frames += 1
        if int(self.env._sim_step_counter) != before:
            raise RuntimeError("Camera changes unexpectedly advanced physics")

    def encode(self):
        if not self.args.output:
            return
        for name, filename in (("wide", "trajectory.mp4"), ("follow", "closeup.mp4")):
            check_deadline(self.deadline)
            command = [
                str(self.args.ffmpeg),
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(self.args.fps),
                "-i",
                str(Path(self.temporary.name) / name / "%06d.png"),
                "-c:v",
                "libx264",
                "-threads",
                "4",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.directory / filename),
            ]
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=max(0.01, self.deadline - time.perf_counter()),
            )

    def close(self):
        if self.force_log is not None:
            self.force_log.close()
        if self.temporary:
            self.temporary.cleanup()


def record_episode(env, policy, recorder, args, deadline):
    """Play exactly one first episode, capturing its terminal pose before reset."""
    import torch

    env.review_capture_error = None
    env.review_terminal_hook = lambda _: recorder.capture(terminal=True)
    result = {"status": "step_limit", "completed": False, "metrics": None}
    try:
        recorder.capture(warmup=True)
        observations = env._get_observations()
        limit = env.max_episode_length + 2 if args.max_steps is None else args.max_steps
        with torch.inference_mode():
            for step in range(1, limit + 1):
                check_deadline(deadline)
                if args.gui and not env.visualization_app.is_running():
                    result["status"] = "window_closed"
                    break
                started = time.perf_counter()
                observations, reward, terminated, truncated, _ = env.step(
                    policy(observations["policy"])
                )
                if bool((terminated | truncated)[0]):
                    result.update(
                        status="failed" if bool(terminated[0]) else "complete",
                        completed=True,
                        metrics={name: float(value[0]) for name, value in env.last_metrics.items()},
                    )
                    if env.review_capture_error is not None:
                        raise env.review_capture_error
                    if not recorder.terminal_captured:
                        raise RuntimeError("The original terminal frame was not captured")
                    break
                if (
                    not torch.isfinite(observations["policy"]).all()
                    or not torch.isfinite(reward).all()
                ):
                    raise RuntimeError("Nonfinite observation or reward during playback")
                if step % args.capture_stride == 0:
                    recorder.capture()
                if args.gui:
                    time.sleep(max(0.0, env.step_dt - (time.perf_counter() - started)))
            else:
                if args.max_steps is None:
                    raise RuntimeError("The first episode exceeded its reference step limit")
        recorder.encode()
        return result
    finally:
        env.review_terminal_hook = None


def settle_grasp(env, policy, recorder, args, deadline):
    import torch

    observations = env._get_observations()
    with torch.inference_mode():
        for _ in range(env.setup_steps):
            check_deadline(deadline)
            observations, reward, terminated, truncated, _ = env.step(
                policy(observations["policy"])
            )
            if bool((terminated | truncated)[0]):
                raise RuntimeError("The saved grasp failed before completing free settling")
            if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Nonfinite state during grasp settling")
    recorder.grasp_views()
    if args.gui:
        until = min(deadline, time.perf_counter() + args.hold_seconds)
        while time.perf_counter() < until:
            if not env.visualization_app.is_running():
                break
            recorder.render()
            time.sleep(0.02)
    return {"status": "settled_grasp", "completed": True, "metrics": None}


def run_worker(args, experiment, metadata, hashes):
    from .device import isolate_gpu, launch_isolated_app

    assignment = isolate_gpu(args.device)  # Idempotent after the early selection in run().
    launcher = launch_isolated_app(
        assignment,
        headless=not args.gui,
        enable_cameras=bool(args.output),
        livestream=0,
        multi_gpu=False,
        fast_shutdown=True,
        kit_args=PEER_SETTINGS,
    )
    env, recorder, exit_code = None, None, 0
    summary = {
        "task": args.task,
        "mode": args.mode,
        "grasp_index": args.grasp_index,
        "seed": args.seed,
        "randomized": args.randomize,
        "controller": "policy"
        if args.checkpoint and args.mode == "trajectory"
        else "zero_residual",
    }
    try:
        import isaaclab.sim as sim_utils

        from .env import G2AEnv, G2AEnvCfg
        from .rl import load_policy

        cfg = G2AEnvCfg()
        cfg.task, cfg.mode, cfg.grasp_path, cfg.grasp_index = (
            args.task,
            "rl",
            str(args.grasp),
            args.grasp_index,
        )
        cfg.hand_asset, cfg.experiment = str(args.hand_asset), experiment
        cfg.scene.num_envs, cfg.sim.device, cfg.sim.use_fabric = 1, assignment.torch_device, True
        cfg.random_start, cfg.randomize_forces, cfg.paired_evaluation = False, args.randomize, False
        cfg.seed, cfg.num_rerenders_on_reset = args.seed, 0
        env = make_review_env(G2AEnv, CpuRgbCamera, args)(cfg)
        env.visualization_app = launcher.app
        env.cfg.sim.render_interval = env.cfg.decimation * (
            args.capture_stride if args.output else 1
        )
        deadline = time.perf_counter() + args.max_wall_time
        policy = (
            load_policy(args.checkpoint, env)
            if args.checkpoint and args.mode == "trajectory"
            else (
                lambda observation: observation.new_zeros(
                    (observation.shape[0], env.cfg.control.action_dim)
                )
            )
        )
        env.seed(args.seed)
        env.reset()
        state = snapshot(env)
        direction = trajectory_camera_direction(
            state["palm_pose_world"],
            state["object_pose_world"],
            env.trajectory.poses.detach().cpu().numpy(),
        )
        force_margin = 0.0
        if args.mode == "trajectory" and not args.hide_forces:
            maximum = np.linalg.norm(env.task.force) + (
                env.task.max_resistance if env.task.resistance else 0
            )
            force_margin = min(
                MAX_FORCE_ARROW_M, args.force_scale * maximum * float(env.force_multiplier[0])
            )
        wide_pose = fixed_camera_pose(trajectory_points(env, force_margin), args, direction)
        if args.mode == "trajectory":
            summary["trajectory_camera"] = {
                "fixed": True,
                "position_world_m": wide_pose[0].tolist(),
                "quaternion_wxyz": wide_pose[1].tolist(),
                "target_world_m": wide_pose[2].tolist(),
            }
            summary["force_display"] = {
                "enabled": not args.hide_forces,
                "source": "actual last physics-substep task force, including resistance and load multiplier",
                "scale_m_per_n": args.force_scale,
                "maximum_arrow_length_m": MAX_FORCE_ARROW_M,
                "samples": "forces.jsonl" if args.output and not args.hide_forces else None,
            }
        if args.gui:
            sim_utils.set_camera_view(wide_pose[0], wide_pose[2])
        recorder = Recorder(env, args, deadline, args.output, wide_pose)
        summary.update(
            (settle_grasp if args.mode == "grasp" else record_episode)(
                env, policy, recorder, args, deadline
            )
        )
        if {path: file_hash(path) for path in hashes} != hashes:
            raise RuntimeError("An input artifact changed during visualization")
        summary.update(
            frames=recorder.frames,
            renderer_body_checks=recorder.body_checks,
            renderer_position_error_m=recorder.max_position_error,
            renderer_angle_error_rad=recorder.max_angle_error,
        )
        print(f"{args.task}: {summary['status']}", flush=True)
    except BaseException as error:
        exit_code = 1
        summary.update(status="error", error=str(error))
        raise
    finally:
        try:
            if args.output:
                write_json(args.output / "summary.json", summary)
        finally:
            try:
                if recorder is not None:
                    recorder.close()
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    import omni.kit.app

                    omni.kit.app.get_app().post_quit(exit_code)
                    launcher.app.close()


def worker_argv(args):
    result = ["--_worker"]
    for key in (
        "task",
        "config",
        "grasp",
        "checkpoint",
        "grasp_index",
        "hand_asset",
        "mode",
        "device",
        "output",
        "seed",
        "width",
        "height",
        "fps",
        "max_wall_time",
        "startup_timeout",
        "max_steps",
        "ffmpeg",
        "hold_seconds",
        "force_scale",
    ):
        value = getattr(args, key)
        if value is not None:
            result.extend(("--" + key.replace("_", "-"), str(value)))
    for key in ("gui", "randomize", "hide_forces"):
        if getattr(args, key):
            result.append("--" + key)
    return result


def stop_owned(process):
    """Stop the worker and its encoder group even if the leader already exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def supervise(args):
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "g2a.visualize", *worker_argv(args)]
    handlers = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Visualization interrupted by signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupt)
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=args.max_wall_time + args.startup_timeout)
            except BaseException:
                stop_owned(process)
                raise
            if code:
                stop_owned(process)
                log.seek(0)
                tail = log.read()[-6000:]
                raise RuntimeError(f"Native visualization failed (exit {code}):\n{tail}")
        if args.output:
            summary_path = args.output / "summary.json"
            if not summary_path.is_file():
                raise RuntimeError("The native renderer did not save its summary")
            summary = json.loads(summary_path.read_text())
            if summary.get("status") == "error":
                raise RuntimeError(summary.get("error", "Native rendering failed"))
            print(f"{summary['task']}: {summary['status']}. Saved to {args.output}")
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)


def run(args):
    # Canonical config imports Torch; isolation MUST precede it in the worker.
    # The CPU supervisor preserves public device ordinals for its fresh worker.
    if args._worker:
        from .device import isolate_gpu

        isolate_gpu(args.device)
    experiment, metadata, hashes = resolve_inputs(args)
    if args._worker:
        return run_worker(args, experiment, metadata, hashes)
    return supervise(args)


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        run(args)
    except (
        ValueError,
        FileNotFoundError,
        TimeoutError,
        subprocess.TimeoutExpired,
        RuntimeError,
    ) as error:
        argument_parser.exit(1, f"Visualization failed: {error}\n")


if __name__ == "__main__":
    main()
