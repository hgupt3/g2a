"""Load, validate, smooth and resample object poses once, before simulation."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .tasks import TaskSpec


def read_transforms(path: str | Path, expected_fps: float | None = None) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"), key=lambda p: int(p.stem))
        if not files:
            raise ValueError(f"No numbered pose files in {path}")
        numbers = [int(p.stem) for p in files]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError(f"Pose frame numbering has gaps in {path}")
        poses = np.stack([np.loadtxt(p) for p in files])
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            poses = data["transforms"]
            if expected_fps is not None and "source_fps" in data:
                recorded_fps = np.asarray(data["source_fps"]).item()
                if not math.isclose(recorded_fps, expected_fps, rel_tol=1e-7):
                    raise ValueError(
                        "Task source_fps does not match the rate stored in the trajectory"
                    )
    elif path.suffix == ".npy":
        poses = np.load(path, allow_pickle=False)
    else:
        raise ValueError("Trajectory must be a numbered .txt directory, .npy, or .npz")
    poses = np.asarray(poses, dtype=np.float64)
    validate_transforms(poses)
    return poses


def validate_transforms(poses: np.ndarray) -> None:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("A trajectory needs at least two 4x4 pose matrices")
    if not np.isfinite(poses).all():
        raise ValueError("Trajectory contains NaN or infinity")
    if not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Pose matrices must have homogeneous bottom rows [0, 0, 0, 1]")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-3):
        raise ValueError("Trajectory rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
        raise ValueError("Trajectory rotations must have determinant +1")


def resample_transforms(
    poses: np.ndarray, source_fps: float, target_fps: float, smoothing_window: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate SO(3) with SLERP. Return exact sample times, including the final pose."""
    validate_transforms(poses)
    if not all(math.isfinite(x) and x > 0 for x in (source_fps, target_fps)):
        raise ValueError("Frame rates must be finite and positive")
    if smoothing_window < 1 or smoothing_window % 2 != 1:
        raise ValueError("Smoothing window must be positive and odd")
    xyz = poses[:, :3, 3].copy()
    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    signs = np.ones(len(quats))
    signs[1:] = np.cumprod(np.where(np.sum(quats[1:] * quats[:-1], axis=1) < 0, -1, 1))
    quats *= signs[:, None]
    if smoothing_window > 1:
        smooth_xyz = uniform_filter1d(xyz, smoothing_window, axis=0, mode="nearest")
        smooth_q = uniform_filter1d(quats, smoothing_window, axis=0, mode="nearest")
        norms = np.linalg.norm(smooth_q, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("Quaternion smoothing window cancels rotations; use a smaller window")
        smooth_q /= norms
        smooth_xyz[[0, -1]] = xyz[[0, -1]]
        smooth_q[[0, -1]] = quats[[0, -1]]
        xyz, quats = smooth_xyz, smooth_q
    old_times = np.arange(len(poses)) / source_fps
    duration = old_times[-1]
    times = np.arange(math.floor(duration * target_fps) + 1) / target_fps
    if duration - times[-1] > 1e-9:
        times = np.append(times, duration)
    else:
        times[-1] = duration
    rotations = Slerp(old_times, Rotation.from_quat(quats))(times)
    out = np.broadcast_to(np.eye(4), (len(times), 4, 4)).copy()
    out[:, :3, :3] = rotations.as_matrix()
    out[:, :3, 3] = np.column_stack([np.interp(times, old_times, xyz[:, i]) for i in range(3)])
    return out, times


@dataclass
class Trajectory:
    poses: torch.Tensor
    velocities: torch.Tensor
    duration: float
    source_frames: int
    fingerprint: str

    @classmethod
    def load(cls, task: TaskSpec, control_dt: float, device: str = "cpu") -> "Trajectory":
        raw = read_transforms(task.trajectory, expected_fps=task.source_fps)
        sampled, times = resample_transforms(
            raw, task.source_fps, 1 / control_dt, task.smoothing_window
        )
        initial = np.eye(4)
        initial[:3, :3] = Rotation.from_euler("xyz", task.initial_euler).as_matrix()
        initial[:3, 3] = task.initial_position
        sampled = initial @ np.linalg.inv(sampled[0]) @ sampled
        rotations = Rotation.from_matrix(sampled[:, :3, :3])
        positions = sampled[:, :3, 3] + rotations.apply(
            np.broadcast_to(task.body_offset, (len(times), 3)).copy()
        )
        xyzw = rotations.as_quat()
        poses = np.column_stack((positions, xyzw[:, [3, 0, 1, 2]]))
        velocities = np.zeros((len(times), 6))
        delta_t = np.diff(times)
        velocities[:-1, :3] = np.diff(positions, axis=0) / delta_t[:, None]
        velocities[:-1, 3:] = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / delta_t[:, None]
        return cls(
            torch.as_tensor(poses, dtype=torch.float32, device=device),
            torch.as_tensor(velocities, dtype=torch.float32, device=device),
            float(times[-1]),
            len(raw),
            hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        )
