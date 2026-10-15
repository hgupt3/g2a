"""Calibrated SE(3) grasp areas and geometry initialization.

An area is a union of boxes around wrist poses, not a collection of grasp
keypoints. Translations and rotation vectors are expressed in each wrist's local
frame. Functional areas can pivot rotations at a measured grip center and bind
the hand's signed thumb side and palm normal to a tool's working frame.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .math import compose_pose, quat_apply, quat_conjugate, rotvec_to_quat

AREA_SCHEMA_VERSION = 1
WRIST_DIM = 6
LEAP_DISTAL_CONTACT_GROUPS = (
    ("thumb_dip", "thumb_fingertip"),
    ("pip", "dip", "fingertip"),
    ("pip_2", "dip_2", "fingertip_2"),
    ("pip_3", "dip_3", "fingertip_3"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numbers shaped {shape}")
    return array


def validate_transform(value, name: str = "transform") -> np.ndarray:
    matrix = _array(value, (4, 4), name)
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-7):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{name} rotation must be right-handed and orthonormal")
    return matrix


def _pose(matrix: np.ndarray) -> tuple[float, ...]:
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    if quaternion[3] < 0:
        quaternion *= -1
    return tuple(np.r_[matrix[:3, 3], quaternion[3], quaternion[:3]].tolist())


def _unit(value, name: str) -> np.ndarray:
    vector = _array(value, (3,), name)
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
        raise ValueError(f"{name} must be a unit vector")
    return vector


def signed_hand_frame(knuckles, thumb_base, palm_normal) -> np.ndarray:
    """Return [palm normal, thumb-side finger span, third axis] in root axes.

    A PCA finger-row axis is unsigned. Projecting the thumb base onto it fixes
    that sign using anatomy, rather than its largest world-coordinate component.
    """
    knuckles = _array(knuckles, (3, 3), "three knuckle centers")
    thumb = _array(thumb_base, (3,), "thumb base")
    normal = _unit(palm_normal, "palm normal")
    centered = knuckles - knuckles.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered)
    if singular[0] < 0.01 or singular[1] > 0.1 * singular[0]:
        raise ValueError("Finger knuckles must define a nondegenerate row")
    across = axes[0] - normal * (axes[0] @ normal)
    length = np.linalg.norm(across)
    if length < 1e-5:
        raise ValueError("Palm normal cannot be parallel to the finger row")
    across /= length
    signed_offset = float((thumb - knuckles.mean(axis=0)) @ across)
    if abs(signed_offset) < 0.005:
        raise ValueError("Thumb base does not identify a signed side of the finger row")
    across *= 1 if signed_offset > 0 else -1
    return np.column_stack((normal, across, np.cross(normal, across)))


@dataclass(frozen=True)
class FunctionalGripFrame:
    """An explicit tool-use convention, independent of PCA eigenvector signs.

    The thumb side points toward the working end. ``palm_side`` optionally fixes
    which signed working-plane normal the palm faces; otherwise either side is
    allowed. A per-patch center is the handle axis expressed in the hand root,
    including the surface-to-axis distance, and is the rotation pivot.
    """

    thumb_side_axis_root: tuple[float, float, float]
    palm_normal_root: tuple[float, float, float]
    working_direction_object: tuple[float, float, float]
    working_plane_normal_object: tuple[float, float, float]
    handle_segment_m: tuple[tuple[float, ...], tuple[float, ...]]
    grip_centers_root_m: tuple[tuple[float, ...], ...]
    max_axis_deviation_rad: float = 0.30
    max_palm_deviation_rad: float = 0.30
    max_transverse_distance_m: float = 0.015
    opposition: dict | None = None
    contact_segment_m: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    palm_side: int | None = None
    contact_center_bounds_root_m: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    handle_center_bounds_root_m: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def validate(self, patch_count: int) -> None:
        if self.palm_side is not None and (
            type(self.palm_side) is not int or self.palm_side not in (-1, 1)
        ):
            raise ValueError("Functional palm_side must be -1, +1, or null")
        axes = [
            _unit(getattr(self, name), name)
            for name in (
                "thumb_side_axis_root",
                "palm_normal_root",
                "working_direction_object",
                "working_plane_normal_object",
            )
        ]
        if abs(axes[0] @ axes[1]) > 1e-5 or abs(axes[2] @ axes[3]) > 1e-5:
            raise ValueError("Functional thumb/palm and tool axes must be orthogonal")
        segment = _array(self.handle_segment_m, (2, 3), "handle_segment_m")
        direction = segment[1] - segment[0]
        length = np.linalg.norm(direction)
        if length <= 0 or abs((direction / length) @ axes[2]) < 1 - 1e-5:
            raise ValueError("Functional handle segment must follow the working axis")
        _array(self.grip_centers_root_m, (patch_count, 3), "grip_centers_root_m")
        if self.contact_segment_m is not None:
            physical = _array(self.contact_segment_m, (2, 3), "contact_segment_m")
            contact_direction = physical[1] - physical[0]
            contact_length = np.linalg.norm(contact_direction)
            if (
                contact_length <= 0
                or abs((contact_direction / contact_length) @ axes[2]) < 1 - 1e-5
            ):
                raise ValueError(
                    "Physical contact segment must be parallel to the functional handle"
                )
            contact_direction /= contact_length
            offset = segment - physical[0]
            projection = offset @ contact_direction
            transverse = np.linalg.norm(offset - projection[:, None] * contact_direction, axis=-1)
            if (
                (projection < -1e-7).any()
                or (projection > contact_length + 1e-7).any()
                or (transverse > 1e-6).any()
            ):
                raise ValueError(
                    "Physical contact segment must contain the proposal-center segment"
                )
        if self.contact_center_bounds_root_m is not None:
            bounds = _array(
                self.contact_center_bounds_root_m, (2, 3), "contact_center_bounds_root_m"
            )
            if (bounds[0] >= bounds[1]).any() or self.opposition is None:
                raise ValueError(
                    "Hand contact-center bounds require ordered bounds and native opposition"
                )
            pivots = np.asarray(self.grip_centers_root_m)
            if ((pivots < bounds[0]) | (pivots > bounds[1])).any():
                raise ValueError(
                    "Functional proposal pivots must lie within hand contact-center bounds"
                )
        if self.handle_center_bounds_root_m is not None:
            bounds = _array(self.handle_center_bounds_root_m, (2, 3), "handle_center_bounds_root_m")
            if (bounds[0] >= bounds[1]).any():
                raise ValueError("Handle-center bounds must be strictly ordered")
        for name in ("max_axis_deviation_rad", "max_palm_deviation_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < math.pi / 2:
                raise ValueError(f"Functional {name} must lie between zero and pi/2")
        if not math.isfinite(self.max_transverse_distance_m) or self.max_transverse_distance_m <= 0:
            raise ValueError("Functional transverse distance must be positive")
        if self.opposition is not None:
            spec = self.opposition
            if not isinstance(spec, dict):
                raise ValueError("Opposition requirements must be an object")
            if set(spec) != {
                "body_groups",
                "minimum_contact_force_n",
                "minimum_contact_fraction",
                "minimum_finger_groups",
            }:
                raise ValueError(
                    "Opposition requires explicit distal body groups, force, fraction, and finger count"
                )
            groups = spec["body_groups"]
            if (
                not isinstance(groups, (list, tuple))
                or len(groups) != 4
                or any(not isinstance(group, (list, tuple)) or not group for group in groups)
                or any(not isinstance(name, str) or not name for group in groups for name in group)
            ):
                raise ValueError(
                    "Opposition body_groups must declare thumb and three finger groups"
                )
            names = [name for group in groups for name in group]
            if len(names) != len(set(names)):
                raise ValueError("Opposition groups must contain distinct distal body names")
            if any(
                set(group) != set(expected)
                for group, expected in zip(groups, LEAP_DISTAL_CONTACT_GROUPS, strict=True)
            ):
                raise ValueError(
                    "LEAP opposition requires distal thumb/finger groups; MCP, base, and thumb_pip are excluded"
                )
            count = spec["minimum_finger_groups"]
            if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 3:
                raise ValueError("Opposition minimum_finger_groups must be an integer in [2,3]")
            for key in ("minimum_contact_force_n", "minimum_contact_fraction"):
                value = spec[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"Opposition {key} must be finite and positive")
            if spec["minimum_contact_fraction"] > 1:
                raise ValueError("Opposition contact fraction must not exceed one")


@dataclass(frozen=True)
class AreaPatch:
    pose_wxyz: tuple[float, ...]
    translation_bounds_m: tuple[tuple[float, ...], tuple[float, ...]]
    rotation_bounds_rad: tuple[tuple[float, ...], tuple[float, ...]]
    name: str = "wrist"

    def validate(self) -> None:
        pose = _array(self.pose_wxyz, (7,), "pose_wxyz")
        if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-5):
            raise ValueError("Area center must use a unit wxyz quaternion")
        for field in ("translation_bounds_m", "rotation_bounds_rad"):
            bounds = _array(getattr(self, field), (2, 3), field)
            if (bounds[1] < bounds[0]).any():
                raise ValueError(f"{field} upper bound must be at least its lower bound")
            if (bounds[0] > 0).any() or (bounds[1] < 0).any():
                raise ValueError(f"{field} must contain the center at zero")
        angular = np.asarray(self.rotation_bounds_rad)
        if np.linalg.norm(np.abs(angular).max(axis=0)) >= math.pi:
            raise ValueError("Rotation-vector box must lie inside the pi-radius ball")


@dataclass(frozen=True)
class GraspArea:
    patches: tuple[AreaPatch, ...]
    object_asset_sha256: str
    calibration: dict
    source: dict
    schema_version: int = AREA_SCHEMA_VERSION
    frame: str = "object_body"
    perturbation_frame: str = "wrist_local"

    @property
    def functional_frame(self) -> FunctionalGripFrame | None:
        value = self.calibration.get("functional_grip")
        return None if value is None else FunctionalGripFrame(**value)

    def validate(self) -> None:
        if self.schema_version != AREA_SCHEMA_VERSION:
            raise ValueError("Unsupported grasp-area schema version")
        if self.frame != "object_body" or self.perturbation_frame != "wrist_local":
            raise ValueError("Grasp area must use object_body poses and wrist_local perturbations")
        if not self.patches:
            raise ValueError("A grasp area must contain at least one box")
        for patch in self.patches:
            patch.validate()
        if len(self.object_asset_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.object_asset_sha256
        ):
            raise ValueError("Grasp area must record the object asset SHA256")
        _array(self.calibration.get("body_offset_m"), (3,), "calibration.body_offset_m")
        if self.source.get("kind") not in ("geometry", "human_rgbd", "defined"):
            raise ValueError("Area source.kind must be geometry, human_rgbd, or defined")
        if not isinstance(self.source.get("description"), str) or not self.source["description"]:
            raise ValueError("Grasp area must explain its source")
        frame = self.functional_frame
        if frame is not None:
            frame.validate(len(self.patches))
            # With grip-center rotation, translations alone move that center.
            # Check every box corner, rather than trusting only its center.
            segment = np.asarray(frame.handle_segment_m, dtype=np.float64)
            direction = segment[1] - segment[0]
            length = np.linalg.norm(direction)
            direction /= length
            for patch, pivot in zip(self.patches, frame.grip_centers_root_m, strict=True):
                rotation = Rotation.from_quat(np.r_[patch.pose_wxyz[4:], patch.pose_wxyz[3]])
                angle_radius = np.linalg.norm(np.abs(patch.rotation_bounds_rad).max(axis=0))
                for root_axis, object_axis, budget, unsigned in (
                    (
                        frame.thumb_side_axis_root,
                        frame.working_direction_object,
                        frame.max_axis_deviation_rad,
                        False,
                    ),
                    (
                        frame.palm_normal_root,
                        np.asarray(frame.working_plane_normal_object) * (frame.palm_side or 1),
                        frame.max_palm_deviation_rad,
                        frame.palm_side is None,
                    ),
                ):
                    alignment = float(rotation.apply(root_axis) @ object_axis)
                    center_angle = math.acos(
                        np.clip(abs(alignment) if unsigned else alignment, -1, 1)
                    )
                    if center_angle + angle_radius > budget + 1e-7:
                        raise ValueError(
                            "Functional area rotation box exceeds its signed axis constraint"
                        )
                corners = np.array(
                    [
                        [x, y, z]
                        for x in np.asarray(patch.translation_bounds_m)[:, 0]
                        for y in np.asarray(patch.translation_bounds_m)[:, 1]
                        for z in np.asarray(patch.translation_bounds_m)[:, 2]
                    ]
                )
                contact = np.asarray(patch.pose_wxyz[:3]) + rotation.apply(pivot)
                delta = contact + rotation.apply(corners) - segment[0]
                along = delta @ direction
                distance = np.linalg.norm(delta - along[:, None] * direction, axis=-1)
                if (
                    along.min() < -1e-7
                    or along.max() > length + 1e-7
                    or distance.max() > frame.max_transverse_distance_m + 1e-7
                ):
                    raise ValueError(
                        "Functional area translation box leaves its usable handle segment"
                    )
        # This also catches non-JSON provenance and NaNs hidden in metadata.
        json.dumps(asdict(self), allow_nan=False)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        object_asset: Path | None = None,
        body_offset: tuple[float, ...] | None = None,
        hand_asset: Path | None = None,
    ) -> GraspArea:
        data = json.loads(Path(path).read_text())
        data["patches"] = tuple(
            AreaPatch(
                pose_wxyz=tuple(patch["pose_wxyz"]),
                translation_bounds_m=tuple(tuple(row) for row in patch["translation_bounds_m"]),
                rotation_bounds_rad=tuple(tuple(row) for row in patch["rotation_bounds_rad"]),
                name=patch.get("name", "wrist"),
            )
            for patch in data["patches"]
        )
        area = cls(**data)
        area.validate()
        if object_asset is not None and sha256(object_asset) != area.object_asset_sha256:
            raise ValueError("Grasp area was calibrated for a different object asset")
        if body_offset is not None and not np.allclose(
            body_offset, area.calibration["body_offset_m"], atol=1e-8, rtol=0
        ):
            raise ValueError("Grasp area was calibrated for a different body_offset")
        expected_hand = area.calibration.get("hand_asset_sha256")
        if hand_asset is not None and expected_hand and sha256(hand_asset) != expected_hand:
            raise ValueError("Grasp area was calibrated for a different hand asset")
        return area

    def save(self, path: Path | str) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def fingerprint(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(
                asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def uniform_parameters(self, count: int, *, device: str = "cpu", seed: int = 0) -> torch.Tensor:
        if count < 1:
            raise ValueError("Area sample count must be positive")
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.rand((count, WRIST_DIM), device=device, generator=generator) * 2 - 1

    def decode_wrist(self, parameters: torch.Tensor) -> torch.Tensor:
        """Map six normalized coordinates to wrist poses inside the union of boxes.

        Coordinate 0 selects a patch in equal-width intervals and is remapped to
        the local box's x coordinate. The other five coordinates retain their
        usual meaning. Independent uniform parameters therefore sample every
        box equally and sample translations/rotation-vector components uniformly
        within it. This is not uniform Haar sampling over SO(3).
        """
        if parameters.shape[-1] != WRIST_DIM or not parameters.is_floating_point():
            raise ValueError(
                "Wrist search parameters must be floating tensors ending in six values"
            )
        if not torch.isfinite(parameters).all() or (parameters.abs() > 1 + 1e-6).any():
            raise ValueError("Wrist search parameters must be finite and lie in [-1, 1]")
        coordinate = (parameters[..., 0].clamp(-1, 1) + 1) * 0.5 * len(self.patches)
        patch_id = coordinate.floor().long().clamp_max(len(self.patches) - 1)
        local = parameters.clone()
        local[..., 0] = 2 * (coordinate - patch_id) - 1
        centers = parameters.new_tensor([patch.pose_wxyz for patch in self.patches])[patch_id]
        translation = parameters.new_tensor([patch.translation_bounds_m for patch in self.patches])[
            patch_id
        ]
        angular = parameters.new_tensor([patch.rotation_bounds_rad for patch in self.patches])[
            patch_id
        ]
        translation_delta = translation[..., 0, :] + (local[..., :3] + 1) * 0.5 * (
            translation[..., 1, :] - translation[..., 0, :]
        )
        rotation_delta = angular[..., 0, :] + (local[..., 3:] + 1) * 0.5 * (
            angular[..., 1, :] - angular[..., 0, :]
        )
        decoded = compose_pose(
            centers, torch.cat((translation_delta, rotvec_to_quat(rotation_delta)), -1)
        )
        frame = self.functional_frame
        if frame is not None:
            pivot = parameters.new_tensor(frame.grip_centers_root_m)[patch_id]
            # Rotate at the selected handle center, not the wrist root: otherwise
            # a 0.35-rad perturbation moves a 90-mm lever by more than 30 mm.
            decoded[..., :3] += quat_apply(centers[..., 3:], pivot) - quat_apply(
                decoded[..., 3:], pivot
            )
        return decoded

    def functional_metrics(
        self,
        poses: torch.Tensor,
        patch_ids: torch.Tensor,
        *,
        contact_centers_object_m: torch.Tensor | None = None,
        contact_center_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Measure signed tool-use axes and the declared handle-center region.

        The fixed FK pivot defines proposal coordinates. When final native
        contact geometry is supplied, the midpoint of thumb contact and the
        distinct finger-group contacts defines the actual handle center instead.
        This avoids treating a calibration joint posture as a final contact point.
        Areas without a functional convention retain their original decoding.
        """
        if (
            not poses.is_floating_point()
            or poses.shape[-1] != 7
            or not torch.isfinite(poses).all()
            or patch_ids.shape != poses.shape[:-1]
            or patch_ids.dtype != torch.long
            or (patch_ids < 0).any()
            or (patch_ids >= len(self.patches)).any()
        ):
            raise ValueError("Functional metrics require finite wrist poses and valid patch IDs")
        zero = torch.zeros_like(poses[..., 0])
        frame = self.functional_frame
        if frame is None:
            return {"feasible": torch.ones_like(zero, dtype=torch.bool)}
        rotation = poses[..., 3:]

        def angle(root_axis, object_axis, unsigned=False):
            hand_axis = quat_apply(rotation, poses.new_tensor(root_axis).expand_as(poses[..., :3]))
            tool_axis = poses.new_tensor(object_axis).expand_as(hand_axis)
            dot = (hand_axis * tool_axis).sum(-1)
            cross = torch.linalg.vector_norm(torch.cross(hand_axis, tool_axis, dim=-1), dim=-1)
            return torch.atan2(cross, dot.abs() if unsigned else dot)

        axis_angle = angle(frame.thumb_side_axis_root, frame.working_direction_object)
        palm_angle = angle(
            frame.palm_normal_root,
            np.asarray(frame.working_plane_normal_object) * (frame.palm_side or 1),
            frame.palm_side is None,
        )
        center = poses[..., :3] + quat_apply(
            rotation, poses.new_tensor(frame.grip_centers_root_m)[patch_ids]
        )
        measured_contact_valid = None
        if contact_centers_object_m is not None or contact_center_valid is not None:
            if (
                frame.opposition is None
                or not isinstance(contact_centers_object_m, torch.Tensor)
                or contact_centers_object_m.shape != (*poses.shape[:-1], 4, 3)
                or not contact_centers_object_m.is_floating_point()
                or not torch.isfinite(contact_centers_object_m).all()
                or not isinstance(contact_center_valid, torch.Tensor)
                or contact_center_valid.shape != (*poses.shape[:-1], 4)
                or contact_center_valid.dtype != torch.bool
            ):
                raise ValueError(
                    "Functional contact centers require finite native four-group geometry"
                )
            finger_valid = contact_center_valid[..., 1:]
            finger_count = finger_valid.sum(-1)
            measured_contact_valid = contact_center_valid[..., 0] & (
                finger_count >= frame.opposition["minimum_finger_groups"]
            )
            finger_center = (contact_centers_object_m[..., 1:, :] * finger_valid[..., None]).sum(
                -2
            ) / finger_count.clamp_min(1)[..., None]
            measured_center = (contact_centers_object_m[..., 0, :] + finger_center) * 0.5
            # Missing groups remain invalid. Use the proposal pivot only for
            # steering diagnostics, never as a replacement physical certificate.
            center = torch.where(measured_contact_valid[..., None], measured_center, center)
        segment = poses.new_tensor(
            frame.contact_segment_m
            if measured_contact_valid is not None and frame.contact_segment_m is not None
            else frame.handle_segment_m
        )
        direction = segment[1] - segment[0]
        length = torch.linalg.vector_norm(direction)
        direction /= length
        delta = center - segment[0]
        along = (delta * direction).sum(-1)
        excess = torch.maximum(-along, along - length).clamp_min(0)
        transverse = torch.linalg.vector_norm(delta - along[..., None] * direction, dim=-1)
        metrics = {
            "feasible": (axis_angle <= frame.max_axis_deviation_rad + 1e-6)
            & (palm_angle <= frame.max_palm_deviation_rad + 1e-6)
            & (excess <= 1e-6)
            & (transverse <= frame.max_transverse_distance_m + 1e-6),
            "axis_angle_rad": axis_angle,
            "palm_angle_rad": palm_angle,
            "segment_excess_m": excess,
            "transverse_distance_m": transverse,
        }
        if frame.handle_center_bounds_root_m is not None:
            # A fixed object landmark locates the tool along its handle. A
            # force-weighted contact centroid can move as the thumb loads and
            # therefore cannot define the tool's axial placement in the hand.
            reference = poses.new_tensor(
                frame.contact_segment_m
                if frame.contact_segment_m is not None
                else frame.handle_segment_m
            ).mean(0)
            root_reference = quat_apply(quat_conjugate(rotation), reference - poses[..., :3])
            bounds = poses.new_tensor(frame.handle_center_bounds_root_m)
            excess_root = torch.maximum(
                bounds[0] - root_reference, root_reference - bounds[1]
            ).clamp_min(0)
            metrics["handle_center_root_m"] = root_reference
            metrics["handle_placement_excess_m"] = torch.linalg.vector_norm(excess_root, dim=-1)
            metrics["feasible"] &= metrics["handle_placement_excess_m"] <= 1e-6
        if frame.contact_center_bounds_root_m is not None:
            bounds = poses.new_tensor(frame.contact_center_bounds_root_m)
            root_center = quat_apply(quat_conjugate(rotation), center - poses[..., :3])
            root_excess = torch.maximum(bounds[0] - root_center, root_center - bounds[1]).clamp_min(
                0
            )
            metrics["contact_center_root_m"] = root_center
            metrics["hand_pocket_excess_m"] = torch.linalg.vector_norm(root_excess, dim=-1)
            metrics["feasible"] &= metrics["hand_pocket_excess_m"] <= 1e-6
        if measured_contact_valid is not None:
            group_along = ((contact_centers_object_m - segment[0]) * direction).sum(-1)
            group_excess = torch.maximum(-group_along, group_along - length).clamp_min(0)
            groups_in_handle = ((group_excess <= 1e-6) | ~contact_center_valid).all(-1)
            metrics["feasible"] &= measured_contact_valid & groups_in_handle
            metrics["contact_center_valid"] = measured_contact_valid
            metrics["contact_groups_in_handle"] = groups_in_handle
            metrics["contact_group_segment_excess_m"] = group_excess
            metrics["contact_center_object_m"] = center
        return metrics

    def opposition_contacts(
        self, forces: torch.Tensor, body_names
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpret tool-filtered native forces without counting MCP/base contact.

        The caller supplies a fresh public PhysX force matrix and accumulates
        simultaneous thumb/finger coverage over the declared free-settle window.
        Forces from hand self-contact must never be supplied as tool contact.
        """
        frame = self.functional_frame
        if frame is None or frame.opposition is None:
            raise ValueError("Area does not declare opposition contact requirements")
        spec = frame.opposition
        if (
            not isinstance(forces, torch.Tensor)
            or not forces.is_floating_point()
            or forces.ndim != 3
            or forces.shape[1:] != (len(body_names), 3)
            or not torch.isfinite(forces).all()
            or len(set(body_names)) != len(body_names)
        ):
            raise ValueError(
                "Opposition needs finite tool-filtered forces [N,body,3] and unique body names"
            )
        if any(name not in body_names for group in spec["body_groups"] for name in group):
            raise ValueError("Opposition distal body is missing from the native hand")
        magnitude = torch.linalg.vector_norm(forces, dim=-1)
        groups = torch.stack(
            [
                (
                    magnitude[:, [body_names.index(name) for name in group]]
                    >= spec["minimum_contact_force_n"]
                ).any(-1)
                for group in spec["body_groups"]
            ],
            dim=-1,
        )
        return groups[:, 0] & (groups[:, 1:].sum(-1) >= spec["minimum_finger_groups"]), groups

    poses = decode_wrist


@dataclass(frozen=True)
class PalmCalibration:
    """Hand-wide geometric calibration, independent of tool/task names.

    ``contact_root_m`` is a nominal power-grasp center in the robot root frame.
    +X points out of the palm; +Y spans the fingers. Values are inferred from
    palm bounds and knuckle transforms by ``calibrate_palm``.
    """

    contact_root_m: tuple[float, float, float]
    finger_span_m: float
    finger_width_m: float
    hand_asset_sha256: str
    method: str = "USD palm surface bounds and three finger knuckle transforms"
    pinch_centers_root_m: tuple[tuple[float, float, float], ...] = ()
    workspace_evidence: dict | None = None
    enveloping_contact_root_m: tuple[float, float, float] | None = None

    def validate(self) -> None:
        _array(self.contact_root_m, (3,), "palm contact_root_m")
        if (
            not math.isfinite(self.finger_span_m)
            or not math.isfinite(self.finger_width_m)
            or self.finger_width_m <= 0
            or self.finger_span_m < self.finger_width_m
        ):
            raise ValueError("Palm finger span/width must be positive measured dimensions")
        if len(self.hand_asset_sha256) != 64:
            raise ValueError("Palm calibration must record its hand asset SHA256")
        for center in self.pinch_centers_root_m:
            _array(center, (3,), "pinch workspace center")
        if self.enveloping_contact_root_m is not None:
            _array(self.enveloping_contact_root_m, (3,), "enveloping contact center")


@dataclass(frozen=True)
class DiscoveryConfig:
    surface_samples: int = 48000
    slices: int = 48
    minimum_slice_points: int = 12
    cluster_radius_m: float = 0.007
    minimum_thickness_m: float = 0.006
    maximum_width_m: float = 0.070
    minimum_roundness: float = 0.35
    # A handle may support two digits while the outer digit overhangs its end.
    minimum_handle_span_fraction: float = 0.6
    approach_clearance_m: float = 0.003
    outward_search_m: float = 0.015
    transverse_search_m: float = 0.015
    angle_range_rad: float = 0.35
    approaches: int = 4
    maximum_handles: int = 2
    seed: int = 0
    axis_signs: tuple[int, ...] = (1, -1)
    workspace_mode: str = "auto"

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if key in ("axis_signs", "workspace_mode"):
                continue
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Discovery setting {key} must be finite")
        if self.workspace_mode not in ("auto", "palm", "enveloping", "pinch", "both"):
            raise ValueError("workspace_mode must be auto, palm, enveloping, pinch, or both")
        if (
            not isinstance(self.axis_signs, (tuple, list))
            or not self.axis_signs
            or any(
                isinstance(sign, bool) or not isinstance(sign, int) or sign not in (-1, 1)
                for sign in self.axis_signs
            )
            or len(set(self.axis_signs)) != len(self.axis_signs)
        ):
            raise ValueError("axis_signs must contain distinct +1 and/or -1 directions")
        for key in (
            "surface_samples",
            "slices",
            "minimum_slice_points",
            "approaches",
            "maximum_handles",
            "seed",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < (
                0 if key == "seed" else 1
            ):
                raise ValueError(f"Discovery setting {key} must be a positive integer")
        if self.slices < 4 or self.surface_samples < self.slices * self.minimum_slice_points:
            raise ValueError("Geometry discovery needs at least four adequately sampled slices")
        if self.minimum_thickness_m <= 0 or self.maximum_width_m <= self.minimum_thickness_m:
            raise ValueError("Geometry discovery needs positive, increasing thickness/width limits")
        if not 0 < self.minimum_roundness <= 1 or not 0 < self.minimum_handle_span_fraction <= 2:
            raise ValueError("Invalid handle roundness or minimum span fraction")
        if self.cluster_radius_m <= 0 or not 0 < self.angle_range_rad < math.pi / math.sqrt(3):
            raise ValueError("Invalid clustering distance or wrist angular search range")
        if min(self.approach_clearance_m, self.outward_search_m, self.transverse_search_m) < 0:
            raise ValueError("Discovery clearance and search distances cannot be negative")


def read_usd_mesh(path: Path, root_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read visual/collision triangles with units and instance transforms applied.

    USD Python is supplied by Isaac Sim after AppLauncher, or by optional
    ``usd-core`` for CPU preparation. Geometry is expressed in the rigid root's
    frame, not the mesh's untransformed point coordinates.
    """
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "USD discovery requires Isaac Sim or the optional usd-core package"
        ) from exc
    stage = Usd.Stage.Open(str(Path(path).resolve()))
    if stage is None:
        raise ValueError(f"Could not open USD asset: {path}")
    root = stage.GetPrimAtPath(root_path) if root_path else stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise ValueError(f"USD root is missing: {root_path or 'defaultPrim'}")
    if root_path is None:
        bodies = [
            prim
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(bodies) > 1:
            raise ValueError(
                "Tool USD must contain one rigid body; specify a geometry root explicitly"
            )
        if bodies:
            root = bodies[0]
    cache = UsdGeom.XformCache()
    inverse_root = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(root)))
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    if not math.isfinite(meters) or meters <= 0:
        raise ValueError("USD metersPerUnit must be finite and positive")
    vertices, faces = [], []
    count = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim)) @ inverse_root
        transformed = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3] * meters
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        sizes = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        triangles = []
        cursor = 0
        for size in sizes:
            polygon = indices[cursor : cursor + size]
            triangles.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, size - 1))
            cursor += size
        if triangles:
            vertices.append(transformed)
            faces.append(np.asarray(triangles) + count)
            count += len(points)
    if not vertices:
        raise ValueError(f"USD root contains no triangular geometry: {path}")
    return np.concatenate(vertices), np.concatenate(faces)


def _pinch_workspace(
    hand_asset: Path, stage, cache, prefix: str, fractions=(0.4, 0.5)
) -> tuple[tuple, dict]:
    """Measure fingertip-midpoint hypotheses from the authored LEAP joint tree.

    Reference configurations use fractions of the *asset's* positive flexion
    limits, not a per-tool wrist offset. These midpoints do not establish
    opposing contact normals or self-collision-free closure, so discovery uses
    them only when pinch hypotheses are requested explicitly.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    from .stress import CLOSING_JOINTS

    meters = UsdGeom.GetStageMetersPerUnit(stage)

    def transform(position, quaternion):
        matrix = np.eye(4)
        matrix[:3, 3] = list(position)
        matrix[:3, :3] = Rotation.from_quat(
            [*quaternion.GetImaginary(), quaternion.GetReal()]
        ).as_matrix()
        return matrix

    joints = []
    upper = np.zeros(16)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix)):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        number = int(prim.GetName().removeprefix("a_"))
        parent = str(joint.GetBody0Rel().GetTargets()[0])
        child = str(joint.GetBody1Rel().GetTargets()[0])
        axis = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
        joints.append(
            (
                number,
                parent,
                child,
                transform(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                transform(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
                axis,
            )
        )
        upper[number] = math.radians(joint.GetUpperLimitAttr().Get())
    if len(joints) != 16:
        raise ValueError("LEAP workspace calibration requires its complete 16-joint tree")

    def forward(positions):
        links = {prefix + "/base": np.eye(4)}
        pending = joints.copy()
        while pending:
            previous_count = len(pending)
            for entry in pending.copy():
                number, parent, child, parent_joint, child_joint, axis = entry
                if parent not in links:
                    continue
                delta = np.eye(4)
                delta[:3, :3] = Rotation.from_rotvec(axis * positions[number]).as_matrix()
                links[child] = links[parent] @ parent_joint @ delta @ np.linalg.inv(child_joint)
                pending.remove(entry)
            if len(pending) == previous_count:
                raise ValueError("Hand joint tree is disconnected or cyclic")
        return links

    root_world = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prefix + "/base"))).T
    zero = forward(np.zeros(16))
    mismatch = max(
        np.abs(
            pose
            - np.linalg.inv(root_world)
            @ np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
        ).max()
        for path, pose in zero.items()
    )
    if mismatch > 1e-5:
        raise ValueError("Authored hand poses disagree with zero-angle USD joint frames")
    tips = {}
    for role, link in (("middle", "fingertip_2"), ("thumb", "thumb_fingertip")):
        points, _ = read_usd_mesh(hand_asset, prefix + "/" + link + "/visuals")
        # Geometry is local to visuals. Remove that local transform so the pad
        # center can be carried by the rigid fingertip's forward kinematics.
        visual_world = np.asarray(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prefix + "/" + link + "/visuals"))
        ).T
        link_world = np.asarray(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prefix + "/" + link))
        ).T
        visual_link = np.linalg.inv(link_world) @ visual_world
        points = (np.c_[points / meters, np.ones(len(points))] @ visual_link.T)[:, :3]
        tips[role] = (prefix + "/" + link, np.quantile(points, (0.02, 0.98), axis=0).mean(axis=0))
    centers, references = [], []
    for fraction in fractions:
        positions = np.zeros(16)
        positions[list(CLOSING_JOINTS)] = fraction * np.maximum(upper[list(CLOSING_JOINTS)], 0)
        positions[12] = 0.5 * max(upper[12], 0)
        links = forward(positions)
        pad_centers = {
            role: (links[path] @ np.r_[point, 1])[:3] * meters
            for role, (path, point) in tips.items()
        }
        center = (pad_centers["middle"] + pad_centers["thumb"]) * 0.5
        centers.append(tuple(center.tolist()))
        references.append(
            {
                "positive_limit_fraction": fraction,
                "joint_positions_rad": positions.tolist(),
                "pad_centers_root_m": {role: point.tolist() for role, point in pad_centers.items()},
                "tip_link_poses_root_m": {
                    role: _pose(
                        np.block(
                            [
                                [links[path][:3, :3], links[path][:3, 3:4] * meters],
                                [np.zeros((1, 3)), np.ones((1, 1))],
                            ]
                        )
                    )
                    for role, (path, _) in tips.items()
                },
                "pad_separation_m": float(
                    np.linalg.norm(pad_centers["middle"] - pad_centers["thumb"])
                ),
            }
        )
    return tuple(centers), {
        "method": "midpoint of middle/thumb fingertip geometry under USD forward kinematics",
        "zero_angle_max_transform_error": float(mismatch),
        "reference_configurations": references,
        "contact_interpretation": "hypothesized rod center between digits, not a verified contact pocket; unlike surface centers, no object radius is added",
    }


def calibrate_palm(
    hand_asset: Path, *, workspace_fractions: tuple[float, ...] = (0.4, 0.5)
) -> PalmCalibration:
    """Derive root-frame geometry and explicit flexion workspace hypotheses.

    Fractions sample the asset's positive joint limits, including deeper
    enveloping configurations when requested. These are geometric hypotheses;
    fresh native closure/contact/stability audits must establish actual support.
    The default retains the original discovery hypotheses and serialization.
    """
    if (
        not isinstance(workspace_fractions, (tuple, list))
        or not workspace_fractions
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value < 1
            for value in workspace_fractions
        )
        or any(
            left >= right
            for left, right in zip(workspace_fractions, workspace_fractions[1:], strict=False)
        )
    ):
        raise ValueError("Workspace fractions must be distinct increasing finite values in (0, 1)")
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError("Palm calibration requires Isaac Sim or optional usd-core") from exc
    stage = Usd.Stage.Open(str(Path(hand_asset).resolve()))
    root = stage.GetDefaultPrim()
    prefix = str(root.GetPath())
    base = stage.GetPrimAtPath(prefix + "/base")
    if not base.IsValid():
        raise ValueError("LEAP calibration requires its rigid root body named base")
    points, _ = read_usd_mesh(hand_asset, prefix + "/base/visuals")
    # Articulation root poses use the rigid base link, which can differ from
    # the USD asset container. Normalize every geometric family to that link.
    cache = UsdGeom.XformCache()
    origin = cache.GetLocalToWorldTransform(base)
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    relative = np.asarray(
        cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prefix + "/base/visuals"))
    ) @ np.linalg.inv(np.asarray(origin))
    points = (np.c_[points / meters, np.ones(len(points))] @ relative)[:, :3] * meters
    knuckles = []
    for link in ("mcp_joint", "mcp_joint_2", "mcp_joint_3"):
        prim = stage.GetPrimAtPath(prefix + "/" + link)
        if not prim.IsValid():
            raise ValueError(
                "Automatic palm calibration supports the LEAP three-finger layout; supply explicit calibration for another hand"
            )
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)) @ np.linalg.inv(
            np.asarray(origin)
        )
        knuckles.append(transform[3, :3] * meters)
    knuckles = np.asarray(knuckles)
    lower, upper = np.quantile(points, (0.002, 0.998), axis=0)
    finger_spacing = np.diff(np.sort(knuckles[:, 1])).mean()
    finger_width = min(float(finger_spacing), float(upper[0] - lower[0]))
    contact = (
        float(upper[0]),
        float(knuckles[:, 1].mean()),
        float(knuckles[:, 2].mean() - finger_width * 0.5),
    )
    pinch_centers, evidence = _pinch_workspace(
        hand_asset, stage, cache, prefix, fractions=tuple(workspace_fractions)
    )
    enveloping = (float(upper[0]), float(knuckles[:, 1].mean()), float(knuckles[:, 2].mean()))
    evidence["root_body_path"] = str(base.GetPath())
    thumb = stage.GetPrimAtPath(prefix + "/thumb_temp_base")
    if not thumb.IsValid():
        raise ValueError("LEAP functional calibration requires its thumb base transform")
    thumb_transform = np.asarray(cache.GetLocalToWorldTransform(thumb)) @ np.linalg.inv(
        np.asarray(origin)
    )
    thumb_position = thumb_transform[3, :3] * meters
    functional_axes = signed_hand_frame(knuckles, thumb_position, (1, 0, 0))
    evidence["functional_hand_frame"] = {
        "palm_normal_root": functional_axes[:, 0].tolist(),
        "thumb_side_axis_root": functional_axes[:, 1].tolist(),
        "third_axis_root": functional_axes[:, 2].tolist(),
        "knuckles_root_m": knuckles.tolist(),
        "thumb_base_root_m": thumb_position.tolist(),
        "method": "finger-row PCA signed by thumb-base projection; LEAP palm front is root +X, all positions normalized to rigid base",
    }
    evidence["enveloping_contact_interpretation"] = (
        "palm front at mean finger MCP height; add handle radius and approach clearance. "
        "Thin-stem enveloping grips need support at the knuckles rather than the lower palm or a fingertip-centroid midpoint."
    )
    return PalmCalibration(
        contact,
        float(np.ptp(knuckles[:, 1]) + finger_width),
        finger_width,
        sha256(hand_asset),
        pinch_centers_root_m=pinch_centers,
        workspace_evidence=evidence,
        enveloping_contact_root_m=enveloping,
    )


def sample_surface(
    vertices: np.ndarray, triangles: np.ndarray, count: int, seed: int
) -> np.ndarray:
    """Area-weighted surface sampling avoids dependence on mesh tessellation."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite (N,3) points")
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(triangles) == 0
        or triangles.min() < 0
        or triangles.max() >= len(vertices)
    ):
        raise ValueError("Mesh triangles must index three existing vertices")
    mesh = vertices[triangles]
    weights = np.linalg.norm(np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]), axis=1)
    if not weights.sum() > 0 or count < 1:
        raise ValueError("Surface sampling needs positive area and sample count")
    generator = np.random.default_rng(seed)
    selected = mesh[generator.choice(len(mesh), count, p=weights / weights.sum())]
    barycentric = generator.random((count, 2))
    barycentric[:, 0] = np.sqrt(barycentric[:, 0])
    return (
        (1 - barycentric[:, :1]) * selected[:, 0]
        + barycentric[:, :1] * (1 - barycentric[:, 1:]) * selected[:, 1]
        + barycentric[:, :1] * barycentric[:, 1:] * selected[:, 2]
    )


def _profile_tracks(local: np.ndarray, config: DiscoveryConfig) -> list[list[dict]]:
    lower, upper = np.quantile(local[:, 2], (0.002, 0.998))
    edges = np.linspace(lower, upper, config.slices + 1)
    tracks: list[list[dict]] = []
    active: list[int] = []
    for index, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        slice_points = local[(local[:, 2] >= a) & (local[:, 2] < b)]
        if len(slice_points) < config.minimum_slice_points:
            active = []
            continue
        pairs = cKDTree(slice_points[:, :2]).query_pairs(
            config.cluster_radius_m, output_type="ndarray"
        )
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(slice_points), len(slice_points)),
        ).tocsr()
        components, labels = connected_components(graph, directed=False)
        profiles = []
        for label in range(components):
            component = slice_points[labels == label]
            if len(component) < config.minimum_slice_points:
                continue
            bounds = np.quantile(component[:, :2], (0.02, 0.98), axis=0)
            width = bounds[1] - bounds[0]
            if (
                width.min() < config.minimum_thickness_m
                or width.max() > config.maximum_width_m
                or width.min() / width.max() < config.minimum_roundness
            ):
                continue
            profiles.append(
                {
                    "index": index,
                    "lo": a,
                    "hi": b,
                    "center": np.r_[bounds.mean(axis=0), (a + b) * 0.5],
                    "bounds": bounds,
                    "width": width,
                }
            )
        previous = active
        active = []
        for profile in profiles:
            matches = [
                track
                for track in previous
                if np.linalg.norm(tracks[track][-1]["center"][:2] - profile["center"][:2])
                <= 2 * config.cluster_radius_m
            ]
            if matches:
                track = min(
                    matches,
                    key=lambda i: np.linalg.norm(
                        tracks[i][-1]["center"][:2] - profile["center"][:2]
                    ),
                )
                previous.remove(track)
                tracks[track].append(profile)
            else:
                track = len(tracks)
                tracks.append([profile])
            active.append(track)
    return tracks


def discover_from_points(
    points: np.ndarray,
    palm: PalmCalibration,
    *,
    object_asset_sha256: str,
    body_offset: tuple[float, ...],
    config: DiscoveryConfig | None = None,
) -> GraspArea:
    """Find graspable stems, loop-handle rails, and transverse handle bars.

    Slice components are joined along each principal axis; thin blades and wide
    working ends are rejected using thickness, width, roundness, and span limits.
    This geometric prior cannot infer semantic tool use for arbitrary shapes.
    Ambiguous tools need an explicitly defined functional area.
    """
    config = config or DiscoveryConfig()
    config.validate()
    palm.validate()
    if config.workspace_mode == "pinch" and not palm.pinch_centers_root_m:
        raise ValueError("Pinch discovery requires fingertip workspace calibration for the hand")
    if config.workspace_mode == "enveloping" and palm.enveloping_contact_root_m is None:
        raise ValueError("Enveloping discovery requires the hand's MCP-height surface calibration")
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or len(points) < config.minimum_slice_points * config.slices
    ):
        raise ValueError("Discovery requires a finite, adequately sampled (N,3) surface")
    center = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(points - center, rowvar=False))
    for i in range(3):
        if axes[np.abs(axes[:, i]).argmax(), i] < 0:
            axes[:, i] *= -1
    minimum_span = palm.finger_span_m * config.minimum_handle_span_fraction
    candidates = []
    for axis_id in (2, 1, 0):
        longitudinal = axes[:, axis_id]
        transverse = axes[:, (axis_id + 1) % 3]
        second = np.cross(longitudinal, transverse)
        basis = np.column_stack((transverse, second, longitudinal))
        local = (points - center) @ basis
        for track in _profile_tracks(local, config):
            if track[-1]["hi"] - track[0]["lo"] >= minimum_span:
                candidates.append({"track": track, "basis": basis})
    if not candidates:
        raise ValueError(
            "No graspable continuous handle region found; provide a defined box or calibrate the discovery limits"
        )
    candidates.sort(
        key=lambda candidate: (
            (candidate["track"][-1]["hi"] - candidate["track"][0]["lo"])
            / np.mean([p["width"].max() for p in candidate["track"]])
        ),
        reverse=True,
    )
    patches, handles = [], []
    for handle_id, candidate in enumerate(candidates[: config.maximum_handles]):
        track, basis = candidate["track"], candidate["basis"]
        transverse, second, longitudinal = basis.T
        a, b = track[0]["lo"], track[-1]["hi"]
        transverse_center = np.mean([profile["center"][:2] for profile in track], axis=0)
        widths = np.max([profile["width"] for profile in track], axis=0)
        mode = config.workspace_mode
        if mode == "auto":
            # A rod narrower than one digit needs a knuckle-height enveloping
            # pocket. Broader handles can use the lower palm surface. Distal
            # midpoint pockets remain explicit hypotheses, not automatic ones.
            mode = (
                "enveloping"
                if palm.enveloping_contact_root_m is not None and widths.max() < palm.finger_width_m
                else "palm"
            )
        workspaces = []
        if mode in ("palm", "both"):
            workspaces.append(("palm", palm.contact_root_m))
        if mode in ("enveloping", "both") and palm.enveloping_contact_root_m is not None:
            workspaces.append(("enveloping", palm.enveloping_contact_root_m))
        if mode in ("pinch", "both"):
            workspaces.extend(
                (f"pinch_{i}", point) for i, point in enumerate(palm.pinch_centers_root_m)
            )
        handle_center = center + basis @ np.r_[transverse_center, (a + b) * 0.5]
        span = b - a
        along_half = max(0.0, (span - minimum_span) * 0.5)
        handles.append(
            {
                "center_m": handle_center.tolist(),
                "axis": longitudinal.tolist(),
                "span_m": float(span),
                "cross_width_m": widths.tolist(),
                "longitudinal_interval_m": [float(a), float(b)],
                "profile_count": len(track),
                "workspace_families": [name for name, _ in workspaces],
            }
        )
        for approach_id in range(config.approaches):
            angle = 2 * math.pi * approach_id / config.approaches
            radial = math.cos(angle) * transverse + math.sin(angle) * second
            radius = (abs(math.cos(angle)) * widths[0] + abs(math.sin(angle)) * widths[1]) * 0.5
            angular = ((-config.angle_range_rad,) * 3, (config.angle_range_rad,) * 3)
            for axis_sign in config.axis_signs:
                across = axis_sign * longitudinal
                rotation = np.column_stack((-radial, across, np.cross(-radial, across)))
                for workspace_name, contact_root in workspaces:
                    hand_contact = np.asarray(contact_root).copy()
                    if workspace_name in ("palm", "enveloping"):
                        hand_contact[0] += radius + config.approach_clearance_m
                        x_bounds = (-config.outward_search_m, config.approach_clearance_m)
                    else:
                        x_bounds = (-config.outward_search_m, config.outward_search_m)
                    matrix = np.eye(4)
                    matrix[:3, :3] = rotation
                    matrix[:3, 3] = handle_center - rotation @ hand_contact
                    translation = (
                        (x_bounds[0], -along_half, -config.transverse_search_m),
                        (x_bounds[1], along_half, config.transverse_search_m),
                    )
                    patches.append(
                        AreaPatch(
                            _pose(matrix),
                            translation,
                            angular,
                            f"handle_{handle_id}_approach_{approach_id}_axis_{axis_sign:+d}_{workspace_name}",
                        )
                    )
    tool_axis = axes[:, 2]
    tool_coordinate = (points - center) @ tool_axis
    lower, upper = np.quantile(tool_coordinate, (0.002, 0.998))
    grip_coordinate = (np.asarray(handles[0]["center_m"]) - center) @ tool_axis
    # The working end is estimated from the longitudinal endpoint furthest from
    # the discovered handle. Its centroid is useful for load torque placement.
    endpoint = lower if grip_coordinate > (lower + upper) * 0.5 else upper
    terminal_span = max((upper - lower) / config.slices * 3, minimum_span * 0.5)
    endpoint_points = points[np.abs(tool_coordinate - endpoint) <= terminal_span]
    working_center = endpoint_points.mean(axis=0)
    area = GraspArea(
        tuple(patches),
        object_asset_sha256,
        {
            "body_offset_m": list(body_offset),
            "hand_asset_sha256": palm.hand_asset_sha256,
            "palm": asdict(palm),
        },
        {
            "kind": "geometry",
            "description": "Area-weighted USD surface slices identify continuous handle regions; root placement uses measured palm/MCP-height surfaces or explicitly requested fingertip-midpoint hypotheses, with both allowed axial orientations. No human RGB-D extraction was used.",
            "algorithm": "surface_profile_workspace_v3",
            "settings": asdict(config),
            "handles": handles,
            "working_end": {
                "point_m": working_center.tolist(),
                "bounds_m": [endpoint_points.min(0).tolist(), endpoint_points.max(0).tolist()],
                "method": "surface centroid of terminal segment opposite longest narrow handle",
                "semantic_use_verified": False,
            },
        },
    )
    area.validate()
    return area


def define_functional_area(
    points: np.ndarray,
    palm: PalmCalibration,
    *,
    object_asset_sha256: str,
    body_offset: tuple[float, ...],
    handle_segment_m,
    working_end_m,
    description: str,
    transverse_search_m: float = 0.008,
    outward_search_m: float = 0.010,
    clearance_m: float = 0.003,
    angle_range_rad: float = 0.12,
    workspace_mode: str = "surface",
    opposition: dict | None = None,
    palm_side: int | None = None,
) -> GraspArea:
    """Define a functional power-grip volume using measured tool/hand frames.

    ``handle_segment_m`` bounds acceptable *grip-center* positions, not the full
    physical handle. Its direction is signed toward ``working_end_m``. Geometry
    supplies the broad working plane and local handle radius; an operator still
    declares the usable segment and intended tool-use convention explicitly.
    ``palm_side`` selects a signed face of the geometry-derived plane; omitting
    it retains both faces. ``opposing_pads`` uses the measured FK thumb/
    middle pad midpoints directly; ``surface`` uses lower-palm/MCP surfaces.
    """
    palm.validate()
    if workspace_mode not in ("surface", "opposing_pads"):
        raise ValueError("Functional workspace_mode must be surface or opposing_pads")
    if workspace_mode == "opposing_pads" and not palm.pinch_centers_root_m:
        raise ValueError("Opposing-pad areas require measured digit FK workspace calibration")
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 30
        or not np.isfinite(points).all()
    ):
        raise ValueError("Functional area needs a finite, adequately sampled (N,3) surface")
    if (
        not all(
            math.isfinite(value)
            for value in (transverse_search_m, outward_search_m, clearance_m, angle_range_rad)
        )
        or min(transverse_search_m, outward_search_m, clearance_m) <= 0
        or not 0 < angle_range_rad < 0.30 / math.sqrt(3)
    ):
        raise ValueError(
            "Functional search dimensions must be positive and rotations inside the axis cone"
        )
    segment = _array(handle_segment_m, (2, 3), "handle_segment_m")
    working = _array(working_end_m, (3,), "working_end_m")
    center = segment.mean(axis=0)
    axis = segment[1] - segment[0]
    length = np.linalg.norm(axis)
    if length < 0.005:
        raise ValueError("Functional grip-center segment must span at least 5 mm")
    axis /= length
    signed_distance = float((working - center) @ axis)
    if abs(signed_distance) <= length / 2:
        raise ValueError("Working end must lie beyond the grip-center segment")
    working_axis = axis * (1 if signed_distance > 0 else -1)
    _, eigenvectors = np.linalg.eigh(np.cov(points, rowvar=False))
    plane_normal = eigenvectors[:, 0]
    plane_normal -= working_axis * (plane_normal @ working_axis)
    norm = np.linalg.norm(plane_normal)
    if norm < 0.5:
        raise ValueError("Tool geometry does not define a transverse working-plane normal")
    plane_normal /= norm
    if plane_normal[np.abs(plane_normal).argmax()] < 0:
        plane_normal *= -1
    calibration = (palm.workspace_evidence or {}).get("functional_hand_frame")
    if calibration is None:
        raise ValueError("Functional areas need the hand's signed thumb-side calibration")
    hand_normal = _unit(calibration["palm_normal_root"], "palm_normal_root")
    thumb_axis = _unit(calibration["thumb_side_axis_root"], "thumb_side_axis_root")
    hand_frame = np.column_stack((hand_normal, thumb_axis, np.cross(hand_normal, thumb_axis)))
    relative = points - center
    axial = relative @ axis
    handle_points = points[np.abs(axial) <= length / 2]
    if len(handle_points) < 30:
        raise ValueError("Usable segment has insufficient sampled handle geometry")
    radial_coordinate = (handle_points - center) @ plane_normal
    radial_bounds = np.quantile(radial_coordinate, (0.002, 0.998))
    patches, pivots = [], []
    if workspace_mode == "opposing_pads":
        workspaces = [
            (f"opposing_pads_{i}", point) for i, point in enumerate(palm.pinch_centers_root_m)
        ]
    else:
        workspaces = [("palm", palm.contact_root_m)]
        if palm.enveloping_contact_root_m is not None:
            workspaces.append(("enveloping", palm.enveloping_contact_root_m))
    if palm_side is not None and (type(palm_side) is not int or palm_side not in (-1, 1)):
        raise ValueError("Functional palm_side must be -1, +1, or null")
    for side in (1, -1) if palm_side is None else (palm_side,):
        normal = side * plane_normal
        tool_frame = np.column_stack((normal, working_axis, np.cross(normal, working_axis)))
        rotation = tool_frame @ hand_frame.T
        # +normal points from palm into the handle. The palm sits on its
        # opposite surface, whose support distance is measured independently.
        radius = float(-radial_bounds[0] if side == 1 else radial_bounds[1])
        if radius <= 0:
            raise ValueError("Grip-center segment must pass inside the handle cross-section")
        for family, contact in workspaces:
            pivot = np.asarray(contact).copy()
            if workspace_mode == "surface":
                pivot += (radius + clearance_m) * hand_normal
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = center - rotation @ pivot
            # Hand-local Y differs slightly from its measured finger-row axis.
            # Project the longitudinal extent conservatively at validation.
            along_half = length * 0.5 - transverse_search_m * abs(thumb_axis[2])
            patches.append(
                AreaPatch(
                    _pose(matrix),
                    (
                        (-outward_search_m, -along_half, -transverse_search_m),
                        (
                            outward_search_m if workspace_mode == "opposing_pads" else clearance_m,
                            along_half,
                            transverse_search_m,
                        ),
                    ),
                    ((-angle_range_rad,) * 3, (angle_range_rad,) * 3),
                    f"functional_palm_side_{side:+d}_{family}",
                )
            )
            pivots.append(tuple(pivot.tolist()))
    constraint = FunctionalGripFrame(
        tuple(thumb_axis),
        tuple(hand_normal),
        tuple(working_axis),
        tuple(plane_normal),
        tuple(tuple(row) for row in segment),
        tuple(pivots),
        max_transverse_distance_m=math.hypot(outward_search_m, transverse_search_m) + 0.001,
        opposition=opposition,
        palm_side=palm_side,
    )
    area = GraspArea(
        tuple(patches),
        object_asset_sha256,
        {
            "body_offset_m": list(body_offset),
            "hand_asset_sha256": palm.hand_asset_sha256,
            "palm": asdict(palm),
            "functional_grip": asdict(constraint),
        },
        {
            "kind": "defined",
            "description": description,
            "algorithm": "signed_functional_power_grip_v1",
            "working_end": {"point_m": working.tolist(), "semantic_use_verified": False},
            "working_plane": {
                "normal_object": plane_normal.tolist(),
                "method": "smallest surface-covariance axis, projected perpendicular to the declared handle segment",
            },
            "handle_grip_center_segment_m": segment.tolist(),
            "settings": {
                "transverse_search_m": transverse_search_m,
                "outward_search_m": outward_search_m,
                "clearance_m": clearance_m,
                "angle_range_rad": angle_range_rad,
                "workspace_mode": workspace_mode,
            },
            "orientation_convention": (
                "thumb side toward working end; palm normal perpendicular to broad working plane, "
                + ("either face" if palm_side is None else f"signed side {palm_side:+d}")
                + "; perturb rotations about measured handle center"
            ),
        },
    )
    area.validate()
    return area


def discover_area(
    task,
    output: Path,
    *,
    hand_asset: Path | None = None,
    config: DiscoveryConfig | None = None,
) -> GraspArea:
    if hand_asset is None:
        hand_asset = Path(__file__).with_name("resources") / "assets/leap_hand/leap.usd"
    config = config or DiscoveryConfig()
    config.validate()
    vertices, triangles = read_usd_mesh(task.object_asset)
    points = sample_surface(vertices, triangles, config.surface_samples, config.seed)
    area = discover_from_points(
        points,
        calibrate_palm(hand_asset),
        object_asset_sha256=sha256(task.object_asset),
        body_offset=task.body_offset,
        config=config,
    )
    area.save(output)
    return area
