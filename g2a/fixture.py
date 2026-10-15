"""Releasable world support for grasp acquisition.

Author a FixedJoint before cloning. Frame zero is a world reset goal; frame one
is the dynamic tool's rigid frame. Each clone has independent frame/enabled USD
opinions. Release before free settling and keep disabled during stress/motion.
No integration-time tool pose writes, extra colliders, kinematic actor or
unsupported GPU target API is required.
"""

from dataclasses import dataclass

import numpy as np

FIXTURE_RELATIVE_PATH = "grasp_fixture"
JOINT_RELATIVE_PATH = f"{FIXTURE_RELATIVE_PATH}/joint"


@dataclass(frozen=True)
class FixturePaths:
    joint_path: str
    object_path: str


def fixture_paths(environment_path: str, object_path: str | None = None) -> FixturePaths:
    environment_path = environment_path.rstrip("/")
    if not environment_path.startswith("/") or not environment_path:
        raise ValueError("Fixture environment path must be absolute")
    object_path = object_path or f"{environment_path}/object"
    if not object_path.startswith(environment_path + "/"):
        raise ValueError("Fixture tool must be inside its own environment")
    return FixturePaths(f"{environment_path}/{JOINT_RELATIVE_PATH}", object_path)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validated_pose(value) -> np.ndarray:
    pose = _numpy(value).astype(np.float64, copy=True)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Fixture pose must contain finite position+wxyz quaternion")
    norm = np.linalg.norm(pose[3:])
    if norm == 0:
        raise ValueError("Fixture quaternion must be nonzero")
    pose[3:] /= norm
    return pose


def author_fixture(
    stage, *, environment_path="/World/envs/env_0", object_path=None, initial_pose=None
) -> FixturePaths:
    """Author source support; ``initial_pose`` is WORLD position+wxyz.

    None derives the authored tool's world pose. Every clone must receive its
    own world goal through write_target_poses before supported integration:
    physics cloning does not automatically rebase an unbound world joint frame.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    paths = fixture_paths(environment_path, object_path)
    environment = stage.GetPrimAtPath(environment_path)
    tool = stage.GetPrimAtPath(paths.object_path)
    if not environment or not tool or not tool.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Fixture requires an existing environment and dynamic tool rigid body")
    body = UsdPhysics.RigidBodyAPI(tool)
    if body.GetKinematicEnabledAttr().Get() or body.GetRigidBodyEnabledAttr().Get() is False:
        raise ValueError("Fixture tool must remain an enabled dynamic rigid body")
    if stage.GetPrimAtPath(paths.joint_path):
        raise ValueError("Fixture support already exists")
    if initial_pose is None:
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(tool)
        decomposition = Gf.Transform(transform)
        if not np.allclose(decomposition.GetScale(), (1, 1, 1), atol=1e-6, rtol=0):
            raise ValueError("Fixture tool rigid frame must not contain scale")
        quaternion = decomposition.GetRotation().GetQuat()
        initial_pose = [
            *decomposition.GetTranslation(),
            quaternion.GetReal(),
            *quaternion.GetImaginary(),
        ]
    pose = _validated_pose(initial_pose)
    UsdGeom.Xform.Define(stage, f"{environment_path}/{FIXTURE_RELATIVE_PATH}")
    joint = UsdPhysics.FixedJoint.Define(stage, paths.joint_path)
    joint.CreateBody0Rel().SetTargets([])
    joint.CreateBody1Rel().SetTargets([paths.object_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*pose[:3].tolist()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(pose[3]), Gf.Vec3f(*pose[4:].tolist())))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    # Cloned world frames are not rebased automatically. Keep every joint off
    # until reset has written that environment's actual world goal.
    joint.CreateJointEnabledAttr().Set(False)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    return paths


class TemporaryFixture:
    """Independent world support frames with sparse, reset-only updates."""

    def __init__(self, stage, environment_paths, *, object_relative_path="object"):
        from pxr import Gf, Sdf, UsdPhysics

        self.paths = tuple(
            fixture_paths(path, f"{path.rstrip('/')}/{object_relative_path}")
            for path in environment_paths
        )
        if not self.paths or len({path.joint_path for path in self.paths}) != len(self.paths):
            raise ValueError("Fixture environment paths must be nonempty and unique")
        attributes, enabled, frames = [], [], []
        for paths in self.paths:
            joint = UsdPhysics.FixedJoint.Get(stage, paths.joint_path)
            tool = stage.GetPrimAtPath(paths.object_path)
            if (
                not joint
                or not tool
                or not tool.HasAPI(UsdPhysics.RigidBodyAPI)
                or UsdPhysics.RigidBodyAPI(tool).GetKinematicEnabledAttr().Get()
                or UsdPhysics.RigidBodyAPI(tool).GetRigidBodyEnabledAttr().Get() is False
            ):
                raise ValueError("Fixture clone requires an enabled dynamic tool")
            if joint.GetBody0Rel().GetTargets() or [
                str(path) for path in joint.GetBody1Rel().GetTargets()
            ] != [paths.object_path]:
                raise ValueError(
                    "Fixture endpoints must be world and the tool in its own environment"
                )
            if joint.GetLocalPos1Attr().Get() != Gf.Vec3f(
                0
            ) or joint.GetLocalRot1Attr().Get() != Gf.Quatf(1):
                raise ValueError("Fixture tool-side joint frame must be identity")
            attribute = joint.GetJointEnabledAttr()
            value = attribute.Get()
            attributes.append(attribute)
            enabled.append(True if value is None else bool(value))
            position, rotation = joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()
            if position.Get() is None or rotation.Get() is None:
                raise ValueError("Fixture world position and orientation must be authored")
            frames.append((position, rotation, position.Get(), rotation.Get()))
        self._attributes = tuple(attributes)
        self._frame_attributes = tuple((position, rotation) for position, rotation, _, _ in frames)
        self._enabled = np.array(enabled, dtype=np.bool_)
        # Resolve handles before ChangeBlock. Explicit local opinions isolate
        # every clone from changes to source frame/enable attributes.
        with Sdf.ChangeBlock():
            for attribute, value in zip(self._attributes, self._enabled, strict=True):
                if not attribute.Set(bool(value)):
                    raise RuntimeError("Could not initialize independent fixture enable state")
            for position, rotation, p, q in frames:
                position.Set(p)
                rotation.Set(q)

    @property
    def num_envs(self) -> int:
        return len(self.paths)

    @property
    def enabled(self) -> np.ndarray:
        return self._enabled.copy()

    def _indices(self, env_ids, *, preserve_order=False) -> np.ndarray:
        if env_ids is None:
            return np.arange(self.num_envs, dtype=np.int64)
        ids = _numpy(env_ids)
        if ids.shape == (0,):
            return np.empty(0, dtype=np.int64)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("Fixture environment IDs must be a one-dimensional integer array")
        if (ids < 0).any() or (ids >= self.num_envs).any():
            raise IndexError("Fixture environment ID is outside the scene")
        unique = np.unique(ids)
        if preserve_order and len(unique) != len(ids):
            raise ValueError("Fixture target reset IDs must be unique")
        return (ids if preserve_order else unique).astype(np.int64, copy=False)

    def set_enabled(self, env_ids, enabled: bool) -> int:
        from pxr import Sdf

        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("Fixture enable state must be Boolean")
        ids = self._indices(env_ids)
        ids = ids[self._enabled[ids] != enabled]
        if not len(ids):
            return 0
        with Sdf.ChangeBlock():
            for index in ids:
                if not self._attributes[index].Set(bool(enabled)):
                    raise RuntimeError("Could not update fixture enable state")
        self._enabled[ids] = enabled
        return len(ids)

    def write_target_poses(self, poses, *, env_ids=None) -> None:
        """Set WORLD support goals once at reset; do not write body state.

        Preserve incoming sparse row order and normalize world rotations. Frame
        writes never enable a joint; the environment explicitly controls release.
        """
        from pxr import Gf, Sdf

        ids = self._indices(env_ids, preserve_order=True)
        values = _numpy(poses).astype(np.float64, copy=True)
        if values.shape != (len(ids), 7) or not np.isfinite(values).all():
            raise ValueError("Fixture target poses must be finite position+wxyz rows matching IDs")
        norms = np.linalg.norm(values[:, 3:], axis=-1)
        if (norms == 0).any():
            raise ValueError("Fixture target quaternion must be nonzero")
        if not len(ids):
            return
        values[:, 3:] /= norms[:, None]
        with Sdf.ChangeBlock():
            for index, pose in zip(ids, values, strict=True):
                position, rotation = self._frame_attributes[index]
                if not position.Set(Gf.Vec3f(*pose[:3].tolist())) or not rotation.Set(
                    Gf.Quatf(float(pose[3]), Gf.Vec3f(*pose[4:].tolist()))
                ):
                    raise RuntimeError("Could not update fixture world goal")

    def update_mask(self, enabled_mask) -> int:
        mask = _numpy(enabled_mask)
        if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
            raise ValueError("Fixture enable mask must be Boolean with one value per environment")
        changed = self._enabled != mask
        return self.set_enabled(np.flatnonzero(changed & mask), True) + self.set_enabled(
            np.flatnonzero(changed & ~mask), False
        )

    def release_all(self) -> int:
        return self.set_enabled(None, False)

    def metadata(self) -> dict:
        return {
            "mode": "fixed_joint",
            "mechanism": "releasable world FixedJoint to dynamic tool",
            "joint_relative_path": JOINT_RELATIVE_PATH,
            "support_pose_updates": "USD localPos0/localRot0 once at reset in world frame",
            "tool_pose_writes": "reset/restoration only; never used as support during integration",
            "release": "joint disabled before free settling and throughout wrench/trajectory motion",
            "joint_frames": "body0=world reset goal; body1=identity in tool rigid-link frame",
            "independent_clone_frame_and_enable_opinions": True,
            "kinematic_actor_required": False,
            "added_collision_shapes": 0,
            "unsupported_gpu_target_api_used": False,
            "initially_disabled_until_reset": True,
        }
