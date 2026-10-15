"""Native-pose collision feasibility against authored boxes and cooked hulls.

Contact offsets generate contacts before geometric touching; they are not an
allowance for shape penetration. This module reports geometric overlap in
meters, with explicit numerical acceptance budgets supplied by the caller.
Adjacent links excluded by authored joints and explicit filtered pairs are
honored. Other self intersections are checked, including palm/finger overlap.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _unit_directions(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=-1)
    values = (
        values[lengths > np.finfo(np.float64).eps]
        / lengths[lengths > np.finfo(np.float64).eps, None]
    )
    if not len(values):
        return np.empty((0, 3))
    # Direction sign does not affect SAT. Canonicalization avoids testing the
    # same hull face or edge repeatedly because of polygon triangulation.
    nonzero = np.abs(values) > 32 * np.finfo(np.float64).eps
    first = nonzero.argmax(axis=-1)
    values *= np.sign(values[np.arange(len(values)), first])[:, None]
    return np.unique(np.round(values, 12), axis=0)


@dataclass(frozen=True)
class CollisionBox:
    name: str
    body: str
    center: np.ndarray
    axes: np.ndarray
    half_extents: np.ndarray

    def __post_init__(self):
        center = np.asarray(self.center, dtype=np.float64)
        axes = np.asarray(self.axes, dtype=np.float64)
        extents = np.asarray(self.half_extents, dtype=np.float64)
        if center.shape != (3,) or axes.shape != (3, 3) or extents.shape != (3,):
            raise ValueError("Box requires center[3], column axes[3,3], half_extents[3]")
        if (
            not all(np.isfinite(value).all() for value in (center, axes, extents))
            or (extents <= 0).any()
        ):
            raise ValueError("Collision boxes must have finite, positive dimensions")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-7):
            raise ValueError("Collision box transforms contain unsupported shear")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "half_extents", extents)

    def fingerprint_data(self) -> dict:
        return {
            "name": self.name,
            "body": self.body,
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "half_extents": self.half_extents.tolist(),
        }


@dataclass(frozen=True)
class CollisionHull:
    name: str
    vertices: np.ndarray

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) < 4
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("A collision hull requires at least four finite 3D vertices")
        vertices = np.unique(vertices, axis=0)
        try:
            hull = ConvexHull(vertices)
        except QhullError as error:
            raise ValueError("Collision hull is not a full-dimensional convex solid") from error
        vertices = vertices[hull.vertices]
        vertices = vertices[np.lexsort(vertices.T[::-1])]
        hull = ConvexHull(vertices)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "normals", _unit_directions(hull.equations[:, :3]))
        # Edges between coplanar triangles are internal face diagonals, rather
        # than polyhedron edges. Drop them to bound the edge-cross-axis bank.
        edge_faces = {}
        for i, face in enumerate(hull.simplices):
            for j, k in ((0, 1), (1, 2), (2, 0)):
                edge_faces.setdefault(tuple(sorted((int(face[j]), int(face[k])))), []).append(i)
        directions = []
        for (i, j), faces in edge_faces.items():
            if len(faces) == 2:
                normals = hull.equations[faces, :3]
                if np.allclose(normals[0], normals[1], atol=1e-9, rtol=0):
                    continue
            directions.append(vertices[j] - vertices[i])
        object.__setattr__(self, "edges", _unit_directions(np.asarray(directions)))

    def fingerprint_data(self) -> dict:
        return {"name": self.name, "vertices": np.round(self.vertices, 12).tolist()}


@dataclass(frozen=True)
class HandCollisionGeometry:
    boxes: tuple[CollisionBox, ...]
    allowed_self_pairs: frozenset[tuple[int, int]] = frozenset()
    exclusions: tuple[dict, ...] = ()
    asset_sha256: str | None = None
    body_names: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.boxes or len({box.name for box in self.boxes}) != len(self.boxes):
            raise ValueError("Hand collision geometry requires uniquely named boxes")
        count = len(self.boxes)
        for i, j in self.allowed_self_pairs:
            if not 0 <= i < j < count:
                raise ValueError("Allowed self pairs must be ordered valid box indices")

    @classmethod
    def from_usd(cls, path: str | Path, *, body_names=None) -> "HandCollisionGeometry":
        """Read active box colliders in each native rigid-link frame, including instances."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(Path(path).resolve()))
        if stage is None or not stage.GetDefaultPrim():
            raise ValueError("Hand USD has no valid default primitive")
        root = stage.GetDefaultPrim()
        cache = UsdGeom.XformCache()
        meters = UsdGeom.GetStageMetersPerUnit(stage)
        bodies = {
            str(prim.GetPath()): prim
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        if len({prim.GetName() for prim in bodies.values()}) != len(bodies):
            raise ValueError("Hand rigid-link names must be unique")
        boxes, box_paths = [], []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            collision_ancestor = prim
            while collision_ancestor and not collision_ancestor.HasAPI(UsdPhysics.CollisionAPI):
                collision_ancestor = collision_ancestor.GetParent()
            if (
                not collision_ancestor
                or not UsdPhysics.CollisionAPI(collision_ancestor).GetCollisionEnabledAttr().Get()
            ):
                continue
            if not prim.IsA(UsdGeom.Gprim):
                continue
            if not prim.IsA(UsdGeom.Cube):
                raise ValueError(
                    f"Unsupported hand collider: {prim.GetPath()} ({prim.GetTypeName()})"
                )
            prim_path = str(prim.GetPath())
            ancestors = [body_path for body_path in bodies if prim_path.startswith(body_path + "/")]
            if not ancestors:
                raise ValueError(f"Box collider is not inside a rigid link: {prim_path}")
            body_path = max(ancestors, key=len)
            body = bodies[body_path]
            local = (
                np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)
                @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
            )
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * meters
            scale = np.linalg.norm(local[:3, :3], axis=0)
            boxes.append(
                CollisionBox(
                    prim_path.removeprefix(str(root.GetPath())),
                    body.GetName(),
                    local[:3, 3] * meters,
                    local[:3, :3] / scale,
                    scale * size / 2,
                )
            )
            box_paths.append(prim_path)
        allowed, exclusions = set(), []
        # Same rigid body is one compound collider and cannot self collide.
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                if a.body == boxes[j].body:
                    allowed.add((i, j))
        disabled_body_pairs = set()
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                parents, children = (
                    joint.GetBody0Rel().GetTargets(),
                    joint.GetBody1Rel().GetTargets(),
                )
                if parents and children and not joint.GetCollisionEnabledAttr().Get():
                    names = sorted(
                        (
                            stage.GetPrimAtPath(parents[0]).GetName(),
                            stage.GetPrimAtPath(children[0]).GetName(),
                        )
                    )
                    disabled_body_pairs.add(tuple(names))
                    exclusions.append(
                        {
                            "bodies": names,
                            "reason": "authored_joint_collision_disabled",
                            "joint": prim.GetName(),
                        }
                    )
            # Filter relations may name a link, a collider group or one shape.
            # Expand only their actual descendant boxes, never an entire hand
            # merely because a filtered shape belongs to it.
            relation = prim.GetRelationship("physics:filteredPairs")
            if relation:
                source = str(prim.GetPath())
                source_boxes = [
                    i for i, p in enumerate(box_paths) if p == source or p.startswith(source + "/")
                ]
                for target in relation.GetTargets():
                    target = str(target)
                    target_boxes = [
                        i
                        for i, p in enumerate(box_paths)
                        if p == target or p.startswith(target + "/")
                    ]
                    for i in source_boxes:
                        for j in target_boxes:
                            if i != j:
                                allowed.add(tuple(sorted((i, j))))
                    exclusions.append(
                        {
                            "source": source.removeprefix(str(root.GetPath())),
                            "target": target.removeprefix(str(root.GetPath())),
                            "reason": "authored_filtered_pair",
                        }
                    )
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                if tuple(sorted((a.body, boxes[j].body))) in disabled_body_pairs:
                    allowed.add((i, j))
        if body_names is not None and (
            len(set(body_names)) != len(body_names)
            or set(body_names) != {p.GetName() for p in bodies.values()}
        ):
            raise ValueError("Explicit native body_names do not match the hand USD rigid links")
        return cls(
            tuple(boxes),
            frozenset(allowed),
            tuple(exclusions),
            _file_hash(path),
            tuple(body_names or ()),
        )


@dataclass(frozen=True)
class ToolCollisionGeometry:
    hulls: tuple[CollisionHull, ...]
    asset_sha256: str | None = None

    def __post_init__(self):
        if not self.hulls:
            raise ValueError("Tool collision geometry requires cooked convex pieces")

    @classmethod
    def from_stage(cls, stage, root_path: str, *, asset_path: str | Path | None = None):
        """Query the public SDK's actual configured convex collision representation."""
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

        root = stage.GetPrimAtPath(root_path)
        if not root or not root.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Tool geometry root must be its native rigid-body primitive")
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        cache = UsdGeom.XformCache()
        inverse_root = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(root)).T)
        meters = UsdGeom.GetStageMetersPerUnit(stage)
        hulls = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI) or UsdPhysics.MeshCollisionAPI(
                prim
            ).GetApproximationAttr().Get() not in ("convexHull", "convexDecomposition"):
                raise ValueError(
                    f"Tool collider is not a supported convex representation: {prim.GetPath()}"
                )
            returned = []
            get_physx_cooking_interface().request_convex_collision_representation(
                stage_id=stage_id,
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                run_asynchronously=False,
                on_result=lambda result, pieces, returned=returned: returned.append(
                    (result, pieces)
                ),
            )
            if not returned or returned[0][0] != PhysxCollisionRepresentationResult.RESULT_VALID:
                raise RuntimeError(f"Could not read actual cooked hulls: {prim.GetPath()}")
            local = inverse_root @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
            pieces = []
            for piece in returned[0][1]:
                vertices = np.asarray([[v.x, v.y, v.z] for v in piece.vertices], dtype=np.float64)
                vertices = (vertices @ local[:3, :3].T + local[:3, 3]) * meters
                pieces.append(vertices)
            # The SDK documents that decomposition order is not reproducible.
            # Bind geometry independently of its enumeration order.
            pieces.sort(key=lambda v: json.dumps(np.unique(np.round(v, 12), axis=0).tolist()))
            for i, vertices in enumerate(pieces):
                hulls.append(
                    CollisionHull(
                        str(prim.GetPath()).removeprefix(root_path) + f"/convex_{i}", vertices
                    )
                )
        return cls(tuple(hulls), _file_hash(asset_path) if asset_path is not None else None)


def read_native_link_poses(
    hand, tool, *, simulation_view=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read public native transforms without a same-timestamp asset pose cache.

    After root/joint writes callers must run sim.forward(), or supply the public
    simulation view so forward articulation kinematics is updated here. These
    are rigid-link/actor transforms, rather than center-of-mass transforms.
    """
    if simulation_view is not None:
        simulation_view.update_articulations_kinematic()
    links = hand.root_physx_view.get_link_transforms().clone()
    obj = tool.root_physx_view.get_transforms().clone()
    links[..., 3:] = links[..., [6, 3, 4, 5]]
    obj[..., 3:] = obj[..., [6, 3, 4, 5]]
    return links, obj


def _rotation(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def obb_pair_penetration(
    centers_a: torch.Tensor,
    axes_a: torch.Tensor,
    half_a: torch.Tensor,
    centers_b: torch.Tensor,
    axes_b: torch.Tensor,
    half_b: torch.Tensor,
) -> torch.Tensor:
    """Exact 15-axis SAT overlap for oriented box pairs; zero means separated/touching."""
    cross = torch.cross(
        axes_a.transpose(-1, -2).unsqueeze(-2), axes_b.transpose(-1, -2).unsqueeze(-3), dim=-1
    )
    axes = torch.cat(
        (axes_a.transpose(-1, -2), axes_b.transpose(-1, -2), cross.flatten(-3, -2)), dim=-2
    )
    lengths = axes.norm(dim=-1)
    valid = lengths > 4 * torch.finfo(axes.dtype).eps
    axes = axes / lengths.clamp_min(torch.finfo(axes.dtype).tiny).unsqueeze(-1)
    ra = (torch.einsum("...ai,...ij->...aj", axes, axes_a).abs() * half_a.unsqueeze(-2)).sum(-1)
    rb = (torch.einsum("...ai,...ij->...aj", axes, axes_b).abs() * half_b.unsqueeze(-2)).sum(-1)
    delta = torch.einsum("...ai,...i->...a", axes, centers_b - centers_a).abs()
    overlap = (ra + rb - delta).masked_fill(~valid, torch.inf)
    return overlap.amin(-1).clamp_min(0)


def box_hull_penetration(center, box_axes, half_extents, vertices, normals, edges):
    """Complete SAT: both face banks and box-edge × hull-edge directions."""
    count = len(center)
    box_faces = box_axes.transpose(-1, -2)
    cross = torch.cross(box_faces[:, :, None], edges[None, None], dim=-1).flatten(1, 2)
    axes = torch.cat((box_faces, normals[None].expand(count, -1, -1), cross), dim=1)
    lengths = axes.norm(dim=-1)
    valid = lengths > 4 * torch.finfo(axes.dtype).eps
    axes = axes / lengths.clamp_min(torch.finfo(axes.dtype).tiny).unsqueeze(-1)
    projected_center = torch.einsum("pai,pi->pa", axes, center)
    radius = (torch.einsum("pai,pij->paj", axes, box_axes).abs() * half_extents[:, None]).sum(-1)
    projection = torch.einsum("pai,vi->pav", axes, vertices)
    overlap = torch.minimum(
        projected_center + radius - projection.amin(-1),
        projection.amax(-1) - (projected_center - radius),
    )
    overlap.masked_fill_(~valid, torch.inf)
    return overlap.amin(-1).clamp_min(0)


class CollisionModel:
    """Cached vectorized collision checks, with bounded temporary pair batches."""

    def __init__(
        self,
        hand: HandCollisionGeometry,
        tool: ToolCollisionGeometry,
        body_names: list[str] | tuple[str, ...] | None = None,
        *,
        device="cpu",
        dtype=torch.float32,
        max_tool_penetration_m: float,
        max_self_penetration_m: float,
        pair_chunk_size: int,
    ):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Collision SAT requires float32 or float64")
        for value in (max_tool_penetration_m, max_self_penetration_m):
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("Collision penetration budgets must be finite and nonnegative")
        if (
            isinstance(pair_chunk_size, bool)
            or not isinstance(pair_chunk_size, int)
            or pair_chunk_size < 1
        ):
            raise ValueError("Collision pair chunk size must be a positive integer")
        body_names = tuple(body_names) if body_names is not None else hand.body_names
        if not body_names:
            raise ValueError("Supply explicit native body_names to geometry or model")
        if len(set(body_names)) != len(body_names):
            raise ValueError("Native body names must be unique")
        ids = {name: i for i, name in enumerate(body_names)}
        if any(box.body not in ids for box in hand.boxes):
            raise ValueError("Authored collision boxes do not match native rigid-link names")
        self.hand, self.tool = hand, tool
        self.device, self.dtype = torch.device(device), dtype
        self.body_names = tuple(body_names)
        self.max_tool_penetration_m = float(max_tool_penetration_m)
        self.max_self_penetration_m = float(max_self_penetration_m)
        self.pair_chunk_size = pair_chunk_size

        def tensor(value):
            return torch.as_tensor(np.asarray(value), device=self.device, dtype=dtype)

        self.body_ids = torch.tensor([ids[box.body] for box in hand.boxes], device=self.device)
        self.centers = tensor([box.center for box in hand.boxes])
        self.axes = tensor([box.axes for box in hand.boxes])
        self.extents = tensor([box.half_extents for box in hand.boxes])
        self.hulls = [(tensor(h.vertices), tensor(h.normals), tensor(h.edges)) for h in tool.hulls]
        self.hull_bounds = [(v.amin(0), v.amax(0)) for v, _, _ in self.hulls]
        self.hull_lower = torch.stack([bounds[0] for bounds in self.hull_bounds])
        self.hull_upper = torch.stack([bounds[1] for bounds in self.hull_bounds])
        pairs = [
            (i, j)
            for i in range(len(hand.boxes))
            for j in range(i + 1, len(hand.boxes))
            if hand.boxes[i].body != hand.boxes[j].body and (i, j) not in hand.allowed_self_pairs
        ]
        self.self_pairs = torch.tensor(pairs, dtype=torch.int64, device=self.device).reshape(-1, 2)
        data = {
            "boxes": [box.fingerprint_data() for box in hand.boxes],
            "hulls": [hull.fingerprint_data() for hull in sorted(tool.hulls, key=lambda h: h.name)],
            "self_excluded_box_pairs": sorted(hand.allowed_self_pairs),
            "hand_asset_sha256": hand.asset_sha256,
            "tool_asset_sha256": tool.asset_sha256,
        }
        self.geometry_sha256 = _canonical_hash(data)

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "method": "native_box_cooked_convex_sat",
            "native_pose_source": "PhysX body_link_state_w/root_link_state_w",
            "native_pose_readback": "public get_link_transforms/get_transforms after forward kinematics; bypasses same-timestamp asset caches",
            "geometry_sha256": self.geometry_sha256,
            "hand_asset_sha256": self.hand.asset_sha256,
            "tool_asset_sha256": self.tool.asset_sha256,
            "hand_box_count": len(self.hand.boxes),
            "tool_convex_count": len(self.tool.hulls),
            "max_tool_penetration_m": self.max_tool_penetration_m,
            "max_self_penetration_m": self.max_self_penetration_m,
            "tolerance_m": {
                "tool_hand": self.max_tool_penetration_m,
                "nonadjacent_self": self.max_self_penetration_m,
            },
            "self_collision_exclusions": [
                entry["bodies"] if "bodies" in entry else [entry["source"], entry["target"]]
                for entry in self.hand.exclusions
            ],
            "self_collision_exclusion_details": list(self.hand.exclusions),
            "self_collision_exclusions_source": "same rigid link; authored joint collisionEnabled=False; authored physics:filteredPairs expanded only to affected boxes",
            "frames": "boxes native link-local; cooked hulls object rigid-body-local; inputs world-frame position+wxyz quaternion",
            "contact_offset_is_penetration_allowance": False,
            "phases": ["initial", "closure", "free_settle", "wrench", "nominal_trajectory"],
        }

    def provenance(self) -> dict:
        return self.metadata()

    @torch.no_grad()
    def metrics(
        self, hand_link_poses: torch.Tensor, object_poses: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if (
            hand_link_poses.ndim != 3
            or hand_link_poses.shape[1:] != (len(self.body_names), 7)
            or object_poses.shape != (len(hand_link_poses), 7)
        ):
            raise ValueError("Expected hand_link_poses[N,bodies,7] and object_poses[N,7]")
        if (
            hand_link_poses.device != self.device
            or object_poses.device != self.device
            or hand_link_poses.dtype != self.dtype
            or object_poses.dtype != self.dtype
        ):
            raise ValueError("Collision model and native poses must share device and dtype")
        count = len(hand_link_poses)
        if not count:
            empty = torch.empty(0, device=self.device, dtype=self.dtype)
            return {
                "tool_hand_penetration_m": empty,
                "nonadjacent_self_penetration_m": empty.clone(),
                "feasible": torch.empty(0, device=self.device, dtype=torch.bool),
            }
        valid = torch.isfinite(hand_link_poses).all(dim=(-1, -2)) & torch.isfinite(
            object_poses
        ).all(-1)
        valid &= (hand_link_poses[..., 3:].norm(dim=-1) > 0).all(-1) & (
            object_poses[..., 3:].norm(dim=-1) > 0
        )
        poses = torch.nan_to_num(hand_link_poses)
        obj = torch.nan_to_num(object_poses)
        link_rotation, object_rotation = _rotation(poses[..., 3:]), _rotation(obj[..., 3:])
        box_rotation = link_rotation[:, self.body_ids]
        world_centers = poses[:, self.body_ids, :3] + torch.einsum(
            "nkij,kj->nki", box_rotation, self.centers
        )
        world_axes = box_rotation @ self.axes
        inverse_object = object_rotation.transpose(-1, -2)
        centers = torch.einsum("nij,nkj->nki", inverse_object, world_centers - obj[:, None, :3])
        axes = inverse_object[:, None] @ world_axes
        radius = torch.einsum("nkij,kj->nki", axes.abs(), self.extents)
        lower, upper = centers - radius, centers + radius
        tool_depth = torch.zeros(count, device=self.device, dtype=self.dtype)
        # One broadphase/compaction for every hull avoids a device synchronization
        # for each empty hull in small verification scenes. Construct one Boolean
        # N x boxes x hulls mask coordinate by coordinate, rather than allocating
        # a three-coordinate floating-point tensor of that size.
        overlapping = (upper[..., 0, None] > self.hull_lower[:, 0]) & (
            lower[..., 0, None] < self.hull_upper[:, 0]
        )
        for coordinate in (1, 2):
            overlapping &= (upper[..., coordinate, None] > self.hull_lower[:, coordinate]) & (
                lower[..., coordinate, None] < self.hull_upper[:, coordinate]
            )
        # nonzero orders indices lexicographically: hull first makes every hull's
        # rows contiguous without a GPU sort or another per-hull nonzero call.
        _, rows, boxes = overlapping.permute(2, 0, 1).nonzero(as_tuple=True)
        hull_counts = overlapping.sum(dim=(0, 1)).tolist()
        offset = 0
        for hull_id, pairs_in_hull in enumerate(hull_counts):
            if not pairs_in_hull:
                continue
            vertices, normals, edges = self.hulls[hull_id]
            stop = offset + pairs_in_hull
            for begin in range(offset, stop, self.pair_chunk_size):
                end = min(begin + self.pair_chunk_size, stop)
                r, b = (
                    rows[begin:end],
                    boxes[begin:end],
                )
                depth = box_hull_penetration(
                    centers[r, b], axes[r, b], self.extents[b], vertices, normals, edges
                )
                tool_depth.scatter_reduce_(0, r, depth, reduce="amax", include_self=True)
            offset = stop
        self_depth = torch.zeros_like(tool_depth)
        # Chunk the pair bank before expanding it across environments, so a
        # large scene never allocates N × all-box-pairs × all-SAT-axes.
        # AABBs need only three coordinates per pair. They can use the full
        # configured bank chunk; expensive 15-axis SAT runs only on compacted
        # overlapping pairs and remains limited to pair_chunk_size.
        pair_bank_chunk = self.pair_chunk_size
        for start in range(0, len(self.self_pairs), pair_bank_chunk):
            pair = self.self_pairs[start : start + pair_bank_chunk]
            first, second = pair.unbind(-1)
            overlap = (
                torch.minimum(upper[:, first], upper[:, second])
                - torch.maximum(lower[:, first], lower[:, second])
            ).amin(-1) > 0
            rows, columns = overlap.nonzero(as_tuple=True)
            for begin in range(0, len(rows), self.pair_chunk_size):
                r, c = (
                    rows[begin : begin + self.pair_chunk_size],
                    columns[begin : begin + self.pair_chunk_size],
                )
                a, b = first[c], second[c]
                depth = obb_pair_penetration(
                    centers[r, a],
                    axes[r, a],
                    self.extents[a],
                    centers[r, b],
                    axes[r, b],
                    self.extents[b],
                )
                self_depth.scatter_reduce_(0, r, depth, reduce="amax", include_self=True)
        tool_depth.masked_fill_(~valid, torch.inf)
        self_depth.masked_fill_(~valid, torch.inf)
        return {
            "tool_hand_penetration_m": tool_depth,
            "nonadjacent_self_penetration_m": self_depth,
            "feasible": valid
            & (tool_depth <= self.max_tool_penetration_m)
            & (self_depth <= self.max_self_penetration_m),
        }


def validate_native_shape_settings(model: CollisionModel, hand, tool) -> dict:
    """Check that exported geometry represents all actual native shapes.

    Negative rest offsets intentionally allow shape overlap and need explicit
    shape-pair budgets; they are not supported by the zero-rest model used here.
    """
    if hand.root_physx_view.max_shapes != len(model.hand.boxes):
        raise RuntimeError("Authored hand box count differs from native PhysX shape count")
    if tool.root_physx_view.max_shapes != len(model.tool.hulls):
        raise RuntimeError("Cooked tool hull count differs from native PhysX shape count")
    result = {}
    for name, asset in (("hand", hand), ("tool", tool)):
        rest = asset.root_physx_view.get_rest_offsets()
        contact = asset.root_physx_view.get_contact_offsets()
        if not torch.isfinite(rest).all() or (rest < 0).any():
            raise ValueError(
                "Negative/nonfinite native rest offsets require explicit shape-pair allowances"
            )
        result[name] = {
            "rest_offset_min_m": float(rest.min()),
            "rest_offset_max_m": float(rest.max()),
            "contact_offset_min_m": float(contact.min()),
            "contact_offset_max_m": float(contact.max()),
        }
    return result


# Short public aliases retained for environment construction.
CollisionGeometry = HandCollisionGeometry
ToolConvexGeometry = ToolCollisionGeometry
read_native_poses = read_native_link_poses
