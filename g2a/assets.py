"""Prepare a mesh as a portable rigid-body USD for an additional task."""

import hashlib
import math
import os
import shutil
from pathlib import Path


def validate_conversion(source: Path, output: Path, mass: float, scale: float) -> None:
    if not source.is_file() or source.suffix.lower() not in (".obj", ".stl", ".fbx"):
        raise ValueError("Source must be an existing OBJ, STL, or FBX mesh")
    if output.suffix != ".usd":
        raise ValueError("Output must end in .usd")
    if not all(math.isfinite(value) and value > 0 for value in (mass, scale)):
        raise ValueError("Mass (kg) and scale (mesh units to meters) must be finite and positive")


def convert_mesh(source: Path, output: Path, mass: float, scale: float) -> None:
    """Call after AppLauncher. Keep physics geometry, textures and coordinate units portable."""
    from isaaclab.sim import (
        CollisionPropertiesCfg,
        ConvexDecompositionPropertiesCfg,
        MassPropertiesCfg,
        RigidBodyPropertiesCfg,
    )
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from pxr import Sdf, Usd, UsdGeom

    source, output = source.resolve(), output.resolve()
    validate_conversion(source, output, mass, scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=str(source),
            usd_dir=str(output.parent),
            usd_file_name=output.name,
            make_instanceable=False,
            force_usd_conversion=True,
            scale=(scale, scale, scale),
            mass_props=MassPropertiesCfg(mass=mass),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=ConvexDecompositionPropertiesCfg(),
        )
    )
    stage = Usd.Stage.Open(converter.usd_path)
    portable = Usd.Stage.Open(stage.Flatten())
    UsdGeom.SetStageMetersPerUnit(portable, 1.0)
    for prim in portable.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if (
                not isinstance(value, Sdf.AssetPath)
                or not value.path
                or value.path.endswith(".mdl")
            ):
                continue
            asset = Path(value.resolvedPath or value.path)
            if not asset.is_absolute():
                asset = output.parent / asset
            if not asset.is_file():
                raise FileNotFoundError(f"Unresolved mesh resource: {asset}")
            with asset.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            destination = output.parent / "textures" / f"{digest}{asset.suffix.lower()}"
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                with destination.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                        raise ValueError(
                            f"Existing mesh resource has unexpected contents: {destination}"
                        )
            else:
                shutil.copy2(asset, destination)
            attribute.Set(Sdf.AssetPath(f"textures/{destination.name}"))
    temporary = output.with_suffix(".tmp.usd")
    portable.GetRootLayer().Export(str(temporary))
    os.replace(temporary, output)
    print(f"Converted {source} to {output} (mass={mass:g} kg, scale={scale:g})", flush=True)
