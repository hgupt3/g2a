"""Portable, validated task manifests. All asset paths are relative to the manifest."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

RESOURCE_ROOT = Path(__file__).with_name("resources")
RELEASE_TASKS = ("hammer", "knife", "saw", "stir", "scoop")


@dataclass(frozen=True)
class ContactObject:
    name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: float = 1.0
    description: str = "Defined simulation contact proxy."

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name):
            raise ValueError(
                "Contact object name must contain letters, digits, underscores or hyphens"
            )
        for name in ("size", "position", "euler"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"Contact {name} must be three finite values")
        if any(value <= 0 for value in self.size):
            raise ValueError("Contact object size must be positive")
        if not math.isfinite(self.friction) or self.friction < 0:
            raise ValueError("Contact object friction must be finite and nonnegative")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Contact object must describe its calibration or definition")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    object_asset: Path
    trajectory: Path
    mass: float
    body_offset: tuple[float, float, float]
    area_path: Path | None = None
    force_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_objects: tuple[ContactObject, ...] = ()
    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.7)
    initial_euler: tuple[float, float, float] = (0.0, math.pi / 2, 0.0)
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_frame: str = "object"
    resistance: float = 0.0
    max_resistance: float = 3.0
    force_interval: tuple[float, float] = (0.0, 1000.0)
    source_fps: float = 30.0
    smoothing_window: int = 5

    @classmethod
    def load(cls, task: str | Path, root: Path = RESOURCE_ROOT) -> "TaskSpec":
        manifest = Path(task).expanduser()
        if manifest.suffix != ".json":
            manifest = root / "tasks" / f"{task}.json"
        manifest = manifest.resolve()
        with manifest.open() as stream:
            data = json.load(stream)
        for key in ("object_asset", "trajectory"):
            data[key] = (manifest.parent / data[key]).resolve()
            if not data[key].is_file():
                raise FileNotFoundError(f"{key} does not exist: {data[key]}")
        if data.get("area_path"):
            # Discovery creates this file; only optimization requires that it already exists.
            data["area_path"] = (manifest.parent / data["area_path"]).resolve()
        data["contact_objects"] = tuple(
            ContactObject(
                **{
                    key: tuple(value) if key in ("size", "position", "euler") else value
                    for key, value in item.items()
                }
            )
            for item in data.get("contact_objects", ())
        )
        for key in (
            "body_offset",
            "force_point",
            "initial_position",
            "initial_euler",
            "force",
            "force_interval",
        ):
            if key in data:
                data[key] = tuple(data[key])
        spec = cls(**data)
        spec.validate()
        return spec

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name):
            raise ValueError("Task name must contain only letters, digits, underscores or hyphens")
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if isinstance(value, tuple) and name != "contact_objects":
                expected = 2 if name == "force_interval" else 3
                if len(value) != expected or not all(math.isfinite(v) for v in value):
                    raise ValueError(f"{name} must contain {expected} finite numbers")
        if self.mass <= 0 or self.source_fps <= 0:
            raise ValueError("mass and source_fps must be positive")
        if self.resistance < 0 or self.max_resistance < 0:
            raise ValueError("resistance settings cannot be negative")
        if self.force_frame not in ("object", "world"):
            raise ValueError("force_frame must be object or world")
        if self.force_interval[0] < 0 or self.force_interval[1] < self.force_interval[0]:
            raise ValueError("force_interval must be an increasing nonnegative interval")
        if (
            not isinstance(self.smoothing_window, int)
            or self.smoothing_window < 1
            or self.smoothing_window % 2 != 1
        ):
            raise ValueError("smoothing_window must be a positive odd integer")
        for item in self.contact_objects:
            item.validate()
        names = [item.name for item in self.contact_objects]
        if len(names) != len(set(names)):
            raise ValueError("Contact object names must be unique")

    def fingerprint(self) -> str:
        """Hash numerical settings and input bytes, independent of checkout location."""
        config = asdict(self)
        config.pop("object_asset")
        config.pop("trajectory")
        config.pop("area_path")
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        for path in (self.object_asset, self.trajectory):
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        if self.area_path is not None:
            digest.update(self.area_path.read_bytes())
        return digest.hexdigest()
