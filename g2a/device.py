"""Select one physical GPU before CUDA-capable libraries are imported.

CUDA indices are local to CUDA_VISIBLE_DEVICES, whereas Isaac Sim 5.1's
renderer/activeGpu uses the host Vulkan enumeration. Keep the two assignments
separate. Inventory queries use nvidia-smi/NVML and do not create CUDA contexts.
"""

import csv
import io
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import ClassVar


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    pci_bus_id: str


@dataclass(frozen=True)
class GpuAssignment:
    requested_device: str
    physical_index: int
    uuid: str
    pci_bus_id: str
    torch_device: str
    renderer_index: int
    inherited_cuda_visible_devices: str | None
    _current: ClassVar["GpuAssignment | None"] = None

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "cuda_visible_devices": self.uuid,
            "cuda_device_order": "PCI_BUS_ID",
            "physics_cuda_index": 0,
            "renderer_index_source": "host_nvidia_smi_index_native_audit_required",
        }


def gpu_inventory() -> tuple[GpuInfo, ...]:
    """Query stable physical identifiers without importing a CUDA runtime."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot determine the physical GPU assignment with nvidia-smi") from error
    rows = []
    try:
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != 3:
                raise ValueError("Expected index, UUID and PCI bus ID")
            index, uuid, bus_id = (value.strip() for value in row)
            if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid) or not bus_id:
                raise ValueError("Invalid GPU UUID or PCI bus ID")
            rows.append(GpuInfo(int(index), uuid, bus_id))
    except (ValueError, TypeError) as error:
        raise ValueError("nvidia-smi returned an invalid physical GPU inventory") from error
    if (
        not rows
        or any(row.index < 0 for row in rows)
        or len({row.index for row in rows}) != len(rows)
        or len({row.uuid for row in rows}) != len(rows)
    ):
        raise ValueError("nvidia-smi returned an empty or ambiguous physical GPU inventory")
    return tuple(sorted(rows, key=lambda row: row.index))


def _resolve_visible_token(token: str, inventory: tuple[GpuInfo, ...]) -> GpuInfo:
    if token.isdecimal():
        matches = [gpu for gpu in inventory if gpu.index == int(token)]
    elif token.startswith("GPU-"):
        matches = [gpu for gpu in inventory if gpu.uuid.startswith(token)]
    else:
        matches = []
    if len(matches) != 1:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES entry does not identify one physical GPU: {token!r}"
        )
    return matches[0]


def environment_for_physical_gpu(
    physical_gpu: int,
    environment: Mapping[str, str],
    inventory: tuple[GpuInfo, ...] | None = None,
) -> tuple[dict[str, str], GpuAssignment]:
    """Build a child UUID mask for a host index without mutating global state.

    A physical index stays physical even when the parent has an inherited
    visible list. That list is an allowance to narrow, never an index remapping
    for this API. This helper is safe to use from independent runner threads.
    """
    if isinstance(physical_gpu, bool) or not isinstance(physical_gpu, int) or physical_gpu < 0:
        raise ValueError("A physical GPU index must be a nonnegative integer")
    inventory = gpu_inventory() if inventory is None else inventory
    selected = [gpu for gpu in inventory if gpu.index == physical_gpu]
    if len(selected) != 1:
        raise ValueError(f"Host physical GPU {physical_gpu} is absent or ambiguous")
    selected = selected[0]
    inherited = environment.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        tokens = [token.strip() for token in inherited.split(",")]
        if not inherited.strip() or any(not token or token == "-1" for token in tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES does not permit a GPU; refusing to expand it")
        allowed = [_resolve_visible_token(token, inventory) for token in tokens]
        if len({gpu.uuid for gpu in allowed}) != len(allowed):
            raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate physical GPUs")
        if selected.uuid not in {gpu.uuid for gpu in allowed}:
            raise ValueError(
                f"Physical GPU {physical_gpu} is outside the inherited CUDA_VISIBLE_DEVICES allowance"
            )
    assignment = GpuAssignment(
        requested_device=f"physical:{physical_gpu}",
        physical_index=physical_gpu,
        uuid=selected.uuid,
        pci_bus_id=selected.pci_bus_id,
        torch_device="cuda:0",
        renderer_index=physical_gpu,
        inherited_cuda_visible_devices=inherited,
    )
    child_environment = dict(environment)
    child_environment["CUDA_VISIBLE_DEVICES"] = selected.uuid
    child_environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return child_environment, assignment


def isolate_gpu(device: str) -> GpuAssignment:
    """Narrow the inherited GPU allowance to one UUID before Torch or Isaac Sim.

    With no inherited mask, cuda:N selects host nvidia-smi GPU N. With a mask,
    N indexes that visible list; a cold child therefore reuses cuda:0 while
    inheriting its parent's single physical UUID. An empty or restrictive mask
    is never expanded. CPU physics still needs an explicitly isolated renderer
    and selects the first GPU permitted by the inherited mask/host inventory.
    """
    if device != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("Device must be cpu, cuda, or cuda:N with a nonnegative index")
    current = GpuAssignment._current
    if current is not None:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != current.uuid:
            raise RuntimeError("GPU visibility changed after initialization; start a fresh process")
        if device not in (current.requested_device, current.torch_device):
            raise RuntimeError("A process may select only one physical GPU; start a fresh process")
        return current
    loaded = [
        name
        for name in ("torch", "isaaclab", "isaacsim", "omni", "warp", "cupy", "jax", "pycuda")
        if name in sys.modules
    ]
    if loaded:
        raise RuntimeError(
            "GPU isolation must run before CUDA-capable imports; already imported: "
            + ", ".join(loaded)
            + ". Start a fresh process."
        )
    inventory = gpu_inventory()
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    ordinal = int(device.split(":", 1)[1]) if ":" in device else 0
    if inherited is None:
        matches = [gpu for gpu in inventory if gpu.index == ordinal]
        if not matches:
            raise ValueError(f"Host physical GPU {ordinal} is absent from nvidia-smi")
        selected = matches[0]
    else:
        tokens = [token.strip() for token in inherited.split(",")]
        if not inherited.strip() or any(not token or token == "-1" for token in tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES does not permit a GPU; refusing to expand it")
        visible = [_resolve_visible_token(token, inventory) for token in tokens]
        if len({gpu.uuid for gpu in visible}) != len(visible):
            raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate physical GPUs")
        if ordinal >= len(visible):
            raise ValueError(
                f"Device ordinal {ordinal} is outside the inherited CUDA_VISIBLE_DEVICES list "
                f"of {len(visible)} GPUs"
            )
        selected = visible[ordinal]
    assignment = GpuAssignment(
        requested_device=device,
        physical_index=selected.index,
        uuid=selected.uuid,
        pci_bus_id=selected.pci_bus_id,
        torch_device="cpu" if device == "cpu" else "cuda:0",
        renderer_index=selected.index,
        inherited_cuda_visible_devices=inherited,
    )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = selected.uuid
    GpuAssignment._current = assignment
    return assignment


def launch_isolated_app(assignment: GpuAssignment, *, launcher_class=None, **kwargs):
    """Adapt the pinned AppLauncher without modifying shared SDK files.

    Isaac Lab overwrites active_gpu with its CUDA index. Override only that
    resolution step so Vulkan retains its host index and CUDA remains local 0.
    Verify physical placement with the native process allocation audit before
    starting long jobs; host Vulkan and NVML enumeration can differ on a host.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != assignment.uuid:
        raise RuntimeError(
            "Single-GPU UUID isolation is missing or changed before simulator launch"
        )
    if launcher_class is None:
        from isaaclab.app import AppLauncher

        launcher_class = AppLauncher
    if "device" in kwargs:
        raise TypeError("The isolated assignment supplies the logical simulation device")
    if kwargs.get("multi_gpu", False) is not False:
        raise ValueError("Isolated launches cannot enable multiple GPUs")
    if kwargs.get("max_gpu_count", 1) != 1:
        raise ValueError("Isolated launches must limit rendering to one GPU")

    class IsolatedAppLauncher(launcher_class):
        def _resolve_device_settings(self, launcher_args):
            super()._resolve_device_settings(launcher_args)
            launcher_args["physics_gpu"] = 0
            launcher_args["active_gpu"] = assignment.renderer_index

    return IsolatedAppLauncher(
        device=assignment.torch_device,
        **{**kwargs, "multi_gpu": False, "max_gpu_count": 1},
    )
