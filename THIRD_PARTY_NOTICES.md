# Third-party notices

The project-wide license is undecided. This document preserves upstream attribution and third-party terms; it does not assign a license to G2A's new project code.

The original G2A simulator scripts carried the Isaac Lab developers' BSD-3-Clause copyright notice. That notice is retained in `licenses/IsaacLab-BSD-3-Clause.txt`. Isaac Lab and Isaac Sim are installed separately and retain their upstream licenses.

The bundled LEAP-hand asset was flattened from the original repository's modified LEAP asset. Its MIT notice, including the dex_urdf modification attribution, is preserved at `g2a/resources/assets/leap_hand/LICENSE.txt` and included in the wheel.

The four object scans and recorded pose matrices come from the original G2A repository. Flattening USD layers, repairing relative resource references and packing pose matrices preserve that provenance.

RL-Games is installed from the pinned NVIDIA-maintained fork and retains its upstream MIT license. It is a training dependency; exported TorchScript policies do not import RL-Games or Isaac Sim.
