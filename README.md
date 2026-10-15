# Grasp-to-Act (G2A)

[Grasp to Act: Dexterous Grasping for Tool Use in Dynamic Settings](https://arxiv.org/abs/2602.20466)

Harsh Gupta, Mohammad Amin Mirzaee, Wenzhen Yuan

[Project page](https://grasp2act.github.io/)

G2A optimizes LEAP-hand grasps and trains residual PPO policies to carry objects through recorded pose trajectories in Isaac Lab. It supports the bundled **hammer, knife, saw, stir and scoop** tasks, and custom object meshes and trajectories.

Grasp search samples wrist poses inside grasp-area boxes, optimizes finger closure, tests stability under external wrenches, and checks the full trajectory. PPO then learns bounded wrist and finger corrections around the saved grasp.

## Install

The supported Linux GPU environment uses Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2 and PyTorch 2.7.0 with CUDA 12.8. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this checkout:

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m g2a validate
```

If that simulator stack is already installed in your Python environment:

```bash
python -m pip install -e '.[rl,visual]'
python -m g2a validate
```

The task manifests, trajectories, object assets and grasp areas are included with the package. Pretrained grasp and policy files are organized in [pretrained/](pretrained/README.md).

## Evaluate a pretrained model

Use the model's saved configuration and grasp together:

```bash
python -m g2a evaluate --model pretrained/scoop --device cuda:0 \
  --baseline --disturbances --num-envs 64 --batches 2 --parallel-batches \
  --seed 2100 --output runs/scoop/evaluation.json
```

This loads the matching task, config, grasp and policy from the model directory and compares the policy with zero residuals on the same grasp. The JSON includes position and orientation errors, completion and success rates, and per-episode outcomes. Omit `--disturbances` for nominal physical parameters. [The training guide](docs/training.md#evaluate) includes grasp-only evaluation and explicit file paths.

## Optimize and train

```bash
python -m g2a optimize --task scoop --device cuda:0 --num-envs 4096 \
  --config examples/scoop.json --iterations 8 \
  --output runs/scoop/grasps.pt

python -m g2a train --task scoop --device cuda:0 --num-envs 4096 \
  --config examples/scoop.json --grasp runs/scoop/grasps.pt \
  --output runs/scoop/ppo
```

Use [examples/train.json](examples/train.json) for a general starting configuration, or the task-specific [hammer](examples/hammer.json), [knife](examples/knife.json) and [scoop](examples/scoop.json) profiles. Keep the same physical configuration across optimization, training and evaluation. All stages have finite iteration and time limits; see [training](docs/training.md) for budgets, checkpoints and evaluation options.

## Visualize

```bash
python -m g2a visualize --model pretrained/scoop --mode grasp \
  --device cuda:0 --output runs/scoop/grasp_view

python -m g2a visualize --model pretrained/scoop --mode trajectory \
  --device cuda:0 --output runs/scoop/trajectory_view
```

Grasp mode saves a front view facing the palm with the complete tool, plus contact and oblique views. Trajectory mode saves a fixed-camera video of the full motion and a following close-up, with arrows showing applied task forces and resistance at their application point. Add `--gui` to open the simulator window. Explicit grasp/config paths and zero-residual trajectory playback are also supported; see [visualization](docs/visualization.md).

## Use your own data

Supply an object USD, an object-pose trajectory and a task manifest, then discover or define grasp-area boxes. Pass the manifest path through `--task` to use the same optimization, training and evaluation commands. [The data guide](docs/data.md) covers formats, coordinate frames, mesh conversion and area discovery. [The controller guide](docs/controller.md) describes policy inputs, actions and portable inference.

Task forces and fixed contact objects are configurable simulation models; cutting, fluid dynamics and deformable materials are outside the simulator model.

## Project layout

```text
g2a/                   Simulation, grasp optimization, PPO and visualization
  resources/assets/    LEAP hand, object USDs and textures
  resources/data/      Recorded object-pose trajectories
  resources/areas/     Grasp-area boxes and orientation constraints
  resources/tasks/     Task manifests and physical loads
pretrained/<task>/     Matching grasp, inference policy and training checkpoint
examples/              Experiment profiles and a custom-task template
docs/                  Data, training, controller and visualization guides
scripts/setup.sh       Installation of the supported simulator stack
```

Generated grasps, training outputs, measurements and media are written to `runs/` by the examples. This directory is ignored by Git and excluded from distribution archives.

Third-party asset and dependency terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For development, install `pip install -e '.[dev,rl,visual]'`. Run `ruff check g2a scripts` and `ruff format --check g2a scripts` to check code style.
