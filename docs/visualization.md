# Visualizing grasps and trajectories

`g2a visualize` runs the simulator with a saved grasp and optional policy. Use the configuration saved with the model so object, hand, timing and control settings match.

## Pretrained models

```bash
python -m g2a visualize --model pretrained/scoop --mode grasp \
  --device cuda:0 --output runs/scoop/grasp_view

python -m g2a visualize --model pretrained/scoop --mode trajectory \
  --device cuda:0 --output runs/scoop/trajectory_view
```

The trajectory command above uses the learned policy. To replay the same optimized grasp with zero RL corrections, run a separate grasp-only trajectory:

```bash
python -m g2a visualize --task scoop --mode trajectory --device cuda:0 \
  --config pretrained/scoop/experiment.json --grasp pretrained/scoop/grasps.pt \
  --output runs/scoop/grasp_only_trajectory
```

Both trajectory runs save the full-motion and close-up videos. Use the same `--seed` and `--randomize` settings when comparing them.

`--mode grasp` inspects the acquired grasp with zero residuals and saves `front.png` (palm facing the camera, complete hand and tool), `contacts.png` (a tighter hand view) and `oblique.png` (complete hand and tool from an angled view).

`--mode trajectory` replays the complete recording and saves `trajectory.mp4`, a following `closeup.mp4`, and `summary.json`. The trajectory camera stays fixed throughout playback and frames the full motion of the hand and tool, with its viewing direction chosen from the palm-facing side of the recorded motion. The close-up camera follows the palm and keeps the complete hand and tool in view. These are native simulation renders; task interactions use the forces and fixed contact objects defined in the task manifest. A reference-path trail is not drawn.

Trajectory playback shows an orange arrow at the applied task-load point. Its direction follows the actual world-frame force sent to the simulator, including resistance and any randomized load multiplier. Saved videos also show the magnitude and XYZ components in newtons; `forces.jsonl` records each captured frame's force and application point. Arrow length defaults to 2 cm/N and is capped at 25 cm for readability. Change the scale with `--force-scale` (metres per newton), or hide the overlay with `--hide-forces`. Zero loads have no arrow. Gravity and collision contact forces are not included in this overlay.

With `--output`, rendering is headless unless you add `--gui`. Without `--output`, the command opens a simulator window for a finite episode. Rendering requires the `visual` dependencies, installed by `scripts/setup.sh` or `pip install -e '.[visual]'`.

`--hold-seconds` controls how long a settled grasp stays visible in the GUI.

## Your own run

Pass the task, configuration and grasp explicitly:

```bash
python -m g2a visualize --task scoop --device cuda:0 \
  --config runs/scoop/ppo/experiment.json --grasp runs/scoop/grasps.pt \
  --checkpoint runs/scoop/ppo/policy.pt --mode trajectory \
  --output runs/scoop/policy_view
```

Omit `--checkpoint` to follow the trajectory with zero residuals around the optimized grasp. For a custom recording, replace `scoop` with its task-manifest path. `--checkpoint` accepts the exported inference `policy.pt`, rather than a full PPO training checkpoint.

Choose a new or empty output directory for each view. `--randomize` enables the configuration's physical randomization, and `--seed` selects the episode seed. `--width`, `--height` and `--fps` control saved media. The default cameras frame the complete hand and tool.

Visualizations stop after acquisition or the trajectory's terminal state. `--max-wall-time` bounds replay and rendering, with a separate `--startup-timeout` for simulator startup. `--max-steps` can shorten an inspection; its summary identifies incomplete playback. Use `python -m g2a visualize --help` for all options.

Inspect the wrist approach, finger/thumb opposition and retention through the complete motion. Rendered appearance complements numerical evaluation; use `g2a evaluate` to compute tracking and episode outcomes.
