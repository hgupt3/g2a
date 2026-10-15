# Grasp optimization and policy training

Every command accepts a bundled task name (`hammer`, `knife`, `saw`, `stir`, `scoop`) or the path to a custom task JSON. Run commands from the source checkout after installation. The simulator is headless by default; `--gui` opens its window.

## Configuration

[examples/train.json](../examples/train.json) is a starting profile for wrist and finger residual control with explicit tracking observations. [hammer.json](../examples/hammer.json), [knife.json](../examples/knife.json) and [scoop.json](../examples/scoop.json) use higher closing-torque thresholds. [experiment.json](../examples/experiment.json) lists all configuration fields and their defaults. A config can contain only the fields you want to override.

For a pretrained model, use `pretrained/<task>/experiment.json` together with that directory's grasp and policy. Changing the object, trajectory, hand, physical settings or saved grasp requires matching newly generated artifacts. Observation and action changes require a matching policy.

`--device cuda:N` selects a GPU from the visible CUDA devices. For example, `CUDA_VISIBLE_DEVICES=4 python -m g2a train ... --device cuda:0` selects physical GPU 4. Use separate output directories for separate runs.

## Optimize a grasp

```bash
python -m g2a optimize --task scoop --config examples/scoop.json \
  --device cuda:0 --num-envs 4096 --iterations 8 \
  --output runs/scoop/grasps.pt
```

Search uses independent distributions for each grasp-area box. It samples wrist translation and orientation, finger spread, thumb opposition, closing speeds and torque thresholds. Candidates close while the object is temporarily supported, then settle freely. Independent force/torque trials measure stability, and a shortlist follows the complete trajectory with task loads. The output stores the accepted grasp poses, measured joint positions and motor targets.

`search.iterations` and `search.max_wall_time_s` bound proposal search; `evaluation.verification_max_wall_time_s` separately bounds complete-trajectory verification. A search with no accepted complete grasp reports failure rather than creating a trainable result. Increase the search/verification budget or improve the area and closure settings before training.

Refine a compatible search with `--warm-start runs/scoop/grasps.pt`. If you deliberately change physical or closure settings, `--warm-proposals` reuses compatible search coordinates and reconstructs every grasp under the new settings. Both options still perform closure and stability tests.

## Train PPO

```bash
python -m g2a train --task scoop --config examples/scoop.json \
  --grasp runs/scoop/grasps.pt --device cuda:0 --num-envs 4096 \
  --output runs/scoop/ppo
```

PPO learns residual commands around the optimized motor targets and nominal wrist trajectory. The starting profile uses 4096 environments, horizon 32, adaptive learning rate and random trajectory starts. Its epoch cap, total runtime, validation interval and final-validation reserve are explicit fields under `training`. Periodic validation replays complete trajectories from phase zero and compares the policy with the optimized-grasp controller.

The main output files are:

- `policy.pt` and `policy.json`: selected deterministic controller and its input/action settings.
- `learned_policy.pt` and its JSON: the best available learned candidate.
- `checkpoint.pth`, `contract.json` and `agent.json`: full training state and settings for resume.
- `latest_checkpoint.pth`: recent recovery state, also saved before expensive validation.
- `experiment.json`, `training.json` and `validation.json`: resolved configuration and run summaries.

If no learned controller meets the configured selection criteria, `policy.json` identifies `policy.pt` as an `optimized_grasp_fallback`, which returns zero residuals. Check `controller_kind` when interpreting a run.

Resume a full checkpoint into a new output directory:

```bash
python -m g2a train --task scoop --device cuda:0 --num-envs 4096 \
  --config runs/scoop/ppo/experiment.json --grasp runs/scoop/grasps.pt \
  --checkpoint runs/scoop/ppo/checkpoint.pth --iterations 140 \
  --output runs/scoop/continued
```

For resume, `--iterations` bounds additional epochs. Keep `contract.json` and `agent.json` beside the checkpoint. The optimizer and normalization state are restored; the live simulator state and random generators start a new run.

## Evaluate

```bash
python -m g2a evaluate --task scoop --device cuda:0 \
  --config runs/scoop/ppo/experiment.json --grasp runs/scoop/grasps.pt \
  --checkpoint runs/scoop/ppo/policy.pt --baseline --disturbances \
  --num-envs 64 --batches 2 --parallel-batches --seed 2100 \
  --output runs/scoop/evaluation.json
```

`--baseline` adds zero-residual grasp control. With a checkpoint and baseline, `--num-envs` is the number of episodes per controller per seed batch. `--batches 2 --seed 2100` uses seeds 2100 and 2101. `--parallel-batches` evaluates both batches in one scene, so this example allocates 256 physical environments. Without it, batches run sequentially. Disturbance ranges come from `randomization`; omit `--disturbances` for nominal parameters.

For a pretrained directory, `--model pretrained/scoop` supplies the task, config, grasp, policy and grasp index; explicit file options override those defaults. To evaluate the grasp alone, use the explicit command above with `--baseline` and omit `--checkpoint` and `--parallel-batches`.

Every environment contributes its first episode, including early failures. `position_rmse_m` is the mean of episode position RMSEs; `position_mean_m` is the mean Euclidean distance. `angle_mean_rad`, success, failure rate and completion are reported separately. Completion means reaching the final trajectory reference; success additionally applies the configured tracking thresholds. Reports also retain per-episode penetration depths and configured collision-stop outcomes.

Use fresh evaluation seeds when assessing a selected controller. Training selection checks and evaluation measurements serve different purposes. Runtime limits are controlled by the saved config or `--max-wall-time`; they exclude initial simulator startup. Evaluation stops at its finite limit and reports incomplete work when it cannot finish.
