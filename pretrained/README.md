# Pretrained grasps and policies

Each available model is stored in a task directory:

```text
pretrained/<task>/
  grasps.pt
  policy.pt
  policy.json
  experiment.json
  checkpoint.pth
  contract.json
  agent.json
```

Task names are `hammer`, `knife`, `saw`, `stir` and `scoop`. Keep each model's files together. `grasps.pt` stores the optimized physical grasp; `policy.pt` is the exported deterministic controller; `policy.json` records its input/action settings and selected grasp. `experiment.json` supplies the matching simulation configuration and finite defaults for new or resumed training. Object assets, trajectories and area definitions are included in the G2A package.

A policy is specific to its saved grasp, task and physical configuration. Reoptimizing the grasp or changing its orientation requires training a matching policy. The loaders check compatibility and reject mismatched files.

## Evaluate

```bash
python -m g2a evaluate --model pretrained/scoop --device cuda:0 \
  --baseline --disturbances \
  --num-envs 64 --batches 2 --parallel-batches --seed 2100 \
  --output runs/scoop/evaluation.json
```

`--model` loads the matching task, configuration, grasp, policy and grasp index. Remove `--disturbances` for nominal parameters. Results are written to the requested JSON path. For grasp-only evaluation, use explicit `--task`, `--config` and `--grasp` paths with `--baseline`, omitting `--model`, `--checkpoint` and `--parallel-batches`; see [evaluation options](../docs/training.md#evaluate).

## Visualize

```bash
python -m g2a visualize --model pretrained/scoop --mode trajectory \
  --device cuda:0 --output runs/scoop/visualization
```

Use `--mode grasp` for a grasp view or add `--gui` for the simulator window.

## Resume training

```bash
python -m g2a train --task scoop --device cuda:0 --num-envs 4096 \
  --config pretrained/scoop/experiment.json --grasp pretrained/scoop/grasps.pt \
  --checkpoint pretrained/scoop/checkpoint.pth --iterations 140 \
  --output runs/scoop/continued
```

`checkpoint.pth` restores PPO model, normalizer and optimizer state; keep its matching `contract.json` and `agent.json` beside it. Resume starts fresh simulator episodes rather than restoring a live simulation. Use `policy.pt` for inference, not training resume.

See [training](../docs/training.md), [visualization](../docs/visualization.md) and [the controller API](../docs/controller.md) for details.
