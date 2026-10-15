# Residual controller and inference

The policy adapts a saved LEAP grasp while the wrist follows the recorded object trajectory. The saved `policy.json` specifies the action mode, observation fields and scales, control timestep, joint order and matching grasp. Keep it beside `policy.pt`.

## Actions

`finger_residual` has 16 actions in numerical joint-name order `a_0` through `a_15`. Each action commands a bounded absolute offset around the optimized motor target. A held action does not accumulate target drift. Zero action retains the optimized targets and nominal wrist path.

`wrist_finger_residual` has 22 actions: translation in `0:3`, rotation vector in `3:6`, and finger offsets in `6:22`. Wrist offsets use the nominal wrist's local axes. Translation and rotation limits are norm bounds; finger limits are radians per joint. Filtering, target-rate limits and asset joint limits apply at the saved control timestep. Read the actual limits from the policy metadata rather than assuming a profile's defaults.

## Observations

`g2a.control.observation()` constructs features in this order:

1. Joint positions, velocities and targets; previous filtered action; estimated actuator torque and torque utilization.
2. Object and palm poses/velocities.
3. Current and future object-relative goal poses, plus goal velocity.
4. Current and expected task wrenches, nominal mass/friction/load context, trajectory phase and setup flag.
5. Optional explicit COM tracking error, orientation correction and measured object-to-palm grip offset.

The base layout has 168 features for finger control or 174 for wrist/finger control. `observation.tracking_features=true` adds twelve features, giving 180 or 186 inputs respectively. A change of layout or scaling requires a matching trained policy.

Pose tensors use `[x, y, z, qw, qx, qy, qz]`, meters and unit scalar-first quaternions. Object poses refer to the USD rigid-body origin; palm poses refer to the hand root. All measurements and reference poses use one calibrated environment frame, without simulator grid translations. Velocities use `[vx, vy, vz, wx, wy, wz]` in m/s and rad/s. Rotation features are the first two rotation-matrix columns.

Joint vectors follow `contract["joint_names"]`; simulator articulation order can differ. Torque is an actuator estimate and its utilization is a load proxy, rather than tactile sensing. Physical randomization remains hidden: context fields use nominal task values.

Task wrenches use `[Fx, Fy, Fz, Tx, Ty, Tz]` in the environment frame, with moments about the object COM. They describe the configured external-load model, excluding gravity and unmeasured contact reactions. `g2a.stress.point_load_wrench()` implements the same model outside Isaac Sim.

Explicit tracking errors use the nominal goal wrist frame. The measured grip offset compares the actual object-to-palm pose with the saved grasp. `observation()` requires the nominal grasp and body-local COM when these features are enabled.

An observation describes the last completed control step. Decode the next action against the next reference pose. Retain the filtered action and previous motor targets between steps, and reset them when acquiring a new grasp. The model's `control_dt` determines the update rate.

## Model files

The exported `policy.pt` contains input normalization and returns deterministic, bounded action means. Evaluation and visualization load the matching `policy.json` and check its grasp, observation, action and physics settings automatically. Use the full `checkpoint.pth` with its `contract.json` and `agent.json` to resume PPO training.
