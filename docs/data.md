# Objects, trajectories and grasp areas

A task JSON connects an object asset, recorded poses, grasp-area boxes and optional task loads. All paths in the manifest are relative to that JSON. Bundled manifests, meshes, trajectories and areas live under `g2a/resources/`; the task names are `hammer`, `knife`, `saw`, `stir` and `scoop`. Stir and scoop share a spoon asset. The original recording name `scoup` is exposed as `scoop`.

## Object-pose recordings

Provide at least two homogeneous transforms with shape `(T, 4, 4)`. Each transform maps the tracked object mesh frame into one consistent world or camera frame. Translation is in meters; rotations are right-handed and orthonormal; the bottom row is `[0, 0, 0, 1]`.

Inputs can be a NumPy `.npy`, an `.npz` with a `transforms` array, or a dedicated directory of contiguous numbered text files such as `000000.txt`, `000001.txt`, and so on. Pack a recording with:

```bash
python -m g2a import-trajectory my_task/raw_poses my_task/trajectory.npz --fps 30
```

Set the task's `source_fps` to the actual recording rate. Packed recordings store that rate and a conflicting manifest is rejected. The loader rebases the first pose to the manifest's `initial_position` and `initial_euler`, applies the configured smoothing window, and resamples positions and rotations to the control clock while retaining the endpoints. Use `smoothing_window: 1` to disable smoothing.

## Object assets

Use a meter-scale USD with one dynamic rigid-body root and suitable collision geometry. Convert an OBJ, STL or FBX through:

```bash
python -m g2a convert-mesh my_task/object.obj my_task/object.usd \
  --mass 0.2 --scale 1.0 --device cuda:0
```

`--scale` converts the input mesh units to meters; use `0.001` for millimeters. The converter creates rigid-body, mass and collision settings and copies textures beside the USD. Keep its generated texture directory with the asset. Mesh conversion starts Isaac Sim.

The tracked mesh and USD must use the same orientation axes. `body_offset` is the local translation from the tracked mesh origin to the USD rigid-body origin, expressed in mesh coordinates:

```text
p_body = p_mesh + R_object * body_offset
```

Use `[0, 0, 0]` if the origins already match. Fix differing orientation axes in the mesh or recording before importing.

## Task manifest

Copy [examples/task_template.json](../examples/task_template.json) into your task directory and edit it. Required fields are `name`, `object_asset`, `trajectory`, `mass` and `body_offset`. Set `area_path` to the grasp-area JSON you will discover or define.

`initial_position` is in meters and `initial_euler` uses XYZ Euler angles in radians. These choose the recorded trajectory's placement in the simulation; they do not alter the recorded relative motion.

Optional task forces use `force` in newtons and `force_frame` set to `object` or `world`. `force_point` is the application point in object-body coordinates. `resistance` opposes that point's motion, including rotational motion, and `max_resistance` caps the force. `force_interval` uses seconds from the beginning of trajectory motion, excluding grasp acquisition. The simulator applies the resulting moment about the object's center of mass.

`contact_objects` optionally defines fixed cuboids by `name`, `size`, `position`, `euler`, `friction` and `description`. Positions and orientations use the simulation world frame. These are rigid interaction proxies; they do not model material cutting or fluid motion.

## Grasp-area boxes

An area defines volumes of candidate wrist poses, rather than individual grasp keypoints. The bundled tasks include calibrated area JSON files. Hammer and knife define functional handle regions and signed hand orientations; saw uses geometry-discovered boxes; stir and scoop use defined boxes. Original RGB-D recordings and human-hand estimates are not included or required.

For a new object, first create its manifest, then discover an area:

```bash
python -m g2a discover-area --task my_task/task.json \
  --output my_task/area.json --device cuda:0
```

Discovery samples the object surface and finds handle-like regions using object and LEAP-hand geometry. `--settings settings.json` overrides `DiscoveryConfig` fields such as handle thickness/width, approach clearance, translation/rotation ranges, sampling budget and seed. A mesh may need a directly defined area if its useful grip cannot be identified geometrically.

Area JSONs use `frame: "object_body"` and `perturbation_frame: "wrist_local"`. Each patch contains:

- `pose_wxyz`: center `[x, y, z, qw, qx, qy, qz]` in the object-body frame.
- `translation_bounds_m`: lower and upper XYZ translation bounds in the center wrist's axes.
- `rotation_bounds_rad`: lower and upper rotation-vector bounds in the same axes.
- `name`: an optional patch label.

Without a functional frame, the sampled wrist pose is `T_object_wrist = T_center * T_perturb`. Bounds must contain zero and the center quaternion must have unit length. Multiple patches represent distinct approaches; search maintains a distribution for each one.

For a tool-use grasp, `calibration.functional_grip` specifies the working-end direction, working-plane normal, usable handle segment, and calibrated hand axes and grip centers. The hammer and knife definitions point the hand's thumb side toward the head or blade and constrain the palm to the selected side of the tool. `palm_side: 1` aligns the palm normal with the working-plane normal; `palm_side: -1` aligns it with the opposite normal. Omitting `palm_side` allows either side. Set these directions explicitly: the sign of a mesh principal axis does not identify the working end or the intended palm side.

Rotation perturbations pivot about the grip center; they do not sweep it around the wrist root. Search refines placement and finger closure within the declared bounds and rejects grasps outside the required orientation or handle region. Inspect a palm-facing render before training a controller. Changing the area or grasp requires new matching policy artifacts; rotating a saved grasp alone does not preserve its contact, collision or trajectory verification.

An optional `functional_grip.opposition` requirement identifies distal thumb/finger body groups and declares `minimum_contact_force_n`, `minimum_contact_fraction` and `minimum_finger_groups`. Search measures tool-filtered contact forces during free settling, independently of the wrench tests. The Hammer definition requires sustained thumb contact together with at least two finger groups; palm or MCP contact alone does not satisfy it. Use the bundled JSON as a schema example when defining a functional grip for a new tool.

`handle_segment_m` bounds proposal-center placement, and `grip_centers_root_m` records the corresponding hand-frame pivots. For opposed grasps, `contact_segment_m` defines the physical handle interval: every force-qualified thumb/finger group's measured contact centroid must lie within it. Keep these physical contact bounds separate from any narrower interval used to place the hand.

For an opposed grip, optional `functional_grip.contact_center_bounds_root_m` contains lower and upper XYZ bounds in the actual hand-root frame. It constrains the midpoint between the measured thumb-contact centroid and the mean of the valid finger-group centroids. This constrains where the opposed contacts form after closure. The bounds must contain every proposal grip center and require valid native opposition contacts. Hammer uses this constraint to keep the handle toward the finger pads instead of allowing it to settle low across the palm.

Optional `functional_grip.handle_center_bounds_root_m` also contains lower and upper XYZ bounds in the actual hand-root frame. It constrains a fixed object landmark: the midpoint of `contact_segment_m`, or of `handle_segment_m` when no contact segment is supplied. Use it to control the handle's placement across the hand. Unlike contact centroids, this landmark stays fixed in the object when contact forces change.

Opposed grasps use actual native contact points to check final handle placement. `resources.tool_contact_points_per_filter` sets reporting storage per hand-body filter (default `32`); increase this positive integer for contact-rich compound meshes if contact geometry is reported incomplete. This changes reporting memory, while saturated reports remain invalid and physical acceptance limits stay unchanged.

To define boxes directly, construct `AreaPatch` and `GraspArea` from `g2a.regions`. Save the actual object asset hash, `calibration.body_offset_m`, hand asset hash, and `source` with `kind: "defined"` plus a description. `GraspArea.save()` validates the JSON; `GraspArea.load()` checks it against the task and hand. Use a center pose calibrated for your object and hand. Geometry discovery is a convenient starting file to inspect and edit.

## Run a custom task

After preparing the asset, recording, manifest and area:

```bash
python -m g2a validate --task my_task/task.json --config examples/train.json

python -m g2a optimize --task my_task/task.json --device cuda:0 \
  --config examples/train.json --num-envs 4096 \
  --output my_task/grasps.pt

python -m g2a train --task my_task/task.json --device cuda:0 \
  --config examples/train.json --grasp my_task/grasps.pt \
  --num-envs 4096 --output my_task/ppo

python -m g2a visualize --task my_task/task.json --device cuda:0 \
  --config my_task/ppo/experiment.json --grasp my_task/grasps.pt \
  --checkpoint my_task/ppo/policy.pt --mode trajectory \
  --output my_task/visualization
```

`validate` checks task files, trajectory transforms, area compatibility and configuration without starting Isaac Sim. Inspect the collision geometry and acquired grasp before committing to training. Changing the mesh, area calibration or recording requires regenerating matching grasps and policies. The current hand model is LEAP; a different hand requires an explicit joint/action mapping and grasp model.
