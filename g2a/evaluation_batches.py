"""Counterbalanced independent seed batches in one physical evaluation scene.

Only evaluation opts into this layout. Each seed retains the same draw order as
the sequential evaluator: mass, load, then bucketed friction. Private generators
keep a failed row's later reset from changing another batch's initial physics.
"""

import hashlib
from dataclasses import dataclass

import torch

from .stress import sample_physics_multipliers

PARALLEL_MATCHING_METHOD = "simultaneous_seed_batches_neighbor_parity_mirrored_initial_physics_v2"


@dataclass(frozen=True)
class PairedBatchPlan:
    seeds: tuple[int, ...]
    per_controller: int

    def __post_init__(self):
        if (
            not isinstance(self.per_controller, int)
            or isinstance(self.per_controller, bool)
            or self.per_controller < 1
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32
                for seed in self.seeds
            )
        ):
            raise ValueError(
                "Paired batches require unique uint32 seeds and a positive controller size"
            )

    @property
    def physical_count(self):
        return 2 * self.per_controller * len(self.seeds)

    def groups(self):
        groups = []
        for index, seed in enumerate(self.seeds):
            begin = index * 2 * self.per_controller
            # Every adjacent pair gets one controller each. Binary digit parity
            # alternates their orientation across both grid axes; swapping the
            # next seed also counterbalances each local slot. Unlike contiguous
            # blocks, the 64-world/2-seed grid has identical per-controller
            # origin means, ranges, and second moments within every seed.
            policy = tuple(
                begin + 2 * pair + ((pair.bit_count() + index) % 2)
                for pair in range(self.per_controller)
            )
            baseline = tuple(env_id ^ 1 for env_id in policy)
            groups.append(
                {"batch_index": index, "seed": seed, "policy": policy, "baseline": baseline}
            )
        return groups

    def initial_physics(self, spec, buckets: torch.Tensor, device, *, enabled=True):
        result = {
            name: torch.ones(self.physical_count, device=device)
            for name in ("mass", "load", "friction")
        }
        if not enabled:
            return result
        for group in self.groups():
            generator = torch.Generator(device=device).manual_seed(group["seed"])
            indices = torch.tensor(group["policy"] + group["baseline"], device=device)
            for name, interval, bank in (
                ("mass", spec.mass_range, None),
                ("load", spec.load_range, None),
                ("friction", spec.friction_range, buckets),
            ):
                draws = sample_physics_multipliers(
                    2 * self.per_controller,
                    interval,
                    device,
                    paired=True,
                    buckets=bank,
                    generator=generator,
                )
                result[name][indices] = draws
        return result

    def controller(self, policy, action_dim, device):
        indices = torch.tensor(
            [index for group in self.groups() for index in group["policy"]], device=device
        )

        def act(observations):
            if len(observations) != self.physical_count:
                raise ValueError("Parallel policy observations must cover every physical world")
            actions = policy(observations[indices])
            if actions.shape != (len(indices), action_dim):
                raise ValueError("Policy actions do not match the parallel controller groups")
            result = actions.new_zeros((self.physical_count, action_dim))
            result[indices] = actions
            return result

        return act

    def split_metrics(self, metrics):
        if not metrics or any(
            not isinstance(value, torch.Tensor) or value.shape != (self.physical_count,)
            for value in metrics.values()
        ):
            raise ValueError(
                "Parallel episode metrics must include every physical world's first episode"
            )
        return [
            {
                label: {name: value[list(group[label])].clone() for name, value in metrics.items()}
                for label in ("policy", "baseline")
            }
            for group in self.groups()
        ]

    def origin_statistics(self, origins):
        """Record actual clone positions; arbitrary grid sizes may be unbalanced."""
        if (
            not isinstance(origins, torch.Tensor)
            or origins.shape != (self.physical_count, 3)
            or not torch.isfinite(origins).all()
        ):
            raise ValueError(
                "Origin proof requires finite native scene positions for every physical world"
            )
        positions = origins.detach().cpu().double().contiguous()

        def statistics(ids):
            selected = positions[list(ids)]
            return {
                "mean_m": selected.mean(0).tolist(),
                "minimum_m": selected.amin(0).tolist(),
                "maximum_m": selected.amax(0).tolist(),
                "second_moment_m2": (selected.T @ selected / len(selected)).tolist(),
            }

        groups = self.groups()
        aggregate = {
            label: statistics([env_id for group in groups for env_id in group[label]])
            for label in ("policy", "baseline")
        }
        batches = [
            {
                "batch_index": group["batch_index"],
                "seed": group["seed"],
                **{label: statistics(group[label]) for label in ("policy", "baseline")},
            }
            for group in groups
        ]
        return {
            "source": "native_scene_env_origins",
            "positions_m": positions.tolist(),
            "sha256": hashlib.sha256(positions.numpy().tobytes()).hexdigest(),
            "hash_dtype": "float64",
            "controller_statistics": {"aggregate": aggregate, "batches": batches},
            "exact_moments_and_ranges_balanced": (
                aggregate["policy"] == aggregate["baseline"]
                and all(batch["policy"] == batch["baseline"] for batch in batches)
            ),
            "limitations": (
                "Pairs occupy distinct physical origins. Matching origin moments does not establish "
                "identical floating-point contact dynamics or equivalence to sequential evaluation. "
                "Arbitrary scene sizes need not have exactly balanced origin moments or ranges."
            ),
        }

    def validate_native_proof(self, proof):
        """Validate persisted evaluation provenance without claiming fresh measurements."""
        required = {
            "object_mass",
            "object_inertia",
            "hand_mass",
            "hand_inertia",
            "object_material",
            "hand_material",
            "mass",
            "load",
            "friction",
        }
        message = "Parallel evaluation requires actual native initial-physics pairing proof and scene origins"
        if not (
            isinstance(proof, dict)
            and proof.get("verified") is True
            and proof.get("matching_method") == PARALLEL_MATCHING_METHOD
            and proof.get("physical_worlds") == self.physical_count
            and proof.get("matched_pairs") == self.per_controller * len(self.seeds)
            and isinstance(proof.get("field_sha256"), dict)
            and set(proof["field_sha256"]) >= required
            and all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in proof["field_sha256"].values()
            )
        ):
            raise ValueError(message)
        try:
            groups = [
                {**group, "policy": tuple(group["policy"]), "baseline": tuple(group["baseline"])}
                for group in proof["batch_groups"]
            ]
            origins = proof["scene_origins"]
            positions = torch.tensor(origins["positions_m"], dtype=torch.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(message) from error
        if groups != self.groups() or origins != self.origin_statistics(positions):
            raise ValueError(message)

    def verify_native_pairs(self, values, *, origins):
        """Check actual properties read back after the native per-row writes."""
        if not values or any(
            not isinstance(value, torch.Tensor)
            or value.ndim < 1
            or len(value) != self.physical_count
            or not torch.isfinite(value).all()
            for value in values.values()
        ):
            raise ValueError("Native pairing proof requires finite values for every physical world")
        for group in self.groups():
            for name, value in values.items():
                if not torch.equal(value[list(group["policy"])], value[list(group["baseline"])]):
                    raise RuntimeError(
                        f"Native paired physics differs for seed {group['seed']}: {name}"
                    )
        return {
            "verified": True,
            "matching_method": PARALLEL_MATCHING_METHOD,
            "physical_worlds": self.physical_count,
            "matched_pairs": self.per_controller * len(self.seeds),
            "field_sha256": {
                name: hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
                for name, value in values.items()
            },
            "batch_groups": self.groups(),
            "scene_origins": self.origin_statistics(origins),
        }
