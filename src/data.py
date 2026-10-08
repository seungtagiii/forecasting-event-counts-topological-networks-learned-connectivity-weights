"""Marked branching simulator audit adapter; duration is NEVER a learner input.

Source lineage: gCastle THPSimulation 1.0.3. Deliberate differences: local RNG,
explicit horizon clipping, event safety limits, sorted records, stable array schema.
This is not asserted to be the exact original TNPAR simulation implementation.
"""
from __future__ import annotations

from collections import deque
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def graph_masks(topology: np.ndarray, max_hop: int) -> np.ndarray:
    """Exact-hop masks [hop, destination node, source node], including own node."""
    n = len(topology)
    if topology.shape != (n, n) or not np.array_equal(topology, topology.T):
        raise ValueError("Physical topology must be a symmetric square matrix")
    result = np.zeros((max_hop + 1, n, n), dtype=np.float32)
    for target in range(n):
        distances = {target: 0}
        queue = deque([target])
        while queue:
            current = queue.popleft()
            if distances[current] >= max_hop:
                continue
            for neighbor in np.flatnonzero(topology[current]):
                if int(neighbor) not in distances:
                    distances[int(neighbor)] = distances[current] + 1
                    queue.append(int(neighbor))
        for source, distance in distances.items():
            result[distance, target, source] = 1
    return result


def random_graphs(rng: np.random.Generator, config: dict) -> tuple[np.ndarray, np.ndarray]:
    p, n = config["event_types"], config["physical_nodes"]
    if p < 2 or n < 1 or config["max_hop"] < 0:
        raise ValueError("Invalid graph dimensions")
    # Mirrors the ER probability convention in gCastle, then orient by an order.
    probability = min(1., 2 * config["causal_edges_expected"] / p**2)
    upper = np.triu(rng.random((p, p)) < probability, 1)
    permutation = rng.permutation(p)
    causal = upper[permutation][:, permutation].astype(np.int8)  # source -> target
    topo_probability = min(1., 2 * config["topology_edges_expected"] / n**2)
    upper_topo = np.triu(rng.random((n, n)) < topo_probability, 1)
    topology = (upper_topo | upper_topo.T).astype(np.int8)
    return causal, topology


def simulate(config: dict, seed: int, deadline: float | None = None) -> tuple[dict, dict, dict]:
    if config.get("sampling_mode") == "event_count_chronological":
        return simulate_event_prefix(config, seed, deadline)
    rng = np.random.default_rng(seed)
    causal, topology = random_graphs(rng, config)
    p, n = config["event_types"], config["physical_nodes"]
    T, delta = float(config["horizon"]), float(config["delta"])
    if T <= 0 or delta <= 0 or not np.isclose(T / delta, round(T / delta)):
        raise ValueError("Horizon must be a positive integer multiple of delta")
    mu = rng.uniform(*config["mu_range"], p)
    weights = rng.uniform(*config["alpha_range"], (p, p)) * causal
    masks = graph_masks(topology, config["max_hop"])
    neighbors = {source: np.flatnonzero(masks[:, :, source].sum(0)) for source in range(n)}
    all_events: list[tuple[int, float, int, float, int, int]] = []

    def trigger(rates, source_start, length, node, parent, generation):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Declared wall-clock limit reached during simulation")
        generated = []
        for event_type, rate in enumerate(rates):
            if rate <= 0:
                continue
            clock = float(source_start)
            while True:
                clock = float(np.rint(clock + rng.exponential(1. / rate)))
                if clock > source_start + length or clock > T:
                    break
                duration = float(np.rint(rng.exponential(config["duration_mean"])))
                event_id = len(all_events)
                event = (event_type, clock, node, duration, parent, generation)
                all_events.append(event)
                if len(all_events) % 1000 == 0 and deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Declared wall-clock limit reached during simulation")
                if len(all_events) > config["maximum_events"]:
                    raise RuntimeError("Event safety limit exceeded; no silent truncation")
                generated.append(event_id)
        return generated

    generation = []
    for node in range(n):
        generation.extend(trigger(mu, 0., T, node, -1, 0))
    roots = len(generation)
    depth = 0
    while generation:
        depth += 1
        if depth > config["maximum_generation"]:
            raise RuntimeError("Generation safety limit exceeded")
        offspring = []
        for parent in generation:
            event_type, start, node, duration, _, _ = all_events[parent]
            for target_node in neighbors[node]:
                offspring.extend(trigger(weights[event_type], start, duration,
                                         int(target_node), parent, depth))
        generation = offspring
    if not all_events:
        raise RuntimeError("No events generated; do not change the seed silently")
    event_array = np.asarray(all_events, dtype=np.float64)
    order = np.lexsort((event_array[:, 0], event_array[:, 2], event_array[:, 1]))
    sorted_events = event_array[order]
    counts = np.zeros((int(round(T / delta)), n, p), dtype=np.int32)
    # TNPAR definition: intervals ((t-1)*delta,t*delta]; time zero goes in bin 0.
    bins = np.clip(np.ceil(sorted_events[:, 1] / delta).astype(int) - 1, 0, len(counts) - 1)
    np.add.at(counts, (bins, sorted_events[:, 2].astype(int), sorted_events[:, 0].astype(int)), 1)
    observations = {"counts": counts, "topology": topology, "hop_masks": masks,
                    "event_type": sorted_events[:, 0].astype(np.int32),
                    "timestamp": sorted_events[:, 1], "node": sorted_events[:, 2].astype(np.int32),
                    "delta": np.asarray(delta)}
    # Never delivered to model fitting. Marks/parents are diagnostic oracle state.
    latent = {"causal_graph_source_target": causal, "excitation_weights": weights,
              "baseline_intensity": mu, "duration": sorted_events[:, 3],
              "unsorted_branching_records": event_array, "sorting_order": order}
    metadata = {"generator": "gCastle_duration_branching_reimplementation",
                "exact_TNPAR_reproduction": False, "seed": seed, "data_config": config,
                "config_sha256": canonical_hash(config), "event_records": len(sorted_events),
                "root_events": roots, "maximum_generation_observed": depth - 1,
                "causal_edges_actual": int(causal.sum()), "physical_edges_actual": int(topology.sum() // 2),
                "nonzero_count_fraction": float(np.mean(counts > 0)),
                "hidden_duration_used_by_learners": False,
                "lambda_truth_for_common_observed_history_available": False,
                "event_count_matches_paper_nominal": len(sorted_events) == config.get("paper_nominal_event_records"),
                "changes": ["fixed_horizon_clipping", "local_RNG", "chronological_sort",
                            "durations_retained_only_in_oracle_diagnostics", "no_event_count_truncation"]}
    return observations, latent, metadata


def simulate_event_prefix(config: dict, seed: int, deadline: float | None = None):
    """Chronological marked branching prefix without materializing all descendants.

    Each heap item represents the next arrival in an independent Poisson stream.
    A processed event schedules both the next sibling and its offspring streams.
    Clock/duration rounding follows gCastle. RNG draw order is deliberately not
    identical to its generation-by-generation algorithm. Exact author identity
    is not claimed. Only the incomplete last time bin is excluded downstream.
    """
    rng = np.random.default_rng(seed)
    causal, topology = random_graphs(rng, config)
    p, n = config["event_types"], config["physical_nodes"]
    delta = float(config["delta"])
    requested = int(config["paper_nominal_event_records"])
    if delta <= 0 or requested < 1 or requested > config["maximum_events"]:
        raise ValueError("Invalid event-prefix budget")
    mu = rng.uniform(*config["mu_range"], p)
    weights = rng.uniform(*config["alpha_range"], (p, p)) * causal
    masks = graph_masks(topology, config["max_hop"])
    neighbors = {source: np.flatnonzero(masks[:, :, source].sum(0)) for source in range(n)}
    queue, serial = [], 0

    def schedule(event_type, node, clock, rate, end, parent, generation):
        nonlocal serial
        arrival = float(np.rint(clock + rng.exponential(1. / rate)))
        if arrival <= end:
            serial += 1
            heapq.heappush(queue, (arrival, serial, event_type, node, rate, end, parent, generation))
            if len(queue) > 2_000_000:
                raise RuntimeError("Pending-stream safety limit exceeded; no silent clipping")

    for node in range(n):
        for event_type, rate in enumerate(mu):
            schedule(event_type, node, 0., rate, float("inf"), -1, 0)
    records = []
    roots, deepest = 0, 0
    while len(records) < requested:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Declared wall-clock limit reached during prefix simulation")
        if not queue:
            raise RuntimeError("Poisson streams unexpectedly exhausted")
        start, _, event_type, node, rate, end, parent, generation = heapq.heappop(queue)
        duration = float(np.rint(rng.exponential(config["duration_mean"])))
        event_id = len(records)
        records.append((event_type, start, node, duration, parent, generation))
        roots += int(parent == -1)
        deepest = max(deepest, generation)
        if generation > config["maximum_generation"]:
            raise RuntimeError("Generation safety limit exceeded")
        schedule(event_type, node, start, rate, end, parent, generation)
        for target_node in neighbors[node]:
            for target_type in np.flatnonzero(weights[event_type]):
                schedule(int(target_type), int(target_node), start, weights[event_type, target_type],
                         start + duration, event_id, generation + 1)
    events = np.asarray(records, dtype=np.float64)
    number_bins = max(1, int(np.ceil(events[-1, 1] / delta)))
    if number_bins > config.get("maximum_time_bins", 20000):
        raise RuntimeError("Time-bin memory safety limit exceeded; no silent seed replacement")
    counts = np.zeros((number_bins, n, p), dtype=np.int32)
    bins = np.clip(np.ceil(events[:, 1] / delta).astype(int) - 1, 0, number_bins - 1)
    np.add.at(counts, (bins, events[:, 2].astype(int), events[:, 0].astype(int)), 1)
    observations = {"counts": counts, "topology": topology, "hop_masks": masks,
                    "event_type": events[:, 0].astype(np.int32), "timestamp": events[:, 1],
                    "node": events[:, 2].astype(np.int32), "delta": np.asarray(delta)}
    latent = {"causal_graph_source_target": causal, "excitation_weights": weights,
              "baseline_intensity": mu, "duration": events[:, 3],
              "unsorted_branching_records": events, "sorting_order": np.arange(len(events))}
    metadata = {"generator": "chronological_marked_Poisson_branching_prefix_reimplementation",
                "exact_TNPAR_reproduction": False, "seed": seed, "data_config": config,
                "config_sha256": canonical_hash(config), "event_records": len(events),
                "root_events": roots, "maximum_generation_observed": deepest,
                "causal_edges_actual": int(causal.sum()), "physical_edges_actual": int(topology.sum() // 2),
                "nonzero_count_fraction": float(np.mean(counts > 0)),
                "hidden_duration_used_by_learners": False,
                "lambda_truth_for_common_observed_history_available": False,
                "event_count_matches_paper_nominal": len(events) == requested,
                "last_partial_bin_excluded_from_all_fit_and_evaluation": bool(config["exclude_partial_final_bin"]),
                "complete_bin_event_records": int(counts[:-1].sum()),
                "time_bins_including_partial": number_bins,
                "simulation_end_timestamp": float(events[-1, 1]),
                "changes": ["chronological_event_count_stopping", "local_RNG_draw_order",
                            "durations_retained_only_in_oracle_diagnostics", "incomplete_last_bin_exclusion"]}
    return observations, latent, metadata


def save_dataset(path: Path, config: dict, seed: int, deadline: float | None = None) -> dict:
    path.mkdir(parents=True, exist_ok=False)
    observations, latent, metadata = simulate(config, seed, deadline)
    np.savez_compressed(path / "observations.npz", **observations)
    np.savez_compressed(path / "oracle_diagnostics.npz", **latent)
    metadata["observation_sha256"] = hashlib.sha256((path / "observations.npz").read_bytes()).hexdigest()
    metadata["oracle_sha256"] = hashlib.sha256((path / "oracle_diagnostics.npz").read_bytes()).hexdigest()
    with (path / "data_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    return metadata


def load_observations(path: Path) -> dict:
    with np.load(path / "observations.npz", allow_pickle=False) as archive:
        expected = {"counts", "topology", "hop_masks", "event_type", "timestamp", "node", "delta"}
        if set(archive.files) != expected:
            raise ValueError("Observation schema changed or contains privileged simulator state")
        return {key: archive[key] for key in archive.files}


def features_and_split(observations: dict, config: dict):
    counts = observations["counts"].astype(np.float32)
    if config["data"].get("exclude_partial_final_bin", False):
        counts = counts[:-1]
    masks = observations["hop_masks"]
    L = config["model"]["lags"]
    T, n, p = counts.shape
    if T < L + 15:
        raise ValueError("Not enough time bins")
    # Information-preserving w.r.t. counts is NOT claimed: sums per hop are
    # explicitly recorded as a reimplementation choice, not official TNPAR code.
    aggregated = np.einsum("knm,tmp->tnkp", masks, counts, optimize=True)
    features = np.stack([aggregated[L - lag:T - lag] for lag in range(1, L + 1)], axis=-1)
    target_times = np.arange(L, T)
    train_end = int(T * config["training"]["train_fraction"])
    val_end = train_end + int(T * config["training"]["validation_fraction"])
    split = {"train": np.flatnonzero(target_times < train_end),
             "validation": np.flatnonzero((target_times >= train_end) & (target_times < val_end)),
             "test": np.flatnonzero(target_times >= val_end)}
    if any(len(index) == 0 for index in split.values()):
        raise ValueError("Empty chronological split")
    return features, counts[L:], split, target_times
