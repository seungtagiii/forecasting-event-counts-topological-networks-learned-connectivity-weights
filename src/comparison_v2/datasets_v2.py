"""Immutable prepared count grids, with graph labels kept outside fit inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from data import graph_masks


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_dataset(destination, counts, topology, truth, metadata):
    destination.mkdir(parents=True, exist_ok=False)
    if counts.ndim != 3 or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError("Invalid count observations")
    if topology.shape != (counts.shape[1], counts.shape[1]) or truth.shape != (counts.shape[2], counts.shape[2]):
        raise ValueError("Declared node/type ID dimensions differ")
    masks = graph_masks(topology, 2)
    np.savez_compressed(destination / "observations.npz", counts=counts, topology=topology, hop_masks=masks)
    np.savez_compressed(destination / "evaluation_labels.npz", graph_source_target=truth)
    metadata.update(shape=list(counts.shape), event_records_complete_bins=int(counts.sum()),
                    observation_sha256=digest(destination / "observations.npz"),
                    labels_sha256=digest(destination / "evaluation_labels.npz"),
                    truth_delivered_to_fits=False, duration_as_input=False)
    save(destination / "dataset.json", metadata)


def prepare():
    protocol = read(ROOT / "protocol.json")
    output = ROOT / "prepared_data"
    output.mkdir(exist_ok=False)
    for seed in protocol["simulation"]["seeds"]:
        source = ROOT.parent / "preflight/fixed_time_20261004T090033Z_84c6c32a" / f"data_seed_{seed}"
        manifest = read(source / "data_manifest.json")
        if digest(source / "observations.npz") != manifest["observation_sha256"] or digest(source / "oracle_diagnostics.npz") != manifest["oracle_sha256"]:
            raise ValueError("Original simulation hashes changed")
        with np.load(source / "observations.npz", allow_pickle=False) as obs, np.load(source / "oracle_diagnostics.npz", allow_pickle=False) as labels:
            counts, topology = obs["counts"], obs["topology"]
            write_dataset(output / f"simulation_{seed}", counts, topology, labels["causal_graph_source_target"],
                          {"kind": "simulation", "name": f"simulation_{seed}", "seed": seed,
                           "delta_seconds": 2., "train": [250, 3250], "validation": [3250, 4250], "test": [4250, 5250],
                           "original_observation_sha256": manifest["observation_sha256"],
                           "same_previously_inspected_path": True, "new_independent_repetition": False,
                           "graph_labels": "simulator_excitation_graph_not_certified_binned_Granger_graph"})
    raw = ROOT / "public_inputs"
    source_manifest = read(raw / "source_manifest.json")
    for filename in ["Alarm.csv", "DAG.npy", "Topology.npy"]:
        if digest(raw / filename) != source_manifest["sources"][filename]["sha256"]:
            raise ValueError("Public data source hash changed")
    with (raw / "Alarm.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = np.asarray([[int(row[k]) for k in ["alarm_id", "device_id", "start_timestamp", "end_timestamp"]] for row in csv.DictReader(stream)], dtype=np.int64)
    topology, truth = np.load(raw / "Topology.npy", allow_pickle=False), np.load(raw / "DAG.npy", allow_pickle=False)
    if records[:, 0].min() < 0 or records[:, 0].max() >= len(truth) or records[:, 1].min() < 0 or records[:, 1].max() >= len(topology):
        raise ValueError("Public IDs are not the declared matrix IDs")
    delta = protocol["real"]["delta_seconds"]
    origin = int(records[:, 2].min())
    complete_bins = int((records[:, 2].max() - origin) // delta)
    bins = (records[:, 2] - origin) // delta
    keep = bins < complete_bins
    counts = np.zeros((complete_bins, len(topology), len(truth)), dtype=np.int32)
    np.add.at(counts, (bins[keep], records[keep, 1], records[keep, 0]), 1)
    train_end, val_end = int(.6 * complete_bins), int(.8 * complete_bins)
    write_dataset(output / "real_24V_439N_Microwave", counts, topology, truth,
                  {"kind": "real", "name": "real_24V_439N_Microwave", "seed": 96100440,
                   "delta_seconds": float(delta), "train": [5, train_end], "validation": [train_end, val_end], "test": [val_end, complete_bins],
                   "source_commit": source_manifest["dataset_commit"], "source_hashes": source_manifest["sources"],
                   "original_rows": len(records), "excluded_partial_final_bin_rows": int((~keep).sum()),
                   "duration_positive_fraction": float(np.mean(records[:, 3] > records[:, 2])),
                   "bin_convention": "half_open_[start,end);_no_duplicate_removal",
                   "graph_labels": "provided_reference_DAG_not_identified_true_Granger_graph"})
    hashes = {str(path.relative_to(output)).replace("\\", "/"): digest(path) for path in output.rglob("*") if path.is_file()}
    save(output / "manifest.json", hashes)
    print(json.dumps({"prepared_datasets": 3, "new_simulation_repetitions": 0, "real_complete_rows": int(counts.sum()), "hashes_saved": True}))


def load(path):
    path = Path(path)
    metadata = read(path / "dataset.json")
    if digest(path / "observations.npz") != metadata["observation_sha256"]:
        raise ValueError("Prepared observations changed")
    with np.load(path / "observations.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"counts", "topology", "hop_masks"}:
            raise ValueError("Privileged input in observation archive")
        observations = {name: archive[name] for name in archive.files}
    return observations, metadata


def load_labels(path, metadata):
    if digest(Path(path) / "evaluation_labels.npz") != metadata["labels_sha256"]:
        raise ValueError("Evaluation reference changed")
    with np.load(Path(path) / "evaluation_labels.npz", allow_pickle=False) as archive:
        return archive["graph_source_target"]


if __name__ == "__main__":
    prepare()
