"""Read-only old Kaggle Outputs + a compact public real count carrier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from datasets_v2 import digest, read, save


def array_digest(array):
    array = np.asarray(array)
    header = json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def materialize(source, input_root, destination):
    source, input_root, destination = Path(source), Path(input_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    blueprint = read(source / "data_blueprint.json")
    report = {}
    for name, spec in blueprint.items():
        metadata = spec["metadata"]
        if name.startswith("simulation_"):
            seed = metadata["seed"]
            candidates = [path for path in input_root.rglob("observations.npz") if path.parent.name == f"data_seed_{seed}"]
            selected = None
            for path in candidates:
                with np.load(path, allow_pickle=False) as archive:
                    observations = {key: archive[key] for key in ["counts", "topology", "hop_masks"]}
                if all(array_digest(observations[key]) == spec["arrays"][key] for key in observations):
                    selected = path
                    break
            if selected is None:
                raise ValueError("Attached simulation Outputs missing or altered")
            with np.load(selected.parent / "oracle_diagnostics.npz", allow_pickle=False) as archive:
                truth = archive["causal_graph_source_target"]
        else:
            carrier = source / "real_sparse_input.npz"
            if digest(carrier) != spec["carrier_sha256"]:
                raise ValueError("Real count carrier altered")
            with np.load(carrier, allow_pickle=False) as archive:
                shape = tuple(archive["shape"])
                counts = np.zeros(int(np.prod(shape)), dtype=np.int32)
                counts[archive["indices"]] = archive["values"]
                observations = {"counts": counts.reshape(shape), "topology": archive["topology"], "hop_masks": archive["hop_masks"]}
                truth = archive["truth"]
        actual = {key: array_digest(value) for key, value in observations.items()}
        actual["graph_source_target"] = array_digest(truth)
        if actual != spec["arrays"]:
            raise ValueError("Materialized arrays differ from immutable local preparation")
        path = destination / name
        path.mkdir(exist_ok=False)
        np.savez_compressed(path / "observations.npz", **observations)
        np.savez_compressed(path / "evaluation_labels.npz", graph_source_target=truth)
        metadata = dict(metadata, observation_sha256=digest(path / "observations.npz"), labels_sha256=digest(path / "evaluation_labels.npz"))
        save(path / "dataset.json", metadata)
        report[name] = {"all_array_hashes_match": True, "array_hashes": actual,
                        "files": {filename: digest(path / filename) for filename in ["observations.npz", "evaluation_labels.npz", "dataset.json"]}}
    save(destination / "materialization_manifest.json", report)
    return report
