"""Convert the verified public alarm archive to the study's binned format."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "comparison_v2"))
from datasets_v2 import read, write_dataset  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--output", type=Path, default=REPO / "data" / "prepared" / "real_24V_439N_Microwave")
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.output.resolve()
    manifest = read(raw / "source_manifest.json")
    if manifest["dataset_commit"] != "1858e9c39d90a524e7d8ff8b9456f4b1f231ff03":
        raise ValueError("Unexpected dataset revision")
    for name in ("Alarm.csv", "DAG.npy", "Topology.npy"):
        source = manifest["sources"][name]
        if sha256(raw / name) != source["sha256"]:
            raise ValueError(f"Hash mismatch: {name}")

    with (raw / "Alarm.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = np.asarray([[int(row[key]) for key in ("alarm_id", "device_id", "start_timestamp", "end_timestamp")]
                              for row in csv.DictReader(stream)], dtype=np.int64)
    topology = np.load(raw / "Topology.npy", allow_pickle=False)
    truth = np.load(raw / "DAG.npy", allow_pickle=False)
    if records[:, 0].min() < 0 or records[:, 0].max() >= len(truth):
        raise ValueError("Alarm IDs do not match the provided reference graph")
    if records[:, 1].min() < 0 or records[:, 1].max() >= len(topology):
        raise ValueError("Device IDs do not match the provided topology")

    protocol = read(REPO / "src" / "comparison_v2" / "protocol.json")
    delta = int(protocol["real"]["delta_seconds"])
    origin = int(records[:, 2].min())
    bins = (records[:, 2] - origin) // delta
    complete_bins = int((records[:, 2].max() - origin) // delta)
    keep = bins < complete_bins
    counts = np.zeros((complete_bins, len(topology), len(truth)), dtype=np.int32)
    np.add.at(counts, (bins[keep], records[keep, 1], records[keep, 0]), 1)
    train_end, validation_end = int(.6 * complete_bins), int(.8 * complete_bins)
    write_dataset(output, counts, topology, truth,
                  {"kind": "real", "name": "real_24V_439N_Microwave", "seed": 96100440,
                   "delta_seconds": float(delta), "train": [5, train_end],
                   "validation": [train_end, validation_end], "test": [validation_end, complete_bins],
                   "source_commit": manifest["dataset_commit"], "source_hashes": manifest["sources"],
                   "original_rows": len(records), "excluded_partial_final_bin_rows": int((~keep).sum()),
                   "duration_positive_fraction": float(np.mean(records[:, 3] > records[:, 2])),
                   "bin_convention": "half_open_[start,end);_no_duplicate_removal",
                   "graph_labels": "provided_reference_DAG_not_identified_true_Granger_graph"})
    print(json.dumps({"status": "prepared", "name": "real_24V_439N_Microwave",
                      "shape": list(counts.shape), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
