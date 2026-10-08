"""Tiny CPU-only repeat wrapper integration test."""
import copy
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from datasets_v2 import read, write_dataset, save
from minimum_repeat import repeat_dataset, materialize
from materialize_v2 import array_digest
from test_v2 import fixture


def main():
    torch.set_num_threads(2)
    base = read(Path(__file__).with_name("protocol.json"))
    small = copy.deepcopy(base)
    small["neural"].update(hidden_units=4, hidden_layers=1, maximum_updates=2,
                          minimum_updates=2, validation_every_updates=1, patience_checks=2,
                          batch_pairs=8, graph_calibration_TRAIN_pairs_max=12,
                          validation_pairs_max=12, test_batch_pairs=8,
                          validation_graph_draws=2, test_graph_draws=2)
    small["s2"]["maximum_updates"] = 2
    protocol = {"base_protocol": small, "repeat_models": ["tnpar", "direct", "deterministic", "s2"]}
    with tempfile.TemporaryDirectory(prefix="minimum_repeat_check_") as temp:
        root = Path(temp)
        obs = fixture()
        truth = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8)
        metadata = {"kind": "simulation", "name": "tiny", "seed": 42, "train": [5, 20],
                    "validation": [20, 25], "test": [25, 30], "delta_seconds": 2,
                    "graph_labels": "synthetic_test_only"}
        write_dataset(root / "inputs/tiny", obs["counts"], obs["topology"], truth, metadata)
        source = root / "source"
        source.mkdir()
        arrays = {key: array_digest(value) for key, value in obs.items()}
        arrays["graph_source_target"] = array_digest(truth)
        save(source / "data_blueprint.json", {"tiny": {"arrays": arrays}})
        materialize(source, root / "inputs", root / "materialized")
        assert read(root / "materialized/materialization_manifest.json")["tiny"]["all_array_hashes_match"]
        for repetition in (1, 2):
            output = root / f"repeat_{repetition}"
            repeat_dataset(root / "materialized/tiny", output, protocol, repetition, torch.device("cpu"), time.monotonic() + 60)
            complete = read(output / "completion.json")
            assert complete["fits"] == 10 and complete["tests"] == 4
            selected = read(output / "validation_selection.json")
            for name in protocol["repeat_models"]:
                candidate = read(output / f"{name}_selection.json")
                minimum = min(candidate["candidates"], key=lambda x: (x["report"]["best_validation_mean_score"], x["trial"]))
                assert selected[name] == minimum
                for row in candidate["candidates"]:
                    assert row["report"]["seed"] == 42 + {1: 200000, 2: 400000}[repetition] + row["trial"]
                    assert not row["report"]["true_graph_or_TEST_used_for_fit"]
                checkpoint = torch.load(output / selected[name]["fit_path"] / "best.pt", map_location="cpu", weights_only=True)
                assert checkpoint["update"] == selected[name]["report"]["best_update"]
                result = read(output / "evaluations" / name / "result.json")
                assert not result["TEST_used_to_change_model"]
                assert result["structure"]["threshold_fixed_before_test"] == (.001 if name == "s2" else .5)
    report = {"status": "passed", "CPU_tiny_fits": 20, "CPU_tiny_evaluations": 8,
              "seed_offsets_verified": True, "validation_selection_verified": True,
              "checkpoint_verified": True, "original_structure_thresholds_retained": True,
              "materialized_arrays_verified": True,
              "GPU_used": False, "base_protocol_unchanged": base == read(Path(__file__).with_name("protocol.json"))}
    save(Path(__file__).with_name("minimum_repeat_test_" + str(time.time_ns()) + ".json"), report)
    print(__import__("json").dumps(report))


if __name__ == "__main__":
    main()
