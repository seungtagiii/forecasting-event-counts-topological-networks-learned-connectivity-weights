"""Two extra fit seeds only; original model/fit/evaluation code unchanged."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from datasets_v2 import load, load_labels, read, save
from models_v2 import CountHistory
from study_v2 import fit_candidate, restore, evaluate
from materialize_v2 import array_digest


def materialize(source, inputs, destination):
    destination.mkdir(parents=True, exist_ok=False)
    blueprint = read(source / "data_blueprint.json")
    rows = {}
    for name, spec in blueprint.items():
        paths = list(inputs.rglob(name + "/observations.npz"))
        matches = []
        for path in paths:
            obs, metadata = load(path.parent)
            actual = {key: array_digest(value) for key, value in obs.items()}
            labels = load_labels(path.parent, metadata)
            actual["graph_source_target"] = array_digest(labels)
            if actual == spec["arrays"]:
                matches.append((path.parent, obs, labels, metadata))
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous matching original dataset")
        original, obs, labels, metadata = matches[0]
        target = destination / name
        target.mkdir(exist_ok=False)
        np.savez_compressed(target / "observations.npz", **obs)
        np.savez_compressed(target / "evaluation_labels.npz", graph_source_target=labels)
        from datasets_v2 import digest
        metadata = dict(metadata, observation_sha256=digest(target / "observations.npz"),
                        labels_sha256=digest(target / "evaluation_labels.npz"))
        save(target / "dataset.json", metadata)
        rows[name] = {"all_array_hashes_match": True, "arrays": spec["arrays"]}
        del obs, labels, matches
    save(destination / "materialization_manifest.json", rows)


def repeat_dataset(data, output, protocol, repetition, device, deadline):
    output.mkdir(parents=True, exist_ok=False)
    obs, metadata = load(data)
    history = CountHistory(obs, device, deadline)
    del obs
    selections = {}
    fits = 0
    offset = {1: 200000, 2: 400000}[repetition]
    for name in protocol["repeat_models"]:
        setting = protocol["base_protocol"]["s2" if name == "s2" else "neural"]
        penalties = setting["regularization_candidates"] if name == "s2" else [None]
        candidates = []
        for lr in setting["learning_rates"]:
            for penalty in penalties:
                trial = len(candidates)
                relative = f"fits/{name}_trial{trial}"
                seed = metadata["seed"] + offset + trial
                fit = fit_candidate(name, history, metadata, protocol["base_protocol"], output / relative, lr, penalty, seed, deadline)
                candidates.append({"fit_path": relative, "trial": trial, "report": fit})
                fits += 1
        selected = min(candidates, key=lambda row: (row["report"]["best_validation_mean_score"], row["trial"]))
        selections[name] = selected
        save(output / f"{name}_selection.json", {"selected": selected, "candidates": candidates, "test_or_graph_used": False})
    save(output / "validation_selection.json", selections)
    truth = load_labels(data, metadata)
    results = []
    for name, selected in selections.items():
        model, graph = restore(name, history, metadata, protocol["base_protocol"], selected["report"], output / selected["fit_path"])
        report = evaluate(model, history, metadata, protocol["base_protocol"], graph, truth,
                          output / "evaluations" / name, deadline)
        results.append(report)
        del model
    save(output / "completion.json", {"status": "complete", "dataset": metadata["name"],
         "repetition": repetition, "fit_seed_offset": offset, "fits": fits, "tests": len(results),
         "data_split_validation_MC_and_TEST_MC_seeds_unchanged": True,
         "new_independent_data": False, "results": results})
    del history
    if device.type == "cuda":
        torch.cuda.empty_cache()


def worker(data, output, protocol_path, datasets, device, seconds):
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + seconds
    protocol = read(protocol_path)
    torch.set_num_threads(2)
    save(output / "environment.json", {"device": str(device), "torch": str(torch.__version__),
         "visible_cuda_devices": torch.cuda.device_count(),
         "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    for name in datasets:
        for repetition in (1, 2):
            repeat_dataset(data / name, output / name / f"repeat_{repetition}", protocol, repetition, device, deadline)


def main(data, output, protocol_path, seconds):
    output.mkdir(parents=True, exist_ok=False)
    protocol = read(protocol_path)
    if torch.cuda.device_count() != 2 or not all("T4" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Expected exactly two T4 devices")
    from orchestrator_v2 import gpu_preflight
    save(output / "GPU_preflight.json", gpu_preflight(protocol["base_protocol"]))
    save(output / "protocol.json", protocol)
    start = time.monotonic()
    deadline = start + seconds
    processes, streams = [], []
    jobs = [("simulation_96100431", "simulation_96100432"), ("real_24V_439N_Microwave",)]
    try:
        for i, names in enumerate(jobs):
            stream = (output / f"worker_{i}.log").open("x", encoding="utf-8")
            streams.append(stream)
            args = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker",
                    "--data", str(data), "--output", str(output / f"worker_{i}"),
                    "--protocol", str(protocol_path), "--datasets", *names,
                    "--device", f"cuda:{i}", "--seconds", str(max(1, int(deadline - time.monotonic())))]
            processes.append(subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if time.monotonic() >= deadline:
                for p in processes:
                    if p.poll() is None:
                        p.terminate()
                break
            time.sleep(2)
        for p in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for stream in streams:
            stream.close()
    completions = [read(x) for x in output.glob("worker_*/*/repeat_*/completion.json")]
    fits, tests = sum(x["fits"] for x in completions), sum(x["tests"] for x in completions)
    success = fits == 60 and tests == 24 and len(completions) == 6 and all(p.returncode == 0 for p in processes)
    save(output / "completion.json", {"status": "complete" if success else "incomplete",
         "planned_new_fits": 60, "planned_new_selected_tests": 24, "completed_fits": fits, "completed_tests": tests,
         "completed_dataset_repetitions": len(completions), "worker_exit_codes": [p.returncode for p in processes],
         "elapsed_seconds": time.monotonic() - start, "automatic_retry": False,
         "original_fits_reused": 33, "original_tests_reused": 15, "new_data_repetitions": 0})
    if not success:
        raise RuntimeError("Incomplete bounded seed study; no retries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--seconds", type=int, default=6900)
    args = parser.parse_args()
    if args.worker:
        worker(args.data, args.output, args.protocol, args.datasets, torch.device(args.device), args.seconds)
    else:
        main(args.data, args.output, args.protocol, args.seconds)
