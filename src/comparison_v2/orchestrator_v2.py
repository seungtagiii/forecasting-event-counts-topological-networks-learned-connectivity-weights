"""Two bounded GPU workers, one submission, no retries or follow-up studies."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from datasets_v2 import read, save
from models_v2 import CountHistory, NeuralGraph, log_probability


def gpu_preflight(protocol):
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This bounded protocol needs two visible T4 devices")
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError("Hardware differs from requested T4 budget")
    report = {"devices": names, "checks": []}
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        counts = np.zeros((12, 2, 3), dtype=np.int32)
        counts[2, 0, 0], counts[4, 1, 1], counts[6, 0, 2] = 1, 2, 1
        obs = {"counts": counts, "hop_masks": np.stack([np.eye(2), np.asarray([[0, 1], [1, 0]]), np.zeros((2, 2))]).astype(np.float32)}
        history = CountHistory(obs, device, time.monotonic() + 60)
        history.build_exponential(.2, time.monotonic() + 60)
        pairs = torch.tensor([12, 13, 14, 15], device=device)
        _, nodes = history.indices(pairs)
        for model_name in ["tnpar", "direct", "deterministic"]:
            model = NeuralGraph(model_name, protocol["neural"], history.counts[:6].mean(0).clamp_min(1e-5), 3).to(device)
            loss, _ = model.objective(history.lagged(pairs, 5), nodes, history.y(pairs), 36)
            loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("GPU preflight gradient failed")
            report["checks"].append({"device": str(device), "model": model_name, "finite_forward_backward": True})
            del model
        del history
        torch.cuda.empty_cache()
    return report


def main(data_root, output, protocol_path, max_seconds):
    output.mkdir(parents=True, exist_ok=False)
    protocol = read(protocol_path)
    start = time.monotonic()
    deadline = start + max_seconds
    preflight = gpu_preflight(protocol)
    save(output / "GPU_preflight.json", preflight)
    save(output / "protocol.json", protocol)
    sources = {path.name: __import__("hashlib").sha256(path.read_bytes()).hexdigest() for path in protocol_path.parent.iterdir() if path.suffix in {".py", ".json"} and path.name != "payload_manifest.json"}
    save(output / "executed_source_manifest.json", sources)
    jobs = [([f"simulation_{seed}" for seed in protocol["simulation"]["seeds"]], "cuda:0"),
            (["real_24V_439N_Microwave"], "cuda:1")]
    processes, logs = [], []
    try:
        for index, (names, device) in enumerate(jobs):
            log = (output / f"worker_{index}.log").open("x", encoding="utf-8")
            logs.append(log)
            args = [sys.executable, "-X", "utf8", str(protocol_path.parent / "study_v2.py"),
                    "--data-root", str(data_root), "--output", str(output / f"worker_{index}"),
                    "--datasets", *names, "--protocol", str(protocol_path), "--device", device,
                    "--max-seconds", str(max(1, int(deadline - time.monotonic())))]
            processes.append(subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= deadline:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            time.sleep(2)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    completions = [read(path) for path in output.glob("worker_*/*/completion.json")]
    fits = sum(row["fits_completed"] for row in completions)
    tests = sum(row["selected_test_evaluations_completed"] for row in completions)
    success = len(processes) == 2 and all(process.returncode == 0 for process in processes) and fits == 33 and tests == 15 and len(completions) == 3
    save(output / "completion.json", {"status": "complete" if success else "incomplete", "planned_fits": 33,
                                       "planned_selected_tests": 15, "completed_fits_in_complete_datasets": fits,
                                       "completed_selected_tests_in_complete_datasets": tests, "completed_datasets": len(completions),
                                       "worker_exit_codes": [process.returncode for process in processes],
                                       "elapsed_seconds": time.monotonic() - start, "automatic_retry": False,
                                       "additional_experiments": 0, "GPU_devices": ["cuda:0", "cuda:1"]})
    if not success:
        raise RuntimeError("Bounded comparison incomplete; retained partial outputs, no retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--max-seconds", type=int, default=13800)
    args = parser.parse_args()
    main(args.data_root, args.output, args.protocol, args.max_seconds)
