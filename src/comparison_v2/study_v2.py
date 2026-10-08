"""Bounded five-model study; no network calls or automatic retries."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from evaluation import graph_metrics
from datasets_v2 import load, load_labels, read, save
from models_v2 import CountHistory, NeuralGraph, S2Rectangle, THP_EM, check_time, log_probability


def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def pair_sample(interval, nodes, maximum, seed, device):
    first, last = interval[0] * nodes, interval[1] * nodes
    rng = np.random.default_rng(seed)
    index = np.arange(first, last, dtype=np.int64)
    if len(index) > maximum:
        index = np.sort(rng.choice(index, size=maximum, replace=False))
    return torch.as_tensor(index, device=device)


def input_for(model, history, indices, protocol):
    if model.name == "s2":
        return history.rectangle(indices, protocol["s2"]["window_bins"])
    if model.name == "thp":
        return history.exp_history(indices)
    return history.lagged(indices, protocol["neural"]["lags"])


@torch.no_grad()
def freeze(model, history, calibration_indices, protocol, deadline):
    if model.name == "s2":
        return model.weights().detach()
    if model.name == "thp":
        return torch.as_tensor(model.graph, device=history.device)
    if model.name != "tnpar":
        return model.probabilities().detach()
    total = torch.zeros((history.H, history.P, history.P), device=history.device, dtype=torch.float64)
    for chunk in calibration_indices.split(4096):
        check_time(deadline)
        probabilities = model.probabilities(history.lagged(chunk, protocol["neural"]["lags"]), history.y(chunk))
        total += probabilities.to(torch.float64).sum(0)
    return (total / len(calibration_indices)).to(torch.float32)


@torch.no_grad()
def forecast_chunks(model, history, indices, graph, protocol, draws, seed, deadline):
    if model.name in {"tnpar", "direct"}:
        generator = torch.Generator(device=history.device).manual_seed(seed)
        graphs = (torch.rand((draws,) + tuple(graph.shape), device=history.device, generator=generator) < graph).to(torch.float32)
        batch = min(1024, protocol["neural"]["test_batch_pairs"])
    else:
        graphs, batch = None, protocol["neural"]["test_batch_pairs"]
    for index in indices.split(batch):
        check_time(deadline)
        _, nodes = history.indices(index)
        x, y = input_for(model, history, index, protocol), history.y(index)
        if graphs is None:
            mean = model.means(x, nodes, graph)
            mass = log_probability(y, mean)
        else:
            logged = torch.log1p(x)
            output = []
            for target, decoder in enumerate(model.decoders):
                gated = logged[None] * graphs[..., target][:, None, :, :, None]
                effect = decoder(gated.flatten(2)).squeeze(-1)
                output.append(torch.nn.functional.softplus(model.baseline_raw[nodes, target][None] + effect) + 1e-8)
            samples = torch.stack(output, -1)
            mean = samples.to(torch.float64).mean(0)
            mass = torch.logsumexp(log_probability(y[None], samples.to(torch.float64)), 0) - math_log(draws)
        if not bool(torch.isfinite(mean).all()) or not bool((mean > 0).all()) or not bool(torch.isfinite(mass).all()):
            raise FloatingPointError("Invalid forecast")
        yield index, y.to(torch.float64), mean.to(torch.float64), mass.to(torch.float64)


def math_log(value):
    return float(np.log(value))


@torch.no_grad()
def validate(model, history, indices, graph, protocol, deadline, seed):
    sums, count = np.zeros(2, dtype=float), 0
    for _, y, mean, mass in forecast_chunks(model, history, indices, graph, protocol, protocol["neural"]["validation_graph_draws"], seed, deadline):
        sums += [float((mean - y * mean.log()).sum()), float(-mass.sum())]
        count += y.numel()
    return {"mean_score": float(sums[0] / count), "predictive_NLL": float(sums[1] / count), "counts": count}


def fit_candidate(name, history, metadata, protocol, path, lr, regularization, seed, deadline):
    path.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    if history.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_start, train_end = metadata["train"]
    baseline = history.counts[train_start:train_end].mean(0).clamp_min(1e-5)
    model = (S2Rectangle(baseline, protocol["s2"], regularization) if name == "s2" else
             NeuralGraph(name, protocol["neural"], baseline, history.H)).to(history.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device=history.device).manual_seed(seed + 8000)
    calibration = pair_sample(metadata["train"], history.N, protocol["neural"]["graph_calibration_TRAIN_pairs_max"], metadata["seed"] + 7100, history.device)
    validation = pair_sample(metadata["validation"], history.N, protocol["neural"]["validation_pairs_max"], metadata["seed"] + 7200, history.device)
    settings = protocol["neural"]
    maximum = protocol["s2"]["maximum_updates"] if name == "s2" else settings["maximum_updates"]
    best, best_update, waited, journal = float("inf"), 0, 0, []
    start = time.perf_counter()
    train_seconds, validation_seconds = 0., 0.
    n_train_counts = (train_end - train_start) * history.N * history.P
    accumulated = []
    termination = "maximum_updates"
    for update in range(1, maximum + 1):
        check_time(deadline)
        begin = time.perf_counter()
        model.train()
        index = torch.randint(train_start * history.N, train_end * history.N, (settings["batch_pairs"],), generator=generator, device=history.device)
        _, nodes = history.indices(index)
        optimizer.zero_grad(set_to_none=True)
        loss, components = model.objective(input_for(model, history, index, protocol), nodes, history.y(index), n_train_counts)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        train_seconds += time.perf_counter() - begin
        accumulated.append({"objective": float(loss.detach()), **components})
        if update % settings["validation_every_updates"] and update != maximum:
            continue
        model.eval()
        begin = time.perf_counter()
        graph = freeze(model, history, calibration, protocol, deadline)
        result = validate(model, history, validation, graph, protocol, deadline, metadata["seed"] + 7300)
        validation_seconds += time.perf_counter() - begin
        row = {"update": update, "validation": result,
               "TRAIN_components": {key: float(np.mean([sample[key] for sample in accumulated])) for key in accumulated[0]},
               "sampled_TRAIN_pairs_seen": update * settings["batch_pairs"], "elapsed_seconds": time.perf_counter() - start}
        accumulated = []
        if result["mean_score"] < best:
            best, best_update, waited = result["mean_score"], update, 0
            torch.save({"state_dict": cpu_state(model), "frozen_TRAIN_graph": graph.cpu(), "update": update,
                        "validation_mean_score": best}, path / "best.pt")
        else:
            waited += 1
        journal.append(row)
        save(path / f"validation_{update:05d}.json", row)
        if update >= settings["minimum_updates"] and waited >= settings["patience_checks"]:
            termination = "validation_patience"
            break
    report = {"model": name, "learning_rate": lr, "regularization": regularization, "seed": seed,
              "best_update": best_update, "best_validation_mean_score": best, "updates_completed": update,
              "termination": termination, "train_seconds": train_seconds, "validation_seconds": validation_seconds,
              "elapsed_seconds": time.perf_counter() - start, "parameters": sum(p.numel() for p in model.parameters()),
              "history": journal, "true_graph_or_TEST_used_for_fit": False,
              "validation_pairs": len(validation), "graph_calibration_TRAIN_pairs": len(calibration)}
    save(path / "fit.json", report)
    del optimizer, model
    if history.device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def restore(name, history, metadata, protocol, fit, path):
    baseline = history.counts[slice(*metadata["train"])].mean(0).clamp_min(1e-5)
    model = (S2Rectangle(baseline, protocol["s2"], fit["regularization"]) if name == "s2" else
             NeuralGraph(name, protocol["neural"], baseline, history.H)).to(history.device)
    checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if checkpoint["update"] != fit["best_update"] or checkpoint["validation_mean_score"] != fit["best_validation_mean_score"]:
        raise ValueError("Selected checkpoint differs from validation selection")
    return model, checkpoint["frozen_TRAIN_graph"].to(history.device)


def graph_result(model, graph, truth, protocol):
    if model.name == "thp":
        scores = model.alpha.max(0).values.cpu().numpy()
        binary = model.graph.copy()
        np.fill_diagonal(binary, 0)
        result = graph_metrics(truth, binary.astype(float), .5)
        from evaluation import average_precision
        off = ~np.eye(len(truth), dtype=bool)
        result["average_precision"] = average_precision(truth[off], scores[off])
        result["threshold_fixed_before_test"] = "BIC_searched_binary_adjacency"
        result["score_kind"] = "searched_excitation_weight_not_probability"
    else:
        scores = (graph if graph.ndim == 2 else graph.max(0).values).detach().cpu().numpy().copy()
        threshold = protocol["s2"]["structure_threshold_per_bin_excitation"] if model.name == "s2" else protocol["neural"]["structure_threshold"]
        result = graph_metrics(truth, scores, threshold)
        binary = (scores >= threshold).astype(np.int8)
        np.fill_diagonal(binary, 0)
        result["score_kind"] = "excitation_weight" if model.name == "s2" else "continuous_gate" if model.name == "deterministic" else "posterior_score"
    np.fill_diagonal(scores, 0)
    result["is_DAG"] = nx.is_directed_acyclic_graph(nx.DiGraph(binary))
    result["predicted_edges"] = int(binary.sum())
    return result, scores, binary


@torch.no_grad()
def evaluate(model, history, metadata, protocol, graph, truth, path, deadline):
    path.mkdir(parents=True, exist_ok=False)
    intervals = metadata["test"]
    indices = torch.arange(intervals[0] * history.N, intervals[1] * history.N, device=history.device)
    total = np.zeros(5, dtype=float)
    count, per_time, means_saved, mass_saved = 0, np.zeros((intervals[1] - intervals[0], 5)), [], []
    begin = time.perf_counter()
    for index, y, mean, mass in forecast_chunks(model, history, indices, graph, protocol, protocol["neural"]["test_graph_draws"], metadata["seed"] + 7400, deadline):
        error = mean - y
        mean_score = mean - y * mean.log()
        stats = torch.stack([mean_score, -log_probability(y, mean), -mass, error.abs(), error.square()], -1).sum(1)
        array = stats.cpu().numpy()
        total += array.sum(0)
        times = (index // history.N - intervals[0]).cpu().numpy()
        np.add.at(per_time, times, array)
        count += y.numel()
        means_saved.append(mean.cpu().numpy().astype(np.float32))
        mass_saved.append(mass.cpu().numpy().astype(np.float32))
    metric = {"Poisson_mean_score": float(total[0] / count), "mean_Poisson_count_NLL": float(total[1] / count),
              "predictive_count_NLL": float(total[2] / count), "count_MAE": float(total[3] / count),
              "count_RMSE": float(np.sqrt(total[4] / count)), "counts": count,
              "latent_lambda_RMSE": None, "lambda_truth_status": "not_observed_not_replaced_by_counts_over_delta"}
    structure, scores, binary = graph_result(model, graph, truth, protocol)
    report = {"model": model.name, "dataset": metadata["name"], "metrics": metric, "structure": structure,
              "elapsed_seconds": time.perf_counter() - begin, "graph_labels_role": metadata["graph_labels"],
              "physical_seconds_per_bin": metadata["delta_seconds"], "TEST_used_to_change_model": False,
              "authors_exact_reproduction": False}
    shape = (intervals[1] - intervals[0], history.N, history.P)
    np.savez_compressed(path / "predictions.npz", mean_count=np.concatenate(means_saved).reshape(shape),
                        predictive_log_mass=np.concatenate(mass_saved).reshape(shape),
                        graph_scores=scores, binary_graph=binary,
                        per_time_mean_metrics=per_time / (history.N * history.P))
    save(path / "result.json", report)
    return report


def dataset_run(dataset_path, output, protocol, device, deadline):
    output.mkdir(parents=True, exist_ok=False)
    observations, metadata = load(dataset_path)
    history = CountHistory(observations, device, deadline)
    del observations
    save(output / "dataset.json", metadata)
    baseline = history.counts[slice(*metadata["train"])].mean(0)
    coverage = {"TRAIN_events_by_type": history.counts[slice(*metadata["train"])].sum((0, 1)).cpu().numpy().astype(np.int64).tolist(),
                "TRAIN_unobserved_node_type_cells": int((baseline == 0).sum()),
                "TRAIN_total_node_type_cells": int(baseline.numel())}
    save(output / "TRAIN_coverage.json", coverage)
    selections, reports = {}, []
    thp_model = None
    start = time.perf_counter()
    for name in protocol["models"]:
        if name == "thp":
            path = output / "fits/thp"
            path.mkdir(parents=True, exist_ok=False)
            begin = time.perf_counter()
            thp_model = THP_EM(history, metadata["train"], protocol["thp"], deadline)
            fit = thp_model.fit(lambda row: save(path / f"move_{row['move']:03d}.json", row))
            fit.update(model="thp", elapsed_seconds=time.perf_counter() - begin, best_update=fit["graph_moves"], selection="TRAIN_BIC_not_TEST")
            graph = freeze(thp_model, history, None, protocol, deadline)
            validation = pair_sample(metadata["validation"], history.N, protocol["neural"]["validation_pairs_max"], metadata["seed"] + 7200, history.device)
            fit["validation_report_only"] = validate(thp_model, history, validation, graph, protocol, deadline, metadata["seed"] + 7300)
            save(path / "fit.json", fit)
            torch.save(cpu_state(thp_model), path / "best.pt")
            selections[name] = {"fit_path": "fits/thp", "selection": "TRAIN_BIC", "report": fit}
            reports.append(fit)
            continue
        lrs = protocol["s2"]["learning_rates"] if name == "s2" else protocol["neural"]["learning_rates"]
        penalties = protocol["s2"]["regularization_candidates"] if name == "s2" else [None]
        candidates = []
        for lr in lrs:
            for penalty in penalties:
                trial = len(candidates)
                path = output / "fits" / f"{name}_trial{trial}"
                fit = fit_candidate(name, history, metadata, protocol, path, lr, penalty, metadata["seed"] + trial, deadline)
                candidate = {"fit_path": str(path.relative_to(output)).replace("\\", "/"), "trial": trial, "report": fit}
                candidates.append(candidate)
                reports.append(fit)
        selected = min(candidates, key=lambda row: (row["report"]["best_validation_mean_score"], row["trial"]))
        selections[name] = selected
        save(output / f"{name}_selection.json", {"selected": selected, "candidates": candidates, "test_or_graph_used": False})
    save(output / "validation_selection.json", selections)
    # The reference graph is first read here, after every fit and selection.
    truth = load_labels(dataset_path, metadata)
    results = []
    for name in protocol["models"]:
        selected = selections[name]
        if name == "thp":
            model, graph = thp_model, freeze(thp_model, history, None, protocol, deadline)
        else:
            model, graph = restore(name, history, metadata, protocol, selected["report"], output / selected["fit_path"])
        result = evaluate(model, history, metadata, protocol, graph, truth, output / "evaluations" / name, deadline)
        result["selected_fit"] = selected["fit_path"]
        results.append(result)
        if name != "thp":
            del model
    test_y = history.counts[slice(*metadata["test"])].to(torch.float64)
    constant = baseline.to(torch.float64).clamp_min(1e-8)[None]
    constant_result = {"mean_score": float((constant - test_y * constant.log()).mean()),
                       "count_NLL": float(-log_probability(test_y, constant).mean()),
                       "count_RMSE": float((constant - test_y).square().mean().sqrt()),
                       "role": "TRAIN_only_diagnostic_not_sixth_architecture"}
    save(output / "TRAIN_constant_baseline.json", constant_result)
    report = {"status": "complete", "dataset": metadata["name"], "fits_completed": len(reports),
              "selected_test_evaluations_completed": len(results), "elapsed_seconds": time.perf_counter() - start,
              "results": results, "constant_baseline": constant_result, "independent_confirmatory_study": False}
    save(output / "completion.json", report)
    return report


def run_worker(data_root, output_parent, dataset_names, protocol_path, device, max_seconds):
    protocol = read(protocol_path)
    start = time.monotonic()
    deadline = start + max_seconds
    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=False)
    environment = {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
                   "visible_cuda_devices": torch.cuda.device_count(),
                   "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    save(output_parent / "environment.json", environment)
    torch.set_num_threads(2)
    completions = []
    try:
        for name in dataset_names:
            report = dataset_run(Path(data_root) / name, output_parent / name, protocol, device, deadline)
            completions.append({"dataset": name, "fits": report["fits_completed"], "tests": report["selected_test_evaluations_completed"]})
        save(output_parent / "worker_completion.json", {"status": "complete", "datasets": completions,
                                                        "elapsed_seconds": time.monotonic() - start})
    except Exception as error:
        save(output_parent / "worker_completion.json", {"status": "incomplete", "datasets_completed": completions,
                                                        "failure_type": type(error).__name__, "automatic_retry": False,
                                                        "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--datasets", required=True, nargs="+")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-seconds", type=int, default=13800)
    args = parser.parse_args()
    run_worker(args.data_root, args.output, args.datasets, args.protocol, torch.device(args.device), args.max_seconds)
