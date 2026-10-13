"""Exploratory frozen-model analysis. No fitting, GPU, threshold or data tuning.

VAL uses the original 25,000 uniformly sampled node-time pairs and all 24
targets. TEST is used only for previously saved forecast error decomposition.
Counterfactual removals measure fixed-model reliance, not causal effects.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
from torch.nn import functional as F

from ig_numpy import integrated_gradients, forward as numpy_forward


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "xai_20261013"
GROUPS = ["all_history", "own_device_same_type", "neighbor_same_type", "cross_type",
          "hop_0", "hop_1", "hop_2", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
          "reference_cross_type", "nonreference_cross_type"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_sample(interval, nodes, maximum, seed):
    values = np.arange(interval[0] * nodes, interval[1] * nodes)
    if len(values) > maximum:
        values = np.sort(np.random.default_rng(seed).choice(values, maximum, replace=False))
    return values


def gather(counts, hop_masks, pairs, lags=5):
    """Exact integer neighbor sums; avoid materializing the 1.1 GB pooled grid."""
    nodes = pairs % counts.shape[1]
    times = pairs // counts.shape[1]
    x = np.zeros((len(pairs), len(hop_masks), counts.shape[2], lags), dtype=np.float32)
    for node in np.unique(nodes):
        rows = np.flatnonzero(nodes == node)
        for hop, masks in enumerate(hop_masks):
            neighbors = np.flatnonzero(masks[node])
            if len(neighbors):
                for lag in range(lags):
                    x[rows, hop, :, lag] = counts[times[rows, None] - lag - 1,
                                                        neighbors[None, :], :].sum(1)
    return x, nodes, counts[times, nodes]


def feature_mask(group, target, H, P, L, truth):
    mask = np.zeros((H, P, L), dtype=bool)
    if group == "all_history":
        mask[:] = True
    elif group == "own_device_same_type":
        mask[0, target] = True
    elif group == "neighbor_same_type":
        mask[1:, target] = True
    elif group == "cross_type":
        mask[:, np.arange(P) != target] = True
    elif group.startswith("hop_"):
        mask[int(group.rsplit("_", 1)[1])] = True
    elif group.startswith("lag_"):
        mask[:, :, int(group.rsplit("_", 1)[1]) - 1] = True
    elif group in {"reference_cross_type", "nonreference_cross_type"}:
        source = truth[:, target].astype(bool)
        if group == "nonreference_cross_type":
            source = ~source
        source[target] = False
        mask[:, source] = True
    else:
        raise ValueError("Unknown predefined feature group")
    return mask


def poisson_nll(y, mean):
    mean = mean.double().clamp_min(1e-8)
    y = y.double()
    return mean - y * mean.log() + torch.lgamma(y + 1)


def metrics(y, mean):
    y, mean = y.double(), mean.double()
    nll, error = poisson_nll(y, mean), mean - y
    positive = y > 0
    return {"NLL": float(nll.mean()), "MAE": float(error.abs().mean()),
            "RMSE": float(error.square().mean().sqrt()), "cells": y.numel(),
            "positive_cells": int(positive.sum()),
            "positive_NLL": float(nll[positive].mean()) if positive.any() else None,
            "zero_NLL": float(nll[~positive].mean()) if (~positive).any() else None,
            "mean_predicted_count": float(mean.mean()),
            "positive_predicted_count": float(mean[positive].mean()) if positive.any() else None}


def load_record(repetition, model_name, models_module, protocol, N, P):
    if repetition == 0:
        metadata_root = OUT / "recovered_original_metadata" / "comparison_v2_results" / "worker_1" / "real_24V_439N_Microwave"
        models_root = OUT / "recovered_original_models" / "comparison_v2_results" / "worker_1" / "real_24V_439N_Microwave"
    else:
        metadata_root = OUT / "recovered_repeated_metadata" / "minimum_repeat_results" / "worker_1" / "real_24V_439N_Microwave" / f"repeat_{repetition}"
        models_root = OUT / "recovered_repeated_models" / "minimum_repeat_results" / "worker_1" / "real_24V_439N_Microwave" / f"repeat_{repetition}"
    selected = read(metadata_root / "validation_selection.json")[model_name]
    fit = read(models_root / selected["fit_path"] / "fit.json")
    checkpoint_path = models_root / selected["fit_path"] / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["update"] == selected["report"]["best_update"] == fit["best_update"]
    assert checkpoint["validation_mean_score"] == selected["report"]["best_validation_mean_score"] == fit["best_validation_mean_score"]
    dummy = torch.ones((N, P))
    model = (models_module.S2Rectangle(dummy, protocol["s2"], fit["regularization"]) if model_name == "s2"
             else models_module.NeuralGraph(model_name, protocol["neural"], dummy, 3))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    graph = checkpoint["frozen_TRAIN_graph"]
    if model_name == "deterministic":
        assert torch.allclose(graph, model.probabilities(), atol=2e-7, rtol=2e-6)
    else:
        assert torch.allclose(graph, model.weights(), atol=2e-7, rtol=2e-6)
    return model, graph, models_root / "evaluations" / model_name, {
        "model": model_name, "repetition": repetition, "trial": selected["trial"],
        "update": checkpoint["update"], "checkpoint_sha256": sha(checkpoint_path)}


def decoder_tail(decoder, first_pre, baseline):
    effect = decoder[4](torch.tanh(decoder[2](torch.tanh(first_pre)))).squeeze(-1)
    return F.softplus(baseline + effect) + 1e-8


@torch.no_grad()
def frozen_analysis(model, graph, x, nodes, y, truth):
    B, H, P, L = x.shape
    all_means = torch.empty_like(y)
    removed_means = {group: torch.empty_like(y) for group in GROUPS}
    edges = np.zeros((P, P), dtype=float)
    base_per_target = np.zeros(P)
    if model.name == "deterministic":
        logged = torch.log1p(x).flatten(1)
        for target, decoder in enumerate(model.decoders):
            gate = graph[..., target, None].expand(H, P, L).reshape(-1)
            gated = logged * gate
            baseline = model.baseline_raw[nodes, target]
            pre = decoder[0](gated)
            mean = decoder_tail(decoder, pre, baseline)
            all_means[:, target] = mean
            original_loss = poisson_nll(y[:, target], mean).mean()
            base_per_target[target] = float(original_loss)
            for group in GROUPS:
                mask = torch.as_tensor(feature_mask(group, target, H, P, L, truth).ravel())
                contribution = F.linear(gated[:, mask], decoder[0].weight[:, mask])
                removed_means[group][:, target] = decoder_tail(decoder, pre - contribution, baseline)
            for source in range(P):
                mask = torch.zeros((H, P, L), dtype=torch.bool)
                mask[:, source] = True
                mask = mask.flatten()
                contribution = F.linear(gated[:, mask], decoder[0].weight[:, mask])
                new_mean = decoder_tail(decoder, pre - contribution, baseline)
                edges[source, target] = float(poisson_nll(y[:, target], new_mean).mean() - original_loss)
    else:
        rectangle = x.sum((1, 3))
        weights = model.weights()
        baseline = F.softplus(model.baseline_raw[nodes])
        all_means = baseline + rectangle @ weights + 1e-8
        for target in range(P):
            original_loss = poisson_nll(y[:, target], all_means[:, target]).mean()
            base_per_target[target] = float(original_loss)
            for group in GROUPS:
                mask = torch.as_tensor(feature_mask(group, target, H, P, L, truth))
                remaining_rectangle = (x * ~mask[None]).sum((1, 3))
                removed_means[group][:, target] = baseline[:, target] + remaining_rectangle @ weights[:, target] + 1e-8
            for source in range(P):
                remaining = rectangle.clone()
                remaining[:, source] = 0
                new_mean = baseline[:, target] + remaining @ weights[:, target] + 1e-8
                edges[source, target] = float(poisson_nll(y[:, target], new_mean).mean() - original_loss)
    base = metrics(y, all_means)
    group_rows = []
    for group, means in removed_means.items():
        row = metrics(y, means)
        row.update(group=group, delta_NLL=row["NLL"] - base["NLL"],
                   delta_MAE=row["MAE"] - base["MAE"], delta_RMSE=row["RMSE"] - base["RMSE"])
        group_rows.append(row)
    return base, group_rows, edges, all_means, base_per_target


def ranks(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        result[order[begin:end]] = (begin + end - 1) / 2
        begin = end
    return result


def spearman(a, b):
    a, b = ranks(np.asarray(a)), ranks(np.asarray(b))
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def edge_summary(graph, importance, truth):
    scores = graph.numpy()
    if scores.ndim == 3:
        scores = scores.max(0)
    off = ~np.eye(len(truth), dtype=bool)
    rows = []
    for source, target in zip(*np.where(off)):
        rows.append({"source": int(source), "target": int(target), "reference": int(truth[source, target]),
                     "original_graph_score": float(scores[source, target]),
                     "delta_target_NLL": float(importance[source, target]),
                     "delta_overall_NLL": float(importance[source, target] / len(truth))})
    groups = {}
    for reference in (0, 1):
        values = importance[off & (truth == reference)]
        groups[str(reference)] = {"edges": len(values), "mean_delta_target_NLL": float(values.mean()),
                                 "median_delta_target_NLL": float(np.median(values)),
                                 "positive_fraction": float((values > 1e-12).mean()),
                                 "negative_fraction": float((values < -1e-12).mean())}
    return {"score_importance_spearman": spearman(scores[off], importance[off]),
            "groups_by_reference_label": groups, "rows": rows,
            "interpretation": "fixed-model removal reliance; not Granger causality; score/AP unchanged"}


def saved_test_decomposition(path, counts, metadata):
    start, end = metadata["test"]
    y = counts[start:end]
    with np.load(path / "predictions.npz", allow_pickle=False) as archive:
        mean = archive["mean_count"]
        mass = archive["predictive_log_mass"]
    assert mean.shape == y.shape == mass.shape
    rows = []
    for target in range(y.shape[-1]):
        current_y, current_mean, current_nll = y[..., target], mean[..., target], -mass[..., target]
        error = current_mean.astype(np.float64) - current_y
        positive = current_y > 0
        rows.append({"target": target, "cells": current_y.size, "events": int(current_y.sum()),
                     "positive_cells": int(positive.sum()), "NLL_sum": float(current_nll.sum(dtype=np.float64)),
                     "NLL": float(current_nll.mean(dtype=np.float64)), "MAE": float(np.abs(error).mean()),
                     "RMSE": float(np.sqrt(np.square(error).mean())),
                     "positive_NLL": float(current_nll[positive].mean(dtype=np.float64)) if positive.any() else None,
                     "zero_NLL": float(current_nll[~positive].mean(dtype=np.float64)) if (~positive).any() else None,
                     "predicted_events": float(current_mean.sum(dtype=np.float64))})
    time_nll = -mass.sum((1, 2), dtype=np.float64) / (y.shape[1] * y.shape[2])
    total = float(-mass.mean(dtype=np.float64))
    assert abs(total - read(path / "result.json")["metrics"]["predictive_count_NLL"]) < 2e-8
    return rows, time_nll


@torch.no_grad()
def checkpoint_prediction_check(model, graph, x, nodes, pairs, saved_path, metadata, N):
    if model.name == "deterministic":
        actual = model.means(x, nodes, graph).numpy()
    else:
        actual = model.means(x.sum((1, 3)), nodes).numpy()
    with np.load(saved_path / "predictions.npz", allow_pickle=False) as archive:
        saved = archive["mean_count"][pairs // N - metadata["test"][0], pairs % N]
    difference = np.abs(actual.astype(float) - saved)
    if difference.max() > 2e-5 or not np.allclose(actual, saved, atol=2e-6, rtol=2e-4):
        raise ValueError("CPU checkpoint forecasts differ from saved original forecasts")
    return {"pairs": len(pairs), "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()), "passed": True}


def decoder_numpy_parameters(model, graph, target, nodes):
    decoder = model.decoders[target]
    gate = graph[..., target, None].expand(3, model.p, model.config["lags"]).reshape(-1).numpy()
    return {"W0": decoder[0].weight.numpy(), "b0": decoder[0].bias.numpy(),
            "W2": decoder[2].weight.numpy(), "b2": decoder[2].bias.numpy(),
            "W4": decoder[4].weight.numpy(), "b4": decoder[4].bias.numpy(),
            "gate_flat": gate, "baseline_raw": model.baseline_raw[nodes, target].numpy()}


def attribution_analysis(model, graph, s2_model, x, nodes, y, train_mean_logged, truth):
    logged = np.log1p(x).reshape(len(x), -1).astype(np.float64)
    H, P, L = x.shape[1:]
    outputs = {}
    for baseline_name, baseline_input in [("zero_history", np.zeros(logged.shape[1])),
                                          ("TRAIN_mean_logged_history", train_mean_logged)]:
        group_absolute = {group: 0.0 for group in GROUPS}
        group_signed = {group: 0.0 for group in GROUPS}
        source_target = np.zeros((P, P))
        hop_absolute, lag_absolute = np.zeros(H), np.zeros(L)
        positive_mass, negative_mass, residuals, changes = 0.0, 0.0, [], []
        convergence_abs = 0.0
        for target in range(P):
            kwargs = decoder_numpy_parameters(model, graph, target, torch.as_tensor(nodes))
            result = integrated_gradients(logged, **kwargs, baseline_input=baseline_input, steps=64)
            check = integrated_gradients(logged, **kwargs, baseline_input=baseline_input, steps=128)
            attr = check["attributions"].reshape(-1, H, P, L)
            absolute = np.abs(attr)
            convergence_abs += float(np.abs(check["attributions"] - result["attributions"]).sum())
            source_target[:, target] = absolute.sum((0, 1, 3))
            hop_absolute += absolute.sum((0, 2, 3))
            lag_absolute += absolute.sum((0, 1, 2))
            positive_mass += float(attr[attr > 0].sum())
            negative_mass += float(-attr[attr < 0].sum())
            residuals.extend(check["completeness_residual"].tolist())
            changes.extend(check["output_change"].tolist())
            for group in GROUPS:
                mask = feature_mask(group, target, H, P, L, truth)
                group_absolute[group] += float(absolute[:, mask].sum())
                group_signed[group] += float(attr[:, mask].sum())
            expected = model.means(torch.as_tensor(x), torch.as_tensor(nodes), graph)[:, target].numpy()
            if not np.allclose(expected, check["input_mean"], atol=2e-6, rtol=2e-4):
                raise ValueError("NumPy IG forward differs from original PyTorch decoder")
        total = positive_mass + negative_mass
        outputs[baseline_name] = {
            "absolute_attribution_sum": total,
            "positive_absolute_share": positive_mass / total if total else None,
            "negative_absolute_share": negative_mass / total if total else None,
            "groups": [{"group": group, "absolute_share": group_absolute[group] / total if total else None,
                        "signed_attribution_sum": group_signed[group]} for group in GROUPS],
            "hop_absolute_shares": (hop_absolute / total).tolist(),
            "lag_absolute_shares": (lag_absolute / total).tolist(),
            "source_target_absolute_share": (source_target / total).tolist(),
            "max_abs_completeness_residual": float(np.max(np.abs(residuals))),
            "mean_abs_completeness_residual": float(np.mean(np.abs(residuals))),
            "completeness_absolute_relative_to_output_changes": float(np.sum(np.abs(residuals)) / max(1e-12, np.sum(np.abs(changes)))),
            "64_to_128_step_L1_relative_change": convergence_abs / total if total else None,
            "groups_overlap_do_not_sum_all_group_shares": True}
    weights = s2_model.weights().numpy()
    exact = x[..., None] * weights[None, None, :, None, :]
    denominator = float(exact.sum())
    outputs["S2_exact_positive_linear_terms_zero_history"] = {
        "negative_absolute_share": 0.0,
        "hop_absolute_shares": (exact.sum((0, 2, 3, 4)) / denominator).tolist() if denominator else None,
        "lag_absolute_shares": (exact.sum((0, 1, 2, 4)) / denominator).tolist() if denominator else None,
        "own_type_share": 0.0, "linear_history_effect_sum": denominator,
        "not_directly_same_attribution_scale_as_IG": True}
    return outputs


def reparameterization_check(model, graph, x, nodes, truth):
    """Construct exact function-preserving gate/first-layer compensation."""
    original = model.means(x, nodes, graph).detach().numpy()
    off = ~np.eye(model.p, dtype=bool)
    scores = graph.max(0).values.detach().numpy()
    # Fixed, label-blind factors; not chosen to improve prediction or AP.
    scales = np.random.default_rng(96101309).uniform(.25, .95, size=(3, model.p, model.p)).astype(np.float32)
    changed_graph = graph * torch.as_tensor(scales)
    changed = []
    for target, decoder in enumerate(model.decoders):
        scale = torch.as_tensor(scales[..., target, None]).expand(3, model.p, model.config["lags"]).reshape(-1)
        changed.append(decoder[0].weight.detach().clone())
        decoder[0].weight.copy_(decoder[0].weight / scale)
    altered = model.means(x, nodes, changed_graph).detach().numpy()
    for decoder, weights in zip(model.decoders, changed):
        decoder[0].weight.copy_(weights)
    difference = np.abs(original.astype(float) - altered)
    altered_scores = changed_graph.max(0).values.numpy()
    return {"label_blind_in_memory_transform_only": True,
            "original_checkpoint_unchanged": True, "new_predictions_not_used_for_primary_evaluation": True,
            "max_abs_prediction_difference": float(difference.max()),
            "mean_abs_prediction_difference": float(difference.mean()),
            "raw_gate_rank_spearman_before_after": spearman(scores[off], altered_scores[off]),
            "interpretation": "raw gate rankings are not uniquely determined by the predictive function; does not establish why model won"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["analysis", "attribution"], default="analysis")
    args = parser.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    source = OUT / "recovered_original_metadata" / "comparison_v2_source" / "models_v2.py"
    spec = importlib.util.spec_from_file_location("original_models", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_root = OUT / "recovered_original_metadata" / "comparison_v2_data" / "real_24V_439N_Microwave"
    metadata = read(data_root / "dataset.json")
    protocol = read(OUT / "recovered_original_metadata" / "comparison_v2_results" / "protocol.json")
    if args.phase == "analysis":
        audit = read(OUT / "recovery_audit.json")
        if audit.get("status") != "verified_fixed_artifact_recovery":
            raise ValueError("Recovery audit has not passed")
        save_new(OUT / "analysis_protocol.json", {
            "purpose": "exploratory_frozen_model_reliance_not_causal_confirmation",
            "models": ["deterministic", "s2"], "fit_repetitions": [0, 1, 2],
            "new_training": False, "new_GPU_runs": 0, "model_or_threshold_selection": False,
            "VAL_sample": {"pairs": 25000, "seed": metadata["seed"] + 7200,
                           "same_as_original_validation_selection": True, "all_targets": 24},
            "TEST_role": "saved_original_forecasts_descriptive_decomposition_only",
            "predefined_removal_groups": GROUPS, "all_source_target_removals": True,
            "IG": {"fit": 0, "VAL_uniform_pairs": 128, "VAL_positive_pair_supplement": 128,
                   "case_seed": 96101301, "baselines": ["zero_history", "TRAIN_mean_logged_history"],
                   "TRAIN_reference_pairs": 10000, "TRAIN_reference_seed": 96101302,
                   "integration_steps": [64, 128], "selection_not_based_on_prediction_success": True},
            "immutable_sources_sha256": {"models_v2.py": sha(source),
                                          "observations.npz": sha(data_root / "observations.npz")}})
    elif not (OUT / "analysis_protocol.json").is_file():
        raise ValueError("Registered exploratory protocol missing")
    print(json.dumps({"stage": "loading_verified_real_data", "phase": args.phase}), flush=True)
    with np.load(data_root / "observations.npz", allow_pickle=False) as archive:
        counts, hop_masks = archive["counts"], archive["hop_masks"]
    with np.load(data_root / "evaluation_labels.npz", allow_pickle=False) as archive:
        truth = archive["graph_source_target"]
    N, P = counts.shape[1:]
    validation_pairs = pair_sample(metadata["validation"], N, 25000, metadata["seed"] + 7200)
    if args.phase == "analysis":
        x_np, nodes_np, y_np = gather(counts, hop_masks, validation_pairs)
        x, nodes, y = torch.as_tensor(x_np), torch.as_tensor(nodes_np), torch.as_tensor(y_np, dtype=torch.float32)
        test_pairs = pair_sample(metadata["test"], N, 1024, 96101303)
        test_x, test_nodes, _ = gather(counts, hop_masks, test_pairs)
        test_x, test_nodes = torch.as_tensor(test_x), torch.as_tensor(test_nodes)
        rows, test_rows, test_time, edges_saved = [], [], {}, {}
        for repetition in range(3):
            for name in ["deterministic", "s2"]:
                model, graph, saved_path, record = load_record(repetition, name, module, protocol, N, P)
                verification = checkpoint_prediction_check(model, graph, test_x, test_nodes, test_pairs, saved_path, metadata, N)
                print(json.dumps({"stage": "VAL_frozen_removals", **record}), flush=True)
                base, groups, importance, mean, per_target = frozen_analysis(model, graph, x, nodes, y, truth)
                if name == "s2":
                    assert abs(next(row for row in groups if row["group"] == "own_device_same_type")["delta_NLL"]) < 1e-8
                result = {**record, "checkpoint_prediction_check": verification, "original_VAL": base,
                          "removed_groups": groups, "edge_removal": edge_summary(graph, importance, truth),
                          "original_VAL_target_NLL": per_target.tolist()}
                save_new(OUT / f"frozen_{name}_repeat{repetition}.json", result)
                rows.append(result)
                edges_saved[f"{name}_repeat{repetition}"] = importance
                current_test, current_time = saved_test_decomposition(saved_path, counts, metadata)
                test_rows.append({"model": name, "repetition": repetition, "by_target": current_test})
                test_time[f"{name}_repeat{repetition}"] = current_time
                del model, graph
        save_new(OUT / "frozen_group_summary.json", {"VAL": rows, "elapsed_seconds": time.perf_counter() - start,
                   "new_GPU_runs": 0, "new_fits": 0})
        save_new(OUT / "TEST_type_decomposition.json", {"rows": test_rows, "post_result_descriptive_only": True})
        with (OUT / "edge_importance_and_TEST_time.npz").open("xb") as stream:
            np.savez_compressed(stream, **edges_saved, **{f"TEST_time_{k}": v for k, v in test_time.items()})
    else:
        rng = np.random.default_rng(96101301)
        uniform = rng.choice(validation_pairs, 128, replace=False)
        positive_pairs = np.flatnonzero((counts[slice(*metadata["validation"])] > 0).any(-1).ravel()) + metadata["validation"][0] * N
        supplement = rng.choice(positive_pairs, min(128, len(positive_pairs)), replace=False)
        cases = np.unique(np.r_[uniform, supplement])
        case_x, case_nodes, case_y = gather(counts, hop_masks, cases)
        train_pairs = pair_sample(metadata["train"], N, 10000, 96101302)
        train_x, _, _ = gather(counts, hop_masks, train_pairs)
        train_mean_logged = np.log1p(train_x).reshape(len(train_x), -1).mean(0, dtype=np.float64)
        del train_x
        model, graph, _, record = load_record(0, "deterministic", module, protocol, N, P)
        s2, _, _, _ = load_record(0, "s2", module, protocol, N, P)
        print(json.dumps({"stage": "IG_two_reference_inputs", "cases": len(cases)}), flush=True)
        result = attribution_analysis(model, graph, s2, case_x, case_nodes, case_y, train_mean_logged, truth)
        compensation = reparameterization_check(model, graph, torch.as_tensor(case_x), torch.as_tensor(case_nodes), truth)
        save_new(OUT / "input_attribution.json", {"selected_model": record, "VAL_case_pairs": cases.tolist(),
                  "cases": len(cases), "cases_containing_observed_event": int((case_y > 0).any(-1).sum()),
                  "outcome_enriched_sample_not_population_weighted": True,
                  "analysis": result, "gate_decoder_compensation": compensation,
                  "elapsed_seconds": time.perf_counter() - start, "new_GPU_runs": 0})
    print(json.dumps({"status": "completed", "phase": args.phase, "seconds": time.perf_counter() - start,
                      "new_GPU_runs": 0, "new_fits": 0}), flush=True)


if __name__ == "__main__":
    main()
