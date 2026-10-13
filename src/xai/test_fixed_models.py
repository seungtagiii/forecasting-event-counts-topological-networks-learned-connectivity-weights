"""Small CPU checks of frozen removal, axis conventions and S2 contributions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import analyze_fixed_models as analysis


def self_test():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.manual_seed(96101317)
    rng = np.random.default_rng(96101317)
    source = Path(__file__).resolve().parents[2] / "src" / "comparison_v2" / "models_v2.py"
    spec = importlib.util.spec_from_file_location("fixed_model_test_original", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    N, P, H, L, T = 4, 4, 3, 5, 14
    masks = np.zeros((H, N, N), dtype=np.float32)
    masks[0] = np.eye(N)
    for node in range(N):
        masks[1, node, (node + 1) % N] = 1
        masks[2, node, (node + 2) % N] = 1
    # Directed toy neighborhoods make a reversed node/source convention visible.
    counts = rng.poisson(2.0, (T, N, P)).astype(np.int32)
    counts[:, 2, 1] += 7
    pairs = np.array([5 * N, 5 * N + 1, 6 * N + 2, 7 * N + 3, 9 * N, 11 * N + 1])
    history = original.CountHistory({"counts": counts, "hop_masks": masks}, torch.device("cpu"), time.monotonic() + 60)
    x_array, node_array, y_array = analysis.gather(counts, masks, pairs, L)
    x = torch.as_tensor(x_array)
    nodes = torch.as_tensor(node_array)
    y = torch.as_tensor(y_array, dtype=torch.float32)
    if not torch.equal(x, history.lagged(torch.as_tensor(pairs), L)):
        raise AssertionError("gather differs from original CountHistory in source/node/hop/lag axes")
    if not torch.equal(y, history.y(torch.as_tensor(pairs))):
        raise AssertionError("gather current targets differ from original CountHistory")
    truth = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int8)
    for target in range(P):
        group_masks = {name: analysis.feature_mask(name, target, H, P, L, truth) for name in analysis.GROUPS}
        own, neighbor, cross = (group_masks[key] for key in ("own_device_same_type", "neighbor_same_type", "cross_type"))
        if not np.array_equal(own.astype(int) + neighbor + cross, np.ones((H, P, L))):
            raise AssertionError("Same-type and cross-type masks do not partition all features")
        if not np.array_equal(sum(group_masks[f"hop_{hop}"].astype(int) for hop in range(H)), np.ones((H, P, L))):
            raise AssertionError("Hop masks do not partition all features")
        if not np.array_equal(sum(group_masks[f"lag_{lag}"].astype(int) for lag in range(1, L + 1)), np.ones((H, P, L))):
            raise AssertionError("Lag masks do not partition all features")
        expected_sources = truth[:, target].astype(bool)
        expected_sources[target] = False
        if not np.array_equal(group_masks["reference_cross_type"][0, :, 0], expected_sources):
            raise AssertionError("Reference graph mask must use source,target rather than target,source")
        if not np.array_equal(group_masks["reference_cross_type"] | group_masks["nonreference_cross_type"], cross):
            raise AssertionError("Reference and nonreference cross-type masks do not partition cross-type features")
    config = {"lags": L, "hidden_units": 7, "hidden_layers": 2, "prior_probability": 0.1}
    baseline = torch.as_tensor(rng.uniform(0.1, 0.8, (N, P)), dtype=torch.float32)
    neural = original.NeuralGraph("deterministic", config, baseline, H)
    s2 = original.S2Rectangle(baseline, {}, {"l1": 0, "dag": 0})
    # Stronger toy parameters prevent an almost-constant initial decoder hiding bugs.
    for parameter in neural.parameters():
        parameter.copy_(torch.as_tensor(rng.normal(0, 0.3, parameter.shape), dtype=torch.float32))
        parameter.requires_grad_(False)
    s2.weight_raw.copy_(torch.as_tensor(rng.normal(-2, 0.3, (P, P)), dtype=torch.float32))
    for parameter in s2.parameters():
        parameter.requires_grad_(False)
    max_route_error, max_group_metric_error, max_edge_loss_error = 0.0, 0.0, 0.0
    diagonal_s2_removal_error = 0.0
    for model in (neural, s2):
        graph = model.probabilities() if model.name == "deterministic" else model.weights()
        base, groups, edges, original_mean, _ = analysis.frozen_analysis(model, graph, x, nodes, y, truth)
        means = lambda value: (model.means(value, nodes, graph) if model.name == "deterministic" else
                               model.means(value.sum((1, 3)), nodes))
        if not torch.equal(original_mean, means(x)):
            raise AssertionError("Unremoved frozen analysis differs from original model means")
        reported = {row["group"]: row for row in groups}
        for group in analysis.GROUPS:
            direct_mean = torch.empty_like(y)
            for target in range(P):
                mask = torch.as_tensor(analysis.feature_mask(group, target, H, P, L, truth))
                removed_x = x.clone()
                removed_x[:, mask] = 0
                direct_mean[:, target] = means(removed_x)[:, target]
                if model.name == "deterministic":
                    decoder = model.decoders[target]
                    flat_gate = graph[..., target, None].expand(H, P, L).reshape(-1)
                    gated = torch.log1p(x).flatten(1) * flat_gate
                    first = decoder[0](gated)
                    contribution = F.linear(gated[:, mask.flatten()], decoder[0].weight[:, mask.flatten()])
                    fast_mean = analysis.decoder_tail(decoder, first - contribution, model.baseline_raw[nodes, target])
                    max_route_error = max(max_route_error, float((fast_mean - direct_mean[:, target]).abs().max()))
            expected_metrics = analysis.metrics(y, direct_mean)
            for metric in ("NLL", "MAE", "RMSE", "zero_NLL", "positive_NLL", "mean_predicted_count"):
                if expected_metrics[metric] is not None:
                    max_group_metric_error = max(max_group_metric_error, abs(expected_metrics[metric] - reported[group][metric]))
            if model.name == "s2" and group in ("own_device_same_type", "neighbor_same_type"):
                diagonal_s2_removal_error = max(diagonal_s2_removal_error, float((direct_mean - original_mean).abs().max()))
        for source_index in range(P):
            removed_x = x.clone()
            removed_x[:, :, source_index, :] = 0
            altered = means(removed_x)
            for target in range(P):
                expected_delta = float(analysis.poisson_nll(y[:, target], altered[:, target]).mean() -
                                       analysis.poisson_nll(y[:, target], original_mean[:, target]).mean())
                max_edge_loss_error = max(max_edge_loss_error, abs(expected_delta - edges[source_index, target]))
    if max_route_error > 2e-6 or max_group_metric_error > 3e-6 or max_edge_loss_error > 3e-6:
        raise AssertionError("Fast frozen removal differs materially from explicit input masking")
    if diagonal_s2_removal_error > 2e-6:
        raise AssertionError("S2 same-type removal changes means beyond floating-point tolerance")
    exact = x[..., None].numpy() * s2.weights().numpy()[None, None, :, None, :]
    source_terms = exact.sum((1, 2, 3))
    matrix_effect = (x.sum((1, 3)) @ s2.weights()).numpy()
    linear_error = float(np.max(np.abs(source_terms - matrix_effect)))
    if not np.allclose(source_terms, matrix_effect, atol=2e-6, rtol=2e-6):
        raise AssertionError("Expanded S2 history terms do not reproduce the linear history effect")
    diagonal_terms = exact[..., np.arange(P), :, np.arange(P)]
    if float(np.abs(diagonal_terms).max()) != 0:
        raise AssertionError("S2 diagonal linear contributions should be zero")
    return {"status": "passed", "gather_matches_original_CountHistory": True,
            "feature_masks_partition_and_source_target_axes": True,
            "neural_removal_route_max_abs_mean_difference": max_route_error,
            "group_removal_max_abs_metric_difference": max_group_metric_error,
            "single_source_removal_max_abs_NLL_difference": max_edge_loss_error,
            "S2_same_type_removal_max_abs_mean_difference": diagonal_s2_removal_error,
            "S2_expanded_linear_terms_max_abs_difference": linear_error,
            "roundoff_tolerance_used_not_exact_zero_claim": True,
            "new_fits": 0, "GPU_used": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), ensure_ascii=False, allow_nan=False))
