"""Small CPU arithmetic, information-boundary and end-to-end checks only."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import tempfile
import time
import unittest

import networkx as nx
import numpy as np
import torch

from datasets_v2 import ROOT, read, save, write_dataset
from models_v2 import CountHistory, NeuralGraph, THP_EM, log_probability
from study_v2 import dataset_run, forecast_chunks


def fixture():
    counts = np.zeros((40, 3, 3), dtype=np.int32)
    for t in range(2, 37, 3):
        counts[t, (t // 3) % 3, (t // 3) % 3] = 1
        counts[t + 1, (t // 3 + 1) % 3, (t // 3 + 1) % 3] = 2
    topology = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    masks = np.stack([np.eye(3), topology, np.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]])]).astype(np.float32)
    return {"counts": counts, "topology": topology, "hop_masks": masks}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.obs = fixture()
        self.history = CountHistory(self.obs, torch.device("cpu"), time.monotonic() + 60)
        self.protocol = read(ROOT / "protocol.json")

    def test_masked_pooling_and_pair_order(self):
        expected = np.einsum("knm,tmp->tnkp", self.obs["hop_masks"], self.obs["counts"])
        np.testing.assert_array_equal(self.history.pooled.numpy(), expected)
        indices = torch.tensor([5 * 3 + 1, 8 * 3 + 2])
        got = self.history.lagged(indices, 5).numpy()
        for row, (t, node) in enumerate([(5, 1), (8, 2)]):
            np.testing.assert_array_equal(got[row], np.stack([expected[t - lag, node] for lag in range(1, 6)], -1))

    def test_full_exponential_tail_and_no_future(self):
        self.history.build_exponential(.2, time.monotonic() + 60)
        expected = np.zeros_like(self.history.exponential.numpy())
        factor = -np.expm1(-.2) / .2
        for t in range(1, 40):
            expected[t] = factor * self.history.pooled[t - 1].numpy() + np.exp(-.2) * expected[t - 1]
        np.testing.assert_allclose(self.history.exponential.numpy(), expected, rtol=2e-6, atol=2e-6)
        future = fixture()
        future["counts"][20:] += 7
        changed = CountHistory(future, torch.device("cpu"), time.monotonic() + 60)
        changed.build_exponential(.2, time.monotonic() + 60)
        np.testing.assert_array_equal(changed.exponential[:21].numpy(), self.history.exponential[:21].numpy())

    def test_EM_matches_pinned_native_EM_given_same_statistics(self):
        config = copy.deepcopy(self.protocol["thp"])
        model = THP_EM(self.history, [5, 25], config, time.monotonic() + 60)
        graph = np.eye(3, dtype=np.int8)
        graph[0, 1] = graph[1, 2] = 1
        for target in range(3):
            model.fit_parent_sets(target, [model.parents(graph, target)])
        source = ast.parse((ROOT / "public_inputs/ttpm.py").read_text(encoding="utf-8"))
        method = next(node for node in next(node for node in source.body if isinstance(node, ast.ClassDef)).body if isinstance(node, ast.FunctionDef) and node.name == "_em")
        namespace = {"np": np, "nx": nx}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "pinned_ttpM_EM", "exec"), namespace)
        class Column:
            def __init__(self, values):
                self.values = values
            def sum(self):
                return self.values.sum()
        class Native:
            pass
        class TensorLike(dict):
            def __len__(self):
                return len(self["times"].values)
        native = Native()
        native._N, native._max_hop, native._penalty, native._epsilon = 3, 2, "BIC", 1
        native._event_names = np.arange(3)
        native._event_indexes = np.concatenate([np.full(len(data[1]), target) for target, data in enumerate(model.positive_data)])
        native.tensor = TensorLike(times=Column(np.concatenate([data[1].numpy() for data in model.positive_data])))
        combined = np.concatenate([data[0].numpy() for data in model.positive_data])
        native._effect_tensor_decays = combined.reshape(-1, 3, 3).transpose(1, 0, 2)
        native._decay_effects = model.excitations_exposure.numpy().reshape(3, 3).T
        native._T = model.exposure
        score, alpha, mu = namespace["_em"](native, graph)
        got_alpha = np.stack([model.cache[target][model.parents(graph, target)]["alpha"].numpy().reshape(3, 3) for target in range(3)], -1)
        got_mu = np.asarray([float(model.cache[target][model.parents(graph, target)]["mu"]) for target in range(3)])
        np.testing.assert_allclose(got_alpha, alpha, rtol=2e-7, atol=1e-8)
        np.testing.assert_allclose(got_mu, mu, rtol=2e-7, atol=1e-8)
        self.assertAlmostEqual(model.graph_score(graph), score, places=6)

    def test_search_only_DAGs_and_train_no_labels(self):
        model = THP_EM(self.history, [5, 25], self.protocol["thp"], time.monotonic() + 60)
        graph = np.eye(3, dtype=np.int8)
        graph[0, 1] = graph[1, 2] = 1
        for candidate in model.candidate_graphs(graph):
            off = candidate.copy()
            np.fill_diagonal(off, 0)
            self.assertTrue(nx.is_directed_acyclic_graph(nx.DiGraph(off)))
        self.assertFalse(hasattr(model, "truth"))

    def test_vector_MC_equals_sequential_with_identical_graph_draws(self):
        torch.manual_seed(4)
        model = NeuralGraph("direct", self.protocol["neural"], torch.ones(3, 3), 3)
        q = model.probabilities()
        indices = torch.arange(15, 24)
        chunk = list(forecast_chunks(model, self.history, indices, q, self.protocol, 5, 78, time.monotonic() + 60))[0]
        generator = torch.Generator().manual_seed(78)
        graphs = (torch.rand((5,) + tuple(q.shape), generator=generator) < q).float()
        _, nodes = self.history.indices(indices)
        x, y = self.history.lagged(indices, 5), self.history.y(indices)
        samples = torch.stack([model.means(x, nodes, graph) for graph in graphs]).double()
        np.testing.assert_allclose(chunk[2].numpy(), samples.mean(0).detach().numpy(), atol=2e-6)
        expected_mass = torch.logsumexp(log_probability(y[None], samples), 0) - np.log(5)
        np.testing.assert_allclose(chunk[3].numpy(), expected_mass.detach().numpy(), atol=2e-6)

    def test_tiny_five_model_end_to_end(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["neural"].update(hidden_units=4, hidden_layers=1, learning_rates=[.001], maximum_updates=2,
                                 minimum_updates=1, validation_every_updates=1, patience_checks=2,
                                 batch_pairs=8, graph_calibration_TRAIN_pairs_max=30, validation_pairs_max=30,
                                 validation_graph_draws=2, test_graph_draws=3, test_batch_pairs=20)
        protocol["s2"].update(learning_rates=[.05], regularization_candidates=[[.001, .1]], maximum_updates=2)
        protocol["thp"].update(maximum_graph_moves=2)
        with tempfile.TemporaryDirectory(prefix="tnpar_v2_smoke_") as temp:
            path = Path(temp)
            data = path / "input"
            truth = np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8)
            write_dataset(data, self.obs["counts"], self.obs["topology"], truth,
                          {"kind": "CPU_smoke", "name": "toy", "seed": 42, "delta_seconds": 2.,
                           "train": [5, 25], "validation": [25, 32], "test": [32, 40], "graph_labels": "toy_not_efficacy"})
            result = dataset_run(data, path / "output", protocol, torch.device("cpu"), time.monotonic() + 60)
            self.assertEqual(result["fits_completed"], 5)
            self.assertEqual(result["selected_test_evaluations_completed"], 5)
            self.assertEqual({row["model"] for row in result["results"]}, set(protocol["models"]))
            for row in result["results"]:
                self.assertTrue(np.isfinite(row["metrics"]["predictive_count_NLL"]))
                self.assertIsNone(row["metrics"]["latent_lambda_RMSE"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    target = ROOT / ("test_report_" + str(time.time_ns()) + ".json")
    save(target, {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "elapsed_seconds": time.perf_counter() - start, "GPU_used": False,
                  "native_EM_statistics_equivalence_only_not_original_performance_reproduction": True})
    raise SystemExit(not result.wasSuccessful())
