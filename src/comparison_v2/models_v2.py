"""Paper-referenced models; common count observations, no event-end inputs.

TNPAR incoming-edge encoder is a documented reduced parameterization, not
author code. Binary Concrete is the binary Gumbel-Softmax relaxation.
THP EM/BIC equations follow gCastle TTPM (Apache-2.0); search uses the paper's
best add/delete/reverse neighbor. Binned likelihood and exposure differ from
the native timestamp implementation. See protocol.json for all adaptations.
"""
from __future__ import annotations

import math
import time

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Declared study execution budget reached")


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def mlp(inputs, outputs, hidden=32, layers=2):
    blocks = []
    for _ in range(layers):
        blocks += [nn.Linear(inputs, hidden), nn.Tanh()]
        inputs = hidden
    return nn.Sequential(*blocks, nn.Linear(inputs, outputs))


def log_probability(y, mean):
    mean = mean.clamp_min(1e-8)
    return y * mean.log() - mean - torch.lgamma(y + 1)


def kl_bernoulli(q, prior):
    q = q.clamp(1e-6, 1 - 1e-6)
    return q * (q.log() - math.log(prior)) + (1 - q) * ((1 - q).log() - math.log(1 - prior))


def dag_penalty(matrix, squared=False):
    p = matrix.shape[-1]
    off = matrix * (1 - torch.eye(p, device=matrix.device))
    if squared:
        off = off.square()
    result = torch.eye(p, device=matrix.device) + off / p
    return torch.diagonal(torch.linalg.matrix_power(result, p), dim1=-2, dim2=-1).sum(-1) - p


class CountHistory:
    """Gather node-time pairs without materializing the giant 5-lag tensor."""
    def __init__(self, observations, device, deadline):
        self.counts = torch.as_tensor(observations["counts"], dtype=torch.float32, device=device)
        self.T, self.N, self.P = self.counts.shape
        masks = observations["hop_masks"]
        self.H = len(masks)
        flat = self.counts.transpose(0, 1).reshape(self.N, -1).contiguous()
        self.pooled = torch.empty((self.T, self.N, self.H, self.P), dtype=torch.float32, device=device)
        for hop in range(self.H):
            check_time(deadline)
            mask = torch.as_tensor(masks[hop], device=device).to_sparse_coo()
            result = torch.sparse.mm(mask, flat).reshape(self.N, self.T, self.P).transpose(0, 1)
            self.pooled[:, :, hop] = result
        self.exponential = None

    @property
    def device(self):
        return self.counts.device

    def indices(self, pair_indices):
        return pair_indices // self.N, pair_indices % self.N

    def y(self, pair_indices):
        times, nodes = self.indices(pair_indices)
        return self.counts[times, nodes]

    def lagged(self, pair_indices, lags):
        times, nodes = self.indices(pair_indices)
        offsets = torch.arange(1, lags + 1, device=self.device)
        return self.pooled[times[:, None] - offsets, nodes[:, None]].permute(0, 2, 3, 1)

    def rectangle(self, pair_indices, window):
        return self.lagged(pair_indices, window).sum((1, 3))

    def build_exponential(self, decay, deadline):
        """Full tail, with bounded-block prefix sums to avoid exp overflow."""
        self.exponential = torch.empty_like(self.pooled)
        state = torch.zeros_like(self.pooled[0])
        average_factor = -math.expm1(-decay) / decay
        for begin in range(0, self.T, 32):
            check_time(deadline)
            block = self.pooled[begin:begin + 32]
            powers = torch.exp(-decay * torch.arange(1, len(block) + 1, device=self.device, dtype=torch.float32))
            powers = powers[:, None, None, None]
            after = powers * (state[None] + torch.cumsum(block / powers, 0))
            before = torch.cat([state[None], after[:-1]], 0)
            self.exponential[begin:begin + len(block)] = before * average_factor
            state = after[-1].clone()

    def exp_history(self, pair_indices):
        times, nodes = self.indices(pair_indices)
        return self.exponential[times, nodes]


class NeuralGraph(nn.Module):
    def __init__(self, name, config, baseline, hops):
        super().__init__()
        self.name, self.config = name, config
        self.p, self.hops = baseline.shape[1], hops
        self.baseline_raw = nn.Parameter(inverse_softplus(baseline.clamp_min(1e-5)))
        inputs = hops * self.p * config["lags"]
        self.decoders = nn.ModuleList([mlp(inputs, 1, config["hidden_units"], config["hidden_layers"]) for _ in range(self.p)])
        initial = math.log(config["prior_probability"] / (1 - config["prior_probability"]))
        if name == "tnpar":
            self.encoders = nn.ModuleList([mlp(inputs + 1, hops * self.p, config["hidden_units"], config["hidden_layers"]) for _ in range(self.p)])
            for encoder in self.encoders:
                nn.init.normal_(encoder[-1].weight, std=.01)
                nn.init.constant_(encoder[-1].bias, initial)
        else:
            self.graph_logits = nn.Parameter(torch.full((hops, self.p, self.p), initial))
        for decoder in self.decoders:
            nn.init.normal_(decoder[-1].weight, std=.01)
            nn.init.zeros_(decoder[-1].bias)

    def probabilities(self, x=None, current_y=None):
        if self.name != "tnpar":
            return torch.sigmoid(self.graph_logits)
        if current_y is None:
            raise ValueError("Encoder requires TRAIN reconstruction observations")
        flat = torch.log1p(x).flatten(1)
        outputs = []
        for target, encoder in enumerate(self.encoders):
            current = torch.log1p(current_y[:, target:target + 1])
            logits = encoder(torch.cat([flat, current], -1)).reshape(-1, self.hops, self.p)
            outputs.append(torch.sigmoid(logits))
        return torch.stack(outputs, -1)

    def means(self, x, nodes, graph):
        outputs, logged = [], torch.log1p(x)
        for target, decoder in enumerate(self.decoders):
            gate = graph[..., target]
            if gate.ndim == 2:
                gate = gate[None]
            effect = decoder((logged * gate.unsqueeze(-1)).flatten(1)).squeeze(-1)
            outputs.append(F.softplus(self.baseline_raw[nodes, target] + effect) + 1e-8)
        return torch.stack(outputs, -1)

    def objective(self, x, nodes, y, n_train_counts):
        q = self.probabilities(x, y if self.name == "tnpar" else None)
        if self.name == "deterministic":
            gate, kl = q, q.new_zeros(())
        else:
            noise = torch.rand_like(q).clamp(1e-6, 1 - 1e-6)
            gate = torch.sigmoid((torch.logit(q.clamp(1e-6, 1 - 1e-6)) + torch.logit(noise)) / self.config["temperature"])
            terms = kl_bernoulli(q, self.config["prior_probability"])
            kl = terms.sum((-3, -2)).mean() if self.name == "tnpar" else terms.sum() / n_train_counts
        mean = self.means(x, nodes, gate)
        nll = -log_probability(y, mean).mean()
        global_gate = gate.mean(0) if gate.ndim == 4 else gate
        sparse = global_gate.sum() / self.p
        dag = dag_penalty(gate.sum(-3)).mean()
        loss = nll + kl + self.config["sparsity_weight"] * sparse + self.config["dag_weight"] * dag
        return loss, {"nll": float(nll.detach()), "kl": float(kl.detach()), "sparsity": float(sparse.detach()), "dag": float(dag.detach())}


class S2Rectangle(nn.Module):
    def __init__(self, baseline, config, regularization):
        super().__init__()
        self.name, self.config, self.regularization = "s2", config, regularization
        self.p = baseline.shape[-1]
        self.baseline_raw = nn.Parameter(inverse_softplus(baseline.clamp_min(1e-5)))
        self.weight_raw = nn.Parameter(torch.full((self.p, self.p), math.log(math.expm1(1e-7))))
        self.register_buffer("off_diagonal", 1 - torch.eye(self.p))

    def weights(self):
        return F.softplus(self.weight_raw) * self.off_diagonal

    def means(self, history, nodes, graph=None):
        return F.softplus(self.baseline_raw[nodes]) + history @ self.weights() + 1e-8

    def objective(self, x, nodes, y, n_train_counts):
        weights = self.weights()
        nll = -log_probability(y, self.means(x, nodes)).mean()
        sparse, dag = weights.sum(), dag_penalty(weights, squared=True)
        loss = nll + self.regularization[0] * sparse + self.regularization[1] * dag
        return loss, {"nll": float(nll.detach()), "kl": 0., "sparsity": float(sparse.detach()), "dag": float(dag.detach())}


class THP_EM:
    """EM + BIC cached DAG search, retaining original weighted hop excitation.

    Initialization, multiplicative EM, absolute .1 likelihood stopping and BIC
    parent-count penalty follow pinned TTPM. Native timestamp exposures are
    replaced by sums over the common binned Poisson observation grid.
    """
    name = "thp"

    def __init__(self, history, train_interval, config, deadline):
        self.history, self.config = history, config
        self.p, self.hops = history.P, history.H
        self.start, self.end = train_interval
        self.deadline = deadline
        self.fit_deadline = min(deadline, time.monotonic() + config["maximum_fit_seconds"])
        history.build_exponential(config["decay_per_bin"], deadline)
        self.exposure = (self.end - self.start) * history.N
        train_x = history.exponential[self.start:self.end].flatten(0, 1).flatten(1).to(torch.float64)
        train_y = history.counts[self.start:self.end].flatten(0, 1).to(torch.float64)
        self.excitations_exposure = train_x.sum(0)
        self.event_counts = max(2., float(train_y.sum()))
        self.positive_data = []
        for target in range(self.p):
            index = train_y[:, target] > 0
            self.positive_data.append((train_x[index].contiguous(), train_y[index, target].contiguous()))
        del train_x, train_y
        self.cache = [{} for _ in range(self.p)]
        self.EM_caps = 0
        self.EM_fits = 0

    def fit_parent_sets(self, target, parent_sets):
        todo = sorted(set(tuple(sorted(parents)) for parents in parent_sets) - set(self.cache[target]))
        if not todo:
            return
        X, y = self.positive_data[target]
        allowed = torch.zeros((self.hops * self.p, len(todo)), dtype=torch.bool, device=X.device)
        for column, parents in enumerate(todo):
            for hop in range(self.hops):
                allowed[hop * self.p + np.asarray(parents, dtype=int), column] = True
        alpha = allowed.to(torch.float64)
        mu = torch.ones(len(todo), dtype=torch.float64, device=X.device)
        old = torch.full_like(mu, -1e30)
        done = torch.zeros(len(todo), dtype=torch.bool, device=X.device)
        epochs = torch.zeros(len(todo), dtype=torch.int64, device=X.device)
        if not len(y):
            alpha.zero_()
            mu.fill_(1e-8)
            done.fill_(True)
            old.fill_(-self.exposure * 1e-8)
        for update in range(1, self.config["maximum_EM_updates"] + 1):
            check_time(self.deadline)
            if bool(done.all()):
                break
            means = (X @ alpha + mu).clamp_min(1e-12)
            likelihood = (y[:, None] * means.log()).sum(0) - self.excitations_exposure @ alpha - self.exposure * mu
            if not bool(torch.isfinite(likelihood).all()):
                raise FloatingPointError("Nonfinite EM likelihood")
            improvement = likelihood - old
            if bool((improvement[~done] < -1e-5).any()):
                raise FloatingPointError("EM likelihood decreased beyond rounding tolerance")
            newly_done = (~done) & (improvement < self.config["EM_absolute_log_likelihood_tolerance"])
            old = torch.where(done, old, likelihood)
            epochs = torch.where(~done, update, epochs)
            done |= newly_done
            ratio = y[:, None] / means
            next_mu = (mu * ratio.sum(0) / self.exposure).clamp_min(1e-12)
            next_alpha = alpha * (X.T @ ratio) / self.excitations_exposure.clamp_min(1e-12)[:, None]
            next_alpha = torch.where(self.excitations_exposure[:, None] > 0, next_alpha, 0.) * allowed
            mu = torch.where(done, mu, next_mu)
            alpha = torch.where(done[None], alpha, next_alpha)
        # At a hard EM cap, score the final parameter values, not the previous iterate.
        final_likelihood = (y[:, None] * (X @ alpha + mu).clamp_min(1e-12).log()).sum(0) - self.excitations_exposure @ alpha - self.exposure * mu
        for column, parents in enumerate(todo):
            penalty = (1 + self.config["bic_epsilon"] * len(parents) * self.hops) * math.log(self.event_counts) / 2
            converged = bool(done[column])
            self.cache[target][parents] = {"score": float(final_likelihood[column]) - penalty,
                                           "log_likelihood": float(final_likelihood[column]),
                                           "alpha": alpha[:, column].detach().clone(), "mu": mu[column].detach().clone(),
                                           "EM_updates": int(epochs[column]), "EM_converged": converged}
            self.EM_caps += int(not converged)
            self.EM_fits += 1

    def parents(self, graph, target):
        return tuple(np.flatnonzero(graph[:, target]).tolist())

    def candidate_graphs(self, graph):
        for source in range(self.p):
            for target in range(self.p):
                if source == target:
                    continue
                candidate = graph.copy()
                candidate[source, target] = 1 - candidate[source, target]
                off = candidate.copy()
                np.fill_diagonal(off, 0)
                if nx.is_directed_acyclic_graph(nx.DiGraph(off)):
                    yield candidate
                if graph[source, target] and not graph[target, source]:
                    candidate = graph.copy()
                    candidate[source, target], candidate[target, source] = 0, 1
                    off = candidate.copy()
                    np.fill_diagonal(off, 0)
                    if nx.is_directed_acyclic_graph(nx.DiGraph(off)):
                        yield candidate

    def graph_score(self, graph):
        return sum(self.cache[target][self.parents(graph, target)]["score"] for target in range(self.p))

    def fit(self, progress=None):
        graph = np.eye(self.p, dtype=np.int8)
        for target in range(self.p):
            self.fit_parent_sets(target, [self.parents(graph, target)])
        score = self.graph_score(graph)
        journal = [{"move": 0, "TRAIN_BIC": score, "edges": 0}]
        reason = "maximum_graph_moves"
        for move in range(1, self.config["maximum_graph_moves"] + 1):
            check_time(self.deadline)
            if time.monotonic() >= self.fit_deadline:
                reason = "declared_fit_time_budget"
                break
            candidates = list(self.candidate_graphs(graph))
            for target in range(self.p):
                self.fit_parent_sets(target, [self.parents(candidate, target) for candidate in candidates])
            best = max(candidates, key=self.graph_score) if candidates else graph
            best_score = self.graph_score(best)
            if best_score <= score + 1e-8:
                reason = "BIC_local_optimum"
                break
            graph, score = best, best_score
            journal.append({"move": move, "TRAIN_BIC": score, "edges": int(graph.sum() - self.p),
                            "parent_set_EM_fits": self.EM_fits, "EM_caps": self.EM_caps})
            if progress:
                progress(journal[-1])
        alpha, mu = [], []
        for target in range(self.p):
            result = self.cache[target][self.parents(graph, target)]
            alpha.append(result["alpha"].reshape(self.hops, self.p))
            mu.append(result["mu"])
        self.alpha, self.mu = torch.stack(alpha, -1), torch.stack(mu)
        self.graph = graph
        self.fit_report = {"termination": reason, "graph_search_converged": reason == "BIC_local_optimum",
                           "graph_moves": len(journal) - 1, "parent_set_EM_fits": self.EM_fits,
                           "EM_capped_parent_fits": self.EM_caps, "history": journal,
                           "TRAIN_BIC": score, "fits_use_test_or_true_graph": False}
        del self.positive_data, self.cache
        return self.fit_report

    def means(self, history, nodes=None, graph=None):
        return (torch.einsum("bhp,hpv->bv", history.to(torch.float64), self.alpha) + self.mu).clamp_min(1e-8)

    def state_dict(self):
        return {"alpha": self.alpha, "mu": self.mu, "graph": torch.as_tensor(self.graph)}
