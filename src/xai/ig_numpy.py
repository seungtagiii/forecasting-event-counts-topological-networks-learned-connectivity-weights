"""Fixed-model explanations of a two-hidden-layer Tanh count decoder.

The inputs are already ``log1p(history)``. Connectivity gates multiply these
features before the first affine layer. This module never fits parameters and
does not explain the graph-learning objective or identify causal effects.

PyTorch Linear parameters retain their original (output, input) orientation.
The device/target ``baseline_raw`` is fixed during each IG integration path.
``steps`` denotes trapezoidal intervals: the path has ``steps + 1`` points.
"""
from __future__ import annotations

import json
import numpy as np


def stable_softplus(value):
    return np.logaddexp(0.0, np.asarray(value, dtype=np.float64))


def stable_log_sigmoid(value):
    return -np.logaddexp(0.0, -np.asarray(value, dtype=np.float64))


def stable_sigmoid(value):
    return np.exp(stable_log_sigmoid(value))


def _array(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite values in {name}")
    return result


def _prepare(inputs, W0, b0, W2, b2, W4, b4, gate_flat, baseline_raw, epsilon):
    x = _array(inputs, "inputs")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("inputs must have nonempty shape (batch, features)")
    w0, w2, w4 = (_array(value, name) for value, name in ((W0, "W0"), (W2, "W2"), (W4, "W4")))
    bias0, bias2, bias4 = (_array(value, name) for value, name in ((b0, "b0"), (b2, "b2"), (b4, "b4")))
    if w0.ndim != 2 or w0.shape[1] != x.shape[1] or bias0.shape != (w0.shape[0],):
        raise ValueError("W0/b0 dimensions do not match input features")
    if w2.ndim != 2 or w2.shape[1] != w0.shape[0] or bias2.shape != (w2.shape[0],):
        raise ValueError("W2/b2 dimensions do not match first hidden layer")
    if w4.shape == (w2.shape[0],):
        w4 = w4.reshape(1, -1)
    if w4.shape != (1, w2.shape[0]) or bias4.size != 1:
        raise ValueError("W4/b4 must define one target output")
    gate = _array(gate_flat, "gate_flat")
    if gate.shape != (x.shape[1],):
        raise ValueError("gate_flat must have one weight per flattened input feature")
    if ((gate < 0) | (gate > 1)).any():
        raise ValueError("Expected continuous sigmoid connectivity gates in [0, 1]")
    baseline = _array(baseline_raw, "baseline_raw")
    if baseline.ndim == 0:
        baseline = np.full(x.shape[0], baseline.item(), dtype=np.float64)
    elif baseline.shape == (x.shape[0], 1):
        baseline = baseline[:, 0]
    elif baseline.shape != (x.shape[0],):
        raise ValueError("baseline_raw must be a scalar or one value per sample")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    return x, w0, bias0, w2, bias2, w4[0], bias4.reshape(-1)[0], gate, baseline, epsilon


def forward_and_gradient(inputs, W0, b0, W2, b2, W4, b4, gate_flat, baseline_raw, epsilon=1e-8):
    """Return (positive predictive mean, derivative wrt logged history input).

    Derivative includes the connectivity gate and the final softplus derivative.
    No derivative is taken with respect to the gate or node-specific baseline.
    """
    x, w0, bias0, w2, bias2, w4, bias4, gate, baseline, epsilon = _prepare(
        inputs, W0, b0, W2, b2, W4, b4, gate_flat, baseline_raw, epsilon
    )
    h1 = np.tanh((x * gate) @ w0.T + bias0)
    h2 = np.tanh(h1 @ w2.T + bias2)
    raw_mean = baseline + h2 @ w4 + bias4
    mean = stable_softplus(raw_mean) + epsilon
    derivative_h2 = stable_sigmoid(raw_mean)[:, None] * w4[None, :]
    derivative_h1 = (derivative_h2 * (1.0 - h2 * h2)) @ w2
    gradient = ((derivative_h1 * (1.0 - h1 * h1)) @ w0) * gate
    if not np.isfinite(mean).all() or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite decoder output or gradient")
    return mean, gradient


def forward(inputs, W0, b0, W2, b2, W4, b4, gate_flat, baseline_raw, epsilon=1e-8):
    return forward_and_gradient(inputs, W0, b0, W2, b2, W4, b4, gate_flat, baseline_raw, epsilon)[0]


def analytic_gradient(inputs, W0, b0, W2, b2, W4, b4, gate_flat, baseline_raw, epsilon=1e-8):
    return forward_and_gradient(inputs, W0, b0, W2, b2, W4, b4, gate_flat, baseline_raw, epsilon)[1]


def integrated_gradients(inputs, W0, b0, W2, b2, W4, b4, gate_flat, baseline_raw,
                         baseline_input=None, steps=128, epsilon=1e-8):
    """Trapezoidal IG of one target mean in logged-feature coordinates.

    ``baseline_input=None`` uses an all-zero history. Pass a feature vector
    (e.g. a TRAIN mean of log1p history) or a batch-shaped array for alternatives.
    Returns a dict of NumPy arrays, including per-sample completeness residual:
    ``sum(attributions) - (input_mean - baseline_mean)``.
    """
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer number of path intervals")
    x = _array(inputs, "inputs")
    if x.ndim != 2:
        raise ValueError("inputs must have shape (batch, features)")
    if baseline_input is None:
        reference = np.zeros_like(x)
    else:
        reference = _array(baseline_input, "baseline_input")
        if reference.shape == (x.shape[1],):
            reference = np.broadcast_to(reference, x.shape)
        elif reference.shape != x.shape:
            raise ValueError("baseline_input must have shape (features,) or inputs.shape")
    params = dict(W0=W0, b0=b0, W2=W2, b2=b2, W4=W4, b4=b4,
                  gate_flat=gate_flat, baseline_raw=baseline_raw, epsilon=epsilon)
    delta = x - reference
    gradient_sum = np.zeros_like(x)
    baseline_mean = input_mean = None
    for step in range(int(steps) + 1):
        alpha = step / int(steps)
        mean, gradient = forward_and_gradient(reference + alpha * delta, **params)
        weight = 0.5 if step in (0, int(steps)) else 1.0
        gradient_sum += weight * gradient
        if step == 0:
            baseline_mean = mean
        if step == int(steps):
            input_mean = mean
    attributions = delta * (gradient_sum / int(steps))
    explained_change = attributions.sum(axis=1)
    output_change = input_mean - baseline_mean
    return {
        "attributions": attributions,
        "input_mean": input_mean,
        "baseline_mean": baseline_mean,
        "output_change": output_change,
        "attribution_sum": explained_change,
        "completeness_residual": explained_change - output_change,
        "steps": int(steps),
        "integration_points": int(steps) + 1,
        "input_coordinate": "log1p_history_before_connectivity_gate",
        "explained_quantity": "single_target_Poisson_mean",
        "node_target_baseline_held_fixed": True,
    }


def self_test():
    """Finite differences, two IG baselines, zero gates and extreme stability."""
    rng = np.random.default_rng(20261013)
    batch, features, h1, h2 = 4, 7, 8, 6
    inputs = np.log1p(rng.poisson(2.0, size=(batch, features))).astype(np.float64)
    params = {
        "W0": rng.normal(0, 0.3, (h1, features)), "b0": rng.normal(0, 0.1, h1),
        "W2": rng.normal(0, 0.3, (h2, h1)), "b2": rng.normal(0, 0.1, h2),
        "W4": rng.normal(0, 0.3, (1, h2)), "b4": np.array([0.17]),
        "gate_flat": rng.uniform(0.05, 0.8, features),
        "baseline_raw": np.array([-6.0, -0.8, 0.2, 2.0]),
    }
    analytic = analytic_gradient(inputs, **params)
    finite_difference = np.empty_like(inputs)
    step = 1e-6
    for feature in range(features):
        plus, minus = inputs.copy(), inputs.copy()
        plus[:, feature] += step
        minus[:, feature] -= step
        finite_difference[:, feature] = (forward(plus, **params) - forward(minus, **params)) / (2 * step)
    gradient_error = float(np.max(np.abs(analytic - finite_difference)))
    if gradient_error >= 2e-8:
        raise AssertionError("Analytic gradient differs from finite differences")
    ig_zero = integrated_gradients(inputs, **params, steps=512)
    ig_mean = integrated_gradients(inputs, **params, baseline_input=inputs.mean(axis=0), steps=512)
    residual_zero = float(np.max(np.abs(ig_zero["completeness_residual"])))
    residual_mean = float(np.max(np.abs(ig_mean["completeness_residual"])))
    if max(residual_zero, residual_mean) >= 2e-6:
        raise AssertionError("IG completeness failed at tested resolution")
    zero_params = dict(params, gate_flat=np.zeros(features))
    if not np.array_equal(analytic_gradient(inputs, **zero_params), np.zeros_like(inputs)):
        raise AssertionError("Zero gates should eliminate input gradients")
    zero_ig = integrated_gradients(inputs, **zero_params, steps=8)
    if not np.array_equal(zero_ig["attributions"], np.zeros_like(inputs)):
        raise AssertionError("Zero gates should eliminate IG attributions")
    extreme = np.array([-1e6, -1e3, 0.0, 1e3, 1e6])
    for output in (stable_softplus(extreme), stable_sigmoid(extreme), stable_log_sigmoid(extreme)):
        if not np.isfinite(output).all():
            raise AssertionError("Stable activation failed for extreme values")
    return {
        "status": "passed", "numpy_version": str(np.__version__),
        "finite_difference_step": step, "maximum_gradient_absolute_error": gradient_error,
        "IG_test_intervals": 512,
        "zero_baseline_maximum_completeness_residual": residual_zero,
        "mean_baseline_maximum_completeness_residual": residual_mean,
        "zero_gate_gradients_and_IG": "passed", "extreme_activation_stability": "passed",
        "new_training_runs": 0, "GPU_used": False,
    }


if __name__ == "__main__":
    print(json.dumps(self_test(), ensure_ascii=False, allow_nan=False))
