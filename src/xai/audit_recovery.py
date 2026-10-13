"""Audit recovered fixed-model artifacts without fitting or modifying sources."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


DATASET = "real_24V_439N_Microwave"
MODELS = ("deterministic", "s2")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(array):
    """Match materialize_v2.array_digest without concatenating full array bytes."""
    array = np.ascontiguousarray(array)
    header = json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode()
    digest = hashlib.sha256(header)
    data = memoryview(array).cast("B")
    for start in range(0, len(data), 8 * 1024 * 1024):
        digest.update(data[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def unique(root, name, contains=None):
    paths = [path for path in root.rglob(name) if contains is None or contains in path.parts]
    if len(paths) != 1:
        raise ValueError(f"Expected one recovered {name} under {root.name}; found {len(paths)}")
    return paths[0]


def artifact(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError("Missing or out-of-scope recovered artifact")
    return candidate


def compare_source(recovered, repository, executed_hash):
    received = recovered.read_bytes()
    local = repository.read_bytes()
    received_text, local_text = received.decode("utf-8-sig"), local.decode("utf-8-sig")
    normalize = lambda text: text.replace("\r\n", "\n").replace("\r", "\n")
    received_ast = ast.dump(ast.parse(received_text), include_attributes=False)
    local_ast = ast.dump(ast.parse(local_text), include_attributes=False)
    return {
        "recovered_path": str(recovered.resolve()), "repository_path": str(repository.resolve()),
        "recovered_sha256": hashlib.sha256(received).hexdigest(),
        "repository_sha256": hashlib.sha256(local).hexdigest(),
        "executed_manifest_sha256": executed_hash,
        "raw_bytes_match_repository": received == local,
        "LF_normalized_text_match_repository": normalize(received_text) == normalize(local_text),
        "Python_AST_match_repository": received_ast == local_ast,
        "recovered_matches_executed_manifest": hashlib.sha256(received).hexdigest() == executed_hash,
    }


def audit_dataset(root, blueprint):
    obs_path = unique(root, "observations.npz", DATASET)
    labels_path = obs_path.parent / "evaluation_labels.npz"
    metadata_path = obs_path.parent / "dataset.json"
    metadata = read(metadata_path)
    arrays = {}
    shapes = {}
    with np.load(obs_path, allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            arrays[key], shapes[key] = array_digest(value), list(value.shape)
            del value
            gc.collect()
    with np.load(labels_path, allow_pickle=False) as archive:
        if archive.files != ["graph_source_target"]:
            raise ValueError("Unexpected recovered label array names")
        truth = archive["graph_source_target"]
        arrays["graph_source_target"] = array_digest(truth)
    expected = blueprint[DATASET]["arrays"]
    output = {
        "metadata_path": str(metadata_path.resolve()), "metadata_sha256": file_digest(metadata_path),
        "observations_file_sha256": file_digest(obs_path), "labels_file_sha256": file_digest(labels_path),
        "observations_match_own_metadata_file_hash": file_digest(obs_path) == metadata["observation_sha256"],
        "labels_match_own_metadata_file_hash": file_digest(labels_path) == metadata["labels_sha256"],
        "canonical_array_hashes": arrays, "canonical_arrays_match_original_blueprint": arrays == expected,
        "array_shapes": shapes, "metadata": metadata,
        "compressed_bytes_and_array_identity_are_distinct": True,
    }
    return output, obs_path, truth


def expected_shapes(model, n, p, hops, protocol):
    output = {"baseline_raw": (n, p)}
    if model == "s2":
        output.update(weight_raw=(p, p), off_diagonal=(p, p))
        return output
    output["graph_logits"] = (hops, p, p)
    hidden = protocol["neural"]["hidden_units"]
    layers = protocol["neural"]["hidden_layers"]
    features = hops * p * protocol["neural"]["lags"]
    for target in range(p):
        previous = features
        for layer in range(layers):
            index = 2 * layer
            output[f"decoders.{target}.{index}.weight"] = (hidden, previous)
            output[f"decoders.{target}.{index}.bias"] = (hidden,)
            previous = hidden
        output[f"decoders.{target}.{2 * layers}.weight"] = (1, previous)
        output[f"decoders.{target}.{2 * layers}.bias"] = (1,)
    return output


def average_precision(truth, scores):
    truth, scores = truth.astype(bool).ravel(), scores.ravel()
    order = np.argsort(-scores, kind="stable")
    labels, ranked = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    precision, recall = tp / (ends + 1), tp / int(truth.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def close_report(actual, expected, tolerance):
    difference = float(actual) - float(expected)
    return {"actual": float(actual), "expected": float(expected), "difference": difference,
            "absolute_tolerance": tolerance, "matches": abs(difference) <= tolerance}


def audit_fit(model, repetition, metadata_root, models_root, selection_path, test_counts,
              truth, protocol, expected_record, n, p, hops):
    selection = read(selection_path)[model]
    relative_dataset = selection_path.parent.relative_to(metadata_root)
    relative_fit = Path(selection["fit_path"])
    fit_path = artifact(models_root, relative_dataset / relative_fit / "fit.json")
    checkpoint_path = artifact(models_root, relative_dataset / relative_fit / "best.pt")
    result_path = artifact(models_root, relative_dataset / "evaluations" / model / "result.json")
    prediction_path = artifact(models_root, relative_dataset / "evaluations" / model / "predictions.npz")
    fit, result = read(fit_path), read(result_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state, frozen = checkpoint["state_dict"], checkpoint["frozen_TRAIN_graph"]
    shapes = expected_shapes(model, n, p, hops, protocol)
    actual_shapes = {key: tuple(value.shape) for key, value in state.items()}
    if model == "deterministic":
        state_graph = torch.sigmoid(state["graph_logits"])
        state_parameter_count = sum(value.numel() for value in state.values())
    else:
        state_graph = torch.nn.functional.softplus(state["weight_raw"]) * state["off_diagonal"]
        state_parameter_count = sum(value.numel() for key, value in state.items() if key != "off_diagonal")
    graph_difference = float((frozen - state_graph).abs().max())
    best_row = min(fit["history"], key=lambda row: row["validation"]["mean_score"])
    checks = {
        "selected_report_exactly_matches_fit_json": selection["report"] == fit,
        "selected_fit_name_matches_selected_trial": relative_fit.name == f"{model}_trial{selection['trial']}",
        "checkpoint_update_matches_selected_fit": checkpoint["update"] == fit["best_update"],
        "checkpoint_validation_score_matches_selected_fit": checkpoint["validation_mean_score"] == fit["best_validation_mean_score"],
        "best_validation_row_matches_saved_checkpoint": (best_row["update"] == checkpoint["update"] and
            best_row["validation"]["mean_score"] == checkpoint["validation_mean_score"]),
        "checkpoint_expected_state_keys_and_shapes": actual_shapes == shapes,
        "state_tensors_CPU_and_finite": all(value.device.type == "cpu" and bool(torch.isfinite(value).all()) for value in state.values()),
        "frozen_graph_CPU_and_finite": frozen.device.type == "cpu" and bool(torch.isfinite(frozen).all()),
        "frozen_graph_matches_checkpoint_state_with_tolerance": bool(torch.allclose(frozen, state_graph, atol=1e-7, rtol=1e-6)),
        "state_parameter_count_matches_fit": state_parameter_count == fit["parameters"],
        "result_model_and_dataset_match": result["model"] == model and result["dataset"] == DATASET,
        "fit_declares_no_true_graph_or_TEST_use": fit["true_graph_or_TEST_used_for_fit"] is False,
        "test_result_declares_no_TEST_selection": result["TEST_used_to_change_model"] is False,
    }
    if model == "s2":
        checks["S2_off_diagonal_buffer_exact"] = torch.equal(state["off_diagonal"], 1 - torch.eye(p))
    if "selected_trial" in expected_record:
        checks["selected_trial_matches_previously_verified_record"] = selection["trial"] == expected_record["selected_trial"]
    if "selected_fit_seed" in expected_record:
        checks["fit_seed_matches_previously_verified_record"] = fit["seed"] == expected_record["selected_fit_seed"]
    expected_result = expected_record["report"]
    metrics = {}
    for field in ("predictive_count_NLL", "count_MAE", "count_RMSE"):
        metrics[f"result_vs_verified_{field}"] = close_report(result["metrics"][field], expected_result["metrics"][field], 1e-12)
    metrics["result_vs_verified_AP"] = close_report(result["structure"]["average_precision"], expected_result["structure"]["average_precision"], 1e-12)
    with np.load(prediction_path, allow_pickle=False) as predictions:
        mass = predictions["predictive_log_mass"]
        checks["predictive_log_mass_shape_and_finite"] = mass.shape == test_counts.shape and bool(np.isfinite(mass).all())
        saved_nll = -float(mass.mean(dtype=np.float64))
        metrics["saved_predictions_vs_result_NLL"] = close_report(saved_nll, result["metrics"]["predictive_count_NLL"], 2e-8)
        del mass
        gc.collect()
        mean = predictions["mean_count"]
        checks["mean_count_shape_positive_finite"] = mean.shape == test_counts.shape and bool(np.isfinite(mean).all()) and bool((mean > 0).all())
        absolute_sum = squared_sum = 0.0
        for start in range(0, len(mean), 128):
            error = mean[start:start + 128].astype(np.float64) - test_counts[start:start + 128]
            absolute_sum += float(np.abs(error).sum(dtype=np.float64))
            squared_sum += float(np.square(error).sum(dtype=np.float64))
        metrics["saved_predictions_vs_result_MAE"] = close_report(absolute_sum / test_counts.size, result["metrics"]["count_MAE"], 2e-8)
        metrics["saved_predictions_vs_result_RMSE"] = close_report(math.sqrt(squared_sum / test_counts.size), result["metrics"]["count_RMSE"], 2e-8)
        del mean
        gc.collect()
        scores, binary = predictions["graph_scores"], predictions["binary_graph"]
        expected_scores = frozen.numpy().copy() if frozen.ndim == 2 else frozen.max(0).values.numpy().copy()
        np.fill_diagonal(expected_scores, 0)
        checks["saved_graph_scores_exactly_match_frozen_graph"] = np.array_equal(scores, expected_scores)
        threshold = (protocol["s2"]["structure_threshold_per_bin_excitation"] if model == "s2" else protocol["neural"]["structure_threshold"])
        expected_binary = (scores >= threshold).astype(np.int8)
        np.fill_diagonal(expected_binary, 0)
        checks["saved_binary_graph_matches_fixed_threshold"] = np.array_equal(binary, expected_binary)
        off = ~np.eye(p, dtype=bool)
        metrics["saved_scores_vs_result_AP"] = close_report(average_precision(truth[off], scores[off]), result["structure"]["average_precision"], 1e-12)
    checks.update({key: row["matches"] for key, row in metrics.items()})
    output = {
        "model": model, "repetition": repetition, "selected_trial": selection["trial"],
        "fit_path": str(fit_path.resolve()), "checkpoint_path": str(checkpoint_path.resolve()),
        "fit_sha256": file_digest(fit_path), "checkpoint_sha256": file_digest(checkpoint_path),
        "result_sha256": file_digest(result_path), "predictions_sha256": file_digest(prediction_path),
        "checkpoint_update": checkpoint["update"], "checkpoint_validation_mean_score": checkpoint["validation_mean_score"],
        "state_parameters": state_parameter_count, "frozen_graph_maximum_state_difference": graph_difference,
        "checks": checks, "metrics": metrics, "passed": all(checks.values()),
        "all_candidate_trial_selection_independently_reverified": False,
        "checkpoint_forward_predictions_reverified_here": False,
    }
    del checkpoint, state, frozen, state_graph
    gc.collect()
    return output


def build(args):
    executed_path = unique(args.original_metadata, "executed_source_manifest.json")
    executed = read(executed_path)
    source_checks = {}
    for label, root in (("original", args.original_metadata), ("repeated", args.repeat_metadata)):
        source_checks[label] = {}
        for name in ("models_v2.py", "study_v2.py", "datasets_v2.py", "evaluation.py"):
            local = args.repository / "src" / ("evaluation.py" if name == "evaluation.py" else "comparison_v2/" + name)
            source_checks[label][name] = compare_source(unique(root, name), local, executed[name])
    blueprint = read(unique(args.original_metadata, "data_blueprint.json"))
    original, original_obs, truth = audit_dataset(args.original_metadata, blueprint)
    repeated, _, repeated_truth = audit_dataset(args.repeat_metadata, blueprint)
    metadata = original["metadata"]
    original_protocol = read(unique(args.original_metadata, "protocol.json", "comparison_v2_results"))
    repeated_protocol = read(unique(args.repeat_metadata, "protocol.json", "minimum_repeat_results"))
    verified = read(args.verified)
    verified_datasets = [row for row in verified["datasets"] if row["dataset"] == DATASET]
    if len(verified_datasets) != 1:
        raise ValueError("Expected one verified real-data dataset")
    records = verified_datasets[0]["raw_verified_records"]
    with np.load(original_obs, allow_pickle=False) as archive:
        counts = archive["counts"]
        test_counts = counts[slice(*metadata["test"])].copy()
        n, p = counts.shape[1:]
        del counts
        hops = archive["hop_masks"].shape[0]
    gc.collect()
    fits = []
    for repetition in (0, 1, 2):
        metadata_root, models_root = ((args.original_metadata, args.original_models) if repetition == 0 else
                                     (args.repeat_metadata, args.repeat_models))
        selection = unique(metadata_root, "validation_selection.json", DATASET if repetition == 0 else f"repeat_{repetition}")
        for model in MODELS:
            matched = [row for row in records[model] if row["repetition"] == repetition]
            if len(matched) != 1:
                raise ValueError("Missing unique previously verified model record")
            fits.append(audit_fit(model, repetition, metadata_root, models_root, selection, test_counts,
                                  truth, original_protocol, matched[0], n, p, hops))
    checks = {
        "all_recovered_sources_match_original_executed_manifest": all(
            row["recovered_matches_executed_manifest"] for stage in source_checks.values() for row in stage.values()),
        "all_recovered_source_ASTs_match_repository": all(
            row["Python_AST_match_repository"] for stage in source_checks.values() for row in stage.values()),
        "all_recovered_source_LF_texts_match_repository": all(
            row["LF_normalized_text_match_repository"] for stage in source_checks.values() for row in stage.values()),
        "both_observation_files_match_own_metadata": original["observations_match_own_metadata_file_hash"] and repeated["observations_match_own_metadata_file_hash"],
        "both_label_files_match_own_metadata": original["labels_match_own_metadata_file_hash"] and repeated["labels_match_own_metadata_file_hash"],
        "original_and_repeated_canonical_arrays_identical": original["canonical_array_hashes"] == repeated["canonical_array_hashes"],
        "both_canonical_arrays_match_original_blueprint": original["canonical_arrays_match_original_blueprint"] and repeated["canonical_arrays_match_original_blueprint"],
        "original_and_repeated_truth_arrays_identical": np.array_equal(truth, repeated_truth),
        "original_and_repeated_base_protocol_equal": original_protocol == repeated_protocol["base_protocol"],
        "metadata_equal_except_recompressed_file_hashes": {k: v for k, v in metadata.items() if k not in {"observation_sha256", "labels_sha256"}} ==
            {k: v for k, v in repeated["metadata"].items() if k not in {"observation_sha256", "labels_sha256"}},
        "six_selected_fit_artifacts_passed": len(fits) == 6 and all(row["passed"] for row in fits),
    }
    return {
        "status": "verified_fixed_artifact_recovery" if all(checks.values()) else "audit_failed_with_explicit_mismatches",
        "created_at_UTC": datetime.now(timezone.utc).isoformat(), "dataset": DATASET,
        "executed_source_manifest_sha256": file_digest(executed_path),
        "previous_verified_report_sha256": file_digest(args.verified),
        "sources": source_checks, "datasets": {"original": original, "repeated": repeated},
        "selected_fits": fits, "checks": checks,
        "compressed_observation_files_identical": original["observations_file_sha256"] == repeated["observations_file_sha256"],
        "canonical_hash_definition": "SHA256(sorted JSON {shape,dtype} header + contiguous C-order array bytes), identical to materialize_v2.array_digest",
        "new_training_runs": 0, "GPU_used": False, "source_predictions_or_manuscript_modified": False,
        "limitations_ko": [
            "원본·추가 반복에서 선택된 연속 모형과 S²GCSL의 6개 체크포인트 및 저장 예측만 감사했다.",
            "선택된 적합 내부의 최적 검증 update는 확인했지만 미회수된 모든 후보 trial 간 선택은 독립적으로 다시 검증하지 않았다.",
            "이 감사는 체크포인트 상태·그래프·저장 예측의 산술 일치를 확인하며 전체 테스트의 체크포인트 forward 재실행과는 구별된다.",
            "compressed NPZ 파일 바이트는 재압축으로 달라질 수 있어 각자 metadata 파일 해시 및 canonical 배열 해시를 별도로 확인했다.",
            "세 반복은 동일 데이터의 학습 반복이며 독립적인 새 데이터 반복이 아니다.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-metadata", "repeat-metadata", "original-models", "repeat-models", "repository", "verified", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing audit result will not be overwritten")
    torch.set_num_threads(2)
    report = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "selected_fits": len(report["selected_fits"]),
                      "failed_checks": [key for key, value in report["checks"].items() if not value],
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    if not all(report["checks"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
