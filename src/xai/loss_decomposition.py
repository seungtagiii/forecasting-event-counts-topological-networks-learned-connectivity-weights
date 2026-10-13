"""Read-only decomposition of existing real-data forecast losses; no refitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


REAL_DATASET = "real_24V_439N_Microwave"
CANDIDATE = "deterministic"
REFERENCE = "s2"


def number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Not numeric: {description}")
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite number: {description}")
    return float(value)


def summary(values):
    return {
        "mean": statistics.mean(values),
        "sample_SD": statistics.stdev(values) if len(values) > 1 else None,
        "per_fit_values": values,
    }


def decompose(record):
    diagnostic = record["event_occurrence_diagnostic"]
    zero = diagnostic["zero"]
    positive = diagnostic["nonzero"]
    counts = (zero["cells"], positive["cells"])
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts):
        raise ValueError("Expected positive integer group cell counts")
    total = sum(counts)
    if record["report"]["metrics"]["counts"] != total:
        raise ValueError("Diagnostic group counts differ from total evaluation count")
    weights = (counts[0] / total, counts[1] / total)
    losses = (
        number(zero["predictive_NLL"], "zero predictive NLL"),
        number(positive["predictive_NLL"], "positive predictive NLL"),
    )
    contributions = (weights[0] * losses[0], weights[1] * losses[1])
    reconstructed = sum(contributions)
    reported = number(record["report"]["metrics"]["predictive_count_NLL"], "overall predictive NLL")
    difference = reconstructed - reported
    if abs(difference) > 1e-8:
        raise ValueError("Group loss decomposition does not reconstruct reported NLL")
    return {
        "repetition": record["repetition"],
        "source": record["source"],
        "zero_cells": counts[0],
        "positive_cells": counts[1],
        "total_cells": total,
        "zero_fraction": weights[0],
        "positive_fraction": weights[1],
        "zero_conditional_NLL": losses[0],
        "positive_conditional_NLL": losses[1],
        "weighted_zero_NLL": contributions[0],
        "weighted_positive_NLL": contributions[1],
        "reconstructed_overall_NLL": reconstructed,
        "reported_overall_NLL": reported,
        "reconstruction_error": difference,
        "total_predicted_counts": number(diagnostic["total_predicted_counts"], "total predicted counts"),
        "average_precision": number(record["report"]["structure"]["average_precision"], "AP"),
    }


def build(source_path):
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes.decode("utf-8-sig"))
    datasets = [row for row in source["datasets"] if row["dataset"] == REAL_DATASET]
    if len(datasets) != 1:
        raise ValueError("Missing or duplicated real-data dataset")
    records = datasets[0]["raw_verified_records"]
    models = {}
    for model, rows in records.items():
        if not rows or any(row["report"]["dataset"] != REAL_DATASET for row in rows):
            raise ValueError("Unexpected dataset record")
        decomposed = [decompose(row) for row in rows]
        if len({row["repetition"] for row in decomposed}) != len(decomposed):
            raise ValueError("Duplicated model repetition")
        fields = (
            "zero_conditional_NLL", "positive_conditional_NLL", "weighted_zero_NLL",
            "weighted_positive_NLL", "reconstructed_overall_NLL", "reported_overall_NLL",
            "total_predicted_counts", "average_precision",
        )
        models[model] = {
            "training_fits": len(rows),
            "per_fit": decomposed,
            "summary": {field: summary([row[field] for row in decomposed]) for field in fields},
        }
    all_rows = [row for model in models.values() for row in model["per_fit"]]
    count_specs = {(row["zero_cells"], row["positive_cells"], row["total_cells"]) for row in all_rows}
    if len(count_specs) != 1:
        raise ValueError("Model repetitions do not share evaluation cell counts")
    candidate = {row["repetition"]: row for row in models[CANDIDATE]["per_fit"]}
    reference = {row["repetition"]: row for row in models[REFERENCE]["per_fit"]}
    if set(candidate) != set(reference):
        raise ValueError("Candidate and reference repetitions cannot be paired")
    pairs = []
    for repetition in sorted(candidate):
        c, r = candidate[repetition], reference[repetition]
        zero_improvement = r["weighted_zero_NLL"] - c["weighted_zero_NLL"]
        positive_improvement = r["weighted_positive_NLL"] - c["weighted_positive_NLL"]
        improvement = zero_improvement + positive_improvement
        reported_improvement = r["reported_overall_NLL"] - c["reported_overall_NLL"]
        if abs(improvement - reported_improvement) > 1e-8:
            raise ValueError("Paired decomposition does not reconstruct overall improvement")
        pairs.append({
            "repetition": repetition,
            "weighted_zero_improvement": zero_improvement,
            "weighted_positive_improvement": positive_improvement,
            "total_improvement": improvement,
            "reported_total_improvement": reported_improvement,
            "zero_share_of_improvement_pct": 100 * zero_improvement / improvement if improvement else None,
            "positive_share_of_improvement_pct": 100 * positive_improvement / improvement if improvement else None,
            "zero_conditional_reduction_pct": 100 * (1 - c["zero_conditional_NLL"] / r["zero_conditional_NLL"]),
            "positive_conditional_reduction_pct": 100 * (1 - c["positive_conditional_NLL"] / r["positive_conditional_NLL"]),
        })
    zero_mean = statistics.mean(row["weighted_zero_improvement"] for row in pairs)
    positive_mean = statistics.mean(row["weighted_positive_improvement"] for row in pairs)
    total_mean = zero_mean + positive_mean
    zero_count, positive_count, total_count = next(iter(count_specs))
    return {
        "analysis": "posthoc_real_data_predictive_NLL_loss_decomposition",
        "created_at_UTC": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source_path.resolve()),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "declared_status": source.get("status"),
            "declared_repetition_kernel_id": source.get("kernel_id"),
            "original_arrays_or_checkpoints_reverified_by_this_script": False,
        },
        "dataset": REAL_DATASET,
        "groups": {
            "zero_cells": zero_count,
            "positive_cells": positive_count,
            "total_cells": total_count,
            "zero_fraction": zero_count / total_count,
            "positive_fraction": positive_count / total_count,
            "definition": "Observed TEST count equal to zero versus strictly positive",
        },
        "formula": "overall_NLL = (n_zero/n_all)*NLL_zero + (n_positive/n_all)*NLL_positive",
        "models": models,
        "paired_comparison": {
            "candidate": CANDIDATE,
            "candidate_display_name_ko": "연속 연결 가중치 모형",
            "reference": REFERENCE,
            "reference_display_name_ko": "S²GCSL",
            "positive_improvement_definition": "reference_NLL minus candidate_NLL; positive means candidate lower loss",
            "per_fit": pairs,
            "summary": {
                "weighted_zero_improvement": summary([row["weighted_zero_improvement"] for row in pairs]),
                "weighted_positive_improvement": summary([row["weighted_positive_improvement"] for row in pairs]),
                "total_improvement": summary([row["total_improvement"] for row in pairs]),
                "aggregate_zero_share_of_mean_improvement_pct": 100 * zero_mean / total_mean if total_mean else None,
                "aggregate_positive_share_of_mean_improvement_pct": 100 * positive_mean / total_mean if total_mean else None,
                "both_groups_improved_in_every_fit": all(
                    row["weighted_zero_improvement"] > 0 and row["weighted_positive_improvement"] > 0 for row in pairs
                ),
            },
        },
        "checks": {
            "maximum_absolute_reconstruction_error": max(abs(row["reconstruction_error"]) for row in all_rows),
            "reconstruction_error_tolerance": 1e-8,
            "same_cell_counts_all_models_and_fits": True,
            "THP_fits": models.get("thp", {}).get("training_fits"),
            "new_training_runs": 0,
            "GPU_used": False,
            "source_or_manuscript_modified": False,
        },
        "type_level_decomposition": {
            "status": "not_available_from_saved_aggregate_report",
            "required_for_further_analysis": [
                "Original TEST observations with event-type axis",
                "Selected-model predictions.npz predictive_log_mass with the same axes",
            ],
            "values_imputed_or_reconstructed_without_arrays": False,
        },
        "interpretation_ko": [
            "연속 연결 가중치 모형의 S²GCSL 대비 NLL 개선은 0인 관측과 양수 관측 모두에서 확인된다.",
            "따라서 전체 개선이 0인 관측에 대한 개선만으로 발생했다는 설명은 이 집계 결과와 맞지 않는다.",
            "이는 손실이 어디서 개선되는지에 대한 기술적 분해이지, AP 차이나 특정 아키텍처의 인과적 효과를 규명한 결과는 아니다.",
        ],
        "limitations_ko": [
            "테스트 발생 여부에 따라 나눈 사후 탐색 분석이며 모형·임계값·평가 기준을 선택하는 데 사용하지 않는다.",
            "세 반복은 동일 자료 및 분할의 학습 시드 반복이고 독립적인 새 데이터 반복이 아니다.",
            "THP는 한 번 적합한 결과이며 세 번의 독립 적합처럼 취급하지 않는다.",
            "소스 JSON의 기존 집계값만 읽었으며 이 단계에서는 원본 예측 배열이나 체크포인트를 재검증하지 않았다.",
            "모형 입력, 예측 함수, 규제 방식의 차이가 있어 연결 표현 자체의 효과로 단정할 수 없다.",
            "연결 제거와 Integrated Gradients는 아직 수행하지 않았고, 모형 의존도와 실제 인과관계도 구별해야 한다.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must differ")
    result = build(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "saved_new_analysis", "output": str(args.output.resolve()),
                      "mean_NLL_improvement": result["paired_comparison"]["summary"]["total_improvement"]["mean"],
                      "zero_share_pct": result["paired_comparison"]["summary"]["aggregate_zero_share_of_mean_improvement_pct"],
                      "positive_share_pct": result["paired_comparison"]["summary"]["aggregate_positive_share_of_mean_improvement_pct"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
