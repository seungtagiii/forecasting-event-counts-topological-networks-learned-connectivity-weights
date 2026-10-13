"""Make a compact, auditable summary without modifying paper/results."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "xai_20261013"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def mean_sd(values):
    return {"mean": statistics.mean(values), "sample_SD": statistics.stdev(values),
            "per_fit_values": values}


def main():
    data = read("frozen_group_summary.json")
    rows = data["VAL"]
    models = {}
    for name in ["deterministic", "s2"]:
        current = [row for row in rows if row["model"] == name]
        groups = {}
        for group in current[0]["removed_groups"]:
            key = group["group"]
            values = [next(g for g in row["removed_groups"] if g["group"] == key) for row in current]
            groups[key] = {metric: mean_sd([value[metric] for value in values])
                           for metric in ["NLL", "delta_NLL", "MAE", "RMSE", "positive_NLL", "zero_NLL"]}
        models[name] = {
            "original_VAL": {metric: mean_sd([row["original_VAL"][metric] for row in current])
                             for metric in ["NLL", "MAE", "RMSE", "positive_NLL", "zero_NLL"]},
            "group_removals": groups,
            "gate_importance_spearman": mean_sd([row["edge_removal"]["score_importance_spearman"] for row in current]),
            "checkpoint_forward_max_error": max(row["checkpoint_prediction_check"]["max_abs_difference"] for row in current)}
    gap = [next(row["original_VAL"]["NLL"] for row in rows if row["model"] == "s2" and row["repetition"] == i) -
           next(row["original_VAL"]["NLL"] for row in rows if row["model"] == "deterministic" and row["repetition"] == i)
           for i in range(3)]
    after = [next(row["original_VAL"]["NLL"] for row in rows if row["model"] == "s2" and row["repetition"] == i) -
             next(g["NLL"] for row in rows if row["model"] == "deterministic" and row["repetition"] == i
                  for g in row["removed_groups"] if g["group"] == "own_device_same_type") for i in range(3)]
    subtype = read("TEST_type_decomposition.json")["rows"]
    types = []
    for target in range(24):
        values = {name: [next(t for t in row["by_target"] if t["target"] == target)
                         for row in subtype if row["model"] == name] for name in models}
        improvements = [values["s2"][i]["NLL"] - values["deterministic"][i]["NLL"] for i in range(3)]
        types.append({"target": target, "TEST_events": values["s2"][0]["events"],
                      "positive_cells": values["s2"][0]["positive_cells"],
                      "mean_NLL_improvement": statistics.mean(improvements),
                      "mean_NLL_sum_improvement": statistics.mean(improvements) * values["s2"][0]["cells"],
                      "all_fits_improve": all(value > 0 for value in improvements),
                      "per_fit_improvements": improvements})
    total = sum(row["mean_NLL_sum_improvement"] for row in types)
    for row in types:
        row["share_of_total_NLL_improvement"] = row["mean_NLL_sum_improvement"] / total
    sorted_types = sorted(types, key=lambda row: row["mean_NLL_sum_improvement"], reverse=True)
    ig = read("input_attribution.json")
    loss = read("loss_decomposition.json")
    source_files = ["analysis_protocol.json", "recovery_audit.json", "loss_decomposition.json",
                    "frozen_group_summary.json", "TEST_type_decomposition.json", "input_attribution.json"]
    summary = {
        "status": "completed_exploratory_frozen_model_analysis",
        "new_fits": 0, "new_GPU_runs": 0, "paper_modified": False, "original_AP_and_thresholds_unchanged": True,
        "VAL": {"pairs": 25000, "count_cells": 600000, "same_original_validation_sample": True,
                "models": models, "original_S2_minus_continuous_NLL_gap": mean_sd(gap),
                "S2_minus_continuous_gap_after_own_history_removal": mean_sd(after),
                "relative_gap_reduction_descriptive_only": 1 - statistics.mean(after) / statistics.mean(gap)},
        "TEST": {"role": "original_saved_forecasts_no_new_model_selection",
                 "types_improved_on_mean": sum(row["mean_NLL_improvement"] > 0 for row in types),
                 "types_improved_in_all_fits": sum(row["all_fits_improve"] for row in types),
                 "by_type": types, "top_3_share_of_total_NLL_improvement": sum(row["share_of_total_NLL_improvement"] for row in sorted_types[:3]),
                 "zero_positive_decomposition_source": "loss_decomposition.json"},
        "IG": ig,
        "elapsed_analysis_seconds": data["elapsed_seconds"] + ig["elapsed_seconds"],
        "findings_ko": [
            "양수 관측과 0인 관측 모두에서 연속 모형의 예측 NLL 개선이 확인되었다.",
            "연속 모형의 같은 장비·같은 경보 유형 과거 입력을 제거하면 검증 손실이 크게 증가한다.",
            "이 자기 이력의 기여는 비대각 유형 간 연결만 평가하는 기존 AP에 포함되지 않는다.",
            "원래 게이트 순위는 고정 모형에서 연결을 제거했을 때의 손실 변화 순위와 일치하지 않는다.",
            "이 결과는 연결 표현 자체만의 우월성이 아니라 입력 활용과 예측 함수의 차이에 일관된 탐색적 증거이다."],
        "limitations_ko": [
            "단일 실제 데이터와 기존 분할의 세 학습 반복이며 새로운 독립 검증이 아니다.",
            "같은 유형의 과거 입력 허용 여부, 비선형성, 홉·시차 구분, 규제 방식이 모형 간 다르다.",
            "입력 제거는 고정 모형의 의존도이며 학습 가능한 대체 모형의 성능이나 실제 인과 효과가 아니다.",
            "IG는 첫 학습 반복의 검증 사례 256개 이내에서 수행했으며 사건 발생 사례를 보충해 모집단 가중치로 해석하지 않는다.",
            "참고 연결은 검증된 실제 인과 정답이 아니며 참고/비참고 그룹은 연결 개수와 발생 빈도도 다르다.",
            "입력 제거 그룹은 겹치고 비선형 상호작용이 있어 개선 비율을 독립 효과의 합으로 해석하지 않는다.",
            "성능차이의 정확한 인과적 원인을 확정하려면 입력·규제·학습 조건을 맞춘 별도 적합 비교가 필요하다."],
        "input_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in source_files}}
    with (OUT / "xai_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": summary["status"], "VAL_original_NLL": {name: models[name]["original_VAL"]["NLL"] for name in models},
                      "VAL_own_history_removed_NLL": {name: models[name]["group_removals"]["own_device_same_type"]["NLL"] for name in models},
                      "VAL_all_history_removed_NLL": {name: models[name]["group_removals"]["all_history"]["NLL"] for name in models},
                      "raw_weight_vs_removal_importance_spearman": {name: models[name]["gate_importance_spearman"] for name in models},
                      "VAL_gap_reduction_descriptive": summary["VAL"]["relative_gap_reduction_descriptive_only"],
                      "TEST_types_improved": summary["TEST"]["types_improved_on_mean"],
                      "TEST_types_all_fits_improved": summary["TEST"]["types_improved_in_all_fits"],
                      "TEST_top3_share": summary["TEST"]["top_3_share_of_total_NLL_improvement"],
                      "IG": {key: {name: value for name, value in row.items()
                                   if name != "source_target_absolute_share"}
                             for key, row in ig["analysis"].items()},
                      "compensation": ig["gate_decoder_compensation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
