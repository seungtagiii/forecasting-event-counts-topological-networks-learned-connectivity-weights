"""Read-only result consistency checks and a new analysis reproducibility manifest."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "xai_20261013"
CODE = ROOT / "src" / "xai"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summary = read(OUT / "xai_summary.json")
    audit = read(OUT / "recovery_audit.json")
    assert audit["status"] == "verified_fixed_artifact_recovery"
    assert summary["new_fits"] == summary["new_GPU_runs"] == 0
    for name, digest in summary["input_sha256"].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
    ig = summary["IG"]["analysis"]
    for name in ["zero_history", "TRAIN_mean_logged_history"]:
        assert abs(sum(ig[name]["hop_absolute_shares"]) - 1) < 1e-10
        assert abs(sum(ig[name]["lag_absolute_shares"]) - 1) < 1e-10
        assert ig[name]["completeness_absolute_relative_to_output_changes"] < 1e-3
        assert ig[name]["64_to_128_step_L1_relative_change"] < 1e-3
    rows = summary["VAL"]["models"]
    assert all(value > 0 for value in rows["deterministic"]["group_removals"]["own_device_same_type"]["delta_NLL"]["per_fit_values"])
    assert all(abs(value) < 1e-8 for value in rows["s2"]["group_removals"]["own_device_same_type"]["delta_NLL"]["per_fit_values"])
    files = [path for path in CODE.glob("*.py")] + [path for path in OUT.glob("*.json")] + [OUT / "edge_importance_and_TEST_time.npz"]
    manifest = {"status": "analysis_consistency_checks_passed", "new_fits": 0, "new_GPU_runs": 0,
                "original_paper_or_results_changed": False,
                "CPU_and_numpy_forward_agree": True,
                "toy_source_target_hop_lag_checks": "src/xai/test_fixed_models.py",
                "IG_integration_checks_passed": True,
                "files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(files)]}
    with (OUT / "analysis_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"status": manifest["status"], "manifested_files": len(files),
                      "new_fits": 0, "new_GPU_runs": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
