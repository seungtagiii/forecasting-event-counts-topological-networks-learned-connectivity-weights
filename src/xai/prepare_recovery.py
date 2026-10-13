"""Build explicit, minimal output download whitelists (no network)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["metadata", "selected"], required=True)
    parser.add_argument("--recovered", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    names = {row["name"] for row in inventory["files"]}
    chosen = set()
    if args.stage == "metadata":
        for name in names:
            if "simulation_" in name:
                continue
            leaf = name.rsplit("/", 1)[-1]
            if "real_24V_439N_Microwave/" in name and leaf in {
                "dataset.json", "evaluation_labels.npz", "observations.npz",
                "validation_selection.json", "completion.json"}:
                chosen.add(name)
            if leaf in {"protocol.json", "repeat_protocol.json", "executed_source_manifest.json",
                        "data_blueprint.json", "materialization_manifest.json"}:
                chosen.add(name)
            if "source/" in name and leaf in {"models_v2.py", "study_v2.py", "datasets_v2.py",
                                             "minimum_repeat.py", "evaluation.py"}:
                chosen.add(name)
    else:
        if args.recovered is None:
            raise ValueError("Recovered metadata required")
        for selection in args.recovered.rglob("validation_selection.json"):
            if "real_24V_439N_Microwave" not in str(selection):
                continue
            rows = json.loads(selection.read_text(encoding="utf-8"))
            for model in ["deterministic", "s2"]:
                row = rows[model]
                root = selection.parent.relative_to(args.recovered).as_posix()
                chosen.update({f"{root}/{row['fit_path']}/best.pt", f"{root}/{row['fit_path']}/fit.json",
                               f"{root}/evaluations/{model}/predictions.npz",
                               f"{root}/evaluations/{model}/result.json"})
        if not chosen or chosen - names:
            raise ValueError("Missing explicitly selected model files")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(sorted(chosen), stream, indent=2)
    print(json.dumps({"status": "whitelist_saved", "stage": args.stage, "files": len(chosen)}))


if __name__ == "__main__":
    main()
