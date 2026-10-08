"""Download the pinned public dataset and verify every file before saving it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request


DATASET_COMMIT = "1858e9c39d90a524e7d8ff8b9456f4b1f231ff03"
CODE_COMMIT = "8675af0bc532c7f138158ded62e8cf303e9af655"
EXPECTED = {
    "ttpm.py": ("65bf5a349c22b0c4916deff3fe7a63bee8f9fe6bf3dd28d70cdd8e89794478b5", 16382),
    "gcastle_LICENSE": ("75b52537769ab89be6be947d14ca742e79938859094563a8a3f792e7137bc9d8", 11390),
    "dataset_README.md": ("970e77ee3c5de36f79a937ce8ea838ec9cae2c5a552385e761c55fbea08cc5ea", 2797),
    "24V_439N_Microwave.tar.gz": ("fad2fdb1e4095b7f50dee89725e5a6d6568f370060704480aac5d00626550de1", 324945),
    "Alarm.csv": ("d91028f43d7c324a0883d07920a3023c4a1631ce25031badbc31223402305827", 1257470),
    "DAG.npy": ("0ad422aeeafa3922aa4e8a35904d4406fd0387fe2cde0a2d7eb10c019ed9afa3", 4736),
    "Topology.npy": ("577c49ecf388533f61d25513ccce93fdf527de2ba915204ce93c295521599a93", 1541896),
}
URLS = {
    "ttpm.py": f"https://raw.githubusercontent.com/huawei-noah/trustworthyAI/{CODE_COMMIT}/gcastle/castle/algorithms/ttpm/ttpm.py",
    "gcastle_LICENSE": f"https://raw.githubusercontent.com/huawei-noah/trustworthyAI/{CODE_COMMIT}/LICENSE",
    "dataset_README.md": f"https://raw.githubusercontent.com/gcastle-hub/dataset/{DATASET_COMMIT}/README.md",
    "24V_439N_Microwave.tar.gz": f"https://raw.githubusercontent.com/gcastle-hub/dataset/{DATASET_COMMIT}/alarm/24V_439N_Microwave.tar.gz",
}
ARCHIVE_MEMBERS = {name: f"24V_439N_Microwave/{name}" for name in ("Alarm.csv", "DAG.npy", "Topology.npy")}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "topological-event-counts-reproduction"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def verify(name: str, payload: bytes) -> dict:
    digest = hashlib.sha256(payload).hexdigest()
    expected_digest, expected_size = EXPECTED[name]
    if digest != expected_digest or len(payload) != expected_size:
        raise ValueError(f"Integrity check failed for {name}")
    return {"sha256": digest, "bytes": len(payload)}


def save_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in URLS.items():
        payload = fetch(url)
        records[name] = {"url": url, **verify(name, payload)}
        save_new(root / name, payload)

    archive_path = root / "24V_439N_Microwave.tar.gz"
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for name, expected_member in ARCHIVE_MEMBERS.items():
            member = members.get(expected_member)
            if member is None or PurePosixPath(member.name).name != name:
                raise ValueError(f"Expected archive member missing: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {name}")
            payload = source.read()
            records[name] = {"archive_member": expected_member, **verify(name, payload)}
            save_new(root / name, payload)

    manifest = {"dataset_commit": DATASET_COMMIT, "sources": records}
    save_new(root / "source_manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    print(json.dumps({"status": "verified", "dataset_commit": DATASET_COMMIT,
                      "files_verified": len(records), "output": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
