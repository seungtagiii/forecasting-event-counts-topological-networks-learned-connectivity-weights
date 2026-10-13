"""Read/download only existing Kaggle output files; never run a notebook.

Only whitelisted metadata is printed/saved. Credentials, signed URLs, raw API
responses, logs and exception text are never printed or saved.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import socket

KERNELS = {
    "original": "seankim0736/tnpar-five-20261004-3af9979e",
    "repeated": "seankim0736/tnpar-minimum-20261004-1616721f",
}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def safe_relative(name):
    name = str(name).replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name or not name:
        raise ValueError("Unsafe output path")
    return name


def inventory(api, kernel):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    rows, token = [], None
    with api.build_kaggle_client() as client:
        for _ in range(30):
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = kernel.split("/")
            api._set_paging(request, 100, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                name = safe_relative(item.file_name)
                size = getattr(item, "size", None)
                rows.append({"name": name, "bytes": int(size) if isinstance(size, (int, float)) else None,
                             "item": item})
            token = response.next_page_token
            if not token:
                return rows
    raise RuntimeError("Inventory page cap exceeded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=KERNELS, required=True)
    parser.add_argument("--mode", choices=["inventory", "download"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected", type=Path)
    args = parser.parse_args()
    stage = "authentication"
    try:
        socket.setdefaulttimeout(45)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from kaggle import api
            if not api._authenticated:
                api.authenticate()
            stage = "output_inventory"
            rows = inventory(api, KERNELS[args.kernel])
            clean = [{"name": row["name"], "bytes": row["bytes"]} for row in rows]
            if args.mode == "inventory":
                write_new(args.output, {"kernel_id": KERNELS[args.kernel], "files": clean,
                                       "new_GPU_runs": 0, "uploads": 0,
                                       "API_version_selection_not_supported": True})
                result = {"status": "inventory_saved", "file_count": len(rows),
                          "output": str(args.output), "new_GPU_runs": 0}
            else:
                if args.selected is None:
                    raise ValueError("Explicit output whitelist required")
                wanted = set(json.loads(args.selected.read_text(encoding="utf-8")))
                found = {row["name"]: row for row in rows}
                if wanted - found.keys():
                    raise ValueError("Requested file missing from outputs")
                args.output.mkdir(parents=True, exist_ok=False)
                receipts = []
                import requests
                stage = "selected_file_download"
                for name in sorted(wanted):
                    row = found[name]
                    if not str(row["item"].url).startswith("https://"):
                        raise ValueError("Expected HTTPS output URL")
                    target = args.output.joinpath(*PurePosixPath(safe_relative(name)).parts)
                    if not target.resolve().is_relative_to(args.output.resolve()):
                        raise ValueError("Output escaped destination")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest, count = hashlib.sha256(), 0
                    with requests.get(row["item"].url, stream=True, timeout=(20, 60)) as response:
                        response.raise_for_status()
                        with target.open("xb") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk:
                                    count += len(chunk)
                                    if count > 500 * 1024 * 1024:
                                        raise ValueError("Single-file size budget exceeded")
                                    digest.update(chunk)
                                    stream.write(chunk)
                    receipts.append({"name": name, "bytes": count, "sha256": digest.hexdigest()})
                write_new(args.output / "recovery_receipt.json", {"kernel_id": KERNELS[args.kernel],
                          "files": receipts, "new_GPU_runs": 0, "uploads": 0,
                          "original_outputs_unchanged": True})
                result = {"status": "selected_outputs_downloaded", "file_count": len(receipts),
                          "total_bytes": sum(r["bytes"] for r in receipts), "new_GPU_runs": 0}
        print(json.dumps(result, ensure_ascii=False))
    except (Exception, SystemExit) as error:
        print(json.dumps({"status": "failed", "stage": stage,
                          "error_type": type(error).__name__, "raw_error_suppressed": True,
                          "new_GPU_runs": 0}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
