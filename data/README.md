# Data provenance

The real 24V_439N_Microwave raw archive is not redistributed in this repository. Download it from the immutable gcastle-hub revision using `../scripts/fetch_public_data.py`; the script verifies SHA-256 and byte counts against `source_manifest.json` before storing files under `data/raw/`.

The two compact simulation inputs under `prepared/` were generated for the reported development study. They are retained with the repository so that the previously reported simulation comparisons can be inspected without retaining Kaggle run outputs. They are not new independent simulation repetitions.

The public source manifest records pinned URLs and hashes for the archive, extracted data files, dataset README, gCastle TTPM source, and its license. The reference DAG is supplied as a comparison label only, not asserted to be verified Granger-causal truth.
