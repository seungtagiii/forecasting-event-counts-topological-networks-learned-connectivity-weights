# Frozen-model exploratory analysis

This directory adds explanatory analyses to the completed real-data comparison. It never fits a model, submits a Kaggle notebook, tunes a threshold, or changes the original evaluations. The internal model label `deterministic` denotes the **continuous connectivity weight model**, not a sixth model.

Published numerical snapshots are under `results/xai_20261013/`. Local recovery and regenerated results use the ignored `outputs/xai_20261013/` directory. Publication edits to code change artifact locations and the toy test's source lookup only. Absolute workspace paths in the published audit are normalized; original-run SHA-256 hashes and publication hashes are distinguished in `publication_manifest.json`.

## Scope and limitations

- Three selected fits each of the continuous model and the fixed-window S²GCSL implementation; one dataset and one chronological split.
- Frozen removals use the original uniform 25,000 validation node-time pairs and all 24 targets. Each source→target removal masks all its hops and lags. Removal measures model reliance, not an intervention in the real system or the performance of a retrained ablation.
- TEST loss decomposition reads original saved forecasts only. It does not select a model or explanation technique.
- Integrated Gradients uses fit 0, 128 uniformly sampled validation pairs plus up to 128 pairs with observed events, selected without regard to prediction success. Zero-history and TRAIN-mean logged-history references are compared; 64 and 128 integration intervals check numerical stability. This case set is outcome-enriched and is not population-weighted.
- Same-type history is allowed by the continuous model but excluded by this S²GCSL implementation. AP evaluates off-diagonal type pairs, so it does not measure that history's usefulness.
- Nonlinear decoding, input aggregation, self-type input availability and regularization are not matched. Results do not identify continuous weighting alone as the cause of improved forecasts.

## CPU checks without Kaggle access

Install the repository's numerical dependencies using your preferred environment. The completed analysis used Python 3.12, NumPy 2.2.6 and **PyTorch 2.6.0+cpu**, with two CPU threads and no GPU. The recovery helper additionally requires the official `kaggle` package. Credentials must stay outside the repository.

```powershell
python src/xai/ig_numpy.py
python src/xai/test_fixed_models.py
python src/xai/loss_decomposition.py --input results/verified_repetition_results.json --output outputs/xai_20261013/loss_decomposition.json
```

The toy test checks node/source/target/hop/lag axes and fast removals against explicit input masking. The IG self-test checks analytic gradients against finite differences and the attribution completeness property.

## Recover the already completed outputs (optional)

Full original predictions and selected checkpoints are deliberately not committed. Reanalysis requires access to the retained outputs of these two completed notebooks:

- Original fits: `seankim0736/tnpar-five-20261004-3af9979e`.
- Additional training repetitions: `seankim0736/tnpar-minimum-20261004-1616721f`.

Authenticate locally using the official Kaggle CLI. Do not paste credentials into code or commit them. The helper authenticates through the SDK but suppresses signed URLs, API responses, logs and raw exception messages. It performs only output listing and explicitly whitelisted downloads. The SDK does not pin a historical version for this endpoint: **source/data/checkpoint audits must pass before analysis**, and unavailable or changed artifacts are not replaced with new fits.

Create `outputs/xai_20261013/` first. For each stage `original` and `repeated`, list outputs, create a metadata whitelist, and download it:

```powershell
python src/xai/recover_outputs.py --kernel original --mode inventory --output outputs/xai_20261013/kaggle_original_inventory.json
python src/xai/prepare_recovery.py --inventory outputs/xai_20261013/kaggle_original_inventory.json --stage metadata --output outputs/xai_20261013/original_metadata_whitelist.json
python src/xai/recover_outputs.py --kernel original --mode download --selected outputs/xai_20261013/original_metadata_whitelist.json --output outputs/xai_20261013/recovered_original_metadata
```

Repeat these commands with `original` replaced by `repeated`. Then use the recovered validation selections to whitelist only the two models' selected fits and forecasts:

```powershell
python src/xai/prepare_recovery.py --inventory outputs/xai_20261013/kaggle_original_inventory.json --stage selected --recovered outputs/xai_20261013/recovered_original_metadata --output outputs/xai_20261013/original_selected_whitelist.json
python src/xai/recover_outputs.py --kernel original --mode download --selected outputs/xai_20261013/original_selected_whitelist.json --output outputs/xai_20261013/recovered_original_models
```

Again repeat with `original` replaced by `repeated`. Approximately 0.5 GB of forecasts/checkpoints were recovered in the completed analysis. Do not overwrite or delete original outputs. New destinations are required; helpers fail rather than overwrite existing result files.

## Audit and analyze

Run the following from the repository root after both recoveries. PowerShell backticks below are line continuations.

```powershell
python src/xai/audit_recovery.py `
  --original-metadata outputs/xai_20261013/recovered_original_metadata `
  --repeat-metadata outputs/xai_20261013/recovered_repeated_metadata `
  --original-models outputs/xai_20261013/recovered_original_models `
  --repeat-models outputs/xai_20261013/recovered_repeated_models `
  --repository . --verified results/verified_repetition_results.json `
  --output outputs/xai_20261013/recovery_audit.json
python src/xai/analyze_fixed_models.py --phase analysis
python src/xai/analyze_fixed_models.py --phase attribution
python src/xai/summarize_xai.py
python src/xai/finalize_audit.py
```

The audit compares executed source manifests, canonical array hashes, validation selections, chosen updates, frozen graph states, and stored prediction arithmetic against previous verified records. Analysis additionally compares CPU checkpoint forwards with 1,024 original saved test forecasts, and verifies the NumPy IG forward against the PyTorch decoder. Small CPU/GPU roundoff is recorded rather than interpreted as a model change.

## Files

- `loss_decomposition.py`: zero/positive observation decomposition from previous verified JSON.
- `analyze_fixed_models.py`: predefined validation input groups, all source→target removals, original TEST type decomposition, IG, and label-blind gate/first-layer compensation check.
- `ig_numpy.py`: analytic gradient and trapezoidal IG for the fixed two-layer Tanh/softplus decoder.
- `audit_recovery.py`, `finalize_audit.py`: provenance and consistency checks.
- `recover_outputs.py`, `prepare_recovery.py`: existing-output recovery only; no submission or execution methods.
- `summarize_xai.py`, `test_fixed_models.py`: summary generation and small CPU checks.

The compensation check alters parameters only in memory and restores them; its forecasts are not new primary results. Published AP, thresholds and fitted checkpoints are unchanged.
