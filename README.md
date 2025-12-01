# IDH Prediction (Stage III)

Time-series and tabular baselines for Intradialytic Hypotension (IDH) using EMR features. Data is not included; see `DATA_GUIDE.md` for required files, columns, and placement.

## Project Layout
- `VI.MLP/` tabular MLP notebooks and script (`run_mlp_cv.py`).
- `utils/prepare_tabular_splits.py` time-based split (Train 2016-2018, Val 2019, Test 2020).
- `utils/run_tabular_experiments.py` tabular MLP runs with A/B/C datasets and Wilcoxon.
- `utils/run_sequence_gru.py` GRU (window=8) sequence benchmark.
- `paper.md` study summary and conclusions.
- `DATA_GUIDE.md` how to place data (filenames, columns) before running.

## Quickstart
1) Place data per `DATA_GUIDE.md` under `dataset/` (raw) and run:
   ```bash
   python utils/prepare_tabular_splits.py
   ```
2) Tabular MLP (A/B/C, time split):
   ```bash
   python utils/run_tabular_experiments.py
   ```
   Outputs metrics and Wilcoxon to `results/`.
3) Time-series GRU (Dataset B, window=8):
   ```bash
   python utils/run_sequence_gru.py
   ```
   Outputs metrics to `results/`.

## Data Handling
- No PHI or data is tracked in this repo/zip. Add data locally only.
- Keep paths relative; avoid absolute mounts.

## Notes
- `../idh_no_data.zip` contains this codebase without data/checkpoints for safe GitHub upload.
- Additional models (Informer, CatBoost, TabTransformer) can be added by reusing the time split and feature set defined in `DATA_GUIDE.md`.
