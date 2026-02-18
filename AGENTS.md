# Donald Trump Speaker-Specific ASD Playbook  
_Status: 18 Feb 2026 (post-repo audit)_

---

## 0. Repo Status Snapshot
- `train.py:1-6` is a stub, so there is no end-to-end CLI that runs the baseline workflows mandated in §5 or produces the logging artifacts from §§6–9.
- `visualize.py:10-15` imports `ResidualFusion` from `train_residual`, but that module is missing; fusion-mode UMAP generation fails immediately unless `FeatureConfig.extract_fusion_features` is turned off.
- `test.py` depends on `utils_combined.AudioSVMClassifier`, which never sets `self.feature`, never imports `librosa`, and calls unbound names (e.g., `single_audio_feature_extraction`); inference will crash before producing predictions.
- `e2e_utils.AudioSVMClassifier` handles SSL feature extraction, but it ignores several playbook requirements (no caching, no run directory writing, no hyperparam sweeps).
- `deepsvdd_runner.py` can call the vendored Deep-SVDD repo but currently just trains for a fixed epoch count and writes one `metrics.json`; it does not emit per-dataset JSONs, hyperparam logs, or the `runs.jsonl` append described later in this playbook.
- No code writes `results/*.json`, `runs.jsonl`, `hyperparam_trials.jsonl`, `outliers_*.csv`, or fusion weight files—the directory template in Appendix B is aspirational until supporting scripts are implemented.
- `scripts/run_baselines.py` now exposes the Phase‑1 CLI and a working `extract` subcommand that materializes the Donald Trump embedding caches plus `runs/<run-id>/extract_summary.json` and `runs.jsonl` entries; `ocsvm`, `deepsvdd`, `evaluate`, and `fusion` remain stubs until their phases land.

> **Action items:** implement an orchestrator script (or extend `train.py`) to run the OC-SVM/Deep SVDD baselines, add a reusable embedding cache, restore a working `AudioSVMClassifier` API, drop the missing fusion dependency or add `ResidualFusion`, and extend the Deep SVDD runner plus evaluation utilities to honor the logging contract below.

---

## 1. Environment & Dependencies
1. Clone `speaker-specific-ASD`.
2. Create/activate env:
   ```bash
   conda env create -f environment.yaml
   conda activate speaker-specific-asd
   pip install -r requirements.txt
   ```
3. Sanity check:
   ```bash
   python -c "import torch, torchaudio, s3prl, speechbrain, umap"
   ```
4. Install the vendored Deep-SVDD repo in editable mode (already cloned under `./Deep-SVDD-PyTorch`):
   ```bash
   pip install -e Deep-SVDD-PyTorch
   ```

> **18 Feb 2026 update:** Rebuilt the env with `python==3.10` (pyannote.audio>=4.0.3 now requires ≥3.10) and installed the requirements sequentially with `pip --use-deprecated=legacy-resolver` to avoid the resolver depth limit. Added a minimal `pyproject.toml` + `setup.cfg` inside `Deep-SVDD-PyTorch/` so `pip install -e Deep-SVDD-PyTorch` succeeds, and verified `python -c "import torch, torchaudio, s3prl, speechbrain, umap"` inside `speaker-specific-asd`. Pip warns that `hdbscan==0.8.37` still declares `numpy<2`, but we keep `numpy==2.2.6` to satisfy pyannote; the warning is benign because wespeaker runs against numpy 2.x in practice. Always activate the env (`conda activate speaker-specific-asd`) or wrap commands with `conda run -n speaker-specific-asd …` before touching repo scripts so the validated toolchain stays in use.

---

## 2. Data Inventory & Loader Behavior

| Dataset | Audio Dir | Protocol CSV | Tag |
|---------|-----------|--------------|-----|
| Famous Figures (train bona fide) | `/data/FF_V2/FF_V2/` | `/data/FF_V2/FF_V2_meta_Data/protocol_Donald_Trump_v1.csv` | `ff` |
| In-The-Wild | `/data/Data/ds_wild/release_in_the_wild/` | `/data/Data/ds_wild/protocols/meta.csv` | `itw` |
| DeepfakeEval 2024 | `/data/Data/Deepfake_Eval_2024/audio-data` | `/data/Data/Deepfake_Eval_2024/protocols/final_Deepfakeeval2024_Speakerverification.csv` | `DFEval2024` |

`dataset_loader.AudioDataset` already merges arbitrary protocol lists, injects missing columns, normalizes `Speaker` values, filters to Donald Trump, and pads/clips each waveform to 64,600 samples at 16 kHz (`dataset_loader.py:15-74`). Keep `FeatureConfig.protocol_tags` in sync with the table because those tags drive the synthesized `Source` column.

---

## 3. Feature Extractors & Configs

| Extractor | Code | Output Dim | Notes |
|-----------|------|------------|-------|
| SSL (XLS-R 300M) | `ssl_utils.SSLModel` | 1,024 | `S3PRLUpstream` + `Featurizer`, mean-pool per layer. |
| SpeechBrain ECAPA | `speaker_utils.Speaker_Model` | 192 | `speechbrain/spkrec-ecapa-voxceleb` by default. |
| Fusion (Residual) | _blocked_ | Needs a real `ResidualFusion` module plus checkpoint; currently broken. |

`config.FeatureConfig` exposes toggles for SSL vs. speaker vs. fusion embeddings, SSL layer index, UMAP hyperparams, and output dirs (`config.py:4-57`). Update the config before each run to specify the target dataset(s), extractor, and batch size. Do **not** rely on `extract_fusion_features=True` until `train_residual.py` exists.

---

## 4. Deep SVDD Integration Plan
1. Ensure `Deep-SVDD-PyTorch/src` is on `PYTHONPATH` (the repo-level runner already injects it).
2. Use the provided embedding networks (`Deep-SVDD-PyTorch/src/networks/embedding_net.py`) instead of editing their CIFAR/MNIST heads.
   - XLS-R head: `embed_mlp_1024` (1024 -> 512 -> 128 -> 64).
   - SpeechBrain head: `embed_mlp_192` (192 -> 128 -> 64).
3. Use the **soft-boundary** objective with `nu=0.1` so tolerances align with OC-SVM sweeps.
4. Stick to Adam (or AMSGrad) with lr `1e-4`, weight decay `1e-6`, batch `64`.
5. Training schedule: **exactly 40 epochs** (per user directive—no early stopping). Track validation EER each epoch, but only stop after the 40th epoch. Save the best checkpoint (center `c`, radius `R`, optimizer state).

---

## 5. Baseline Workflows (to implement in `train.py` or equivalent)

> Finish every baseline (XLS-R + SpeechBrain, each with OC-SVM and Deep SVDD) before attempting threshold extension or fusion experiments.

### 5.1 SSL (XLS-R 300M) + OC-SVM
1. Run the Phase‑1 `extract` workflow to materialize protocol-driven embedding caches (`ff_train.npz`, `ff_val.npz`, `itw_test.npz`, `dfeval_test.npz`) using `e2e_utils.AudioSVMClassifier` for layer-7 mean-pooled features.
2. For each OC-SVM trial, load the cached FF embeddings (rather than scanning folders) and honor the requested pooling mode (`aggregate_emb` / `layer_number`). Persist scaler/SVM artifacts under `/data/Speaker_Specific_Models/xlsr300m_ocsvm_<UTCtimestamp>/Donald_Trump/{scaling_models,svm_models}`.
3. Sweep `nu ∈ {0.05, 0.1, 0.2}` × `gamma ∈ {0.05, 0.1, 0.2}`, recording validation metrics for every trial in `runs/xlsr300m_ocsvm_<timestamp>/hyperparam_trials.jsonl`.
4. Evaluate the selected model on FF validation, ITW, and DFEval using the cached embeddings plus Appendix-A EER helper; write dataset JSONs beneath the run directory and append a summary line to the repo-level `runs.jsonl`.

### 5.2 SSL (XLS-R 300M) + Deep SVDD
1. Feed the cached SSL embeddings into `deepsvdd_runner.py` using `--net-name embed_mlp_1024`.
2. Extend the runner to:
   - Log validation EER each epoch (FF hold-out or ITW subset).
   - Always train for `--epochs 40`.
   - Save the checkpoint with the best validation EER.
   - Emit per-dataset JSONs (`ff.json`, `itw.json`, `dfeval.json`) with `R - ||z - c||` scores and append `runs.jsonl`.

### 5.3 SpeechBrain ECAPA + OC-SVM / Deep SVDD
1. Add a SpeechBrain extraction path (192-D) using `speaker_utils.Speaker_Model`.
2. Mirror the OC-SVM grid from §5.1 and the Deep SVDD schedule from §5.2 (using `embed_mlp_192`).
3. Save everything under `/data/Speaker_Specific_Models/speechbrain_<model>_<timestamp>/...` with the same logging format.

### 5.4 Automation Notes
- `train.py` must become the orchestrator: parse CLI flags for extractor/model/datasets, call the appropriate helper, and centralize logging/tracking.
- When hyperparameter sweeps finish, retrain the selected model on the full FF bona fide set before exporting checkpoints.

---

## 6. Metrics & Logging Contract
For each dataset evaluation (FF validation, ITW, DFEval):

```json
{
  "run_id": "xlsr300m_ocsvm_20260218T1530Z",
  "dataset": "itw",
  "phase": "baseline",
  "threshold": -0.18,
  "num_real": 240,
  "num_fake": 220,
  "accuracy": 0.948,
  "FAR": 0.032,
  "FRR": 0.082,
  "EER": 0.057,
  "eer_threshold": -0.22,
  "timestamp_utc": "2026-02-18T15:41:02Z"
}
```

- Use UTC timestamps.
- Store dataset JSONs under `results/` inside each run directory.
- Append a compact summary line (extractor, model, hyperparameters, validation EER, checkpoints, notes) to the repo-level `runs.jsonl`.

---

## 7. Score Fusion (blocked until fusion code exists)
1. Reserve ~10% of FF bona fide plus an equal spoof set for fusion tuning.
2. Once `ResidualFusion` is implemented, support:
   - Simple average: `0.5 * (zscore_xlsr + zscore_speechbrain)`.
   - Weighted fusion: grid `w1 ∈ {0.3, 0.5, 0.7}`, `w2 = 1 - w1`.
3. Log `fusion_avg.json` and `fusion_weighted.json` plus ROC/score histograms under `plots/`.
4. Until `train_residual.py` exists, document this section as pending and set `FeatureConfig.extract_fusion_features = False` to avoid runtime errors.

---

## 8. Visualization Outputs
- **UMAP (`visualize.py`)**: Works for SSL and SpeechBrain paths. Disable fusion unless you backfill `ResidualFusion`. Keep outputs under `/data/Speaker_Specific_Models/umap_plots/Donald_Trump/...` with filenames `umap_<protocol_tags>_<model>_<desc>.(html|png)`.
- **Heatmaps (`heatmap_vis.py`)**: Static visualization of historic EERs; regenerate after every major benchmark update and version the PDFs/PNGs in `plots/`.
- Capture residual norms (`visualize.py:118-152`) for analytics whenever fusion resumes.

---

## 9. Threshold Extension (Post-Baseline)
_Do not start until §§5–7 are complete and logged._
1. For each dataset, list bona fide samples outside the baseline boundary (OC-SVM decision < threshold, Deep SVDD score < 0). Save `outliers_itw.csv`, `outliers_dfeval.csv`.
2. Sweep thresholds from -0.30 to 0.00 (step 0.02). Target ≥95% recall on ITW + DFEval bona fide while keeping FAR ≤5%.
3. For the chosen relaxed threshold, regenerate metrics JSONs (phase=`"threshold_extension"`) and append new entries to `runs.jsonl`.
4. Archive the relaxed threshold value inside the run directory (e.g., `threshold_extension.json`).

---

## 10. Manual CLI Sequence (until automation lands)
1. Activate env and verify CUDA device availability (`config.py:12` currently defaults to `cuda:1`).
2. Build SSL embeddings (`extracting_features/feature_extract.py`) or speaker embeddings and cache them.
3. Run OC-SVM hyperparam sweeps for each extractor → select best config → retrain full model.
4. Run Deep SVDD for each extractor (soft-boundary, fixed 40 epochs) on the cached embeddings.
5. Evaluate every baseline on FF val, ITW, DFEval; write dataset JSONs and append `runs.jsonl`.
6. Generate UMAP plots (SSL and SpeechBrain) and update heatmaps.
7. When fusion is unblocked, run simple average then weighted fusion and log metrics.
8. Perform threshold extension sweeps and update outlier lists.
9. Sync all artifacts into the timestamped run directory per Appendix B.

---

## 11. Baseline Orchestrator Script Plan — Phase 1 Scope
- **Current operating assumption:** the repo is incomplete by design and we will implement features in controlled phases. **Phase 1** delivers a minimal vertical slice: SSL (XLS-R 300M) embeddings + OC-SVM baselines, logging, and run tracking. _Everything else stays untouched._
- **Script:** `scripts/run_baselines.py` (new file, keeps `train.py` free for legacy workflows). Entry point guarded with `if __name__ == "__main__": main()`.
- **Phase-1 status:** the CLI scaffolding and `extract` subcommand are implemented. Running `python scripts/run_baselines.py --extractor xlsr300m extract` builds the cache hierarchy under `data_cache/<extractor>/Donald_Trump`, writes per-split `.npz` archives, and records summaries under `runs/<run-id>/extract_summary.json` plus `runs.jsonl`. The remaining subcommands are stubbed until the corresponding features are ready.
- **CLI (final):** `argparse` with shared options and subcommands.
  - Global options (apply to all subcommands):
    - `--config CONFIG` (optional YAML/JSON). When omitted, default to `config.FeatureConfig`.
    - `--extractor {xlsr300m,speechbrain}` (required if `--config` missing; otherwise overrides config).
    - `--model {ocsvm,deepsvdd}` (used by `evaluate`/`fusion` in addition to subcommand context).
    - `--device DEVICE` (default `auto`, resolves to `cuda` if `torch.cuda.is_available()` else `cpu`).
    - `--output-root DIR` (default `/data/Speaker_Specific_Models`).
    - `--cache-dir DIR` (default `./data_cache`).
    - `--run-id RUN_ID` (if absent, auto-generate `<extractor>_<model>_<UTCtimestamp>`).
    - `--seed INT` (default `42`).
    - `--log-level {DEBUG,INFO,WARNING,ERROR}` (default `INFO`).
  - Subcommands (Phase‑1 commitments):
    1. `extract` → build SSL embeddings; writes standardized per-split `.npz` archives plus summary logs.
    2. `ocsvm` → run OC-SVM hyperparam sweeps, pick best config, retrain full model, write checkpoints + `hyperparam_trials.jsonl`, append `runs.jsonl`, emit dataset metrics JSONs.
    3. `deepsvdd` → **stub in Phase‑1** (must print “not implemented in Phase‑1” and exit with non-zero status).
    4. `evaluate` → **stub in Phase‑1** (reserved for post-Phase‑1 when multiple models exist).
    5. `fusion` → **stub in Phase‑1** (enable only after `ResidualFusion` lands).
- **Directory structure enforced by the script:**
  ```
  scripts/
    run_baselines.py
  runs/
    <extractor>_<model>_<timestamp>/
      hyperparam_trials.jsonl
      validation_curve.json
  data_cache/
    xlsr300m/
      Donald_Trump/
        ff_train.npz
        ff_val.npz
        itw_test.npz
        dfeval_test.npz
  ```
- **Logging contract:** every subcommand writes a concise summary to STDOUT and mirrors it into a unified `runs.jsonl` at repo root. Per-dataset metrics always land in `results/` under the timestamped run dir, while sweep diagnostics remain in `runs/<...>/hyperparam_trials.jsonl`.
- **Integration points:** `run_baselines.py` should call into existing helpers (`dataset_loader`, `ssl_utils`, `speaker_utils`, `e2e_utils`, `deepsvdd_runner`) rather than re-implementing the logic. Define lightweight wrapper classes/modules if any helper needs additional hooks (e.g., caching embeddings or writing metrics).
- **Embedding cache schema (locked):** store **one `.npz` per dataset split** inside `<cache-dir>/<extractor>/<speaker>/`, e.g.
  - `ff_train.npz`, `ff_val.npz`, `itw_test.npz`, `dfeval_test.npz`.
  - Each file contains:
    - `embeddings`: float32 array `[N, D]`.
    - `labels`: int64 array `[N]` (**1 = bona fide, 0 = spoof**; for FF train, spoof rows may be absent).
    - `paths`: optional list/array of original audio paths (for traceability).
    - `metadata`: serialized dict such as `{"dataset": "<tag>", "split": "<train|val|test>", "extractor": "<name>", "layer": <int>, "speaker": "Donald_Trump"}`.
  - All splits must share the same feature dimension `D`. Downstream jobs reconstruct combined datasets by loading the relevant `.npz` files instead of scanning raw audio folders.
- **Metrics JSON schema (locked):** every dataset evaluation writes a JSON with the following keys:
  - `run_id` (string, matches directory name or CLI `--run-id`).
  - `extractor` (`"xlsr300m"` or `"speechbrain"`).
  - `model_type` (`"ocsvm"` or `"deepsvdd"`).
  - `dataset` (`"ff"`, `"itw"`, `"dfeval"`, etc.).
  - `phase` (`"baseline"` or `"threshold_extension"`).
  - `threshold` (float; OC-SVM decision threshold or Deep SVDD score threshold used).
  - `num_real` (int, bona fide count).
  - `num_fake` (int, spoof count).
  - `accuracy`, `FAR`, `FRR`, `EER` (floats).
  - `eer_threshold` (float cut point returned by Appendix-A helper).
  - `timestamp_utc` (ISO-8601 string).
  - Optional extras: `nu`, `gamma` for OC-SVM or `nu`, `epochs`, `objective` for Deep SVDD, plus `notes`.
  This schema applies to `results/*.json`, `fusion_*.json`, and threshold-extension outputs so downstream tooling can parse them uniformly.
- **Phase 1 guardrails (do _not_ touch until later phases):**
  - Skip `visualize.py` fusion path entirely; leave `ResidualFusion` unresolved.
  - Ignore `test.py`, `utils_combined.py`, and any legacy inference code.
  - Do not implement threshold extension, fusion, or Deep SVDD.
  - Do not refactor unrelated modules unless strictly required for SSL embedding extraction, OC-SVM training, or metric logging.
- **Phase 1 deliverables:**
  1. **Embedding extraction** driven by protocol configs (Donald Trump only) that writes `.npz` caches following the locked schema.
  2. **OC-SVM hyperparameter sweep** (nu/gamma grid), model training, and checkpoint export for SSL embeddings.
  3. **Metrics emission**: dataset JSONs per evaluation set plus `runs.jsonl` append per run.
  4. **No changes** to fusion, Deep SVDD, visualization, or inference scripts—leave them for later phases.

---

## Appendix A — EER Helper
```python
from sklearn.metrics import roc_curve
import numpy as np

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    return eer, eer_threshold
```
- OC-SVM scores: `svm_model.decision_function(X)`.
- Deep SVDD scores: `R - torch.norm(z - c, dim=1)`.
- Label convention: `1` always denotes bona fide (positive class) and `0` denotes spoof, so ensure any scoring function is aligned before calling `compute_eer`.

---

## Appendix B — Run Directory Template (goal state)
```
/data/Speaker_Specific_Models/
  └── xlsr300m_ocsvm_20260218T1530Z/
        ├── Donald_Trump/
        │     ├── scaling_models/xls_r_300m.pkl
        │     └── svm_models/xls_r_300m.pkl
        ├── results/
        │     ├── ff.json
        │     ├── itw.json
        │     ├── dfeval.json
        │     ├── fusion_avg.json
        │     ├── fusion_weighted.json
        │     └── itw_threshold_extended.json
        ├── hyperparam_trials.jsonl
        ├── outliers_itw.csv
        ├── outliers_dfeval.csv
        └── plots/
              ├── umap_ff_itw_DFEval2024_xls_r_300m_layer_7.html
              └── heatmap_update_20260218.png
```
Mirror this structure for `xlsr300m_deepsvdd_*`, `speechbrain_ocsvm_*`, `speechbrain_deepsvdd_*`, and future fusion/threshold-extension runs.
Keep the master `runs.jsonl` at the repo root so it can aggregate entries from every run directory.
