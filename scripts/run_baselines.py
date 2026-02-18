#!/usr/bin/env python3
"""Phase-1 baseline orchestrator for Donald Trump speaker-specific ASD."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

try:  # Optional dependency for YAML configs
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

from config import FeatureConfig
from dataset_loader import AudioDataset
from e2e_utils import AudioSVMClassifier

SUPPORTED_EXTRACTORS = ("xlsr300m", "speechbrain")
SUPPORTED_MODELS = ("ocsvm", "deepsvdd")
PHASE1_DEFAULT_SPLITS = ("ff_train", "ff_val", "itw_test", "dfeval_test")
RUNS_ROOT = Path("runs")
PHASE1_STUB_COMMANDS = ("deepsvdd", "evaluate", "fusion")


@dataclass
class RuntimeState:
    """Holds runtime settings shared across subcommands."""

    extractor: str
    model: str
    device: str
    feature_config: FeatureConfig
    output_root: Path
    output_run_dir: Path
    cache_dir: Path
    run_id: str
    runs_dir: Path
    seed: int
    log_level: str
    raw_config: Dict[str, Any]


@dataclass(frozen=True)
class SplitSpec:
    dataset_tags: List[str]
    split_values: Optional[List[str]] = None
    split_column: str = "split"


DEFAULT_SPLIT_SPECS: Dict[str, SplitSpec] = {
    "ff_train": SplitSpec(dataset_tags=["ff"], split_values=["train"]),
    "ff_val": SplitSpec(dataset_tags=["ff"], split_values=["val", "dev", "validation"]),
    "itw_test": SplitSpec(dataset_tags=["itw"]),
    "dfeval_test": SplitSpec(dataset_tags=["DFEval2024"]),
}

POSITIVE_LABEL_VALUES = {
    "1",
    "bonafide",
    "bona_fide",
    "bona fide",
    "genuine",
    "human",
    "real",
    "true",
}
NEGATIVE_LABEL_VALUES = {"0", "spoof", "fake", "attack", "false"}
RUNS_LOG_PATH = REPO_ROOT / "runs.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline automation entrypoint (Phase-1). "
            "Implements SSL embedding extraction + OC-SVM scaffolding"
        ),
    )
    parser.add_argument("--config", type=str, help="Optional YAML/JSON config file path.")
    parser.add_argument(
        "--extractor",
        choices=SUPPORTED_EXTRACTORS,
        help="Feature extractor to run (required when --config is omitted).",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="ocsvm",
        help="Model family to orchestrate (ocsvm or deepsvdd).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--output-root",
        default="/data/Speaker_Specific_Models",
        help="Directory root for exported artifacts (default: /data/Speaker_Specific_Models).",
    )
    parser.add_argument(
        "--cache-dir",
        default="data_cache",
        help="Embedding cache directory (default: ./data_cache).",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Optional run identifier. When omitted the orchestrator auto-generates "
            "<extractor>_<model>_<UTCtimestamp>."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Root logger level (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    extract_parser = subparsers.add_parser(
        "extract", help="Build cached embeddings (.npz) for the Donald Trump protocols."
    )
    extract_parser.add_argument(
        "--splits",
        nargs="+",
        default=list(PHASE1_DEFAULT_SPLITS),
        help="Dataset splits to cache (default: ff_train ff_val itw_test dfeval_test).",
    )
    extract_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files instead of skipping.",
    )
    extract_parser.set_defaults(func=handle_extract)

    ocsvm_parser = subparsers.add_parser(
        "ocsvm", help="Run OC-SVM hyperparameter sweeps + evaluations (Phase-1 scope)."
    )
    ocsvm_parser.add_argument(
        "--nu-grid",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2],
        help="List of nu values to sweep (default: 0.05 0.1 0.2).",
    )
    ocsvm_parser.add_argument(
        "--gamma-grid",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2],
        help="List of gamma values to sweep (default: 0.05 0.1 0.2).",
    )
    ocsvm_parser.add_argument(
        "--train-split",
        default="ff_train",
        help="Training split name inside the embedding cache (default: ff_train).",
    )
    ocsvm_parser.add_argument(
        "--val-splits",
        nargs="+",
        default=["ff_val"],
        help="Validation split names used to score each trial (default: ff_val).",
    )
    ocsvm_parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["ff_val", "itw_test", "dfeval_test"],
        help="Splits evaluated after selecting the best config.",
    )
    ocsvm_parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel workers for sweeps (default: 1 / serial).",
    )
    ocsvm_parser.set_defaults(func=handle_ocsvm)

    for cmd in PHASE1_STUB_COMMANDS:
        stub_parser = subparsers.add_parser(
            cmd,
            help=(
                f"Placeholder subcommand. {cmd} is deferred until after Phase-1, "
                "see AGENTS.md Section 11."
            ),
        )
        stub_parser.set_defaults(func=handle_phase1_stub, stub_name=cmd)

    return parser


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_run_id(extractor: str, model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{extractor}_{model}_{stamp}"


def load_structured_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    text = cfg_path.read_text().strip()
    if not text:
        return {}
    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configs but is not installed.")
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config format: {cfg_path.suffix}")


def apply_feature_overrides(feature_config: FeatureConfig, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(feature_config, key):
            setattr(feature_config, key, value)
        else:
            logging.warning("Ignoring unknown FeatureConfig override: %s", key)


def build_runtime_state(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    raw_cfg: Dict[str, Any],
) -> RuntimeState:
    extractor = args.extractor or raw_cfg.get("extractor")
    if extractor is None:
        parser.error("--extractor is required when --config does not specify it.")
    if extractor not in SUPPORTED_EXTRACTORS:
        parser.error(f"Unsupported extractor '{extractor}'. Valid options: {SUPPORTED_EXTRACTORS}")

    model = args.model or raw_cfg.get("model") or "ocsvm"
    if model not in SUPPORTED_MODELS:
        parser.error(f"Unsupported model '{model}'. Valid options: {SUPPORTED_MODELS}")

    feature_cfg = FeatureConfig()
    feature_cfg.extract_fusion_features = False
    feature_overrides = raw_cfg.get("feature_config") or {}
    apply_feature_overrides(feature_cfg, feature_overrides)

    if extractor == "xlsr300m":
        feature_cfg.extract_ssl = True
        feature_cfg.extract_speaker_embed = False
        feature_cfg.ssl_model = "xls_r_300m"
    elif extractor == "speechbrain":
        feature_cfg.extract_ssl = False
        feature_cfg.extract_speaker_embed = True

    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    cache_dir = Path(args.cache_dir)
    run_id = args.run_id or raw_cfg.get("run_id") or generate_run_id(extractor, model)
    runs_dir = RUNS_ROOT / run_id
    output_run_dir = output_root / run_id

    return RuntimeState(
        extractor=extractor,
        model=model,
        device=device,
        feature_config=feature_cfg,
        output_root=output_root,
        output_run_dir=output_run_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        runs_dir=runs_dir,
        seed=args.seed,
        log_level=args.log_level,
        raw_config=raw_cfg,
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def persist_plan(path: Path, payload: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def append_runs_log(entry: Dict[str, Any]) -> None:
    ensure_directory(RUNS_LOG_PATH.parent)
    with RUNS_LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")


def normalize_label_value(raw_value: Any) -> int:
    if raw_value is None:
        return 0
    if isinstance(raw_value, (int, float)):
        return 1 if int(raw_value) > 0 else 0
    value = str(raw_value).strip().lower()
    if value in POSITIVE_LABEL_VALUES:
        return 1
    if value in NEGATIVE_LABEL_VALUES:
        return 0
    try:
        numeric = float(value)
        return 1 if numeric > 0 else 0
    except ValueError:
        pass
    if value in {"", "nan"}:
        return 0
    logging.warning("Unrecognized label '%s'; defaulting to spoof (0).", raw_value)
    return 0


def infer_dataset_tag(source_value: Any, audio_path: str, protocol_tags: List[str]) -> str:
    source_str = str(source_value or "").strip()
    lower_source = source_str.lower()
    for tag in protocol_tags:
        if lower_source.startswith(tag.lower()):
            return tag
    audio_lower = audio_path.lower()
    for tag in protocol_tags:
        if f"/{tag.lower()}" in audio_lower or f"\\{tag.lower()}" in audio_lower:
            return tag
    if "_" in source_str:
        return source_str.split("_", 1)[0]
    return source_str or "unknown"


def build_protocol_metadata(feature_config: FeatureConfig) -> pd.DataFrame:
    dataset = AudioDataset(feature_config)
    meta = dataset.meta.copy()
    if meta.empty:
        raise RuntimeError("Dataset loader returned zero rows for the configured protocols.")

    if "Audio" not in meta.columns:
        raise KeyError("Protocol metadata is missing the 'Audio' column.")

    if "split" not in meta.columns:
        meta["split"] = "unspecified"
    meta["split"] = meta["split"].fillna("unspecified")
    meta["split_norm"] = meta["split"].astype(str).str.strip().str.lower()

    if "Label" not in meta.columns:
        raise KeyError("Protocol metadata is missing the 'Label' column.")
    meta["label_norm"] = meta["Label"].astype(str).str.strip().str.lower()

    if "Source" not in meta.columns:
        meta["Source"] = ""

    protocol_tags = list(feature_config.protocol_tags or [])
    meta["dataset_tag"] = meta.apply(
        lambda row: infer_dataset_tag(row["Source"], str(row["Audio"]), protocol_tags),
        axis=1,
    )
    return meta


def filter_split_rows(meta: pd.DataFrame, split_name: str, spec: SplitSpec) -> pd.DataFrame:
    subset = meta[meta["dataset_tag"].isin(spec.dataset_tags)].copy()
    if spec.split_values:
        valid = {value.lower() for value in spec.split_values}
        norm_column = f"{spec.split_column}_norm"
        if norm_column not in subset.columns:
            raise KeyError(f"Metadata is missing the normalized column '{norm_column}' required for split filtering.")
        subset = subset[subset[norm_column].isin(valid)].copy()
    return subset


def instantiate_ssl_extractor(state: RuntimeState) -> AudioSVMClassifier:
    if state.extractor != "xlsr300m":
        raise NotImplementedError(
            "Phase-1 extract currently supports only the SSL XLS-R 300M path. "
            f"Requested extractor='{state.extractor}'."
        )
    device = torch.device(state.device)
    return AudioSVMClassifier(
        model_name=state.feature_config.ssl_model,
        device=device,
        speaker=state.feature_config.speaker_name,
    )


def inspect_existing_cache(cache_path: Path, split_name: str) -> Dict[str, Any]:
    with np.load(cache_path, allow_pickle=True) as npz:
        embeddings_shape = npz["embeddings"].shape
        labels = npz["labels"]
        metadata = {}
        if "metadata" in npz.files:
            raw_meta = npz["metadata"]
            try:
                metadata_str = raw_meta.item() if hasattr(raw_meta, "item") else str(raw_meta)
                metadata = json.loads(metadata_str)
            except Exception:  # pragma: no cover - best effort
                metadata = {"raw": str(raw_meta)}
        label_counts = Counter(int(v) for v in labels.tolist())
    summary = {
        "split": split_name,
        "cache_path": str(cache_path.resolve()),
        "num_examples": int(embeddings_shape[0]),
        "embedding_dim": int(embeddings_shape[1]) if len(embeddings_shape) > 1 else None,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "status": "existing",
    }
    if metadata:
        summary["metadata"] = metadata
    return summary


def extract_split_cache(
    split_name: str,
    subset: pd.DataFrame,
    cache_path: Path,
    extractor: AudioSVMClassifier,
    layer_idx: int,
    ssl_model_name: str,
    overwrite: bool,
) -> Dict[str, Any]:
    if subset.empty:
        raise RuntimeError(f"No rows matched the '{split_name}' split specification.")

    if cache_path.exists() and not overwrite:
        logging.info("Cache %s already exists; skipping (use --overwrite to rebuild).", cache_path)
        summary = inspect_existing_cache(cache_path, split_name)
        summary["status"] = "skipped_existing"
        return summary

    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []
    errors: List[str] = []
    start = time.perf_counter()
    for idx, row in subset.iterrows():
        path = Path(str(row["Audio"])).expanduser()
        try:
            emb = extractor.extract_features(
                str(path),
                aggregate_emb=False,
                layer_number=layer_idx,
            )
            emb_np = np.asarray(emb)
            if emb_np.ndim > 1:
                emb_np = np.squeeze(emb_np, axis=0)
            embeddings.append(emb_np.astype(np.float32))
        except Exception as exc:  # pragma: no cover - runtime safety
            message = f"{split_name}: failed to extract {path}: {exc}"
            logging.error(message)
            errors.append(message)
            continue
        labels.append(normalize_label_value(row["Label"]))
        paths.append(str(path))

    if not embeddings:
        raise RuntimeError(f"Failed to create embeddings for split '{split_name}'. See logs for details.")

    embedding_matrix = np.stack(embeddings).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    paths_arr = np.asarray(paths, dtype=object)

    metadata = {
        "split": split_name,
        "dataset_tags": sorted(subset["dataset_tag"].unique().tolist()),
        "speaker": subset["Speaker"].iloc[0] if "Speaker" in subset.columns else "",
        "extractor": "xlsr300m",
        "ssl_model": ssl_model_name,
        "layer": layer_idx,
        "num_embeddings": int(embedding_matrix.shape[0]),
        "embedding_dim": int(embedding_matrix.shape[1]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }

    ensure_directory(cache_path.parent)
    np.savez(
        cache_path,
        embeddings=embedding_matrix,
        labels=labels_arr,
        paths=paths_arr,
        metadata=json.dumps(metadata),
    )
    duration = time.perf_counter() - start
    label_counts = Counter(labels_arr.tolist())
    summary = {
        "split": split_name,
        "cache_path": str(cache_path.resolve()),
        "num_examples": int(embedding_matrix.shape[0]),
        "embedding_dim": int(embedding_matrix.shape[1]),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "status": "written",
        "duration_sec": round(duration, 2),
    }
    if errors:
        summary["errors"] = errors
    return summary


def handle_extract(args: argparse.Namespace, state: RuntimeState) -> int:
    cache_root = state.cache_dir / state.extractor / state.feature_config.speaker_name
    ensure_directory(cache_root)
    ensure_directory(state.runs_dir)
    ensure_directory(state.output_run_dir)

    logging.info("Loading protocol metadata for speaker '%s'.", state.feature_config.speaker_name)
    feature_cfg = state.feature_config
    feature_cfg.device = state.device
    feature_cfg.extract_fusion_features = False
    meta = build_protocol_metadata(feature_cfg)
    logging.info("Loaded %d protocol rows spanning datasets: %s", len(meta), sorted(meta["dataset_tag"].unique()))

    extractor = instantiate_ssl_extractor(state)
    split_summaries: List[Dict[str, Any]] = []
    for split_name in args.splits:
        spec = DEFAULT_SPLIT_SPECS.get(split_name)
        if spec is None:
            raise ValueError(f"Unknown split '{split_name}'. Supported splits: {sorted(DEFAULT_SPLIT_SPECS)}")
        subset = filter_split_rows(meta, split_name, spec)
        cache_path = cache_root / f"{split_name}.npz"
        logging.info(
            "Processing split '%s' (%d rows) -> %s",
            split_name,
            len(subset),
            cache_path,
        )
        summary = extract_split_cache(
            split_name=split_name,
            subset=subset,
            cache_path=cache_path,
            extractor=extractor,
            layer_idx=feature_cfg.output_layer,
            ssl_model_name=feature_cfg.ssl_model,
            overwrite=bool(args.overwrite),
        )
        split_summaries.append(summary)
        logging.info(
            "Split '%s': wrote %s (samples=%s, labels=%s, status=%s).",
            split_name,
            summary.get("cache_path"),
            summary.get("num_examples"),
            summary.get("label_counts"),
            summary.get("status"),
        )

    summary_payload = {
        "run_id": state.run_id,
        "command": "extract",
        "extractor": state.extractor,
        "model": state.model,
        "output_layer": feature_cfg.output_layer,
        "cache_root": str(cache_root.resolve()),
        "splits": split_summaries,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = state.runs_dir / "extract_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    logging.info("Wrote extract summary to %s", summary_path)

    append_runs_log(summary_payload)
    return 0


def handle_ocsvm(args: argparse.Namespace, state: RuntimeState) -> int:
    ensure_directory(state.runs_dir)
    ensure_directory(state.output_run_dir)

    sweep_plan = {
        "command": "ocsvm",
        "run_id": state.run_id,
        "extractor": state.extractor,
        "model": state.model,
        "train_split": args.train_split,
        "val_splits": args.val_splits,
        "eval_splits": args.eval_splits,
        "nu_grid": args.nu_grid,
        "gamma_grid": args.gamma_grid,
        "max_workers": args.max_workers,
        "cache_dir": str(state.cache_dir),
        "output_root": str(state.output_run_dir),
        "notes": (
            "Phase-1 placeholder. Implement OC-SVM training, hyperparam logging, and "
            "metrics emission per AGENTS.md."
        ),
    }
    plan_path = state.runs_dir / "ocsvm_plan.json"
    persist_plan(plan_path, sweep_plan)
    logging.info("Recorded OC-SVM sweep placeholder at %s.", plan_path)
    logging.info(
        "TODO: load cached embeddings, run sweeps over nu/gamma, and emit metrics + runs.jsonl entries."
    )
    return 0


def handle_phase1_stub(args: argparse.Namespace, state: RuntimeState) -> int:
    message = (
        f"The '{args.stub_name}' subcommand is intentionally disabled during Phase-1. "
        "Refer to AGENTS.md Section 11 before enabling it."
    )
    logging.error(message)
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_cfg = load_structured_config(args.config)
    state = build_runtime_state(args, parser, raw_cfg)
    logging.basicConfig(
        level=getattr(logging, state.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.debug("Runtime state: %s", state)
    result = args.func(args, state)
    return result


if __name__ == "__main__":
    sys.exit(main())
