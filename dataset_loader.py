import pandas as pd
import numpy as np
import torchaudio
import os
import torch
from torch import Tensor
import librosa
import torchaudio.functional as AF
import random
from pathlib import Path

class AudioDataset(torch.utils.data.Dataset):
    # def __init__(self, protocol_paths, audio_dir, speaker_name, sep=','):
    def __init__(self, config):
        protocol_paths = config.protocol_path
        audio_dir = config.audio_dir
        speaker_name = config.speaker_name
        sep = config.sep
        required_cols = config.required_cols
        # Accepts a list or string for protocol files
        if isinstance(protocol_paths, str):
            protocol_paths = [protocol_paths]

        # If audio_dir is a list, require it to have same length as protocol_paths
        if isinstance(audio_dir, (list, tuple)):
            if len(audio_dir) != len(protocol_paths):
                raise ValueError("When passing multiple audio_dir entries, its length must match protocol_paths length")

        # path_reconstruction_modes: one entry per protocol ("auto" or "reconstruct").
        # "reconstruct" ignores the absolute path stored in the CSV and rebuilds it as:
        #   <audio_dir_for_df> / <parent_dirname_from_csv> / <basename_from_csv>
        # This is used when the CSV was created on a different machine and the root
        # prefix must be replaced with the local dataset root.
        path_modes = list(getattr(config, 'path_reconstruction_modes', None) or [])

        dfs = []
        for idx, p in enumerate(protocol_paths):
            if not Path(p).exists():
                import warnings
                warnings.warn(
                    f"Protocol file not found, skipping: {p}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            df = pd.read_csv(p, sep=sep)
            # Strip BOM (\ufeff) and surrounding whitespace from column names.
            # Some CSV files created on Windows or by certain tools embed a BOM
            # at the start of the file, turning "Audio" into "\ufeffAudio".
            df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]

            # Normalize common lowercase/aliased column names to canonical names.
            # Handles protocols like the itw meta.csv that use "file"/"speaker"/"label"
            # instead of the canonical "Audio"/"Speaker"/"Label".
            _COL_ALIASES = {"file": "Audio", "filename": "Audio", "label": "Label", "speaker": "Speaker"}
            df = df.rename(columns={k: v for k, v in _COL_ALIASES.items() if k in df.columns and v not in df.columns})

            path_mode = path_modes[idx] if idx < len(path_modes) else "auto"

            # ----------------------------------------------------------------
            # Reconstruct-mode column normalization.
            # The OC eval CSV uses "audiofilepath" instead of "Audio", has no
            # "Label" column (real vs. fake is encoded in the path's parent
            # directory), and no "Source" column.  Normalize all three before
            # the required_cols check so the rest of the pipeline is unchanged.
            # ----------------------------------------------------------------
            if path_mode == "reconstruct":
                # 1. Normalize path column names to canonical "Audio".
                if "Audio" not in df.columns:
                    for candidate in ("audiofilepath", "audio_file", "path", "filepath", "file", "filename"):
                        if candidate in df.columns:
                            df = df.rename(columns={candidate: "Audio"})
                            break

                # If there is still no path column, this protocol cannot be reconstructed.
                if "Audio" not in df.columns:
                    import warnings
                    warnings.warn(
                        f"Protocol at '{p}' is missing an Audio path column required for reconstruct mode; skipping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                # 2. Infer Label: parent dir "Original" (and aliases) → bonafide,
                #    anything else (COZYVOICE2, E2TTS, …) → spoof.
                if "Label" not in df.columns:
                    _REAL_ALIASES = frozenset({"original", "real", "bonafide", "genuine"})
                    df["Label"] = df["Audio"].apply(
                        lambda a: "bonafide"
                        if Path(str(a)).parent.name.lower() in _REAL_ALIASES
                        else "spoof"
                    )

                # 3. Infer Source from the system subdirectory name, prefixed with
                #    the protocol tag so that infer_dataset_tag() in run_baselines.py
                #    can map every row back to the correct dataset tag (e.g. "oc").
                if "Source" not in df.columns:
                    _REAL_ALIASES = frozenset({"original", "real", "bonafide", "genuine"})
                    _tag = config.protocol_tags[idx]
                    df["Source"] = df["Audio"].apply(
                        lambda a, _t=_tag: f"{_t}_bonafide"
                        if Path(str(a)).parent.name.lower() in _REAL_ALIASES
                        else f"{_t}_{Path(str(a)).parent.name}"
                    )

            for col in required_cols:
                if col not in df.columns:
                    if col == "Source":
                        df[col] = config.protocol_tags[idx] + '_' + df['Label'].astype(str)
                    else:
                        df[col] = None

            df = df[required_cols]

            # Use the corresponding audio_dir for each protocol file when audio_dir is a list.
            audio_dir_for_df = audio_dir[idx] if isinstance(audio_dir, (list, tuple)) else audio_dir

            # Directory names in old CSV paths that correspond to real (bonafide) audio.
            # On Great Lakes these live under the "-" subdirectory.
            _REAL_DIR_ALIASES = frozenset({"original", "real", "bonafide", "genuine"})
            _REALS_DIR = "-"

            def _join_audio_path(a, _mode=path_mode, _root=audio_dir_for_df):
                a_str = str(a)
                if _mode == "reconstruct":
                    # Take only <system_subdir>/<filename> from the stored path and
                    # graft onto the local dataset root.  Map "Original" (and aliases)
                    # to the "-" directory used on Great Lakes for real audio.
                    parts = Path(a_str).parts
                    filename = parts[-1]
                    parent = parts[-2] if len(parts) >= 2 else ""
                    subdir = _REALS_DIR if parent.lower() in _REAL_DIR_ALIASES else parent
                    return os.path.join(_root, subdir, filename)
                # Default "auto" mode: keep absolute paths as-is when they exist.
                # If an absolute path points to a stale mount prefix (common across
                # cluster migrations), fall back to <audio_root>/<basename> when that
                # candidate exists locally.
                if os.path.isabs(a_str):
                    if os.path.exists(a_str):
                        return a_str
                    fallback = os.path.join(_root, os.path.basename(a_str))
                    if os.path.exists(fallback):
                        return fallback
                    return a_str
                return os.path.join(_root, a_str)

            df['Audio'] = df['Audio'].astype(str).apply(_join_audio_path)
            dfs.append(df)
        if not dfs:
            raise RuntimeError(
                "No protocol files could be loaded. Check that at least one CSV path "
                "in FeatureConfig.protocol_path exists."
            )
        meta = pd.concat(dfs, ignore_index=True)
        
        # Normalize speaker names: replace spaces and underscores, lowercase
        meta['Speaker_norm'] = meta['Speaker'].str.replace(' ', '_').str.replace('-', '_')
        speaker_name_norm = speaker_name.replace(' ', '_').replace('-', '_')
        self.meta = meta[meta['Speaker_norm'] == speaker_name_norm].reset_index(drop=True)

        # print(self.meta)
        
        self.audio_dir = audio_dir
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        # file_path = os.path.join(self.audio_dir, row['filename'])
        file_path = row["Audio"]
        # waveform, sr = torchaudio.load(file_path)
        X, fs = librosa.load(file_path, sr=16000, dtype=np.float32)
        
        X_pad= self.pad(X,self.cut)
        x_inp= torch.from_numpy(X_pad)

        label = row.get('Label', None)
        attack_type = row.get('Source', None)
        
        return x_inp, label, file_path, attack_type
    
    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x	


class BonaFideOnly(torch.utils.data.Dataset):
    def __init__(self, base_dataset, bona_fide_label=1, split_value="train", split_col="split"):
        self.base = base_dataset

        # --- safety checks ---
        if split_col not in self.base.meta.columns:
            raise ValueError(
                f"BonaFideOnly: split_col='{split_col}' not found in protocol. "
                f"Available columns: {list(self.base.meta.columns)}"
            )
        
        # Build mask infer indices that are bona fide and in the specified split`
        labels = self.base.meta["Label"].astype(str)
        splits = self.base.meta[split_col].astype(str)

        mask = (labels == str(bona_fide_label)) & (splits == str(split_value))

        self.idxs = self.base.meta.index[mask].to_list()

        if len(self.idxs) == 0:
            raise ValueError(
                "BonaFideOnly: No samples matched "
                f"Label={bona_fide_label} and {split_col}={split_value}. "
                "Check your protocol values."
            )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        x, label, path, attack = self.base[self.idxs[i]]
        return x  # only waveform needed
    

class TwoViewWrapper(torch.utils.data.Dataset):
    def __init__(self, base_wave_dataset, augment_fn):
        self.base = base_wave_dataset
        self.aug = augment_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x = self.base[idx]              # Tensor [T]
        x1 = self.aug(x)
        x2 = self.aug(x)
        return x1, x2
    

def augment_wave(x, sr=16000):
    # x: Tensor [T]
    x = x.clone()

    # random gain
    if random.random() < 0.8:
        gain_db = random.uniform(-6, 6)
        x = x * (10 ** (gain_db / 20))

    # additive noise
    if random.random() < 0.8:
        snr_db = random.uniform(5, 30)
        noise = torch.randn_like(x)
        x_power = x.pow(2).mean().clamp_min(1e-8)
        n_power = noise.pow(2).mean().clamp_min(1e-8)
        scale = torch.sqrt(x_power / (10 ** (snr_db / 10) * n_power))
        x = x + scale * noise

    # random band-pass-ish via biquad (optional)
    if random.random() < 0.5:
        # pick lowpass or highpass randomly
        if random.random() < 0.5:
            cutoff = random.uniform(2000, 7000)
            x = AF.lowpass_biquad(x, sr, cutoff)
        else:
            cutoff = random.uniform(50, 300)
            x = AF.highpass_biquad(x, sr, cutoff)

    # clamp
    x = torch.clamp(x, -1.0, 1.0)
    return x

# =========================
# Augmentations (bona fide only)
# =========================
def augment_wave(
    x: torch.Tensor,
    sr: int = 16000,
    gain_db_min: float = -6.0,
    gain_db_max: float = 6.0,
    snr_db_min: float = 5.0,
    snr_db_max: float = 30.0,
    time_mask_prob: float = 0.3,
    time_mask_min_frac: float = 0.02,
    time_mask_max_frac: float = 0.10,
) -> torch.Tensor:
    """
    x: Tensor [T]
    Torch-only augmentations (safe in DataLoader workers).
    """
    x = x.clone().float()

    # random gain
    if random.random() < 0.8:
        gain_db = random.uniform(gain_db_min, gain_db_max)
        x = x * (10 ** (gain_db / 20.0))

    # additive noise at random SNR
    if random.random() < 0.8:
        snr_db = random.uniform(snr_db_min, snr_db_max)
        noise = torch.randn_like(x)
        x_power = x.pow(2).mean().clamp_min(1e-8)
        n_power = noise.pow(2).mean().clamp_min(1e-8)
        scale = torch.sqrt(x_power / (10 ** (snr_db / 10.0) * n_power))
        x = x + scale * noise

    # random band-pass-ish via biquad (optional)
    if random.random() < 0.5:
        # pick lowpass or highpass randomly
        if random.random() < 0.5:
            cutoff = random.uniform(2000, 7000)
            x = AF.lowpass_biquad(x, sr, cutoff)
        else:
            cutoff = random.uniform(50, 300)
            x = AF.highpass_biquad(x, sr, cutoff)

    # time masking
    if random.random() < time_mask_prob:
        T = x.shape[0]
        width = int(random.uniform(time_mask_min_frac, time_mask_max_frac) * T)
        if width > 0:
            start = random.randint(0, max(0, T - width))
            x[start:start + width] = 0.0

    # clamp
    x = torch.clamp(x, -1.0, 1.0)
    return x