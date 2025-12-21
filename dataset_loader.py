import pandas as pd
import numpy as np
import torchaudio
import os
import torch
from torch import Tensor
import librosa

class AudioDataset(torch.utils.data.Dataset):
    # def __init__(self, protocol_paths, audio_dir, speaker_name, sep=','):
    def __init__(self, config):
        protocol_paths = config.protocol_path
        audio_dir = config.audio_dir
        speaker_name = config.speaker_name
        sep = config.sep
        # Accepts a list or string for protocol files
        if isinstance(protocol_paths, str):
            protocol_paths = [protocol_paths]
        
        required_cols = ["Audio", "Label", "Speaker", "Source"]
        
        # If audio_dir is a list, require it to have same length as protocol_paths
        if isinstance(audio_dir, (list, tuple)):
            if len(audio_dir) != len(protocol_paths):
                raise ValueError("When passing multiple audio_dir entries, its length must match protocol_paths length")
        
        dfs = []
        for idx, p in enumerate(protocol_paths):
            df = pd.read_csv(p, sep=sep)
            for col in required_cols:
                if col not in df.columns:
                    if col == "Source":
                        df[col] = config.protocol_tags[idx] + '_' + df['Label'].astype(str)
                    else:
                        df[col] = None
            
            df = df[required_cols]

            # Use the corresponding audio_dir for each protocol file when audio_dir is a list.
            audio_dir_for_df = audio_dir[idx] if isinstance(audio_dir, (list, tuple)) else audio_dir

            # Prepend audio_dir to Audio entries that are not absolute paths
            def _join_audio_path(a):
                a_str = str(a)
                if os.path.isabs(a_str):
                    return a_str
                return os.path.join(audio_dir_for_df, a_str)

            df['Audio'] = df['Audio'].astype(str).apply(_join_audio_path)
            dfs.append(df)
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
        X, fs = librosa.load(file_path, sr=16000)
        
        X_pad= self.pad(X,self.cut)
        x_inp= Tensor(X_pad)

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