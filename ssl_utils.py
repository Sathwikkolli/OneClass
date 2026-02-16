from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from s3prl.nn import S3PRLUpstream, Featurizer

class SSLModel(nn.Module):
    def __init__(self, n_layers, device, args):
        super(SSLModel, self).__init__()
        self.device = device
        self.model_name = args.ssl_model
        self.model = S3PRLUpstream(self.model_name).to(self.device)
        self.featurizer = Featurizer(self.model).to(self.device)
        self.n_layers=n_layers
        self.out_dim = self.featurizer.output_size

    def extract_feat_featurizer(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])
        with torch.no_grad():
            all_hs, all_hs_len = self.model(waveform.to(self.device), wavs_len.to(self.device))
        hs, hs_len = self.featurizer(all_hs, all_hs_len)
        return hs, hs_len
    
    def extract_feat(self, waveform):
        waveform = waveform.squeeze(1)
        wavs_len = torch.LongTensor([waveform.size(1) for _ in range(waveform.size(0))])
        with torch.no_grad():
            all_hs, all_hs_len = self.model(waveform.to(self.device), wavs_len.to(self.device))
        # return torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in all_hs[:self.n_layers]], dim=1)
        return all_hs
    
    
# -----------------------------
# SSL layer embedding extraction (robust)
# -----------------------------
def get_ssl_layer_emb(all_hs, layer_idx: int, batch_size: int) -> torch.Tensor:
    """
    Returns pooled SSL embedding: [B, D] from a chosen layer.

    all_hs: list-like of layer outputs from S3PRLUpstream
    layer outputs can be Tensor or tuple (Tensor, lengths)
    Common shapes:
      - [B, T, D]
      - [T, B, D]
    """
    h = all_hs[layer_idx]
    if isinstance(h, tuple):
        h = h[0]

    if not torch.is_tensor(h):
        raise TypeError(f"Layer output is not a tensor/tuple tensor. Got type={type(h)}")

    if h.dim() != 3:
        raise ValueError(f"Expected 3D hidden state, got shape={tuple(h.shape)} at layer {layer_idx}")

    # Make it [B, T, D]
    if h.shape[0] == batch_size:
        # already [B, T, D]
        pass
    elif h.shape[1] == batch_size:
        # [T, B, D] -> [B, T, D]
        h = h.transpose(0, 1)
    else:
        # fallback: try to guess; but better to print and fix
        raise ValueError(
            f"Cannot infer batch dimension for h shape={tuple(h.shape)}, batch_size={batch_size}. "
            "Print shapes from your upstream and adjust."
        )

    # mean pool over time
    e = h.mean(dim=1)  # [B, D]
    return e
