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