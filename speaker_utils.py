import torch
import torch.nn as nn
from torch import Tensor

import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# from pyannote.audio.core.task import Specifications, Problem

# torch.serialization.add_safe_globals([
#     torch.torch_version.TorchVersion,
#     Specifications,
#     Problem
# ])

try:
    # from pyannote.audio import Model as PyannoteModel
    from pyannote.audio import Model
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

from pyannote.audio import Inference

########## pyannote pipeline Names ##########
# pyannote/speaker-diarization-3.1
# "pyannote/speaker-diarization-community-1

########## Pyannote Model Names ##########
# pyannote/wespeaker-voxceleb-resnet34-LM

class Speaker_Model(nn.Module):
    """Speaker embedding extractor"""
    def __init__(self, device, model_name="speechbrain/spkrec-ecapa-voxceleb", embedding_type="Speechbrain"):
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        self.embedding_type = embedding_type

        if self.embedding_type == "Speechbrain":
            self.classifier = EncoderClassifier.from_hparams(source=self.model_name, run_opts={"device": torch.device(self.device)})
        
        elif self.embedding_type == "Pyannote":
            self.model = Model.from_pretrained(self.model_name)
            self.inference = Inference(self.model, window="whole", device=torch.device(self.device))
       
    def extract_speaker_embedding(self, waveform, sr=16000):
        """
        Extract embeddings from waveform
        Args:
            waveform: tensor of shape (batch, 1, samples) or (batch, samples)
            sr: sample rate of the audio
        Returns:
            embeddings: tensor of shape (batch, embedding_dim)
        """
        # Squeeze channel dimension if present
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        if self.embedding_type == "Speechbrain":
             embeddings = self.classifier.encode_batch(waveform).squeeze(1)

        elif self.embedding_type == "Pyannote":

            # add channel dimension back for pyannote
            waveform = waveform.unsqueeze(1)

            embeddings = self.inference.infer(waveform)
            # embeddings = []

            
            # print(waveform.shape)
            # with torch.no_grad():
            #     for wav in waveform:
            #         print(wav.shape)
            #         emb = self.inference({"waveform": wav, "sample_rate": sr})
            #         # emb = self.inference({wav})
            #         embeddings.append(emb.squeeze(0).cpu())
        
            # print(embeddings.shape)
        
        # with torch.no_grad():
        #     for wav in waveform:
        #         emb = self.classifier.encode_batch(wav.unsqueeze(0))
        #         print(emb.shape)
        #         embeddings.append(emb.squeeze(0).cpu())
        
       
        # print(embeddings.shape)
        return embeddings
