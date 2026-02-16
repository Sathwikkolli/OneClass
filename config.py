# config.py
from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class FeatureConfig:
    extract_ssl: bool = False
    speaker_name: str = "Donald_Trump"
    # ssl_model: str = "wavlm_large"
    ssl_model: str = "xls_r_300m"
    # ssl_model: str = "unispeech_sat_large"
    n_layers: int = 13
    device: str = "cuda:1"
    # audio_dir: str = "/data/FF_V2/FF_V2/"
    # protocol_path: str = "/data/FF_V2/FF_V2_meta_Data/protocol_Barack_Obama.csv"
    # audio_dir: list = field(default_factory=lambda: ["/data/FF_V2/FF_V2/", "/data/Data/ds_wild/release_in_the_wild/"])
    # protocol_path: list = field(default_factory=lambda: ["/data/FF_V2/FF_V2_meta_Data/protocol_Kamala_Harris.csv", "/data/Data/ds_wild/protocols/meta.csv"])
    # protocol_tags: list = field(default_factory=lambda: ["ff", "itw"])
    audio_dir: list = field(default_factory=lambda: ["/data/FF_V2/FF_V2/", "/data/Data/ds_wild/release_in_the_wild/", "/data/Data/Deepfake_Eval_2024/audio-data"])
    protocol_path: list = field(default_factory=lambda: ["/data/FF_V2/FF_V2_meta_Data/protocol_Donald_Trump_v1.csv", "/data/Data/ds_wild/protocols/meta.csv", "/data/Data/Deepfake_Eval_2024/protocols/final_Deepfakeeval2024_Speakerverification.csv"])
    protocol_tags: list = field(default_factory=lambda: ["ff", "itw", "DFEval2024"])
    sep = ','  # or '\t' for tsv files
    required_cols: List[str] = field(default_factory=lambda: ["Audio", "Label", "Speaker", "Source"])
    
    # output_layer: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # output_layer: list = field(default_factory=lambda: [7, 8, 9, 10, 11, 12])
    output_layer: int = 7
    out_dir: str = "/data/Speaker_Specific_Models/umap_plots/"
    color_by: str = "attack_type"
    batch_size: int = 32
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    # Pyannote settings
    extract_speaker_embed: bool = False
    speaker_embedding_type: str = "Speechbrain"
    # speaker_embedding_type: str = "Pyannote"
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    
    extract_fusion_features: bool = True
    fusion_ckpt_path: str = "/data/Speaker_Specific_Models/fusion_ckpts/trump_residual_fusion_a2p0_mag0p05/fusion_best.pt"
    run_name: str = "fusion_a2p0_mag0p05"
    fused_feature_type: str = "residual"
    
########## pyannote pipelines Names ##########
# pyannote/speaker-diarization-3.1
# "pyannote/speaker-diarization-community-1

########## Speakaer Model Names ##########
# pyannote/wespeaker-voxceleb-resnet34-LM
# speechbrain/spkrec-ecapa-voxceleb

@dataclass
class FusionTrainConfig:
    # --- training data selection ---
    speaker_name: str = "Donald_Trump"
    train_audio_dir: Union[str, List[str]] = "/data/FF_V2/FF_V2/"
    train_protocol_path: Union[str, List[str]] = "/data/FF_V2/FF_V2_meta_Data_OC/protocol_Donald_Trump_v1.csv"
   
    train_protocol_tag: str = "ff"      # only train on FF
    bona_fide_value: str = "bonafide"   # or 1/0 depending on your protocol
    sep: str = ","                   # keep here so training doesn't depend on FeatureConfig
    required_cols: List[str] = field(default_factory=lambda: ["Audio", "Label", "Speaker", "Source", "split"])
    
    sampling_rare: int = 16000

    # ---- Models ----
    # ssl_model: str = "unispeech_sat_large"
    ssl_model: str = "xls_r_300m"
    n_layers: int = 10
    ssl_layer: int = 7

    speaker_embedding_type: str = "Speechbrain"
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"

    device: str = "cuda:1"

    # ---- Residual fusion module ----
    alpha: float = 2.0
    hidden: int = 512

    # ---- Optimization ----
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    seed: int = 1234

    # ---- Loss knobs ----
    use_mag_loss: bool = True
    tau: float = 1.0
    mag_weight: float = 0.05

    # --- augmentation knobs ---
    snr_db_min: float = 5.0
    snr_db_max: float = 30.0
    gain_db_min: float = -6.0
    gain_db_max: float = 6.0
    time_mask_prob: float = 0.3
    time_mask_min_frac: float = 0.02
    time_mask_max_frac: float = 0.10

     # --- outputs ---
    save_dir: str = "/data/Speaker_Specific_Models/fusion_ckpts/"
    run_name: str = "trump_residual_fusion_a2p0_mag0p05"