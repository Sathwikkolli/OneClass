# config.py
from dataclasses import dataclass, field

@dataclass
class FeatureConfig:
    speaker_name: str = "Donald_Trump"
    # ssl_model: str = "wavlm_large"
    ssl_model: str = "xls_r_300m"
    n_layers: int = 13
    device: str = "cuda:1"
    # audio_dir: str = "/data/FF_V2/FF_V2/"
    # protocol_path: str = "/data/FF_V2/FF_V2_meta_Data/protocol_Barack_Obama.csv"
    audio_dir: list = field(default_factory=lambda: ["/data/FF_V2/FF_V2/", "/data/Data/ds_wild/release_in_the_wild/"])
    protocol_path: list = field(default_factory=lambda: ["/data/FF_V2/FF_V2_meta_Data/protocol_Donald_Trump.csv", "/data/Data/ds_wild/protocols/meta.csv"])
    protocol_tags: list = field(default_factory=lambda: ["ff", "itw"])
    sep = ','  # or '\t' for tsv files
    output_layer: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # output_layer: int = 3
    out_dir: str = "/data/Speaker_Specific_Models/umap_plots/"
    color_by: str = "attack_type"
    batch_size: int = 32
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    