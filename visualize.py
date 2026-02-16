import torch
import numpy as np
import pandas as pd
import umap
import os
from tqdm import tqdm
from plotly import express as px
from torch.utils.data import DataLoader

from ssl_utils import SSLModel
from speaker_utils import Speaker_Model
from config import FeatureConfig
from dataset_loader import AudioDataset
from train_residual import ResidualFusion
from ssl_utils import get_ssl_layer_emb


# Distinct color palette (colorblind-friendly, expanded for many categories)
available_colors = [
    "#0072B2",  # Blue (for bonafide)
    "#E69F00",  # Orange
    "#CC79A7",  # Magenta/Pink
    "#D55E00",  # Red/Orange
    "#009E73",  # Green
    "#F0E442",  # Yellow
    "#56B4E9",  # Light Blue
    "#999999",  # Gray
    "#E1BE6A",  # Tan
    "#1B9E77",  # Teal
    "#7570B3",  # Purple
    "#A6761D",  # Brown
    "#F8766D",  # Red
    "#00BA38",  # Bright Green
    "#619CFF",  # Periwinkle
    "#FF61C3",  # Hot Pink
    "#00BFC4",  # Cyan
    "#7CAE00",  # Lime Green
    "#C77CFF",  # Lavender
    "#00BCD4",  # Light Cyan
]


def extract_SSL_features(model, dataloader, output_layer):
    feats, labels, files, attack_type = [], [], [], []
    for waveform, label, path, attack in tqdm(dataloader, desc="Extracting SSL features"):
        with torch.no_grad():
            all_hs = model.extract_feat(waveform)
            # print(all_hs.shape)
            layer_feat = all_hs[output_layer].mean(dim=1).cpu().numpy()
        feats.append(layer_feat)
        # Use extend to handle batches of any size
        labels.extend(label)
        files.extend(path)
        attack_type.extend(attack)
    feats = np.vstack(feats)
    return feats, np.array(labels), files, attack_type


def extract_spkr_features(model, dataloader):
    feats, labels, files, attack_type = [], [], [], []
    for waveform, label, path, attack in tqdm(dataloader, desc="Extracting speaker features"):
        with torch.no_grad():
            spkr_emb = model.extract_speaker_embedding(waveform)
            # Ensure tensor is on CPU before converting to numpy
            if isinstance(spkr_emb, torch.Tensor):
                spkr_emb = spkr_emb.detach().cpu().numpy()
                # if spkr_emb.device.type != "cpu":
                #     spkr_emb = spkr_emb.cpu().numpy()
        
        feats.append(spkr_emb)
        # Use extend to handle batches of any size
        labels.extend(label)
        files.extend(path)
        attack_type.extend(attack)
    feats = np.vstack(feats)
    return feats, np.array(labels), files, attack_type 


def extract_fused_embeddings(ssl_model, spk_model, fusion_ckpt_path: str, dataloader, ssl_layer: int, device: str):
    
    # ---- load fusion checkpoint ----
    ckpt = torch.load(fusion_ckpt_path, map_location=device)

    d_ssl = int(ckpt["d_ssl"])
    d_sb  = int(ckpt["d_sb"])
    alpha = float(ckpt.get("alpha", 0.2))
    hidden = int(ckpt.get("hidden", 512))

    fusion = ResidualFusion(d_ssl=d_ssl, d_sb=d_sb, alpha=alpha, hidden=hidden).to(device)
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    fusion.eval()

    ssl_model.eval()
    spk_model.eval()

    residual_feats = []
    residual_norms = []

    feats, labels, files, attack_type = [], [], [], []

    for waveform, label, path, attack in tqdm(dataloader, desc="Extracting fused embeddings"):
        # waveform from your AudioDataset is [B,T] or [T] depending on your collate
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1,T]

        # Your SSLModel/Speaker_Model expect [B,1,T] based on your code patterns
        waveform = waveform.to(device).unsqueeze(1)  # [B,1,T]

        with torch.no_grad():
            # Speaker embedding: [B, D_sb]
            e_sb = spk_model.extract_speaker_embedding(waveform)

            # SSL embedding: [B, D_ssl]
            all_hs = ssl_model.extract_feat(waveform)  # list of layers
            e_ssl = get_ssl_layer_emb(all_hs, ssl_layer, batch_size=waveform.size(0))

            # Fused embedding: [B, D_sb]
            # z = fusion(e_sb, e_ssl).detach().cpu().numpy()
            z, r = fusion(e_sb, e_ssl)          # unpack

        z_np = z.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        r_norm = torch.norm(r, dim=1).detach().cpu().numpy()

        feats.append(z_np)
        residual_feats.append(r_np)
        residual_norms.append(r_norm)

        # keep metadata consistent with your existing code
        labels.extend(label if isinstance(label, (list, tuple)) else [label])
        files.extend(path if isinstance(path, (list, tuple)) else [path])
        attack_type.extend(attack if isinstance(attack, (list, tuple)) else [attack])
 
    feats = np.vstack(feats) # [N, D_sb]
    residual_feats = np.vstack(residual_feats)
    residual_norms = np.concatenate(residual_norms)
    
    return feats, residual_feats, residual_norms, np.array(labels), files, attack_type


def visualize_umap(features, labels, file_paths, attack_type, config, desc="layer_0"):
    reducer = umap.UMAP(
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=42
    )
    embedding = reducer.fit_transform(features)

    # Map '-' to 'bonafide' for clarity
    attack_type = ["bonafide" if x == "-" else x for x in attack_type]
    
    # Determine marker based on whether it's ITW (in-the-wild) dataset
    markers = []
    for source in attack_type:
        is_itw = "itw" in source.lower()
        is_dfeval = "dfeval2024" in source.lower()
        if is_itw:
            markers.append("star")
        elif is_dfeval:
            markers.append("cross")
        else:  
            markers.append("circle")
    
    df = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "Label": labels,
        "filename": [os.path.basename(p) for p in file_paths],
        "Source": attack_type,
        "Marker": markers
    })

    # Create color mapping: blue for bonafide, unique colors for other sources
    unique_sources = df["Source"].unique()
    color_map = {}
    
    # Separate bonafide and attack sources
    attack_sources = [s for s in unique_sources if s != "bonafide"]
    
    # Assign blue to bonafide
    color_map["bonafide"] = available_colors[0]
    
    # Assign each attack source a unique color from available colors
    for i, source in enumerate(attack_sources):
        color_map[source] = available_colors[(i + 1) % len(available_colors)]

    # Build symbol_map generically from the Marker column so Plotly
    # uses the literal marker names (e.g. 'star', 'circle') instead
    # of remapping them to its default symbol sequence.
    symbol_map = {}
    for m in df["Marker"].unique():
        # fallback to 'circle' if marker value is missing/empty
        symbol_map[m] = m if (m and isinstance(m, str)) else "circle"
    
    if config.extract_ssl:
        model_name = config.ssl_model + "_" + desc
    elif config.extract_speaker_embed:
        model_name = config.speaker_embedding_type
    elif config.extract_fusion_features:
        model_name = config.ssl_model + "_" + config.speaker_embedding_type + "_" + config.fused_feature_type
    
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Source",
        symbol="Marker",
        symbol_map=symbol_map,
        color_discrete_map=color_map,
        hover_data=["filename", "Label"],
        title=f"UMAP projection of {model_name} features {desc}",
        width=900,
        height=700
    )

    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        legend_title_text="Labels",
            # Place legend outside the plot area on the right
            legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
            # Add margin on the right to make room for the outside legend
            margin=dict(r=220),
    )

    protocol_tag = "_".join(config.protocol_tags)

    # Save HTML file
    if config.extract_ssl:
        out_dir = os.path.join(config.out_dir, config.speaker_name, config.ssl_model)
    else:
        out_dir = os.path.join(config.out_dir, config.speaker_name)
    os.makedirs(out_dir, exist_ok=True)
    
    html_path = os.path.join(out_dir, f"umap_{protocol_tag}_{model_name}_{desc}.html")
    png_path = os.path.join(out_dir, f"umap_{protocol_tag}_{model_name}_{desc}.png")
    # html_path = os.path.join(out_dir, f"umap_{config.ssl_model}_layer_{layer_number}.html")
    # png_path = os.path.join(out_dir, f"umap_{config.ssl_model}_layer_{layer_number}.png")

    # Save interactive HTML
    fig.write_html(html_path)
    print(f"✅ Saved interactive UMAP visualization to: {html_path}")

    # Save static PNG (requires kaleido)
    try:
        fig.write_image(png_path, scale=2)
        print(f"✅ Saved static PNG visualization to: {png_path}")
    except Exception as e:
        print(f"⚠️ Could not save PNG file (install kaleido?): {e}")


    # fig.show()


if __name__ == "__main__":
    
    config = FeatureConfig()
    dataset = AudioDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    if config.extract_ssl:
        # Extract and visualize SSL features
        if isinstance(config.output_layer, list):
            for layer in config.output_layer:
                print(f"Visualizing SSL layer {layer}...")
                model = SSLModel(config.n_layers, config.device, args=config)
                model.eval()

                features, labels, file_paths, source = extract_SSL_features(model, dataloader, layer)
                visualize_umap(features, labels, file_paths, source, config, desc="layer " + str(layer))
        else:
            print("Visualizing SSL layer...")
            model = SSLModel(config.n_layers, config.device, args=config)
            model.eval()

            features, labels, file_paths, source = extract_SSL_features(model, dataloader, config.output_layer)
            visualize_umap(features, labels, file_paths, source, config, desc="layer " + str(config.output_layer))

    # Extract and visualize Speaker embeddings
    if config.extract_speaker_embed:
        print("Extracting speaker embeddings...")

        speaker_model = Speaker_Model(config.device, config.speaker_model, embedding_type=config.speaker_embedding_type)

        features, labels, file_paths, source = extract_spkr_features(speaker_model, dataloader)
        print(features.shape)
        visualize_umap(features, labels, file_paths, source, config, desc=config.speaker_embedding_type)


    # Extract and visualize fused embeddings
    if config.extract_fusion_features:
        print("Extracting fused embeddings...")

        speaker_model = Speaker_Model(config.device, config.speaker_model, embedding_type=config.speaker_embedding_type)
        ssl_model = SSLModel(config.n_layers, config.device, args=config)

        features, residual, residual_norm, labels, file_paths, source = extract_fused_embeddings(ssl_model, speaker_model, config.fusion_ckpt_path, dataloader, config.output_layer, config.device)

        print("Residual norm stats:")
        print("Mean:", residual_norm.mean())
        print("Std:", residual_norm.std())
        print("Min:", residual_norm.min())
        print("Max:", residual_norm.max())

        np.savez(
            "fused_debug.npz",
            fused=features,
            residual=residual,
            residual_norm=residual_norm,
            labels=np.array(labels),
            files=np.array(file_paths, dtype=object),
            attack_type=np.array(source, dtype=object),
        )

        # df = pd.DataFrame({
        #     "file": file_paths,
        #     "label": labels,
        #     "attack_type": source,
        #     "residual_norm": residual_norm
        # })

        # df.to_csv("residual_debug.csv", index=False)

        print(features.shape)

        visualize_umap(features, labels, file_paths, source, config, desc=config.run_name)
