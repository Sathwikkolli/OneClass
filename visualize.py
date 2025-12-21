import torch
import numpy as np
import pandas as pd
import umap
import os
from tqdm import tqdm
from plotly import express as px
from torch.utils.data import DataLoader

from ssl_utils import SSLModel
from config import FeatureConfig
from dataset_loader import AudioDataset
import random
import os

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


def extract_features(model, dataloader, output_layer):
    feats, labels, files, attack_type = [], [], [], []
    for waveform, label, path, attack in tqdm(dataloader, desc="Extracting features"):
        with torch.no_grad():
            all_hs = model.extract_feat(waveform)
            layer_feat = all_hs[output_layer].mean(dim=1).cpu().numpy()
        feats.append(layer_feat)
        # Use extend to handle batches of any size
        labels.extend(label)
        files.extend(path)
        attack_type.extend(attack)
    feats = np.vstack(feats)
    return feats, np.array(labels), files, attack_type


def visualize_umap(features, labels, file_paths, attack_type, config, layer_number):
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
        markers.append("x" if is_itw else "circle")
    
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
    
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Source",
        symbol="Marker",
        color_discrete_map=color_map,
        hover_data=["filename", "Label"],
        title=f"UMAP projection of {config.ssl_model} features",
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
    out_dir = os.path.join(config.out_dir, config.speaker_name)
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"umap_{protocol_tag}_{config.ssl_model}_layer_{layer_number}.html")
    png_path = os.path.join(out_dir, f"umap_{protocol_tag}_{config.ssl_model}_layer_{layer_number}.png")
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

    if isinstance(config.output_layer, list):
        for layer in config.output_layer:
            print(f"Visualizing layer {layer}...")
            model = SSLModel(config.n_layers, config.device, args=config)
            model.eval()

            features, labels, file_paths, source = extract_features(model, dataloader, layer)
            visualize_umap(features, labels, file_paths, source, config, layer)
    else:
        model = SSLModel(config.n_layers, config.device, args=config)
        model.eval()

        features, labels, file_paths, source = extract_features(model, dataloader, config.output_layer)
        visualize_umap(features, labels, file_paths, source, config, config.output_layer)