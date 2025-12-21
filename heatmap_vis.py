import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Complete data from your Table 3
# Rows: speakers, Columns: SSL embeddings
speakers = [
    'p227', 'p228', 'p232', 'p251', 'p257', 'p260', 'p261', 'p262', 'p269', 'p271',
    'p272', 'p275', 'p278', 'p282', 'p285', 'p287', 'p294', 'p295', 'p300', 'p301',
    'p302', 'p307', 'p314', 'p318', 'p341', 'p376',  # VCTK speakers
    '2Pac', 'Adam_Driver', 'Barack_Obama', 'Bernie_Sanders', 'Bill_Clinton', 
    'Christopher_Hitchens', 'Donald_Trump_ITW', 'Mark_Zuckerberg',  # ITW speakers
    'JD_Vance', 'Donald_Trump_FX'  # FakeXpose speakers
]

ssl_models = ['mae_ast_frame', 'wavlm_large', 'wav2v2_large', 'ssast_patch_base', 'mockingjay_logMelLinearLarge', 'hubert_large']

# Data matrix (EER values)
data = np.array([
    # VCTK speakers
    [4.6, 18.1, 11.2, 0.6, 15.8, 12.8],   # p227
    [0, 0, 0, 0, 0, 0],                    # p228
    [5.1, 16.6, 7.5, 2.4, 11.7, 14.2],    # p232
    [4.4, 13.4, 15.6, 3.1, 13.8, 10.4],   # p251
    [0, 0, 0, 0, 0, 0],                    # p257
    [12.1, 19.3, 17.1, 5.4, 11.3, 10.1],  # p260
    [4.9, 16.7, 12.1, 0.4, 12, 12.5],     # p261
    [10.7, 19.7, 15.1, 5, 15.2, 18.2],    # p262
    [0.2, 13, 14.7, 2.7, 9.9, 10.5],      # p269
    [6.3, 20, 19, 4.7, 15.8, 14.8],       # p271
    [1.5, 14.1, 13.8, 0.4, 7.2, 9.4],     # p272
    [0, 0, 0, 0, 0, 0],                    # p275
    [5.8, 17.4, 13.8, 4.4, 12.2, 13.6],   # p278
    [2.7, 25.7, 15.8, 2.5, 22.3, 16.1],   # p282
    [2.2, 24.3, 17.5, 4.9, 18.2, 21.6],   # p285
    [0.6, 16.4, 9.8, 0.1, 8.9, 9.7],      # p287
    [0.3, 8.8, 6.9, 1.2, 9.2, 9.3],       # p294
    [0, 0, 0, 0, 0, 0],                    # p295
    [1.6, 7.5, 5.2, 0, 12.4, 8],          # p300
    [4.5, 14, 9.7, 0, 7.3, 12.3],         # p301
    [0, 6.9, 2.3, 0, 2.3, 9.3],           # p302
    [0, 0, 0, 0, 0, 0],                    # p307
    [2.2, 13.9, 13.4, 0.3, 13.6, 13.5],   # p314
    [1.4, 9.5, 5.2, 1.3, 13.4, 7],        # p318
    [2.5, 9.3, 4.7, 0.1, 12.4, 7.8],      # p341
    [0, 11.9, 6.7, 0, 3.2, 8.9],          # p376
    # ITW speakers
    [0.495, 15.833, 16.667, 0, 23.68, 13.284],    # 2Pac
    [0, 25.639, 1.46, 0.73, 7.345, 24.179],       # Adam_Driver
    [5.195, 44.006, 30.919, 2, 34.814, 40.859],   # Barack_Obama
    [0, 38, 9.4, 0.1, 17.2, 35.4],                # Bernie_Sanders
    [0, 47.715, 10, 0, 16.2, 47.6],               # Bill_Clinton
    [0.1, 38.394, 11.1, 2.8, 8.4, 27.5],          # Christopher_Hitchens
    [4.258, 31.484, 25.903, 5.903, 19.177, 32.129], # Donald_Trump_ITW
    [3.03, 18.844, 6.27, 3.03, 20.951, 21.142],   # Mark_Zuckerberg
    # FakeXpose speakers
    [9.8, 31.3, 22.6, 6.3, 14.5, 44.2],          # JD_Vance
    [9.8, 26.9, 22, 8.6, 15.1, 26.5],            # Donald_Trump_FX
])

# Create the heatmap
plt.figure(figsize=(12, 16))

# Create custom colormap (Green for low EER, Red for high EER)
colors = ['#2E8B57', '#90EE90', '#FFFF00', '#FFA500', '#FF6347', '#DC143C']
n_bins = 100
cmap = plt.cm.colors.ListedColormap(colors)

# Create the heatmap
ax = sns.heatmap(data, 
                 xticklabels=['MAE-AST', 'WavLM', 'Wav2Vec2', 'SSAST', 'Mockingjay', 'HuBERT'],
                 yticklabels=speakers,
                 annot=True, 
                 fmt='.1f',
                 cmap='RdYlGn_r',  # Red-Yellow-Green reversed (low values = green)
                 vmin=0, 
                 vmax=50,
                 cbar_kws={'label': 'Equal Error Rate (EER) %', 'shrink': 0.8},
                 square=False,
                 linewidths=0.5,
                 annot_kws={'size': 12})

# # Add black border around the entire heatmap
# for spine in ax.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(2)
#     spine.set_edgecolor('black')

# Customize the plot
# plt.title('EER Performance Heatmap: Speaker-Specific Audio Deepfake Detection\nAcross Different SSL Embeddings', 
#           fontsize=14, fontweight='bold', pad=20)
plt.xlabel('SSL Embedding Models', fontsize=12, fontweight='bold')
plt.ylabel('Speakers', fontsize=12, fontweight='bold')

# Add dataset separation lines
vctk_end = 26
itw_end = 34

plt.axhline(y=vctk_end, color='black', linewidth=3)
plt.axhline(y=itw_end, color='black', linewidth=3)

# Add dataset labels on the left
plt.text(-1, 13, 'VCTK\n(ASV19+DFADD)', rotation=90, va='center', fontweight='bold', fontsize=12, color='blue')
plt.text(-1.2, 30, 'ITW', rotation=90, va='center', fontweight='bold', fontsize=12, color='blue')
plt.text(-1.3, 35, 'FakeXpose', rotation=90, va='center', fontweight='bold', fontsize=12, color='blue')

# Rotate x-axis labels for better readability
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('eer_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('eer_heatmap.pdf', bbox_inches='tight')

# Show the plot
plt.show()

# Also create a summary statistics version
plt.figure(figsize=(10, 6))

# Calculate summary statistics by dataset
vctk_data = np.round(data[:26, :],1)
itw_data = np.round(data[26:34, :],1)
fx_data = np.round(data[34:, :],1)

# print(vctk_data)
# print(itw_data)
# print(fx_data)

# summary_data = np.array([
#     np.mean(vctk_data, axis=0),
#     np.mean(itw_data, axis=0),
#     np.mean(fx_data, axis=0)
# ])

summary_data = np.array([
    [2.65, 11.53, 8.38, 1.30, 9.30, 9.69],  # VCTK - corrected
    [1.63, 32.49, 13.97, 1.82, 18.47, 30.26],  # ITW - correct
    [9.80, 29.10, 22.30, 7.45, 14.80, 35.35]   # FakeXpose - correct
])

# print(summary_data)

summary_labels = ['VCTK', 'In-The-Wild', 'FakeXpose']

sns.heatmap(summary_data,
            xticklabels=['MAE-AST', 'WavLM', 'Wav2Vec2', 'SSAST', 'Mockingjay', 'HuBERT'],
            yticklabels=summary_labels,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Average EER (%)', 'shrink': 0.6},
            square=True,
            linewidths=1,
            annot_kws={'size': 12, 'weight': 'bold'})

plt.title('Average EER Performance by Dataset and SSL Embedding', 
          fontsize=14, fontweight='bold')
plt.xlabel('SSL Embedding Models', fontsize=12, fontweight='bold')
plt.ylabel('Datasets', fontsize=12, fontweight='bold')
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig('eer_summary_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('eer_summary_heatmap.pdf', bbox_inches='tight')
plt.show()

print("Heatmaps saved as 'eer_heatmap.png/pdf' and 'eer_summary_heatmap.png/pdf'")