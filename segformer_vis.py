import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import umap

# --- 1. Configuration & Model Loading ---
MODEL_PATH = "best_ARAS400k_Segformer_efficientnet-b7.pth"
REAL_DIR = "ARAS400k/train/images"
SYN_DIR = "ARAS400k/synth/images"
GEN_DIR = "ARAS400k/train/generated"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(path):
    model = smp.Segformer(
        encoder_name="efficientnet-b7",
        encoder_weights=None,
        in_channels=3,
        classes=7
    )

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model.encoder


# --- 2. Feature Extraction ---
def extract_features(folder, encoder, max_images=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    features_list = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
    
    # 🔥 LIMIT HERE
    if max_images is not None and len(files) > max_images:
        np.random.shuffle(files)
        files = files[:max_images]

    print(f"Extracting features from {len(files)} images in {folder}...")

    for img_name in files:
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                stages = encoder(tensor)
                bottleneck = stages[-1]
                pooled = torch.nn.functional.adaptive_avg_pool2d(bottleneck, (1, 1))
                features_list.append(pooled.flatten().cpu().numpy())

        except Exception as e:
            print(f"Skipping {img_name}: {e}")

    return np.array(features_list)


# --- 3. Execution ---
encoder = load_encoder(MODEL_PATH)

real_feats = extract_features(REAL_DIR, encoder, max_images=50000)
syn_feats = extract_features(SYN_DIR, encoder, max_images=50000)
gen_feats = extract_features(GEN_DIR, encoder, max_images=50000)

# Combine
X = np.vstack([real_feats, syn_feats, gen_feats])

labels_str = (
    ['Real'] * len(real_feats) +
    ['Synthetic'] * len(syn_feats) +
    ['Generated'] * len(gen_feats)
)
labels_str = np.array(labels_str)

# Encode labels numerically (required for silhouette)
label_encoder = LabelEncoder()
labels_num = label_encoder.fit_transform(labels_str)

# Scale features
X_scaled = StandardScaler().fit_transform(X)

print("Feature matrix shape:", X_scaled.shape)

# --- 4. Dimensionality Reduction ---
print("Running t-SNE...")
tsne_results = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(X_scaled)

print("Running UMAP...")
umap_results = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
).fit_transform(X_scaled)


# --- 5. Visualization ---
def plot_results(data, labels, title, filename):
    plt.figure(figsize=(12, 8))

    unique_labels = np.unique(labels)
    colors = ['#2ca02c', '#d62728', '#1f77b4']
    markers = ['o', 'x', '^']

    for label, color, marker in zip(unique_labels, colors, markers):
        mask = labels == label
        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            c=color,
            label=label,
            alpha=0.6,
            s=20,
            marker=marker
        )

    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


plot_results(tsne_results, labels_str,
             "t-SNE Projection (B7 Encoder)",
             "tsne_comparison.png")

plot_results(umap_results, labels_str,
             "UMAP Projection (B7 Encoder)",
             "umap_comparison.png")


# --- 6. Silhouette Scores ---
print("\n=== Silhouette Scores ===")

# overall
overall_score = silhouette_score(X_scaled, labels_num)
print(f"Overall silhouette score: {overall_score:.4f}")

# pairwise
unique_classes = label_encoder.classes_

for i in range(len(unique_classes)):
    for j in range(i + 1, len(unique_classes)):
        mask = np.isin(labels_str, [unique_classes[i], unique_classes[j]])
        score = silhouette_score(
            X_scaled[mask],
            labels_num[mask]
        )
        print(f"{unique_classes[i]} vs {unique_classes[j]}: {score:.4f}")