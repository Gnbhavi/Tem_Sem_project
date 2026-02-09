# ────────────────────────────────────────────────
# Full Pipeline: Hu Moments + ResNet Hybrid Model
# ────────────────────────────────────────────────

import os, datetime
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix

# ────────────────────────────────────────────────
# 1. Paths and setup
# ────────────────────────────────────────────────
root_dir = os.path.abspath("..")  # assuming you are in root/Notebooks
charts_dir = os.path.join(root_dir, "Charts")
os.makedirs(charts_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ────────────────────────────────────────────────
# 2. Data transforms + dataset
# ────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

dataset_root = os.path.join(root_dir, "Dataset")
dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

# ────────────────────────────────────────────────
# 3. Hu moments calculation
# ────────────────────────────────────────────────
def hu_moments_from_path(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    return hu

hu_features = []
labels_list = []

for path, label in dataset.samples:
    hu = hu_moments_from_path(path)
    hu_features.append(hu)
    labels_list.append(label)

hu_features = np.array(hu_features)
labels = np.array(labels_list)

# ────────────────────────────────────────────────
# 4. Visualize Hu moments with t-SNE
# ────────────────────────────────────────────────
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
hu_tsne = tsne.fit_transform(hu_features)

df_hu = pd.DataFrame({
    "TSNE1": hu_tsne[:,0],
    "TSNE2": hu_tsne[:,1],
    "label": labels
})

sns.set(style="whitegrid")
sns.set_palette("husl")  # bright colors

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_hu, x="TSNE1", y="TSNE2", hue="label", s=80, alpha=0.8)
plt.title("Hu Moments (t-SNE projection)", fontsize=14)

save_path_hu_tsne = os.path.join(charts_dir, f"hu_tsne_{timestamp}.png")
plt.savefig(save_path_hu_tsne, dpi=600, bbox_inches="tight")
plt.close()
print(f"Hu t-SNE plot saved → {save_path_hu_tsne}")

# ────────────────────────────────────────────────
# 5. ResNet18 feature extractor
# ────────────────────────────────────────────────
resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

features_list = []
with torch.no_grad():
    for inputs, _ in DataLoader(dataset, batch_size=8, shuffle=False):
        inputs = inputs.to(device)
        outputs = resnet(inputs)
        features_list.append(outputs.cpu())

cnn_features = torch.cat(features_list).numpy()

# ────────────────────────────────────────────────
# 6. PCA with 3 components (Hu + CNN features)
# ────────────────────────────────────────────────
combined_features = np.concatenate([cnn_features, hu_features], axis=1)

pca = PCA(n_components=3)
combined_pca = pca.fit_transform(combined_features)

df_pca = pd.DataFrame({
    "PC1": combined_pca[:,0],
    "PC2": combined_pca[:,1],
    "PC3": combined_pca[:,2],
    "label": labels
})

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", s=80, alpha=0.8)
plt.title("Hybrid Features (PCA projection)", fontsize=14)

save_path_pca = os.path.join(charts_dir, f"hybrid_pca_{timestamp}.tif")
plt.savefig(save_path_pca, dpi=600, bbox_inches="tight")
plt.close()
print(f"Hybrid PCA plot saved → {save_path_pca}")

# ────────────────────────────────────────────────
# 7. Hybrid Model definition
# ────────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self, cnn_backbone, hu_dim=7):
        super().__init__()
        self.cnn = cnn_backbone
        self.fc = nn.Sequential(
            nn.Linear(512 + hu_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # binary classification
        )

    def forward(self, x_img, x_hu):
        cnn_feat = self.cnn(x_img)  # (batch, 512)
        combined = torch.cat((cnn_feat, x_hu), dim=1)
        return self.fc(combined)

# ────────────────────────────────────────────────
# 8. Training loop
# ────────────────────────────────────────────────
cnn_backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
cnn_backbone.fc = nn.Identity()
cnn_backbone = cnn_backbone.to(device)

model = HybridModel(cnn_backbone).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, lbls) in enumerate(train_loader):
        inputs = inputs.to(device)
        lbls = lbls.to(device)
        hu_batch = torch.tensor(hu_features[train_dataset.indices[i*8:(i+1)*8]],
                                dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(inputs, hu_batch)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f}")

# ────────────────────────────────────────────────
# 9. Validation predictions
# ────────────────────────────────────────────────
model.eval()
val_preds, val_true = [], []
with torch.no_grad():
    for i, (inputs, lbls) in enumerate(val_loader):
        inputs = inputs.to(device)
        lbls = lbls.to(device)
        hu_batch = torch.tensor(hu_features[val_dataset.indices[i*8:(i+1)*8]],
                                dtype=torch.float32).to(device)

        outputs = model(inputs, hu_batch)
        preds = torch.argmax(outputs, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(lbls.cpu().numpy())

acc = accuracy_score(val_true, val_preds)
cm = confusion_matrix(val_true, val_preds)

print("Validation Accuracy:", acc)
print("Confusion Matrix:\n", cm)
