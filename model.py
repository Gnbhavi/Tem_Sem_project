import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datetime import datetime
from utils import (ROOT, DATASET_DIR, CAN_DIR, CANNOT_HOLD_DIR, CHART_DIR)


os.makedirs(CHART_DIR, exist_ok=True)
current_date = datetime.now().strftime('%Y-%m-%d')

# === Hu Moments Loader ===
def load_images_and_compute_hu_moments(folder, label):
    hu_moments_list, image_paths = [], []
    for file in os.listdir(folder):
        if file.lower().endswith('.tif'):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            moments = cv2.moments(img)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments_list.append(hu_moments)
            image_paths.append(img_path)
    labels = [label] * len(hu_moments_list)
    return np.array(hu_moments_list), labels, image_paths

# Load Hu moments
hu_can, labels_can, paths_can = load_images_and_compute_hu_moments(CAN_DIR, 1)
hu_cannot, labels_cannot, paths_cannot = load_images_and_compute_hu_moments(CANNOT_HOLD_DIR, 0)

hu_moments = np.vstack((hu_can, hu_cannot))
labels = np.array(labels_can + labels_cannot)
image_paths = paths_can + paths_cannot

print(f"Loaded {len(hu_moments)} images. Hu moments shape: {hu_moments.shape}")

# Create DataFrame for easy handling and charts
hu_df = pd.DataFrame(hu_moments, columns=[f'Hu_{i+1}' for i in range(7)])
hu_df['label'] = labels
hu_sns_path = os.path.join(CHART_DIR, f'hu_moments_pairplot_{current_date}.png')
sns.pairplot(hu_df, hue="label", vars=[f"Hu_{i+1}" for i in range(7)])
plt.savefig(hu_sns_path, dpi=600)
plt.show()

# # === PCA on Hu moments ===
pca = PCA(n_components=3)
pca_features = pca.fit_transform(hu_moments)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Plot PCA scatter (2D for simplicity, using PC1 and PC2)
pca_df = pd.DataFrame(pca_features, columns=['PC1', 'PC2', 'PC3'])
pca_df['label'] = labels
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette='viridis')
plt.title('PCA of Hu Moments (3 Components, Projected to 2D)')
plt.tight_layout()

# Save plot
pca_plot_path = os.path.join(CHART_DIR, f'pca_scatter_{current_date}.png')
plt.savefig(pca_plot_path, dpi=600)
plt.show()


# === ResNet Feature Extraction ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove classifier
resnet.eval()

def extract_resnet_features(image_paths):
    features = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = resnet(img_tensor).squeeze()
        features.append(feat.numpy())
    return np.array(features)

resnet_features = extract_resnet_features(image_paths)
print(f"ResNet features shape: {resnet_features.shape}")

# === Feature Fusion ===
combined_features = np.hstack((pca_features, resnet_features))

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    combined_features, labels, test_size=0.2, random_state=42
)

# === Dataset Class ===
class ImageDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_loader = DataLoader(ImageDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(ImageDataset(X_test, y_test), batch_size=32, shuffle=False)
# #
# # # === Simple MLP Classifier ===
# class SimpleModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 1)  # binary output
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)  # raw logits
#
# model = SimpleModel(combined_features.shape[1])
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # === Training Loop ===
# epochs = 20
# for epoch in range(epochs):
#     model.train()
#     for features, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(features).squeeze()
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
#
# # === Evaluation ===
# model.eval()
# all_probs, all_preds, all_true = [], [], []
# with torch.no_grad():
#     for features, labels in test_loader:
#         logits = model(features).squeeze()
#         probs = torch.sigmoid(logits)
#         preds = (probs > 0.5).int()
#         all_probs.extend(probs.cpu().numpy())
#         all_preds.extend(preds.cpu().numpy())
#         all_true.extend(labels.cpu().numpy())
#
# acc = accuracy_score(all_true, all_preds)
# prec = precision_score(all_true, all_preds)
# rec = recall_score(all_true, all_preds)
# roc = roc_auc_score(all_true, all_probs)
#
# print("\n=== Final Performance Metrics ===")
# print(f"Accuracy : {acc:.4f}")
# print(f"Precision: {prec:.4f}")
# print(f"Recall   : {rec:.4f}")
# print(f"ROC-AUC  : {roc:.4f}")


# Simple MLP model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Regression output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze()

model = SimpleModel(combined_features.shape[1])
criterion = nn.MSELoss()  # For RMSE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 50

train_rmse_history = []
test_rmse_history = []
epochs_list = list(range(1, epochs + 1))

for epoch in range(epochs):
    # Training phase
    model.train()
    train_preds = []
    train_true = []

    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collect all predictions to calculate RMSE later
        train_preds.extend(outputs.detach().cpu().numpy())
        train_true.extend(labels.cpu().numpy())

    # Calculate training RMSE after the whole epoch
    train_rmse = np.sqrt(mean_squared_error(train_true, train_preds))
    train_rmse_history.append(train_rmse)

    # Evaluation phase (test set)
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            test_preds.extend(outputs.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
    test_rmse_history.append(test_rmse)

    # Print progress
    print(f"Epoch {epoch+1:3d}/{epochs} | Train RMSE: {train_rmse:.6f} | Test RMSE: {test_rmse:.6f}")

print(f"\nFinal Test RMSE: {test_rmse_history[-1]:.6f}")

# === Plotting the learning curve ===
plt.figure(figsize=(10, 6))

plt.plot(epochs_list, train_rmse_history, marker='o', linestyle='-', color='blue', label='Training RMSE')
plt.plot(epochs_list, test_rmse_history, marker='D', linestyle='-', color='orange', label='Testing RMSE')

plt.fill_between(epochs_list,
                 train_rmse_history,
                 test_rmse_history,
                 color='gray', alpha=0.12)

plt.title('Training vs Testing RMSE over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

plt.tight_layout()

# Save the plot
learning_curve_path = os.path.join(CHART_DIR, f'rmse_vs_epochs_{current_date}.png')
plt.savefig(learning_curve_path, dpi=600, bbox_inches='tight')
plt.show()



# # Save RMSE to file
# rmse_path = os.path.join(chart_dir, f'rmse_{current_date}.txt')
# with open(rmse_path, 'w') as f:
#     f.write(f"RMSE: {rmse:.4f}\n")