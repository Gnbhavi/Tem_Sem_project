import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

# ────────────────────────────────────────────────
# 1. Data transforms
# ────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ────────────────────────────────────────────────
# 2. Load dataset (assuming folders: Dataset/can/ and Dataset/cannot_hold/)
# ────────────────────────────────────────────────
dataset_root = 'Dataset/'
dataset = torchvision.datasets.ImageFolder(root=dataset_root, transform=transform)

print("Classes:", dataset.classes)
print("Class → index mapping:", dataset.class_to_idx)
print(f"Total images: {len(dataset)}")

# 80/20 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ────────────────────────────────────────────────
# 3. Model: ResNet18 + binary head
# ────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ────────────────────────────────────────────────
# 4. Differential learning rates
# ────────────────────────────────────────────────
base_lr = 5e-4  # full speed for front & back
lr_middle = base_lr * 0.15  # 15% speed for middle layers

optimizer = optim.Adam([
    # Front ─ full speed
    {'params': model.conv1.parameters(), 'lr': base_lr},
    {'params': model.bn1.parameters(), 'lr': base_lr},
    {'params': model.layer1.parameters(), 'lr': base_lr},

    # Middle ─ 15% speed
    {'params': model.layer2.parameters(), 'lr': lr_middle},
    {'params': model.layer3.parameters(), 'lr': lr_middle},

    # Back ─ full speed
    {'params': model.layer4.parameters(), 'lr': base_lr},
    {'params': model.fc.parameters(), 'lr': base_lr},
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=4
)


# ────────────────────────────────────────────────
# 5. Evaluation function (loss + accuracy + RMSE on probabilities)
# ────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # prob of class 1
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(loader)
    accuracy = 100.0 * correct / total
    rmse = np.sqrt(mean_squared_error(all_labels, all_probs))

    return avg_loss, accuracy, rmse


# ────────────────────────────────────────────────
# 6. Training loop — now also collecting train RMSE
# ────────────────────────────────────────────────
num_epochs = 25

train_losses = []
val_losses = []
val_accs = []
val_rmses = []
train_rmses = []  # ← NEW: for training RMSE curve

print("\nStarting training...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # ── Compute train RMSE (on probabilities) ────────
    model.eval()
    train_probs = []
    train_labels_list = []
    with torch.no_grad():
        for t_inputs, t_labels in train_loader:
            t_inputs = t_inputs.to(device)
            t_outputs = model(t_inputs)
            t_probs = F.softmax(t_outputs, dim=1)[:, 1].cpu().numpy()  # prob of class 1
            train_probs.extend(t_probs)
            train_labels_list.extend(t_labels.cpu().numpy())

    train_rmse = np.sqrt(mean_squared_error(train_labels_list, train_probs))
    train_rmses.append(train_rmse)
    model.train()  # back to training mode

    # Validation
    val_loss, val_acc, val_rmse = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_rmses.append(val_rmse)

    print(f"Epoch {epoch + 1:2d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}% | Val RMSE: {val_rmse:.4f}")

    current_lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
    print(f"   LRs: {current_lrs}")

    scheduler.step(val_loss)

# ────────────────────────────────────────────────
# 7. Save model
# ────────────────────────────────────────────────
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H")
model_filename = f"can_hold_model_{timestamp}.pth"
model_path = os.path.join(models_dir, model_filename)

torch.save(model.state_dict(), model_path)
print(f"\nModel saved → {model_path}")

print(model)

# ────────────────────────────────────────────────
# 8. Plot RMSE vs Epochs with shading + auto-save
# ────────────────────────────────────────────────


# Create folder if needed
charts_dir = "Charts"
os.makedirs(charts_dir, exist_ok=True)

# Timestamp for filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(charts_dir, f"rmse_vs_epochs_{timestamp}.png")

# ── Plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

epochs_range = list(range(1, num_epochs + 1))

# Training RMSE
ax.plot(epochs_range, train_rmses, 'o-', color='blue', label='Training RMSE',
        linewidth=1.6, markersize=5, alpha=0.9)
ax.fill_between(epochs_range,
                np.array(train_rmses) - 0.012,
                np.array(train_rmses) + 0.012,
                color='blue', alpha=0.11)

# Validation RMSE
ax.plot(epochs_range, val_rmses, 'D-', color='orange', label='Validation RMSE',
        linewidth=1.6, markersize=5, alpha=0.9)
ax.fill_between(epochs_range,
                np.array(val_rmses) - 0.015,
                np.array(val_rmses) + 0.015,
                color='orange', alpha=0.11)

ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE')
ax.set_title('Training & Validation RMSE vs Epochs')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.35)

# Nice y-limit (adjust multiplier if your values are very different)
max_rmse = max(max(train_rmses), max(val_rmses))
ax.set_ylim(0, max_rmse * 1.25 if max_rmse > 0 else 0.2)

plt.tight_layout()

# Save high-res
plt.savefig(save_path, dpi=160, bbox_inches='tight')
print(f"RMSE plot saved → {save_path}")

# Show (comment out if you run in non-interactive env)
plt.show()

print("Done!")