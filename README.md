# Can/Cannot Hold Object Classification using Hu Moments + ResNet50 Features

**Date of last run:** (automatically uses current date in script)  
**Objective:** Binary classification of images into two classes:  
- **1** = Can hold (graspable object)  
- **0** = Cannot hold (non-graspable object)

This pipeline combines **hand-crafted geometric features (Hu moments)** with **deep learning extracted features (ResNet50)** and trains a simple MLP regressor/classifier.

## 1. Dataset

- **Source directories**:
  - `CAN_DIR` → images of graspable objects (label = 1)
  - `CANNOT_HOLD_DIR` → images of non-graspable objects (label = 0)
- **Image format**: `.tif` grayscale images
- **Total images loaded**: Printed during execution

## 2. Feature Extraction Pipeline

### 2.1 Hu Moments (Shape-based features)
- Computed using OpenCV `cv2.moments` → `cv2.HuMoments`
- 7 Hu invariant moments per image
- Used as rotation/scale/translation invariant descriptors

### 2.2 PCA on Hu Moments
- Reduced 7 Hu moments → **3 principal components**
- Explained variance ratio is printed
- Visualization: 2D scatter plot (PC1 vs PC2) colored by class

### 2.3 Deep Features – ResNet50
- Pre-trained ResNet50 (ImageNet)
- Removed final classification head → 2048-dimensional global average pooling features
- Input preprocessing:
  - Resize → 256
  - Center crop → 224×224
  - Normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 2.4 Feature Fusion
- Concatenation: **PCA (3 dims) + ResNet50 (2048 dims) → 2051-dimensional feature vector**

## 3. Model Architecture

Simple **MLP regressor**:

```python
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)        # single output (regression style)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze()
```

- Loss: MSE (Mean Squared Error)
- Optimizer: Adam, lr = 0.001
- Output interpretation: 
  - trained with labels 0.0 / 1.0
  - usually threshold at 0.5 for classification

## 4. Training & Evaluation Setup

- Train / Test split: 80/20 (random_state=42)
- Batch size: 32
- Epochs: 50
- Metrics tracked every epoch:
  - **Train RMSE**
  - **Test RMSE**
- Final reported metric: Test RMSE (lower is better)

## 5. Visualizations (saved in `CHART_DIR`)
| File name pattern                            | Description                                    |
|----------------------------------------------|------------------------------------------------|
| `hu_moments_pairplot_YYYY-MM-DD.png`         | Pairplot of all 7 Hu moments + class hue       |
| `pca_scatter_YYYY-MM-DD.png`                 | 2D PCA projection scatter plot (PC1 vs PC2)    |
| `rmse_vs_epochs_YYYY-MM-DD.png`              | Learning curve: Train vs Test RMSE over epochs |
## 6. Current Performance

- Final Test RMSE: (printed at the end of training)
- Check the learning curve plot for signs of overfitting / underfitting

## 7. How to Run
```Bash
  python model.py
```
Make sure paths are correctly defined in `utils.py`:
```python
ROOT, DATASET_DIR, CAN_DIR, CANNOT_HOLD_DIR, CHART_DIR
```

## 8. Experiments & Model Comparison
This section is reserved for comparing alternative models.
You can copy-paste and adapt the training/evaluation loop below each new model.
8.1 Baseline (Current Model)

- Features: Hu PCA (3) + ResNet50 (2048)
- Architecture: 2051 → 512 → 256 → 1
- Loss: MSE
- Epochs: 50
- Test RMSE: [record after run]