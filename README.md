# Oral Cancer Detection using Deep Learning

This repository contains deep learning models designed for the automated classification of oral cancer from images. It uses pre-trained convolutional neural networks (CNNs) augmented with a custom **CBAM (Convolutional Block Attention Module)** for improved spatial and channel attention.

## Architecture
The system employs transfer learning with different backbone models including:
- **DenseNet201** (`densnet201.py`)
- **EfficientNet** (`efficientnet.py`)
- **MobileNet** (`mobilenet.py`)

### Model Pipeline
1. **Pre-trained CNN Backbone**: Extracts feature maps (with pre-trained `imagenet` weights frozen)
2. **CBAM Attention Block:**
   - **Channel Attention**: Global Average & Max Pooling -> Shared Dense layers -> Sigmoid activation -> Multiply with feature maps
   - **Spatial Attention**: Convolutional layer (7x7 kernel) over concatenated Average/Max pools -> Sigmoid activation -> Multiply with channel-attended features
3. **Global Max Pooling 2D**
4. **Dense Layer (256, ReLU)** + **Dropout (0.5)** for regularization
5. **Output Layer**: 2 Nodes (Softmax) for `Cancer` and `Non-Cancer` classification

## How to Run

1. **Environment Setup**:
   Ensure you have a Python environment with TensorFlow and scientific libraries installed:
   ```bash
   pip install tensorflow matplotlib opencv-python pandas numpy seaborn scikit-learn scikit-image tqdm
   ```

2. **Dataset Setup**:
   - The code originally targets Google Drive paths (e.g., `/content/drive/MyDrive/Project2025/oralcancer1`).
   - Before running locally, update the `train_path` and image loading paths in the scripts to match your local dataset directory.

3. **Training & Evaluation**:
   Execute the model pipeline of your choice:
   ```bash
   python densnet201.py
   ```
   - The scripts automatically load images, perform rigorous data augmentation, split train/test partitions, and train the model.
   - Comprehensive model analytics are generated: accuracy/loss graphs, ROC curves, confusion matrices, and visualizations of intermediate 2D convolution feature maps (saved as `.tiff` files).

## Dataset Structure
The dataset directory should be structured with subfolders acting as class labels:
```text
oralcancer1/
├── Cancer/
│   ├── oral_scc_0001.jpg
│   └── ...
└── Non-Cancer/
    ├── normal_0001.jpg
    └── ...
```

## Performance Metrics
Upon completion of testing, the scripts emit several crucial medical imaging metrics including:
- **Accuracy** and Mis-Classification 
- **Sensitivity** (True Positive Rate)
- **Specificity** (True Negative Rate)
- **Precision** and **NPV** (Negative Predictive Value)
- **F1 Score** and **ROC AUC**
