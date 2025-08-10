# Severstal Steel Defect Detection – U-Net Segmentation

This repository contains a U-Net-based deep learning model for semantic segmentation of steel defects, built using the [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) dataset.

# 📌 Project Overview
The goal of this project is to automatically detect and segment defects in steel surfaces from high-resolution images.  
We use a U-Net architecture with convolutional layers, batch normalization, and skip connections to perform pixel-level classification.

# 📂 Dataset
- Source: Kaggle Severstal Steel Defect Detection competition
- Format: Images in .jpg, masks in RLE (Run-Length Encoding)
- Classes: 4 defect types + background

# 🛠️ Methodology
1. Data Loading – Images and masks are loaded directly from the CSV (RLE decoding).
2. Preprocessing – Resizing to 800×128, normalization.
3. Model Architecture – Custom U-Net implemented in PyTorch.
4. Training – BCE + Dice loss, Adam optimizer, learning rate scheduling.
5. Prediction – Generates pixel-wise probability maps for defect detection.

# 📊 Results
- Model outputs segmentation masks for each defect type.
- Can be used to generate submission.csv for Kaggle evaluation.

# 📦 Requirements
`bash
pip install torch torchvision albumentations opencv-python pandas tqdm
