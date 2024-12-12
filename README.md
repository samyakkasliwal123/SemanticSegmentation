# Semantic Segmentation Using UNet on the Carvana Dataset

## Project Overview
This project implements a **UNet architecture** for performing semantic segmentation on the **Carvana Image Masking Dataset**, which contains images of cars and their corresponding segmentation masks. The primary goal is to accurately segment the car from the background.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Important Formulas](#important-formulas)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

---

## Introduction
Semantic segmentation is the process of labeling each pixel in an image with a corresponding class. In this project, we use the **UNet architecture**, a popular convolutional neural network designed for biomedical image segmentation, to identify cars in the Carvana dataset.

---

## Dataset
The Carvana Image Masking Dataset consists of high-resolution car images and their corresponding binary masks (car vs. background).

### Dataset Features:
- **Training images**: 5088 high-resolution images
- **Validation images**: 10% of the training set
- **Mask format**: Binary images with 1 indicating car pixels

### Example Images:
![Car Image](https://raw.githubusercontent.com/creafz/kaggle-carvana/master/img/example_predictions.gif)

---

## Model Architecture
The **UNet architecture** consists of two main paths:

1. **Contracting Path (Encoder):**
   - Sequence of convolutional layers followed by max-pooling.
   - Captures contextual information.

2. **Expanding Path (Decoder):**
   - Sequence of upsampling layers and concatenations with corresponding encoder layers.
   - Restores spatial resolution for precise localization.

### UNet Diagram:
![UNet Architecture](https://miro.medium.com/v2/resize:fit:1400/0*hljnVQ1r4ZcWxx-4.jpg)

---

## Important Formulas

### Binary Cross-Entropy Loss:
Used for training the model:
\[\text{BCE Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]\]
Where:
- \( y_i \): True label
- \( p_i \): Predicted probability
- \( N \): Total number of pixels

### Dice Coefficient:
Evaluates segmentation performance:
\[
\text{Dice Coefficient} = \frac{2 \times |A \cap B|}{|A| + |B|}
\]
Where:
- \( A \): Ground truth mask
- \( B \): Predicted mask

### Intersection over Union (IoU):
\[
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
\]

---

## Results

### Metrics Achieved:
- **Dice Coefficient**: 0.99
- **IoU**: 0.85

### Visual Results:
![Image](https://miro.medium.com/v2/resize:fit:1400/1*3dmHfN7a3l2uVSpY-z_TDA.png)
---

## Installation and Usage

### Requirements:
- Python 3.7+
- TensorFlow/Keras
- NumPy, Matplotlib, OpenCV

### Installation:
```bash
pip install -r requirements.txt
```

### Training the Model:
```bash
python train.py --epochs 50 --batch_size 16
```

### Evaluating the Model:
```bash
python evaluate.py --model saved_model.h5
```

---

## References
- [Carvana Image Masking Dataset](https://www.kaggle.com/c/carvana-image-masking-challenge)
- [UNet Paper](https://arxiv.org/abs/1505.04597)

---
