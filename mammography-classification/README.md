# Mammography Image Classification Project - Template

This is a template project structure for developing a deep learning model for mammography image analysis and classification. The goal is to create a binary classification system to distinguish between benign and malignant mammography images.

## Project Overview

🎯 **Objective**: Build a complete pipeline for mammography image classification using deep learning
- **Input**: Mammography images (DICOM, PNG, JPEG)
- **Output**: Binary classification (benign vs malignant)
- **Approach**: Transfer learning with modern CNN architectures

## Project Structure

```
mammography-classification/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # TODO: Dataset loading and handling
│   │   └── preprocessing.py    # TODO: Image preprocessing and CLAHE
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py        # TODO: Custom CNN architectures
│   │   └── transfer_learning.py # TODO: EfficientNet, ResNet, DenseNet
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py            # TODO: Training loop and validation
│   │   └── evaluate.py         # TODO: Model evaluation and metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── gradcam.py          # TODO: Grad-CAM visualization
│   ├── utils/
│   │   ├── __init__.py
│   │   └── metrics.py          # TODO: Evaluation metrics
│   └── main.py                 # TODO: Main pipeline orchestration
├── configs/
│   └── config.yaml             # TODO: Configuration parameters
├── requirements.txt            # TODO: Project dependencies
└── README.md
```

## Getting Started

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (update requirements.txt first)
pip install -r requirements.txt
```

### 2. Dataset Preparation
Organize your dataset in the following structure:
```
data/
├── train/
│   ├── benign/
│   └── malignant/
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

### 3. Implementation Roadmap

#### Phase 1: Data Pipeline
- [ ] Implement `dataset.py` - Custom PyTorch Dataset class
- [ ] Implement `preprocessing.py` - CLAHE, normalization, augmentation
- [ ] Test data loading with sample images

#### Phase 2: Model Architecture
- [ ] Implement `transfer_learning.py` - EfficientNet/ResNet/DenseNet
- [ ] Modify final layers for binary classification
- [ ] Test model creation and forward pass

#### Phase 3: Training Pipeline
- [ ] Implement `train.py` - Training loop with validation
- [ ] Add early stopping and model checkpointing
- [ ] Configure hyperparameters in `config.yaml`

#### Phase 4: Evaluation
- [ ] Implement `evaluate.py` - Comprehensive metrics
- [ ] Add ROC curve, confusion matrix plotting
- [ ] Calculate accuracy, precision, recall, specificity, F1-score

#### Phase 5: Explainability
- [ ] Implement `gradcam.py` - Grad-CAM visualization
- [ ] Generate attention maps for model interpretation
- [ ] Save visualization results

#### Phase 6: Integration
- [ ] Implement `main.py` - Complete pipeline
- [ ] Add logging and experiment tracking
- [ ] Final testing and validation

## Hardware Requirements

- **GPU**: RTX 4050 (6GB VRAM) - Use lightweight models and batch size 8-16
- **CPU**: Intel i7
- **Memory**: Sufficient RAM for data loading

## Model Recommendations

For your hardware constraints, consider:
- **EfficientNet-B0/B1**: Excellent accuracy-efficiency trade-off
- **ResNet50**: Well-established, good performance
- **DenseNet121**: Efficient parameter usage

## Key Implementation Tips

1. **Memory Management**: Use batch size 8-16, enable gradient checkpointing if needed
2. **Preprocessing**: Implement CLAHE for mammography contrast enhancement
3. **Data Augmentation**: Horizontal flip, small rotations (±10°), minimal zoom
4. **Transfer Learning**: Freeze backbone initially, fine-tune later
5. **Evaluation**: Focus on sensitivity (recall) for medical applications
6. **Visualization**: Use Grad-CAM to understand model decisions

## Expected Deliverables

- ✅ Working training pipeline
- ✅ Comprehensive evaluation metrics
- ✅ Grad-CAM visualizations
- ✅ Clean, modular code structure
- ✅ Documentation and usage instructions

## Getting Help

Each Python file contains TODO comments indicating what needs to be implemented. Start with the data pipeline (`dataset.py`, `preprocessing.py`) and work your way through the phases.

Good luck with your university project! 🚀