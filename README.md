# Projet Traitement d'Image - Classification et Segmentation de Mammographies

**Étudiants:** Omar HAMMAMI, Elyes FANDOULI, Badie SAKKA  
**Institution:** INSAT (Institut National des Sciences Appliquées et de Technologie)  
**Cours:** Traitement d'Image  
**Date:** Janvier 2026

---

## 📋 Description

Système complet d'aide au diagnostic du cancer du sein utilisant le Deep Learning:
- **Classification multi-classe** (EfficientNet-B0): Bénin / Malin / Bénin sans rappel
- **Segmentation** (U-Net): Localisation précise des tumeurs
- **Dataset**: CBIS-DDSM (3,504 mammographies)

---

## 🎯 Objectifs du Projet

1. Classifier automatiquement les lésions mammographiques
2. Segmenter les zones tumorales avec masques précis
3. Fournir un outil d'aide à la décision pour radiologues

---

## 📁 Structure du Projet

```
PROJET-TRAITEMENT-IMAGE/
├── mammography-classification/    # Module Classification
│   ├── configs/config.yaml        # Configuration
│   ├── src/                       # Code source
│   │   ├── data/                  # Chargement données
│   │   ├── models/                # Architectures
│   │   ├── training/              # Entraînement
│   │   └── utils/                 # Utilitaires
│   ├── train.py                   # Script entraînement
│   ├── evaluate.py                # Évaluation
│   └── predict.py                 # Inférence
│
├── mamography_segment/            # Module Segmentation
│   ├── train_segmentation.py     # Entraînement U-Net
│   ├── visualize_predictions.py  # Visualisation
│   └── saved_models/              # Modèles entraînés
│
├── RAPPORT_PROJET.md              # Rapport académique complet
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

---

## 🚀 Installation

### Prérequis
- Python 3.12+
- GPU NVIDIA avec CUDA 12.1+ (recommandé)
- 16 GB RAM minimum

### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/OmarHammami123/PROJET-TRAITEMENT-IMAGE.git
cd PROJET-TRAITEMENT-IMAGE

# 2. Créer environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Installer PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Installer autres dépendances
pip install -r requirements.txt

# 5. Télécharger le dataset CBIS-DDSM
# Placer dans: mammography-classification/data/
```

---

## 📊 Dataset

**CBIS-DDSM** (Curated Breast Imaging Subset of DDSM)

- **Source**: The Cancer Imaging Archive (TCIA)
- **Taille**: 3,504 mammographies
- **Classes**: 
  - Bénin (benign)
  - Bénin sans rappel (benign_without_callback)
  - Malin (malignant)

**Distribution**:
- Training: 2,436 images (69.5%)
- Validation: 521 images (14.9%)
- Test: 547 images (15.6%)

---

## 🧠 Modèles

### Classification: EfficientNet-B0
- **Architecture**: Transfer learning depuis ImageNet
- **Paramètres**: 4M trainable
- **Input**: 224×224 RGB
- **Output**: 3 classes (logits)
- **Prétraitement**: CLAHE + Normalisation ImageNet

### Segmentation: U-Net
- **Architecture**: Encoder-Decoder avec skip connections
- **Loss**: Dice Loss
- **Output**: Masque binaire de segmentation
- **Métrique**: Dice Score (0.82)

---

## 🏋️ Entraînement

### Classification

```bash
# Entraîner le modèle
python mammography-classification/train.py

# Évaluer sur test set
python mammography-classification/evaluate.py

# Prédire sur nouvelle image
python mammography-classification/predict.py path/to/image.png
```

**Configuration** (dans `configs/config.yaml`):
```yaml
training:
  epochs: 30
  learning_rate: 0.0005
  batch_size: 32
  class_weights: [1.0, 3.5, 1.2]  # Gestion déséquilibre
```

### Segmentation

```bash
# Visualiser ground truth
python mamography_segment/visualize_ground_truth.py

# Visualiser prédictions
python mamography_segment/visualize_predictions.py
```

---

## 📈 Résultats

### Classification

**Métriques (Test Set - 547 images)**:
```
Overall Accuracy: 56.12% → 75% (après weighted loss)
ROC-AUC (macro): 0.73
F1-Score (macro): 0.38 → 0.68 (amélioré)
```

**Problème initial**: Déséquilibre classes (69 vs 271 images)  
**Solution**: Weighted Cross-Entropy Loss

### Segmentation

```
Dice Score: 0.82
IoU: 0.71
Pixel Accuracy: 0.94
```

---

## 🛠️ Technologies

- **Framework**: PyTorch 2.5.1
- **Architecture**: EfficientNet-B0 (timm), U-Net custom
- **Prétraitement**: OpenCV (CLAHE), torchvision
- **Métriques**: scikit-learn, matplotlib, seaborn
- **Hardware**: NVIDIA RTX 4050 (6GB VRAM)

---

## 📖 Documentation

### Fichiers principaux

- **`RAPPORT_PROJET.md`**: Rapport académique complet (25 pages)
  - Introduction et état de l'art
  - Méthodologie détaillée
  - Résultats et discussion
  - Références bibliographiques

- **`requirements.txt`**: Liste des dépendances
- **`configs/config.yaml`**: Configuration centralisée

---

## 🎓 Contexte Académique

Projet réalisé dans le cadre du cours **Traitement d'Image** à l'INSAT.

**Objectifs pédagogiques**:
- Maîtrise du Deep Learning appliqué à l'imagerie médicale
- Techniques de prétraitement (CLAHE, normalisation)
- Transfer Learning et fine-tuning
- Gestion de datasets déséquilibrés
- Évaluation rigoureuse (métriques, visualisations)

---

## 📝 Licence

Projet à usage éducatif uniquement.  
Pour toute utilisation clinique, validation médicale requise.

---

## 👥 Contributeurs

- **Omar HAMMAMI** - Classification pipeline
- **Elyes FANDOULI** - Segmentation pipeline
- **Badie SAKKA** - Segmentation pipeline

---

## 📧 Contact

Repository: https://github.com/OmarHammami123/PROJET-TRAITEMENT-IMAGE

---

**INSAT - Traitement d'Image - Janvier 2026**
